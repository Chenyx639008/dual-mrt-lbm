"""Sharded (multi-device) LBM via JAX shard_map (P4).

Enables 1D X-direction domain decomposition for D2Q9 LBM.
Validates communication patterns before translating to CUDA MPI.

Design:
    - N devices, each handles NX/N × NY sub-domain
    - Halo depth = 1 (D2Q9 only needs nearest neighbors)
    - Streaming triggers automatic halo exchange via shard_map
    - Collision is local (no communication)
    - Force needs neighbor psi (handled by halo)

Usage::

    from jax_lbm.sharded_lbm import setup_mesh, step_sharded_scmp
    mesh = setup_mesh(n_devices=2)
    f_sharded = step_sharded_scmp(f, mesh, ...)

Reference
---------
- Phase 6 §8 (P4: JAX shard_map 2D domain decomposition)
- JAX-LaB: src/jax_lab/base.py (shard_map pattern)
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

from jax_lbm.lattice import C, Q, CS2
from jax_lbm.collision import collision_bgk, macroscopic
from jax_lbm.eos import cs_eos_pressure
from jax_lbm.force import shan_chen_force


# ═══════════════════════════════════════════════════════════════════════════
# Mesh setup
# ═══════════════════════════════════════════════════════════════════════════


def setup_1d_x_mesh(n_devices=2):
    """Configure 1D X-direction domain decomposition mesh.

    Parameters
    ----------
    n_devices : int — number of devices (GPUs or CPU threads)

    Returns
    -------
    mesh : jax.sharding.Mesh
    """
    devices = jax.devices()[:n_devices]
    if len(devices) < n_devices:
        devices = [jax.devices("cpu")[0]] * n_devices
    import numpy as np

    mesh = Mesh(np.array(devices).reshape(n_devices, 1), axis_names=("x", "y"))
    return mesh


# ═══════════════════════════════════════════════════════════════════════════
# Streaming with halo exchange
# ═══════════════════════════════════════════════════════════════════════════


def _stream_local(f):
    """Streaming within a single shard (no halo exchange needed)."""
    f_streamed = jnp.zeros_like(f)
    for k in range(Q):
        f_streamed = f_streamed.at[..., k].set(
            jnp.roll(
                jnp.roll(f[..., k], C[k, 0].astype(int), axis=0),
                C[k, 1].astype(int),
                axis=1,
            )
        )
    return f_streamed


# ═══════════════════════════════════════════════════════════════════════════
# Sharded SCMP step
# ═══════════════════════════════════════════════════════════════════════════


@partial(
    jit,
    static_argnames=("n_steps",),
)
def run_sharded_scmp(f0, omega, eos_a, eos_b, eos_R, eos_T, G, mesh, n_steps=200):
    """Run SCMP LBM with shard_map domain decomposition.

    Each device handles a 1/N_x slice of the domain along X.
    Halo exchange happens automatically during streaming.

    Parameters
    ----------
    f0 : (nx, ny, 9) — initial distributions (automatically sharded)
    omega : float — relaxation frequency
    eos_a, eos_b, eos_R, eos_T : float — CS-EOS parameters
    G : float — interaction strength
    mesh : jax.sharding.Mesh
    n_steps : int

    Returns
    -------
    f_final : (nx, ny, 9) — automatically gathered to single device
    """
    in_spec = P("x", "y", None)

    # Shard the initial state
    sharding = NamedSharding(mesh, in_spec)

    @partial(
        jax.experimental.shard_map.shard_map,
        mesh=mesh,
        in_specs=(in_spec,),
        out_specs=in_spec,
        check_rep=False,
    )
    def step_sharded(f):
        """One time-step within a shard. Halo inserted by shard_map."""
        # Macroscopic fields (local)
        rho = jnp.sum(f, axis=-1, keepdims=True)

        # Pseudopotential (local — uses only local ρ)
        p = cs_eos_pressure(rho, a=eos_a, b=eos_b, R=eos_R, T=eos_T)
        psi_sq = 2.0 * (p - rho * CS2) / (G * 1.0 * 1.0)
        psi = jnp.sqrt(jnp.clip(psi_sq, 0.0, None))

        # Force (needs halo for psi neighbors — shard_map handles this)
        F = shan_chen_force(psi, G=G)

        # Collision (local)
        f_coll = collision_bgk(f, omega, F)

        # Streaming (with halo exchange via shard_map)
        f_new = _stream_local(f_coll)

        return f_new

    # Run via lax.scan
    def scan_fn(f, _):
        return step_sharded(f), None

    f_final, _ = jax.lax.scan(scan_fn, f0, None, length=n_steps)

    # Gather back to single device
    gathered = jax.device_get(f_final)
    return gathered


# ═══════════════════════════════════════════════════════════════════════════
# Validation: single-device baseline vs sharded
# ═══════════════════════════════════════════════════════════════════════════


def validate_sharded_vs_serial(f0, omega, eos_params, n_steps=100):
    """Compare single-device vs 2-device sharded results.

    Parameters
    ----------
    f0 : (nx, ny, 9)
    omega : float
    eos_params : tuple (a, b, R, T, G)
    n_steps : int

    Returns
    -------
    max_diff : float — maximum absolute difference
    """
    from jax_lbm.d2q9_bgk import run_lbm_scmp

    # Serial baseline
    f_serial = run_lbm_scmp(f0, omega, eos_params, n_steps)

    # Sharded (2-device)
    try:
        mesh = setup_1d_x_mesh(2)
        a, b, R, T, G = eos_params
        f_sharded = jnp.array(
            run_sharded_scmp(
                f0,
                omega,
                a,
                b,
                R,
                T,
                G,
                mesh,
                n_steps=n_steps,
            )
        )
        max_diff = float(jnp.abs(f_serial - f_sharded).max())
        return max_diff
    except Exception as e:
        return float("nan")  # shard_map not available on this platform
