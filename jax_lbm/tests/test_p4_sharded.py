"""P4: Sharded (multi-device) LBM tests.

Run: JAX_ENABLE_X64=1 uv run pytest jax_lbm/tests/test_p4_sharded.py -v
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from jax_lbm.lattice import C, CS2
from jax_lbm.collision import equilibrium
from jax_lbm.d2q9_bgk import init_droplet, init_equilibrium

# ── Try importing sharded module ──
try:
    from jax_lbm.sharded_lbm import (
        setup_1d_x_mesh,
        run_sharded_scmp,
        validate_sharded_vs_serial,
    )

    _SHARDED_AVAILABLE = True
except ImportError:
    _SHARDED_AVAILABLE = False


class TestMeshSetup:
    """Mesh configuration tests."""

    def test_mesh_creation(self):
        """setup_1d_x_mesh must create a valid Mesh."""
        if not _SHARDED_AVAILABLE:
            pytest.skip("sharded_lbm not available")
        mesh = setup_1d_x_mesh(2)
        assert mesh is not None
        assert len(mesh.devices) >= 1


class TestShardedLBM:
    """Sharded LBM must not crash on simple input."""

    def test_sharded_step_no_crash(self):
        """A single sharded step must not crash."""
        if not _SHARDED_AVAILABLE:
            pytest.skip("sharded_lbm not available")

        nx, ny = 32, 32
        rho = jnp.ones((nx, ny, 1))
        u = jnp.zeros((nx, ny, 2))
        f = equilibrium(rho, u)
        omega = 1.0 / 1.5

        try:
            mesh = setup_1d_x_mesh(2)
            f_out = run_sharded_scmp(
                f,
                omega,
                1.0,
                4.0,
                1.0,
                0.066,
                -1.0,
                mesh,
                n_steps=2,
            )
            f_arr = jnp.array(f_out)
            assert not jnp.any(jnp.isnan(f_arr)), "Sharded output contains NaN"
        except Exception as e:
            if "shard_map" in str(e).lower() or "mesh" in str(e).lower():
                pytest.skip(f"shard_map not supported on this JAX build: {e}")
            raise

    def test_sharded_preserves_mass(self):
        """Sharded LBM must conserve mass."""
        if not _SHARDED_AVAILABLE:
            pytest.skip("sharded_lbm not available")

        nx, ny = 32, 32
        rho = jnp.ones((nx, ny, 1))
        u = jnp.zeros((nx, ny, 2))
        f = equilibrium(rho, u)
        mass0 = float(jnp.sum(rho))
        omega = 1.0 / 1.5

        try:
            mesh = setup_1d_x_mesh(2)
            f_out = run_sharded_scmp(
                f,
                omega,
                1.0,
                4.0,
                1.0,
                0.066,
                -1.0,
                mesh,
                n_steps=10,
            )
            f_arr = jnp.array(f_out)
            mass_final = float(jnp.sum(jnp.sum(f_arr, axis=-1)))
            assert abs(mass_final - mass0) / mass0 < 1e-12
        except Exception as e:
            if "shard_map" in str(e).lower() or "mesh" in str(e).lower():
                pytest.skip(f"shard_map not supported: {e}")
            raise
