"""jax_lbm — JAX differentiable LBM for SCMP validation.

Minimal, pure-JAX D2Q9 LBM framework inspired by JAX-LaB.
Designed for CUDA cross-validation and autograd sensitivity analysis.

Modules
-------
lattice      — D2Q9 constants (velocities, weights, sound speed)
eos          — Multiple equations of state (CS, PR, RK, RKS, VdW)
collision    — BGK and MRT collision operators
boundary     — Boundary conditions (bounce-back, Zou/He, equilibrium BC)
force        — Shan-Chen + adsorption + body force
wetting      — Contact angle via ψ ghost BC or virtual density

Quick Start
-----------
>>> from jax_lbm import (
...     run_lbm, init_droplet,
...     EOS_REGISTRY, get_eos,
...     COLLISION_REGISTRY, get_collision,
...     BC_REGISTRY, get_bc,
...     WETTING_REGISTRY, get_wetting,
... )
>>>
>>> # Run with CS-EOS + MRT + BounceBack wall + contact angle
>>> eos_fn = get_eos("cs")
>>> collision_fn = get_collision("mrt")
>>> bc_fn = get_bc("bounce_back_halfway")
>>> wetting_fn = get_wetting("psi_ghost_bc")
"""

# ── Lattice constants ──
from jax_lbm.lattice import C, W, OPP, Q, CS2, INV_CS2, CS, G_FF

# ── EOS ──
from jax_lbm.eos import (
    EOS_REGISTRY,
    get_eos,
    list_eos,
    register_eos,
    cs_eos_pressure,
    pr_eos_pressure,
    rk_eos_pressure,
    rks_eos_pressure,
    vdw_eos_pressure,
    pseudopotential,
    critical_temperature,
)

# ── Collision ──
from jax_lbm.collision import (
    COLLISION_REGISTRY,
    get_collision,
    macroscopic,
    equilibrium,
    collision_bgk,
    collision_mrt,
    MRT_M,
    MRT_MINV,
    MRT_S_DEFAULT,
)

# ── Boundary ──
from jax_lbm.boundary import (
    BC_REGISTRY,
    get_bc,
    wall_mask,
    ghost_mask,
    bounce_back_fullway,
    bounce_back_halfway,
    equilibrium_bc,
    zou_he_bottom_wall,
)

# ── Force ──
from jax_lbm.force import (
    shan_chen_force,
    adsorption_force,
    body_force,
    total_force,
)

# ── Wetting ──
from jax_lbm.wetting import (
    WETTING_REGISTRY,
    get_wetting,
    apply_psi_ghost_bc,
    apply_improved_virtual_density,
)

# ── For backward compatibility: re-export from original d2q9_bgk.py ──
# These will be updated to use the new modules internally.
from jax_lbm.d2q9_bgk import (
    cs_eos_pressure as _,
    cs_eos_pseudopotential,
    _find_coexistence,
    streaming,
    step_scmp,
    run_lbm_scmp,
    init_droplet,
    init_equilibrium,
    make_permeability_fn,
)

# Clean up the unused import
del _

__all__ = [
    # Lattice
    "C",
    "W",
    "OPP",
    "Q",
    "CS2",
    "INV_CS2",
    "CS",
    "G_FF",
    # EOS
    "EOS_REGISTRY",
    "get_eos",
    "list_eos",
    "register_eos",
    "cs_eos_pressure",
    "pr_eos_pressure",
    "rk_eos_pressure",
    "rks_eos_pressure",
    "vdw_eos_pressure",
    "pseudopotential",
    "critical_temperature",
    # Collision
    "COLLISION_REGISTRY",
    "get_collision",
    "macroscopic",
    "equilibrium",
    "collision_bgk",
    "collision_mrt",
    "MRT_M",
    "MRT_MINV",
    "MRT_S_DEFAULT",
    # Boundary
    "BC_REGISTRY",
    "get_bc",
    "wall_mask",
    "ghost_mask",
    "bounce_back_fullway",
    "bounce_back_halfway",
    "equilibrium_bc",
    "zou_he_bottom_wall",
    # Force
    "shan_chen_force",
    "adsorption_force",
    "body_force",
    "total_force",
    # Wetting
    "WETTING_REGISTRY",
    "get_wetting",
    "apply_psi_ghost_bc",
    "apply_improved_virtual_density",
    # Backward compat
    "cs_eos_pseudopotential",
    "_find_coexistence",
    "streaming",
    "step_scmp",
    "run_lbm_scmp",
    "init_droplet",
    "init_equilibrium",
    "make_permeability_fn",
]
