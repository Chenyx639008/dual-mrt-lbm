"""JAX LBM eos tests.

Run: JAX_ENABLE_X64=1 uv run pytest jax_lbm/tests/test_eos.py -v
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from jax_lbm.eos import cs_eos_pressure
from jax_lbm.lattice import CS2

class TestPseudopotential:
    """ψ must be zero exactly where p ≥ ρ·cs²."""

    def test_psi_nonzero_below_critical(self):
        """At T < Tc, ψ must be non-zero in the liquid phase."""
        rho_l = jnp.array([[[0.30]]])
        p = cs_eos_pressure(rho_l, T=0.066)  # T/Tc ≈ 0.70
        psi_sq = jnp.clip(2.0 * (p - rho_l * CS2) / (-1.0), 0, None)
        assert float(psi_sq[0, 0, 0]) > 0.0, "ψ should be non-zero in liquid below Tc"

    def test_psi_decreases_with_temperature(self):
        """ψ must decrease as temperature increases (weaker phase separation)."""
        rho_test = jnp.array([[[0.20]]])  # intermediate density
        T_cold = 0.050  # T/Tc ≈ 0.53
        T_hot = 0.120  # T/Tc ≈ 1.27 (above Tc)
        p_cold = cs_eos_pressure(rho_test, T=T_cold)
        p_hot = cs_eos_pressure(rho_test, T=T_hot)
        psi_sq_cold = jnp.clip(2.0 * (p_cold - rho_test * CS2) / (-1.0), 0, None)
        psi_sq_hot = jnp.clip(2.0 * (p_hot - rho_test * CS2) / (-1.0), 0, None)
        assert float(psi_sq_cold[0, 0, 0]) > float(psi_sq_hot[0, 0, 0]), (
            f"ψ² should decrease with T: cold={float(psi_sq_cold):.4f}, hot={float(psi_sq_hot):.4f}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Test 8: MRT equilibrium moments match CUDA meq_gpu (P0b)
# ═══════════════════════════════════════════════════════════════════════════

