"""JAX LBM collision tests.

Run: JAX_ENABLE_X64=1 uv run pytest jax_lbm/tests/test_collision.py -v
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from jax_lbm.collision import collision_bgk, collision_mrt, macroscopic, equilibrium, MRT_M, MRT_MINV, meq_mrt
from jax_lbm.lattice import C

class TestCollisionConservation:
    """BGK and MRT collision must conserve mass exactly."""

    def test_bgk_mass_conservation(self):
        """BGK: Σ f_out = Σ f_in."""
        nx, ny = 8, 8
        rho = 1.0 + 0.1 * jax.random.normal(jax.random.PRNGKey(0), (nx, ny, 1))
        u = 0.05 * jax.random.normal(jax.random.PRNGKey(1), (nx, ny, 2))
        f = equilibrium(rho, u)
        F = jnp.zeros((nx, ny, 2))

        for omega in [0.5, 0.667, 1.0, 1.5]:
            f_out = collision_bgk(f, omega, F)
            assert jnp.allclose(jnp.sum(f_out), jnp.sum(f), rtol=1e-12), (
                f"BGK mass not conserved at ω={omega}"
            )

    def test_mrt_mass_conservation(self):
        """MRT: Σ f_out = Σ f_in for various relaxation rates."""
        nx, ny = 8, 8
        rho = 1.0 + 0.1 * jax.random.normal(jax.random.PRNGKey(0), (nx, ny, 1))
        u = 0.05 * jax.random.normal(jax.random.PRNGKey(1), (nx, ny, 2))
        f = equilibrium(rho, u)

        # Match CUDA SCMP relaxation rates
        tau = 1.5
        s_paper = 1.0 / tau
        s_q = 1.0 / (0.5 + (1.0 / 12.0) / (tau - 0.5))
        s_relax = jnp.array(
            [1.0, s_paper, s_paper, 1.0, s_q, 1.0, s_q, s_paper, s_paper]
        )

        f_out = collision_mrt(f, s_relax, F=jnp.zeros((nx, ny, 2)))
        assert jnp.allclose(jnp.sum(f_out), jnp.sum(f), rtol=1e-12), (
            "MRT mass not conserved"
        )

    def test_bgk_mrt_identical_at_tau1(self):
        """τ=1 (ω=1, s_k=1) → BGK ≡ MRT (both relax fully to equilibrium)."""
        nx, ny = 8, 8
        rho = 1.0 + 0.1 * jax.random.normal(jax.random.PRNGKey(0), (nx, ny, 1))
        u = 0.05 * jax.random.normal(jax.random.PRNGKey(1), (nx, ny, 2))
        f = equilibrium(rho, u)
        F = jnp.zeros((nx, ny, 2))

        f_bgk = collision_bgk(f, 1.0, F)
        f_mrt = collision_mrt(f, jnp.ones(9), F=F)
        assert jnp.allclose(f_bgk, f_mrt, atol=1e-12), "BGK and MRT differ at τ=1"


# ═══════════════════════════════════════════════════════════════════════════
# Test 4: MRT matrix correctness
# ═══════════════════════════════════════════════════════════════════════════

class TestMRTMatrix:
    """MRT transform matrix must be invertible."""

    def test_orthogonality(self):
        """M · M⁻¹ = I₉."""
        I9 = jnp.dot(MRT_M, MRT_MINV)
        assert jnp.allclose(I9, jnp.eye(9), atol=1e-12), "MRT matrix not invertible"


# ═══════════════════════════════════════════════════════════════════════════
# Test 5: Droplet stability (smoke test)
# ═══════════════════════════════════════════════════════════════════════════

class TestMRTEquilibrium:
    """JAX meq_mrt must produce identical values to CUDA meq_gpu."""

    # Reference from CUDA meq_gpu (k, formula)
    # k=0: rho
    # k=1: (-2 + 3u²)rho
    # k=2: (1 - 3u²)rho
    # k=3: ux*rho
    # k=4: -ux*rho
    # k=5: uy*rho
    # k=6: -uy*rho
    # k=7: (ux² - uy²)rho
    # k=8: ux*uy*rho

    def test_meq_slot_values(self):
        """Each meq slot must match CUDA formula."""
        from jax_lbm.collision import meq_mrt

        rho = jnp.array([[[2.0]]])
        u = jnp.array([[[0.3]], [[0.1]]]).transpose(1, 2, 0)  # (1,1,2)
        ux, uy = 0.3, 0.1
        u2 = ux * ux + uy * uy

        meq = meq_mrt(rho, u)  # shape (1, 1, 9)
        vals = [float(meq[0, 0, k]) for k in range(9)]
        expected = [
            2.0,  # k=0: rho
            2.0 * (-2.0 + 3.0 * u2),  # k=1
            2.0 * (1.0 - 3.0 * u2),  # k=2
            2.0 * ux,  # k=3
            2.0 * (-ux),  # k=4
            2.0 * uy,  # k=5
            2.0 * (-uy),  # k=6
            2.0 * (ux * ux - uy * uy),  # k=7
            2.0 * (ux * uy),  # k=8
        ]
        for k, (v, e) in enumerate(zip(vals, expected)):
            assert abs(v - e) < 1e-14, f"meq[{k}] mismatch: {v:.15f} vs {e:.15f}"

    def test_meq_alpha_handling(self):
        """alpha_meq must correctly modify slot 2 (ε moment)."""
        rho = jnp.ones((4, 4, 1))
        u = jnp.zeros((4, 4, 2))
        f = equilibrium(rho, u)

        # α=1.0 and α=0.5 should give different results
        f1 = collision_mrt(f, jnp.ones(9), alpha_meq=1.0)
        f2 = collision_mrt(f, jnp.ones(9), alpha_meq=0.5)
        assert not jnp.allclose(f1, f2), "α=1.0 and α=0.5 should differ"


# ═══════════════════════════════════════════════════════════════════════════
# Test 9: MRT Guo+C mode droplet stability (P0b)
# ═══════════════════════════════════════════════════════════════════════════

