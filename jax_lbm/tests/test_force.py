"""JAX LBM force tests.

Run: JAX_ENABLE_X64=1 uv run pytest jax_lbm/tests/test_force.py -v
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from jax_lbm.lattice import C, CS2, G_FF
from jax_lbm.force import shan_chen_force, compute_Q_huang, compute_S_guo, adsorption_force
from jax_lbm.eos import cs_eos_pressure

class TestForceUniformField:
    """In a uniform pseudopotential field, SC force must be exactly zero."""

    def test_force_zero_uniform_liquid(self):
        """Uniform liquid density → uniform psi → F = 0."""
        nx, ny = 8, 8
        rho = jnp.full((nx, ny, 1), 0.30)
        p = cs_eos_pressure(rho, T=0.066)
        psi = jnp.sqrt(jnp.clip(2.0 * (p - rho * CS2) / (-1.0), 0, None))
        F = shan_chen_force(psi, G=-1.0)
        assert jnp.allclose(F, 0.0, atol=1e-15), (
            f"Force should be zero, got max |F|={float(jnp.abs(F).max()):.2e}"
        )

    def test_force_zero_uniform_gas(self):
        """Uniform gas density → uniform psi → F = 0."""
        nx, ny = 8, 8
        rho = jnp.full((nx, ny, 1), 0.01)
        p = cs_eos_pressure(rho, T=0.0849)
        psi = jnp.sqrt(jnp.clip(2.0 * (p - rho * CS2) / (-1.0), 0, None))
        F = shan_chen_force(psi, G=-1.0)
        assert jnp.allclose(F, 0.0, atol=1e-15)


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: Collision conservation
# ═══════════════════════════════════════════════════════════════════════════

class TestQmCorrection:
    """Q_m surface tension correction basic properties."""

    def test_qm_zero_in_bulk(self):
        """Q_m must be zero where force is zero (bulk phases)."""
        nx, ny = 16, 16
        psi = jnp.full((nx, ny, 1), 0.4)
        Fx = jnp.zeros((nx, ny, 1))
        Fy = jnp.zeros((nx, ny, 1))

        C = compute_Q_huang(psi, Fx, Fy, k1=-0.2125)
        assert jnp.allclose(C, 0.0, atol=1e-15), (
            f"Q_m non-zero in bulk: max={float(jnp.abs(C).max()):.2e}"
        )

    def test_qm_epsilon_scaling(self):
        """Q_m magnitude must scale linearly with k1 (= −ε/8)."""
        nx, ny = 16, 16
        # Create a simple interface-like psi gradient
        psi = jnp.linspace(0.1, 0.4, nx)[:, jnp.newaxis, jnp.newaxis] * jnp.ones(
            (1, ny, 1)
        )
        F_mol = shan_chen_force(psi, G=-1.0)
        Fx, Fy = F_mol[..., 0:1], F_mol[..., 1:2]

        k1_a = 1.0 / 12.0
        k1_b = -0.2125
        Ca = compute_Q_huang(psi, Fx, Fy, k1=k1_a)
        Cb = compute_Q_huang(psi, Fx, Fy, k1=k1_b)

        ratio = jnp.abs(Cb).max() / max(jnp.abs(Ca).max(), 1e-15)
        expected = abs(k1_b / k1_a)
        assert abs(ratio - expected) / expected < 0.01, (
            f"Q_m scaling wrong: {ratio:.4f} vs {expected:.4f}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Test 7: Pseudopotential correctness
# ═══════════════════════════════════════════════════════════════════════════

class TestAdsorptionForce:
    """Adsorption force must point toward/away from walls based on G_ads sign."""

    def test_adsorption_zero_without_wall(self):
        """No wall nodes → no adsorption force."""
        nx, ny = 16, 16
        psi = jnp.ones((nx, ny, 1)) * 0.3
        mask = jnp.zeros((nx, ny), dtype=bool)
        F = adsorption_force(psi, mask, G_ads=1.0)
        assert jnp.allclose(F, 0.0, atol=1e-15)

    def test_adsorption_nonzero_at_wall(self):
        """Wall present → force non-zero near wall."""
        nx, ny = 16, 16
        psi = jnp.ones((nx, ny, 1)) * 0.3
        mask = jnp.zeros((nx, ny), dtype=bool)
        mask = mask.at[:, 0].set(True)  # bottom wall
        F = adsorption_force(psi, mask, G_ads=0.1)
        # Force should be non-zero at fluid nodes adjacent to wall
        assert not jnp.allclose(F[1:-1, 1, :], 0.0, atol=1e-15)

    def test_adsorption_sign(self):
        """Positive G_ads → force toward wall (hydrophilic), negative → away."""
        nx, ny = 16, 16
        psi = jnp.ones((nx, ny, 1)) * 0.3
        mask = jnp.zeros((nx, ny), dtype=bool)
        mask = mask.at[:, 0].set(True)

        F_pos = adsorption_force(psi, mask, G_ads=0.1)
        F_neg = adsorption_force(psi, mask, G_ads=-0.1)
        # Forces should have opposite y-components
        assert jnp.allclose(F_pos[..., 1], -F_neg[..., 1], atol=1e-14)


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: Contact angle ghost BC
# ═══════════════════════════════════════════════════════════════════════════

