"""JAX LBM wetting tests.

Run: JAX_ENABLE_X64=1 uv run pytest jax_lbm/tests/test_wetting.py -v
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from jax_lbm.wetting import apply_psi_ghost_bc
from jax_lbm.lattice import C

class TestContactAngleBC:
    """ψ-based ghost BC must produce correct contact angle behavior."""

    def test_theta_90_neutral(self):
        """θ=90° → ψ_ghost = ψ_ref (neutral)."""
        nx, ny = 32, 32
        # Create a gradient in ψ
        psi = jnp.ones((nx, ny, 1)) * 0.2
        psi = psi.at[:, 1:, :].set(0.4)  # step at y=1
        psi_updated = apply_psi_ghost_bc(psi, 90.0, wall="bottom")
        # At θ=90°, cot=0, so ψ_ghost should be close to ψ_ref
        diff = jnp.abs(psi_updated[:, 0, 0] - psi[:, 2, 0])
        assert float(jnp.mean(diff)) < 0.01

    def test_theta_60_wets_wall(self):
        """θ=60° → ψ_ghost > ψ_ref (hydrophilic, higher ψ near wall)."""
        nx, ny = 32, 32
        # Create vertical gradient: bottom low, top high
        y = jnp.arange(ny, dtype=jnp.float64)
        psi = 0.1 + 0.3 * (y / ny)
        psi = psi[jnp.newaxis, :, jnp.newaxis] * jnp.ones((nx, 1, 1))
        psi_updated = apply_psi_ghost_bc(psi, 60.0, wall="bottom")
        # ψ_ghost (y=0) should be larger than ψ_ref (y=2) for hydrophilic
        diff = float(jnp.mean(psi_updated[:, 0, 0] - psi[:, 2, 0]))
        assert diff > 0, f"θ=60 should increase ψ at wall, got diff={diff:.4f}"

    def test_theta_120_repels(self):
        """θ=120° → ψ_ghost < ψ_ref (hydrophobic)."""
        nx, ny = 32, 32
        y = jnp.arange(ny, dtype=jnp.float64)
        psi = 0.1 + 0.3 * (y / ny)
        psi = psi[jnp.newaxis, :, jnp.newaxis] * jnp.ones((nx, 1, 1))
        psi_updated = apply_psi_ghost_bc(psi, 120.0, wall="bottom")
        diff = float(jnp.mean(psi_updated[:, 0, 0] - psi[:, 2, 0]))
        assert diff < 0, f"θ=120 should decrease ψ at wall, got diff={diff:.4f}"


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: Laplace law — ΔP = σ/R
# ═══════════════════════════════════════════════════════════════════════════

