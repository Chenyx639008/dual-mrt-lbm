"""P2a: Adjoint gold-standard tests.

Run: JAX_ENABLE_X64=1 uv run pytest jax_lbm/tests/test_adjoint.py -v
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from jax_lbm.adjoint import (
    droplet_mass_loss,
    grad_mass_loss,
    finite_difference_gradient,
)


class TestAdjointGradient:
    """Adjoint gradients must be consistent with finite differences."""

    def test_gradient_exists(self):
        """Gradient function must be callable without error."""
        d_eps, d_T = grad_mass_loss(1.7, 0.70)  # must not crash
        assert jnp.isfinite(d_eps), "∂loss/∂ε must be finite"
        assert jnp.isfinite(d_T), "∂loss/∂T must be finite"

    def test_gradient_finite_difference_consistency(self):
        """Adjoint gradient must match central finite difference within tolerance."""

        # Use a simpler loss function for speed
        def loss_eps(eps):
            return droplet_mass_loss(eps, 0.90, nx=32, ny=32, n_steps=50)

        # Exact gradient via JAX
        d_exact = float(jax.grad(loss_eps)(-0.48))

        # Finite-difference approximation
        d_fd = finite_difference_gradient(loss_eps, -0.48, h=1e-4)

        # Relative error should be small
        rel_err = abs(d_exact - d_fd) / max(abs(d_exact), 1e-15)
        assert rel_err < 0.1, (
            f"FD error too large: exact={d_exact:.6e}, fd={d_fd:.6e}, err={rel_err:.4f}"
        )


class TestMassLoss:
    """Mass loss function basic properties."""

    def test_mass_loss_nonnegative(self):
        """Mass loss should be >= 0 (mass cannot increase in closed system)."""
        loss = droplet_mass_loss(1.7, 0.70, nx=32, ny=32, n_steps=50)
        assert float(loss) >= -1e-10, f"Mass loss negative: {float(loss):.6e}"

    def test_mass_loss_small(self):
        """Mass loss should be < 1% for stable parameters."""
        loss = droplet_mass_loss(-0.48, 0.90, nx=32, ny=32, n_steps=50)
        assert float(loss) < 0.01, f"Mass loss too large: {float(loss):.6e}"
