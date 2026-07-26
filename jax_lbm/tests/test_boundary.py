"""JAX LBM boundary condition tests.

Run: JAX_ENABLE_X64=1 uv run pytest jax_lbm/tests/test_boundary.py -v
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from jax_lbm.lattice import C, Q
from jax_lbm.collision import equilibrium, collision_bgk
from jax_lbm.boundary import (
    wall_mask,
    ghost_mask,
    bounce_back_halfway,
    equilibrium_bc,
    zou_he_bottom_wall,
)


class TestPeriodicBC:
    """Periodic BC is the default — no special handling needed."""

    def test_periodic_preserves_mass(self):
        """Streaming with periodic BC conserves mass."""
        nx, ny = 16, 16
        rho = jnp.ones((nx, ny, 1))
        u = jnp.zeros((nx, ny, 2))
        f = equilibrium(rho, u)
        from jax_lbm.d2q9_bgk import streaming

        fs = streaming(f)
        assert jnp.allclose(jnp.sum(fs), jnp.sum(f), rtol=1e-14)


class TestBounceBack:
    """Halfway bounce-back at wall nodes."""

    def test_bounce_back_mass_conservation(self):
        """Bounce-back should not change total mass."""
        nx, ny = 16, 16
        rho = jnp.ones((nx, ny, 1))
        u = jnp.zeros((nx, ny, 2))
        f = equilibrium(rho, u)
        mask = jnp.zeros((nx, ny), dtype=bool)
        mask = mask.at[:, 0].set(True)  # bottom wall
        f_bc = bounce_back_halfway(f, f, mask)
        assert jnp.allclose(jnp.sum(f_bc), jnp.sum(f), rtol=1e-14)

    def test_bounce_back_no_effect_off_wall(self):
        """Bounce-back should not affect fluid nodes."""
        nx, ny = 16, 16
        rho = jnp.ones((nx, ny, 1))
        u = jnp.zeros((nx, ny, 2))
        f = equilibrium(rho, u)
        mask = jnp.zeros((nx, ny), dtype=bool)  # no walls
        f_bc = bounce_back_halfway(f, f, mask)
        # With no walls, bounce-back should be identity
        assert jnp.allclose(f_bc, f, rtol=1e-14)


class TestEquilibriumBC:
    """Equilibrium boundary condition sets f = feq at wall."""

    def test_equilibrium_bc_no_nan(self):
        """Equilibrium BC should not produce NaN."""
        nx, ny = 16, 16
        rho = jnp.ones((nx, ny, 1))
        u = jnp.zeros((nx, ny, 2))
        f = equilibrium(rho, u)
        mask = jnp.zeros((nx, ny), dtype=bool)
        mask = mask.at[:, 0].set(True)
        f_bc = equilibrium_bc(f, mask, u_target=None)
        assert not jnp.any(jnp.isnan(f_bc))


class TestZouHeBC:
    """Zou/He bottom wall no-slip BC."""

    def test_zou_he_no_nan(self):
        """Zou/He BC should not produce NaN."""
        nx, ny = 16, 16
        rho = jnp.ones((nx, ny, 1))
        u = jnp.zeros((nx, ny, 2))
        f = equilibrium(rho, u)
        Fx = jnp.zeros((nx, ny))
        Fy = jnp.zeros((nx, ny))
        f_bc = zou_he_bottom_wall(f, f, Fx, Fy)
        assert not jnp.any(jnp.isnan(f_bc))


class TestWallMask:
    """Wall mask and ghost mask utilities."""

    def test_wall_mask_bottom(self):
        nx, ny = 16, 16
        mask = wall_mask(nx, ny, wall="bottom")
        # wall at bottom: y=1 is first fluid layer above ghost (y=0)
        assert mask[0, 1].item(), f"y=1 should be wall, got {mask[0, 1]}"
        assert not mask[0, 2].item(), "y=2 should be fluid"

    def test_ghost_mask_bottom(self):
        nx, ny = 16, 16
        mask = ghost_mask(nx, ny, wall="bottom")
        assert bool(jnp.all(mask[:, 0])), "Bottom row should be ghost"
        assert not bool(jnp.any(mask[:, 1])), "Second row should not be ghost"
