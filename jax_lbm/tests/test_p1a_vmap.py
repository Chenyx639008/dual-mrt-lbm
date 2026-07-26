"""P1a: vmap batch sweep tests.

Run: JAX_ENABLE_X64=1 uv run pytest jax_lbm/tests/test_p1a_vmap.py -v
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from jax_lbm.parameter_sweep import _single_droplet_run, sweep_surface_tension


class TestVmapSweep:
    """vmap batch sweep must produce consistent results."""

    def test_single_run_returns_finite(self):
        """Single droplet run must return finite mass."""
        mass = _single_droplet_run(0.90, 0.06, nx=32, ny=32, n_steps=50)
        assert jnp.isfinite(mass), f"Mass not finite: {float(mass)}"
        assert float(mass) > 0, "Mass should be positive"

    def test_vmap_equals_serial(self):
        """vmap of 2 runs must equal 2 serial runs."""
        T_vals = jnp.array([0.90, 0.90])
        k1_vals = jnp.array([0.06, 0.06])

        # vmap
        from jax_lbm.parameter_sweep import batch_droplet_run

        masses_vmap = batch_droplet_run(T_vals, k1_vals, 32, 32, 50)

        # Serial
        m1 = _single_droplet_run(0.90, 0.06, nx=32, ny=32, n_steps=50)
        m2 = _single_droplet_run(0.90, 0.06, nx=32, ny=32, n_steps=50)

        assert jnp.allclose(masses_vmap[0], m1, rtol=1e-12)
        assert jnp.allclose(masses_vmap[1], m2, rtol=1e-12)

    def test_sweep_produces_grid(self):
        """sweep_surface_tension must produce correct shape grid."""
        masses, T_vals, eps_vals = sweep_surface_tension(
            T_range=(0.85, 0.95, 2),
            eps_range=(1.0, 2.0, 2),
            nx=32,
            ny=32,
            n_steps=50,
        )
        assert masses.shape == (2, 2), f"Wrong shape: {masses.shape}"
        assert not jnp.any(jnp.isnan(masses)), "NaN in sweep results"
