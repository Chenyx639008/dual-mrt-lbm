"""JAX LBM integration tests.

Run: JAX_ENABLE_X64=1 uv run pytest jax_lbm/tests/test_integration.py -v
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from jax_lbm.d2q9_bgk import init_droplet, init_equilibrium, _step_scmp_huang_core

class TestFullSCMPStep:
    """Complete step function with adsorption + contact angle BC."""

    def test_step_with_adsorption_no_crash(self):
        """Step with non-zero G_ads must not crash."""
        nx, ny = 32, 32
        Tc = 0.3773 / 4.0
        eos_T = 0.70 * Tc
        tau = 1.5
        s_p = 1.0 / tau
        Lambda = 1.0 / 12.0
        s_q = 1.0 / (0.5 + Lambda / (tau - 0.5))
        s_relax = jnp.array([1.0, s_p, s_p, 1.0, s_q, 1.0, s_q, s_p, s_p])

        # Droplet on bottom wall
        rho, u = init_droplet(
            nx, ny, radius=12, center=(16, 4), rho_l=0.305202, rho_g=0.009170, width=3.0
        )
        f = init_equilibrium(rho, u)

        f_new = _step_scmp_huang_core(
            f,
            s_relax,
            1.0,
            4.0,
            1.0,
            eos_T,
            k1=0.06,
            k2=0.0,
            psi_cut=1e-3,
            G_mol=-1.0,
            c_lat=1.0,
            se=s_p,
            st=s_p,
            sp=s_p,
            alpha_meq=1.0,
            G_ads=0.1,
            theta_deg=60.0,
            wall="bottom",
        )

        assert not jnp.any(jnp.isnan(f_new)), "Step with adsorption produced NaN"

    def test_step_without_adsorption_backward_compat(self):
        """Call without G_ads/theta_deg must work (backward compat)."""
        nx, ny = 16, 16
        Tc = 0.3773 / 4.0
        eos_T = 0.70 * Tc
        tau = 1.5
        s_p = 1.0 / tau
        s_q = 1.0 / (0.5 + 1.0 / 12.0 / (tau - 0.5))
        s_relax = jnp.array([1.0, s_p, s_p, 1.0, s_q, 1.0, s_q, s_p, s_p])

        rho, u = init_droplet(
            nx, ny, radius=8, center=(8, 8), rho_l=0.30, rho_g=0.01, width=3.0
        )
        f = init_equilibrium(rho, u)

        # Old-style call (no G_ads, no theta_deg)
        f_new = _step_scmp_huang_core(
            f,
            s_relax,
            1.0,
            4.0,
            1.0,
            eos_T,
            k1=0.06,
            k2=0.0,
            psi_cut=1e-3,
            G_mol=-1.0,
            c_lat=1.0,
            se=s_p,
            st=s_p,
            sp=s_p,
            alpha_meq=1.0,
        )

        assert not jnp.any(jnp.isnan(f_new)), "Backward-compat call produced NaN"

