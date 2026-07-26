"""JAX LBM droplet tests.

Run: JAX_ENABLE_X64=1 uv run pytest jax_lbm/tests/test_droplet.py -v
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from jax_lbm.d2q9_bgk import init_droplet, init_equilibrium, streaming
from jax_lbm.collision import collision_bgk, collision_mrt, equilibrium
from jax_lbm.eos import cs_eos_pressure
from jax_lbm.force import shan_chen_force, compute_Q_huang, compute_S_guo
from jax_lbm.lattice import C, CS2

class TestDropletStability:
    """Droplet must survive 500 steps without blow-up."""

    # Validated production parameters (from HUANG_SCMP_VALIDATION_GUIDE.md §4.6)
    PROD_PARAMS = dict(
        rho_l=0.305202,
        rho_g=0.009170,
        T_reduced=0.70,
        tau=1.5,
        nx=128,
        ny=128,
        radius=30,
    )

    def test_droplet_survives_500_steps_bgk(self):
        """BGK droplet must not diverge within 500 steps."""
        p = self.PROD_PARAMS
        Tc = 0.3773 / 4.0
        eos_T = p["T_reduced"] * Tc

        rho, u = init_droplet(
            p["nx"],
            p["ny"],
            radius=p["radius"],
            center=(p["nx"] // 2, p["ny"] // 2),
            rho_l=p["rho_l"],
            rho_g=p["rho_g"],
            width=3.0,
        )
        f = init_equilibrium(rho, u)
        omega = 1.0 / p["tau"]

        for _ in range(500):
            rho_cur = jnp.sum(f, axis=-1, keepdims=True)
            p_eos = cs_eos_pressure(rho_cur, T=eos_T)
            psi = jnp.sqrt(jnp.clip(2.0 * (p_eos - rho_cur * CS2) / (-1.0), 0, None))
            F = shan_chen_force(psi, G=-1.0)
            f = streaming(collision_bgk(f, omega, F))

        rf = jnp.sum(f, axis=-1)
        assert not jnp.any(jnp.isnan(rf)), "Droplet diverged (NaN) after 500 steps"
        assert not jnp.any(jnp.isinf(rf)), "Droplet diverged (Inf) after 500 steps"
        assert rf.min() > 0.001, f"Gas density too low: {float(rf.min()):.6f}"
        assert rf.max() < 1.0, f"Liquid density too high: {float(rf.max()):.6f}"


# ═══════════════════════════════════════════════════════════════════════════
# Test 6: Q_m correction sanity
# ═══════════════════════════════════════════════════════════════════════════

class TestMRTGuoCDroplet:
    """MRT collision with Guo+C forcing must sustain a droplet."""

    # Production parameters
    RHO_L = 0.305202
    RHO_G = 0.009170

    def test_mrt_guoc_droplet_200_steps(self):
        """MRT Guo+C droplet must not diverge within 200 steps."""
        nx, ny = 64, 64
        Tc = 0.3773 / 4.0
        eos_T = 0.70 * Tc
        tau = 1.5

        s_paper = 1.0 / tau
        Lambda = 1.0 / 12.0
        s_q = 1.0 / (0.5 + Lambda / (tau - 0.5))
        s_relax = jnp.array(
            [1.0, s_paper, s_paper, 1.0, s_q, 1.0, s_q, s_paper, s_paper]
        )
        k1 = -1.7 / 8.0

        rho, u = init_droplet(
            nx,
            ny,
            radius=18,
            center=(32, 32),
            rho_l=self.RHO_L,
            rho_g=self.RHO_G,
            width=3.0,
        )
        f = init_equilibrium(rho, u)

        for _ in range(200):
            rho_c = jnp.sum(f, axis=-1, keepdims=True)
            p_eos = cs_eos_pressure(rho_c, T=eos_T)
            psi = jnp.sqrt(jnp.clip(2.0 * (p_eos - rho_c * CS2) / (-1.0), 0, None))

            F_mol = shan_chen_force(psi, G=-1.0)

            C_qm = compute_Q_huang(
                psi,
                F_mol[..., 0:1],
                F_mol[..., 1:2],
                k1=k1,
                psi_cut=1e-3,
                G=-1.0,
                c=1.0,
                se=s_paper,
                st=s_paper,
                sp=s_paper,
            )

            rho_safe = jnp.clip(rho_c, 1e-8, None)
            u_raw = jnp.dot(f, C.astype(jnp.float64)) / rho_safe
            u_corr = u_raw + 0.5 * F_mol / rho_safe
            S_guo = compute_S_guo(
                u_corr[..., 0:1], u_corr[..., 1:2], F_mol[..., 0:1], F_mol[..., 1:2]
            )

            f = streaming(collision_mrt(f, s_relax, S_guo=S_guo, C=C_qm))

        rf = jnp.sum(f, axis=-1)
        assert not jnp.any(jnp.isnan(rf)), "MRT Guo+C droplet diverged (NaN)"
        assert not jnp.any(jnp.isinf(rf)), "MRT Guo+C droplet diverged (Inf)"

