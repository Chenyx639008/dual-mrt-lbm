"""JAX LBM physics tests.

Run: JAX_ENABLE_X64=1 uv run pytest jax_lbm/tests/test_physics.py -v
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from jax_lbm.d2q9_bgk import init_droplet, init_equilibrium, streaming
from jax_lbm.collision import collision_mrt
from jax_lbm.eos import cs_eos_pressure
from jax_lbm.force import shan_chen_force, compute_Q_huang, compute_S_guo
from jax_lbm.lattice import C, CS2

class TestLaplaceLaw:
    """Laplace law: pressure difference across droplet interface ∝ 1/R."""

    # Use near-critical params for faster convergence
    T_REDUCED = 0.90
    RHO_L = 0.248108
    RHO_G = 0.045410
    K1 = 0.06  # ε = -0.48

    def _run_droplet(self, nx, ny, radius, n_steps=300):
        """Run a droplet simulation and return final density and pressure."""
        Tc = 0.3773 / 4.0
        eos_T = self.T_REDUCED * Tc
        tau = 1.5
        s_p = 1.0 / tau
        Lambda = 1.0 / 12.0
        s_q = 1.0 / (0.5 + Lambda / (tau - 0.5))
        s_relax = jnp.array([1.0, s_p, s_p, 1.0, s_q, 1.0, s_q, s_p, s_p])

        rho, u = init_droplet(
            nx,
            ny,
            radius=radius,
            center=(nx // 2, ny // 2),
            rho_l=self.RHO_L,
            rho_g=self.RHO_G,
            width=3.0,
        )
        f = init_equilibrium(rho, u)

        for _ in range(n_steps):
            rho_c = jnp.sum(f, axis=-1, keepdims=True)
            p_eos = cs_eos_pressure(rho_c, T=eos_T)
            psi = jnp.sqrt(jnp.clip(2.0 * (p_eos - rho_c * CS2) / (-1.0), 0, None))
            F_mol = shan_chen_force(psi, G=-1.0)
            C_qm = compute_Q_huang(
                psi,
                F_mol[..., 0:1],
                F_mol[..., 1:2],
                k1=self.K1,
                se=s_p,
                st=s_p,
                sp=s_p,
            )
            rho_s = jnp.clip(rho_c, 1e-8, None)
            u_raw = jnp.dot(f, C.astype(jnp.float64)) / rho_s
            u_corr = u_raw + 0.5 * F_mol / rho_s
            S_guo = compute_S_guo(
                u_corr[..., 0:1], u_corr[..., 1:2], F_mol[..., 0:1], F_mol[..., 1:2]
            )
            f = streaming(collision_mrt(f, s_relax, S_guo=S_guo, C=C_qm))

        rho_f = jnp.sum(f, axis=-1)
        return rho_f

    def _measure_delta_p(self, rho, nx, ny):
        """Measure pressure difference: inside vs outside droplet."""
        r_center = jnp.sqrt(
            (jnp.arange(nx)[:, None] - nx // 2) ** 2
            + (jnp.arange(ny)[None, :] - ny // 2) ** 2
        )
        inside = r_center < 8
        outside = r_center > nx // 2 - 5

        rho_in = float(jnp.mean(rho[inside]))
        rho_out = float(jnp.mean(rho[outside]))

        Tc = 0.3773 / 4.0
        p_in = float(
            cs_eos_pressure(jnp.array([[[rho_in]]]), T=self.T_REDUCED * Tc)[0, 0, 0]
        )
        p_out = float(
            cs_eos_pressure(jnp.array([[[rho_out]]]), T=self.T_REDUCED * Tc)[0, 0, 0]
        )
        return p_in - p_out

    def test_laplace_dp_vs_radius(self):
        """Larger radius → smaller ΔP (Laplace law direction)."""
        nx, ny = 128, 128
        # Two different radii
        rho_1 = self._run_droplet(nx, ny, radius=30, n_steps=300)
        rho_2 = self._run_droplet(nx, ny, radius=40, n_steps=300)

        assert not jnp.any(jnp.isnan(rho_1)), "R=30 NaN"
        assert not jnp.any(jnp.isnan(rho_2)), "R=40 NaN"

        dp_1 = self._measure_delta_p(rho_1, nx, ny)
        dp_2 = self._measure_delta_p(rho_2, nx, ny)

        # ΔP(R=30) > ΔP(R=40) — Laplace law direction
        assert dp_1 > dp_2, (
            f"Laplace direction wrong: dp(30)={dp_1:.4e} <= dp(40)={dp_2:.4e}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Test 4: Full SCMP step with adsorption + wetting
# ═══════════════════════════════════════════════════════════════════════════

