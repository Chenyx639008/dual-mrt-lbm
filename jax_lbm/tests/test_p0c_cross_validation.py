"""P0c: Cross-validation tests — adsorption, contact angle, Laplace law.

Run with:
    JAX_ENABLE_X64=1 uv run pytest jax_lbm/tests/test_p0c_cross_validation.py -v
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from jax_lbm.lattice import C, CS2
from jax_lbm.collision import collision_mrt, equilibrium
from jax_lbm.force import (
    shan_chen_force,
    compute_Q_huang,
    compute_S_guo,
    adsorption_force,
)
from jax_lbm.eos import cs_eos_pressure
from jax_lbm.wetting import apply_psi_ghost_bc
from jax_lbm.d2q9_bgk import (
    init_droplet,
    init_equilibrium,
    streaming,
    _step_scmp_huang_core,
)


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: Adsorption force basic properties
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
