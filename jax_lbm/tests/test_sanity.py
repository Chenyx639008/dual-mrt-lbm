"""JAX LBM validation — basic sanity tests that MUST pass before any code change.

These tests catch fundamental bugs like the streaming direction error (2026-07-25).
Run with:

    JAX_ENABLE_X64=1 uv run pytest jax_lbm/tests/test_sanity.py -v

Reference
---------
- HUANG_SCMP_VALIDATION_GUIDE.md §5 (instability diagnosis)
- Phase 6 P0a implementation record
"""

import jax
import jax.numpy as jnp
import pytest

# ── Force x64 for validation ──
jax.config.update("jax_enable_x64", True)

from jax_lbm.lattice import C, W, Q, CS2, G_FF
from jax_lbm.collision import (
    collision_bgk,
    collision_mrt,
    macroscopic,
    equilibrium,
    MRT_M,
    MRT_MINV,
)
from jax_lbm.force import shan_chen_force, compute_Q_huang, compute_S_guo
from jax_lbm.eos import cs_eos_pressure
from jax_lbm.d2q9_bgk import streaming, init_droplet, init_equilibrium


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: Streaming direction (THE bug that took 4 hours to find)
# ═══════════════════════════════════════════════════════════════════════════


class TestStreamingDirection:
    """Particle at (x,y) with velocity c_k should move to (x+cx, y+cy)."""

    # Lattice velocities (must match C array)
    _CX = [0, 1, 0, -1, 0, 1, -1, -1, 1]
    _CY = [0, 0, 1, 0, -1, 1, 1, -1, -1]

    @pytest.mark.parametrize("k", range(1, 9))  # skip rest particle (k=0)
    def test_particle_moves_in_c_direction(self, k):
        """f at (nx/2, ny/2) in direction k moves to (nx/2+cx, ny/2+cy)."""
        nx, ny = 8, 8
        cx, cy = self._CX[k], self._CY[k]

        f = jnp.zeros((nx, ny, Q))
        f = f.at[nx // 2, ny // 2, k].set(1.0)

        fs = streaming(f)

        expected_x = (nx // 2 + cx) % nx
        expected_y = (ny // 2 + cy) % ny

        assert fs[expected_x, expected_y, k] == 1.0, (
            f"k={k} c=({cx},{cy}): value should be at ({expected_x},{expected_y}), "
            f"not elsewhere. Streaming direction may be wrong!"
        )
        # Original position should be empty
        assert fs[nx // 2, ny // 2, k] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: Force at uniform field must be zero
# ═══════════════════════════════════════════════════════════════════════════


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


class TestPseudopotential:
    """ψ must be zero exactly where p ≥ ρ·cs²."""

    def test_psi_nonzero_below_critical(self):
        """At T < Tc, ψ must be non-zero in the liquid phase."""
        rho_l = jnp.array([[[0.30]]])
        p = cs_eos_pressure(rho_l, T=0.066)  # T/Tc ≈ 0.70
        psi_sq = jnp.clip(2.0 * (p - rho_l * CS2) / (-1.0), 0, None)
        assert float(psi_sq[0, 0, 0]) > 0.0, "ψ should be non-zero in liquid below Tc"

    def test_psi_decreases_with_temperature(self):
        """ψ must decrease as temperature increases (weaker phase separation)."""
        rho_test = jnp.array([[[0.20]]])  # intermediate density
        T_cold = 0.050  # T/Tc ≈ 0.53
        T_hot = 0.120  # T/Tc ≈ 1.27 (above Tc)
        p_cold = cs_eos_pressure(rho_test, T=T_cold)
        p_hot = cs_eos_pressure(rho_test, T=T_hot)
        psi_sq_cold = jnp.clip(2.0 * (p_cold - rho_test * CS2) / (-1.0), 0, None)
        psi_sq_hot = jnp.clip(2.0 * (p_hot - rho_test * CS2) / (-1.0), 0, None)
        assert float(psi_sq_cold[0, 0, 0]) > float(psi_sq_hot[0, 0, 0]), (
            f"ψ² should decrease with T: cold={float(psi_sq_cold):.4f}, hot={float(psi_sq_hot):.4f}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Test 8: MRT equilibrium moments match CUDA meq_gpu (P0b)
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
