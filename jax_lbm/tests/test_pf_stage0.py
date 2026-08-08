"""Phase-field JAX track — Stage 0 (single-phase NS) + Stage 1 (AC) tests.

Run: JAX_ENABLE_X64=1 uv run pytest jax_lbm/tests/test_pf_stage0.py -v
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from jax_lbm.pf.phase_field import (
    run_poiseuille,
    conservative_ac_step,
    tanh_interface_profile,
    run_lid_driven_cavity,
    ghia_uy_at,
)
from jax_lbm.collision import collision_bgk, equilibrium, macroscopic
from jax_lbm.d2q9_bgk import streaming


class TestStage0SinglePhaseNS:
    """Stage 0: single-phase incompressible NS — plane Poiseuille benchmark."""

    def test_poiseuille_profile(self):
        """Velocity profile matches analytic parabola (< 2% L2 relative error)."""
        nx, ny = 64, 32
        u_profile, u_exact = run_poiseuille(
            nx=nx, ny=ny, omega=1.0, gx=1e-4, n_steps=8000
        )
        # skip wall rows (y=0,1,ny-2,ny-1) for comparison
        interior = slice(2, ny - 2)
        err = jnp.linalg.norm(
            u_profile[interior] - u_exact[interior]
        ) / jnp.linalg.norm(u_exact[interior])
        assert float(err) < 0.02, f"Poiseuille L2 rel err = {float(err):.4f} (>2%)"
        assert float(u_profile[interior].max()) > 0.5 * float(u_exact.max()), (
            "flow not developed"
        )

    def test_no_slip_walls(self):
        """Velocity is (near) zero at the no-slip walls."""
        u_profile, _ = run_poiseuille(nx=32, ny=16, omega=1.0, gx=1e-4, n_steps=4000)
        assert float(jnp.abs(u_profile[1])) < 1e-6
        assert float(jnp.abs(u_profile[-2])) < 1e-6

    def test_mass_conserved(self):
        """Total density conserved in periodic single-phase flow (EDM force)."""
        nx, ny = 16, 8
        rho0 = 1.0
        u0 = jnp.zeros((nx, ny, 2))
        f0 = equilibrium(jnp.full((nx, ny, 1), rho0), u0)
        F = jnp.broadcast_to(jnp.array([1e-5, 0.0], dtype=jnp.float64), (nx, ny, 2))
        f = f0
        for _ in range(200):
            f = streaming(collision_bgk(f, 1.0, F))
        rho0sum = float(jnp.sum(f0))
        rho1sum = float(jnp.sum(f))
        assert abs(rho1sum - rho0sum) / rho0sum < 1e-6, "mass not conserved"

    def test_lid_driven_cavity_vs_ghia(self):
        """Lid-driven cavity vertical-centreline u(y) vs Ghia (Re=100)."""
        nx = ny = 96
        u_lid = 0.05
        # Re = u_lid·L/ν = 100 → ν = u_lid·L/100; ν = (1/ω − 0.5)/3
        nu = u_lid * nx / 100.0
        omega = 1.0 / (0.5 + 3.0 * nu)
        u_centerline, _ = run_lid_driven_cavity(
            nx=nx, ny=ny, u_lid=u_lid, omega=omega, n_steps=30000
        )
        # compare interior points (exclude wall layers) to Ghia reference
        errs = []
        for y_norm in (0.1, 0.25, 0.5, 0.75, 0.9):
            u_ghia = ghia_uy_at(y_norm) * u_lid
            # map normalised y (0..1) to lattice y (1..ny-2)
            y_lat = 1 + int(round(y_norm * (ny - 3)))
            u_sim = float(u_centerline[y_lat])
            if abs(u_ghia) > 1e-3:
                errs.append(abs(u_sim - u_ghia) / abs(u_ghia))
        assert len(errs) > 0
        max_err = max(errs)
        # BGK LBM cavity at 96²: ~15% centreline error is typical (Ghia is a
        # fine-grid benchmark); 25% confirms the cavity vortex structure.
        assert max_err < 0.25, f"cavity max rel err vs Ghia = {max_err:.3f} (>25%)"


class TestStage1ConservativeAC:
    """Stage 1: conservative Allen-Cahn order-parameter evolution (no flow)."""

    def test_phi_conserved_no_flow(self):
        """Without advection, ∫φ stays constant (conservative AC)."""
        nx, ny = 64, 64
        x = jnp.arange(nx, dtype=jnp.float64)
        phi0 = jnp.broadcast_to(
            tanh_interface_profile(x, x0=32.0, W=3.0)[None, :], (ny, nx)
        )
        ux = jnp.zeros((ny, nx))
        uy = jnp.zeros((ny, nx))
        phi = phi0
        for _ in range(200):
            phi = conservative_ac_step(phi, ux, uy, M=0.02, W=3.0)
        total0 = float(phi0.sum())
        total1 = float(phi.sum())
        assert abs(total1 - total0) / total0 < 0.02, (
            f"AC φ drift = {abs(total1 - total0) / total0:.4f} (>2%)"
        )

    def test_profile_shape(self):
        """Interface profile stays tanh-like (φ∈[0,1])."""
        nx, ny = 32, 32
        x = jnp.arange(nx, dtype=jnp.float64)
        phi0 = jnp.broadcast_to(
            tanh_interface_profile(x, x0=16.0, W=3.0)[None, :], (ny, nx)
        )
        ux = jnp.zeros((ny, nx))
        uy = jnp.zeros((ny, nx))
        phi = phi0
        for _ in range(100):
            phi = conservative_ac_step(phi, ux, uy, M=0.02, W=3.0)
        assert float(phi.min()) >= -0.1 and float(phi.max()) <= 1.1, "φ out of bound"
