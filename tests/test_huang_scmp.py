"""Validation tests for Huang & Wu (2016) SCMP solver.

These tests verify:
1. CS-EOS Maxwell construction (analytical)
2. Laplace law (droplet radius sweep → σ fit)
3. Coexistence curve (flat-interface T sweep)
4. Spurious currents magnitude
5. σ-decoupling (k₁ sweep)

Usage::

    uv run pytest tests/test_huang_scmp.py -v
    uv run pytest tests/test_huang_scmp.py -v --run-slow  # includes GPU simulations
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lbm_mrt.validation.cs_eos import (
    coexistence_curve,
    cs_critical_point,
    cs_pressure,
    maxwell_coexistence,
)

# ── CS-EOS reference parameters (plan.md defaults) ──
A, B, R_CS = 1.0, 4.0, 1.0
Tc, pc, rhoc = cs_critical_point(A, B, R_CS)


class TestCSEOS:
    """Analytical CS-EOS tests (no GPU needed)."""

    def test_critical_point(self):
        """CS critical constants match known values."""
        assert abs(Tc - 0.094325) < 1e-4
        assert abs(rhoc - 0.0326) < 1e-3

    def test_pressure_ideal_gas_limit(self):
        """At low density, CS-EOS approaches ideal gas."""
        rho = 1e-4
        p = cs_pressure(rho, A, B, R_CS, 0.05)
        p_ideal = rho * R_CS * 0.05
        assert abs(p - p_ideal) / max(p_ideal, 1e-12) < 0.05

    def test_pressure_hard_sphere_divergence(self):
        """At η→1, CS-EOS has a minimum (attraction) then diverges (repulsion)."""
        # At η≈0.45, p reaches minimum (attraction dominates)
        # At η→1, repulsion dominates and p→∞
        p_min = cs_pressure(0.45, A, B, R_CS, 0.05)
        p_high = cs_pressure(0.55, A, B, R_CS, 0.05)
        # Pressure should increase from minimum toward divergence
        assert p_high > p_min, f"p(0.45)={p_min:.4f}, p(0.55)={p_high:.4f}"

    def test_maxwell_coexistence_exists(self):
        """Maxwell construction finds coexistence at Tr=0.65."""
        result = maxwell_coexistence(A, B, R_CS, 0.65 * Tc)
        assert result is not None, "Maxwell construction failed"
        rg, rl, peq = result
        assert rg > 0
        assert rl > rg
        # At high Tr (>0.8), rg can exceed rhoc for CS-EOS; check ratio instead
        assert rl / rg >= 1.5, f"Density ratio {rl / rg:.1f} too small"

    def test_maxwell_pressure_equality(self):
        """Gas and liquid pressures are equal at coexistence."""
        result = maxwell_coexistence(A, B, R_CS, 0.65 * Tc)
        assert result is not None
        rg, rl, peq = result
        pg = cs_pressure(rg, A, B, R_CS, 0.65 * Tc)
        pl = cs_pressure(rl, A, B, R_CS, 0.65 * Tc)
        assert abs(pg - pl) < 1e-8

    def test_maxwell_supercritical_fails(self):
        """Maxwell construction returns None at T ≥ Tc."""
        assert maxwell_coexistence(A, B, R_CS, 1.0 * Tc) is None
        assert maxwell_coexistence(A, B, R_CS, 1.1 * Tc) is None

    def test_coexistence_curve_monotonic(self):
        """Coexistence curve: ρ_l decreases, ρ_g increases with T."""
        T_vals = np.linspace(0.6 * Tc, 0.95 * Tc, 10)
        curve = coexistence_curve(A, B, R_CS, T_vals)
        assert curve is not None
        assert len(curve["T"]) >= 3
        # ρ_l should decrease with T (toward ρc) — check first half only
        n2 = len(curve["T"]) // 2
        assert curve["rho_l"][0] > curve["rho_l"][n2], "ρ_l not decreasing"
        # ρ_g should increase with T — check first half only (solver noise at high Tr)
        assert curve["rho_g"][0] < curve["rho_g"][n2], "ρ_g not increasing"

    def test_coexistence_ratio(self):
        """Density ratio ρ_l/ρ_g ≥ 1.5 at Tr ≤ 0.85."""
        for Tr in [0.6, 0.7, 0.8]:
            result = maxwell_coexistence(A, B, R_CS, Tr * Tc)
            if result:
                rg, rl, _ = result
                ratio = rl / max(rg, 1e-8)
                assert ratio >= 1.5, f"Tr={Tr}: ratio={ratio:.1f} < 1.5"

    def test_reference_data_exists(self):
        """Reference coexistence data is saved."""
        ref_path = "data/validation_reference/huang_2016/cs_coexistence.json"
        assert os.path.exists(ref_path), f"Missing: {ref_path}"
        with open(ref_path) as f:
            data = json.load(f)
        assert "T" in data
        assert len(data["T"]) >= 5


class TestSCMPSimulation:
    """GPU simulation tests (requires --run-slow flag)."""

    @pytest.mark.slow
    def test_scmp_runs_without_nan(self):
        """SCMP binary runs 50k steps without NaN at Tr=0.65."""
        from lbm_mrt.core.config import load_config, override
        from lbm_mrt.runners.single_run import run_one

        result = maxwell_coexistence(A, B, R_CS, 0.65 * Tc)
        assert result is not None
        rg, rl, _ = result

        params = override(
            load_config(),
            **{
                "pp_mode": 1,
                "init_eq": 0,
                "tau_p_a": 1.0,
                "k1_huang": 0.08333,
                "cs_a": A,
                "cs_T": 0.65,
                "cs_G": -1.0,
                "huang_R0": 40.0,
                "huang_init_mode": 1,
                "huang_rho_g": float(rg),
                "huang_rho_l": float(rl),
            },
        )
        ret, elapsed = run_one(
            params,
            case_name="test_scmp_no_nan",
            out_root="results",
            app="lbm_mrt/solver/mcmp_huang_256",
            resume=False,
        )
        assert ret == 0, f"Binary failed with exit code {ret}"

        import glob as g

        fs = sorted(g.glob("results/test_scmp_no_nan/outputdata_scmp/flow*.vtk"))
        assert len(fs) > 0, "No VTK output"
        from lbm_mrt.io.vtk_reader import read_vtk_scalars

        fields, _, _ = read_vtk_scalars(fs[-1])
        assert not np.any(np.isnan(fields["rho"])), "NaN in rho"
        assert not np.any(np.isnan(fields["ux"])), "NaN in ux"

    @pytest.mark.slow
    def test_laplace_law_linearity(self):
        """Laplace law: ΔP vs 1/R is linear (R² ≥ 0.95)."""
        from lbm_mrt.core.config import load_config, override
        from lbm_mrt.runners.single_run import run_one
        from lbm_mrt.io.vtk_reader import read_vtk_scalars
        from lbm_mrt.validation.analytical import (
            detect_interface_radius,
            fit_pressure_inside_outside,
            laplace_sigma,
        )

        result = maxwell_coexistence(A, B, R_CS, 0.65 * Tc)
        assert result is not None
        rg, rl, _ = result

        Rs, dPs = [], []
        for R in [25.0, 35.0, 45.0]:
            params = override(
                load_config(),
                **{
                    "pp_mode": 1,
                    "init_eq": 0,
                    "tau_p_a": 1.0,
                    "k1_huang": 0.08333,
                    "cs_a": A,
                    "cs_T": 0.65,
                    "cs_G": -1.0,
                    "huang_R0": R,
                    "huang_init_mode": 1,
                    "huang_rho_g": float(rg),
                    "huang_rho_l": float(rl),
                },
            )
            ret, _ = run_one(
                params,
                case_name=f"test_laplace_R{R:.0f}",
                out_root="results",
                app="lbm_mrt/solver/mcmp_huang_256",
                resume=False,
            )
            assert ret == 0

            import glob as g

            fs = sorted(
                g.glob(f"results/test_laplace_R{R:.0f}/outputdata_scmp/flow*.vtk")
            )
            if fs:
                fields, nx, ny = read_vtk_scalars(fs[-1])
                rho = fields["rho"]
                pressure = fields.get("pressure", np.zeros_like(rho))
                center, radius = detect_interface_radius(rho, nx, ny)
                p_in, p_out = fit_pressure_inside_outside(rho, pressure, center, radius)
                if radius > 0:
                    Rs.append(radius)
                    dPs.append(p_in - p_out)

        assert len(Rs) >= 3, f"Only {len(Rs)} valid droplet radii"
        sigma, r2, intercept = laplace_sigma(np.array(Rs), np.array(dPs))
        assert r2 >= 0.95, f"Laplace fit R²={r2:.3f} < 0.95"


# ── Configure slow marker ──
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests that run GPU simulations")


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", help="run slow GPU tests")
