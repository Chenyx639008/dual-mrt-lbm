#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LBM unit conversion utilities.

Purpose
-------
Convert between lattice units (L.U.) and physical units (P.U.)
under dynamic similarity constraints.

Base scales
-----------
- Length scale: X_r [m / LU]
- Time scale:   T_r [s / LU]
- Density scale: rho_r [kg/m^3 / LU]

Main relations
--------------
- x_pu   = x_lu * X_r
- t_pu   = t_lu * T_r
- rho_pu = rho_lu * rho_r
- u_pu   = u_lu * X_r / T_r
- nu_pu  = nu_lu * X_r^2 / T_r
- mu_pu  = rho_pu * nu_pu
- sigma_pu = sigma_lu * rho_r * X_r^3 / T_r^2

Notes
-----
1) mu (dynamic viscosity) in Pa.s, or mPa.s by multiplying 1e3.
2) sigma (surface tension) in N/m, or mN/m by multiplying 1e3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class EqPreset:
    """Single equilibrium-state preset (Table-like values)."""

    name: str
    pressure_mpa: float
    temperature_k: float

    # Physical properties (P.U.)
    rho_w_kg_m3: float
    rho_g_kg_m3: float
    mu_w_mpa_s: float
    mu_g_mpa_s: float
    sigma_mn_m: float

    # Lattice properties (L.U.)
    rho_w_lu: float
    rho_g_lu: float
    nu_w_lu: float
    nu_g_lu: float
    sigma_lu: float


@dataclass
class LBMUnitConverter:
    """Converter between lattice units and physical units."""

    x_r_m: float = 2.5e-7
    t_r_s: float = 1.5e-9
    rho_r_kg_m3: float = 134.22

    # ------------------ basic scale conversions ------------------
    def length_lu_to_pu(self, x_lu: float) -> float:
        return x_lu * self.x_r_m

    def length_pu_to_lu(self, x_m: float) -> float:
        return x_m / self.x_r_m

    def time_lu_to_pu(self, t_lu: float) -> float:
        return t_lu * self.t_r_s

    def time_pu_to_lu(self, t_s: float) -> float:
        return t_s / self.t_r_s

    def density_lu_to_pu(self, rho_lu: float) -> float:
        return rho_lu * self.rho_r_kg_m3

    def density_pu_to_lu(self, rho_kg_m3: float) -> float:
        return rho_kg_m3 / self.rho_r_kg_m3

    def velocity_lu_to_pu(self, u_lu: float) -> float:
        return u_lu * self.x_r_m / self.t_r_s

    def velocity_pu_to_lu(self, u_m_s: float) -> float:
        return u_m_s * self.t_r_s / self.x_r_m

    # ------------------ viscosity conversions ------------------
    def nu_lu_to_pu(self, nu_lu: float) -> float:
        """Kinematic viscosity: LU -> m^2/s."""
        return nu_lu * (self.x_r_m ** 2) / self.t_r_s

    def nu_pu_to_lu(self, nu_m2_s: float) -> float:
        """Kinematic viscosity: m^2/s -> LU."""
        return nu_m2_s * self.t_r_s / (self.x_r_m ** 2)

    def mu_from_rho_nu_pu(self, rho_kg_m3: float, nu_m2_s: float) -> float:
        """Dynamic viscosity in Pa.s from rho (kg/m^3) and nu (m^2/s)."""
        return rho_kg_m3 * nu_m2_s

    def mu_from_rho_nu_lu_to_pu(self, rho_lu: float, nu_lu: float) -> float:
        """Dynamic viscosity in Pa.s from lattice rho, nu."""
        rho_pu = self.density_lu_to_pu(rho_lu)
        nu_pu = self.nu_lu_to_pu(nu_lu)
        return self.mu_from_rho_nu_pu(rho_pu, nu_pu)

    @staticmethod
    def pa_s_to_mpa_s(mu_pa_s: float) -> float:
        return mu_pa_s * 1e3

    @staticmethod
    def mpa_s_to_pa_s(mu_mpa_s: float) -> float:
        return mu_mpa_s * 1e-3

    # ------------------ surface tension conversions ------------------
    def sigma_lu_to_pu(self, sigma_lu: float) -> float:
        """Surface tension: LU -> N/m."""
        return sigma_lu * self.rho_r_kg_m3 * (self.x_r_m ** 3) / (self.t_r_s ** 2)

    def sigma_pu_to_lu(self, sigma_n_m: float) -> float:
        """Surface tension: N/m -> LU."""
        return sigma_n_m * (self.t_r_s ** 2) / (self.rho_r_kg_m3 * (self.x_r_m ** 3))

    @staticmethod
    def n_m_to_mn_m(sigma_n_m: float) -> float:
        return sigma_n_m * 1e3

    @staticmethod
    def mn_m_to_n_m(sigma_mn_m: float) -> float:
        return sigma_mn_m * 1e-3

    # ------------------ helpers ------------------
    @staticmethod
    def reference_density_from_water(rho_w_pu: float, rho_w_lu: float) -> float:
        """rho_r from water density at one equilibrium state."""
        if rho_w_lu <= 0:
            raise ValueError("rho_w_lu must be > 0")
        return rho_w_pu / rho_w_lu

    def reference_density_from_surface_tension(self, sigma_pu_n_m: float, sigma_lu: float) -> float:
        """rho_r from surface tension matching with fixed X_r, T_r and sigma_lu."""
        if sigma_lu <= 0:
            raise ValueError("sigma_lu must be > 0")
        return sigma_pu_n_m * (self.t_r_s ** 2) / (sigma_lu * (self.x_r_m ** 3))

    def summary_from_lu(self, rho_lu: float, nu_lu: float, sigma_lu: float) -> Dict[str, float]:
        """Convenient one-shot conversion summary."""
        rho_pu = self.density_lu_to_pu(rho_lu)
        nu_pu = self.nu_lu_to_pu(nu_lu)
        mu_pa_s = self.mu_from_rho_nu_pu(rho_pu, nu_pu)
        sigma_n_m = self.sigma_lu_to_pu(sigma_lu)
        return {
            "rho_kg_m3": rho_pu,
            "nu_m2_s": nu_pu,
            "mu_pa_s": mu_pa_s,
            "mu_mpa_s": self.pa_s_to_mpa_s(mu_pa_s),
            "sigma_n_m": sigma_n_m,
            "sigma_mn_m": self.n_m_to_mn_m(sigma_n_m),
        }


def build_eq_presets() -> Dict[str, EqPreset]:
    """Eq1/Eq2 presets from your table values."""
    return {
        "Eq1": EqPreset(
            name="Eq1",
            pressure_mpa=7.270,
            temperature_k=283.1,
            rho_w_kg_m3=1003.1,
            rho_g_kg_m3=57.821,
            mu_w_mpa_s=1.3015,
            mu_g_mpa_s=0.012376,
            sigma_mn_m=63.91,
            rho_w_lu=7.4734,
            rho_g_lu=0.4352,
            nu_w_lu=0.03113,
            nu_g_lu=0.005133,
            sigma_lu=0.06927,
        ),
        "Eq2": EqPreset(
            name="Eq2",
            pressure_mpa=4.300,
            temperature_k=278.2,
            rho_w_kg_m3=1002.0,
            rho_g_kg_m3=32.91,
            mu_w_mpa_s=1.5099,
            mu_g_mpa_s=0.011262,
            sigma_mn_m=66.68,
            rho_w_lu=7.2744,
            rho_g_lu=0.2373,
            nu_w_lu=0.03613,
            nu_g_lu=0.0082,
            sigma_lu=0.06923,
        ),
    }


def recommended_sigma_anchor_parameters(preset: EqPreset,
                                        x_r_m: float = 2.5e-7,
                                        t_r_s: float = 1.5e-9) -> Dict[str, float]:
    """
    Recommended parameters when enforcing sigma consistency as primary anchor:
      - solve rho_r from sigma
      - convert physical phase densities to lattice densities
    """
    conv = LBMUnitConverter(x_r_m=x_r_m, t_r_s=t_r_s, rho_r_kg_m3=1.0)
    rho_r = conv.reference_density_from_surface_tension(
        sigma_pu_n_m=preset.sigma_mn_m * 1e-3,
        sigma_lu=preset.sigma_lu,
    )
    conv2 = LBMUnitConverter(x_r_m=x_r_m, t_r_s=t_r_s, rho_r_kg_m3=rho_r)
    return {
        "rho_r_kg_m3": rho_r,
        "rho_w_lu": conv2.density_pu_to_lu(preset.rho_w_kg_m3),
        "rho_g_lu": conv2.density_pu_to_lu(preset.rho_g_kg_m3),
    }


def converter_from_preset(preset: EqPreset, x_r_m: float = 2.5e-7, t_r_s: float = 1.5e-9) -> LBMUnitConverter:
    """Create a converter with rho_r inferred from preset water density."""
    rho_r = LBMUnitConverter.reference_density_from_water(preset.rho_w_kg_m3, preset.rho_w_lu)
    return LBMUnitConverter(x_r_m=x_r_m, t_r_s=t_r_s, rho_r_kg_m3=rho_r)


def check_preset_consistency(preset: EqPreset, x_r_m: float = 2.5e-7, t_r_s: float = 1.5e-9) -> Dict[str, Any]:
    """Return converted values and relative errors vs table P.U. values."""
    conv = converter_from_preset(preset, x_r_m=x_r_m, t_r_s=t_r_s)

    rho_w = conv.density_lu_to_pu(preset.rho_w_lu)
    rho_g = conv.density_lu_to_pu(preset.rho_g_lu)

    mu_w = conv.pa_s_to_mpa_s(conv.mu_from_rho_nu_lu_to_pu(preset.rho_w_lu, preset.nu_w_lu))
    mu_g = conv.pa_s_to_mpa_s(conv.mu_from_rho_nu_lu_to_pu(preset.rho_g_lu, preset.nu_g_lu))

    sigma = conv.n_m_to_mn_m(conv.sigma_lu_to_pu(preset.sigma_lu))

    def rel_err(pred: float, ref: float) -> float:
        if ref == 0:
            return 0.0
        return (pred - ref) / ref

    return {
        "name": preset.name,
        "rho_r_kg_m3": conv.rho_r_kg_m3,
        "pred": {
            "rho_w_kg_m3": rho_w,
            "rho_g_kg_m3": rho_g,
            "mu_w_mpa_s": mu_w,
            "mu_g_mpa_s": mu_g,
            "sigma_mn_m": sigma,
        },
        "ref": {
            "rho_w_kg_m3": preset.rho_w_kg_m3,
            "rho_g_kg_m3": preset.rho_g_kg_m3,
            "mu_w_mpa_s": preset.mu_w_mpa_s,
            "mu_g_mpa_s": preset.mu_g_mpa_s,
            "sigma_mn_m": preset.sigma_mn_m,
        },
        "rel_err": {
            "rho_w": rel_err(rho_w, preset.rho_w_kg_m3),
            "rho_g": rel_err(rho_g, preset.rho_g_kg_m3),
            "mu_w": rel_err(mu_w, preset.mu_w_mpa_s),
            "mu_g": rel_err(mu_g, preset.mu_g_mpa_s),
            "sigma": rel_err(sigma, preset.sigma_mn_m),
        },
    }


def check_anchor_sensitivity(preset: EqPreset,
                             x_r_m: float = 2.5e-7,
                             t_r_s: float = 1.5e-9) -> Dict[str, Any]:
    """
    Compare two anchoring choices for rho_r:
    1) water-density anchored
    2) surface-tension anchored
    """
    # water anchor
    rho_r_w = LBMUnitConverter.reference_density_from_water(preset.rho_w_kg_m3, preset.rho_w_lu)

    # sigma anchor
    sigma_n_m = preset.sigma_mn_m * 1e-3
    base = LBMUnitConverter(x_r_m=x_r_m, t_r_s=t_r_s, rho_r_kg_m3=1.0)
    rho_r_s = base.reference_density_from_surface_tension(sigma_n_m, preset.sigma_lu)

    def _calc(rho_r: float) -> Dict[str, float]:
        conv = LBMUnitConverter(x_r_m=x_r_m, t_r_s=t_r_s, rho_r_kg_m3=rho_r)
        rho_w = conv.density_lu_to_pu(preset.rho_w_lu)
        rho_g = conv.density_lu_to_pu(preset.rho_g_lu)
        mu_w = conv.pa_s_to_mpa_s(conv.mu_from_rho_nu_lu_to_pu(preset.rho_w_lu, preset.nu_w_lu))
        mu_g = conv.pa_s_to_mpa_s(conv.mu_from_rho_nu_lu_to_pu(preset.rho_g_lu, preset.nu_g_lu))
        sigma = conv.n_m_to_mn_m(conv.sigma_lu_to_pu(preset.sigma_lu))
        return {
            "rho_w_kg_m3": rho_w,
            "rho_g_kg_m3": rho_g,
            "mu_w_mpa_s": mu_w,
            "mu_g_mpa_s": mu_g,
            "sigma_mn_m": sigma,
        }

    return {
        "name": preset.name,
        "rho_r_water_anchor": rho_r_w,
        "rho_r_sigma_anchor": rho_r_s,
        "delta_rho_r_pct": (rho_r_s - rho_r_w) / rho_r_w * 100.0,
        "pred_water_anchor": _calc(rho_r_w),
        "pred_sigma_anchor": _calc(rho_r_s),
    }


def _demo() -> None:
    presets = build_eq_presets()

    print("=== Recommended defaults (sigma-anchored) ===")
    for name, p in presets.items():
        rec = recommended_sigma_anchor_parameters(p)
        print(f"[{name}] rho_r={rec['rho_r_kg_m3']:.6f}, "
              f"rho_w_lu={rec['rho_w_lu']:.6f}, rho_g_lu={rec['rho_g_lu']:.6f}")

    print("\n=== Current preset consistency (water-anchored check) ===")
    for name, p in presets.items():
        report = check_preset_consistency(p)
        print(f"[{name}] rho_r = {report['rho_r_kg_m3']:.3f} kg/m^3")
        print("  predicted:")
        for k, v in report["pred"].items():
            print(f"    {k}: {v:.6g}")
        print("  relative errors:")
        for k, v in report["rel_err"].items():
            print(f"    {k}: {v*100:.3f}%")

    print("\n=== Anchor sensitivity (water vs surface tension) ===")
    for name, p in presets.items():
        s = check_anchor_sensitivity(p)
        print(f"[{name}] rho_r(water)={s['rho_r_water_anchor']:.3f}, "
              f"rho_r(sigma)={s['rho_r_sigma_anchor']:.3f}, "
              f"delta={s['delta_rho_r_pct']:+.3f}%")


if __name__ == "__main__":
    _demo()
