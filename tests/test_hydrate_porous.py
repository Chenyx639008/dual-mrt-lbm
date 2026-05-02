"""Porous-media hydrate dissociation case validation tests.

Tests cover:
1. hydrate_porous.yaml -> flat params dict conversion and key values
2. thermal_bc_side parameter propagation (Python config + params.txt roundtrip)
3. Derived lattice quantities for the porous-media config
4. Geometry parameter consistency (morph, r_obs, coat_thick, r_mid)
5. Temperature-dependent dissociation rate ratio: k_r(T_hot) / k_r(T_cold) >> 1
6. Diagnostic figure: parameter overview for the porous case
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import pytest

from lbm_mrt.core.config import load_config, override
from lbm_mrt.io.params_writer import write_params_txt

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = REPO_ROOT / "configs" / "default.yaml"
POROUS_CFG = REPO_ROOT / "configs" / "hydrate_porous.yaml"

Params = dict[str, object]


def _f(params: Params, key: str) -> float:
    return float(cast(float | int | bool | str, params[key]))


def _i(params: Params, key: str) -> int:
    return int(cast(float | int | str, params[key]))


def _b(params: Params, key: str) -> bool:
    return bool(cast(bool | int | float | str, params[key]))


def _load_porous() -> Params:
    return cast(Params, load_config(POROUS_CFG))


def _parse_params_txt(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split(maxsplit=1)
        parsed[key] = value
    return parsed


# ── 1. Config Loading ────────────────────────────────────────────────────────

def test_porous_config_loads() -> None:
    """hydrate_porous.yaml must load without error."""
    params = _load_porous()
    assert isinstance(params, dict)
    assert len(params) > 0


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("hydrate_enable",     True),
        ("hydrate_start_step", 0),
        ("T0_init",            274.15),
        ("T0_inlet",           298.15),
        ("thermal_bc_side",    3),        # right boundary
        ("thermal_init_mode",  1),        # linear gradient by default for porous case
        ("lambda_fluid",       0.6),
        ("lambda_hydrate",     0.49),
        ("lambda_solid",       0.9),
        ("rhocp_fluid",        4.2e6),
        ("rhocp_hydrate",      2.1e6),
        ("rhocp_solid",        2.0e6),
        ("D_mol_water",        1.85e-9),
        ("Henry_KH",           0.1),
        ("Cm_init",            0.0),
        ("k0_rxn",             36000.0),
        ("Ea_rxn",             97500.0),
        ("e1_peq",             33.12),
        ("e2_peq",             -9005.5),
        ("latent_heat",        43000.0),
        ("Vm_hydrate",         2.274e-5),
        ("Vh_init",            1.0),
        ("vop_terminate_frac", 0.01),
        ("dx_phys",            1.0e-5),
        ("dt_phys",            1.0e-6),
    ],
)
def test_porous_config_values(key: str, expected: float | bool) -> None:
    """All hydrate_porous.yaml values must land in the flat dict unchanged."""
    params = _load_porous()
    if isinstance(expected, bool):
        assert _b(params, key) is expected
    else:
        assert _f(params, key) == pytest.approx(expected)


# ── 2. thermal_bc_side Propagation ──────────────────────────────────────────

def test_thermal_bc_side_default_sphere() -> None:
    """hydrate_sphere_hot.yaml should have thermal_bc_side=0 (bottom)."""
    sphere_cfg = REPO_ROOT / "configs" / "hydrate_sphere_hot.yaml"
    params = cast(Params, load_config(sphere_cfg))
    assert _i(params, "thermal_bc_side") == 0


def test_thermal_bc_side_porous_is_right() -> None:
    """hydrate_porous.yaml must set thermal_bc_side=3 (right boundary)."""
    params = _load_porous()
    assert _i(params, "thermal_bc_side") == 3


def test_thermal_bc_side_override() -> None:
    """override() must correctly update thermal_bc_side."""
    params = _load_porous()
    updated = override(params, thermal_bc_side=1)
    assert _i(updated, "thermal_bc_side") == 1
    # Original must not be mutated
    assert _i(params, "thermal_bc_side") == 3


def test_thermal_bc_side_params_txt_roundtrip(tmp_path: Path) -> None:
    """thermal_bc_side=3 must survive write → parse roundtrip via params.txt."""
    params = _load_porous()
    out = tmp_path / "params.txt"
    write_params_txt(cast(dict[str, object], params), out)
    parsed = _parse_params_txt(out)
    assert parsed["thermal_bc_side"] == "3"
    assert parsed["thermal_init_mode"] == "1"
    assert parsed["T0_init"] == "274.15"
    assert parsed["T0_inlet"] == "298.15"


# ── 3. Geometry Parameters ───────────────────────────────────────────────────

def test_porous_script_geometry_params() -> None:
    """Script-level geometry constants must be self-consistent."""
    # Import the script's constants without executing main()
    import importlib.util, sys

    script_path = REPO_ROOT / "scripts" / "05_run_porous_hydrate_case.py"
    spec = importlib.util.spec_from_file_location("_porous_script", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Prevent __main__ block from running
    mod.__name__ = "_porous_script"
    sys.modules["_porous_script"] = mod
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except SystemExit:
        pass

    assert mod.R_OBS      == 20
    assert mod.L_GAP      == 20
    assert mod.COAT_THICK == 4
    assert mod.R_MID      == 8
    assert mod.MORPH_COATING  == 2
    assert mod.MORPH_POREFILL == 1
    assert mod.DOMAIN_NX  == 339
    assert mod.DOMAIN_NY  == 212


# ── 4. Lattice Quantity Derivation ───────────────────────────────────────────

def _derive_porous_lattice(params: Params) -> dict[str, float]:
    """Reproduce init_device_variable_hydrate() conversion for porous config."""
    dx = _f(params, "dx_phys")
    dt = _f(params, "dt_phys")
    return {
        "alpha_fluid":    _f(params, "lambda_fluid")   / _f(params, "rhocp_fluid")   * dt / dx**2,
        "alpha_hydrate":  _f(params, "lambda_hydrate") / _f(params, "rhocp_hydrate") * dt / dx**2,
        "alpha_solid":    _f(params, "lambda_solid")   / _f(params, "rhocp_solid")   * dt / dx**2,
        "D_latt":         _f(params, "D_mol_water") * dt / dx**2,
        "k0_latt":        _f(params, "k0_rxn") * dt / dx,
        "Ea_over_R":      _f(params, "Ea_rxn") / 8.314,
        "Vm_latt":        _f(params, "Vm_hydrate") / dx**3,
        "latent_H_latt":  _f(params, "latent_heat") / (_f(params, "rhocp_fluid") * dx),
    }


def test_porous_lattice_alpha_fluid() -> None:
    """alpha_fluid lattice value: λ/(ρcp) * dt/dx² with porous config params."""
    params = _load_porous()
    derived = _derive_porous_lattice(params)
    # λ=0.6, ρcp=4.2e6, dt=1e-6, dx=1e-5 → 0.6/4.2e6 * 1e-6/1e-10 = 1.4286e-3
    assert derived["alpha_fluid"] == pytest.approx(1.4285714286e-3, rel=1e-9)


def test_porous_lattice_alpha_hydrate() -> None:
    params = _load_porous()
    derived = _derive_porous_lattice(params)
    # λ=0.49, ρcp=2.1e6 → 0.49/2.1e6 * 1e-6/1e-10 = 2.3333e-3
    assert derived["alpha_hydrate"] == pytest.approx(2.3333333333e-3, rel=1e-9)


def test_porous_lattice_D_latt() -> None:
    params = _load_porous()
    derived = _derive_porous_lattice(params)
    assert derived["D_latt"] == pytest.approx(1.85e-5, rel=1e-12)


def test_porous_lattice_k0_latt() -> None:
    params = _load_porous()
    derived = _derive_porous_lattice(params)
    # k0=3.6e4, dt=1e-6, dx=1e-5 → 3.6e4 * 1e-6/1e-5 = 3.6e3
    assert derived["k0_latt"] == pytest.approx(3.6e3, rel=1e-12)


# ── 5. Temperature Sensitivity of Kim-Bishnoi Rate ───────────────────────────

def _kim_bishnoi_k_r(T: float, k0_latt: float, Ea_over_R: float,
                     e1: float, e2: float, Cm: float = 0.0) -> float:
    """Evaluate k_r at temperature T (same formula as hydrate.cu:721)."""
    Csat = math.exp(e1 + e2 / T)
    driving = max(0.0, 1.0 - Cm / (Csat + 1e-30))
    return k0_latt * math.exp(-Ea_over_R / T) * driving


def test_dissociation_rate_increases_with_temperature() -> None:
    """k_r(T_hot) must be substantially larger than k_r(T_cold).

    Physical expectation: Arrhenius factor increases ~30× from 274 K to 298 K
    at Ea=97500 J/mol.  Even after accounting for Csat(T) change, the net rate
    should be significantly higher at T_hot.
    """
    params = _load_porous()
    derived = _derive_porous_lattice(params)

    T_cold = _f(params, "T0_init")    # 274.15 K
    T_hot  = _f(params, "T0_inlet")   # 298.15 K
    e1     = _f(params, "e1_peq")
    e2     = _f(params, "e2_peq")
    k0     = derived["k0_latt"]
    Ea_R   = derived["Ea_over_R"]

    k_cold = _kim_bishnoi_k_r(T_cold, k0, Ea_R, e1, e2)
    k_hot  = _kim_bishnoi_k_r(T_hot,  k0, Ea_R, e1, e2)

    # The hot rate must exceed the cold rate by at least one order of magnitude
    assert k_hot > k_cold * 10, (
        f"Expected k_r(T_hot={T_hot}) >> k_r(T_cold={T_cold}), "
        f"got ratio = {k_hot / (k_cold + 1e-300):.2f}"
    )


def test_dissociation_rate_zero_below_equilibrium() -> None:
    """When Cm >= Csat the driving force is zero (no dissociation)."""
    params = _load_porous()
    derived = _derive_porous_lattice(params)

    T     = _f(params, "T0_init")
    e1    = _f(params, "e1_peq")
    e2    = _f(params, "e2_peq")
    Csat  = math.exp(e1 + e2 / T)

    k_r_over = _kim_bishnoi_k_r(
        T,
        derived["k0_latt"], derived["Ea_over_R"], e1, e2,
        Cm=Csat * 2.0,   # supersaturated → no driving force
    )
    assert k_r_over == pytest.approx(0.0, abs=1e-30)


def test_arrhenius_ratio_cold_to_hot() -> None:
    """Pure Arrhenius factor from 274.15 K → 298.15 K should be ~30.

    Reference: exp(Ea/R * (1/T_cold - 1/T_hot)) with Ea=97500, R=8.314.
    """
    params = _load_porous()
    Ea     = _f(params, "Ea_rxn")
    T_cold = _f(params, "T0_init")
    T_hot  = _f(params, "T0_inlet")

    ratio = math.exp(Ea / 8.314 * (1.0 / T_cold - 1.0 / T_hot))
    # Numerical check: ~exp(11727*(1/274.15 - 1/298.15)) ≈ exp(3.45) ≈ 31.5
    assert ratio == pytest.approx(math.exp(97500 / 8.314 * (1 / T_cold - 1 / T_hot)),
                                  rel=1e-9)
    assert ratio > 10.0, f"Arrhenius ratio expected >10, got {ratio:.2f}"


# ── 6. params.txt Full Roundtrip ─────────────────────────────────────────────

def test_porous_params_txt_roundtrip(tmp_path: Path) -> None:
    """All key porous params must survive write → parse via params.txt."""
    params = _load_porous()
    out = tmp_path / "params.txt"
    write_params_txt(cast(dict[str, object], params), out)
    parsed = _parse_params_txt(out)

    assert parsed["hydrate_enable"]      == "true"
    assert parsed["T0_init"]             == "274.15"
    assert parsed["T0_inlet"]            == "298.15"
    assert parsed["thermal_bc_side"]     == "3"
    assert parsed["thermal_init_mode"]   == "1"
    assert parsed["lambda_hydrate"]    == "0.49"
    assert parsed["D_mol_water"]       == "1.85e-09"
    assert parsed["Vm_hydrate"]        == "2.274e-05"
    assert parsed["vop_terminate_frac"] == "0.01"


# ── 7. Sphere vs Porous Config Isolation ────────────────────────────────────

def test_sphere_hot_and_porous_configs_independent() -> None:
    """The two config files must differ on T0_inlet, T0_init, and thermal_bc_side."""
    sphere_cfg   = REPO_ROOT / "configs" / "hydrate_sphere_hot.yaml"
    sphere_params = cast(Params, load_config(sphere_cfg))
    porous_params = _load_porous()

    assert _f(sphere_params, "T0_inlet") != _f(porous_params, "T0_inlet")
    assert _f(sphere_params, "T0_init")  != _f(porous_params, "T0_init")
    assert _i(sphere_params, "thermal_bc_side") == 0   # bottom
    assert _i(porous_params, "thermal_bc_side") == 3   # right
