"""Hydrate-specific validation tests.

These tests focus on two things:
1. hydrate.yaml -> flat params dict conversion
2. derived lattice quantities used by hydrate.cu / hydrate_vop.cu

The plotting test saves a diagnostic figure through lbm_mrt.viz.viz_template
so the conversion chain can be inspected visually without running a full GPU case.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest

from lbm_mrt.core.config import load_config
from lbm_mrt.io.params_writer import write_params_txt
from lbm_mrt.viz import viz_template as V

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = REPO_ROOT / "configs" / "default.yaml"
HYDRATE_CFG = REPO_ROOT / "configs" / "hydrate.yaml"


HydrateParams = dict[str, object]


def _float_param(params: HydrateParams, key: str) -> float:
    """Read a numeric parameter as float from the flattened config dict."""
    return float(cast(float | int | bool | str, params[key]))


def _bool_param(params: HydrateParams, key: str) -> bool:
    """Read a boolean parameter from the flattened config dict."""
    return bool(cast(bool | int | float | str, params[key]))


def _load_configs() -> tuple[dict[str, object], HydrateParams]:
    """Load the default config and the hydrate overlay config."""
    base = cast(dict[str, object], load_config(DEFAULT_CFG))
    hydrate = cast(HydrateParams, load_config(HYDRATE_CFG))
    return base, hydrate


def _compute_hydrate_lattice_quantities(
    params: HydrateParams,
) -> dict[str, float]:
    """Reproduce the lattice-unit conversion used by hydrate.cu.

    The formulas mirror init_device_variable_hydrate():
    - alpha = lambda / rhocp * dt / dx^2
    - D_latt = D * dt / dx^2
    - k0_latt = k0 * dt / dx
    - Ea_over_R = Ea / 8.314
    - Vm_latt = Vm / dx^3
    - latent_H_latt = latent_heat / (rhocp_fluid * dx)
    """
    dx = _float_param(params, "dx_phys")
    dt = _float_param(params, "dt_phys")

    alpha_fluid = (
        _float_param(params, "lambda_fluid")
        / _float_param(params, "rhocp_fluid")
        * dt
        / (dx * dx)
    )
    alpha_hydrate = (
        _float_param(params, "lambda_hydrate")
        / _float_param(params, "rhocp_hydrate")
        * dt
        / (dx * dx)
    )
    alpha_solid = (
        _float_param(params, "lambda_solid")
        / _float_param(params, "rhocp_solid")
        * dt
        / (dx * dx)
    )
    d_latt = _float_param(params, "D_mol_water") * dt / (dx * dx)
    k0_latt = _float_param(params, "k0_rxn") * dt / dx
    ea_over_r = _float_param(params, "Ea_rxn") / 8.314
    vm_latt = _float_param(params, "Vm_hydrate") / (dx * dx * dx)
    latent_h_latt = _float_param(params, "latent_heat") / (
        _float_param(params, "rhocp_fluid") * dx
    )

    return {
        "alpha_fluid": alpha_fluid,
        "alpha_hydrate": alpha_hydrate,
        "alpha_solid": alpha_solid,
        "D_latt": d_latt,
        "k0_latt": k0_latt,
        "Ea_over_R": ea_over_r,
        "Vm_latt": vm_latt,
        "latent_H_latt": latent_h_latt,
    }


def _parse_params_txt(path: Path) -> dict[str, str]:
    """Read a params.txt file back into a key-value mapping."""
    parsed: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split(maxsplit=1)
        parsed[key] = value
    return parsed


def _make_conversion_figure(
    params: HydrateParams, derived: dict[str, float], out_dir: Path
) -> Path:
    """Render a compact conversion report using viz_template."""
    V.init_style(
        mode="thesis_cn", base_fontsize=12, bold=True, axis_linewidth=1.5, verbose=False
    )

    fig, axes = V.plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True)

    source_names = [
        "T0_init",
        "T0_inlet",
        "lambda_fluid",
        "lambda_hydrate",
        "lambda_solid",
        "D_mol_water",
        "k0_rxn",
        "Vm_hydrate",
    ]
    source_values = [params[name] for name in source_names]

    derived_names = [
        "alpha_fluid",
        "alpha_hydrate",
        "alpha_solid",
        "D_latt",
        "k0_latt",
        "Ea_over_R",
        "Vm_latt",
        "latent_H_latt",
    ]
    derived_values = [derived[name] for name in derived_names]

    y0 = np.arange(len(source_names))
    y1 = np.arange(len(derived_names))

    axes[0].barh(y0, source_values, color="#4C72B0")
    axes[0].set_yticks(y0)
    axes[0].set_yticklabels(source_names)
    axes[0].set_xscale("log")
    V.set_title(axes[0], "hydrate.yaml 原始物理参数", fontsize=13, lang="cn")
    V.set_xlabel(axes[0], "数值（对数坐标）", fontsize=12, lang="cn")
    V.format_axes(axes[0], tick_font="en", tick_labelsize=10)

    axes[1].barh(y1, derived_values, color="#55A868")
    axes[1].set_yticks(y1)
    axes[1].set_yticklabels(derived_names)
    axes[1].set_xscale("log")
    V.set_title(axes[1], "hydrate.cu 中使用的格子/派生量", fontsize=13, lang="cn")
    V.set_xlabel(axes[1], "数值（对数坐标）", fontsize=12, lang="cn")
    V.format_axes(axes[1], tick_font="en", tick_labelsize=10)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hydrate_conversion_report.png"
    V.save_figure(fig, out_path, dpi=180)
    V.plt.close(fig)
    return out_path


def test_hydrate_overlay_keeps_default_keys() -> None:
    """hydrate.yaml should override only the hydrate section, not base defaults."""
    base, hydrate = _load_configs()

    for key in ["Sw", "tau_p_a", "tau_p_b", "Gx", "drive_mode", "ENABLE_CKPT"]:
        assert hydrate[key] == base[key]


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("hydrate_enable", True),
        ("hydrate_start_step", 0),
        ("T0_init", 278.15),
        ("T0_inlet", 285.0),
        ("lambda_fluid", 0.6),
        ("lambda_hydrate", 0.49),
        ("lambda_solid", 0.9),
        ("rhocp_fluid", 4.2e6),
        ("rhocp_hydrate", 2.1e6),
        ("rhocp_solid", 2.0e6),
        ("D_mol_water", 1.85e-9),
        ("Henry_KH", 0.1),
        ("Cm_init", 0.0),
        ("k0_rxn", 36000.0),
        ("Ea_rxn", 97500.0),
        ("e1_peq", 33.12),
        ("e2_peq", -9005.5),
        ("latent_heat", 43000.0),
        ("Vm_hydrate", 2.274e-5),
        ("Vh_init", 1.0),
        ("vop_terminate_frac", 0.01),
        ("dx_phys", 1.0e-5),
        ("dt_phys", 1.0e-6),
    ],
)
def test_hydrate_overlay_values_match_yaml(key: str, expected: float | bool) -> None:
    """hydrate.yaml values should land in the flattened params dict unchanged."""
    _, hydrate = _load_configs()
    if isinstance(expected, bool):
        assert _bool_param(hydrate, key) is expected
    else:
        assert _float_param(hydrate, key) == pytest.approx(expected)


def test_hydrate_lattice_conversion_formulas_match_source() -> None:
    """Derived lattice quantities should match the formulas used by hydrate.cu."""
    _, hydrate = _load_configs()
    derived = _compute_hydrate_lattice_quantities(hydrate)

    assert derived["alpha_fluid"] == pytest.approx(1.4285714286e-3, rel=1e-10)
    assert derived["alpha_hydrate"] == pytest.approx(2.3333333333e-3, rel=1e-10)
    assert derived["alpha_solid"] == pytest.approx(4.5e-3, rel=1e-10)
    assert derived["D_latt"] == pytest.approx(1.85e-5, rel=1e-12)
    assert derived["k0_latt"] == pytest.approx(3.6e3, rel=1e-12)
    assert derived["Ea_over_R"] == pytest.approx(11727.2071205, rel=1e-9)
    assert derived["Vm_latt"] == pytest.approx(2.274e10, rel=1e-12)
    assert derived["latent_H_latt"] == pytest.approx(1.0238095238e3, rel=1e-10)


def test_hydrate_params_txt_roundtrip(tmp_path: Path) -> None:
    """The flattened hydrate params should serialize cleanly to params.txt."""
    _, hydrate = _load_configs()
    out = tmp_path / "params.txt"
    write_params_txt(cast(dict[str, object], hydrate), out)

    parsed = _parse_params_txt(out)
    assert parsed["hydrate_enable"] == "true"
    assert parsed["T0_init"] == "278.15"
    assert parsed["T0_inlet"] == "285"
    assert parsed["lambda_hydrate"] == "0.49"
    assert parsed["D_mol_water"] == "1.85e-09"
    assert parsed["Vm_hydrate"] == "2.274e-05"
    assert parsed["vop_terminate_frac"] == "0.01"


def test_hydrate_conversion_report_can_be_plotted(tmp_path: Path) -> None:
    """Generate a diagnostic figure for the hydrate conversion chain."""
    _, hydrate = _load_configs()
    derived = _compute_hydrate_lattice_quantities(hydrate)

    out_path = _make_conversion_figure(hydrate, derived, tmp_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0
