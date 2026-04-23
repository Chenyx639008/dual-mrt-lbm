"""Tests for YAML config loading and params.txt serialization."""

import os
import tempfile

import pytest

from lbm_mrt.core.config import load_config, override
from lbm_mrt.io.params_writer import fmt_value, write_params_txt


# ── load_config ───────────────────────────────────────────────────────────

REQUIRED_KEYS = [
    "morph", "r_obs", "l_gap", "coat_thick", "r_mid",
    "Sw", "water_seed",
    "rhoA_ini_h", "rhoA_ini_l", "rhoB_ini_h", "rhoB_ini_l",
    "thetaA_quartz_deg", "thetaA_hydrate_deg",
    "GBw_quartz", "GBw_hydrate",
    "init_eq", "drive",
    "Gx", "Gy", "drive_mode",
    "tau_p_a", "tau_p_b", "kappa", "GAB", "GBA", "sigmaA",
    "ENABLE_CKPT", "CP_EVERY", "CP_KEEP", "CP_RESUME",
    "eq_tol_rel", "eq_need_consec", "eq_max_steps",
    "flow_tol_rel", "flow_need_consec", "flow_max_steps",
    "hydrate_enable", "hydrate_start_step",
    "T0_init", "T0_inlet",
    "lambda_fluid", "lambda_hydrate", "lambda_solid",
    "k0_rxn", "Ea_rxn", "dx_phys", "dt_phys",
]


def test_load_default_config_has_required_keys() -> None:
    params = load_config()
    for k in REQUIRED_KEYS:
        assert k in params, f"Missing required key: {k}"


def test_load_default_config_types() -> None:
    params = load_config()
    assert isinstance(params["morph"], int)
    assert isinstance(params["Sw"], float)
    assert isinstance(params["ENABLE_CKPT"], bool)
    assert isinstance(params["CP_EVERY"], int)
    assert isinstance(params["Gx"], float)


def test_load_default_config_physics_bounds() -> None:
    params = load_config()
    assert 0.5 < params["tau_p_a"] <= 2.0, "tau_p_a should be in (0.5, 2.0]"
    assert 0.5 < params["tau_p_b"] <= 2.0, "tau_p_b should be in (0.5, 2.0]"
    assert 0.0 <= params["Sw"] <= 1.0, "Sw should be in [0, 1]"
    assert params["CP_EVERY"] > 0
    assert params["T0_init"] > 200.0


def test_load_hydrate_overlay() -> None:
    import os
    hydrate_cfg = os.path.join(
        os.path.dirname(__file__), "..", "configs", "hydrate.yaml"
    )
    params = load_config(hydrate_cfg)
    assert params["hydrate_enable"] is True


# ── override ─────────────────────────────────────────────────────────────

def test_override_applies_changes() -> None:
    params = load_config()
    new_params = override(params, Sw=0.9, Gx=2.0e-8, thetaA_quartz_deg=60)
    assert new_params["Sw"] == pytest.approx(0.9)
    assert new_params["Gx"] == pytest.approx(2.0e-8)
    assert new_params["thetaA_quartz_deg"] == 60


def test_override_does_not_mutate_base() -> None:
    params = load_config()
    original_sw = params["Sw"]
    _ = override(params, Sw=0.9)
    assert params["Sw"] == pytest.approx(original_sw)


def test_override_can_add_new_keys() -> None:
    params = load_config()
    new_params = override(params, geom_file="data/geometry/case0001/geo.plt")
    assert new_params["geom_file"] == "data/geometry/case0001/geo.plt"


# ── fmt_value ────────────────────────────────────────────────────────────

def test_fmt_value_bool_true() -> None:
    assert fmt_value(True) == "true"


def test_fmt_value_bool_false() -> None:
    assert fmt_value(False) == "false"


def test_fmt_value_integer() -> None:
    assert fmt_value(50000) == "50000"


def test_fmt_value_integer_float() -> None:
    assert fmt_value(50000.0) == "50000"


def test_fmt_value_small_float() -> None:
    result = fmt_value(1.5e-8)
    assert "1.5" in result
    assert "e" in result.lower()


# ── write_params_txt ──────────────────────────────────────────────────────

def test_write_params_txt_roundtrip(tmp_path) -> None:
    params = load_config()
    out = tmp_path / "params.txt"
    write_params_txt(params, out)

    lines = out.read_text().splitlines()
    keys_written = {ln.split()[0] for ln in lines if ln.strip()}

    for k in ["Sw", "tau_p_a", "ENABLE_CKPT", "Gx", "CP_EVERY"]:
        assert k in keys_written, f"Key {k!r} not written to params.txt"


def test_write_params_txt_skips_empty_geom(tmp_path) -> None:
    params = load_config()
    params["geom_file"] = ""
    out = tmp_path / "params.txt"
    write_params_txt(params, out)
    content = out.read_text()
    assert "geom_file" not in content


def test_write_params_txt_includes_geom_when_set(tmp_path) -> None:
    params = load_config()
    params["geom_file"] = "data/geometry/case0001/geo.plt"
    out = tmp_path / "params.txt"
    write_params_txt(params, out)
    content = out.read_text()
    assert "geom_file" in content
    assert "case0001" in content
