"""YAML configuration loader for lbm_mrt.

Loads configs/default.yaml (or a user-specified file) into a validated flat dict
whose keys match exactly the key strings expected by the C++ load_params_txt()
function in sim_utils.cu.

Usage::

    from lbm_mrt.core.config import load_config, override

    params = load_config()                     # load default.yaml
    params = load_config("configs/hydrate.yaml")  # load override file

    # Apply per-case overrides without mutating the base dict
    case_params = override(params, Sw=0.5, thetaA_quartz_deg=60)
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from lbm_mrt.core.paths import DEFAULT_CONFIG


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load a YAML config file and return a flat params dict.

    The flat dict uses the exact key names that C++ load_params_txt() reads,
    so it can be passed directly to io.params_writer.write_params_txt().

    Args:
        path: Path to a YAML config file. Defaults to configs/default.yaml.

    Returns:
        Flat dict mapping parameter names to their values.
    """
    config_path = Path(path) if path is not None else Path(DEFAULT_CONFIG)
    with config_path.open() as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return _flatten(raw)


def _flatten(raw: dict[str, Any]) -> dict[str, Any]:
    """Collapse nested YAML sections into the flat params.txt key space.

    Key name mapping follows exactly what sim_utils.cu::load_params_txt() reads.
    For example: YAML fluids.rhoA_ini_h → params key "rhoA_ini_h" → C++ rhoA_hi field.
    """
    p: dict[str, Any] = {}

    # ── geometry / morphology ──────────────────────────────────────────────
    g = raw.get("geometry", {})
    p["morph"] = g.get("morph", 1)
    p["r_obs"] = g.get("r_obs", 20)
    p["l_gap"] = g.get("l_gap", 20)
    p["coat_thick"] = g.get("coat_thick", 6)
    p["r_mid"] = g.get("r_mid", 4)
    p["geom_file"] = g.get("geom_file", "")

    # ── initial fluids ─────────────────────────────────────────────────────
    f = raw.get("fluids", {})
    p["Sw"] = f.get("Sw", 0.3)
    p["water_seed"] = f.get("water_seed", 1234567)
    # params.txt keys rhoA_ini_h/l are mapped to rhoA_hi/lo in RuntimeParams
    p["rhoA_ini_h"] = f.get("rhoA_ini_h", 6.6293)
    p["rhoA_ini_l"] = f.get("rhoA_ini_l", 0.34127)
    p["rhoB_ini_h"] = f.get("rhoB_ini_h", 0.1147)
    p["rhoB_ini_l"] = f.get("rhoB_ini_l", 0.0001)
    p["rhoA_ini_h_1"] = f.get("rhoA_ini_h_1", 7.2243)
    p["rhoA_ini_l_0"] = f.get("rhoA_ini_l_0", 0.0)
    p["rhoB_ini_l_1"] = f.get("rhoB_ini_l_1", 0.0)
    p["rhoB_ini_h_0"] = f.get("rhoB_ini_h_0", 0.2363)

    # ── wettability ────────────────────────────────────────────────────────
    w = raw.get("wettability", {})
    p["thetaA_quartz_deg"] = w.get("thetaA_quartz_deg", 30)
    p["thetaA_hydrate_deg"] = w.get("thetaA_hydrate_deg", 80)
    p["GBw_quartz"] = w.get("GBw_quartz", 0.0)
    p["GBw_hydrate"] = w.get("GBw_hydrate", 0.0)

    # ── scenario switches ──────────────────────────────────────────────────
    s = raw.get("scenario", {})
    p["init_eq"] = s.get("init_eq", 1)
    p["drive"] = s.get("drive", 2)

    # ── driving force ──────────────────────────────────────────────────────
    d = raw.get("driving", {})
    p["Gx"] = d.get("Gx", 1.5e-8)
    p["Gy"] = d.get("Gy", 0.0)
    p["drive_mode"] = d.get("drive_mode", 1)

    # ── relaxation / physics ───────────────────────────────────────────────
    r = raw.get("relaxation", {})
    p["tau_p_a"] = r.get("tau_p_a", 1.0)
    p["tau_p_b"] = r.get("tau_p_b", 0.775)
    p["kappa"] = r.get("kappa", 0.6)
    p["GAB"] = r.get("GAB", 0.24)
    p["GBA"] = r.get("GBA", 0.24)
    p["sigmaA"] = r.get("sigmaA", 0.11)

    # ── Huang & Wu (2016) SCMP ─────────────────────────────────────────────
    hg = raw.get("huang_scmp", {})
    p["pp_mode"] = hg.get("pp_mode", 0)
    p["k1_huang"] = hg.get("k1_huang", 1.0 / 12.0)
    p["epsilon_huang"] = hg.get("epsilon_huang", -2.0 / 3.0)
    p["k2_huang"] = hg.get("k2_huang", 0.0)
    p["kd_huang"] = hg.get("kd_huang", -1.0 / 12.0)
    p["alpha_meq"] = hg.get("alpha_meq", 1.0)
    p["cs_a"] = hg.get("cs_a", 1.0)
    p["cs_b"] = hg.get("cs_b", 4.0)
    p["cs_R"] = hg.get("cs_R", 1.0)
    p["cs_T"] = hg.get("cs_T", 0.0660)
    p["cs_G"] = hg.get("cs_G", -1.0)
    p["huang_R0"] = hg.get("huang_R0", 40.0)
    p["huang_xc"] = hg.get("huang_xc", 128.0)
    p["huang_yc"] = hg.get("huang_yc", 128.0)
    p["huang_W"] = hg.get("huang_W", 3.0)
    p["huang_init_mode"] = hg.get("huang_init_mode", 1)
    p["huang_rho_g"] = hg.get("huang_rho_g", 0.0)
    p["huang_rho_l"] = hg.get("huang_rho_l", 0.0)
    # — additional SCMP params read by solver (sim_utils.cu lines 353-358) —
    p["G_ads"] = hg.get("G_ads", 0.0)
    p["theta_contact_deg"] = hg.get("theta_contact_deg", 0.0)
    p["huang_psi_l_ref"] = hg.get("huang_psi_l_ref", 0.15)
    p["huang_psi_g_ref"] = hg.get("huang_psi_g_ref", 0.01)
    p["tau_huang"] = hg.get("tau_huang", 1.5)
    p["Lambda_huang"] = hg.get("Lambda_huang", 1.0 / 12.0)
    p["huang_u_max"] = hg.get("huang_u_max", 0.15)
    p["huang_psi_cut"] = hg.get("huang_psi_cut", 1.0e-3)
    p["huang_tanh_factor"] = hg.get("huang_tanh_factor", 2.0)
    p["huang_rho_max_init"] = hg.get("huang_rho_max_init", 1.0)

    # ── output / checkpoint ───────────────────────────────────────────────
    c = raw.get("checkpoint", {})
    p["ENABLE_CKPT"] = c.get("enable", True)
    p["OUTPUT_EVERY"] = c.get("every", 5000)
    p["CP_EVERY"] = c.get("every", 50000)
    p["CP_KEEP"] = c.get("keep", 3)
    p["CP_RESUME"] = c.get("resume", 1)

    # ── convergence ───────────────────────────────────────────────────────
    cv = raw.get("convergence", {})
    eq = cv.get("eq", {})
    flow = cv.get("flow", {})
    p["eq_tol_rel"] = eq.get("tol_rel", 1e-4)
    p["eq_need_consec"] = eq.get("need_consec", 2)
    p["eq_max_steps"] = eq.get("max_steps", 200000)
    p["flow_tol_rel"] = flow.get("tol_rel", 1e-4)
    p["flow_need_consec"] = flow.get("need_consec", 3)
    p["flow_max_steps"] = flow.get("max_steps", 5000000)

    # ── hydrate dissociation (optional; only used with mcmp_sim_hydrate) ──
    h = raw.get("hydrate", {})
    p["hydrate_enable"] = h.get("enable", False)
    p["hydrate_start_step"] = h.get("start_step", 0)

    therm = h.get("thermal", {})
    p["T0_init"] = therm.get("T0_init", 278.15)
    p["T0_inlet"] = therm.get("T0_inlet", 285.0)
    # 热 Dirichlet 边界朝向：0=底(y==0) 1=顶(y==NY-1) 2=左(x==0) 3=右(x==NX-1)
    p["thermal_bc_side"] = therm.get("thermal_bc_side", 0)
    # 温度场初始化模式：0=均匀(T0_init) 1=线性梯度(T0_init→T0_inlet 沿 bc_side 方向)
    p["thermal_init_mode"] = therm.get("thermal_init_mode", 0)
    p["lambda_fluid"] = therm.get("lambda_fluid", 0.6)
    p["lambda_hydrate"] = therm.get("lambda_hydrate", 0.49)
    p["lambda_solid"] = therm.get("lambda_solid", 0.9)
    p["rhocp_fluid"] = therm.get("rhocp_fluid", 4200000.0)
    p["rhocp_hydrate"] = therm.get("rhocp_hydrate", 2100000.0)
    p["rhocp_solid"] = therm.get("rhocp_solid", 2000000.0)

    conc = h.get("concentration", {})
    p["D_mol_water"] = conc.get("D_mol_water", 1.85e-9)
    p["Henry_KH"] = conc.get("Henry_KH", 0.1)
    p["Cm_init"] = conc.get("Cm_init", 0.0)

    rxn = h.get("reaction", {})
    p["k0_rxn"] = rxn.get("k0_rxn", 36000.0)
    p["Ea_rxn"] = rxn.get("Ea_rxn", 97500.0)
    p["e1_peq"] = rxn.get("e1_peq", 33.12)
    p["e2_peq"] = rxn.get("e2_peq", -9005.5)
    p["latent_heat"] = rxn.get("latent_heat", 43000.0)

    vop = h.get("vop", {})
    p["Vm_hydrate"] = vop.get("Vm_hydrate", 2.274e-5)
    p["Vh_init"] = vop.get("Vh_init", 1.0)
    p["vop_terminate_frac"] = vop.get("vop_terminate_frac", 0.01)

    units = h.get("units", {})
    p["dx_phys"] = units.get("dx_phys", 1.0e-5)
    p["dt_phys"] = units.get("dt_phys", 1.0e-6)

    return p


def override(base: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Return a deep copy of base with the given key/value overrides applied.

    Args:
        base:   Base params dict (e.g. from load_config()).
        **kwargs: Key-value pairs to override. Keys must be valid params.txt keys.

    Returns:
        New dict with overrides applied; the original base is not mutated.

    Example::

        params = load_config()
        case_params = override(params, Sw=0.7, Gx=2.0e-8, thetaA_quartz_deg=45)
    """
    result = copy.deepcopy(base)
    result.update(kwargs)
    return result
