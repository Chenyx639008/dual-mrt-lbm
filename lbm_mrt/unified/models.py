"""Pre-registered model definitions for lbm_mrt.

Each ModelDefinition declares a complete physics configuration by composing
EOSParams, CollisionParams, ForceParams, and WettingParams. The to_params_dict()
method merges component outputs into the flat key-value format consumed by
the CUDA solver's load_params_txt().

Models are organized by family:
    scmp  — Single-Component Multi-Phase (Huang & Wu 2016, CS-EOS)
    mcmp  — Multi-Component Multi-Phase (Shan-Chen, PR-EOS) [future]

Reference
---------
- Huang, H.-B., & Wu, Y.-C. (2016). "Third-order analysis of pseudopotential
  lattice Boltzmann model." Phys. Rev. E 93, 043311.
- Feasibility assessment: research/unified_framework_feasibility.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .components import (
    CollisionParams,
    CollisionType,
    EOSParams,
    EOSType,
    ForceParams,
    ForceType,
    WettingParams,
    WettingType,
)


# ═══════════════════════════════════════════════════════════════════════════
# Coexistence density helper (numpy — no JAX dependency)
# ═══════════════════════════════════════════════════════════════════════════


def _compute_cs_coexistence(a=1.0, b=4.0, R=1.0, T_reduced=0.70):
    """Compute Carnahan-Starling coexistence densities via Maxwell construction.

    Uses numpy for high-resolution pressure scan.  Returns (rho_g, rho_l)
    that get injected into params.txt as huang_rho_g / huang_rho_l,
    bypassing the inaccurate GPU-side CS critical-scaling estimate.

    Reference values (a=1,b=4,R=1):
      T_reduced=0.55 → ρ_g≈0.0003, ρ_l≈0.42
      T_reduced=0.70 → ρ_g≈0.005,  ρ_l≈0.36
      T_reduced=0.80 → ρ_g≈0.010,  ρ_l≈0.30
      T_reduced=0.90 → ρ_g≈0.025,  ρ_l≈0.20
    """
    import numpy as np

    Tc = 0.3773 * a / (b * R)
    T = T_reduced * Tc

    def p_np(rho):
        eta = b * rho / 4.0
        if eta >= 1.0 or rho <= 0.0:
            return -1e10
        eta2 = eta * eta
        eta3 = eta2 * eta
        denom = (1.0 - eta) * (1.0 - eta) * (1.0 - eta)
        return R * rho * T * (1.0 + eta + eta2 - eta3) / denom - a * rho * rho

    # High-resolution pressure scan
    rho_vals = np.logspace(-5, -0.01, 5000)
    p_vals = np.array([p_np(r) for r in rho_vals], dtype=np.float64)

    positive_mask = p_vals > 0

    # Gas branch: first positive pressure
    gas_idx = int(np.argmax(positive_mask))
    rho_g = float(rho_vals[gas_idx])

    # Liquid branch: first positive after spinodal (negative region)
    neg_starts = np.where(np.diff(positive_mask.astype(int)) == -1)[0]
    if len(neg_starts) > 0:
        unstable_start = neg_starts[0]
        pos_after = np.where(np.diff(positive_mask.astype(int)) == 1)[0]
        pos_after = pos_after[pos_after > unstable_start]
        if len(pos_after) > 0:
            rho_l = float(rho_vals[pos_after[0]])
        else:
            # Fallback: estimate from ideal scaling
            rho_l = 0.38 * (0.7 / max(T_reduced, 0.4)) ** 0.5
    else:
        rho_l = 0.38 * (0.7 / max(T_reduced, 0.4)) ** 0.5

    # Sanity clamps
    rho_g = max(min(rho_g, 0.1), 1e-6)
    rho_l = max(min(rho_l, 0.6), rho_g + 0.02)

    return float(rho_g), float(rho_l)


@dataclass(frozen=True)
class ModelDefinition:
    """Complete physics configuration for one LBM model variant.

    A ModelDefinition is a "recipe" — it specifies which components to use
    and with what parameters. It is NOT a simulator; the runner module
    translates it into params.txt and invokes the appropriate CUDA binary.

    Attributes
    ----------
    name : str
        Unique model identifier, e.g. "scmp_cs_huang_256".
    description : str
        Human-readable summary.
    model_family : str
        "scmp" or "mcmp" — determines binary selection and key namespaces.
    eos : EOSParams
        Equation-of-state configuration.
    collision : CollisionParams
        Collision operator configuration.
    force : ForceParams
        Interaction force configuration.
    wetting : WettingParams
        Wettability / contact-angle configuration.
    n_components : int
        Number of fluid components (1 for SCMP, 2 for MCMP).
    cuda_binary : str | None
        Path or name of the CUDA binary. If None, auto-detected from family.
    grid : tuple[int, int] | None
        Override for (NX, NY) grid size (Huang SCMP only).
    initial : dict[str, Any]
        Initial-condition overrides (droplet radius, center, etc.).
    numerical : dict[str, Any]
        Numerical guard parameters (velocity cap, psi cutoff, etc.).
    notes : str
        Free-form notes about this model configuration.
    """

    name: str
    description: str = ""
    model_family: str = "scmp"
    eos: EOSParams = field(default_factory=EOSParams.carnahan_starling)
    collision: CollisionParams = field(default_factory=CollisionParams.huang_mrt)
    force: ForceParams = field(default_factory=ForceParams.huang_zhang)
    wetting: WettingParams = field(default_factory=WettingParams.scmp_neutral)
    n_components: int = 1
    n_dimensions: int = 2
    cuda_binary: str | None = None
    grid: tuple[int, int] | None = None
    initial: dict[str, Any] = field(default_factory=dict)
    numerical: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    status: str = "stable"  # "stable" | "experimental" | "pending_verification"

    def __post_init__(self):
        """Validate model consistency."""
        if self.n_components not in (1, 2):
            raise ValueError(f"n_components must be 1 or 2, got {self.n_components}")
        if self.model_family not in ("scmp", "mcmp", "pf"):
            raise ValueError(
                f"model_family must be 'scmp', 'mcmp' or 'pf', got {self.model_family}"
            )
        if self.model_family == "scmp" and self.n_components != 1:
            raise ValueError("SCMP models require n_components=1")
        if self.model_family == "mcmp" and self.n_components != 2:
            raise ValueError("MCMP models require n_components=2")
        if self.n_dimensions not in (2, 3):
            raise ValueError(f"n_dimensions must be 2 or 3, got {self.n_dimensions}")
        if self.status not in ("stable", "experimental", "pending_verification"):
            raise ValueError(f"Unknown status '{self.status}'")

    def to_params_dict(self) -> dict[str, Any]:
        """Merge all component params into the flat params.txt dict.

        Returns
        -------
        dict
            Flat key-value mapping suitable for write_params_txt().
            Keys match exactly what sim_utils.cu::load_params_txt() expects.
        """
        # Phase-field family uses its own params schema (pf_* keys)
        if self.model_family == "pf":
            return self._to_pf_params_dict()

        params: dict[str, Any] = {}

        # Core mode flag
        if self.model_family == "scmp":
            params["pp_mode"] = 1
        else:
            params["pp_mode"] = 0

        # Merge component dictionaries (order: EOS → collision → force → wetting)
        params.update(self.eos.to_params_dict())
        params.update(self.collision.to_params_dict(model_family=self.model_family))
        params.update(self.force.to_params_dict(model_family=self.model_family))
        params.update(self.wetting.to_params_dict(model_family=self.model_family))

        # Initial-condition overrides
        if self.model_family == "scmp":
            params.setdefault("huang_init_mode", 1)  # 1 = droplet
            params.setdefault("huang_R0", 40.0)
            params.setdefault("huang_xc", 128.0)
            params.setdefault("huang_yc", 128.0)
            params.setdefault("huang_W", 3.0)

            # 🔧 Inject accurate coexistence densities (bypass GPU CS scaling)
            if self.eos.eos_type == EOSType.CARNAHAN_STARLING:
                rho_g, rho_l = _compute_cs_coexistence(
                    a=self.eos.a,
                    b=self.eos.b,
                    R=self.eos.R,
                    T_reduced=self.eos.T_reduced,
                )
                params.setdefault("huang_rho_g", rho_g)
                params.setdefault("huang_rho_l", rho_l)
        else:
            # MCMP defaults
            params.setdefault("Sw", 0.30)
            params.setdefault("init_eq", 1)
            params.setdefault("drive_mode", 1)
            params.setdefault("Gx", 1e-5)
        params.update(self.initial)

        # Numerical guards
        if self.model_family == "scmp":
            params.setdefault("huang_u_max", 0.15)
            params.setdefault("huang_psi_cut", 1.0e-3)
            params.setdefault("huang_tanh_factor", 2.0)
            params.setdefault("huang_rho_max_init", 1.0)
        params.update(self.numerical)

        # Misc required keys (harmless defaults for SCMP)
        params.setdefault("init_eq", 0)
        params.setdefault("drive_mode", 1)
        params.setdefault("Gx", 0.0)
        params.setdefault("Gy", 0.0)
        params.setdefault("geom_file", "")

        # 🔧 Phase 2: Inject runtime grid for unified builds
        if self.grid is not None and self.grid[0] > 0 and self.grid[1] > 0:
            params.setdefault("nx_override", self.grid[0])
            params.setdefault("ny_override", self.grid[1])

        return params

    def _to_pf_params_dict(self) -> dict[str, Any]:
        """Phase-field family params → flat params.txt dict (pf_* keys).

        The phase-field solver is under development (see
        research/phasefield_development_plan.md). Stage gating via pf_mode:
          0 = single-phase incompressible NS (Stage 0)
          1 = conservative Allen-Cahn only (Stage 1)
          2 = AC + NS coupled (Stage 2+)
        Phase-field params live in the free-form ``initial`` dict.
        The JAX track (jax_lbm/pf/) consumes the same schema for verification.
        """
        pf = self.initial
        params: dict[str, Any] = {}
        params["pf_mode"] = int(pf.get("pf_mode", 0))
        params["pf_scheme"] = str(pf.get("pf_scheme", "conservative_ac"))
        params["pf_W"] = float(pf.get("pf_W", 3.0))         # interface width (lu)
        params["pf_M"] = float(pf.get("pf_M", 0.02))        # mobility
        params["pf_sigma"] = float(pf.get("pf_sigma", 0.01))  # surface tension
        params["pf_rho_g"] = float(pf.get("pf_rho_g", 0.001))
        params["pf_rho_w"] = float(pf.get("pf_rho_w", 1.0))
        params["pf_mu_g"] = float(pf.get("pf_mu_g", 0.1))
        params["pf_mu_w"] = float(pf.get("pf_mu_w", 0.1))
        params["pf_theta_deg"] = float(pf.get("pf_theta_deg", 90.0))
        params["pf_tau"] = float(pf.get("pf_tau", 0.8))     # NS relaxation
        # droplet initial condition (Stage 2)
        params["pf_R0"] = float(pf.get("pf_R0", 30.0))
        params["pf_xc"] = float(pf.get("pf_xc", 0.0))
        params["pf_yc"] = float(pf.get("pf_yc", 0.0))
        # misc required keys (harmless defaults)
        params.setdefault("init_eq", 0)
        params.setdefault("drive_mode", 1)
        params.setdefault("Gx", 0.0)
        params.setdefault("Gy", 0.0)
        params.setdefault("geom_file", "")
        if self.grid is not None and self.grid[0] > 0 and self.grid[1] > 0:
            params.setdefault("nx_override", self.grid[0])
            params.setdefault("ny_override", self.grid[1])
        params.update(self.numerical)
        return params

    def resolve_binary(self) -> str:
        """Return the absolute path to the CUDA binary for this model.

        Returns
        -------
        str
            Absolute path to the compiled CUDA binary.
        """
        import os
        from lbm_mrt.core.paths import SOLVER_DIR

        if self.cuda_binary:
            if os.path.isabs(self.cuda_binary):
                return self.cuda_binary
            return os.path.join(SOLVER_DIR, self.cuda_binary)

        # Auto-detect from model family
        if self.model_family == "scmp":
            return os.path.join(SOLVER_DIR, "mcmp_huang_256")
        elif self.model_family == "mcmp":
            # Check if hydrate is enabled
            if self.initial.get("hydrate_enable"):
                return os.path.join(SOLVER_DIR, "mcmp_sim_hydrate")
            return os.path.join(SOLVER_DIR, "mcmp_sim")
        elif self.model_family == "pf":
            # Phase-field CUDA binary (Stage 0+); JAX track first.
            return os.path.join(SOLVER_DIR, "pf_ns_2d")
        return os.path.join(SOLVER_DIR, "mcmp_sim")


# ═══════════════════════════════════════════════════════════════════════════
# Pre-registered SCMP models
# ═══════════════════════════════════════════════════════════════════════════

SCMP_MODELS: dict[str, ModelDefinition] = {
    # ── Huang & Wu (2016) baseline: 256×256, CS-EOS, half-max tension ──
    "scmp_cs_huang_256": ModelDefinition(
        name="scmp_cs_huang_256",
        description=(
            "Huang & Wu (2016) SCMP baseline: 256×256, CS-EOS, "
            "MRT with Λ=1/12, ε=1.7 (empirically tuned), "
            "T_reduced=0.70 for strong density ratio."
        ),
        model_family="scmp",
        eos=EOSParams.carnahan_starling(a=1.0, b=4.0, R=1.0, T_reduced=0.70),
        collision=CollisionParams.huang_mrt(tau=1.5, Lambda=1.0 / 12.0, alpha_meq=1.0),
        force=ForceParams.huang_zhang(epsilon=1.7, k2=0.0, G=-1.0),
        wetting=WettingParams.scmp_neutral(),
        n_components=1,
        cuda_binary="mcmp_huang_256",
        grid=(256, 256),
        initial={
            "huang_init_mode": 1,
            "huang_R0": 40.0,
            "huang_xc": 128.0,
            "huang_yc": 128.0,
            "huang_W": 3.0,
            # Explicit coexistence densities (empirically validated at T/Tc=0.70)
            "huang_rho_g": 0.009170,
            "huang_rho_l": 0.305202,
        },
        notes=(
            "Empirically tuned ε=1.7 (note: sign convention differs from paper's "
            "ε=−8(k₁+k₂). At T/Tc=0.70 with R₀=40 droplet, ε=1.7 yields "
            "stable droplet with conserved mass. Coexistence densities from "
            "Maxwell construction: ρ_g≈0.009, ρ_l≈0.305. "
            "For different T/Tc or droplet size, re-tune ε."
        ),
    ),
    # ── Huang & Wu (2016) with contact angle ──
    "scmp_cs_huang_256_theta60": ModelDefinition(
        name="scmp_cs_huang_256_theta60",
        description=(
            "Huang SCMP with 60° contact angle (ψ-based ghost BC). "
            "Same physics as scmp_cs_huang_256, adds wettability."
        ),
        model_family="scmp",
        eos=EOSParams.carnahan_starling(a=1.0, b=4.0, R=1.0, T_reduced=0.70),
        collision=CollisionParams.huang_mrt(tau=1.5),
        force=ForceParams.huang_zhang(epsilon=1.7, k2=0.0, G=-1.0),
        wetting=WettingParams.scmp_contact(theta_deg=60.0),
        n_components=1,
        cuda_binary="mcmp_huang_256",
        grid=(256, 256),
        initial={
            "huang_init_mode": 1,
            "huang_R0": 40.0,
            "huang_xc": 128.0,
            "huang_yc": 80.0,  # droplet sits on bottom wall
            "huang_W": 3.0,
        },
        notes="Droplet on bottom wall with 60° contact angle (hydrophilic).",
    ),
    # ── Huang SCMP with θ=120° (hydrophobic) ──
    "scmp_cs_huang_256_theta120": ModelDefinition(
        name="scmp_cs_huang_256_theta120",
        description="Huang SCMP with 120° contact angle (hydrophobic wall).",
        model_family="scmp",
        eos=EOSParams.carnahan_starling(a=1.0, b=4.0, R=1.0, T_reduced=0.70),
        collision=CollisionParams.huang_mrt(tau=1.5),
        force=ForceParams.huang_zhang(),
        wetting=WettingParams.scmp_contact(theta_deg=120.0),
        n_components=1,
        cuda_binary="mcmp_huang_256",
        grid=(256, 256),
        initial={
            "huang_init_mode": 1,
            "huang_R0": 40.0,
            "huang_xc": 128.0,
            "huang_yc": 80.0,
            "huang_W": 3.0,
        },
        notes="Droplet on bottom wall with 120° contact angle (hydrophobic).",
    ),
    # ── Huang SCMP low-T (strong density ratio) ──
    "scmp_cs_huang_256_lowT": ModelDefinition(
        name="scmp_cs_huang_256_lowT",
        description=(
            "Huang SCMP at T_reduced=0.55 — very strong density ratio "
            "(ρ_l/ρ_g ~ O(10³)). Requires careful initialization."
        ),
        model_family="scmp",
        eos=EOSParams.carnahan_starling(a=1.0, b=4.0, R=1.0, T_reduced=0.55),
        collision=CollisionParams.huang_mrt(tau=1.0),  # lower τ for stability
        force=ForceParams.huang_zhang(),
        wetting=WettingParams.scmp_neutral(),
        n_components=1,
        cuda_binary="mcmp_huang_256",
        grid=(256, 256),
        initial={
            "huang_init_mode": 1,
            "huang_R0": 40.0,
            "huang_xc": 128.0,
            "huang_yc": 128.0,
            "huang_W": 3.0,
        },
        numerical={"huang_u_max": 0.10},  # tighter velocity cap
        notes="Low temperature — expect strong density ratio, may need "
        "reduced τ and tighter velocity cap for stability.",
    ),
    # ── Huang SCMP high-T (weak density ratio, near-critical) ──
    "scmp_cs_huang_256_highT": ModelDefinition(
        name="scmp_cs_huang_256_highT",
        description=(
            "Huang SCMP at paper-default T_reduced=0.90 (near-critical). "
            "Weaker density ratio, easier to simulate, good for validation."
        ),
        model_family="scmp",
        eos=EOSParams.carnahan_starling(a=1.0, b=4.0, R=1.0, T_reduced=0.90),
        collision=CollisionParams.huang_mrt(tau=1.5),
        force=ForceParams.huang_zhang(),
        wetting=WettingParams.scmp_neutral(),
        n_components=1,
        cuda_binary="mcmp_huang_256",
        grid=(256, 256),
        initial={
            "huang_init_mode": 1,
            "huang_R0": 40.0,
            "huang_xc": 128.0,
            "huang_yc": 128.0,
            "huang_W": 4.0,
        },
        notes="Paper default T=0.9Tc. Wider interface, easier convergence.",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Pre-registered MCMP models
# ═══════════════════════════════════════════════════════════════════════════

MCMP_MODELS: dict[str, ModelDefinition] = {
    "mcmp_pr_baseline": ModelDefinition(
        name="mcmp_pr_baseline",
        description=(
            "MCMP two-phase baseline: PR-EOS (A) + ideal gas (B), "
            "MRT collision, Shan-Chen force, Sw=0.30, init_eq=1."
        ),
        model_family="mcmp",
        n_components=2,
        eos=EOSParams.peng_robinson(a=2 / 49, b=2 / 21, R=1.0, T_reduced=0.80),
        collision=CollisionParams.mcmp_mrt(
            tau_p_a=0.593418,
            tau_p_b=0.515411,
            kappa=0.7698,
        ),
        force=ForceParams.shan_chen(GAB=0.24, GBA=0.24, sigmaA=0.08),
        wetting=WettingParams.material_mapped(
            theta_by_material={1: 30.0, 2: 80.0},
            GAw_m=1.0 / 399.39,
            GAw_c=85.29,
        ),
        cuda_binary="mcmp_sim",
        initial={
            "Sw": 0.30,
            "init_eq": 1,
            "rhoA_hi": 6.6293,
            "rhoA_lo": 0.5,
            "rhoB_hi": 0.41931,
            "rhoB_lo": 0.05,
            "rhoA_ini_h_1": 7.5511,
            "rhoA_ini_l_0": 0.0,
            "rhoB_ini_l_1": 0.0,
            "rhoB_ini_h_0": 0.4290,
            "drive_mode": 1,
            "Gx": 1e-5,
            "Gy": 0.0,
        },
        notes="Calibrated for init_eq=1. Requires mcmp_sim binary.",
    ),
    "mcmp_pr_wet": ModelDefinition(
        name="mcmp_pr_wet",
        description=("MCMP water-wet: Sw=0.50, init_eq=2, lower capillary number."),
        model_family="mcmp",
        n_components=2,
        eos=EOSParams.peng_robinson(a=2 / 49, b=2 / 21, R=1.0, T_reduced=0.80),
        collision=CollisionParams.mcmp_mrt(
            tau_p_a=0.608377,
            tau_p_b=0.524639,
            kappa=0.5337,
        ),
        force=ForceParams.shan_chen(GAB=0.30, GBA=0.30, sigmaA=0.08),
        wetting=WettingParams.material_mapped(
            theta_by_material={1: 30.0, 2: 80.0},
            GAw_m=1.0 / 375.40,
            GAw_c=85.54,
        ),
        cuda_binary="mcmp_sim",
        initial={
            "Sw": 0.50,
            "init_eq": 2,
            "rhoA_hi": 6.6293,
            "rhoA_lo": 0.005,
            "rhoB_hi": 0.236653,
            "rhoB_lo": 0.014,
            "rhoA_ini_h_1": 7.2243,
            "rhoA_ini_l_0": 0.0,
            "rhoB_ini_l_1": 0.0,
            "rhoB_ini_h_0": 0.2363,
            "drive_mode": 1,
            "Gx": 4e-5,
            "Gy": 0.0,
        },
        notes="Higher Sw, moderate body force for drainage.",
    ),
    # ── Hydrate models (MCMP + thermal + concentration + VOP) ──
    "mcmp_hydrate_sphere": ModelDefinition(
        name="mcmp_hydrate_sphere",
        description=(
            "MCMP hydrate sphere: 300×300, PR-EOS, bottom-wall heating "
            "(T_inlet=323.15K), single hydrate sphere decomposition."
        ),
        model_family="mcmp",
        n_components=2,
        eos=EOSParams.peng_robinson(a=2 / 49, b=2 / 21, R=1.0, T_reduced=0.80),
        collision=CollisionParams.mcmp_mrt(
            tau_p_a=0.593418,
            tau_p_b=0.515411,
            kappa=0.7698,
        ),
        force=ForceParams.shan_chen(GAB=0.24, GBA=0.24, sigmaA=0.11),
        wetting=WettingParams.material_mapped(
            theta_by_material={1: 30.0, 2: 80.0},
        ),
        cuda_binary="mcmp_sim_hydrate",
        grid=(300, 300),
        status="pending_verification",
        initial={
            "Sw": 0.30,
            "init_eq": 1,
            "hydrate_enable": True,
            "hydrate_start_step": 0,
            "T0_init": 278.15,
            "T0_inlet": 323.15,
            "thermal_bc_side": 0,
            "lambda_fluid": 0.6,
            "lambda_hydrate": 0.49,
            "lambda_solid": 0.9,
            "rhocp_fluid": 4200000.0,
            "rhocp_hydrate": 2100000.0,
            "rhocp_solid": 2000000.0,
            "D_mol_water": 1.85e-9,
            "Henry_KH": 0.1,
            "Cm_init": 0.0,
            "k0_rxn": 36000.0,
            "Ea_rxn": 97500.0,
            "e1_peq": 33.12,
            "e2_peq": -9005.5,
            "latent_heat": 43000.0,
            "Vm_hydrate": 2.274e-5,
            "Vh_init": 1.0,
            "vop_terminate_frac": 0.01,
            "dx_phys": 1.0e-5,
            "dt_phys": 1.0e-6,
            "drive_mode": 1,
            "Gx": 0.0,
            "Gy": 0.0,
        },
        notes="Spherical hydrate decomposition with bottom-wall 323K heating.",
    ),
    "mcmp_hydrate_porous": ModelDefinition(
        name="mcmp_hydrate_porous",
        description=(
            "MCMP hydrate porous: right-boundary heating (T_inlet=298.15K), "
            "multi-particle porous media with hydrate decomposition."
        ),
        model_family="mcmp",
        n_components=2,
        eos=EOSParams.peng_robinson(a=2 / 49, b=2 / 21, R=1.0, T_reduced=0.80),
        collision=CollisionParams.mcmp_mrt(
            tau_p_a=0.593418,
            tau_p_b=0.515411,
            kappa=0.7698,
        ),
        force=ForceParams.shan_chen(GAB=0.24, GBA=0.24, sigmaA=0.11),
        wetting=WettingParams.material_mapped(
            theta_by_material={1: 30.0, 2: 80.0},
        ),
        cuda_binary="mcmp_sim_hydrate",
        status="pending_verification",
        initial={
            "Sw": 0.30,
            "init_eq": 1,
            "hydrate_enable": True,
            "hydrate_start_step": 0,
            "T0_init": 278.15,
            "T0_inlet": 298.15,
            "thermal_bc_side": 3,
            "lambda_fluid": 0.6,
            "lambda_hydrate": 0.49,
            "lambda_solid": 0.9,
            "rhocp_fluid": 4200000.0,
            "rhocp_hydrate": 2100000.0,
            "rhocp_solid": 2000000.0,
            "D_mol_water": 1.85e-9,
            "Henry_KH": 0.1,
            "Cm_init": 0.0,
            "k0_rxn": 36000.0,
            "Ea_rxn": 97500.0,
            "e1_peq": 33.12,
            "e2_peq": -9005.5,
            "latent_heat": 43000.0,
            "Vm_hydrate": 2.274e-5,
            "Vh_init": 1.0,
            "vop_terminate_frac": 0.01,
            "dx_phys": 1.0e-5,
            "dt_phys": 1.0e-6,
            "drive_mode": 1,
            "Gx": 0.0,
            "Gy": 0.0,
        },
        notes="Porous media hydrate with right-boundary 298K heating.",
    ),
    # ── 3D models (placeholder — pending D3Q19 lattice + CUDA 3D solver) ──
    # fmt: off
    # "scmp_cs_huang_128_3d": ModelDefinition(
    #     name="scmp_cs_huang_128_3d",
    #     description="Huang SCMP 3D: 128³, CS-EOS, D3Q19 MRT.",
    #     model_family="scmp", n_components=1, n_dimensions=3,
    #     eos=EOSParams.carnahan_starling(T_reduced=0.70),
    #     collision=CollisionParams.huang_mrt(tau=1.5),
    #     force=ForceParams.huang_zhang(epsilon=1.7),
    #     wetting=WettingParams.scmp_neutral(),
    #     cuda_binary="mcmp_huang_128_3d", grid=(128, 128, 128),
    #     status="pending_verification",
    # ),
    # fmt: on
}


# ═══════════════════════════════════════════════════════════════════════════
# Pre-registered Phase-Field models (family "pf") — see
# research/phasefield_development_plan.md
# JAX track (jax_lbm/pf/) is the algorithm-verification path; CUDA binary
# (pf_ns_2d) is under construction (Stage 0+).
# ═══════════════════════════════════════════════════════════════════════════

PF_MODELS: dict[str, ModelDefinition] = {
    "pf_ns_single_2d": ModelDefinition(
        name="pf_ns_single_2d",
        description=(
            "Phase-field Stage 0 scaffold: single-phase D2Q9 MRT incompressible NS "
            "(no Allen-Cahn). Poiseuille / lid-driven cavity baseline."
        ),
        model_family="pf",
        collision=CollisionParams.huang_mrt(tau=0.8),
        n_components=1,
        grid=(128, 64),
        initial={"pf_mode": 0},
        status="experimental",
        notes=(
            "Stage 0 scaffold: single-phase NS only. JAX track: "
            "jax_lbm/pf/phase_field.py (Poiseuille test)."
        ),
    ),
    "pf_ac_droplet_2d": ModelDefinition(
        name="pf_ac_droplet_2d",
        description=(
            "Phase-field Stage 2 scaffold: conservative Allen-Cahn + NS coupled, "
            "static droplet (Laplace pressure / coexistence benchmarks)."
        ),
        model_family="pf",
        collision=CollisionParams.huang_mrt(tau=0.8),
        n_components=1,
        grid=(128, 128),
        initial={
            "pf_mode": 2,
            "pf_W": 3.0,
            "pf_M": 0.02,
            "pf_sigma": 0.01,
            "pf_rho_g": 0.001,
            "pf_rho_w": 1.0,
            "pf_R0": 30.0,
            "pf_xc": 64.0,
            "pf_yc": 64.0,
        },
        status="experimental",
        notes="Droplet equilibrium / Laplace benchmark target (Stage 2).",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Model Registry
# ═══════════════════════════════════════════════════════════════════════════


class ModelRegistry:
    """Central registry for all pre-defined LBM models.

    Provides discovery, creation, and listing of available models.
    Models are organized by family (scmp / mcmp).
    """

    _registry: dict[str, ModelDefinition] = dict(SCMP_MODELS)

    # Merge MCMP + Phase-Field models at class init
    _registry.update(MCMP_MODELS)
    _registry.update(PF_MODELS)

    @classmethod
    def register(cls, model: ModelDefinition) -> None:
        """Register a new model definition.

        Parameters
        ----------
        model : ModelDefinition
            The model to register. Its name must be unique.

        Raises
        ------
        ValueError
            If a model with the same name already exists.
        """
        if model.name in cls._registry:
            raise ValueError(f"Model '{model.name}' is already registered.")
        cls._registry[model.name] = model

    @classmethod
    def get(cls, name: str) -> ModelDefinition:
        """Retrieve a model definition by name.

        Parameters
        ----------
        name : str
            Model identifier.

        Returns
        -------
        ModelDefinition

        Raises
        ------
        KeyError
            If the model name is not found.
        """
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise KeyError(f"Model '{name}' not found. Available models: {available}")
        return cls._registry[name]

    @classmethod
    def list_all(cls) -> list[str]:
        """Return sorted list of all registered model names."""
        return sorted(cls._registry.keys())

    @classmethod
    def list_scmp(cls) -> list[str]:
        """Return sorted list of SCMP model names."""
        return sorted(k for k, v in cls._registry.items() if v.model_family == "scmp")

    @classmethod
    def list_mcmp(cls) -> list[str]:
        """Return sorted list of MCMP model names."""
        return sorted(k for k, v in cls._registry.items() if v.model_family == "mcmp")

    @classmethod
    def list_pf(cls) -> list[str]:
        """Return sorted list of Phase-Field model names."""
        return sorted(k for k, v in cls._registry.items() if v.model_family == "pf")

    @classmethod
    def info(cls, name: str) -> str:
        """Return a human-readable summary of a model.

        Parameters
        ----------
        name : str
            Model identifier.

        Returns
        -------
        str
            Multi-line summary string.
        """
        m = cls.get(name)
        lines = [
            f"Model: {m.name}",
            f"Family: {m.model_family} | Components: {m.n_components} | Dim: {m.n_dimensions}D",
            f"Description: {m.description}",
            f"EOS: {m.eos.eos_type.name} (a={m.eos.a}, b={m.eos.b}, R={m.eos.R}, T/Tc={m.eos.T_reduced})",
            f"Collision: {m.collision.collision_type.name} (τ={m.collision.tau})",
            f"Force: {m.force.force_type.name}",
            f"Wetting: {m.wetting.wetting_type.name} (θ={m.wetting.contact_angle_deg}°)",
            f"Binary: {m.cuda_binary or 'auto-detect'}",
        ]
        if m.grid:
            lines.append(f"Grid: {m.grid[0]}×{m.grid[1]}")
        if m.notes:
            lines.append(f"Notes: {m.notes}")
        return "\n".join(lines)
