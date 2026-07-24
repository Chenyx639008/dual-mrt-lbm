"""Component parameter definitions for LBM models.

Each component type (EOS, collision, force, wetting) is represented as an
immutable parameter container. The `to_params_dict()` methods on each
component produce the flat key-value format consumed by params.txt.

The design mirrors JAX-LaB's component-based model composition while
targeting huang_mrt_2d's params.txt protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════
# Enumerations
# ═══════════════════════════════════════════════════════════════════════════


class EOSType(Enum):
    """Equation of state variants supported by the solver."""

    CARNAHAN_STARLING = auto()  # CS-EOS (Huang & Wu 2016 SCMP)
    PENG_ROBINSON = auto()  # PR-EOS (MCMP)
    REDLICH_KWONG = auto()  # RK-EOS
    REDLICH_KWONG_SOAVE = auto()  # RKS-EOS
    VANDER_WAALS = auto()  # VdW-EOS


class CollisionType(Enum):
    """Collision operator variants."""

    MRT = auto()  # Multiple-Relaxation-Time (current solver)
    BGK = auto()  # Bhatnagar-Gross-Krook (future)
    CASCADED = auto()  # Cascaded / central-moment (future)


class ForceType(Enum):
    """Interaction force model variants."""

    SHAN_CHEN = auto()  # Standard Shan-Chen pseudopotential
    HUANG_ZHANG = auto()  # Huang & Wu (2016) 3rd-order correction
    ZHANG_CHEN = auto()  # Zhang-Chen potential formulation
    MIXED = auto()  # Weighted SC + ZC (JAX-LaB A-matrix approach)


class WettingType(Enum):
    """Wettability / contact-angle model variants."""

    IMPROVED_VIRTUAL_DENSITY = auto()  # Li et al. (2019) wall-density modification
    GEOMETRIC = auto()  # Fei et al. (2023) characteristic-direction interpolation
    WALL_MATERIAL_MAP = auto()  # 256-material lookup table (current MCMP)


# ═══════════════════════════════════════════════════════════════════════════
# Default EOS parameter sets (verified against paper values)
# ═══════════════════════════════════════════════════════════════════════════

EOS_DEFAULT_PARAMS: dict[EOSType, dict[str, float]] = {
    EOSType.CARNAHAN_STARLING: {
        "a": 1.0,
        "b": 4.0,
        "R": 1.0,
        # Tc = 0.3773 * a / (b * R) = 0.09433
        # T_reduced = T / Tc,  paper uses T_reduced = 0.7–0.9
        "T_reduced": 0.70,
    },
    EOSType.PENG_ROBINSON: {
        "a": 2.0 / 49.0,
        "b": 2.0 / 21.0,
        "R": 1.0,
        "omega": 0.344,  # acentric factor (water)
        "T_reduced": 0.80,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Component parameter dataclasses
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class EOSParams:
    """Equation-of-state parameters for pseudopotential computation.

    Attributes
    ----------
    eos_type : EOSType
        Which EOS functional form to use.
    a : float
        Attractive parameter (EOS-dependent definition).
    b : float
        Repulsive / co-volume parameter.
    R : float
        Gas constant (usually 1.0 in lattice units).
    T_reduced : float
        Reduced temperature T / Tc.
    omega : float | None
        Acentric factor (PR-EOS only).

    Notes
    -----
    For CS-EOS (Huang SCMP):  p = ρRT·(1+η+η²−η³)/(1−η)³ − aρ²,  η = bρ/4
    Tc = 0.3773·a/(b·R).  The solver multiplies `cs_T` (=T_reduced) by Tc internally.
    """

    eos_type: EOSType
    a: float = 1.0
    b: float = 4.0
    R: float = 1.0
    T_reduced: float = 0.70
    omega: float | None = None

    @classmethod
    def carnahan_starling(cls, a=1.0, b=4.0, R=1.0, T_reduced=0.70) -> EOSParams:
        """Factory for Carnahan-Starling EOS (Huang SCMP)."""
        return cls(
            eos_type=EOSType.CARNAHAN_STARLING, a=a, b=b, R=R, T_reduced=T_reduced
        )

    @classmethod
    def peng_robinson(
        cls, a=2 / 49, b=2 / 21, R=1.0, T_reduced=0.80, omega=0.344
    ) -> EOSParams:
        """Factory for Peng-Robinson EOS (MCMP)."""
        return cls(
            eos_type=EOSType.PENG_ROBINSON,
            a=a,
            b=b,
            R=R,
            T_reduced=T_reduced,
            omega=omega,
        )

    @property
    def Tc(self) -> float:
        """Critical temperature in lattice units."""
        if self.eos_type == EOSType.CARNAHAN_STARLING:
            return 0.3773 * self.a / (self.b * self.R)
        # PR-EOS: Tc = 0.17014 * a / (b * R)  (approximate)
        return 0.17014 * self.a / (self.b * self.R)

    @property
    def T(self) -> float:
        """Physical temperature T = T_reduced * Tc (lattice units)."""
        return self.T_reduced * self.Tc

    def to_params_dict(self) -> dict[str, Any]:
        """Produce flat dict for params.txt (SCMP Huang keys)."""
        d: dict[str, Any] = {}
        if self.eos_type == EOSType.CARNAHAN_STARLING:
            d["cs_a"] = self.a
            d["cs_b"] = self.b
            d["cs_R"] = self.R
            d["cs_T"] = self.T_reduced  # solver multiplies by Tc internally
            d["cs_G"] = -1.0  # interaction strength (standard)
        elif self.eos_type == EOSType.PENG_ROBINSON:
            # MCMP keys — placeholder for future MCMP model support
            d["cs_a"] = self.a  # TODO: verify MCMP key mapping
        return d

    def __post_init__(self):
        """Validate EOS parameters."""
        if self.T_reduced <= 0:
            raise ValueError(f"T_reduced must be positive, got {self.T_reduced}")
        if self.T_reduced >= 1.0:
            raise ValueError(
                f"T_reduced={self.T_reduced} >= 1.0 — above critical point, "
                f"no two-phase region. For CS-EOS, Tc={self.Tc:.5f}"
            )


@dataclass(frozen=True)
class CollisionParams:
    """Collision operator parameters.

    Attributes
    ----------
    collision_type : CollisionType
        Which collision operator to use (MRT / BGK / Cascaded).
    tau : float
        Relaxation time τ (determines kinematic viscosity via ν = cs²(τ − 0.5)).
    tau_p_a : float
        Shear relaxation for component A (MCMP MRT only).
    tau_p_b : float
        Shear relaxation for component B (MCMP MRT only).
    kappa : float
        Surface-tension adjustment parameter (MRT).
    Lambda : float
        Huang SCMP "magic" parameter Λ = 1/12 (eliminates anisotropy).
    alpha_meq : float
        Huang SCMP α parameter for meq[1] correction.
    """

    collision_type: CollisionType = CollisionType.MRT
    tau: float = 1.0
    tau_p_a: float = 1.0
    tau_p_b: float = 1.0
    kappa: float = 0.0
    Lambda: float = 1.0 / 12.0
    alpha_meq: float = 1.0

    @classmethod
    def mrt(cls, tau=1.0, tau_p_a=1.0, tau_p_b=1.0, kappa=0.0) -> CollisionParams:
        """Factory for MRT collision (default for all current models)."""
        return cls(
            collision_type=CollisionType.MRT,
            tau=tau,
            tau_p_a=tau_p_a,
            tau_p_b=tau_p_b,
            kappa=kappa,
        )

    @classmethod
    def huang_mrt(cls, tau=1.5, Lambda=1 / 12, alpha_meq=1.0) -> CollisionParams:
        """Factory for Huang SCMP MRT with paper-default parameters."""
        return cls(
            collision_type=CollisionType.MRT,
            tau=tau,
            tau_p_a=tau,
            tau_p_b=tau,
            Lambda=Lambda,
            alpha_meq=alpha_meq,
        )

    @classmethod
    def mcmp_mrt(cls, tau_p_a=1.0, tau_p_b=1.0, kappa=0.0) -> CollisionParams:
        """Factory for MCMP MRT — independent τ per component.

        Parameters
        ----------
        tau_p_a : float
            Shear relaxation for component A (e.g. water phase).
        tau_p_b : float
            Shear relaxation for component B (e.g. gas phase).
        kappa : float
            Surface-tension adjustment (MCMP MRT).
        """
        return cls(
            collision_type=CollisionType.MRT,
            tau=tau_p_a,  # default to tau_p_a for generic tau
            tau_p_a=tau_p_a,
            tau_p_b=tau_p_b,
            kappa=kappa,
        )

    def to_params_dict(self, model_family: str = "scmp") -> dict[str, Any]:
        """Produce flat dict for params.txt."""
        d: dict[str, Any] = {}
        if model_family == "scmp":
            d["tau_huang"] = self.tau
            d["Lambda_huang"] = self.Lambda
            d["alpha_meq"] = self.alpha_meq
        else:
            d["tau_p_a"] = self.tau_p_a
            d["tau_p_b"] = self.tau_p_b
            d["kappa"] = self.kappa
        return d


@dataclass(frozen=True)
class ForceParams:
    """Interaction force model parameters.

    Attributes
    ----------
    force_type : ForceType
        Which force scheme to use.
    G : float
        Interaction strength (SCMP Huang).
    k1 : float
        Primary surface-tension coefficient k₁ (Huang SCMP, default 1/12).
    k2 : float
        Secondary coefficient k₂ (Huang SCMP, default 0.0).
    kd : float
        Constant kd = −1/12 for pressure tensor (Huang SCMP).
    epsilon : float
        Computed as ε = −8·k₁ (Huang SCMP).
    GAB : float
        Cross-interaction strength A→B (MCMP Shan-Chen).
    GBA : float
        Cross-interaction strength B→A (MCMP Shan-Chen).
    sigmaA : float
        Component A self-interaction strength (MCMP).
    """

    force_type: ForceType = ForceType.HUANG_ZHANG
    G: float = -1.0
    k1: float = 1.0 / 12.0
    k2: float = 0.0
    kd: float = -1.0 / 12.0
    epsilon: float | None = None
    GAB: float = 0.24
    GBA: float = 0.24
    sigmaA: float = 0.11

    @classmethod
    def huang_zhang(
        cls, k1=None, k2=0.0, kd=-1 / 12, G=-1.0, epsilon=None
    ) -> ForceParams:
        """Factory for Huang & Wu (2016) SCMP force.

        Parameters
        ----------
        epsilon : float or None
            If given, used directly as ε (overrides k1). Empirically, ε=1.7
            works for T/Tc=0.70. Paper value: ε=−2/3 = −0.667 (k₁=1/12).
        k1 : float or None
            If epsilon is None, computed as ε=−8·k₁.
        k2 : float
            Secondary coefficient (default 0).
        kd : float
            Fixed constant for pressure tensor (default −1/12).
        G : float
            Interaction strength (default −1.0).
        """
        if epsilon is None:
            _k1 = k1 if k1 is not None else 1.0 / 12.0
            _epsilon = -8.0 * _k1
        else:
            _k1 = -epsilon / 8.0  # derived from ε=−8·k₁
            _epsilon = epsilon
        return cls(
            force_type=ForceType.HUANG_ZHANG,
            G=G,
            k1=_k1,
            k2=k2,
            kd=kd,
            epsilon=_epsilon,
        )

    @classmethod
    def shan_chen(cls, GAB=0.24, GBA=0.24, sigmaA=0.11) -> ForceParams:
        """Factory for standard Shan-Chen MCMP force."""
        return cls(
            force_type=ForceType.SHAN_CHEN,
            GAB=GAB,
            GBA=GBA,
            sigmaA=sigmaA,
        )

    def to_params_dict(self, model_family: str = "scmp") -> dict[str, Any]:
        """Produce flat dict for params.txt."""
        d: dict[str, Any] = {}
        if model_family == "scmp":
            d["pp_mode"] = 1
            d["epsilon_huang"] = (
                self.epsilon if self.epsilon is not None else -8.0 * self.k1
            )
            d["k2_huang"] = self.k2
            d["kd_huang"] = self.kd
            d["cs_G"] = self.G
        else:
            d["GAB"] = self.GAB
            d["GBA"] = self.GBA
            d["sigmaA"] = self.sigmaA
        return d


@dataclass(frozen=True)
class WettingParams:
    """Wettability / contact-angle parameters.

    Attributes
    ----------
    wetting_type : WettingType
        Which wetting model to use.
    contact_angle_deg : float
        Contact angle in degrees (SCMP).
    psi_l_ref : float
        Liquid reference pseudopotential for ghost BC (SCMP).
    psi_g_ref : float
        Gas reference pseudopotential for ghost BC (SCMP).
    G_ads : float
        Adsorption force strength (SCMP).
    theta_by_material : dict[int, float]
        Per-material contact angles in degrees (MCMP wall-material-map).
    GAw_m : float
        Slope for GAw = m·(θ − c) affine map (MCMP).
    GAw_c : float
        Intercept for GAw affine map (MCMP).
    """

    wetting_type: WettingType = WettingType.IMPROVED_VIRTUAL_DENSITY
    contact_angle_deg: float = 0.0
    psi_l_ref: float = 0.15
    psi_g_ref: float = 0.01
    G_ads: float = 0.0
    theta_by_material: dict[int, float] = field(default_factory=dict)
    GAw_m: float = 1.0 / 456.69
    GAw_c: float = 86.41

    @classmethod
    def scmp_neutral(cls) -> WettingParams:
        """Neutral wetting (θ = 0°, ψ-based ghost BC disabled)."""
        return cls(
            wetting_type=WettingType.IMPROVED_VIRTUAL_DENSITY, contact_angle_deg=0.0
        )

    @classmethod
    def scmp_contact(
        cls, theta_deg=30.0, psi_l_ref=0.15, psi_g_ref=0.01
    ) -> WettingParams:
        """SCMP contact angle with ψ-based ghost BC."""
        return cls(
            wetting_type=WettingType.IMPROVED_VIRTUAL_DENSITY,
            contact_angle_deg=theta_deg,
            psi_l_ref=psi_l_ref,
            psi_g_ref=psi_g_ref,
        )

    @classmethod
    def material_mapped(
        cls,
        theta_by_material=None,
        GAw_m=1.0 / 456.69,
        GAw_c=86.41,
        GBw_quartz=0.0,
        GBw_hydrate=0.0,
    ) -> WettingParams:
        """Factory for MCMP 256-material wettability lookup table.

        The CUDA solver maps material IDs (1=quartz, 2=hydrate) to
        interaction strengths via GAw = m·(θ − c) affine transform.

        Parameters
        ----------
        theta_by_material : dict or None
            {1: theta_quartz_deg, 2: theta_hydrate_deg}. Default 30°/80°.
        GAw_m : float
            Slope of GAw = m·(θ − c).
        GAw_c : float
            Intercept of GAw = m·(θ − c).
        GBw_quartz : float
            Component B wettability for quartz.
        GBw_hydrate : float
            Component B wettability for hydrate.
        """
        if theta_by_material is None:
            theta_by_material = {1: 30.0, 2: 80.0}
        return cls(
            wetting_type=WettingType.WALL_MATERIAL_MAP,
            contact_angle_deg=0.0,
            theta_by_material=theta_by_material,
            GAw_m=GAw_m,
            GAw_c=GAw_c,
        )

    def to_params_dict(self, model_family: str = "scmp") -> dict[str, Any]:
        """Produce flat dict for params.txt."""
        d: dict[str, Any] = {}
        if model_family == "scmp":
            d["G_ads"] = self.G_ads
            d["theta_contact_deg"] = self.contact_angle_deg
            d["huang_psi_l_ref"] = self.psi_l_ref
            d["huang_psi_g_ref"] = self.psi_g_ref
        else:
            if self.theta_by_material:
                for mat, theta in self.theta_by_material.items():
                    if mat == 1:
                        d["thetaA_quartz_deg"] = theta
                    elif mat == 2:
                        d["thetaA_hydrate_deg"] = theta
            d["GAw_m"] = self.GAw_m
            d["GAw_c"] = self.GAw_c
        return d

    def __post_init__(self):
        """Validate contact angle range."""
        if self.contact_angle_deg < 0 or self.contact_angle_deg > 180:
            raise ValueError(
                f"Contact angle must be in [0°, 180°], got {self.contact_angle_deg}°"
            )
        for mat, theta in self.theta_by_material.items():
            if theta < 0 or theta > 180:
                raise ValueError(
                    f"Contact angle for material {mat} must be in [0°, 180°], got {theta}°"
                )
