"""Multiple Equations of State — pure JAX, functional style.

Inspired by JAX-LaB `src/jax_lab/eos.py` (Carnahan_Starling, Peng_Robinson,
Redlich_Kwong, Redlich_Kwong_Soave, VanderWaal).

All functions are @jit-compatible pure functions. No class hierarchy.
EOS selection via string key or direct function reference.

Reference
---------
- JAX-LaB: src/jax_lab/eos.py
- Huang & Wu (2016): Phys. Rev. E 93, 043311 (CS-EOS)
"""

from functools import partial

import jax.numpy as jnp
from jax import jit

from jax_lbm.lattice import CS2

# ═══════════════════════════════════════════════════════════════════════════
# EOS Type Registry
# ═══════════════════════════════════════════════════════════════════════════

EOS_REGISTRY: dict[str, callable] = {}


def register_eos(name: str):
    """Decorator to register an EOS function."""

    def decorator(fn):
        EOS_REGISTRY[name] = fn
        return fn

    return decorator


def get_eos(name: str) -> callable:
    """Get EOS pressure function by name.

    Supported: 'cs', 'pr', 'rk', 'rks', 'vdw', 'carnahan_starling',
               'peng_robinson', 'redlich_kwong', 'redlich_kwong_soave',
               'van_der_waals'.
    """
    name_lower = name.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "cs": "carnahan_starling",
        "pr": "peng_robinson",
        "rk": "redlich_kwong",
        "rks": "redlich_kwong_soave",
        "vdw": "van_der_waals",
    }
    key = aliases.get(name_lower, name_lower)
    if key not in EOS_REGISTRY:
        raise KeyError(
            f"Unknown EOS '{name}'. Available: {list(EOS_REGISTRY.keys())}. "
            f"Aliases: {list(aliases.keys())}."
        )
    return EOS_REGISTRY[key]


def list_eos() -> list[str]:
    """List all registered EOS names."""
    return sorted(EOS_REGISTRY.keys())


# ═══════════════════════════════════════════════════════════════════════════
# EOS Implementations
# ═══════════════════════════════════════════════════════════════════════════


@register_eos("carnahan_starling")
def cs_eos_pressure(rho, a=1.0, b=4.0, R=1.0, T=0.066):
    """Carnahan-Starling EOS.

    p = ρRT·(1+η+η²-η³)/(1-η)³ - a·ρ²,  η = b·ρ/4

    Parameters
    ----------
    rho : array — density field
    a, b, R : float — EOS parameters (lattice units)
    T : float — temperature (lattice units). Tc = 0.3773·a/(b·R) ≈ 0.09433

    Returns
    -------
    p : array — pressure field
    """
    eta = b * rho / 4.0
    eta2 = eta * eta
    eta3 = eta2 * eta
    denom = (1.0 - eta) ** 3
    p_ideal = R * rho * T * (1.0 + eta + eta2 - eta3) / denom
    p_attr = a * rho * rho
    return p_ideal - p_attr


@register_eos("peng_robinson")
def pr_eos_pressure(
    rho, a=1.0 / 49.0, b=2.0 / 21.0, R=1.0, T=0.03099, omega=0.344, Tc=0.036461002037067
):
    """Peng-Robinson EOS.

    p = ρRT/(1-bρ) - a·α(T)·ρ²/(1+2bρ-(bρ)²)

    Default parameters match CUDA LBM.h eos namespace (MCMP water-like):
      a=1/49, b=2/21, R=1, ω=0.344, Tc=0.03646, T/Tc=0.85

    Parameters
    ----------
    rho : array
    a, b, R : float — EOS parameters
    T : float — physical temperature (NOT reduced). Default = 0.03099
    omega : float — acentric factor
    Tc : float or None — critical temperature (auto-computed if None)

    Returns
    -------
    p : array
    """
    if Tc is None:
        Tc = 0.17014 * a / (b * R)  # PR critical constants (≈0.03646 for a=1/49,b=2/21)
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega * omega
    Tr = T / Tc
    alpha = (1.0 + kappa * (1.0 - jnp.sqrt(Tr))) ** 2

    denom1 = 1.0 - b * rho
    denom1 = jnp.clip(denom1, 1e-10)
    denom2 = 1.0 + 2.0 * b * rho - (b * rho) ** 2
    denom2 = jnp.clip(denom2, 1e-10)

    p_ideal = R * rho * T / denom1
    p_attr = a * alpha * rho * rho / denom2
    return p_ideal - p_attr


@register_eos("redlich_kwong")
def rk_eos_pressure(rho, a=1.0, b=4.0 / 21.0, R=1.0, T=0.8):
    """Redlich-Kwong EOS.

    p = ρRT/(1-bρ) - a·ρ²/[√T·(1+bρ)]

    Parameters
    ----------
    rho : array
    a, b, R : float
    T : float

    Returns
    -------
    p : array
    """
    denom1 = 1.0 - b * rho
    denom1 = jnp.clip(denom1, 1e-10)
    denom2 = 1.0 + b * rho
    denom2 = jnp.clip(denom2, 1e-10)

    p_ideal = R * rho * T / denom1
    p_attr = a * rho * rho / (jnp.sqrt(T) * denom2)
    return p_ideal - p_attr


@register_eos("redlich_kwong_soave")
def rks_eos_pressure(rho, a=1.0, b=4.0 / 21.0, R=1.0, T=0.8, omega=0.344, Tc=None):
    """Redlich-Kwong-Soave EOS.

    p = ρRT/(1-bρ) - a·α(T)·ρ²/(1+bρ)

    Parameters
    ----------
    rho : array
    a, b, R : float
    T : float
    omega : float — acentric factor
    Tc : float or None

    Returns
    -------
    p : array
    """
    if Tc is None:
        Tc = 0.08664 * a / (0.42748 * b * R)
    kappa = 0.480 + 1.574 * omega - 0.176 * omega * omega
    Tr = T / Tc
    alpha = (1.0 + kappa * (1.0 - jnp.sqrt(Tr))) ** 2

    denom1 = 1.0 - b * rho
    denom1 = jnp.clip(denom1, 1e-10)
    denom2 = 1.0 + b * rho
    denom2 = jnp.clip(denom2, 1e-10)

    p_ideal = R * rho * T / denom1
    p_attr = a * alpha * rho * rho / denom2
    return p_ideal - p_attr


@register_eos("van_der_waals")
def vdw_eos_pressure(rho, a=1.0, b=4.0, R=1.0, T=0.066):
    """Van der Waals EOS.

    p = ρRT/(1-bρ) - a·ρ²

    Parameters
    ----------
    rho : array
    a, b, R : float
    T : float

    Returns
    -------
    p : array
    """
    denom = 1.0 - b * rho
    denom = jnp.clip(denom, 1e-10)
    p_ideal = R * rho * T / denom
    p_attr = a * rho * rho
    return p_ideal - p_attr


# ═══════════════════════════════════════════════════════════════════════════
# Pseudopotential (Shan-Chen ψ)
# ═══════════════════════════════════════════════════════════════════════════


def pseudopotential(rho, eos_fn, eos_params, G=-1.0):
    """Compute Shan-Chen pseudopotential ψ from EOS.

    ψ = sqrt(2·(p_EOS − ρ·cs²) / G)

    Parameters
    ----------
    rho : array — density field
    eos_fn : callable — pressure function p(rho, **eos_params)
    eos_params : dict — EOS parameters
    G : float — interaction strength

    Returns
    -------
    psi : array — pseudopotential (clamped to ≥ 0)
    """
    p = eos_fn(rho, **eos_params)
    p_ideal_gas = rho * CS2
    raw = 2.0 * (p - p_ideal_gas) / G
    return jnp.sqrt(jnp.maximum(raw, 0.0))


# ═══════════════════════════════════════════════════════════════════════════
# Critical constants helper
# ═══════════════════════════════════════════════════════════════════════════


def critical_temperature(eos_name: str, a=1.0, b=4.0, R=1.0) -> float:
    """Compute critical temperature for a given EOS.

    Parameters
    ----------
    eos_name : str — EOS name
    a, b, R : float — EOS parameters

    Returns
    -------
    Tc : float
    """
    if eos_name in ("carnahan_starling", "cs"):
        return 0.3773 * a / (b * R)
    elif eos_name in ("peng_robinson", "pr"):
        return 0.0778 * a / (0.45724 * b * R)  # Tc = 0.1701·a/(b·R)
    elif eos_name in ("redlich_kwong", "rk"):
        return 0.08664 * a / (0.42748 * b * R)
    elif eos_name in ("redlich_kwong_soave", "rks"):
        return 0.08664 * a / (0.42748 * b * R)
    elif eos_name in ("van_der_waals", "vdw"):
        return 8.0 * a / (27.0 * b * R)
    else:
        raise KeyError(f"Unknown EOS '{eos_name}'")
