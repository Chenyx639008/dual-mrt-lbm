"""Shared fixtures for JAX LBM test suite.

Usage:
    JAX_ENABLE_X64=1 uv run pytest jax_lbm/tests/ -v
"""

import jax
import jax.numpy as jnp
import pytest

# ── Force x64 for all tests ──
jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="session")
def scmp_production():
    """Production SCMP parameters (ε=1.7, T/Tc=0.70)."""
    return {
        "rho_l": 0.305202,
        "rho_g": 0.009170,
        "T_reduced": 0.70,
        "tau": 1.5,
        "epsilon": 1.7,
        "k1": -1.7 / 8.0,
    }


@pytest.fixture(scope="session")
def scmp_near_critical():
    """Near-critical SCMP parameters (ε=-0.48, T/Tc=0.90)."""
    return {
        "rho_l": 0.248108,
        "rho_g": 0.045410,
        "T_reduced": 0.90,
        "tau": 1.5,
        "epsilon": -0.48,
        "k1": 0.06,
    }


@pytest.fixture(scope="session")
def mrt_relaxation():
    """CUDA-matched MRT relaxation rates (τ=1.5, Λ=1/12)."""
    tau = 1.5
    s_p = 1.0 / tau
    Lambda = 1.0 / 12.0
    s_q = 1.0 / (0.5 + Lambda / (tau - 0.5))
    return jnp.array([1.0, s_p, s_p, 1.0, s_q, 1.0, s_q, s_p, s_p])


@pytest.fixture(scope="session")
def cs_eos_temperature():
    """CS-EOS temperature for T/Tc=0.70."""
    Tc = 0.3773 / 4.0
    return 0.70 * Tc


@pytest.fixture(scope="session")
def small_grid():
    """Small grid for quick tests."""
    return 64, 64


@pytest.fixture(scope="session")
def tiny_grid():
    """Tiny grid for unit tests."""
    return 16, 16
