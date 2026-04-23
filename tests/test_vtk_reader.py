"""Tests for lbm_mrt.io.vtk_reader using a synthetic VTK file."""

from __future__ import annotations

import os
import struct

import numpy as np
import pytest

from lbm_mrt.io.vtk_reader import latest_vtk, read_vtk_scalars


# ── Helpers ───────────────────────────────────────────────────────────────

def _write_synthetic_vtk(path: str, nx: int, ny: int, fields: dict) -> None:
    """Write a minimal Legacy VTK Binary structured-points file.

    Only writes the fields given; uses big-endian double format like the CUDA solver.
    """
    n = nx * ny
    with open(path, "wb") as f:
        # ASCII header (must end with LF)
        header = (
            "# vtk DataFile Version 3.0\n"
            "Synthetic LBM output\n"
            "BINARY\n"
            "DATASET STRUCTURED_POINTS\n"
            f"DIMENSIONS {nx} {ny} 1\n"
            "ORIGIN 0 0 0\n"
            "SPACING 1 1 1\n"
            f"POINT_DATA {n}\n"
        )
        f.write(header.encode("ascii"))
        for name, arr in fields.items():
            scalar_hdr = f"SCALARS {name} double 1\nLOOKUP_TABLE default\n"
            f.write(scalar_hdr.encode("ascii"))
            # Write big-endian doubles
            data_be = arr.astype(">f8").tobytes()
            f.write(data_be)


# ── Tests ─────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_vtk(tmp_path):
    """Create a single synthetic VTK file with two scalar fields."""
    nx, ny = 20, 15
    rng = np.random.default_rng(42)
    rho = rng.uniform(0.1, 7.0, size=(ny, nx))
    ux = rng.uniform(-0.01, 0.01, size=(ny, nx))

    vtk_path = tmp_path / "outputdata_flow00050000.vtk"
    _write_synthetic_vtk(str(vtk_path), nx, ny, {"rho": rho, "ux": ux})
    return vtk_path, nx, ny, rho, ux


def test_read_vtk_returns_correct_dims(synthetic_vtk) -> None:
    vtk_path, nx, ny, _, _ = synthetic_vtk
    fields, nx_r, ny_r = read_vtk_scalars(str(vtk_path))
    assert nx_r == nx
    assert ny_r == ny


def test_read_vtk_field_names(synthetic_vtk) -> None:
    vtk_path, _, _, _, _ = synthetic_vtk
    fields, _, _ = read_vtk_scalars(str(vtk_path))
    assert "rho" in fields
    assert "ux" in fields


def test_read_vtk_field_shape(synthetic_vtk) -> None:
    vtk_path, nx, ny, _, _ = synthetic_vtk
    fields, _, _ = read_vtk_scalars(str(vtk_path))
    assert fields["rho"].shape == (ny, nx)
    assert fields["ux"].shape == (ny, nx)


def test_read_vtk_values_match(synthetic_vtk) -> None:
    vtk_path, _, _, rho_orig, ux_orig = synthetic_vtk
    fields, _, _ = read_vtk_scalars(str(vtk_path))
    np.testing.assert_allclose(fields["rho"], rho_orig, rtol=1e-12)
    np.testing.assert_allclose(fields["ux"], ux_orig, rtol=1e-12)


# ── latest_vtk ────────────────────────────────────────────────────────────

def test_latest_vtk_finds_file(tmp_path) -> None:
    nx, ny = 5, 5
    rho = np.ones((ny, nx))
    for step in [50000, 100000, 150000]:
        name = f"outputdata_flow{step:08d}.vtk"
        _write_synthetic_vtk(str(tmp_path / name), nx, ny, {"rho": rho})

    result = latest_vtk(str(tmp_path), tag="flow")
    assert "00150000" in result


def test_latest_vtk_raises_if_missing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        latest_vtk(str(tmp_path), tag="flow")
