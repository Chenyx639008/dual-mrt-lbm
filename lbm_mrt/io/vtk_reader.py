"""VTK Legacy Binary reader for LBM simulation output.

Reads the structured-points VTK files produced by the CUDA solver's
outputvtk() function (big-endian binary, SCALARS double/int format).
"""

from __future__ import annotations

import glob
import os
import struct

import numpy as np


def read_vtk_scalars(vtk_path: str) -> tuple[dict[str, np.ndarray], int, int]:
    """Read scalar fields from a Legacy VTK Binary structured-points file.

    Only supports SCALARS with types double or int (the formats written by
    the LBM solver). Fields are stored in big-endian order on disk and are
    byte-swapped to native endianness on read.

    Args:
        vtk_path: Path to the .vtk file.

    Returns:
        Tuple (fields, nx, ny) where:
        - fields is a dict mapping field names to 2-D arrays of shape (ny, nx).
        - nx, ny are the grid dimensions parsed from the DIMENSIONS header.
    """
    data = open(vtk_path, "rb").read()

    # Locate POINT_DATA section boundary
    header_end = data.find(b"\nPOINT_DATA ")

    # Parse grid dimensions from the ASCII header
    header_str = data[:header_end].decode("ascii", "ignore")
    dims_line = next(
        ln for ln in header_str.split("\n") if ln.startswith("DIMENSIONS")
    )
    nx, ny, _ = map(int, dims_line.split()[1:4])

    fields: dict[str, np.ndarray] = {}
    pos = header_end

    while pos < len(data):
        chunk = data[pos : pos + 512].decode("ascii", "ignore")
        if "SCALARS" not in chunk:
            pos += 1
            continue

        lines = chunk.split("\n")
        found = False
        for ln in lines:
            if not ln.startswith("SCALARS"):
                continue
            parts = ln.split()
            name = parts[1]
            dtype = np.float64 if "double" in ln else np.int32
            nbytes = 8 if dtype == np.float64 else 4

            lookup_pos = data.find(b"LOOKUP_TABLE default\n", pos)
            if lookup_pos < 0:
                break
            raw_start = lookup_pos + len(b"LOOKUP_TABLE default\n")
            n_vals = nx * ny
            raw = data[raw_start : raw_start + n_vals * nbytes]
            arr = np.frombuffer(raw, dtype=dtype).byteswap()  # big-endian → native
            fields[name] = arr.reshape((ny, nx))
            pos = raw_start + n_vals * nbytes
            found = True
            break

        if not found:
            pos += 1

    return fields, nx, ny


def latest_vtk(folder: str, tag: str = "flow") -> str:
    """Return the path to the most recent VTK output file for a given stage.

    Args:
        folder: Directory containing VTK output files.
        tag:    Stage prefix, either "flow" or "eq".

    Returns:
        Absolute path to the last (alphabetically/numerically) matching file.

    Raises:
        FileNotFoundError: If no matching files are found.
    """
    pattern = os.path.join(folder, f"outputdata_{tag}*.vtk")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No VTK files matching: {pattern}")
    return files[-1]
