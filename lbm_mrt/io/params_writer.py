"""Serializes a flat params dict to the key-value params.txt format.

The C++ load_params_txt() in sim_utils.cu reads one "key value" pair per line,
ignoring blank lines and comments (# ...). This module writes that format.
"""

from __future__ import annotations

import os
from typing import Any


def fmt_value(v: Any) -> str:
    """Format a single value for params.txt output.

    - Booleans → 'true' / 'false'
    - Integer-valued floats → plain integer string (e.g. 50000.0 → "50000")
    - Other floats → 10-significant-figure representation
    - Everything else → str(v)

    Args:
        v: The value to format.

    Returns:
        String representation suitable for params.txt.
    """
    if isinstance(v, bool):
        return "true" if v else "false"
    try:
        iv = int(v)
        if float(v) == iv:
            return str(iv)
    except (TypeError, ValueError):
        pass
    try:
        return f"{float(v):.10g}"
    except (TypeError, ValueError):
        return str(v)


def write_params_txt(params: dict[str, Any], path: str | os.PathLike) -> None:
    """Write a flat params dict to a params.txt file.

    Each entry is written as "key value" on its own line, in dict iteration order.
    The CUDA binary's load_params_txt() ignores keys it does not recognize,
    so it is safe to pass a superset of the expected keys.

    Args:
        params: Flat params dict (e.g. from lbm_mrt.core.config.load_config()).
        path:   Destination file path; parent directory must already exist.
    """
    with open(path, "w") as f:
        for k, v in params.items():
            if v == "" or v is None:
                # Skip empty/None values (e.g. geom_file when not set)
                continue
            f.write(f"{k} {fmt_value(v)}\n")
