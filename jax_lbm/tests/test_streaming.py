"""JAX LBM streaming tests.

Run: JAX_ENABLE_X64=1 uv run pytest jax_lbm/tests/test_streaming.py -v
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from jax_lbm.lattice import C, Q
from jax_lbm.d2q9_bgk import streaming

class TestStreamingDirection:
    """Particle at (x,y) with velocity c_k should move to (x+cx, y+cy)."""

    # Lattice velocities (must match C array)
    _CX = [0, 1, 0, -1, 0, 1, -1, -1, 1]
    _CY = [0, 0, 1, 0, -1, 1, 1, -1, -1]

    @pytest.mark.parametrize("k", range(1, 9))  # skip rest particle (k=0)
    def test_particle_moves_in_c_direction(self, k):
        """f at (nx/2, ny/2) in direction k moves to (nx/2+cx, ny/2+cy)."""
        nx, ny = 8, 8
        cx, cy = self._CX[k], self._CY[k]

        f = jnp.zeros((nx, ny, Q))
        f = f.at[nx // 2, ny // 2, k].set(1.0)

        fs = streaming(f)

        expected_x = (nx // 2 + cx) % nx
        expected_y = (ny // 2 + cy) % ny

        assert fs[expected_x, expected_y, k] == 1.0, (
            f"k={k} c=({cx},{cy}): value should be at ({expected_x},{expected_y}), "
            f"not elsewhere. Streaming direction may be wrong!"
        )
        # Original position should be empty
        assert fs[nx // 2, ny // 2, k] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: Force at uniform field must be zero
# ═══════════════════════════════════════════════════════════════════════════

