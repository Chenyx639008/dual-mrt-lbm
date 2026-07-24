"""D2Q9 Lattice constants — pure JAX, no classes.

Provides the fundamental D2Q9 lattice data used by all other modules.
Matches JAX-LaB `src/jax_lab/lattice.py` LatticeD2Q9.

Reference
---------
- JAX-LaB: src/jax_lab/lattice.py (LatticeD2Q9 class)
"""

import jax.numpy as jnp

# ── Discrete velocities (9 directions) ──
C = jnp.array(
    [
        [0, 0],
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
        [1, 1],
        [-1, 1],
        [-1, -1],
        [1, -1],
    ],
    dtype=jnp.int32,
)

# ── Lattice weights ──
W = jnp.array(
    [
        4.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
    ],
    dtype=jnp.float64,
)

# ── Opposite direction indices ──
OPP = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=jnp.int32)

# ── Speed of sound ──
CS2 = 1.0 / 3.0
INV_CS2 = 3.0
CS = jnp.sqrt(CS2)

# ── Number of discrete velocities ──
Q = 9

# ── Green's function weights for Shan-Chen force (Eq. 24 in Huang & Wu 2016) ──
# Rest: 0, Cardinal: 1/3, Diagonal: 1/12
G_FF = jnp.where(
    jnp.linalg.norm(C.astype(jnp.float64), axis=1) > 1.1,
    1.0 / 12.0,
    jnp.where(jnp.linalg.norm(C.astype(jnp.float64), axis=1) > 0.1, 1.0 / 3.0, 0.0),
)

# ── Cardinal direction indices (|c|=1): right, up, left, down ──
CARDINAL_INDICES = jnp.array([1, 2, 3, 4], dtype=jnp.int32)

# ── Diagonal direction indices (|c|=√2) ──
DIAGONAL_INDICES = jnp.array([5, 6, 7, 8], dtype=jnp.int32)
