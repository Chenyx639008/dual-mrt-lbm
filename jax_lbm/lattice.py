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


# ═══════════════════════════════════════════════════════════════════════════
# D3Q19 Lattice (placeholder — 3D models pending)
# ═══════════════════════════════════════════════════════════════════════════

# D3Q19 velocities: 1 rest + 6 cardinal (face) + 12 diagonal (edge)
D3Q19_C = jnp.array(
    [
        [0, 0, 0],  # 0: rest
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],  # 1-6: cardinal
        [1, 1, 0],
        [-1, -1, 0],
        [1, -1, 0],
        [-1, 1, 0],  # 7-10: edge (z=0)
        [1, 0, 1],
        [-1, 0, -1],
        [1, 0, -1],
        [-1, 0, 1],  # 11-14: edge (y=0)
        [0, 1, 1],
        [0, -1, -1],
        [0, 1, -1],
        [0, -1, 1],  # 15-18: edge (x=0)
    ],
    dtype=jnp.int32,
)

D3Q19_Q = 19

# D3Q19 weights: w₀=1/3, w_cardinal=1/18, w_edge=1/36
D3Q19_W = jnp.array(
    [1.0 / 3.0] + [1.0 / 18.0] * 6 + [1.0 / 36.0] * 12,
    dtype=jnp.float64,
)

# D3Q19 MRT matrix placeholder (19×19)
# Reference: d'Humières et al. (2002), Phil. Trans. R. Soc. A 360, 437-451
# TODO: populate full MRT_M_3D when 3D solver is implemented

# ── Diagonal direction indices (|c|=√2) ──
DIAGONAL_INDICES = jnp.array([5, 6, 7, 8], dtype=jnp.int32)
