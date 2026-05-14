#!/usr/bin/env python3
"""Enhanced CS-EOS coexistence visualization matching Huang & Wu (2016) Fig. 3-4.

Generates:
1. Pressure-density isotherm (single T)
2. Coexistence curve with log-scale ρ axis (matching paper format)
3. Optional: numerical simulation points overlay
4. Optional: analytical pressure tensor forms comparison

Paper reference: Huang, H., & Wu, Z. (2016). Third-order analysis of pseudopotential...
Figure 3-4: Coexistence curve comparisons across methods
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lbm_mrt.validation.cs_eos import (
    cs_pressure,
    cs_critical_point,
    coexistence_curve,
    maxwell_coexistence,
)
from lbm_mrt.viz.viz_template import init_style, create_figure_ax, save_figure

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Paper parameters (a=1.0, b=4.0, R=1.0, Tr=0.9 for Laplace validation) ──
a, b, R = 1.0, 4.0, 1.0
Tr_target = 0.9  # Paper's Laplace law validation temperature

Tc, pc, rhoc = cs_critical_point(a, b, R)
T_actual = Tr_target * Tc

print(f"CS-EOS Parameters: a={a}, b={b}, R={R}")
print(f"Critical point: Tc={Tc:.5f}, pc={pc:.5f}, ρc={rhoc:.5f}")
print(f"Target state: Tr={Tr_target}, T_actual={T_actual:.5f}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Pressure-Density Isotherm with Maxwell Construction
# ─────────────────────────────────────────────────────────────────────────────

init_style()
fig, ax = create_figure_ax(figsize=(7, 5))

# Compute isotherm: p vs ρ at fixed T
rhos_iso = np.linspace(1e-4, 0.49, 2000)
ps_iso = cs_pressure(rhos_iso, a, b, R, T_actual)

ax.plot(rhos_iso, ps_iso, "k-", lw=2.0, label=f"CS-EOS (Tr={Tr_target:.2f})")

# Maxwell tie-line (equal-area rule)
result = maxwell_coexistence(a, b, R, T_actual)
if result:
    rg, rl, peq = result
    ax.axhline(peq, color="red", ls="--", lw=1.5, label="Maxwell tie-line")
    ax.axvline(rg, color="red", ls=":", lw=1.0, alpha=0.5)
    ax.axvline(rl, color="red", ls=":", lw=1.0, alpha=0.5)

    # Fill equal-area regions
    mask_mid = (rhos_iso >= rg) & (rhos_iso <= rl)
    ax.fill_between(
        rhos_iso[mask_mid],
        ps_iso[mask_mid],
        peq,
        where=(ps_iso[mask_mid] > peq),
        color="red",
        alpha=0.1,
        label="Correction region",
    )
    ax.fill_between(
        rhos_iso[mask_mid],
        ps_iso[mask_mid],
        peq,
        where=(ps_iso[mask_mid] < peq),
        color="blue",
        alpha=0.1,
    )

    # Annotate coexistence point
    ax.plot([rg, rl], [peq, peq], "r-", lw=2.5, alpha=0.8)
    ax.plot(rg, peq, "ro", ms=8, alpha=0.6)
    ax.plot(rl, peq, "ro", ms=8, alpha=0.6)

# Critical point
ax.axvline(rhoc, color="gray", ls=":", lw=1.0, alpha=0.5)
ax.text(
    rhoc,
    ax.get_ylim()[1] * 0.8,
    f"ρc≈{rhoc:.3f}",
    fontsize=8,
    color="gray",
    ha="right",
)

ax.set_xlabel(r"Density $\rho$")
ax.set_ylabel(r"Pressure $p$")
ax.set_title(
    f"Carnahan-Starling EOS: p-ρ Isotherm (Tr={Tr_target})\n"
    "Maxwell construction: equal-area rule"
)
ax.legend(fontsize=8, loc="upper left")
ax.set_xlim(0, 0.49)
ax.grid(True, alpha=0.2)

save_figure(fig, os.path.join(OUT_DIR, "cs_isotherm_maxwell_improved"), dpi=150)
plt.close(fig)
print(f"✓ Saved: {OUT_DIR}/cs_isotherm_maxwell_improved.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: Coexistence Curve — LOG SCALE (matching Huang & Wu Fig. 3-4)
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = create_figure_ax(figsize=(8, 6))

# Compute coexistence curve: T-sweep
T_vals = np.linspace(0.55 * Tc, 0.99 * Tc, 50)
curve = coexistence_curve(a, b, R, T_vals)

if curve:
    Tr_vals = curve["T"] / Tc
    rho_l = curve["rho_l"]
    rho_g = curve["rho_g"]

    # Quality check: remove solver artifacts (non-monotonic transitions)
    ok = np.ones(len(Tr_vals), dtype=bool)
    for i in range(1, len(Tr_vals) - 1):
        # ρ_l should be decreasing (toward ρc), ρ_g should be increasing
        # Allow some tolerance for numerical noise
        if i > 5:  # Skip first few points which may have transient noise
            if not (rho_l[i - 1] >= rho_l[i] - 0.02 or i < 10):
                ok[i] = False

    # Plot coexistence branches
    valid_idx = np.where(ok)[0]
    if len(valid_idx) > 10:
        ax.plot(
            rho_l[valid_idx],
            Tr_vals[valid_idx],
            "b-",
            lw=2.5,
            label="Maxwell (liquid)",
            alpha=0.8,
        )
        ax.plot(
            rho_g[valid_idx],
            Tr_vals[valid_idx],
            "r-",
            lw=2.5,
            label="Maxwell (gas)",
            alpha=0.8,
        )

    # Critical point
    ax.plot(rhoc, 1.0, "k*", ms=15, label="Critical point", zorder=5)

# ────────────────────────────────────────────────────────────────────────────
# OVERLAY: Numerical simulation points (if available)
# ────────────────────────────────────────────────────────────────────────────

# Try to load numerical data from earlier SCMP simulations
try:
    from lbm_mrt.validation.coexistence import load_coexistence_reference

    ref = load_coexistence_reference()
    # Note: ref contains Maxwell analytical values, but we'll keep structure
    # for future overlay of actual SCMP results
except Exception:
    ref = None

# Placeholder for numerical data from flat-interface SCMP runs
# Format: list of tuples (tau, Tr, rho_g, rho_l)
# Example (to be populated by Phase 5 suite):
# numerical_points = [
#     (1.0, 0.75, 0.012, 0.15),
#     (1.5, 0.75, 0.011, 0.16),
# ]
# if numerical_points:
#     tau_1 = [p for p in numerical_points if abs(p[0] - 1.0) < 0.01]
#     tau_15 = [p for p in numerical_points if abs(p[0] - 1.5) < 0.01]
#     if tau_1:
#         Tr_1, rho_g_1, rho_l_1 = zip(*[(p[1], p[2], p[3]) for p in tau_1])
#         ax.plot(rho_g_1, Tr_1, 'go', ms=7, label='τ=1.0 (numerical, gas)', alpha=0.6)
#         ax.plot(rho_l_1, Tr_1, 'go', ms=7, alpha=0.6)
#     if tau_15:
#         Tr_15, rho_g_15, rho_l_15 = zip(*[(p[1], p[2], p[3]) for p in tau_15])
#         ax.plot(rho_g_15, Tr_15, 'mx', ms=8, label='τ=1.5 (numerical, gas)', alpha=0.6)
#         ax.plot(rho_l_15, Tr_15, 'mx', ms=8, alpha=0.6)

# Annotate reference point (Tr=0.9 for Laplace law validation)
result_09 = maxwell_coexistence(a, b, R, 0.9 * Tc)
if result_09:
    rg09, rl09, _ = result_09
    ax.plot([rg09, rl09], [0.9, 0.9], "g--", lw=1.5, alpha=0.5)
    ax.plot(rg09, 0.9, "go", ms=7, alpha=0.6)
    ax.plot(rl09, 0.9, "go", ms=7, alpha=0.6)
    ax.text(
        rg09 * 0.7,
        0.9 + 0.01,
        f"Tr=0.9 (Laplace)\nρ_g={rg09:.4f}\nρ_l={rl09:.4f}",
        fontsize=7,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

# Formatting (matching Huang & Wu paper style)
ax.set_xlabel(r"Density $\rho$ (log scale)", fontsize=11)
ax.set_ylabel(r"Reduced Temperature $T/T_c$", fontsize=11)
ax.set_title(
    "Carnahan-Starling EOS: Coexistence Curve\nFormat: Huang & Wu (2016) Fig. 3-4",
    fontsize=12,
    fontweight="bold",
)

# LOG SCALE on ρ axis — extended to 0.5 to capture liquid branch (ρ_l up to 0.43)
ax.set_xscale("log")
ax.set_xlim(1e-3, 0.5)
ax.set_ylim(0.55, 1.02)

ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=9, loc="upper right", framealpha=0.95)

save_figure(fig, os.path.join(OUT_DIR, "cs_coexistence_curve_logscale"), dpi=150)
plt.close(fig)
print(f"✓ Saved: {OUT_DIR}/cs_coexistence_curve_logscale.png")

# ─────────────────────────────────────────────────────────────────────────────
# Save numerical data for reference
# ─────────────────────────────────────────────────────────────────────────────

if curve:
    ref_out = {
        "description": "CS-EOS (a=1, b=4, R=1) Maxwell coexistence curve",
        "method": "Equal-area (chemical potential equality) via scipy.optimize.fsolve",
        "a": float(a),
        "b": float(b),
        "R": float(R),
        "Tc": float(Tc),
        "pc": float(pc),
        "rhoc": float(rhoc),
        "T": [float(t) for t in curve["T"]],
        "T_reduced": [float(t / Tc) for t in curve["T"]],
        "rho_g": [float(rg) for rg in curve["rho_g"]],
        "rho_l": [float(rl) for rl in curve["rho_l"]],
        "note": (
            "ρ_g and ρ_l obtained from Maxwell construction. "
            "Compare to numerical SCMP simulation points for validation."
        ),
    }

    with open(os.path.join(OUT_DIR, "cs_coexistence_reference.json"), "w") as f:
        json.dump(ref_out, f, indent=2)

    print(f"✓ Saved: {OUT_DIR}/cs_coexistence_reference.json")

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE — Ready for numerical data comparison")
print("=" * 70)
print(
    "\nNext steps (Phase 5):"
    "\n  1. Run flat-interface SCMP simulations at different T and τ values"
    "\n  2. Extract (ρ_g, ρ_l) from equilibrium density profiles"
    "\n  3. Overlay numerical points on coexistence_curve_logscale.png"
    "\n  4. Verify agreement with Maxwell construction (< 2% deviation)"
)
print("\nFigure output directory: " + os.path.abspath(OUT_DIR))
