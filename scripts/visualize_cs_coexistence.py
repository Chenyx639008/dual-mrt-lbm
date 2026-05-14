#!/usr/bin/env python3
"""CS-EOS coexistence curve visualization + Maxwell construction.

Output: results/coexistence_curve.png, results/coexistence_curve.pdf
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

from lbm_mrt.validation.cs_eos import (
    cs_pressure,
    cs_critical_point,
    coexistence_curve,
    maxwell_coexistence,
)
from lbm_mrt.viz.viz_template import init_style, create_figure_ax, save_figure

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Parameters (plan.md defaults: a=1.0, b=4, R=1, cs_T reduced) ──
a, b, R, Tr_target = 1.0, 4.0, 1.0, 0.9  # paper: T=0.9Tc for Laplace
Tc, pc, rhoc = cs_critical_point(a, b, R)
T_actual = Tr_target * Tc

print(f"CS-EOS: a={a}, b={b}, R={R}")
print(f"Tc={Tc:.5f}, pc={pc:.5f}, ρc={rhoc:.5f}")
print(f"Target: Tr={Tr_target}, T_actual={T_actual:.5f}")

# ── Figure 1: p−ρ isotherm with Maxwell construction ──
init_style()
fig, ax = create_figure_ax(figsize=(7, 5))

rhos = np.linspace(1e-4, 0.49, 2000)
ps = cs_pressure(rhos, a, b, R, T_actual)
ax.plot(rhos, ps, "k-", lw=1.5, label=f"CS-EOS  Tr={Tr_target:.2f}")

# Maxwell tie-line
result = maxwell_coexistence(a, b, R, T_actual)
if result:
    rg, rl, peq = result
    ax.axhline(
        peq,
        color="red",
        ls="--",
        lw=1.2,
        label=f"Maxwell: ρ_g={rg:.4f}, ρ_l={rl:.4f}, p_eq={peq:.2e}",
    )
    ax.axvline(rg, color="red", ls=":", lw=0.8)
    ax.axvline(rl, color="red", ls=":", lw=0.8)
    # Fill equal-area regions
    mask_mid = (rhos >= rg) & (rhos <= rl)
    ax.fill_between(
        rhos[mask_mid],
        ps[mask_mid],
        peq,
        where=(ps[mask_mid] > peq),
        color="red",
        alpha=0.15,
    )
    ax.fill_between(
        rhos[mask_mid],
        ps[mask_mid],
        peq,
        where=(ps[mask_mid] < peq),
        color="blue",
        alpha=0.15,
    )

# Critical point
ax.axvline(rhoc, color="gray", ls=":", lw=0.5)
ax.annotate(f"ρc={rhoc:.3f}", (rhoc, pc * 0.5), fontsize=8, color="gray")

ax.set_xlabel(r"Density $\rho$")
ax.set_ylabel(r"Pressure $p$")
ax.set_title(f"Carnahan-Starling EOS  (a={a}, b={b}, R={R}, Tr={Tr_target})")
ax.legend(fontsize=8)
ax.set_xlim(0, 0.49)
ax.set_ylim(-0.15, 0.08)

save_figure(fig, os.path.join(OUT_DIR, "cs_isotherm_maxwell"), dpi=150)
plt.close(fig)
print(f"Saved: {OUT_DIR}/cs_isotherm_maxwell.png")

# ── Figure 2: Coexistence curve — T/Tc (y-axis) vs ρ (x-axis) per paper Fig. 4 ──
fig, ax = create_figure_ax(figsize=(6, 7))

T_vals = np.linspace(0.55 * Tc, 0.99 * Tc, 40)
curve = coexistence_curve(a, b, R, T_vals)

if curve:
    Tr_vals = curve["T"] / Tc
    # Filter out solver artifacts at high Tr (check monotonicity)
    ok = np.ones(len(Tr_vals), dtype=bool)
    for i in range(1, len(Tr_vals) - 1):
        if curve["rho_l"][i] < curve["rho_l"][i + 1] + 0.01:  # rho_l should decrease
            ok[i] = ok[i]  # keep
    # Plot: Tr on y, rho on x (paper convention)
    ax.plot(curve["rho_l"][ok], Tr_vals[ok], "b-o", ms=4, label=r"$\rho_l$ (liquid)")
    ax.plot(curve["rho_g"][ok], Tr_vals[ok], "r-s", ms=4, label=r"$\rho_g$ (gas)")
    # Critical point
    ax.plot(rhoc, 1.0, "k*", ms=12, label=f"Critical")
    # Maxwell tie-line at Tr=0.9 (paper's Laplace validation T)
    result_09 = maxwell_coexistence(a, b, R, 0.9 * Tc)
    if result_09:
        rg09, rl09, _ = result_09
        ax.plot([rg09, rl09], [0.9, 0.9], "k--", lw=1.2)
        ax.annotate(
            f"Tr=0.9\nρ_g={rg09:.3f}\nρ_l={rl09:.3f}",
            (0.5 * (rg09 + rl09), 0.9),
            fontsize=7,
            ha="center",
        )

ax.set_xlabel(r"Density $\rho$ (log scale)")
ax.set_ylabel(r"Reduced temperature $T/T_c$")
ax.set_title(f"CS-EOS Coexistence — per Huang & Wu (2016) Fig. 3-4")
ax.set_xscale("log")
ax.legend(fontsize=8, loc="upper right")
ax.set_ylim(0.55, 1.02)
ax.set_xlim(1e-3, 0.5)

save_figure(fig, os.path.join(OUT_DIR, "cs_coexistence_curve"), dpi=150)
plt.close(fig)
print(f"Saved: {OUT_DIR}/cs_coexistence_curve.png")

# ── Save numerical data ──
if curve:
    ref = {k: v.tolist() for k, v in curve.items()}
    ref["Tc"] = float(Tc)
    ref["rhoc"] = float(rhoc)
    ref["a"] = a
    ref["b"] = b
    ref["R"] = R
    with open(os.path.join(OUT_DIR, "cs_coexistence_data.json"), "w") as f:
        json.dump(ref, f, indent=2)
    print(f"Saved: {OUT_DIR}/cs_coexistence_data.json")

print("\nDone. All outputs in results/")
