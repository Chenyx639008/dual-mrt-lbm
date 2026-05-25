#!/usr/bin/env python3
"""Quick batch Gx sweep to verify Darcy linearity."""

import subprocess, os
from pathlib import Path

ROOT = Path(
    "/home/server/projects/lbm_twoflow/complex-porous-media/worktrees/huang_mrt_2d"
)
BIN = ROOT / "lbm_mrt/solver/mcmp_huang_porous_300x300"
GEOM = "data/geometry/for_lbm/geometry_case0000.plt"

GX_VALUES = [1e-8, 2e-8, 5e-8, 1e-7]

for gx in GX_VALUES:
    cid = f"Gx{gx:.0e}"
    run_dir = ROOT / "results/batch_test" / cid
    run_dir.mkdir(parents=True, exist_ok=True)

    params = run_dir / "params.txt"
    with open(params, "w") as f:
        f.write(f"""pp_mode 1
huang_init_mode 5
epsilon_huang 1.7
k2_huang 0.0
tau_huang 1.5
Lambda_huang 0.08333
alpha_meq 1.0
cs_a 1.0
cs_b 4.0
cs_R 1.0
cs_T 0.9
cs_G -1.0
huang_rho_l 1.0
huang_rho_g 1.0
huang_u_max 0.15
huang_psi_cut 1.0e-3
theta_contact_deg 90.0
thetaA_quartz_deg 30.0
thetaA_hydrate_deg 30.0
G_ads 0.0
Gx {gx:.8e}
Gy 0.0
drive_mode 1
geom_file {GEOM}
OUTPUT_EVERY 100000
flow_tol_rel 1.0e-5
flow_need_consec 3
flow_max_steps 100000
file_dir {run_dir}
""")

    print(f"\n=== Gx={gx:.1e} ===")
    r = subprocess.run(
        [str(BIN), str(params)], cwd=ROOT, capture_output=True, text=True, timeout=120
    )
    for line in r.stdout.splitlines():
        if any(k in line for k in ["converged", "steady at", "Q="]):
            print(line.strip())
    for line in r.stderr.splitlines():
        if "error" in line.lower():
            print(f"ERR: {line.strip()}")

# Summary
print("\n=== SUMMARY ===")
for gx in GX_VALUES:
    cid = f"Gx{gx:.0e}"
    summary = ROOT / "results/batch_test" / cid / cid / "run_summary.txt"
    if summary.exists():
        with open(summary) as f:
            for line in f:
                if line.startswith("Q "):
                    q = float(line.split()[1])
                    print(f"Gx={gx:.1e}  Q={q:.6e}  Q/Gx={q / gx:.1f}")
