#ifndef PF_NS_2D_H
#define PF_NS_2D_H

// Phase-field 2D LBM — self-contained CUDA module (pf_ns_2d binary).
//
// Reference: Yang et al. (2024, Innov Energy) SI S10–S17 (conservative
// Allen-Cahn + chemical-potential capillary force), Guo et al. (2000)
// pressure-based LBM. The JAX golden reference lives in jax_lbm/pf/phase_field.py.
//
// pf_mode (params.txt):
//   0 = single-phase pressure-based NS only (no phase field)
//   1 = conservative Allen-Cahn only (no NS)
//   2 = AC + NS coupled static droplet (Stage 2)
//   3 = Stage 2 + surface-energy wetting (Stage 3, planned)
//
// Params parsed directly from params.txt (pf_* keys):
//   pf_mode, pf_W, pf_M, pf_sigma, pf_rho_g, pf_rho_w, pf_tau, pf_R0,
//   pf_xc, pf_yc, pf_steps, pf_output_every, pf_nx, pf_ny

int run_pf_ns_2d(const char* params_path);

#endif  // PF_NS_2D_H
