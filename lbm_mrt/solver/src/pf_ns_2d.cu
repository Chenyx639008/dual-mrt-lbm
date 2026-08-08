// pf_ns_2d.cu — Phase-field 2D LBM, self-contained CUDA module.
//
// Implements the conservative Allen-Cahn + pressure-based NS two-phase model
// of Yang et al. (2024, Innov Energy) SI S10–S17 with the Guo (2000)
// pressure-form LBM. Compiled into its own binary (pf_ns_2d); does NOT touch
// the existing SC/MCMP/hydrate kernels.
//
// JAX golden reference: jax_lbm/pf/phase_field.py (formulas cross-verified).
//
// pf_mode:
//   0 = single-phase pressure-based NS (Poiseuille validation)
//   1 = conservative Allen-Cahn only (interface profile / conservation)
//   2 = AC + NS coupled static droplet (Stage 2 benchmarks)
//
// Lattice: D2Q9, c_s² = 1/3, Δx = Δt = 1.
// Layout: row-major (y*nx + x); x = e[.][0], y = e[.][1].
// Boundary: fully periodic (matching the JAX verification track).
//
// Per-step order (matches JAX coupled_ac_ns_step):
//   1. pf_ns_velocity   → u = (Σfe + F/2)/ρ(φ)   (pre-collision velocity)
//   2. pf_ac_step       → φ^{n+1} = AC(φ, u)     (advection + anti-diffusion)
//   3. pf_ns_collide_stream → collision (Guo force) + periodic streaming

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>
#include "../include/pf_ns_2d.h"

// ── Compile-time defaults (overridable via -DPF_NX/-DPF_NY) ──
#ifndef PF_NX
#define PF_NX 256
#endif
#ifndef PF_NY
#define PF_NY 256
#endif

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t e = (call);                                                 \
        if (e != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e),  \
                    __FILE__, __LINE__);                                        \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// D2Q9 lattice (local, independent of LBM.h)
// ─────────────────────────────────────────────────────────────────────────────
__constant__ int   d_e[9][2];
__constant__ double d_w[9];
static const int HOST_E[9][2] = {
    {0, 0},  {1, 0},  {0, 1},  {-1, 0}, {0, -1},
    {1, 1},  {-1, 1}, {-1, -1}, {1, -1},
};
static const double HOST_W[9] = {
    4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
};

// ─────────────────────────────────────────────────────────────────────────────
// Simple params.txt parser (self-contained; only pf_* keys consumed)
// ─────────────────────────────────────────────────────────────────────────────
struct PFParams {
    int    mode        = 2;
    double W           = 6.0;
    double M           = 0.02;
    double sigma       = 0.01;
    double rho_g       = 0.1;
    double rho_w       = 1.0;
    double tau         = 0.8;
    double R0          = 30.0;
    double xc          = -1.0;   // <0 → centre
    double yc          = -1.0;
    double gx          = 0.0;
    double gy          = 0.0;
    int    steps       = 10000;
    int    output_every= 1000;
    int    nx          = 0;      // 0 → PF_NX
    int    ny          = 0;      // 0 → PF_NY
};

static PFParams load_pf_params(const char* path) {
    PFParams P;
    std::ifstream in(path);
    if (!in.is_open()) {
        fprintf(stderr, "[pf] warning: cannot open %s, using defaults\n", path);
        return P;
    }
    std::string line;
    while (std::getline(in, line)) {
        size_t b = line.find_first_not_of(" \t\r");
        if (b == std::string::npos) continue;
        size_t eq = line.find('=', b);
        if (eq == std::string::npos) continue;
        size_t key_end = line.find_first_of(" \t\r", b);
        if (key_end == std::string::npos || key_end > eq) key_end = eq;
        std::string key = line.substr(b, key_end - b);
        double val = atof(line.c_str() + eq + 1);
        if      (key == "pf_mode")          P.mode = (int)val;
        else if (key == "pf_W")             P.W = val;
        else if (key == "pf_M")             P.M = val;
        else if (key == "pf_sigma")         P.sigma = val;
        else if (key == "pf_rho_g")         P.rho_g = val;
        else if (key == "pf_rho_w")         P.rho_w = val;
        else if (key == "pf_tau")           P.tau = val;
        else if (key == "pf_R0")            P.R0 = val;
        else if (key == "pf_xc")            P.xc = val;
        else if (key == "pf_yc")            P.yc = val;
        else if (key == "pf_gx")            P.gx = val;
        else if (key == "pf_gy")            P.gy = val;
        else if (key == "pf_steps")         P.steps = (int)val;
        else if (key == "pf_output_every")  P.output_every = (int)val;
        else if (key == "pf_nx")            P.nx = (int)val;
        else if (key == "pf_ny")            P.ny = (int)val;
        else if (key == "nx_override")      P.nx = (int)val;
        else if (key == "ny_override")      P.ny = (int)val;
        else if (key == "flow_max_steps")   P.steps = (int)val;
    }
    if (P.nx <= 0) P.nx = PF_NX;
    if (P.ny <= 0) P.ny = PF_NY;
    if (P.xc < 0)  P.xc = 0.5 * P.nx;
    if (P.yc < 0)  P.yc = 0.5 * P.ny;
    return P;
}

// ─────────────────────────────────────────────────────────────────────────────
// Device helpers
// ─────────────────────────────────────────────────────────────────────────────

// Isotropic gradient + 9-point Laplacian of phi at (x, y), periodic.
__device__ inline void phi_grad_lap(const double* phi, int nx, int ny, int x, int y,
                                    double& dphix, double& dphiy, double& lap) {
    int xm = (x - 1 + nx) % nx, xp = (x + 1) % nx;
    int ym = (y - 1 + ny) % ny, yp = (y + 1) % ny;
    double c  = phi[y * nx + x];
    double E  = phi[y * nx + xp], W_ = phi[y * nx + xm];
    double N  = phi[yp * nx + x], S = phi[ym * nx + x];
    double NE = phi[yp * nx + xp], NW = phi[yp * nx + xm];
    double SE = phi[ym * nx + xp], SW = phi[ym * nx + xm];
    dphix = (1.0 / 3.0) * (E - W_) + (1.0 / 12.0) * (NE + SE - NW - SW);
    dphiy = (1.0 / 3.0) * (N - S) + (1.0 / 12.0) * (NE + NW - SE - SW);
    lap   = (4.0 * (E + W_ + N + S) + (NE + NW + SE + SW) - 20.0 * c) / 6.0;
}

// anti-diffusion coefficient (1/W)(1−tanh²(½ln(φ/(1−φ))))
__device__ inline double anti_coef(double ph, double W) {
    double phc = fmin(fmax(ph, 1e-10), 1.0 - 1e-10);
    double s = 0.5 * log(phc / (1.0 - phc));
    return (1.0 / W) * (1.0 - tanh(s) * tanh(s));
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: current velocity u = (Σfe + F/2)/ρ(φ)  (pre-collision)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void pf_ns_velocity(const double* f, const double* phi, double* ux,
                               double* uy, double sigma, double W, double rho_g,
                               double rho_w, double gx, double gy, int nx, int ny,
                               int mode) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    int idx = y * nx + x;
    double ph = (mode >= 1) ? phi[idx] : 1.0;
    double rho = rho_g + ph * (rho_w - rho_g);
    double momx = 0.0, momy = 0.0;
    for (int k = 0; k < 9; k++) {
        double fk = f[9 * idx + k];
        momx += fk * d_e[k][0];
        momy += fk * d_e[k][1];
    }
    double fcx = 0.0, fcy = 0.0;
    if (mode == 2) {
        double dphix, dphiy, lap;
        phi_grad_lap(phi, nx, ny, x, y, dphix, dphiy, lap);
        double beta = 12.0 * sigma / W, kappa = 1.5 * sigma * W;
        double mu = 4.0 * beta * ph * (ph - 1.0) * (ph - 0.5) - kappa * lap;
        fcx = mu * dphix;
        fcy = mu * dphiy;
    }
    double Fx = fcx + rho * gx, Fy = fcy + rho * gy;
    ux[idx] = (momx + 0.5 * Fx) / rho;
    uy[idx] = (momy + 0.5 * Fy) / rho;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: conservative Allen-Cahn update (Yang S10, direct divergence form)
//   φ^{n+1} = φ − ∇·(φu) + 0.5·∇·(u·∇φ) + ∇·[M(∇φ − θn)]   (Lax–Wendroff)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void pf_ac_step(const double* phi, const double* ux, const double* uy,
                           double* phi_new, double M, double W, int nx, int ny) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    int idx  = y * nx + x;
    int xp = (x + 1) % nx, xm = (x - 1 + nx) % nx;
    int yp = (y + 1) % ny, ym = (y - 1 + ny) % ny;
    int idxp = y * nx + xp, idxm = y * nx + xm;
    int idxyp = yp * nx + x, idxym = ym * nx + x;
    double ph = phi[idx];

    // diffusion-flux divergence: ∇·[M(∇φ − θn)]
    double jxp, jxm, jyp, jym;
    {
        double a, b, c;
        // +x flux
        phi_grad_lap(phi, nx, ny, xp, y, a, b, c);
        double g = sqrt(a * a + b * b) + 1e-12;
        jxp = M * (a - anti_coef(phi[idxp], W) * (a / g));
        // −x flux
        phi_grad_lap(phi, nx, ny, xm, y, a, b, c);
        g = sqrt(a * a + b * b) + 1e-12;
        jxm = M * (a - anti_coef(phi[idxm], W) * (a / g));
        // +y flux
        phi_grad_lap(phi, nx, ny, x, yp, a, b, c);
        g = sqrt(a * a + b * b) + 1e-12;
        jyp = M * (b - anti_coef(phi[idxyp], W) * (b / g));
        // −y flux
        phi_grad_lap(phi, nx, ny, x, ym, a, b, c);
        g = sqrt(a * a + b * b) + 1e-12;
        jym = M * (b - anti_coef(phi[idxym], W) * (b / g));
    }
    double divJ = 0.5 * (jxp - jxm) + 0.5 * (jyp - jym);

    // advection ∇·(φu) — central differences (matches JAX conservative_ac_step)
    double div_adv = 0.5 * (ux[idxp] * phi[idxp] - ux[idxm] * phi[idxm])
                   + 0.5 * (uy[idxyp] * phi[idxyp] - uy[idxym] * phi[idxym]);

    // Lax–Wendroff correction 0.5·∇·(u·∇φ) — matches JAX exactly:
    //   A[x] = ux[x]·0.5(φ[x+1]−φ[x−1]);  corr_x = 0.5(A[x+1]−A[x−1])
    //   corr = 0.5(corr_x + corr_y)
    int idxp2 = y * nx + ((x + 2) % nx);
    int idxm2 = y * nx + ((x - 2 + nx) % nx);
    int idxyp2 = ((y + 2) % ny) * nx + x;
    int idxym2 = ((y - 2 + ny) % ny) * nx + x;
    double a_xp = ux[idxp] * (0.5 * (phi[idxp2] - ph));    // u·dφdx at x+1
    double a_xm = ux[idxm] * (0.5 * (ph - phi[idxm2]));    // u·dφdx at x−1
    double a_yp = uy[idxyp] * (0.5 * (phi[idxyp2] - ph));  // v·dφdy at y+1
    double a_ym = uy[idxym] * (0.5 * (ph - phi[idxym2]));  // v·dφdy at y−1
    double corr = 0.5 * (0.5 * (a_xp - a_xm) + 0.5 * (a_yp - a_ym));

    phi_new[idx] = ph - div_adv + corr + divJ;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3: NS collision (pressure-based BGK + Guo force) + periodic streaming
// ─────────────────────────────────────────────────────────────────────────────
__global__ void pf_ns_collide_stream(const double* f, double* fnew, const double* phi,
                                     const double* ux, const double* uy,
                                     double sigma, double W, double rho_g,
                                     double rho_w, double omega, int nx, int ny,
                                     int mode) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    int idx = y * nx + x;
    double ph = (mode >= 1) ? phi[idx] : 1.0;
    double rho = rho_g + ph * (rho_w - rho_g);
    double fv[9], feq[9], Fi[9];
    double p = 0.0;
    for (int k = 0; k < 9; k++) { fv[k] = f[9 * idx + k]; p += fv[k]; }
    double fcx = 0.0, fcy = 0.0;
    if (mode == 2) {
        double dphix, dphiy, lap;
        phi_grad_lap(phi, nx, ny, x, y, dphix, dphiy, lap);
        double beta = 12.0 * sigma / W, kappa = 1.5 * sigma * W;
        double mu = 4.0 * beta * ph * (ph - 1.0) * (ph - 0.5) - kappa * lap;
        fcx = mu * dphix;
        fcy = mu * dphiy;
    }
    double uu = ux[idx], vv = uy[idx];
    double u2 = uu * uu + vv * vv;
    double uF = uu * fcx + vv * fcy;
    for (int k = 0; k < 9; k++) {
        double cu = uu * d_e[k][0] + vv * d_e[k][1];
        feq[k] = d_w[k] * (p + rho * (3.0 * cu + 4.5 * cu * cu - 1.5 * u2));
        double edotF = fcx * d_e[k][0] + fcy * d_e[k][1];
        double edotu = uu * d_e[k][0] + vv * d_e[k][1];
        Fi[k] = (1.0 - 0.5 * omega) * d_w[k] * (3.0 * (edotF - uF) + 9.0 * edotu * edotF);
    }
    for (int k = 0; k < 9; k++) {
        double fcoll = fv[k] - omega * (fv[k] - feq[k]) + Fi[k];
        int nx2 = (x + d_e[k][0] + nx) % nx;
        int ny2 = (y + d_e[k][1] + ny) % ny;
        fnew[9 * (ny2 * nx + nx2) + k] = fcoll;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host driver
// ─────────────────────────────────────────────────────────────────────────────

static void write_vtk(const char* dir, int step, int nx, int ny,
                      const double* phi, const double* p, const double* ux,
                      const double* uy, const double* rho, int mode) {
    char fname[512];
    snprintf(fname, sizeof(fname), "%s/flow%d.vtk", dir, step);
    FILE* fp = fopen(fname, "w");
    if (!fp) { fprintf(stderr, "[pf] cannot open %s\n", fname); return; }
    fprintf(fp, "# vtk DataFile Version 3.0\nphase-field LBM\nASCII\n");
    fprintf(fp, "DATASET STRUCTURED_POINTS\nDIMENSIONS %d %d 1\n", nx, ny);
    fprintf(fp, "ORIGIN 0 0 0\nSPACING 1 1 1\nPOINT_DATA %d\n", nx * ny);
    if (mode >= 1) {
        fprintf(fp, "SCALARS phi double 1\nLOOKUP_TABLE default\n");
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++)
                fprintf(fp, "%.12g\n", phi[j * nx + i]);
    }
    fprintf(fp, "SCALARS p double 1\nLOOKUP_TABLE default\n");
    for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
            fprintf(fp, "%.12g\n", p[j * nx + i]);
    fprintf(fp, "SCALARS ux double 1\nLOOKUP_TABLE default\n");
    for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
            fprintf(fp, "%.12g\n", ux[j * nx + i]);
    fprintf(fp, "SCALARS uy double 1\nLOOKUP_TABLE default\n");
    for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
            fprintf(fp, "%.12g\n", uy[j * nx + i]);
    fprintf(fp, "SCALARS rho double 1\nLOOKUP_TABLE default\n");
    for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
            fprintf(fp, "%.12g\n", rho[j * nx + i]);
    fclose(fp);
}

int run_pf_ns_2d(const char* params_path) {
    PFParams P = load_pf_params(params_path);
    int nx = P.nx, ny = P.ny;
    long nn = (long)nx * ny;

    fprintf(stdout, "[pf] phase-field 2D  nx=%d ny=%d mode=%d W=%.2f M=%.4f "
                    "sigma=%.4f rho_g=%.3f rho_w=%.3f tau=%.3f steps=%d\n",
            nx, ny, P.mode, P.W, P.M, P.sigma, P.rho_g, P.rho_w, P.tau, P.steps);

    CUDA_CHECK(cudaMemcpyToSymbol(d_e, HOST_E, sizeof(HOST_E)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_w, HOST_W, sizeof(HOST_W)));

    double* f_d;   double* fnew_d;
    double* phi_d; double* phi_new_d;
    double* ux_d;  double* uy_d;
    CUDA_CHECK(cudaMalloc(&f_d, 9 * nn * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&fnew_d, 9 * nn * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&phi_d, nn * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&phi_new_d, nn * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ux_d, nn * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&uy_d, nn * sizeof(double)));

    double omega = 1.0 / P.tau;
    std::vector<double> phi_h(nn), f_h(9 * nn, 0.0), p_h(nn), ux_h(nn), uy_h(nn), rho_h(nn);
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            double dx = i - P.xc, dy = j - P.yc;
            double r = sqrt(dx * dx + dy * dy);
            double ph = 0.5 * (1.0 - tanh(2.0 * (r - P.R0) / P.W));
            if (P.mode == 0) ph = 1.0;   // single-phase: uniform water
            phi_h[j * nx + i] = ph;
            rho_h[j * nx + i] = P.rho_g + ph * (P.rho_w - P.rho_g);
        }
    }
    CUDA_CHECK(cudaMemcpy(phi_d, phi_h.data(), nn * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(f_d, f_h.data(), 9 * nn * sizeof(double), cudaMemcpyHostToDevice));

    char outdir[512];
    if (const char* e = getenv("LBM_FILE_DIR"); e && *e)
        snprintf(outdir, sizeof(outdir), "%s", e);
    else
        snprintf(outdir, sizeof(outdir), ".");

    dim3 threads(32, 1);
    dim3 grid((nx + 31) / 32, ny);

    write_vtk(outdir, 0, nx, ny, phi_h.data(), p_h.data(), ux_h.data(), uy_h.data(),
              rho_h.data(), P.mode);

    for (int step = 1; step <= P.steps; step++) {
        pf_ns_velocity<<<grid, threads>>>(f_d, phi_d, ux_d, uy_d, P.sigma, P.W,
                                          P.rho_g, P.rho_w, P.gx, P.gy, nx, ny, P.mode);
        if (P.mode >= 1)
            pf_ac_step<<<grid, threads>>>(phi_d, ux_d, uy_d, phi_new_d, P.M, P.W, nx, ny);
        else
            CUDA_CHECK(cudaMemcpy(phi_new_d, phi_d, nn * sizeof(double), cudaMemcpyDeviceToDevice));
        pf_ns_collide_stream<<<grid, threads>>>(f_d, fnew_d, phi_d, ux_d, uy_d,
                                                P.sigma, P.W, P.rho_g, P.rho_w,
                                                omega, nx, ny, P.mode);
        CUDA_CHECK(cudaGetLastError());
        std::swap(f_d, fnew_d);
        std::swap(phi_d, phi_new_d);

        if (P.output_every > 0 && step % P.output_every == 0) {
            CUDA_CHECK(cudaMemcpy(phi_h.data(), phi_d, nn * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(f_h.data(), f_d, 9 * nn * sizeof(double), cudaMemcpyDeviceToHost));
            for (int j = 0; j < ny; j++)
                for (int i = 0; i < nx; i++) {
                    int id = j * nx + i;
                    double s = 0.0, mx = 0.0, my = 0.0;
                    for (int k = 0; k < 9; k++) {
                        double fk = f_h[9 * id + k];
                        s += fk; mx += fk * HOST_E[k][0]; my += fk * HOST_E[k][1];
                    }
                    p_h[id] = s;
                    double rho = P.rho_g + phi_h[id] * (P.rho_w - P.rho_g);
                    ux_h[id] = mx / rho; uy_h[id] = my / rho;
                    rho_h[id] = rho;
                }
            write_vtk(outdir, step, nx, ny, phi_h.data(), p_h.data(), ux_h.data(),
                      uy_h.data(), rho_h.data(), P.mode);
            fprintf(stdout, "[pf] step %d written\n", step);
            fflush(stdout);
        }
    }

    CUDA_CHECK(cudaFree(f_d)); CUDA_CHECK(cudaFree(fnew_d));
    CUDA_CHECK(cudaFree(phi_d)); CUDA_CHECK(cudaFree(phi_new_d));
    CUDA_CHECK(cudaFree(ux_d)); CUDA_CHECK(cudaFree(uy_d));
    fprintf(stdout, "[pf] done\n");
    return 0;
}
