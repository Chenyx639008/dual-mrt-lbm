// hydrate.cu
// 水合物相变扩展模块 —— 温度场（Phase 1）+ 浓度场/VOP 桩（Phase 2-5 后续填充）
// 编译时必须有 -DHYDRATE_ENABLE，否则整个文件为空
//
// 内存布局约定（与 LBM.cu 完全一致）：
//   标量    idx = NX*y + x
//   D2Q5    idx5(k,x,y) = NX*(NY*k + y) + x
//   D2Q9    由 LBM.cu 负责，本文件不直接操作

#ifdef HYDRATE_ENABLE

// ==== HYDRATE_DEFINE_GLOBALS 必须先于 hydrate.h 的包含 ====
// 让 hydrate.h 中的 #ifdef HYDRATE_DEFINE_GLOBALS 块产生定义而非 extern
#define HYDRATE_DEFINE_GLOBALS

#include "../include/LBM.h"             // NX, NY, mem_size_scalar, Mix_dev, Fluid_dev, d_wall_mat …
#include "../include/hydrate.h"
#include "../include/sim_utils.h"       // RuntimeParams
#include "../include/unified_cuda_error_check.cuh"
#include <cstdio>
#include <cmath>

// ============================================================
// 内部宏与全局辅助
// ============================================================

// 与 LBM.cu 共享的全局线程块（定义在 main.cu）
extern dim3 grid, threads;

// D2Q5 标量/分布函数索引
__device__ __forceinline__ size_t idx_scalar(int x, int y)
{
    return (size_t)NX * y + x;
}
__device__ __forceinline__ size_t idx5(int k, int x, int y)
{
    return (size_t)NX * ((size_t)NY * k + y) + x;
}

// D2Q5 平衡分布：线性展开（适用于温度/浓度的对流扩散方程）
//   phi_eq[k] = w5_gpu[k] * phi * (1 + (e5_gpu[k]·u) / cs2_5)
__device__ __forceinline__ double phi_eq5(int k, double phi,
                                          double ux, double uy)
{
    const double eu = e5_gpu[k][0] * ux + e5_gpu[k][1] * uy;
    return w5_gpu[k] * phi * (1.0 + eu / cs2_5);
}

// ============================================================
// §1  设备常量数据（HYDRATE_DEFINE_GLOBALS 已在上方 define）
// ============================================================
// 具体 __constant__ 定义已由 hydrate.h 中的 #ifdef HYDRATE_DEFINE_GLOBALS 块产生。
// 此处只需实现上传函数。

// ============================================================
// §2  设备常量上传
// ============================================================
void init_device_variable_hydrate(const RuntimeParams& P)
{
    // 物理→格子单位换算
    const double dx = P.dx_phys;
    const double dt = P.dt_phys;

    // 热扩散率（格子单位）= alpha_phys * dt / dx²
    double alpha_fluid    = (P.lambda_fluid    / P.rhocp_fluid)    * dt / (dx * dx);
    double alpha_hydrate  = (P.lambda_hydrate  / P.rhocp_hydrate)  * dt / (dx * dx);
    double alpha_solid    = (P.lambda_solid    / P.rhocp_solid)    * dt / (dx * dx);

    // 扩散系数（格子单位）
    double D_latt         = P.D_mol_water * dt / (dx * dx);

    // Kim-Bishnoi 动力学
    //   k_r_phys 单位: mol/(m²·s·Pa)
    //   k_r_latt = k_r_phys * dt / dx   （使 k_r·Δt/Δx 成为无量纲比较浓度梯度用）
    //   此处存前指数因子，指数在核函数内实时计算
    double k0_latt        = P.k0_rxn * dt / dx;
    // Ea/R 直接用物理值，因为 T 也用 K
    double Ea_over_R      = P.Ea_rxn / 8.314;

    // VOP 摩尔体积（格子单位：m³/mol → dx³/mol）
    double Vm_latt        = P.Vm_hydrate / (dx * dx * dx);

    // 潜热源项：ΔH_latt = ΔH_phys * dt / (rhocp_fluid * dx³)
    //   进入热场方程的形式是 S_latent = diss_rate * ΔH / (rhocp·V_cell)
    //   diss_rate [mol/(m²·s)] → diss_rate_latt [mol/(格子²·格子时间)]
    //   此处存换算后的 ΔH/(rhocp·dx) 比例因子
    double rhocp_fluid_latt = P.rhocp_fluid * dx * dx * dx / dt;  // [格子能量单位]
    double latent_H_latt    = P.latent_heat / (P.rhocp_fluid * dx);  // [K·格子]

    // D2Q5 格子常数上传（设备 __constant__ 数组）
    CUDA_CHECK(cudaMemcpyToSymbol(e5_gpu,   e5,   sizeof(e5)));
    CUDA_CHECK(cudaMemcpyToSymbol(opp5_gpu, opp5, sizeof(opp5)));
    CUDA_CHECK(cudaMemcpyToSymbol(w5_gpu,   w5,   sizeof(w5)));

    CUDA_CHECK(cudaMemcpyToSymbol(d_T0_inlet,       &P.T0_inlet,    sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_T0_init,         &P.T0_init,    sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_alpha_fluid,     &alpha_fluid,   sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_alpha_hydrate,   &alpha_hydrate, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_alpha_solid,     &alpha_solid,   sizeof(double)));

    CUDA_CHECK(cudaMemcpyToSymbol(d_D_latt,          &D_latt,        sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_Henry_KH,        &P.Henry_KH,   sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_Cm_init,         &P.Cm_init,    sizeof(double)));

    CUDA_CHECK(cudaMemcpyToSymbol(d_k0_latt,         &k0_latt,       sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_Ea_over_R,       &Ea_over_R,     sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_e1_peq,          &P.e1_peq,     sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_e2_peq,          &P.e2_peq,     sizeof(double)));

    CUDA_CHECK(cudaMemcpyToSymbol(d_Vm_latt,         &Vm_latt,       sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_latent_H_latt,   &latent_H_latt, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_rhocp_fluid_latt,&rhocp_fluid_latt,sizeof(double)));

    printf("[hydrate] 设备常量上传完成\n");
    printf("  alpha_fluid=%.4e  alpha_hydrate=%.4e  D_latt=%.4e\n",
           alpha_fluid, alpha_hydrate, D_latt);
    printf("  k0_latt=%.4e  Ea/R=%.1f K  Vm_latt=%.4e\n",
           k0_latt, Ea_over_R, Vm_latt);
}

// ============================================================
// §3  分配 / 释放
// ============================================================
void alloc_therm(Therm_dev& TH)
{
    CUDA_CHECK(cudaMalloc(&TH.h_in,  mem_size_D2Q5));
    CUDA_CHECK(cudaMalloc(&TH.h_out, mem_size_D2Q5));
    CUDA_CHECK(cudaMalloc(&TH.T,     mem_size_scalar));
}
void free_therm(Therm_dev& TH)
{
    cudaFree(TH.h_in);  cudaFree(TH.h_out);  cudaFree(TH.T);
    TH.h_in = TH.h_out = TH.T = nullptr;
}

void alloc_conc(Conc_dev& CN)
{
    CUDA_CHECK(cudaMalloc(&CN.g_in,  mem_size_D2Q5));
    CUDA_CHECK(cudaMalloc(&CN.g_out, mem_size_D2Q5));
    CUDA_CHECK(cudaMalloc(&CN.Cm,    mem_size_scalar));
}
void free_conc(Conc_dev& CN)
{
    cudaFree(CN.g_in);  cudaFree(CN.g_out);  cudaFree(CN.Cm);
    CN.g_in = CN.g_out = CN.Cm = nullptr;
}

void alloc_vop(VOP_dev& VP)
{
    CUDA_CHECK(cudaMalloc(&VP.Vh,              mem_size_scalar));
    CUDA_CHECK(cudaMalloc(&VP.diss_rate,       mem_size_scalar));
    CUDA_CHECK(cudaMalloc(&VP.S_latent,        mem_size_scalar));
    CUDA_CHECK(cudaMalloc(&VP.new_fluid_flag,  mem_size_flag));
}
void free_vop(VOP_dev& VP)
{
    cudaFree(VP.Vh);  cudaFree(VP.diss_rate);
    cudaFree(VP.S_latent);  cudaFree(VP.new_fluid_flag);
    VP.Vh = VP.diss_rate = VP.S_latent = nullptr;
    VP.new_fluid_flag = nullptr;
}

// ============================================================
// §4  温度场初始化核函数
// ============================================================

// 读取材料类型，返回对应的热弛豫率 ω_T
// mat=0 → 流体; mat=1 → quartz; mat=2 → hydrate
__device__ __forceinline__ double get_omega_T(unsigned char mat)
{
    double alpha;
    if      (mat == 2) alpha = d_alpha_hydrate;
    else if (mat == 1) alpha = d_alpha_solid;
    else               alpha = d_alpha_fluid;
    // τ = 0.5 + α/cs2_5  →  ω = 1/τ
    return 1.0 / (0.5 + alpha / cs2_5);
}

// 将 h_in 和 T 初始化为均匀温度 T0_init
// 流体节点：h_in[k] = h_eq(T0, u=0) = w5[k] * T0
// 固/水合物节点：h_in[k] = h_eq(T0, u=0)（保持温度一致，后续边界会覆盖）
// ghost/boundary 节点：同上
__global__ void kernel_init_thermal(double* h_in, double* T,
                                     const int* pointsflag)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;

    const size_t s = idx_scalar(x, y);
    const double T0 = d_T0_init;

    T[s] = T0;
    for (int k = 0; k < Q5; ++k)
        h_in[idx5(k, x, y)] = w5_gpu[k] * T0;   // u=0 时 h_eq = w5[k]*T
}

// ============================================================
// §5  宏观温度更新
// ============================================================
__global__ void kernel_update_T(const double* h_in, double* T,
                                  const int* pointsflag)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;

    const size_t s = idx_scalar(x, y);
    // 固体/ghost 节点温度由共轭条件隐式确定，但为输出美观仍更新
    double sum = 0.0;
    for (int k = 0; k < Q5; ++k)
        sum += h_in[idx5(k, x, y)];
    T[s] = sum;
}

// ============================================================
// §6  D2Q5 MRT 碰撞核函数（温度场）
// ============================================================
// MRT 碰撞步骤：
//   1) h_in  → 矩空间 m = M5 · h_in
//   2) 计算平衡矩 m_eq
//   3) 弛豫：m_out = m - Λ·(m - m_eq) + S_latent 源项
//   4) m_out → h_out = Minv5 · m_out
//
// 弛豫矩阵 Λ = diag(1, ωT, ωT, 1, 1)
//   只有动量分量（m1, m2 / jx, jy）控制扩散；ωT 按材料空间可变
//   m0=ρ（守恒），m3/m4 取快速弛豫 ω=1（不影响扩散精度，简化实现）
//
// 平衡矩（参考 Yang 2024 D3Q7 退化到 2D）：
//   m_eq = [T, T*ux, T*uy, (3/4)*T, 0]
//
// 潜热源项 S_latent 进入 m0（能量守恒）分量

__global__ void kernel_collide_thermal(
    const double* __restrict__ h_in,
          double* __restrict__ h_out,
    const double* __restrict__ T,
    const double* __restrict__ ux_mix,
    const double* __restrict__ uy_mix,
    const double* __restrict__ S_latent,       // [NX*NY]，nullptr 时忽略
    const int*    __restrict__ pointsflag)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;

    const size_t s  = idx_scalar(x, y);
    const int    fl = pointsflag[s];

    // 仅在流体和鬼节点处做碰撞（固体中不需要，但鬼节点参与共轭热传递）
    // 固体内部节点（flag == -2 且 mat==1/石英）也做碰撞以维持温度场连续
    // 纯固体（-2）和水合物内部（-3）均参与碰撞
    if (fl == -1) {
        // ghost 节点：只复制，不碰撞（由 boundary 核函数处理）
        for (int k = 0; k < Q5; ++k)
            h_out[idx5(k, x, y)] = h_in[idx5(k, x, y)];
        return;
    }

    // 从 d_wall_mat 获取材料编号 → 弛豫率
    unsigned char mat = d_wall_mat[s];
    double omegaT = get_omega_T(mat);

    // 读取分布函数
    double h[Q5];
    for (int k = 0; k < Q5; ++k)
        h[k] = h_in[idx5(k, x, y)];

    double Ti  = T[s];
    double uxi = ux_mix[s];
    double uyi = uy_mix[s];

    // ----- 变换到矩空间 m = M5 · h -----
    // m[0] = h[0]+h[1]+h[2]+h[3]+h[4]
    // m[1] = h[1]-h[2]
    // m[2] = h[3]-h[4]
    // m[3] = -4h[0]+h[1]+h[2]+h[3]+h[4]
    // m[4] = h[1]+h[2]-h[3]-h[4]
    double m0 = h[0] + h[1] + h[2] + h[3] + h[4];
    double m1 = h[1] - h[2];
    double m2 = h[3] - h[4];
    double m3 = -4.0*h[0] + h[1] + h[2] + h[3] + h[4];
    double m4 = h[1] + h[2] - h[3] - h[4];

    // ----- 平衡矩 -----
    double meq0 = Ti;
    double meq1 = Ti * uxi;
    double meq2 = Ti * uyi;
    double meq3 = 0.75 * Ti;   // (3/4)*T，来自 Yang 2024 D3Q7 能量模式退化
    double meq4 = 0.0;

    // ----- 潜热源项（仅作用于 m0，守恒量）-----
    double src0 = (S_latent != nullptr) ? S_latent[s] : 0.0;

    // ----- MRT 弛豫 m_out = m - Λ·(m - m_eq) + dt*src -----
    // Λ = diag(1, omegaT, omegaT, 1, 1)
    // 守恒分量（m0）：弛豫率=1，加源项
    double mo0 = m0 - 1.0 * (m0 - meq0) + src0;
    double mo1 = m1 - omegaT * (m1 - meq1);
    double mo2 = m2 - omegaT * (m2 - meq2);
    double mo3 = m3 - 1.0   * (m3 - meq3);
    double mo4 = m4 - 1.0   * (m4 - meq4);

    // ----- 逆变换 h_out = Minv5 · m_out -----
    // 由 Minv5 解析式：
    //   h[0] = (1/5)*mo0                    - (1/5)*mo3
    //   h[1] = (1/5)*mo0 + (1/2)*mo1        + (1/20)*mo3 + (1/4)*mo4
    //   h[2] = (1/5)*mo0 - (1/2)*mo1        + (1/20)*mo3 + (1/4)*mo4
    //   h[3] = (1/5)*mo0           + (1/2)*mo2 + (1/20)*mo3 - (1/4)*mo4
    //   h[4] = (1/5)*mo0           - (1/2)*mo2 + (1/20)*mo3 - (1/4)*mo4
    const double inv5  = 1.0/5.0;
    const double inv20 = 1.0/20.0;
    const double half  = 0.5;
    const double qtr   = 0.25;

    h_out[idx5(0, x, y)] = inv5*mo0                    - inv5*mo3;
    h_out[idx5(1, x, y)] = inv5*mo0 + half*mo1         + inv20*mo3 + qtr*mo4;
    h_out[idx5(2, x, y)] = inv5*mo0 - half*mo1         + inv20*mo3 + qtr*mo4;
    h_out[idx5(3, x, y)] = inv5*mo0            + half*mo2 + inv20*mo3 - qtr*mo4;
    h_out[idx5(4, x, y)] = inv5*mo0            - half*mo2 + inv20*mo3 - qtr*mo4;
}

// ============================================================
// §7  D2Q5 流步核函数（温度场）
// ============================================================
// 标准拉格朗日流：fin[k](x,y) ← fout[k](x - e5[k], y - e5[k])
// 边界/ghost 节点的 h_in 由 kernel_boundary_thermal 覆盖，此处只做流体节点

__global__ void kernel_stream_thermal(
          double* __restrict__ h_in,
    const double* __restrict__ h_out,
    const int*    __restrict__ pointsflag)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;

    const int fl = pointsflag[idx_scalar(x, y)];
    // 流体节点（flag==1）和固体内部节点（flag<=-2）都接收流
    // ghost 节点（flag==-1）和边界节点（flag==0）也接收，但会被 BC 覆盖
    for (int k = 0; k < Q5; ++k) {
        // 上游坐标（wrap 处理周期边界）
        int xp = (x - e5_gpu[k][0] + NX) % NX;
        int yp = (y - e5_gpu[k][1] + NY) % NY;
        h_in[idx5(k, x, y)] = h_out[idx5(k, xp, yp)];
    }
}

// ============================================================
// §8  边界条件核函数（温度场）
// ============================================================
// 策略：
//   y==0 行（ghost，入口侧）：固定 T=T_inlet，h_in[k] = h_eq(T_inlet, u=0)
//   y==NY-1 行（ghost，出口侧）：全展开（copy 来自 y=NY-2）
//   固体/水合物鬼节点（flag==-1 且 d_wall_mat>0）：全反弹
//     h_in[k] = h_out[opp5[k]]   （保持热绝缘/共轭由弛豫代理完成）
//
// 注意：共轭热传递的温度跳变通过 collide 中空间可变 ωT 自然实现，
//       不需要额外的显式界面处理（Zhang 2019 / Karani & Huber 2015 方法）

__global__ void kernel_boundary_thermal(
          double* __restrict__ h_in,
    const double* __restrict__ h_out,
    const int*    __restrict__ pointsflag,
    double T_inlet)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;

    const size_t s  = idx_scalar(x, y);
    const int    fl = pointsflag[s];

    // ---- 入口：y==0 ghost 层（固定温度 Dirichlet）----
    if (y == 0 && fl == -1) {
        for (int k = 0; k < Q5; ++k)
            h_in[idx5(k, x, y)] = w5_gpu[k] * T_inlet;  // h_eq(T_inlet, u=0)
        return;
    }

    // ---- 出口：y==NY-1 ghost 层（全展开：复制内层）----
    if (y == NY - 1 && fl == -1) {
        for (int k = 0; k < Q5; ++k)
            h_in[idx5(k, x, y)] = h_in[idx5(k, x, NY - 2)];
        return;
    }

    // ---- 固体/水合物鬼节点（flag==-1，mat>0）：全反弹 ----
    // 这些节点夹在固体和流体之间，通过反弹实现零法向通量
    // （共轭热传递在内部通过弛豫自然实现，见 collide_thermal 中 get_omega_T）
    if (fl == -1) {
        unsigned char mat = d_wall_mat[s];
        if (mat > 0) {
            // 全反弹
            for (int k = 0; k < Q5; ++k)
                h_in[idx5(k, x, y)] = h_out[idx5(opp5_gpu[k], x, y)];
        }
        return;
    }

    // ---- 边界节点（flag==0）：流体侧，由 stream 填充，无需额外处理 ----
}

// ============================================================
// §9  宿主函数：每步热场演化
// ============================================================
void init_thermal_field(Therm_dev& TH, const int* pointsflag)
{
    kernel_init_thermal<<<grid, threads>>>(TH.h_in, TH.T, pointsflag);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// h_T_inlet: 宿主侧标量，由调用方从 RuntimeParams 传入，避免直接读 __constant__
void step_thermal(Therm_dev& TH,
                  const double* ux_mix, const double* uy_mix,
                  const double* S_latent,
                  const int* pointsflag,
                  double h_T_inlet)
{
    // 1) 宏观温度更新（碰撞前）
    kernel_update_T<<<grid, threads>>>(TH.h_in, TH.T, pointsflag);
    CUDA_CHECK(cudaGetLastError());

    // 2) MRT 碰撞
    kernel_collide_thermal<<<grid, threads>>>(
        TH.h_in, TH.h_out, TH.T,
        ux_mix, uy_mix, S_latent, pointsflag);
    CUDA_CHECK(cudaGetLastError());

    // 3) 流步
    kernel_stream_thermal<<<grid, threads>>>(TH.h_in, TH.h_out, pointsflag);
    CUDA_CHECK(cudaGetLastError());

    // 4) 边界条件（传入宿主侧 T_inlet）
    kernel_boundary_thermal<<<grid, threads>>>(
        TH.h_in, TH.h_out, pointsflag, h_T_inlet);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// §10  潜热源项（Phase 4 实现）
// ============================================================
// S_latent[idx] = -ΔH_latt * diss_rate[idx]
//   diss_rate 已由 kernel_boundary_conc_reaction 在水合物 ghost 节点写入
//   ΔH_latt = latent_heat / (rhocp_fluid * dx)  [K·格子]  （在 init_device_variable_hydrate 中计算）
// 仅在水合物面 ghost 节点（mat==2，fl==-1）非零；流体节点接收该源项

__global__ void kernel_compute_latent_heat(
          double* __restrict__ S_latent,
    const double* __restrict__ diss_rate,
    const int*    __restrict__ pointsflag)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;

    const size_t s = idx_scalar(x, y);
    // 仅流体节点接收潜热：遍历邻居 ghost 节点累加
    if (pointsflag[s] != 1) {
        S_latent[s] = 0.0;
        return;
    }

    double src = 0.0;
    for (int k = 1; k < Q5; ++k) {
        int xn = x + e5_gpu[k][0];
        int yn = y + e5_gpu[k][1];
        if (xn < 0 || xn >= NX || yn < 0 || yn >= NY) continue;
        const size_t sn = idx_scalar(xn, yn);
        if (pointsflag[sn] == -1 && d_wall_mat[sn] == 2) {
            // 相邻水合物 ghost 节点的分解速率贡献潜热
            src -= d_latent_H_latt * diss_rate[sn];
        }
    }
    S_latent[s] = src;
}

void compute_latent_heat_source(VOP_dev& VP,
                                const Conc_dev& CN,
                                const Therm_dev& TH,
                                const int* pointsflag)
{
    kernel_compute_latent_heat<<<grid, threads>>>(
        VP.S_latent, VP.diss_rate, pointsflag);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// §11  浓度场（Phase 2 实现）
// ============================================================

// ------------------------------------------------------------
// §11.1  宏观浓度更新
// ------------------------------------------------------------
__global__ void kernel_update_Cm(const double* g_in, double* Cm,
                                   const int* pointsflag)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;

    const size_t s = idx_scalar(x, y);
    double sum = 0.0;
    for (int k = 0; k < Q5; ++k)
        sum += g_in[idx5(k, x, y)];
    Cm[s] = sum;
}

// ------------------------------------------------------------
// §11.2  初始化浓度场（均匀 Cm_init）
// ------------------------------------------------------------
__global__ void kernel_init_conc(double* g_in, double* Cm,
                                   const int* pointsflag)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;

    const size_t s  = idx_scalar(x, y);
    const double C0 = d_Cm_init;

    Cm[s] = C0;
    for (int k = 0; k < Q5; ++k)
        g_in[idx5(k, x, y)] = w5_gpu[k] * C0;  // g_eq(Cm, u=0)
}

// ------------------------------------------------------------
// §11.3  D2Q5 MRT 碰撞核函数（浓度场）
//
// 与热场碰撞相同结构，只是：
//   ωD = 1/(0.5 + D_latt/cs2_5)  （均匀扩散，流体节点）
//   m_eq = [Cm, Cm·ux, Cm·uy, 0.75*Cm, 0]
//   CST 源项（气-水界面）：在守恒量 m0 加入松弛到 Henry 平衡
//     气-水界面判据：rhoB/(rhoA+rhoB) > theta_iface（0.3）
//     源项形式（对流-扩散方程 BGK 型）：
//       S_cst = -omega_cst * (Cm - Ceq_latt)   （调整后直接加到 m0）
//       Ceq_latt = KH * Cg = KH * exp(e1 + e2/T)  [格子单位]
//     其中 omega_cst 可取 1（强制平衡）或小值（弱松弛）；
//     此处取 omega_cst = 1（在 CST 界面节点一步直接平衡，Yang 2024 §S3）
// ------------------------------------------------------------
__global__ void kernel_collide_conc(
    const double* __restrict__ g_in,
          double* __restrict__ g_out,
    const double* __restrict__ Cm,
    const double* __restrict__ T,
    const double* __restrict__ ux_mix,
    const double* __restrict__ uy_mix,
    const double* __restrict__ rho_A,
    const double* __restrict__ rho_B,
    const int*    __restrict__ pointsflag)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;

    const size_t s  = idx_scalar(x, y);
    const int    fl = pointsflag[s];

    // ghost 节点：仅复制（交由边界核函数处理）
    if (fl == -1) {
        for (int k = 0; k < Q5; ++k)
            g_out[idx5(k, x, y)] = g_in[idx5(k, x, y)];
        return;
    }

    // 弛豫率（仅流体区域有意义；固/水合物内部可不参与，但复制确保数组有效）
    const double omegaD = 1.0 / (0.5 + d_D_latt / cs2_5);

    double g[Q5];
    for (int k = 0; k < Q5; ++k)
        g[k] = g_in[idx5(k, x, y)];

    const double Ci  = Cm[s];
    const double uxi = ux_mix[s];
    const double uyi = uy_mix[s];

    // 变换到矩空间（与热场完全相同的 M5）
    double m0 = g[0] + g[1] + g[2] + g[3] + g[4];
    double m1 = g[1] - g[2];
    double m2 = g[3] - g[4];
    double m3 = -4.0*g[0] + g[1] + g[2] + g[3] + g[4];
    double m4 = g[1] + g[2] - g[3] - g[4];

    // 平衡矩
    double meq0 = Ci;
    double meq1 = Ci * uxi;
    double meq2 = Ci * uyi;
    double meq3 = 0.75 * Ci;
    double meq4 = 0.0;

    // CST 源项（Henry 平衡，仅在气-水界面流体节点）
    //   Ceq_latt = KH * exp(e1_peq + e2_peq / T)
    //   若 rhoB/(rhoA+rhoB) > 0.3，视为界面节点，强制松弛到 Ceq
    double src_cst = 0.0;
    if (fl == 1) {
        const double rA = rho_A[s];
        const double rB = rho_B[s];
        const double rT = rA + rB;
        if (rT > 1e-12 && rB / rT > 0.3) {
            const double Ti    = T[s];
            const double Ceq   = d_Henry_KH * exp(d_e1_peq + d_e2_peq / Ti);
            // omega_cst = 1 → 一步全松弛
            src_cst = Ceq - Ci;  // 直接修正 m0（守恒量）
        }
    }

    // MRT 弛豫
    double mo0 = m0 - 1.0    * (m0 - meq0) + src_cst;
    double mo1 = m1 - omegaD * (m1 - meq1);
    double mo2 = m2 - omegaD * (m2 - meq2);
    double mo3 = m3 - 1.0    * (m3 - meq3);
    double mo4 = m4 - 1.0    * (m4 - meq4);

    // 逆变换（Minv5，与热场相同）
    const double inv5  = 1.0/5.0;
    const double inv20 = 1.0/20.0;
    const double half  = 0.5;
    const double qtr   = 0.25;

    g_out[idx5(0, x, y)] = inv5*mo0                    - inv5*mo3;
    g_out[idx5(1, x, y)] = inv5*mo0 + half*mo1         + inv20*mo3 + qtr*mo4;
    g_out[idx5(2, x, y)] = inv5*mo0 - half*mo1         + inv20*mo3 + qtr*mo4;
    g_out[idx5(3, x, y)] = inv5*mo0            + half*mo2 + inv20*mo3 - qtr*mo4;
    g_out[idx5(4, x, y)] = inv5*mo0            - half*mo2 + inv20*mo3 - qtr*mo4;
}

// ------------------------------------------------------------
// §11.4  D2Q5 流步核函数（浓度场）
// ------------------------------------------------------------
__global__ void kernel_stream_conc(
          double* __restrict__ g_in,
    const double* __restrict__ g_out,
    const int*    __restrict__ pointsflag)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;

    for (int k = 0; k < Q5; ++k) {
        int xp = (x - e5_gpu[k][0] + NX) % NX;
        int yp = (y - e5_gpu[k][1] + NY) % NY;
        g_in[idx5(k, x, y)] = g_out[idx5(k, xp, yp)];
    }
}

// ------------------------------------------------------------
// §11.5  Kang 方案反应边界 + 出入口边界（浓度场）
//
// 处理三类 ghost/boundary：
//   y==0  (入口 ghost, fl==-1)：Cm = Cm_init  → g_eq
//   y==NY-1 (出口 ghost, fl==-1)：全展开（copy from y=NY-2）
//   水合物面 ghost (fl==-1, mat==2)：Kang 方案 Cm_bc
//     Cm_bc = (D_latt * Cm_nbr + k_r * Csat * dx_latt)
//             / (D_latt + k_r * dx_latt)
//     其中 Csat = exp(e1+e2/T)（无量纲"溶解平衡" Henry 侧）
//     dx_latt = 1
//   同时将反应速率写入 diss_rate（格子单位 mol/(格子²·格子时间)）
//     diss_rate = k_r * (1 - Cm_nbr/Csat)   （当 Cm<Csat 时才分解）
// ------------------------------------------------------------
__global__ void kernel_boundary_conc_reaction(
          double* __restrict__ g_in,
    const double* __restrict__ g_out,
    const double* __restrict__ Cm,
    const double* __restrict__ T,
          double* __restrict__ diss_rate,
    const int*    __restrict__ pointsflag)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;

    const size_t s  = idx_scalar(x, y);
    const int    fl = pointsflag[s];

    if (fl != -1) return;  // 只处理 ghost 节点

    // ---- 入口：y==0，Dirichlet Cm_init ----
    if (y == 0) {
        for (int k = 0; k < Q5; ++k)
            g_in[idx5(k, x, y)] = w5_gpu[k] * d_Cm_init;
        return;
    }

    // ---- 出口：y==NY-1，全展开 ----
    if (y == NY - 1) {
        for (int k = 0; k < Q5; ++k)
            g_in[idx5(k, x, y)] = g_in[idx5(k, x, NY - 2)];
        return;
    }

    // ---- 水合物/固体 ghost（内部边界）----
    const unsigned char mat = d_wall_mat[s];
    if (mat == 0) return;  // 普通固体（石英）：绝质边界（全反弹）

    if (mat == 2) {
        // 水合物面 → Kang 方案
        // 找最近流体邻居（沿 +y 方向优先，然后其他方向）
        // 简化实现：取邻居中第一个流体节点的 Cm 和 T
        double Cm_nbr = d_Cm_init;
        double T_nbr  = d_T0_init;
        bool   found  = false;
        for (int k = 1; k < Q5 && !found; ++k) {  // k=0 静止跳过
            int xn = x + e5_gpu[k][0];
            int yn = y + e5_gpu[k][1];
            if (xn < 0 || xn >= NX || yn < 0 || yn >= NY) continue;
            size_t sn = idx_scalar(xn, yn);
            if (pointsflag[sn] == 1) {
                Cm_nbr = Cm[sn];
                T_nbr  = T[sn];
                found  = true;
            }
        }

        const double Ti    = T_nbr;
        // 平衡溶解浓度（无单位校正，直接用格子温度 K）
        const double Csat  = __expf((float)(d_e1_peq + d_e2_peq / Ti));
        // Kim-Bishnoi 速率（格子单位）
        const double k_r   = d_k0_latt * exp(-d_Ea_over_R / Ti)
                             * fmax(0.0, 1.0 - Cm_nbr / (Csat + 1e-30));
        // Kang BC 浓度
        // Cm_bc = (D·Cm_nbr + k_r·Csat) / (D + k_r)   (dx_latt=1)
        const double D     = d_D_latt;
        const double Cm_bc = (D * Cm_nbr + k_r * Csat) / (D + k_r + 1e-30);

        // 写分解速率到 diss_rate
        diss_rate[s] = k_r * fmax(0.0, 1.0 - Cm_nbr / (Csat + 1e-30));

        // 将 g_in 设为均衡分布（对应 Cm_bc, u=0）
        for (int k = 0; k < Q5; ++k)
            g_in[idx5(k, x, y)] = w5_gpu[k] * Cm_bc;
    } else {
        // 普通固体（石英等）：全反弹，不参与反应，diss_rate=0
        for (int k = 0; k < Q5; ++k)
            g_in[idx5(k, x, y)] = g_out[idx5(opp5_gpu[k], x, y)];
        diss_rate[s] = 0.0;
    }
}

// ------------------------------------------------------------
// §11.6  宿主函数：每步浓度场演化
// ------------------------------------------------------------
void init_conc_field(Conc_dev& CN, const int* pointsflag)
{
    kernel_init_conc<<<grid, threads>>>(CN.g_in, CN.Cm, pointsflag);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void step_conc(Conc_dev& CN, const Therm_dev& TH,
               const double* ux_mix, const double* uy_mix,
               const double* rho_A, const double* rho_B,
               VOP_dev& VP, const int* pointsflag)
{
    // 1) 宏观浓度更新
    kernel_update_Cm<<<grid, threads>>>(CN.g_in, CN.Cm, pointsflag);
    CUDA_CHECK(cudaGetLastError());

    // 2) MRT 碰撞（含 CST Henry 源项）
    kernel_collide_conc<<<grid, threads>>>(
        CN.g_in, CN.g_out, CN.Cm, TH.T,
        ux_mix, uy_mix, rho_A, rho_B, pointsflag);
    CUDA_CHECK(cudaGetLastError());

    // 3) 流步
    kernel_stream_conc<<<grid, threads>>>(CN.g_in, CN.g_out, pointsflag);
    CUDA_CHECK(cudaGetLastError());

    // 4) Kang 反应边界 + 入/出口边界（同时计算 diss_rate）
    kernel_boundary_conc_reaction<<<grid, threads>>>(
        CN.g_in, CN.g_out, CN.Cm, TH.T,
        VP.diss_rate, pointsflag);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// §12  VOP（Phase 3 桩）
// ============================================================
void init_vop(VOP_dev& VP, const int* pointsflag)
{
    // Phase 3 实现：水合物节点 Vh=1，其他=0
    CUDA_CHECK(cudaMemset(VP.Vh,           0, mem_size_scalar));
    CUDA_CHECK(cudaMemset(VP.diss_rate,    0, mem_size_scalar));
    CUDA_CHECK(cudaMemset(VP.S_latent,     0, mem_size_scalar));
    CUDA_CHECK(cudaMemset(VP.new_fluid_flag,0,mem_size_flag));
}

// step_vop 实现在 hydrate_vop.cu（Phase 3）

// ============================================================
// §13  全耦合入口（Phase 5 桩）
// ============================================================
int step_hydrate_physics(VOP_dev& VP, Therm_dev& TH, Conc_dev& CN,
                          Fluid_dev& A, Fluid_dev& B, Mix_dev& MX,
                          const RuntimeParams& P, int current_step)
{
    if (!P.hydrate_enable) return 0;
    if (current_step < P.hydrate_start_step) return 0;

    // Phase 5 耦合顺序（Yang 2024 Figure S2）：
    //   Flow（evolution_all 已在外部调用）→ Conc → LatentHeat → Thermal → VOP
    step_conc(CN, TH, MX.ux, MX.uy, A.rho, B.rho, VP, MX.pointsflag);
    compute_latent_heat_source(VP, CN, TH, MX.pointsflag);
    step_thermal(TH, MX.ux, MX.uy, VP.S_latent, MX.pointsflag, P.T0_inlet);
    int n_conv = step_vop(VP, TH, CN, A, B, MX);

    return n_conv;
}

#endif  // HYDRATE_ENABLE
