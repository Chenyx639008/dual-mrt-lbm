// hydrate_vop.cu
// VOP 固相动态更新模块（Phase 3）
// 实现：Vh 体积分数更新、节点翻转、new-fluid 节点重初始化、ghost 层重建
//
// 编译时必须有 -DHYDRATE_ENABLE

#ifdef HYDRATE_ENABLE

#include "../include/LBM.h"
#include "../include/hydrate.h"
#include "../include/sim_utils.h"
#include "../include/unified_cuda_error_check.cuh"
#include <cstdio>

// ============================================================
// 与 LBM.cu / main.cu 共享的全局线程块
extern dim3 grid, threads;

// LBM.cu 中定义的 D2Q9 设备常量（通过 -rdc=true 跨文件可见）
extern __device__ __constant__ int    e_gpu[9][2];
extern __device__ __constant__ double w_gpu[9];

// LBM.cu 内部核函数（geometry rebuild，通过 rdc=true 跨文件可见）
extern __global__ void mark_boundary(int* pointsflag);
extern __global__ void mark_ghost(int* pointsflag);
extern __global__ void init_wall_mat_from_flag(int* pointsflag);

// D2Q9 feq（本文件本地 inline 定义，与 LBM.cu 中完全相同）
// 避免跨文件引用 __forceinline__ device 函数的链接问题
__device__ __forceinline__ double feq_vop(int k, double rho, const double u[2])
{
    const double eu  = e_gpu[k][0] * u[0] + e_gpu[k][1] * u[1];
    const double uv  = u[0] * u[0] + u[1] * u[1];
    const double cs2 = 1.0 / 3.0;   // cs2_gpu 值（与 LBM.cu 一致）
    return w_gpu[k] * rho * (1.0 + eu / cs2 + eu * eu / (2.0 * cs2 * cs2) - uv / (2.0 * cs2));
}

// ============================================================
// § VOP-1  更新水合物体积分数 Vh
// ------------------------------------------------------------
// Vh[idx] -= d_Vm_latt * diss_rate[idx]
//   diss_rate 由 kernel_boundary_conc_reaction 写入（格子单位）
// Vh <= 0 时标记翻转；只处理 flag == -3 的水合物内部节点
// ============================================================
__global__ void kernel_update_vop(
          double* __restrict__ Vh,
          int*    __restrict__ new_fluid_flag,
    const double* __restrict__ diss_rate,
    const int*    __restrict__ pointsflag)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;

    const size_t s  = (size_t)NX * y + x;
    if (pointsflag[s] != -3) return;

    double v = Vh[s] - d_Vm_latt * diss_rate[s];
    if (v <= 0.0) {
        Vh[s]             = 0.0;
        new_fluid_flag[s] = 1;
    } else {
        Vh[s] = v;
    }
}

// ============================================================
// § VOP-2  应用节点翻转：-3 → 1，清除 d_wall_mat
// ------------------------------------------------------------
// atomicAdd 统计翻转数写入 n_conv[0]
// ============================================================
__global__ void kernel_apply_vop_conversion(
          int*    __restrict__ pointsflag,
    const int*    __restrict__ new_fluid_flag,
          int*    __restrict__ n_conv)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;

    const size_t s = (size_t)NX * y + x;
    if (!new_fluid_flag[s]) return;

    pointsflag[s] = 1;
    d_wall_mat[s] = 0;
    atomicAdd(n_conv, 1);
}

// ============================================================
// § VOP-3  重初始化新流体节点的分布函数
// ------------------------------------------------------------
// rho_A/rho_B = D2Q9 邻居流体节点均值
// fin/fout    = feq(rho_mean, u=0)
// h_in[k]     = w5[k] * T_mean
// g_in[k]     = w5[k] * Cm_mean
// ============================================================
__global__ void kernel_reinit_new_fluid(
    const int*    __restrict__ new_fluid_flag,
    const int*    __restrict__ pointsflag,
          double* __restrict__ fin_A,
          double* __restrict__ fout_A,
          double* __restrict__ rho_A,
          double* __restrict__ fin_B,
          double* __restrict__ fout_B,
          double* __restrict__ rho_B,
          double* __restrict__ h_in,
          double* __restrict__ T,
          double* __restrict__ g_in,
          double* __restrict__ Cm)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;

    const size_t s = (size_t)NX * y + x;
    if (!new_fluid_flag[s]) return;

    // 采样 D2Q9 邻居（仅流体 flag==1 或边界 flag==0）
    double sumrA = 0.0, sumrB = 0.0, sumT = 0.0, sumCm = 0.0;
    int    cnt   = 0;

    for (int k = 1; k < Q; ++k) {
        int xn = x + e_gpu[k][0];
        int yn = y + e_gpu[k][1];
        if (xn < 0 || xn >= NX || yn < 0 || yn >= NY) continue;
        const size_t sn = (size_t)NX * yn + xn;
        const int    fn = pointsflag[sn];
        if (fn == 1 || fn == 0) {
            sumrA += rho_A[sn];
            sumrB += rho_B[sn];
            sumT  += T[sn];
            sumCm += Cm[sn];
            ++cnt;
        }
    }

    const double rA_new = (cnt > 0) ? sumrA / cnt : 0.05;
    const double rB_new = (cnt > 0) ? sumrB / cnt : 0.05;
    const double T_new  = (cnt > 0) ? sumT  / cnt : d_T0_init;
    const double Cm_new = (cnt > 0) ? sumCm / cnt : d_Cm_init;

    rho_A[s] = rA_new;
    rho_B[s] = rB_new;
    T[s]     = T_new;
    Cm[s]    = Cm_new;

    // D2Q9 分布函数：feq(rho, u=0)
    const double u0[2] = {0.0, 0.0};
    for (int k = 0; k < Q; ++k) {
        double fA = feq_vop(k, rA_new, u0);
        double fB = feq_vop(k, rB_new, u0);
        fin_A [(size_t)NX * NY * k + s] = fout_A[(size_t)NX * NY * k + s] = fA;
        fin_B [(size_t)NX * NY * k + s] = fout_B[(size_t)NX * NY * k + s] = fB;
    }

    // D2Q5 分布函数：h_eq = w5[k]*T, g_eq = w5[k]*Cm (u=0)
    for (int k = 0; k < Q5; ++k) {
        h_in[(size_t)NX * ((size_t)NY * k + y) + x] = w5_gpu[k] * T_new;
        g_in[(size_t)NX * ((size_t)NY * k + y) + x] = w5_gpu[k] * Cm_new;
    }
}

// ============================================================
// § VOP-4  清除 new_fluid_flag（每步末尾重置）
// ============================================================
__global__ void kernel_clear_new_fluid_flag(int* __restrict__ new_fluid_flag)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;
    new_fluid_flag[(size_t)NX * y + x] = 0;
}

// ============================================================
// § VOP-5  宿主函数：完整 VOP 步
// ------------------------------------------------------------
// 顺序：Vh 更新 → 翻转 → 新节点重初 → ghost 重建 → 清 flag
// 返回：本步翻转节点数（>0 时主循环应做额外处理如润湿性上传）
// ============================================================
int step_vop(VOP_dev& VP,
             Therm_dev& TH, Conc_dev& CN,
             Fluid_dev& A, Fluid_dev& B, Mix_dev& MX)
{
    // 1) 更新 Vh，标记翻转节点
    kernel_update_vop<<<grid, threads>>>(
        VP.Vh, VP.new_fluid_flag, VP.diss_rate, MX.pointsflag);
    CUDA_CHECK(cudaGetLastError());

    // 2) 统计 + 执行翻转
    int* d_n_conv = nullptr;
    CUDA_CHECK(cudaMalloc(&d_n_conv, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_n_conv, 0, sizeof(int)));

    kernel_apply_vop_conversion<<<grid, threads>>>(
        MX.pointsflag, VP.new_fluid_flag, d_n_conv);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int n_conv = 0;
    CUDA_CHECK(cudaMemcpy(&n_conv, d_n_conv, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_n_conv);

    if (n_conv > 0) {
        // 3) 新流体节点重初始化
        kernel_reinit_new_fluid<<<grid, threads>>>(
            VP.new_fluid_flag, MX.pointsflag,
            A.fin, A.fout, A.rho,
            B.fin, B.fout, B.rho,
            TH.h_in, TH.T,
            CN.g_in, CN.Cm);
        CUDA_CHECK(cudaGetLastError());

        // 4) 全域 ghost 层重建：壁面材质 → 边界 → ghost
        init_wall_mat_from_flag<<<grid, threads>>>(MX.pointsflag);
        CUDA_CHECK(cudaGetLastError());
        mark_boundary<<<grid, threads>>>(MX.pointsflag);
        CUDA_CHECK(cudaGetLastError());
        mark_ghost<<<grid, threads>>>(MX.pointsflag);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        printf("[VOP] 本步翻转 %d 个水合物节点 → 流体\n", n_conv);
    }

    // 5) 清 flag，为下步准备
    kernel_clear_new_fluid_flag<<<grid, threads>>>(VP.new_fluid_flag);
    CUDA_CHECK(cudaGetLastError());

    return n_conv;
}

#endif  // HYDRATE_ENABLE
