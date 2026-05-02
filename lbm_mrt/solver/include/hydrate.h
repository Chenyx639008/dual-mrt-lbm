// hydrate.h
// 甲烷水合物相变扩展模块头文件
// 全部内容在 #ifdef HYDRATE_ENABLE 保护下，不影响原 flow-only 编译
//
// 编译目标：
//   flow-only:   nvcc ... main.cu LBM.cu ... -o mcmp_sim
//   hydrate:     nvcc -DHYDRATE_ENABLE ... hydrate.cu hydrate_vop.cu -o mcmp_sim_hydrate
//
// 物理→格子单位映射（在 init_device_variable_hydrate() 中完成）：
//   格子长度     dx_latt = 1      ←→  dx_phys (m)
//   格子时间     dt_latt = 1      ←→  dt_phys (s)
//   温度         T_latt  = T_phys        （直接用 K，无量纲化困难）
//   扩散系数     D_latt  = D_phys * dt_phys / dx_phys²
//   热扩散率     alpha_latt = alpha_phys * dt_phys / dx_phys²
//   反应速率     k_r_latt = k_r_phys * dt_phys / dx_phys
//   压力         p_latt  = mix.pressure   （已是格子量，直接使用）
//   气体常数     R_gas = 8.314  J/(mol·K)  （物理值，T 也用 K，配套）
//
#pragma once

#ifdef HYDRATE_ENABLE

#include <cuda_runtime.h>
#include "LBM.h"   // NX, NY, mem_size_scalar, Fluid_dev, Mix_dev 等

// ============================================================
// §1  D2Q5 格子常数
// ============================================================
// 离散速度方向：0=(0,0)  1=(+x)  2=(-x)  3=(+y)  4=(-y)
constexpr int    Q5              = 5;
constexpr int    e5[Q5][2]       = {{0,0},{1,0},{-1,0},{0,1},{0,-1}};
constexpr int    opp5[Q5]        = {0, 2, 1, 4, 3};      // 反方向索引
constexpr double w5[Q5]          = {1.0/3, 1.0/6, 1.0/6, 1.0/6, 1.0/6};
constexpr double cs2_5           = 1.0/3.0;               // D2Q5 声速平方
constexpr size_t mem_size_D2Q5   = sizeof(double) * NX * NY * Q5;

// ============================================================
// §2  D2Q5 MRT 变换矩阵 M5 及其逆 Minv5
// ============================================================
// 矩向量定义：m = [ρ, jx, jy, e, pxx]
//   m0 = ρ    (质量，守恒)
//   m1 = jx   (x 动量，守恒)
//   m2 = jy   (y 动量，守恒)
//   m3 = e    (能量，非守恒)  对应行 [-4, 1, 1, 1, 1]
//   m4 = pxx  (应力，非守恒) 对应行 [ 0, 1, 1,-1,-1]
//
//   M5[α][i] = 第 α 矩基对第 i 速度分量的投影
//   速度编号:  0=(0,0)  1=(+x)  2=(-x)  3=(+y)  4=(-y)

inline constexpr double M5[Q5][Q5] = {
    { 1,  1,  1,  1,  1},   // m0 = ρ
    { 0,  1, -1,  0,  0},   // m1 = jx
    { 0,  0,  0,  1, -1},   // m2 = jy
    {-4,  1,  1,  1,  1},   // m3 = e
    { 0,  1,  1, -1, -1}    // m4 = pxx
};

// Minv5 = M5⁻¹，解析推导：
//   行 i 对应方向 i 的分布函数展开系数
//   fi = (1/5)*ρ + [jx 贡献] + [jy 贡献] + (1/20)*e + [pxx 贡献]
//
//   f0 =  (1/5)ρ                                -  (1/5)e
//   f1 =  (1/5)ρ + (1/2)jx             + (1/20)e + (1/4)pxx
//   f2 =  (1/5)ρ - (1/2)jx             + (1/20)e + (1/4)pxx
//   f3 =  (1/5)ρ           + (1/2)jy   + (1/20)e - (1/4)pxx
//   f4 =  (1/5)ρ           - (1/2)jy   + (1/20)e - (1/4)pxx
//
//   列序：[ρ,  jx,   jy,    e,    pxx]

inline constexpr double Minv5[Q5][Q5] = {
    { 1.0/5,    0.0,    0.0, -1.0/5,    0.0  },  // f0
    { 1.0/5,  1.0/2,    0.0,  1.0/20,  1.0/4 },  // f1
    { 1.0/5, -1.0/2,    0.0,  1.0/20,  1.0/4 },  // f2
    { 1.0/5,    0.0,  1.0/2,  1.0/20, -1.0/4 },  // f3
    { 1.0/5,    0.0, -1.0/2,  1.0/20, -1.0/4 }   // f4
};

// ============================================================
// §3  设备端新结构体（GPU 指针）
// ============================================================

// 温度场：D2Q5 DDF 热 LBM
struct Therm_dev {
    double* h_in  = nullptr;   // 分布函数（碰撞后 → 对流前）  [NX*NY*Q5]
    double* h_out = nullptr;   // 分布函数（碰撞输出）          [NX*NY*Q5]
    double* T     = nullptr;   // 宏观温度 T（单位 K）          [NX*NY]
};

// 溶解甲烷浓度场：D2Q5 MRT-CST LBM
struct Conc_dev {
    double* g_in  = nullptr;   // 分布函数                      [NX*NY*Q5]
    double* g_out = nullptr;   // 分布函数（碰撞输出）          [NX*NY*Q5]
    double* Cm    = nullptr;   // 溶解浓度（格子单位）          [NX*NY]
};

// VOP 固相动态更新
struct VOP_dev {
    double* Vh            = nullptr;   // 水合物体积分数 [0,1]  [NX*NY]
    double* diss_rate     = nullptr;   // 每节点分解速率        [NX*NY]
    double* S_latent      = nullptr;   // 潜热源项（暂存）      [NX*NY]
    int*    new_fluid_flag = nullptr;  // 本步翻转节点标记      [NX*NY]
    double* pore_origin   = nullptr;   // 诊断场：1=由水合物分解释放的孔隙，0=原生孔隙 [NX*NY]
};

// ============================================================
// §4  设备常量（由 init_device_variable_hydrate 上传）
// ============================================================
// 遵循与 LBM.h 相同的模式：
//   HYDRATE_DEFINE_GLOBALS 只在 hydrate.cu 的顶部 define，其他 .cu 不 define

#ifdef HYDRATE_DEFINE_GLOBALS

// ----- D2Q5 格子常数（设备侧）-----
__constant__ int    e5_gpu[Q5][2];
__constant__ int    opp5_gpu[Q5];
__constant__ double w5_gpu[Q5];

// ----- 热场参数 -----
__constant__ double d_T0_inlet;        // 入口温度（K）
__constant__ double d_T0_init;         // 初始均匀温度（K）
__constant__ double d_alpha_fluid;     // 热扩散率 α_fluid = λ/(ρcp) [格子]
__constant__ double d_alpha_hydrate;   // 热扩散率 α_hydrate [格子]
__constant__ double d_alpha_solid;     // 热扩散率 α_solid [格子]（quartz 用）

// ----- 浓度场参数 -----
__constant__ double d_D_latt;          // 甲烷扩散系数 [格子]
__constant__ double d_Henry_KH;        // Henry 常数（无量纲）
__constant__ double d_Cm_init;         // 初始浓度 [格子]

// ----- Kim-Bishnoi 反应动力学 -----
__constant__ double d_k0_latt;         // 前指数因子 [格子]
__constant__ double d_Ea_over_R;       // Ea/R [K]（直接存 Ea/R 减少核函数除法）
__constant__ double d_e1_peq;          // 平衡压: pe=exp(e1+e2/T)
__constant__ double d_e2_peq;          // 平衡压经验常数 [K]

// ----- VOP 参数 -----
__constant__ double d_Vm_latt;         // 摩尔体积 [格子³/mol]
__constant__ double d_latent_H_latt;   // 潜热 ΔH [格子·K]（ΔH/(ρcp) 量纲）
__constant__ double d_rhocp_fluid_latt;// ρcp_fluid [格子]

#else

extern __device__ __constant__ int    e5_gpu[Q5][2];
extern __device__ __constant__ int    opp5_gpu[Q5];
extern __device__ __constant__ double w5_gpu[Q5];

extern __device__ __constant__ double d_T0_inlet;
extern __device__ __constant__ double d_T0_init;
extern __device__ __constant__ double d_alpha_fluid;
extern __device__ __constant__ double d_alpha_hydrate;
extern __device__ __constant__ double d_alpha_solid;

extern __device__ __constant__ double d_D_latt;
extern __device__ __constant__ double d_Henry_KH;
extern __device__ __constant__ double d_Cm_init;

extern __device__ __constant__ double d_k0_latt;
extern __device__ __constant__ double d_Ea_over_R;
extern __device__ __constant__ double d_e1_peq;
extern __device__ __constant__ double d_e2_peq;

extern __device__ __constant__ double d_Vm_latt;
extern __device__ __constant__ double d_latent_H_latt;
extern __device__ __constant__ double d_rhocp_fluid_latt;

#endif  // HYDRATE_DEFINE_GLOBALS

// ============================================================
// §5  分配 / 释放
// ============================================================
void alloc_therm(Therm_dev& TH);
void free_therm(Therm_dev& TH);
void alloc_conc(Conc_dev& CN);
void free_conc(Conc_dev& CN);
void alloc_vop(VOP_dev& VP);
void free_vop(VOP_dev& VP);

// ============================================================
// §6  设备常量上传（在 main() 初始化阶段调用）
// ============================================================
// 在 RuntimeParams 解析完成并调用 push_device_constants() 之后调用此函数
struct RuntimeParams;   // forward 声明，避免循环依赖
void init_device_variable_hydrate(const RuntimeParams& P);

// ============================================================
// §7  热场（Phase 1）函数声明
// ============================================================
// thermal_init_mode: 0=uniform(T0_init), 1=linear gradient(T0_init→T0_inlet along bc_side axis)
void init_thermal_field(Therm_dev& TH, const int* pointsflag,
                        int thermal_init_mode = 0,
                        double T0_inlet = 285.0,
                        int    bc_side  = 0);

// 每时间步热场演化（碰撞→流→边界）
// h_T_inlet: 宿主侧标量，来自 RuntimeParams::T0_inlet
void step_thermal(Therm_dev& TH,
                  const double* ux_mix, const double* uy_mix,
                  const double* S_latent,
                  const int* pointsflag,
                  double h_T_inlet,
                  int    bc_side = 0);

// ============================================================
// §8  浓度场（Phase 2）函数声明
// ============================================================
void init_conc_field(Conc_dev& CN, const int* pointsflag);

// 每时间步浓度场演化；Kang 反应边界同时写 VP.diss_rate
void step_conc(Conc_dev& CN, const Therm_dev& TH,
               const double* ux_mix, const double* uy_mix,
               const double* rho_A, const double* rho_B,
               VOP_dev& VP, const int* pointsflag);

// ============================================================
// §9  VOP 固相更新（Phase 3）函数声明
// ============================================================
void init_vop(VOP_dev& VP, const int* pointsflag, double Vh_init);

// 更新 Vh，检测翻转事件，返回本步翻转节点数
int step_vop(VOP_dev& VP,
             Therm_dev& TH, Conc_dev& CN,
             Fluid_dev& A, Fluid_dev& B, Mix_dev& MX);

// ============================================================
// §10  潜热源项（Phase 4）函数声明
// ============================================================
void compute_latent_heat_source(VOP_dev& VP,
                                const Conc_dev& CN,
                                const Therm_dev& TH,
                                const int* pointsflag);

// ============================================================
// §11  全耦合入口（Phase 5）
// ============================================================
// 在 run_stage 的 evolution_all() 调用之后调用此函数
// 返回本步翻转节点数（>0 时调用方需重建 ghost 层）
struct RuntimeParams;   // forward 声明
int step_hydrate_physics(VOP_dev& VP, Therm_dev& TH, Conc_dev& CN,
                         Fluid_dev& A, Fluid_dev& B, Mix_dev& MX,
                         const RuntimeParams& P, int current_step);

#endif  // HYDRATE_ENABLE
