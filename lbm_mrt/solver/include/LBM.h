// LBM.h
// 该文件包含了 CUDA LBM 模型的核心定义和函数声明
#pragma once //作用是防止重复定义，例如结构体、函数原型、constexpr 变量等。
/* 只保留最必要的头文件，不要把 <iostream> <fstream> 等重头文件塞进来 */

// ── 编译期守卫：Huang SCMP 与水合物物理互斥 ──
#if defined(HUANG_256_BUILD) && defined(HYDRATE_ENABLE)
#error "HUANG_256_BUILD and HYDRATE_ENABLE are mutually exclusive. Use separate builds."
#endif

#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <filesystem>
#include <string>

#define BLOCK_SIZE 32

/* ------------ ① 网格尺寸与时间控制 ------------------- */
#ifdef HUANG_256_BUILD
#ifndef HUANG_NX
#define HUANG_NX 256
#endif
#ifndef HUANG_NY
#define HUANG_NY 256
#endif
#ifndef HUANG_NSTEPS
#define HUANG_NSTEPS 200000
#endif
#ifndef HUANG_NOUTPUT
#define HUANG_NOUTPUT 5000
#endif
constexpr unsigned int SCALE   = 1;
constexpr unsigned int NX      = HUANG_NX;
constexpr unsigned int NY      = HUANG_NY;
constexpr unsigned int NSTEPS  = HUANG_NSTEPS;
constexpr unsigned int NOUTPUT = HUANG_NOUTPUT;
#else
constexpr unsigned int SCALE   = 1;
constexpr unsigned int NX      = 339* SCALE;
constexpr unsigned int NY      = 212* SCALE;
constexpr unsigned int NSTEPS  = 5000000;
constexpr unsigned int NOUTPUT = 50000;
#endif
/* ------------ ② D2Q9 几何与权重 ----------------------- */
constexpr int Q = 9;
inline constexpr int   e[Q][2] = {{0,0},{1,0},{0,1},{-1,0},{0,-1},{1,1},{-1,1},{-1,-1},{1,-1}};
inline constexpr int   opp[Q]   = { 0, 3, 4, 1, 2, 7, 8, 5, 6 };
inline constexpr double w [Q]  = { 4.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/36,1.0/36,1.0/36,1.0/36 };
inline constexpr double w_F[Q]  = { 4.0/3, 1.0/3, 1.0/3, 1.0/3, 1.0/3,1.0/12,1.0/12,1.0/12,1.0/12 };
/* ------------ ② MRT 参数 ----------------------- */
inline constexpr double M[Q][Q] = {
	{ 1, 1, 1, 1, 1, 1, 1, 1, 1},
	{-4,-1,-1,-1,-1, 2, 2, 2, 2},
	{ 4,-2,-2,-2,-2, 1, 1, 1, 1},
	{ 0, 1, 0,-1, 0, 1,-1,-1, 1},
	{ 0,-2, 0, 2, 0, 1,-1,-1, 1},
	{ 0, 0, 1, 0,-1, 1, 1,-1,-1},
	{ 0, 0,-2, 0, 2, 1, 1,-1,-1},
	{ 0, 1,-1, 1,-1, 0, 0, 0, 0},
	{ 0, 0, 0, 0, 0, 1,-1, 1,-1}
};
inline constexpr double Minv[9][9] = {
	{4 / 36.0,-4 / 36.0, 4 / 36.0, 0 / 36.0, 0 / 36.0, 0 / 36.0, 0 / 36.0, 0 / 36.0, 0 / 36.0},
	{4 / 36.0,-1 / 36.0,-2 / 36.0, 6 / 36.0,-6 / 36.0, 0 / 36.0, 0 / 36.0, 9 / 36.0, 0 / 36.0},
	{4 / 36.0,-1 / 36.0,-2 / 36.0, 0 / 36.0, 0 / 36.0, 6 / 36.0,-6 / 36.0,-9 / 36.0, 0 / 36.0},
	{4 / 36.0,-1 / 36.0,-2 / 36.0,-6 / 36.0, 6 / 36.0, 0 / 36.0, 0 / 36.0, 9 / 36.0, 0 / 36.0},
	{4 / 36.0,-1 / 36.0,-2 / 36.0, 0 / 36.0, 0 / 36.0,-6 / 36.0, 6 / 36.0,-9 / 36.0, 0 / 36.0},
	{4 / 36.0, 2 / 36.0, 1 / 36.0, 6 / 36.0, 3 / 36.0, 6 / 36.0, 3 / 36.0, 0 / 36.0, 9 / 36.0},
	{4 / 36.0, 2 / 36.0, 1 / 36.0,-6 / 36.0,-3 / 36.0, 6 / 36.0, 3 / 36.0, 0 / 36.0,-9 / 36.0},
	{4 / 36.0, 2 / 36.0, 1 / 36.0,-6 / 36.0,-3 / 36.0,-6 / 36.0,-3 / 36.0, 0 / 36.0, 9 / 36.0},
	{4 / 36.0, 2 / 36.0, 1 / 36.0, 6 / 36.0, 3 / 36.0,-6 / 36.0,-3 / 36.0, 0 / 36.0,-9 / 36.0}
};
/* MRT 松弛率 */
inline constexpr double tau_e = 1 / 0.8, tau_t = 1 / 0.8, tau_q = 1 / 1.1, tau_p = 1;
inline constexpr double tau_p_a = 1.0, tau_p_b = 1.0;
inline constexpr int MAX_OBS = 1000;

namespace phys
{
    /* Lattice unit physics */
    inline constexpr double deltax = 1.0;
    inline constexpr double deltat = 1.0;
    inline constexpr double c      = deltax / deltat;
    inline constexpr double cs2    = c*c / 3.0;
    /* Interaction strength (kept for legacy __constant__ upload) */
    inline constexpr double GAA = -1.0;
    inline constexpr double GBB = 0.0;
}



// —— Device-side __constant__ and device-pointer declarations ——
// Single compilation unit (LBM.cu) #define-s LBM_DEFINE_GLOBALS to own the storage;
// all other .cu files see `extern` declarations via the #else branch.

// ── Group A: Wettability & wall maps (MCMP + future SCMP two-phase) ──
#ifdef LBM_DEFINE_GLOBALS
__constant__ double GAw_by_mat_gpu[256];
__constant__ double GBw_by_mat_gpu[256];
__device__ unsigned char* d_wall_mat = nullptr;
__device__ double*        d_GAw_map  = nullptr;
__device__ double*        d_GBw_map  = nullptr;

// ── Group B: MRT relaxation matrix (common to MCMP + SCMP) ──
__constant__ double A_a_gpu[9];
__constant__ double A_b_gpu[9];

// ── Group C: Body force & drive (common) ──
__constant__ double d_Gx, d_Gy;
__constant__ double d_drive_scale;

// ── Group D: MCMP two-phase parameters ──
__constant__ double d_water_satur;
__constant__ unsigned long long d_water_seed;
__constant__ int    d_drive_mode;
__constant__ double d_rhoA_ini_h, d_rhoA_ini_l;
__constant__ double d_rhoB_ini_h, d_rhoB_ini_l;
__constant__ double d_rhoA_ini_h_1, d_rhoA_ini_l_0;
__constant__ double d_rhoB_ini_h_0, d_rhoB_ini_l_1;
__constant__ double d_tau_p_a, d_tau_p_b;
__constant__ double d_kappa;
__constant__ double d_GAB;
__constant__ double d_GBA;
__constant__ double d_sigmaA;

// ── Group E: Huang & Wu (2016) SCMP ──
__constant__ int    d_pp_mode;
__constant__ double d_k1_huang, d_k2_huang, d_kd_huang, d_alpha_meq;
__constant__ double d_cs_a, d_cs_b, d_cs_R, d_cs_T, d_cs_G;
__constant__ double d_huang_R0, d_huang_xc, d_huang_yc, d_huang_W;
__constant__ double d_huang_rho_g, d_huang_rho_l;
__constant__ double d_tau_huang, d_Lambda_huang;
__constant__ int    d_huang_init_mode;
__constant__ double d_G_ads_scmp;
__constant__ double d_theta_contact_deg;
__constant__ double d_huang_psi_l_ref, d_huang_psi_g_ref;
__constant__ double d_huang_u_max, d_huang_psi_cut;
__constant__ double d_huang_tanh_factor, d_huang_rho_max_init;

#else  // ── extern declarations for non-defining translation units ──

// Group A: Wettability & wall maps
extern __device__ __constant__ double GAw_by_mat_gpu[256];
extern __device__ __constant__ double GBw_by_mat_gpu[256];
extern __device__ unsigned char* d_wall_mat;
extern __device__ double*        d_GAw_map;
extern __device__ double*        d_GBw_map;

// Group B: MRT relaxation matrix
extern __device__ __constant__ double A_a_gpu[9];
extern __device__ __constant__ double A_b_gpu[9];

// Group C: Body force & drive
extern __device__ __constant__ double d_Gx, d_Gy;
extern __device__ __constant__ double d_drive_scale;

// Group D: MCMP two-phase parameters
extern __device__ __constant__ double d_water_satur;
extern __device__ __constant__ unsigned long long d_water_seed;
extern __device__ __constant__ int    d_drive_mode;
extern __device__ __constant__ double d_rhoA_ini_h, d_rhoA_ini_l;
extern __device__ __constant__ double d_rhoB_ini_h, d_rhoB_ini_l;
extern __device__ __constant__ double d_rhoA_ini_h_1, d_rhoA_ini_l_0;
extern __device__ __constant__ double d_rhoB_ini_h_0, d_rhoB_ini_l_1;
extern __device__ __constant__ double d_tau_p_a, d_tau_p_b;
extern __device__ __constant__ double d_kappa;
extern __device__ __constant__ double d_GAB;
extern __device__ __constant__ double d_GBA;
extern __device__ __constant__ double d_sigmaA;

// Group E: Huang & Wu (2016) SCMP
extern __device__ __constant__ int    d_pp_mode;
extern __device__ __constant__ double d_k1_huang, d_k2_huang, d_alpha_meq;
extern __device__ __constant__ double d_cs_a, d_cs_b, d_cs_R, d_cs_T, d_cs_G;
extern __device__ __constant__ double d_huang_R0, d_huang_xc, d_huang_yc, d_huang_W;
extern __device__ __constant__ double d_huang_rho_g, d_huang_rho_l;
extern __device__ __constant__ double d_tau_huang, d_Lambda_huang;
extern __device__ __constant__ int    d_huang_init_mode;
extern __device__ __constant__ double d_G_ads_scmp;
extern __device__ __constant__ double d_theta_contact_deg;
extern __device__ __constant__ double d_huang_psi_l_ref, d_huang_psi_g_ref;
extern __device__ __constant__ double d_huang_u_max, d_huang_psi_cut;
extern __device__ __constant__ double d_huang_tanh_factor, d_huang_rho_max_init;
#endif


// ── Device getter declarations ──
// Bodies are in LBM.cu with __forceinline__.

// Group C: Body force & drive
__device__ double get_Gx();
__device__ double get_Gy();

// Group D: MCMP two-phase
__device__ double get_water_satur();
__device__ unsigned long long get_water_seed();
__device__ int    get_drive_mode();
__device__ double rhoA_hi();
__device__ double rhoA_lo();
__device__ double rhoB_hi();
__device__ double rhoB_lo();
__device__ double rhoA_hi_1();
__device__ double rhoA_lo_0();
__device__ double rhoB_hi_0();
__device__ double rhoB_lo_1();
__device__ double tauA();
__device__ double tauB();
__device__ double get_kappa();
__device__ double get_GAB();
__device__ double get_GBA();
__device__ double get_sigmaA();

// Group E: Huang & Wu (2016) SCMP
__device__ int    get_pp_mode();
__device__ double get_k1_huang();
__device__ double get_k2_huang();
__device__ double get_alpha_meq();
__device__ double get_cs_a();
__device__ double get_cs_b();
__device__ double get_cs_R();
__device__ double get_cs_T();
__device__ double get_cs_G();
__device__ double get_huang_R0();
__device__ double get_huang_xc();
__device__ double get_huang_yc();
__device__ double get_huang_W();
__device__ double get_huang_rho_g();
__device__ double get_huang_rho_l();
__device__ int    get_huang_init_mode();
__device__ double get_G_ads_scmp();
__device__ double get_theta_contact_deg();
__device__ double get_huang_psi_l_ref();
__device__ double get_huang_psi_g_ref();
__device__ double get_huang_u_max();
__device__ double get_huang_psi_cut();
__device__ double get_huang_tanh_factor();
__device__ double get_huang_rho_max_init();



//定义内存大小,mem_size_scalar物理量，mem_size_distfun分布函数
constexpr size_t mem_size_scalar = sizeof(double) * NX * NY;
constexpr size_t mem_size_flag = sizeof(int) * NX * NY;
constexpr size_t mem_size_distfun = sizeof(double) * NX * NY * Q;
/* ========== ⑩  多孔介质圆柱列表  ================================= */
struct Obstacle{
    int cx, cy;   // 圆心（格点坐标）
    double r2;       // 半径平方
    double r2_hydrate; // 水合物半径平方
    int flag;     // -2 固体, -3 水合物, 以后可扩展
    unsigned char mat_id;    // 1=石英, 2=水合物, 自行约定

};

/* Host 侧存储（可变长） */
struct Porous_host{
    std::vector<Obstacle> obs;   // 全部障碍物
    int obst_num = 0;
};


//  混合物主机端  //
struct Mix_host {
    std::vector<double> rho, ux, uy, pressure;
    std::vector<int> pointsflag;
    Mix_host()
    : rho      (NX*NY),
      ux       (NX*NY),
      uy       (NX*NY),
      pressure (NX*NY),
      pointsflag (NX*NY) {} //节点类型：-3:solid, -2:hydrate -1:solid(ghost), 0:boundary, 1:fluid
};

//  混合物设备端  //
struct Mix_dev {
    double *rho     = nullptr;
    double *ux      = nullptr;
    double *uy      = nullptr;
    double *pressure= nullptr;
    int *pointsflag= nullptr;
};

//  单相流体设备端  //
struct Fluid_host {
    std::vector<double> rho, ux, uy, psi, pressure, Fx_mol, Fy_mol, Fx_ads, Fy_ads;
    Fluid_host()
    : rho      (NX*NY),
      ux       (NX*NY),
      uy       (NX*NY),
      psi      (NX*NY),
      pressure (NX*NY),
      Fx_mol       (NX*NY),
      Fy_mol       (NX*NY),
      Fx_ads       (NX*NY),
      Fy_ads       (NX*NY){}
};

//设备端场变量结构体//
struct Fluid_dev {
    double *rho       = nullptr;
    double *ux        = nullptr;
    double *uy        = nullptr;
    double *psi       = nullptr;
    double *pressure  = nullptr;
    double *p_xx      = nullptr;  // pressure tensor normal x (paper Eq. 56)
    double *p_yy      = nullptr;  // pressure tensor normal y
    double *p_xy      = nullptr;  // pressure tensor shear
    double *Fx_mol    = nullptr;
    double *Fy_mol    = nullptr;
    double *Fx_ads    = nullptr;
    double *Fy_ads    = nullptr;
    double *fin       = nullptr;
    double *fout      = nullptr;
    double *min       = nullptr;
    double *mout      = nullptr;
    double *S         = nullptr;
    double *C         = nullptr; // B 相暂时可不使用
};

// 实现在内存中分配和释放
void alloc_fluid(Fluid_dev&);
void free_fluid (Fluid_dev&);
void alloc_mix  (Mix_dev&);
void free_mix   (Mix_dev&);

//声明的函数类型
void copy_and_check(const Fluid_dev& d,Fluid_host& h,const char* tag_prefix);
void copy_back_mix(const Mix_dev& d, Mix_host& h, bool do_nan_check = true);
//void clear_obstacles(); // 若需多次重建几何
// === 多材料润湿性
void upload_wettability_table_host(double thetaA_quartz,double thetaA_hydrate,
                                   double GBw_quartz  = 0.0,double GBw_hydrate = 0.0);
void upload_wettability_table_raw_host(const double* GAw_host256,
                                       const double* GBw_host256);
void alloc_wall_and_wettability_maps_host();  // 分配 d_wall_mat / d_GAw_map / d_GBw_map
void free_wall_and_wettability_maps_host();   // 释放以上映射内存

void build_circle_array(Porous_host& porous, int morph, double r_obs, double coat_thick, double r_mid, double l_gap );
void upload_obstacles(const Porous_host& h);
void read_tecplot_to_flag(const std::string& filename, std::vector<int>& host_flag);
void init_geometry(int* pointsflag);
void init_all(int* pointsflag, double* rho_A,double* fin_A, double* fout_A, double*min_A , double*mout_A, double* rho_B,double* fin_B, double* fout_B, double* min_B ,double*mout_B);
inline void init_all(const Mix_dev& mix, const Fluid_dev& A,const Fluid_dev& B){
    init_all(mix.pointsflag, A.rho,A.fin,A.fout,A.min,A.mout,B.rho,B.fin,B.fout,B.min,B.mout);
}

void init_device_variable();
void evolution_all(
    double* rho_A, double* ux_A, double* uy_A,
    double* psi_A, double* pressure_A,
    double* Fx_mol_A, double* Fy_mol_A, double* Fx_ads_A, double* Fy_ads_A,
    double* fin_A, double* fout_A, double* min_A, double* mout_A, double* S_A, double* C_A,

    double* rho_B, double* ux_B, double* uy_B,
    double* psi_B, double* pressure_B,
    double* Fx_mol_B, double* Fy_mol_B, double* Fx_ads_B, double* Fy_ads_B,
    double* fin_B, double* fout_B, double* min_B, double* mout_B, double* S_B, double* C_B,

	double* rho_host, double* pressure_host,double* ux_host, double* uy_host, int* pointsflag ) ;

inline void evolution_all(const Fluid_dev& A,const Fluid_dev& B, const Mix_dev& mix){
    evolution_all(
        /* A */ A.rho,A.ux,A.uy,A.psi,A.pressure,A.Fx_mol,A.Fy_mol,A.Fx_ads,A.Fy_ads,
                  A.fin,A.fout,A.min,A.mout,A.S,A.C,
        /* B */ B.rho,B.ux,B.uy,B.psi,B.pressure,B.Fx_mol,B.Fy_mol,B.Fx_ads,B.Fy_ads,
                  B.fin,B.fout,B.min,B.mout,B.S,B.C,
        /* mix */ mix.rho,mix.pressure,mix.ux,mix.uy,mix.pointsflag);
}

void display_results(const Fluid_host& AH,const Fluid_host& BH, const Mix_host& MIX, int ini_opt);
void outputdat(int step,const std::string& folder,const std::string& name ,const std::string& title,
               const Fluid_host& AH,const Fluid_host& BH, const Mix_host& MIX);
void outputvtk(int step,const std::string& folder,const std::string& name ,const std::string& title,
            const Fluid_host& AH,const Fluid_host& BH, const Mix_host& MIX);
// 大小端翻转工具
void SwapEnd(double& var);
void SwapEnd_int(int& var) ;
__global__ void dbg_consts_once();

// ===== 水合物扩展模块 =====
// 放在 LBM.h 末尾，确保 Fluid_dev、Mix_dev 等已经定义后再包含
#ifdef HYDRATE_ENABLE
#include "hydrate.h"

// hydrate 场 VTK 追加写（向已打开的 .vtk 文件末尾写 T/Cm/Vh/diss_rate）
// 由 sim_utils.cu 中 write_stage_output_hydrate 调用
void outputvtk_append_hydrate(const std::string& vtk_path,
                               const std::vector<double>& T,
                               const std::vector<double>& Cm,
                               const std::vector<double>& Vh,
                               const std::vector<double>& diss_rate,
                               const std::vector<double>& pore_origin);
#endif

// ── Huang & Wu (2016) SCMP (gated by HUANG_256_BUILD at compile time) ──
#ifdef HUANG_256_BUILD
void evolution_scmp(
    double* rho,   double* ux,   double* uy,
    double* psi,   double* pressure,
    double* Fx,    double* Fy,
    double* Fx_ads, double* Fy_ads,
    double* fin,   double* fout,
    double* min_m, double* mout_m,
    double* S,     double* C,
    double* p_xx,  double* p_yy, double* p_xy,
    double  theta_contact_deg,
    int*    pointsflag);

void init_all_scmp(
    double* rho,   double* fin,   double* fout,
    double* min_m, double* mout_m,
    int*    pointsflag);

void outputvtk_scmp(int step, const std::string& folder,
                     const std::string& prefix,
                     const std::string& title,
                     const std::vector<double>& rho,
                     const std::vector<double>& ux,
                     const std::vector<double>& uy,
                     const std::vector<double>& pressure,
                     const std::vector<double>& p_xx,
                     const std::vector<double>& p_yy,
                     const std::vector<double>& Fx,
                     const std::vector<double>& Fy,
                     const std::vector<double>& psi,
                     const std::vector<int>& pointsflag);
#endif
