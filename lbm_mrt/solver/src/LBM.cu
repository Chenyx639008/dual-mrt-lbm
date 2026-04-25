//LBM.cu文件专门用于函数
#define LBM_DEFINE_GLOBALS
#include "../include/LBM.h"
#include <cmath>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/unified_cuda_error_check.cuh"
#include "../include/steady_monitor.cuh"

#define CK(call) do{ \
  cudaError_t _e=(call); \
  if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s failed @ %s:%d : %s\n", \
            #call,__FILE__,__LINE__,cudaGetErrorString(_e)); \
    exit(1); \
  } \
}while(0)



// getter（可加 __forceinline__）
__device__ __forceinline__ double get_water_satur(){ return d_water_satur; }
__device__ __forceinline__ unsigned long long get_water_seed(){ return d_water_seed; }
__device__ __forceinline__ double get_Gx(){ return d_Gx; }
__device__ __forceinline__ double get_Gy(){ return d_Gy; }
__device__ __forceinline__ int    get_drive_mode(){ return d_drive_mode; }
__device__ __forceinline__ double rhoA_hi(){ return d_rhoA_ini_h; }
__device__ __forceinline__ double rhoA_lo(){ return d_rhoA_ini_l; }
__device__ __forceinline__ double rhoB_hi(){ return d_rhoB_ini_h; }
__device__ __forceinline__ double rhoB_lo(){ return d_rhoB_ini_l; }

__device__ __forceinline__ double rhoA_hi_1(){ return d_rhoA_ini_h_1; }
__device__ __forceinline__ double rhoA_lo_0(){ return d_rhoA_ini_l_0; }
__device__ __forceinline__ double rhoB_hi_0(){ return d_rhoB_ini_h_0; }
__device__ __forceinline__ double rhoB_lo_1(){ return d_rhoB_ini_l_1; }



__device__ __forceinline__ double tauA(){ return d_tau_p_a; }
__device__ __forceinline__ double tauB(){ return d_tau_p_b; }
__device__ __forceinline__ double get_kappa(){ return d_kappa; }
__device__ __forceinline__ double get_GAB(){ return d_GAB; }
__device__ __forceinline__ double get_GBA(){ return d_GBA; }
__device__ __forceinline__ double get_sigmaA(){ return d_sigmaA; }



__constant__ int e_gpu[Q][2];//D2Q9速度方向向量（如 e[9][2]）
__constant__ int opp_gpu[Q];//D2Q9速度方向的对立方向（如 opp[9]）
__constant__ double w_gpu[Q], w_F_gpu[Q];//LBM平衡函数中的方向权重
__constant__ double deltax_gpu, deltat_gpu, c_gpu, cs2_gpu;//网格间距与时间步长,晶格传播速度,声速平方（用于 feq）,吸引力强度（常为负）,松弛时间（决定流体粘度）
__constant__ double GAA_gpu, GBB_gpu;//MRT吸引力参数
__constant__ double M_gpu[Q][Q], Minv_gpu[Q][Q];//MRT变换矩阵与逆变换矩阵
__constant__ double tau_e_gpu, tau_t_gpu, tau_p_gpu, tau_q_gpu;//MRT松弛时间
__constant__ int ini_opt_gpu, x_ini_gpu, y_ini_gpu;//初始条件选择开关以及参数
__constant__ double radius_gpu, w_ini_gpu;
__constant__ int contact_angle_dir_gpu; //0:接触角减小、1:接触角增大
__constant__ double phi_contact_pho_A_gpu, delta_pho_A_gpu;//
__constant__ double reducedT_w_ini_gpu;//初始状态下各组分的温度与密度
__constant__ double a_w_gpu, b_w_gpu, R_w_gpu, omega_w_gpu, Tc_w_gpu, T_gpu;//Peng-Robinson状态方程参数
__constant__ double PR_scalar_gpu;//状态方程参数
__constant__ double a_m_gpu, b_m_gpu, R_m_gpu, omega_m_gpu, Tc_m_gpu;
__constant__ int obst_num_gpu;
__constant__ Obstacle d_obs[MAX_OBS];
extern dim3 grid, threads;


__global__ void dbg_consts_once(){
    if (threadIdx.x==0 && blockIdx.x==0){
        // 读取所有需要由 host 下发的设备常量，避免被设备链接裁剪
        volatile double sink = 0.0;

        sink += d_water_satur;
        sink += (double)d_drive_mode;
        sink += d_Gx + d_Gy;
        sink += d_rhoA_ini_h + d_rhoA_ini_l;
        sink += d_rhoB_ini_h + d_rhoB_ini_l;
        sink += d_tau_p_a + d_tau_p_b;
        sink += d_kappa;
        sink += d_GAB;
        sink += d_GBA;
        sink += d_sigmaA;
        // 也“用”一下小数组和润湿查表
        sink += A_a_gpu[0] + A_a_gpu[7] + A_b_gpu[7];
        sink += GAw_by_mat_gpu[1] + GBw_by_mat_gpu[1];
        sink += get_GAB() + get_GBA() + get_sigmaA();

        // 防止被优化掉
        if (sink < -1e300) printf("sink=%g\n", (double)sink);
    }
}


__device__ __forceinline__ size_t findindex_scalar_gpu(int x, int y)
{
	return NX * y + x;//标量场索引,二维数组行优先展开方式
}


__device__ __forceinline__ size_t findindex_distfun_gpu(int x, int y, int d)
{
	return (NX * (NY * d + y) + x);//分布函数三维展开,每个方向 d 的所有格点（按 y,x）挨着放
}

__device__ __forceinline__ double feq_gpu(const int k, const double rho, const double u[2]) {
	double eu, uv, feq;
	eu = e_gpu[k][0] * u[0] + e_gpu[k][1] * u[1];
	uv = u[0] * u[0] + u[1] * u[1];
	feq = w_gpu[k] * rho * (1.0 + eu / cs2_gpu + eu * eu / (2 * cs2_gpu * cs2_gpu) - uv / (2 * cs2_gpu));
	return feq;
}

__device__ __forceinline__ uint64_t mix64(uint64_t x){
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33; return x;
}

// 0..1 随机数（不需库）
__device__ __forceinline__ double u01(uint64_t h) {
    // 取高 53 位生成 double
    return (h >> 11) * (1.0/9007199254740992.0);
}



__device__ double atomicMin_double(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(fmin(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}
//将 double* 转为 unsigned long long int*；使用 atomicCAS() 做原子比较交换；
// 用 __double_as_longlong 与 __longlong_as_double 实现浮点与整型之间的位级别转换；实现更新 min(val, old) 或 max(val, old) 的原子操作。//
__device__ double atomicMax_double(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}
//润湿性处理部分
double h_GAw_m = 1.0 / 456.69;
double h_GAw_c = 86.41;
// θA -> GAw 的标定（照你原公式）
static inline double G_from_theta_A(double theta_deg){
    return h_GAw_m * (theta_deg - h_GAw_c);
}

// —— 主动上传两种材料（1=石英, 2=水合物）；其余材料默认 0 —— //
// 两种材料（1=石英, 2=水合物），其余默认0
void upload_wettability_table_host(double thetaA_quartz,
                                   double thetaA_hydrate,
                                   double GBw_quartz,
                                   double GBw_hydrate) {
    double h_GAw[256] = {0}, h_GBw[256] = {0};
    h_GAw[1] = G_from_theta_A(thetaA_quartz);   h_GBw[1] = GBw_quartz;
    h_GAw[2] = G_from_theta_A(thetaA_hydrate);  h_GBw[2] = GBw_hydrate;
    cudaMemcpyToSymbol(GAw_by_mat_gpu, h_GAw, sizeof(h_GAw));
    cudaMemcpyToSymbol(GBw_by_mat_gpu, h_GBw, sizeof(h_GBw));
}

// —— 若你要一次性上传完整 256 项表 —— //
void upload_wettability_table_raw_host(const double* GAw_host256,
                                       const double* GBw_host256) {
    cudaMemcpyToSymbol(GAw_by_mat_gpu, GAw_host256, 256*sizeof(double));
    cudaMemcpyToSymbol(GBw_by_mat_gpu, GBw_host256, 256*sizeof(double));
}


// —— 分配 / 释放 ghost 局部 G 与材料图 —— //
void alloc_wall_and_wettability_maps_host() {
    size_t N = size_t(NX) * NY;
    unsigned char* d_mat = nullptr;
    double *d_gAw = nullptr, *d_gBw = nullptr;

    cudaMalloc(&d_mat, N * sizeof(unsigned char));
    cudaMalloc(&d_gAw, N * sizeof(double));
    cudaMalloc(&d_gBw, N * sizeof(double));
    cudaMemset(d_mat, 0, N * sizeof(unsigned char));
    cudaMemset(d_gAw, 0, N * sizeof(double));
    cudaMemset(d_gBw, 0, N * sizeof(double));

    cudaMemcpyToSymbol(d_wall_mat, &d_mat, sizeof(d_mat));
    cudaMemcpyToSymbol(d_GAw_map,  &d_gAw, sizeof(d_gAw));
    cudaMemcpyToSymbol(d_GBw_map,  &d_gBw, sizeof(d_gBw));
}

void free_wall_and_wettability_maps_host() {
    unsigned char* d_mat=nullptr; double *d_gAw=nullptr, *d_gBw=nullptr;
    cudaMemcpyFromSymbol(&d_mat, d_wall_mat, sizeof(d_mat));
    cudaMemcpyFromSymbol(&d_gAw, d_GAw_map,  sizeof(d_gAw));
    cudaMemcpyFromSymbol(&d_gBw, d_GBw_map,  sizeof(d_gBw));
    if (d_mat) cudaFree(d_mat);
    if (d_gAw) cudaFree(d_gAw);
    if (d_gBw) cudaFree(d_gBw);
    d_mat=nullptr; d_gAw=nullptr; d_gBw=nullptr;
    cudaMemcpyToSymbol(d_wall_mat, &d_mat, sizeof(d_mat));
    cudaMemcpyToSymbol(d_GAw_map,  &d_gAw, sizeof(d_gAw));
    cudaMemcpyToSymbol(d_GBw_map,  &d_gBw, sizeof(d_gBw));
}

// 从 Tecplot 文件读取几何信息到 pointsflag
// phase: 0=pore, 1=solid, 2=hydrate
// 输出 host_flag：1 = 流体; -2 = 固体; -3 = 水合物
void read_tecplot_to_flag(const std::string& filename,
                          std::vector<int>& host_flag)
{
    host_flag.assign(NX * NY, 1);  // 默认全域流体 1

    std::ifstream fin(filename);
    if (!fin) {
        std::cerr << "Failed to open: " << filename << std::endl;
        std::exit(1);
    }

    std::string line;
    for (int i = 0; i < 3; ++i) std::getline(fin, line); // 跳头部

    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        int x, y;
        float phase;
        char comma;
        ss >> x >> comma >> y >> comma >> phase;
        if (!ss) continue;
        if (x < 0 || x >= NX || y < 0 || y >= NY) continue;

        int idx = y * NX + x;

        if (std::abs(phase - 1.0f) < 1e-3f)      host_flag[idx] = -2; // 固体
        else if (std::abs(phase - 0.5f) < 1e-3f) host_flag[idx] = -3; // 水合物
        else                                     host_flag[idx] =  1; // 流体
    }
}

// mat_id: 1=固体, 2=水合物
__host__ void build_circle_array(Porous_host& porous,
                                 int   morph,             // 1=pore-fill, 2=coating, 3=mixed
                                 double r_obs,            // 固体半径
                                 double coat_thick,       // coating 壳厚（可为0）
                                 double r_mid,            // pore-fill 半径（可为0）
                                 double l_gap)            // 间隙
{
    porous.obs.clear();

    const double l = 2.0 * r_obs + l_gap;       // 中心到中心
    const double d = l / std::sqrt(2.0);        // 旋转方阵投影
    const double r_hyd = std::max(r_obs, r_obs + coat_thick);
    const int Nmax = static_cast<int>(
        std::ceil((std::max(NX, NY) + std::max(r_hyd, r_mid)) / d)
    );

    const bool use_coating  = (morph == 2 || morph == 3) && (coat_thick > 0.0);
    const bool use_porefill = (morph == 1 || morph == 3) && (r_mid     > 0.0);
    const double r2_mid = r_mid * r_mid;

    // ① 固体 + （可选）外包水合物壳
    for (int i=-Nmax; i<=Nmax; ++i){
        for (int j=-Nmax; j<=Nmax; ++j){
            const double cx=(i-j)*d, cy=(i+j)*d;

            // 包围盒裁剪（使用较大半径，避免漏判）
            const double Rbox = use_coating ? r_hyd : r_obs;
            if (cx + Rbox < 0.0 || cx - Rbox > NX) continue;
            if (cy + Rbox < 0.0 || cy - Rbox > NY) continue;

            Obstacle ob{};
            ob.cx = (int)std::round(cx);
            ob.cy = (int)std::round(cy);
            ob.r2 = r_obs * r_obs;                          // 实心
            ob.r2_hydrate = use_coating ? (r_hyd * r_hyd)   // 外壳半径^2
                                        : ob.r2;            // 不用壳：等于 r2，等效关闭
            ob.flag = -2;                                   // 先按“固体实体”入场
            ob.mat_id = 1;                                   // 1=solid（注意：壳体材料在 kernel 中判断）
            porous.obs.push_back(ob);
        }
    }

    // ② 孔隙填充：在相邻颗粒中点放“纯水合物”圆
    if (use_porefill){
        for (int i=-Nmax; i<=Nmax; ++i){
            for (int j=-Nmax; j<=Nmax; ++j){
                const double cx=(i-j)*d, cy=(i+j)*d;

                auto try_add = [&](double mx, double my){
                    if (mx + r_mid < 0.0 || mx - r_mid > NX) return;
                    if (my + r_mid < 0.0 || my - r_mid > NY) return;
                    Obstacle oh{};
                    oh.cx=(int)std::round(mx);
                    oh.cy=(int)std::round(my);
                    oh.r2=-1.0;                          // 纯水合物，不占固体
                    oh.r2_hydrate=r2_mid;               // 仅水合物半径
                    oh.flag=-3;                         // 直接标为水合物
                    oh.mat_id=2;                        // 材料=水合物
                    porous.obs.push_back(oh);
                };
                // 中点（i+1,j）与（i,j+1）
                try_add(cx + d, cy);
                try_add(cx, cy + d);
            }
        }
    }

    porous.obst_num = (int)porous.obs.size();
}







/* ---------------- 把 多孔介质Host→Device ------------------ */
void upload_obstacles(const Porous_host& h)
{
    int n = h.obst_num;
    if (n > MAX_OBS) throw std::runtime_error("MAX_OBS too small!");
    cudaMemcpyToSymbol(d_obs, h.obs.data(), n * sizeof(Obstacle));
    cudaMemcpyToSymbol(obst_num_gpu, &n, sizeof(int)); // ✅ 这句非常重要
}

/* 若需要清空（可选）
void clear_obstacles(){
    int zero = 0;
    cudaMemcpyToSymbol(obst_num,&zero,sizeof(int));
}
*/
// 工具：统一 cudaMalloc / cudaFree 一个 Fluid_dev
void alloc_fluid(Fluid_dev& f){
    cudaMalloc(&f.rho ,mem_size_scalar); cudaMalloc(&f.ux  ,mem_size_scalar);
    cudaMalloc(&f.uy  ,mem_size_scalar); cudaMalloc(&f.psi ,mem_size_scalar);
    cudaMalloc(&f.pressure,mem_size_scalar);
    cudaMalloc(&f.Fx_mol  ,mem_size_scalar);  cudaMalloc(&f.Fy_mol  ,mem_size_scalar);
    cudaMalloc(&f.Fx_ads  ,mem_size_scalar);  cudaMalloc(&f.Fy_ads  ,mem_size_scalar);
    cudaMalloc(&f.fin ,mem_size_distfun); cudaMalloc(&f.fout,mem_size_distfun);
    cudaMalloc(&f.min ,mem_size_distfun); cudaMalloc(&f.mout,mem_size_distfun);
    cudaMalloc(&f.S   ,mem_size_distfun); cudaMalloc(&f.C   ,mem_size_distfun);
}

void alloc_mix(Mix_dev& m){
    cudaMalloc(&m.rho ,mem_size_scalar); cudaMalloc(&m.ux  ,mem_size_scalar);
    cudaMalloc(&m.uy  ,mem_size_scalar); cudaMalloc(&m.pressure,mem_size_scalar);
    cudaMalloc(&m.pointsflag, mem_size_flag); // 用于存储几何信息
}

void free_fluid(Fluid_dev& f){
    auto df=[&](double*& p){ if(p){ cudaFree(p); p=nullptr;} };
    df(f.rho ); df(f.ux ); df(f.uy ); df(f.psi ); df(f.pressure);
    df(f.Fx_mol  ); df(f.Fy_mol  ); df(f.Fx_ads  ); df(f.Fy_ads  );
    df(f.fin); df(f.fout); df(f.min ); df(f.mout);
    df(f.S   ); df(f.C   );
}

void free_mix(Mix_dev& m){
    auto df=[&](double*& p){ if(p){ cudaFree(p); p=nullptr;} };
	auto df_int=[&](int*& p){ if(p){ cudaFree(p); p=nullptr;} };
    df(m.rho ); df(m.ux ); df(m.uy ); df(m.pressure);
    df_int(m.pointsflag);
}


/* 设备端 → 主机端拷贝，并逐场 NaN 扫描 */
void copy_and_check(const Fluid_dev& d,
                           Fluid_host&      h,
                           const char*      tag)     // 建议 tag = "A_" / "B_"
{
    struct Item { const char* n;
                  const double* dp;   // device pointer
                  double*       hp;   // host pointer
                  size_t        sz; };

    const Item tbl[] = {
        {"rho", d.rho,      h.rho.data(),      mem_size_scalar},
        {"ux",  d.ux,       h.ux.data(),       mem_size_scalar},
        {"uy",  d.uy,       h.uy.data(),       mem_size_scalar},
        {"psi", d.psi,      h.psi.data(),      mem_size_scalar},
        {"P",   d.pressure, h.pressure.data(), mem_size_scalar},
        {"Fx_mol",  d.Fx_mol,       h.Fx_mol.data(),       mem_size_scalar},
        {"Fy_mol",  d.Fy_mol,       h.Fy_mol.data(),       mem_size_scalar},
        {"Fx_ads",  d.Fx_ads,       h.Fx_ads.data(),       mem_size_scalar},
        {"Fy_ads",  d.Fy_ads,       h.Fy_ads.data(),       mem_size_scalar}
    };

    for (const auto& t : tbl) {
        checkCudaErrors(cudaMemcpy(t.hp, t.dp, t.sz, cudaMemcpyDeviceToHost));
        check_nan_all(t.hp, (std::string(tag) + t.n).c_str());
    }
}


void copy_back_mix(const Mix_dev& d,
                          Mix_host&      h,
                          bool do_nan_check)
{
    // 1. 拷贝
    checkCudaErrors(cudaMemcpy(h.rho.data(),      d.rho,      mem_size_scalar, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h.ux.data(),       d.ux,       mem_size_scalar, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h.uy.data(),       d.uy,       mem_size_scalar, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h.pressure.data(), d.pressure, mem_size_scalar, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h.pointsflag.data(), d.pointsflag, mem_size_flag, cudaMemcpyDeviceToHost));

    // 2. 可选 NaN 扫描
    if (do_nan_check) {
        check_nan_all(h.rho.data(),      "mix_rho");
        check_nan_all(h.ux.data(),       "mix_ux");
        check_nan_all(h.uy.data(),       "mix_uy");
        check_nan_all(h.pressure.data(), "mix_P");
    }
}



__device__ __forceinline__ double meq_gpu(const int k, const double rho, const double u[2]) {
	double u_squ = u[0] * u[0] + u[1] * u[1];

	switch (k) {
	case 0: return rho;
	case 1: return (-2.0 + 3.0 * u_squ) * rho;
	case 2: return (1.0 - 3.0 * u_squ) * rho;
	case 3: return u[0] * rho;
	case 4: return -u[0] * rho;
	case 5: return u[1] * rho;
	case 6: return -u[1] * rho;
	case 7: return (u[0] * u[0] - u[1] * u[1]) * rho;
	case 8: return (u[0] * u[1]) * rho;
	default: return 0.0;
	}
}

// forward declarations of kernels核函数前置声明
__global__ void init_wall_mat_from_flag(int* pointsflag);
__global__ void mark_fluid_solid(int* pointsflag);
__global__ void mark_boundary(int* pointsflag);
__global__ void mark_ghost(int* pointsflag);
__global__ void init_gpu(int* pointsflag, double* rho_A, double* fin_A, double* fout_A, double* min_A, double* mout_A,
						  double* rho_B, double* fin_B, double* fout_B, double* min_B, double* mout_B);
__host__ void update_fluid_A_rho_psi_pressure(double* rho_A, double* psi_A, double* pressure_A, const double* fin_A, int* pointsflag);
__host__ void update_fluid_B_rho_psi_pressure(double* rho_B, double* psi_B, double* pressure_B, const double* fin_B, int* pointsflag);

__global__ void compute_total_density_gpu(const double* rho_A, const double* rho_B, double* rho_host,int* pointsflag);
__global__ void compute_total_pressure_gpu(const double* rho_A, const double* rho_B,const double* psi_A, const double* psi_B,double*  pressure_host, int* pointsflag);
__global__ void compute_molecular_force_gpu(
    const double* psi_A,const double*  psi_B, const double* rho_A, const double* rho_B,
    double* Fx_mol_A,double* Fy_mol_A,double* Fx_mol_B,double* Fy_mol_B, int* pointsflag);
__global__ void compute_adsorption_force_gpu(
    const double* psi_A, const double* psi_B, double* Fx_ads_A, double* Fy_ads_A, double* Fx_ads_B, double* Fy_ads_B, int* pointsflag);

__global__ void compute_velocity_gpu_AB(
    const double* rho_A, const double* fin_A, const double* Fx_mol_A, const double* Fy_mol_A,
    const double* Fx_ads_A, const double* Fy_ads_A, double* ux_A, double* uy_A,
    const double* rho_B, const double* fin_B, const double* Fx_mol_B, const double* Fy_mol_B,
    const double* Fx_ads_B, const double* Fy_ads_B, double* ux_B, double* uy_B, int* pointsflag );
__global__ void compute_velocity_gpu_mix(const double* rho_A, const double* ux_A, const double* uy_A,const double* rho_B, const double* ux_B, const double* uy_B,
    			double* ux_host, double* uy_host, int* pointsflag);
__global__ void compute_S_gpu_A(const double* ux_host, const double* uy_host,
    const double* rho_A, const double* Fx_mol_A, const double* Fy_mol_A, const double* Fx_ads_A, const double* Fy_ads_A,
    const double* psi_A, double* S_A, int* pointsflag) ;
__global__ void compute_S_gpu_B(const double* ux_host, const double* uy_host,
    const double* rho_B, const double* Fx_mol_B, const double* Fy_mol_B, const double* Fx_ads_B, const double* Fy_ads_B,
    const double* psi_B,double* S_B , int* pointsflag);
__global__ void compute_C_gpu_A(const double* psi_A, double* C_A,int* pointsflag) ;
__global__ void mrt_collide_two_components_gpu(
        const double* rho_A,
        const double* S_A,  const double* C_A,
        const double* fin_A,      double* fout_A,  double* min_A,      double* mout_A,

        const double* rho_B,
        const double* S_B,   const double* C_B,   // C_B 目前没用，可保留
        const double* fin_B,      double* fout_B,  double* min_B,      double* mout_B,
		const double* ux_host, const double* uy_host,   int* pointsflag );
__global__ void stream_two_components_gpu(double* fin_A, const double* fout_A, double* fin_B, const double* fout_B ,int* pointsflag);
__global__ void boundary_gpu(double*  fin_A,const double* fout_A,double*  fin_B,const double*  fout_B, const int* pointsflag);
__global__ void sanitize_ghost_and_solid(const int* pointsflag,double* ux_A, double* uy_A, double* ux_B, double* uy_B,double* Fx_mA, double* Fy_mA, double* Fx_mB, double* Fy_mB);
// 初始化几何信息

__host__ void init_geometry(int* pointsflag)
{

	init_wall_mat_from_flag<<<grid, threads>>>(pointsflag);
    CUDA_CHECK(cudaGetLastError());
	mark_boundary<<<grid, threads>>>(pointsflag);
	CUDA_CHECK(cudaGetLastError());
	mark_ghost<<<grid, threads>>>(pointsflag);
	CUDA_CHECK(cudaGetLastError()); // 等待所有核函数执行完毕
	// 例如设置网格尺寸、边界条件等
}
/* ---------- Phase-1 : 标记流体(1) / 固体内部(-2) ---------------- */
/* ---------- Phase-1 : 标记流体(1) / 固体(-2) / 水合物(-3) -------- */
//原来理性几何下的使用
__global__ void mark_fluid_solid(int* pointsflag)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x>=NX || y>=NY) return;
    int idx = findindex_scalar_gpu(x,y);

    //extern __device__ unsigned char* d_wall_mat;
    int flag = 1;              // 默认流体
    unsigned char mat = 0;     // 0 表示非壁面

    #pragma unroll 4
    for (int k = 0; k < obst_num_gpu; ++k) {
        const Obstacle ob = d_obs[k];
        const long long dx = (long long)x - (long long)ob.cx;
        const long long dy = (long long)y - (long long)ob.cy;
        const double dist2 = double(dx*dx + dy*dy);

        if (dist2 <= ob.r2) {
            // —— 固体内核 ——
            flag = -2;     // 固体
            mat  = 1;      // 固体材料
            break;
        } else if (dist2 <= ob.r2_hydrate) {
            // —— 水合物壳 或 纯水合物圆 ——
            flag = -3;     // 水合物
            mat  = 2;      // 水合物材料（※ 不再沿用 ob.mat_id）
            break;
        }
    }

    // 边界/ghost 的覆写仅对流体格生效
    if (flag > 0) {
        if (y == 0 || y == NY-1)      flag = -1; //
        else if (y == 1 || y == NY-2) flag =  0; //
    }

    pointsflag[idx] = flag;

    // 只有壁面（-2/-3）写材料
    if (flag == -2 || flag == -3) {
        d_wall_mat[idx] = mat;   // 1=固体, 2=水合物
    }
}

__global__ void init_wall_mat_from_flag(int* pointsflag)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;

    int idx = findindex_scalar_gpu(x, y);
    int f = pointsflag[idx];

    unsigned char mat = 0;
    if      (f == -2) mat = 1;   // -2: 石英，对应材料1
    else if (f == -3) mat = 2;   // -3: 水合物，对应材料2
    // 其它（流体/ghost）保持0

    d_wall_mat[idx] = mat;
}



/* ==========================================================
 * Phase-2 kernel : 把贴固体的流体节点改为边界(0)
 * ----------------------------------------------------------
 * 对所有当前 flag==1 的格点，检查 8 邻中是否存在 -2，
 * 若存在则把自己标记为 0 —— 与 CPU 版第二段完全一致。
 * ========================================================== */
__global__ void mark_boundary(int* pointsflag)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x>=NX || y>=NY) return;
    int idx = findindex_scalar_gpu(x,y);
    // 只有流体格才做边界判断
    if (pointsflag[idx] != 1) return;

    bool isBoundary = false;
    #pragma unroll
    for (int k = 0; k < Q; ++k) {
        int xp = (x + e_gpu[k][0] + NX) % NX;
        int yp = (y + e_gpu[k][1] + NY) % NY;
        int nb = findindex_scalar_gpu(xp, yp);
        int f = pointsflag[nb];
        // 如果邻居是固体(-2)或水合物(-3)，就打边界
        if (f == -2 || f == -3) {
            isBoundary = true;
            break;
        }
    }
    if (isBoundary) pointsflag[idx] = 0;  // 0 标记为边界
}
/* ==========================================================
 * Phase-3 kernel : 把仍为 -2 的内部格点贴上虚拟层(-1)
 * ----------------------------------------------------------
 * 对 flag==0 的边界节点，再扫一圈邻居，
 * 把仍保持 -2 的格点改成 -1（ghost layer）。
 * ========================================================== */
__global__ void mark_ghost(int* pointsflag)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;
	int idx = findindex_scalar_gpu(x, y);
    if (pointsflag[idx] != 0) return;

    #pragma unroll
    for (int k = 0; k < Q; ++k) {
        int xp = (x + e_gpu[k][0] + NX) % NX;
        int yp = (y + e_gpu[k][1] + NY) % NY;
        int nb = findindex_scalar_gpu(xp, yp);
        int f = pointsflag[nb];
        // 如果邻居是固体(-2)或水合物(-3)，就把它设成 ghost(-1)
        if (f == -2 || f == -3) {
            pointsflag[nb] = -1;
            unsigned char m = d_wall_mat[nb];
            // 局部耦合系数：直接由材料查表
            d_GAw_map[nb] = GAw_by_mat_gpu[m];
            d_GBw_map[nb] = GBw_by_mat_gpu[m];
        }
    }
}


//用于在主函数中启动 GPU 上的初始化核函数
__host__ void init_all(int* pointsflag, double* rho_A, double* fin_A, double* fout_A, double*min_A ,double*mout_A,
						double* rho_B,double* fin_B, double* fout_B, double* min_B ,double*mout_B)
{
	init_gpu<<<grid, threads>>>(pointsflag, rho_A, fin_A, fout_A, min_A, mout_A, rho_B,fin_B, fout_B, min_B, mout_B);
	CUDA_CHECK(cudaGetLastError());        // 立即检查 launch 配置错误
	CUDA_CHECK(cudaDeviceSynchronize());   // 等内核跑完再继续
}

//用于将所有需要在 GPU 上共享使用的物理常量和控制参数复制到 __constant__ 内存中。
//cudaMemcpyToSymbol 是专门用于将值写入 __constant__ 内存的
__host__ void init_device_variable() {
	using namespace eos;
    using namespace phys;

	// D2Q9 方向与权重
	CK(cudaMemcpyToSymbol(e_gpu, &e, sizeof(int) * Q * 2));
	CK(cudaMemcpyToSymbol(opp_gpu, &opp, sizeof(int) * Q));
	CK(cudaMemcpyToSymbol(w_gpu, &w, sizeof(double) * Q));
	CK(cudaMemcpyToSymbol(w_F_gpu, &w_F, sizeof(double) * Q));

	// MRT 矩阵与松弛因子
	CK(cudaMemcpyToSymbol(M_gpu, &M, sizeof(double) * Q * Q));
	CK(cudaMemcpyToSymbol(Minv_gpu, &Minv, sizeof(double) * Q * Q));

	CK(cudaMemcpyToSymbol(tau_e_gpu, &tau_e, sizeof(double)));
	CK(cudaMemcpyToSymbol(tau_t_gpu, &tau_t, sizeof(double)));
	//CK(cudaMemcpyToSymbol(tau_p_gpu, &tau_p, sizeof(double)));
	CK(cudaMemcpyToSymbol(tau_q_gpu, &tau_q, sizeof(double)));

	// 各组分吸引力参数
	CK(cudaMemcpyToSymbol(GAA_gpu, &GAA, sizeof(double)));
	//CK(cudaMemcpyToSymbol(GAB_gpu, &GAB, sizeof(double)));
	//CK(cudaMemcpyToSymbol(GBA_gpu, &GBA, sizeof(double)));
	CK(cudaMemcpyToSymbol(GBB_gpu, &GBB, sizeof(double)));
	// 初始条件配置
	CK(cudaMemcpyToSymbol(ini_opt_gpu, &ini_opt, sizeof(int)));
	// 初始位置和半径
	CK(cudaMemcpyToSymbol(contact_angle_dir_gpu, &contact_angle_dir, sizeof(int)));
    CK(cudaMemcpyToSymbol(phi_contact_pho_A_gpu, &phi_contact_pho_A, sizeof(double)));
    CK(cudaMemcpyToSymbol(delta_pho_A_gpu, &delta_pho_A, sizeof(double)));
	CK(cudaMemcpyToSymbol(x_ini_gpu, &x_ini, sizeof(int)));//暂时没用
	CK(cudaMemcpyToSymbol(y_ini_gpu, &y_ini, sizeof(int)));
	CK(cudaMemcpyToSymbol(radius_gpu, &radius, sizeof(double)));
	CK(cudaMemcpyToSymbol(w_ini_gpu, &w_ini, sizeof(double)));
	// 初始密度和温度（两组分）
	CK(cudaMemcpyToSymbol(reducedT_w_ini_gpu, &reducedT_w_ini, sizeof(double)));
	// 网格物理参数
	CK(cudaMemcpyToSymbol(deltax_gpu, &deltax, sizeof(double)));
	CK(cudaMemcpyToSymbol(deltat_gpu, &deltat, sizeof(double)));
	CK(cudaMemcpyToSymbol(c_gpu, &c, sizeof(double)));
	CK(cudaMemcpyToSymbol(cs2_gpu, &cs2, sizeof(double)));
	// 状态方程参数
	CK(cudaMemcpyToSymbol(a_w_gpu, &a_w, sizeof(double)));
	CK(cudaMemcpyToSymbol(b_w_gpu, &b_w, sizeof(double)));
	CK(cudaMemcpyToSymbol(R_w_gpu, &R_w, sizeof(double)));
	CK(cudaMemcpyToSymbol(omega_w_gpu, &omega_w, sizeof(double)));
	CK(cudaMemcpyToSymbol(Tc_w_gpu, &Tc_w, sizeof(double)));
	CK(cudaMemcpyToSymbol(a_m_gpu, &a_m, sizeof(double)));
	CK(cudaMemcpyToSymbol(b_m_gpu, &b_m, sizeof(double)));
	CK(cudaMemcpyToSymbol(R_m_gpu, &R_m, sizeof(double)));
	CK(cudaMemcpyToSymbol(omega_m_gpu, &omega_m, sizeof(double)));
	CK(cudaMemcpyToSymbol(Tc_m_gpu, &Tc_m, sizeof(double)));
	CK(cudaMemcpyToSymbol(T_gpu, &T, sizeof(double)));

	CK(cudaMemcpyToSymbol(PR_scalar_gpu, &PR_scalar, sizeof(double)));
	//CK(cudaMemcpyToSymbol(sigmaA_gpu, &sigmaA, sizeof(double)));

}
//推进一步 LBM 的演化过程（相当于 time step 的执行器）
// 所有步骤在 GPU 上并行进行，整合在 evolution(...) 中，便于在 main() 中统一调用。
__host__ void evolution_all(
    double* rho_A, double* ux_A, double* uy_A,
    double* psi_A, double* pressure_A,
    double* Fx_mol_A, double* Fy_mol_A, double* Fx_ads_A, double* Fy_ads_A,
    double* fin_A, double* fout_A, double* min_A, double* mout_A, double* S_A, double* C_A,

    double* rho_B, double* ux_B, double* uy_B,
    double* psi_B, double* pressure_B,
    double* Fx_mol_B, double* Fy_mol_B, double* Fx_ads_B, double* Fy_ads_B,
    double* fin_B, double* fout_B, double* min_B, double* mout_B, double* S_B, double* C_B,

	double* rho_host, double* pressure_host,double* ux_host, double* uy_host, int* pointsflag )
    {
    update_fluid_A_rho_psi_pressure(rho_A, psi_A, pressure_A, fin_A, pointsflag);
    update_fluid_B_rho_psi_pressure(rho_B, psi_B, pressure_B, fin_B, pointsflag);
	KCALL(compute_total_density_gpu, (rho_A, rho_B, rho_host, pointsflag));
	KCALL(compute_total_pressure_gpu, (rho_A, rho_B, psi_A, psi_B, pressure_host,pointsflag));
    KCALL(compute_molecular_force_gpu, (psi_A, psi_B, rho_A, rho_B, Fx_mol_A, Fy_mol_A, Fx_mol_B, Fy_mol_B, pointsflag));
    KCALL(compute_adsorption_force_gpu, (psi_A, psi_B, Fx_ads_A, Fy_ads_A, Fx_ads_B, Fy_ads_B, pointsflag));
	KCALL(compute_velocity_gpu_AB, (rho_A, fin_A, Fx_mol_A, Fy_mol_A, Fx_ads_A, Fy_ads_A, ux_A, uy_A,
									rho_B, fin_B, Fx_mol_B, Fy_mol_B, Fx_ads_B, Fy_ads_B, ux_B, uy_B, pointsflag));

	KCALL(compute_velocity_gpu_mix, (rho_A, ux_A, uy_A, rho_B, ux_B, uy_B, ux_host, uy_host, pointsflag));
	KCALL(compute_S_gpu_A, (ux_host, uy_host, rho_A, Fx_mol_A, Fy_mol_A, Fx_ads_A, Fy_ads_A, psi_A, S_A, pointsflag));
	KCALL(compute_S_gpu_B, (ux_host, uy_host, rho_B, Fx_mol_B, Fy_mol_B, Fx_ads_B, Fy_ads_B, psi_B, S_B, pointsflag));
	KCALL(compute_C_gpu_A, (psi_A, C_A, pointsflag));
	//调用 MRT 碰撞与流动核函数，处理两组分的分布函数
	KCALL(mrt_collide_two_components_gpu, (rho_A, S_A, C_A, fin_A, fout_A, min_A, mout_A,
       rho_B, S_B, C_B, fin_B, fout_B, min_B, mout_B,ux_host, uy_host, pointsflag));
	KCALL(stream_two_components_gpu, (fin_A, fout_A, fin_B, fout_B, pointsflag));
	KCALL(boundary_gpu, (fin_A, fout_A, fin_B, fout_B, pointsflag));
    KCALL(sanitize_ghost_and_solid,(pointsflag, ux_A, uy_A, ux_B, uy_B, Fx_mol_A, Fy_mol_A, Fx_mol_B, Fy_mol_B));
}


//计算标量场的索引,需要修改
// 温和随机初始化（平滑噪声 + 软阈值 + 远离极端）
__device__ inline uint64_t hash_xy_seed(int x, int y, unsigned long long seed){
    return ( (uint64_t)(uint32_t)x << 32 ) ^ (uint64_t)(uint32_t)y ^ (uint64_t)seed;
}

__device__ inline double rand01_at(int x, int y, unsigned long long seed){
    // 这里要用“传入的 seed”，而不是再去读 get_water_seed()
    return u01( mix64( hash_xy_seed(x, y, seed) ) );
}


// 3×3 平滑核（近似高斯）：1 2 1 / 2 4 2 / 1 2 1  ，总权重 16
__device__ inline double smooth_noise_3x3(int x, int y, int NX, int NY, unsigned long long seed){
    // 边界用 clamp（非周期）
    auto clampi = [](int v, int lo, int hi){ return v<lo?lo:(v>hi?hi:v); };
    double acc = 0.0;
    int w[3][3] = {{1,2,1},{2,4,2},{1,2,1}};
    for(int dy=-1; dy<=1; ++dy){
        for(int dx=-1; dx<=1; ++dx){
            int nx = clampi(x+dx, 0, NX-1);
            int ny = clampi(y+dy, 0, NY-1);
            acc += w[dy+1][dx+1] * rand01_at(nx, ny, seed);
        }
    }
    return acc / 16.0; // 归一化到 [0,1] 附近
}


// 角点随机 + 双线性插值（相关长度 ~ L）
__device__ inline double coarse_bilinear_noise(
    int x, int y, int NX, int NY, unsigned long long seed, int L)
{
    auto clampi = [](int v, int lo, int hi){ return v<lo?lo:(v>hi?hi:v); };

    int x0 = (x / L) * L;
    int y0 = (y / L) * L;
    int x1 = clampi(x0 + L, 0, NX-1);
    int y1 = clampi(y0 + L, 0, NY-1);

    double fx = (L==0) ? 0.0 : (double)(x - x0) / (double)max(L,1);
    double fy = (L==0) ? 0.0 : (double)(y - y0) / (double)max(L,1);

    // 角点 4 个随机数（可用你现有的 mix64/u01）
    double n00 = rand01_at(x0, y0, seed);
    double n10 = rand01_at(x1, y0, seed);
    double n01 = rand01_at(x0, y1, seed);
    double n11 = rand01_at(x1, y1, seed);

    double nx0 = n00*(1.0 - fx) + n10*fx;
    double nx1 = n01*(1.0 - fx) + n11*fx;
    return nx0*(1.0 - fy) + nx1*fy; // ∈[0,1]
}

__global__ void init_gpu_1(
    int* pointsflag,
    double* rho_A, double* fin_A, double* fout_A, double* min_A , double* mout_A,
    double* rho_B, double* fin_B, double* fout_B, double* min_B , double* mout_B)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;

    const int idx = findindex_scalar_gpu(x, y);
    double u0[2] = {0.0, 0.0};
    const double RHO_FLOOR = 1e-5;

    if (pointsflag[idx] < 0){
        rho_A[idx] = 0.0; rho_B[idx] = 0.0;
    } else {
        // ---- 读全局参数 ----
        double Sw = fmin(fmax(get_water_satur(), 0.0), 1.0);
        const double rA_hi = fmax(rhoA_hi(), RHO_FLOOR);
        const double rA_lo = fmax(rhoA_lo(), RHO_FLOOR);
        const double rB_hi = fmax(rhoB_hi(), RHO_FLOOR);
        const double rB_lo = fmax(rhoB_lo(), RHO_FLOOR);

        // ---- Sw 极端：硬特判（使用你封装的 4 个 getter）----
        const double EPSW = 1e-12;
        if (Sw <= EPSW){
            // 没水：A 取“Sw=0 的 A 值（低）”，B 取“Sw=0 的 B 值（高）”
            rho_A[idx] = fmax(rhoA_lo_0(), RHO_FLOOR);
            rho_B[idx] = fmax(rhoB_hi_0(), RHO_FLOOR);
        } else if (Sw >= 1.0 - EPSW){
            // 全水：A 取“Sw=1 的 A 值（高）”，B 取“Sw=1 的 B 值（低）”
            rho_A[idx] = fmax(rhoA_hi_1(), RHO_FLOOR);
            rho_B[idx] = fmax(rhoB_lo_1(), RHO_FLOOR);
        } else {
            // ---- 随机（可复现）：粗+细噪声 + 软阈 ----
            const double minority = fmin(Sw, 1.0 - Sw);
            const double dist_mid = fabs(Sw - 0.5);

            int L = 8
                  + (int)llround(12.0 * exp(-(dist_mid*dist_mid)/0.05))
                  + (int)llround(10.0 * fmax(0.0, 0.12 - minority) / 0.12);
            L = max(8, min(L, 32));

            const double eps = fmin(0.22, fmax(0.08,
                0.08 + 0.06*exp(-(dist_mid*dist_mid)/0.05) + fmax(0.0, 0.40*(0.12 - minority))
            ));
            const double delta = fmin(0.10, fmax(0.03,
                0.03 + 0.02*exp(-(dist_mid*dist_mid)/0.05) + fmax(0.0, 0.25*(0.12 - minority))
            ));
            const double alpha = 0.85 + 0.10*exp(-(dist_mid*dist_mid)/0.05);

            const unsigned long long seed = get_water_seed();     // 固定种子 → 可复现
            const double phi_coarse = coarse_bilinear_noise(x,y,NX,NY,seed,L);
            const double phi_fine   = smooth_noise_3x3(x,y,NX,NY,seed);
            const double phi = alpha * phi_coarse + (1.0 - alpha) * phi_fine;

            // 方向：phi < Sw ⇒ A 多
            double s = 0.5 * (1.0 + tanh((Sw - phi) / eps));
            const double delta_eff = delta * 4.0 * Sw * (1.0 - Sw); // 两端收敛到 0
            s = (1.0 - 2.0*delta_eff) * s + delta_eff;

            rho_A[idx] = rA_lo + s * (rA_hi - rA_lo);
            rho_B[idx] = rB_hi + s * (rB_lo - rB_hi);
        }

        // 保护：避免极小负值
        rho_A[idx] = fmax(rho_A[idx], RHO_FLOOR);
        rho_B[idx] = fmax(rho_B[idx], RHO_FLOOR);
    }

    // —— 分布函数置平衡 —— //
    #pragma unroll
    for (int k=0; k<Q; ++k){
        const int off = findindex_distfun_gpu(x, y, k);
        const double feqA = feq_gpu(k, rho_A[idx], u0);
        const double feqB = feq_gpu(k, rho_B[idx], u0);
        fin_A[off]  = fout_A[off] = feqA;
        fin_B[off]  = fout_B[off] = feqB;

        const double meqA = meq_gpu(k, rho_A[idx], u0);
        const double meqB = meq_gpu(k, rho_B[idx], u0);
        min_A[off]  = mout_A[off] = meqA;
        min_B[off]  = mout_B[off] = meqB;
    }
}

//计算标量场的索引,需要修改
// ====== 仅加入 Sw 极端硬特判 + 数值保护的轻量版 init_gpu ======
__global__ void init_gpu(
    int* pointsflag,
    double* rho_A, double* fin_A, double* fout_A, double* min_A , double* mout_A,
    double* rho_B, double* fin_B, double* fout_B, double* min_B , double* mout_B)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // 每个线程负责 (x,y)
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;

    const int idx = findindex_scalar_gpu(x, y);
    double u0[2] = {0.0, 0.0};

    // ---- 数值保护参数 ----
    const double RHO_FLOOR = 1e-5;
    const double EPSW      = 1e-12;

    if (pointsflag[idx] < 0){
        // 固体/ghost：全零
        rho_A[idx] = 0.0;
        rho_B[idx] = 0.0;
    } else {
        // ---- 读全局参数 ----
        const double Sw    = fmin(fmax(get_water_satur(), 0.0), 1.0);
        const double rA_hi = fmax(rhoA_hi(), RHO_FLOOR);
        const double rA_lo = fmax(rhoA_lo(), RHO_FLOOR);
        const double rB_hi = fmax(rhoB_hi(), RHO_FLOOR);
        const double rB_lo = fmax(rhoB_lo(), RHO_FLOOR);

        // ---- Sw 极端：硬特判（同 init_gpu_1）----
        if (Sw <= EPSW){
            // 没水：A 取“Sw=0 的 A（低）”，B 取“Sw=0 的 B（高）”
            rho_A[idx] = fmax(rhoA_lo_0(), RHO_FLOOR);
            rho_B[idx] = fmax(rhoB_hi_0(), RHO_FLOOR);
        } else if (Sw >= 1.0 - EPSW){
            // 全水：A 取“Sw=1 的 A（高）”，B 取“Sw=1 的 B（低）”
            rho_A[idx] = fmax(rhoA_hi_1(), RHO_FLOOR);
            rho_B[idx] = fmax(rhoB_lo_1(), RHO_FLOOR);
        } else {
            // ---- 维持你原来的“逐格伯努利随机”方案（轻量&可复现）----
            // 说明：此处使用 get_water_seed() 作为固定种子，保持运行间可复现；
            // 若想“一次运行内可复现，换种子可变化”，就让 get_water_seed() 由外部配置。
            const uint64_t base = ( (uint64_t)(uint32_t)x << 32 )
                                 ^ (uint64_t)(uint32_t)y
                                 ^ (uint64_t)get_water_seed();
            const double rnd = u01( mix64(base) );

            if (rnd < Sw) {
                rho_A[idx] = rA_hi;
                rho_B[idx] = rB_lo;
            } else {
                rho_A[idx] = rA_lo;
                rho_B[idx] = rB_hi;
            }
        }

        // ---- 再次数值保护：避免极小负值/0 ----
        rho_A[idx] = fmax(rho_A[idx], RHO_FLOOR);
        rho_B[idx] = fmax(rho_B[idx], RHO_FLOOR);
    }

    // ===== 分布函数/矩量函数初始化为平衡态 =====
    #pragma unroll
    for (int k = 0; k < Q; ++k){
        const int off = findindex_distfun_gpu(x, y, k);

        const double feqA = feq_gpu(k, rho_A[idx], u0);
        const double feqB = feq_gpu(k, rho_B[idx], u0);
        fin_A [off] = fout_A[off] = feqA;
        fin_B [off] = fout_B[off] = feqB;

        const double meqA = meq_gpu(k, rho_A[idx], u0);
        const double meqB = meq_gpu(k, rho_B[idx], u0);
        min_A [off] = mout_A[off] = meqA;
        min_B [off] = mout_B[off] = meqB;
    }
}

/* 共享字节数 = 2*blockDim.x*sizeof(double)  (这里 blockDim.y=1) */
__global__ void compute_rho_fluid_A(
        double*       rho_A,
        const double* __restrict__ fin_A,
        const int*    __restrict__ pointsflag,
        double*       g_minRho,
        double*       g_maxRho )
{
    extern __shared__ double sminmax[];      // 前一段存 min，后一段存 max

    /* —— 全局坐标 —— */
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;

    int idx = findindex_scalar_gpu(x, y);
    int tid = threadIdx.y * blockDim.x + threadIdx.x;  // 1-D 线程索引

    /* —— 每线程本地极值 —— */
    double vmin = 1e100, vmax = -1e100;
    if (pointsflag[idx] >= 0) {
        double rho = 0.0;
        #pragma unroll
        for (int k = 0; k < Q; ++k)
            rho += fin_A[ findindex_distfun_gpu(x, y, k) ];
        rho_A[idx] = fmax(rho, 1e-5);
        vmin = vmax = rho;
    }
    sminmax[tid]               = vmin;
    sminmax[tid + blockDim.x * blockDim.y]  = vmax;
    __syncthreads();

    /* —— block 内 1-D 规约 —— */
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sminmax[tid]               = fmin(sminmax[tid],               sminmax[tid+stride]);
            sminmax[tid+ blockDim.x]    = fmax(sminmax[tid+ blockDim.x],    sminmax[tid+ blockDim.x+stride]);
        }
        __syncthreads();
    }

    /* —— 每 block 原子写一次全局极值 —— */
    if (tid == 0) {
        atomicMin_double(g_minRho, sminmax[0]);
        atomicMax_double(g_maxRho, sminmax[ blockDim.x * blockDim.y]);
    }
}

__global__ void fill_ghost_rho_A(
        double*       rho_A,
        const int*    __restrict__ pointsflag,
        double        min_rho, double max_rho)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;

    int idx = findindex_scalar_gpu(x, y);
    if (pointsflag[idx] != -1) return;       // 只处理虚拟层

    double rho_avg=0.0, denom=0.0;
    #pragma unroll
    for (int k=0;k<Q;++k){
        int xp = (x + e_gpu[k][0] + NX) % NX;
        int yp = (y + e_gpu[k][1] + NY) % NY;
        int nb = findindex_scalar_gpu(xp, yp);
        if (pointsflag[nb] >= 0){
            rho_avg += w_F_gpu[k]*rho_A[nb];
            denom   += w_F_gpu[k];
        }
    }
    rho_avg /= denom;

    double rho_g = (contact_angle_dir_gpu==0) ?                              // 亲液
                   fmin(phi_contact_pho_A_gpu*rho_avg, max_rho)
                   :                                                      // 疏液
                   fmax(rho_avg - delta_pho_A_gpu,  min_rho);

    rho_A[idx] = rho_g;
}


__global__ void compute_p_psi_A_all(
        const double* __restrict__ rho_A,
        double*       pressure_A,
        double*       psi_A,
        const int*    __restrict__ pointsflag,
        double h_minRho, double h_maxRho)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;
    int idx = findindex_scalar_gpu(x, y);
    // —— 1) ghost layer —— 用密度直接做 psi，跳过下面的 EOS
    if (pointsflag[idx] == -2) return;


    double rho_safe = fmax(rho_A[idx], 1e-5);
    double tmp1 = 0.0, tmp2 = 0.0;
    double alpha_gpu = 0.0;

    alpha_gpu = pow(1.0 + (0.37464 + 1.54226 * omega_w_gpu - 0.26992 * omega_w_gpu * omega_w_gpu) * (1.0 - sqrt(reducedT_w_ini_gpu)), 2.0);
    double denom1 = fmax(1.0 - b_w_gpu * rho_safe, 1e-4);
	double denom2 = fmax(1.0 + 2.0 * b_w_gpu * rho_safe - b_w_gpu * b_w_gpu * rho_safe * rho_safe, 1e-4);
    tmp1 = rho_safe * R_w_gpu * T_gpu / denom1;
	tmp2 = a_w_gpu * alpha_gpu * rho_safe * rho_safe / denom2;
    // === 建议版本 ===
    double p_tmp = PR_scalar_gpu * (tmp1 - tmp2);
    pressure_A[idx] = fmin(fmax(p_tmp, 1e-8), 1e3);


    double diff = pressure_A[idx] - rho_A[idx] * cs2_gpu;

    psi_A[idx] = sqrt(fmax(0.0, 2.0*diff/(GAA_gpu*deltax_gpu*deltax_gpu)));
}



__host__ void update_fluid_A_rho_psi_pressure(double* rho_A, double* psi_A, double* pressure_A,
                      const double* fin_A,  int*  pointsflag)
{
    /* —— 线程 & 网格配置：与你给出的保持一致 —— */
    const int nThreadsx = 32, nThreadsy = 1;
    dim3 threads(nThreadsx, nThreadsy, 1);
    dim3 grid((NX + nThreadsx - 1) / nThreadsx,
              (NY + nThreadsy - 1) / nThreadsy, 1);
    size_t shBytes = 2 * threads.x * sizeof(double);        // K1 共享内存

    /* —— 全局极值缓冲 —— */
    static double *d_minRho=nullptr, *d_maxRho=nullptr;
    if(!d_minRho){
        cudaMalloc(&d_minRho,sizeof(double));
        cudaMalloc(&d_maxRho,sizeof(double));
    }
    const double huge=1e100, negHuge=-1e100;
    cudaMemcpy(d_minRho,&huge,   sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_maxRho,&negHuge,sizeof(double),cudaMemcpyHostToDevice);

    /* ① ρ_fluid & block-reduce */
    compute_rho_fluid_A<<<grid,threads,shBytes>>>(
        rho_A, fin_A, pointsflag, d_minRho, d_maxRho);
    CUDA_CHECK(cudaGetLastError());

    double h_minRho, h_maxRho;
    cudaMemcpy(&h_minRho, d_minRho, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_maxRho, d_maxRho, sizeof(double), cudaMemcpyDeviceToHost);

    /* ② ghost ρ */
    fill_ghost_rho_A<<<grid,threads>>>(
        rho_A, pointsflag,
        h_minRho, h_maxRho);
    CUDA_CHECK(cudaGetLastError());

    /* ③ p & ψ */
    compute_p_psi_A_all<<<grid,threads>>>(
        rho_A, pressure_A, psi_A, pointsflag, h_minRho, h_maxRho);
    CUDA_CHECK(cudaGetLastError());
}

/* shared = 0 —— 只算 ρ，不归约极值 */
__global__ void compute_rho_fluid_B(
        double*       rho_B,
        const double* __restrict__ fin_B,
        const int*    __restrict__ pointsflag)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;    // blockDim.y = 1
    if (x>=NX || y>=NY) return;

    int idx = findindex_scalar_gpu(x,y);
    if (pointsflag[idx] >= 0){          // 0 / 1
        double rho = 0.0;
        #pragma unroll
        for(int k=0;k<Q;++k)
            rho += fin_B[ findindex_distfun_gpu(x,y,k) ];
        rho_B[idx] = fmax(rho, 1e-4);
    }
}
__global__ void fill_ghost_rho_B_avg(
        double*       rho_B,
        const int*    __restrict__ pointsflag)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;
    if (x>=NX || y>=NY) return;

    int idx = findindex_scalar_gpu(x,y);
    if (pointsflag[idx] != -1) return;

    double rho_avg = 0.0, denom = 0.0;
    #pragma unroll
    for(int k=0;k<Q;++k){
        int xp = (x + e_gpu[k][0] + NX) % NX;
        int yp = (y + e_gpu[k][1] + NY) % NY;
        int nb=findindex_scalar_gpu(xp,yp);
        if(pointsflag[nb]>=0){
            rho_avg += w_F_gpu[k]*rho_B[nb];
            denom   += w_F_gpu[k];
        }
    }
    rho_B[idx] = rho_avg/denom;          // ψ_B = ρ_B 后面一起算
}

__global__ void compute_p_psi_B_all(
        const double* __restrict__ rho_B,
        double*       pressure_B,
        double*       psi_B,
        const int*    __restrict__ pointsflag)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;
    if (x>=NX || y>=NY) return;

    int idx = findindex_scalar_gpu(x,y);
    if (pointsflag[idx] == -2) return;

    double rho = rho_B[idx];
    /* 理想气体 EOS */
    double p = rho * R_m_gpu * T_gpu;
    pressure_B[idx] = p;

    /*  ψ_B = ρ_B  （Shan-Chen 单伪势轻气体模型） */
    double roh_0 = 1;
    psi_B[idx] = roh_0*(1-exp(-rho/roh_0));
}

void update_fluid_B_rho_psi_pressure(double* rho_B, double* psi_B, double* pressure_B,
                      const double* fin_B, int* pointsflag)
{
    dim3 threads(32,1,1);
    dim3 grid((NX+threads.x-1)/threads.x,
              (NY+threads.y-1)/threads.y,1);

    /* 1) ρ_fluid */
    compute_rho_fluid_B<<<grid,threads>>>(
        rho_B, fin_B, pointsflag);
    CUDA_CHECK(cudaGetLastError());

    /* 2) ghost ρ_B 平均补齐 */
    fill_ghost_rho_B_avg<<<grid,threads>>>(
        rho_B, pointsflag);
    CUDA_CHECK(cudaGetLastError());

    /* 3) p & ψ_B */
    compute_p_psi_B_all<<<grid,threads>>>(
        rho_B, pressure_B, psi_B, pointsflag);
    CUDA_CHECK(cudaGetLastError());
}




__global__ void compute_total_density_gpu(
	const double* rho_A,  const double* rho_B, double* rho_host, int* pointsflag)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= NX || y >= NY) return;
	int index = findindex_scalar_gpu(x, y);
	if (pointsflag[index] >= 0) {
		rho_host[index] = rho_A[index] + rho_B[index];
	}
}


__global__ void compute_total_pressure_gpu(
	const double* rho_A, const double* rho_B,const double* psi_A, const double* psi_B, double* pressure_host, int* pointsflag)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= NX || y >= NY) return;
	int index = findindex_scalar_gpu(x, y);
	if (pointsflag[index] >= 0) {
		double rho_sum = rho_A[index] + rho_B[index];
        double rhoA = rho_A[index];
        double rhoB = rho_B[index];
		double psiA = psi_A[index];
		double psiB = psi_B[index];
        double GAB = get_GAB();
		pressure_host[index] = cs2_gpu * rho_sum + 0.5 * c_gpu * c_gpu * (GAA_gpu * psiA * psiA +GBB_gpu * psiB * psiB +2.0 * GAB * rhoA * rhoB);
	}
}



// 仅计算 fluid–fluid 分子力

// 仅计算 fluid–fluid 分子力
__global__ void compute_molecular_force_gpu(
    const double*  psi_A,
    const double*  psi_B,
    const double*  rho_A,
    const double*  rho_B,
    double*       Fx_mol_A,
    double*       Fy_mol_A,
    double*       Fx_mol_B,
    double*       Fy_mol_B,
    int*    pointsflag)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x>=NX||y>=NY) return;
    int idx = findindex_scalar_gpu(x,y);
    if (pointsflag[idx]<0) return;

    double psiA = psi_A[idx], psiB = psi_B[idx];
    double phiA_new = rho_A[idx];   // 或 S2 的函数
    double phiB_new = rho_B[idx];
    double tmpAA[2] = {0.0, 0.0}, tmpBB[2] = {0.0, 0.0};
	double tmpAB[2] = {0.0, 0.0}, tmpBA[2] = {0.0, 0.0};

    #pragma unroll
	for (int k = 1; k < Q; ++k) {
        int xp = (x + e_gpu[k][0] + NX) % NX;
        int yp = (y + e_gpu[k][1] + NY) % NY;
        int idxp = findindex_scalar_gpu(xp, yp);

		const double ex = w_F_gpu[k] * e_gpu[k][0];
        const double ey = w_F_gpu[k] * e_gpu[k][1];

        double psiA_nb = psi_A[idxp];
        double psiB_nb = psi_B[idxp];
        double phiA_nb_new = rho_A[idxp];
        double phiB_nb_new = rho_B[idxp];
        tmpAA[0] += ex * psiA_nb;  tmpAA[1] += ey * psiA_nb;
        tmpAB[0] += ex * phiB_nb_new;  tmpAB[1] += ey * phiB_nb_new;
        tmpBB[0] += ex * psiB_nb;   tmpBB[1] += ey * psiB_nb;
        tmpBA[0] += ex * phiA_nb_new;  tmpBA[1] += ey * phiA_nb_new;
    }
    double GAB = get_GAB();
    double GBA = get_GBA();
    double Fx_mA = -GAA_gpu * psiA * tmpAA[0] - GAB * phiA_new * tmpAB[0];
    double Fy_mA = -GAA_gpu * psiA * tmpAA[1] - GAB * phiA_new * tmpAB[1];
    double Fx_mB = -GBB_gpu * psiB * tmpBB[0] - GBA * phiB_new * tmpBA[0];
    double Fy_mB = -GBB_gpu * psiB * tmpBB[1] - GBA * phiB_new * tmpBA[1];

    Fx_mol_A[idx] = Fx_mA;   Fy_mol_A[idx] = Fy_mA;
    Fx_mol_B[idx] = Fx_mB;   Fy_mol_B[idx] = Fy_mB;
}
// 仅计算 fluid–wall 吸附力
__global__ void compute_adsorption_force_gpu(
    const double*  psi_A,
    const double*  psi_B,
    double*       Fx_ads_A,
    double*       Fy_ads_A,
    double*       Fx_ads_B,
    double*       Fy_ads_B,
    int*    pointsflag)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x>=NX||y>=NY) return;
    int idx = findindex_scalar_gpu(x,y);
    if (pointsflag[idx]<0) return;

    double psiA = psi_A[idx], psiB = psi_B[idx];
    double wAx=0.0, wAy=0.0, wBx=0.0, wBy=0.0;

    #pragma unroll
    for(int k=1;k<Q;++k){
        int xp=(x+e_gpu[k][0]+NX)%NX,
            yp=(y+e_gpu[k][1]+NY)%NY;
        int idxp = findindex_scalar_gpu(xp,yp);
        if(pointsflag[idxp]==-1){
            const double ex = w_F_gpu[k] * e_gpu[k][0];
            const double ey = w_F_gpu[k] * e_gpu[k][1];
            double GAw_loc = d_GAw_map[idxp];
            double GBw_loc = d_GBw_map[idxp];

            wAx += ex * psiA * GAw_loc;  wAy += ey * psiA * GAw_loc;
            wBx += ex * psiB * GBw_loc;  wBy += ey * psiB * GBw_loc; // ← 用 GBw_loc
        }

    }
    Fx_ads_A[idx] = -wAx;  Fy_ads_A[idx] = -wAy;
    Fx_ads_B[idx] = -wBx;  Fy_ads_B[idx] = -wBy;
}


//分别计算两相的速度+考虑了合成作用力
__global__ void compute_velocity_gpu_AB(
    const double* rho_A, const double* fin_A, const double* Fx_mol_A, const double* Fy_mol_A,
    const double* Fx_ads_A, const double* Fy_ads_A, double* ux_A, double* uy_A,
    const double* rho_B, const double* fin_B, const double* Fx_mol_B, const double* Fy_mol_B,
    const double* Fx_ads_B, const double* Fy_ads_B, double* ux_B, double* uy_B, int* pointsflag ) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= NX || y >= NY) return;
	int idx = findindex_scalar_gpu(x, y);
	if (pointsflag[idx] >= 0) {
		// --- A 组分 ---
		ux_A[idx] = 0.0;
		uy_A[idx] = 0.0;
		#pragma unroll
		for (int k = 0; k < Q; ++k) {
			int idxk = findindex_distfun_gpu(x, y, k);
			ux_A[idx] += e_gpu[k][0] * fin_A[idxk];
			uy_A[idx] += e_gpu[k][1] * fin_A[idxk];
		}
		double rhoA_safe = fmax(rho_A[idx], 1e-4);  // 防除 0
        double gx = get_Gx(), gy = get_Gy();
        const double Fx_A_all = ( Fx_mol_A[idx] + Fx_ads_A[idx] + d_drive_scale*gx*rhoA_safe );//总力
        const double Fy_A_all = ( Fy_mol_A[idx] + Fy_ads_A[idx] + d_drive_scale*gy*rhoA_safe );
        // 速度更新：ux = (ux + 0.
		ux_A[idx] = (ux_A[idx] + 0.5 * Fx_A_all * deltat_gpu ) /  rhoA_safe;
		uy_A[idx] = (uy_A[idx] + 0.5 * Fy_A_all * deltat_gpu ) /  rhoA_safe;
        const double UMAX = 0.12;     // 只数值护栏
        double u2A = ux_A[idx]*ux_A[idx] + uy_A[idx]*uy_A[idx];
        if (u2A > UMAX*UMAX){
            double s = UMAX / sqrt(u2A);
            ux_A[idx]*=s; uy_A[idx]*=s;
        }


		// --- B 组分 ---
		ux_B[idx] = 0.0;
		uy_B[idx] = 0.0;
		for (int k = 0; k < Q; ++k) {
			int idxk = findindex_distfun_gpu(x, y, k);
			ux_B[idx] += e_gpu[k][0] * fin_B[idxk];
			uy_B[idx] += e_gpu[k][1] * fin_B[idxk];
		}
		double rhoB_safe = fmax(rho_B[idx], 1e-4);
        const double Fx_B_all = ( Fx_mol_B[idx] + Fx_ads_B[idx] + d_drive_scale*gx*rhoB_safe );//总力
        const double Fy_B_all = ( Fy_mol_B[idx] + Fy_ads_B[idx] + d_drive_scale*gy*rhoB_safe );
		ux_B[idx] = (ux_B[idx] + 0.5 * Fx_B_all * deltat_gpu ) / rhoB_safe;
		uy_B[idx] = (uy_B[idx] + 0.5 * Fy_B_all * deltat_gpu ) / rhoB_safe;

        double u2B = ux_B[idx]*ux_B[idx] + uy_B[idx]*uy_B[idx];
        if (u2B > UMAX*UMAX){
            double s = UMAX / sqrt(u2B);
            ux_B[idx]*=s; uy_B[idx]*=s;
        }
	}
}

//混合速度计算核函数
__global__ void compute_velocity_gpu_mix(
    const double* rho_A, const double* ux_A, const double* uy_A,
    const double* rho_B, const double* ux_B, const double* uy_B,
    double* ux_host, double* uy_host, int* pointsflag
) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= NX || y >= NY) return;
	int idx = findindex_scalar_gpu(x, y);
	if (pointsflag[idx] >= 0) {
		double rA = rho_A[idx], rB = rho_B[idx];
		double uAx = ux_A[idx], uAy = uy_A[idx];
		double uBx = ux_B[idx], uBy = uy_B[idx];

		double rTot = fmax(rA + rB, 1e-12);
		ux_host[idx] = (rA * uAx + rB * uBx) / rTot;
		uy_host[idx] = (rA * uAy + rB * uBy) / rTot;
	}
}

//张量扰动核函数
__global__ void compute_C_gpu_A(const double* psi_A, double* C_A,  int* pointsflag)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= NX || y >= NY) return;
	int idx = findindex_scalar_gpu(x, y);
	if (pointsflag[idx] >= 0) {

		double psi0 = psi_A[idx];
		double Qxx = 0.0, Qyy = 0.0, Qxy = 0.0;
		#pragma unroll
		for (int k = 0; k < Q; ++k) {
			int xp = (x + e_gpu[k][0] + NX) % NX;
			int yp = (y + e_gpu[k][1] + NY) % NY;
			int idxp = findindex_scalar_gpu(xp, yp);

			double dpsi = psi_A[idxp] - psi0;
			Qxx += w_F_gpu[k] * dpsi * e_gpu[k][0] * e_gpu[k][0];
			Qyy += w_F_gpu[k] * dpsi * e_gpu[k][1] * e_gpu[k][1];
			Qxy += w_F_gpu[k] * dpsi * e_gpu[k][0] * e_gpu[k][1];
		}
        double kap = get_kappa();
		double coeff = 0.5 * kap * GAA_gpu * psi0;
		Qxx *= coeff;
		Qyy *= coeff;
		Qxy *= coeff;
        double tauA_loc = tauA();
		// 写入 moment 空间中的对应索引（直接使用 C_A）
		C_A[findindex_distfun_gpu(x, y, 0)] = 0.0; // C_0 不需要扰动
		C_A[findindex_distfun_gpu(x, y, 1)] = 1.5 / tau_e_gpu * (Qxx + Qyy);
		C_A[findindex_distfun_gpu(x, y, 2)] = -1.5 / tau_t_gpu * (Qxx + Qyy);
		C_A[findindex_distfun_gpu(x, y, 3)] = 0.0;
		C_A[findindex_distfun_gpu(x, y, 4)] = 0.0;
		C_A[findindex_distfun_gpu(x, y, 5)] = 0.0;
		C_A[findindex_distfun_gpu(x, y, 6)] = 0.0;
		C_A[findindex_distfun_gpu(x, y, 7)] = -1.0 / tauA_loc * (Qxx - Qyy);
		C_A[findindex_distfun_gpu(x, y, 8)] = -1.0 / tauA_loc * Qxy;
	}
}


//Guo 力模型下的外力源项,A组分
__global__ void compute_S_gpu_A(const double* ux_host, const double* uy_host,
    const double* rho_A, const double* Fx_mol_A, const double* Fy_mol_A,const double* Fx_ads_A, const double* Fy_ads_A,
    const double* psi_A, double* S_A, int* pointsflag)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= NX || y >= NY) return;
	int idx = findindex_scalar_gpu(x, y);
	if (pointsflag[idx] >= 0) {
        double gx = get_Gx(), gy = get_Gy();
        double rhoA_safe = rho_A[idx];  // 防除 0
		double u_A[2] = { ux_host[idx], uy_host[idx] };
		double F_A[2] = { Fx_mol_A[idx] + Fx_ads_A[idx] + d_drive_scale*gx*rhoA_safe, Fy_mol_A[idx] + Fy_ads_A[idx]+ d_drive_scale*gy*rhoA_safe };//总力
        double Fm[2] = {Fx_mol_A[idx]  , Fy_mol_A[idx]};//A相分子力

		double uF = u_A[0] * F_A[0] + u_A[1] *F_A[1];
		double Fm2 = Fm[0] * Fm[0] + Fm[1] * Fm[1];
		const double PSI_CUT = 1e-3;                          // 只数值截断
        double psi2 = psi_A[idx]*psi_A[idx] + PSI_CUT*PSI_CUT;
        double sigmaA = get_sigmaA();
		// 写入 moment space 的 S_k
		S_A[findindex_distfun_gpu(x, y, 0)] = 0.0;
		S_A[findindex_distfun_gpu(x, y, 1)] = 6.0 * uF + 12.0 * sigmaA * Fm2 / (psi2 * deltat_gpu * (tau_e_gpu - 0.5));
		S_A[findindex_distfun_gpu(x, y, 2)] = -6.0 * uF - 12.0 * sigmaA * Fm2 / (psi2 * deltat_gpu * (tau_t_gpu - 0.5));
		S_A[findindex_distfun_gpu(x, y, 3)] = F_A[0];
		S_A[findindex_distfun_gpu(x, y, 4)] = -F_A[0];
		S_A[findindex_distfun_gpu(x, y, 5)] = F_A[1];
		S_A[findindex_distfun_gpu(x, y, 6)] = -F_A[1];
		S_A[findindex_distfun_gpu(x, y, 7)] = 2*(u_A[0] * F_A[0] - u_A[1] * F_A[1]);
		S_A[findindex_distfun_gpu(x, y, 8)] = u_A[0] * F_A[1] + u_A[1] * F_A[0];
	}
}
__global__ void compute_S_gpu_B(const double* ux_host, const double* uy_host,
    const double* rho_B,const double* Fx_mol_B, const double* Fy_mol_B,const double* Fx_ads_B, const double* Fy_ads_B,
    const double* psi_B,double* S_B , int* pointsflag)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= NX || y >= NY) return;
	int idx = findindex_scalar_gpu(x, y);
	if (pointsflag[idx] >= 0) {
        double rhoB_safe = rho_B[idx];
        double gx = get_Gx(), gy = get_Gy();
		double u_B[2] = { ux_host[idx], uy_host[idx] };
		double F_B[2] = { Fx_mol_B[idx] + Fx_ads_B[idx] + d_drive_scale*gx*rhoB_safe, Fy_mol_B[idx] + Fy_ads_B[idx] + d_drive_scale*gy*rhoB_safe };

		double uF = u_B[0] * F_B[0] + u_B[1] * F_B[1];

        S_B[findindex_distfun_gpu(x, y, 0)] = 0.0;
		S_B[findindex_distfun_gpu(x, y, 1)] = 6.0 * uF;
		S_B[findindex_distfun_gpu(x, y, 2)] = -6.0 * uF;
		S_B[findindex_distfun_gpu(x, y, 3)] = F_B[0];
		S_B[findindex_distfun_gpu(x, y, 4)] = -F_B[0];
		S_B[findindex_distfun_gpu(x, y, 5)] = F_B[1];
		S_B[findindex_distfun_gpu(x, y, 6)] = -F_B[1];
		S_B[findindex_distfun_gpu(x, y, 7)] = 2*(u_B[0] * F_B[0] - u_B[1] * F_B[1]);
		S_B[findindex_distfun_gpu(x, y, 8)] = u_B[0] * F_B[1] + u_B[1] * F_B[0];
	}
}


__device__ __forceinline__
void positivity_limiter(double* __restrict__ fout,
                        int x,int y, double rho, const double u[2],
                        unsigned long long* ct_any,
                        unsigned long long* ct_full)
{
    // 1) 本格 feq
    double feq[Q];
    #pragma unroll
    for (int k=0;k<Q;++k) feq[k] = feq_gpu(k, rho, u);

    const double eps_abs = 1e-18, eps_rel = 1e-12;//想让 limiter 更少介入：把这两个都调小；
    const double eps = fmax(eps_abs, eps_rel * rho);  // 正性下限：数值级别
    double beta_min = 0.0;
    const double FULL_T = 50.0*eps;//想明显减少 full：把 50 改大一点，
    bool need=false, full=false;

    // 2) 找到能让所有方向都 >= eps 的最小 beta
    #pragma unroll
    for (int k=0;k<Q;++k){
        const int off = findindex_distfun_gpu(x,y,k);
        const double fk = fout[off];

        if (!isfinite(fk)) { need=true; full=true; break; }
        if (fk < eps){
            if (fk < -FULL_T){ need=true; full=true; break; }
            const double denom = feq[k] - fk;
            if (denom > 0.0){
                const double b = (eps - fk) / denom;
                if (b > beta_min) beta_min = b;
                need = true;
            }else{
                need=true; full=true; break;
            }
        }
    }

    // 3) 执行修复（最小必要混合；必要时全热化）
    if (need){
        const double beta = full ? 1.0 : fmin(fmax(beta_min,0.0),1.0);
        #pragma unroll
        for (int k=0;k<Q;++k){
            const int off = findindex_distfun_gpu(x,y,k);
            const double fk = fout[off];
            fout[off] = (1.0 - beta)*fk + beta*feq[k];
        }
        if (ct_any) atomicAdd(ct_any, 1ULL);
        if (full && ct_full) atomicAdd(ct_full, 1ULL);
    }
}


__device__ __forceinline__
void positivity_limiter_A(double* __restrict__ fout,
                          int x,int y,int idx, double rho, const double u[2]){
    // 复用你的原函数：把 A 相计数器塞进去
    positivity_limiter(fout, x,y, rho, u, g_ctA_any, g_ctA_full);
    if (g_hitA) atomicAdd(&g_hitA[idx], 1u); // 可选热度图
}

__device__ __forceinline__
void positivity_limiter_B(double* __restrict__ fout,
                          int x,int y,int idx, double rho, const double u[2]){
    positivity_limiter(fout, x,y, rho, u, g_ctB_any, g_ctB_full);
    if (g_hitB) atomicAdd(&g_hitB[idx], 1u);
}

/* ---------- ① 仅碰撞，不做迁移 ---------- */
__global__ void mrt_collide_two_components_gpu(
        const double* rho_A,
        const double* S_A,  const double* C_A,
        const double* fin_A,      double* fout_A,  double* min_A,      double* mout_A,

        const double* rho_B,
        const double* S_B,   const double* C_B,   // C_B 目前没用，可保留
        const double* fin_B,      double* fout_B,  double* min_B,      double* mout_B,
		const double* ux_host, const double* uy_host,   int* pointsflag )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;

    /* ------------ A 相 ------------ */
    {
        int idx = findindex_scalar_gpu(x,y);
		if (pointsflag[idx] >= 0) {
			double u_all[2] = { ux_host[idx], uy_host[idx] };

			/* fin → min */
			#pragma unroll
			for (int k=0;k<Q;++k){
				double mk = 0.0;
				#pragma unroll
				for (int kk=0; kk<Q; ++kk){
					mk += M_gpu[k][kk] * fin_A[ findindex_distfun_gpu(x,y,kk) ];
				}
				min_A[ findindex_distfun_gpu(x,y,k) ] = mk;
			}

			/* 碰撞：min → mout */
			#pragma unroll
			for (int k=0;k<Q;++k){
				int idxk = findindex_distfun_gpu(x,y,k);
				double mk    = min_A[idxk];
				double meq_k = meq_gpu(k, rho_A[idx], u_all);
				mout_A[idxk] = mk - A_a_gpu[k]*(mk-meq_k)
							+ (1.0-0.5*A_a_gpu[k])*S_A[idxk]*deltat_gpu
							+                    C_A[idxk]*deltat_gpu;
			}
			/* 逆变换：mout → fout */
			#pragma unroll
			for (int k=0;k<Q;++k){
				double fk = 0.0;
				#pragma unroll
				for (int kk=0; kk<Q; ++kk){
					fk += Minv_gpu[k][kk] *
						mout_A[ findindex_distfun_gpu(x,y,kk) ];
				}
				fout_A[ findindex_distfun_gpu(x,y,k) ] = fk;
			}
            positivity_limiter_A((double*)fout_A, x,y, idx, rho_A[idx], u_all);
		}
    }

    /* ------------ B 相 ------------ */
    {
        int idx = findindex_scalar_gpu(x,y);
		if (pointsflag[idx] >= 0) {
			double u_all[2] = { ux_host[idx], uy_host[idx] };

			#pragma unroll
			for (int k=0;k<Q;++k){
				double mk = 0.0;
				#pragma unroll
				for (int kk=0; kk<Q; ++kk){
					mk += M_gpu[k][kk] * fin_B[ findindex_distfun_gpu(x,y,kk) ];
				}
				min_B[ findindex_distfun_gpu(x,y,k) ] = mk;
			}

			#pragma unroll
			for (int k=0;k<Q;++k){
				int idxk = findindex_distfun_gpu(x,y,k);
				double mk    = min_B[idxk];
				double meq_k = meq_gpu(k, rho_B[idx],  u_all);
				mout_B[idxk] = mk - A_b_gpu[k]*(mk-meq_k)
							+ (1.0-0.5*A_b_gpu[k])*S_B[idxk]*deltat_gpu;
							/* 若要加 C_B，按 A 相同样法即可 */
			}
			#pragma unroll
			for (int k=0;k<Q;++k){
				double fk = 0.0;
				#pragma unroll
				for (int kk=0; kk<Q; ++kk){
					fk += Minv_gpu[k][kk] *
						mout_B[ findindex_distfun_gpu(x,y,kk) ];
				}
				fout_B[ findindex_distfun_gpu(x,y,k) ] = fk;
			}
            positivity_limiter_B((double*)fout_B, x,y, idx, rho_B[idx], u_all);
		}
    }
}

/* ---------- ② 仅迁移，不再计算 ---------- */
__global__ void stream_two_components_gpu(
        double* fin_A, const double* fout_A,
        double* fin_B, const double* fout_B,  int* pointsflag)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;
	int idx = findindex_scalar_gpu(x,y);
	if (pointsflag[idx] == 1) {
		#pragma unroll
		for (int k=0;k<Q;++k){
			int xp = (x - e_gpu[k][0] + NX) % NX;
			int yp = (y - e_gpu[k][1] + NY) % NY;
			fin_A[ findindex_distfun_gpu(x,y,k) ] =fout_A[ findindex_distfun_gpu(xp,yp,k) ];
			fin_B[ findindex_distfun_gpu(x,y,k) ] =fout_B[ findindex_distfun_gpu(xp,yp,k) ];
		}
	}
}


__global__ void sanitize_ghost_and_solid(
    const int* pointsflag,
    double* ux_A, double* uy_A, double* ux_B, double* uy_B,
    double* Fx_mA, double* Fy_mA, double* Fx_mB, double* Fy_mB)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x>=NX || y>=NY) return;
    int idx = NX*y + x;
    if (pointsflag[idx] < 0){               // -1, -2
        ux_A[idx]=uy_A[idx]=0.0;  ux_B[idx]=uy_B[idx]=0.0;
        Fx_mA[idx]=Fy_mA[idx]=0.0; Fx_mB[idx]=Fy_mB[idx]=0.0;
    }
}




/* -----------------------------------------------------------
 *  边界条件（含凹角专门处理）
 *  - 直壁/凸角：ghost→反弹；否则→搬运（原有逻辑）
 *  - 凹角(两条正交ghost)：
 *      NE：反弹 {E(1),N(2),NE(5)}，置零 {NW(6),SE(8)}
 *      NW：反弹 {W(3),N(2),NW(6)}，置零 {NE(5),SW(7)}
 *      SW：反弹 {W(3),S(4),SW(7)}，置零 {NW(6),SE(8)}
 *      SE：反弹 {E(1),S(4),SE(8)}，置零 {NE(5),SW(7)}
 *  - k=0 不动（同以往）
 * ----------------------------------------------------------- */

//—— 4-邻是否为 ghost(-1)
__device__ __forceinline__
void ghost4(int x,int y, const int* __restrict__ flag,
            bool& gE,bool& gW,bool& gN,bool& gS)
{
    int xE=(x+1+NX)%NX, xW=(x-1+NX)%NX;
    int yN=(y+1+NY)%NY, yS=(y-1+NY)%NY;
    gE = (flag[findindex_scalar_gpu(xE,y )] == -1);
    gW = (flag[findindex_scalar_gpu(xW,y )] == -1);
    gN = (flag[findindex_scalar_gpu(x ,yN)] == -1);
    gS = (flag[findindex_scalar_gpu(x ,yS)] == -1);
}

//—— 反弹 & 置零（用你的 findindex_*）
__device__ __forceinline__
void set_bounce(int x,int y,int k,
                double* __restrict__ fin_A, const double* __restrict__ fout_A,
                double* __restrict__ fin_B, const double* __restrict__ fout_B)
{
    const int idxk     = findindex_distfun_gpu(x,y,k);
    const int idxk_opp = findindex_distfun_gpu(x,y,opp_gpu[k]);
    fin_A[idxk] = fout_A[idxk_opp];
    fin_B[idxk] = fout_B[idxk_opp];
}
__device__ __forceinline__
void set_zero(int x,int y,int k,
              double* __restrict__ fin_A, double* __restrict__ fin_B)
{
    const int idxk = findindex_distfun_gpu(x,y,k);
    fin_A[idxk] = 0.0;  fin_B[idxk] = 0.0;
}

/* -----------------------------------------------------------
 * 边界条件（含凹角）：直壁/凸角走“墙→反弹/流体→搬运”，
 * 凹角：沿壁三向反弹，埋藏对角清零
 * ----------------------------------------------------------- */
__global__ void boundary_gpu_1(double*  __restrict__ fin_A,
                             const double* __restrict__ fout_A,
                             double*  __restrict__ fin_B,
                             const double* __restrict__ fout_B,
                             const int* __restrict__ pointsflag)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;

    const int idx = findindex_scalar_gpu(x,y);
    if (pointsflag[idx] != 0) return; // 只处理边界节点

    // 角点检测（看 4-邻是否有两条正交 ghost）
    bool gE,gW,gN,gS; ghost4(x,y,pointsflag,gE,gW,gN,gS);
    const bool conc_NE = (gE && gN);
    const bool conc_NW = (gN && gW);
    const bool conc_SW = (gW && gS);
    const bool conc_SE = (gS && gE);
    const bool is_conc = (conc_NE || conc_NW || conc_SW || conc_SE);

    // 位掩码记录已写方向（第 k 位=1 表示已写）
    unsigned int filledMask = 0u;
    auto mark = [&](int k){ filledMask |= (1u<<k); };

    if (is_conc){
        if (conc_NE){
            set_bounce(x,y,1,fin_A,fout_A,fin_B,fout_B); mark(1); // E
            set_bounce(x,y,2,fin_A,fout_A,fin_B,fout_B); mark(2); // N
            set_bounce(x,y,5,fin_A,fout_A,fin_B,fout_B); mark(5); // NE
            set_bounce(x,y,6,fin_A,fout_A,fin_B,fout_B); mark(6); // NW ← SE  ← 必须 mark！
            set_bounce(x,y,8,fin_A,fout_A,fin_B,fout_B); mark(8); // SE ← NW  ← 必须 mark！
        }else if (conc_NW){
            set_bounce(x,y,3,fin_A,fout_A,fin_B,fout_B); mark(3); // W
            set_bounce(x,y,2,fin_A,fout_A,fin_B,fout_B); mark(2); // N
            set_bounce(x,y,6,fin_A,fout_A,fin_B,fout_B); mark(6); // NW
            set_bounce(x,y,5,fin_A,fout_A,fin_B,fout_B); mark(5); // NE ← SW  ← 必须 mark！
            set_bounce(x,y,7,fin_A,fout_A,fin_B,fout_B); mark(7); // SW ← NE  ← 必须 mark！
        }else if (conc_SW){
            set_bounce(x,y,3,fin_A,fout_A,fin_B,fout_B); mark(3); // W
            set_bounce(x,y,4,fin_A,fout_A,fin_B,fout_B); mark(4); // S
            set_bounce(x,y,7,fin_A,fout_A,fin_B,fout_B); mark(7); // SW
            set_bounce(x,y,6,fin_A,fout_A,fin_B,fout_B); mark(6); // NW ← SE  ← 必须 mark！
            set_bounce(x,y,8,fin_A,fout_A,fin_B,fout_B); mark(8); // SE ← NW  ← 必须 mark！
        }else{ // conc_SE
            set_bounce(x,y,1,fin_A,fout_A,fin_B,fout_B); mark(1); // E
            set_bounce(x,y,4,fin_A,fout_A,fin_B,fout_B); mark(4); // S
            set_bounce(x,y,8,fin_A,fout_A,fin_B,fout_B); mark(8); // SE
            set_bounce(x,y,5,fin_A,fout_A,fin_B,fout_B); mark(5); // NE ← SW  ← 必须 mark！
            set_bounce(x,y,7,fin_A,fout_A,fin_B,fout_B); mark(7); // SW ← NE  ← 必须 mark！
        }
    }


    // —— 通用逻辑：上游为“墙”(flag<0)→反弹；否则→搬运
    #pragma unroll
    for (int k=0; k<Q; ++k){
        if (filledMask & (1u<<k)) continue; // 角点已写的方向不覆盖

        int ip = (x - e_gpu[k][0] + NX) % NX;
        int jp = (y - e_gpu[k][1] + NY) % NY;
        int idx_nb = findindex_scalar_gpu(ip,jp);
        const int nb_flag = pointsflag[idx_nb];

        const int idxk = findindex_distfun_gpu(x,y,k);
        if (nb_flag < 0) { // ★ 关键：ghost/solid/hydrate 都当墙
            const int idxk_opp = findindex_distfun_gpu(x,y,opp_gpu[k]);
            fin_A[idxk] = fout_A[idxk_opp];
            fin_B[idxk] = fout_B[idxk_opp];
        } else {
            const int idxk_nb = findindex_distfun_gpu(ip,jp,k);
            fin_A[idxk] = fout_A[idxk_nb];
            fin_B[idxk] = fout_B[idxk_nb];
        }
    }
}

/* -----------------------------------------------------------
 *  CUDA kernel: 单层“幽灵反弹”边界条件
 *  则将 fin[k] 赋值为 fout[opp[k]]（A、B 两相各做一次）
 * ----------------------------------------------------------- */
__global__ void boundary_gpu(double*  fin_A,
                             const double* fout_A,
                             double*  fin_B,
                             const double*  fout_B,
                             const int* pointsflag)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;
    int idx = findindex_scalar_gpu(x, y);          // scalar-field 索引

    if (pointsflag[idx] == 0)                    // 只处理 flag == 0
    {
        #pragma unroll
        for (int k = 0; k < Q; ++k)
        {
            int ip = (x - e_gpu[k][0] + NX) % NX;
            int jp = (y - e_gpu[k][1] + NY) % NY;
            int idx_nb = findindex_scalar_gpu(ip, jp);

            if (pointsflag[idx_nb] == -1)        // 邻居为 ghost(-1)
            {
                int idxk      = findindex_distfun_gpu(x, y, k);
                int idxk_opp  = findindex_distfun_gpu(x, y, opp_gpu[k]);

                fin_A[idxk] = fout_A[idxk_opp];    // A 相反弹
                fin_B[idxk] = fout_B[idxk_opp];    // B 相反弹
            }
            else{
                int idxk_nb = findindex_distfun_gpu(ip, jp, k);
                int idxk      = findindex_distfun_gpu(x, y, k);
                fin_A[idxk] = fout_A[idxk_nb];      // 保持不变
                fin_B[idxk] = fout_B[idxk_nb];
            }
        }
    }
}

void outputdat(int                       step,
               const std::string&        folder,
               const std::string&        fname,
               const std::string&        title,
               const Fluid_host&          AH,
               const Fluid_host&          BH,
               const Mix_host&           MIX)
{
    namespace fs = std::filesystem;
    fs::create_directories(folder);

    std::ostringstream num;
    num << std::setw(8) << std::setfill('0') << step;
    std::string file = folder + "/" + fname + num.str() + ".dat";

    std::ofstream out(file);
    out << "Title = \"" << title << "\"\n"
        << "VARIABLES = \"X\",\"Y\",\"rho\",\"rhoA\",\"rhoB\",\"P\",\"U\",\"V\", \"flag\"\n"
        << "ZONE T=\"BOX\", I=" << NX << ",J=" << NY << ",F=POINT\n";

    for (int j = 0; j < NY; ++j)
        for (int i = 0; i < NX; ++i)
        {
            int idx = j*NX + i;
            out << i << ' ' << j << ' '
                << MIX.rho[idx]        << ' '   // ρ 混合
                << AH.rho[idx]    << ' '   // ρ_A
                << BH.rho[idx]    << ' '   // ρ_B
                << MIX.pressure[idx]        << ' '   // P
                << MIX.ux[idx]        << ' '   // U
                << MIX.uy[idx]        << ' ' // V
			    << MIX.pointsflag[idx] << '\n'; // flag
        }
    std::cout << "[output] dat step " << step << " completed\n";
}


/***********  outputvtk 保留 3D 框架 ***********/
/***********  outputvtk – 2D / 可平滑切换到 3D ***********/
__host__ void outputvtk(int                     step,
                        const std::string&      folder,
                        const std::string&      fname,
                        const std::string&      title,
                        const Fluid_host&       AH,
                        const Fluid_host&       BH,
                        const Mix_host&         MIX)
{
    namespace fs = std::filesystem;
    fs::create_directories(folder);

    std::ostringstream num;
    num << std::setw(8) << std::setfill('0') << step;
    std::string file = folder + "/" + fname + num.str() + ".vtk";

    std::ofstream vtk(file, std::ios::binary);

    /* ---------- 头部 ---------- */
    constexpr int NZ = 1;                  // 如需 3D 自行改为 NZ
    vtk << "# vtk DataFile Version 3.0\n"
        << title << '\n'
        << "BINARY\n"
        << "DATASET STRUCTURED_POINTS\n"
        << "DIMENSIONS " << NX << ' ' << NY << ' ' << NZ << '\n'
        << "ORIGIN 0 0 0\n"
        << "SPACING 1 1 1\n"
        << "POINT_DATA " << NX * NY * NZ << '\n';

    /* ---------- velocity VECTORS ---------- */
    vtk << "\nVECTORS velocity double\n";
    for (int k = 0; k < NZ; ++k)
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i)
            {
                const std::size_t idx = (static_cast<std::size_t>(k) * NY + j) * NX + i;
                double U = MIX.ux[idx];
                double V = MIX.uy[idx];
                double W = 0.0;                // 如果有 uz，请替换 W
                SwapEnd(U); SwapEnd(V); SwapEnd(W);
                vtk.write(reinterpret_cast<char*>(&U), sizeof(double));
                vtk.write(reinterpret_cast<char*>(&V), sizeof(double));
                vtk.write(reinterpret_cast<char*>(&W), sizeof(double));
            }

    /* ---------- 通用 lambda 写 double 型标量 ---------- */
    auto write_scalar = [&](const std::string& name,
                            const std::vector<double>& vec)
    {
        vtk << "\nSCALARS " << name << " double\nLOOKUP_TABLE default\n";
        for (double v : vec)
        {
            double t = v; SwapEnd(t);
            vtk.write(reinterpret_cast<char*>(&t), sizeof(double));
        }
    };
    /* ---------- 6 个标量字段：与 .dat 顺序一致 ----------*/
    write_scalar("rho",      MIX.rho);      // 混合密度
    write_scalar("rhoA",     AH.rho);       // ρ_A
    write_scalar("ux_A",     AH.ux);       //
    write_scalar("uy_A",     AH.uy);       //
    write_scalar("Fx_mol",   AH.Fx_mol);       // ρ_B
    write_scalar("Fy_mol",   AH.Fy_mol);       // ρ_B
    write_scalar("Fx_ads",   AH.Fx_ads);       // ρ_B
    write_scalar("Fy_ads",   AH.Fy_ads);       // ρ_B
    write_scalar("rhoB",     BH.rho);       // ρ_B
    write_scalar("ux_B",     BH.ux);       //
    write_scalar("uy_B",     BH.uy);       //
    write_scalar("pressure", MIX.pressure); // 压力
    write_scalar("U",        MIX.ux);       // U 分量
    write_scalar("V",        MIX.uy);       // V 分量

    /* ---------- flag：存成 int，头部也写 int ---------- */
    vtk << "\nSCALARS flag int\nLOOKUP_TABLE default\n";
	for (int f : MIX.pointsflag) {
		int t = f;
		SwapEnd_int(t);  // 用模板版本 ✅
		vtk.write(reinterpret_cast<char*>(&t), sizeof(int));  // ✅ 正确匹配类型
	}
	std::streampos pos = vtk.tellp();
	std::cout << "[debug] binary bytes written = " << pos << '\n';
    vtk.close();
    std::cout << "[output] vtk step " << step << " completed\n";
}

#ifdef HYDRATE_ENABLE
// 向已有 .vtk 文件末尾追加水合物场（以 append 模式打开）
__host__ void outputvtk_append_hydrate(const std::string& vtk_path,
                                        const std::vector<double>& T,
                                        const std::vector<double>& Cm,
                                        const std::vector<double>& Vh,
                                        const std::vector<double>& diss_rate,
                                        const std::vector<double>& pore_origin)
{
    std::ofstream vtk(vtk_path, std::ios::binary | std::ios::app);
    if (!vtk) return;

    auto write_scalar = [&](const std::string& name, const std::vector<double>& vec)
    {
        vtk << "\nSCALARS " << name << " double\nLOOKUP_TABLE default\n";
        for (double v : vec) { double t = v; SwapEnd(t); vtk.write(reinterpret_cast<char*>(&t), sizeof(double)); }
    };
    write_scalar("temperature",  T);
    write_scalar("concentration", Cm);
    write_scalar("hydrate_Vh",   Vh);
    write_scalar("diss_rate",    diss_rate);
    write_scalar("pore_origin",  pore_origin);
}
#endif  // HYDRATE_ENABLE

/***********  大小端翻转工具保持不变 ***********/

__host__ void SwapEnd(double& v)
{
    char* p = reinterpret_cast<char*>(&v);
    for (std::size_t i = 0; i < sizeof(double)/2; ++i)
        std::swap(p[i], p[sizeof(double)-1-i]);
}

__host__ void SwapEnd_int(int& v)
{
    char* p = reinterpret_cast<char*>(&v);
    for (std::size_t i = 0; i < sizeof(int)/2; ++i)
        std::swap(p[i], p[sizeof(int)-1-i]);
}
