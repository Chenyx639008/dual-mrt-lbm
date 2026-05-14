// main.cu 文件是程序的入口点，负责初始化、执行和清理工作
//9.1坂本拟加入，断点处理

#include <iomanip>
#include <string>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <chrono>
#include <map>
#include "../include/unified_cuda_error_check.cuh"
#include "../include/LBM.h"
#include "../include/steady_monitor.cuh"
#include "../include/sim_utils.h"
using namespace std::chrono;

// LBM.cu 中定义的几何 kernel（通过 -rdc=true 跨文件可见）
extern __global__ void mark_fluid_solid(int* pointsflag);
extern __global__ void mark_boundary(int* pointsflag);
extern __global__ void mark_ghost(int* pointsflag);

// 线程网格默认配置（大多核用这个；特殊核可另行覆盖）
const int nThreadsx = 32, nThreadsy = 1;// CUDA 线程配置，以后可修改为16，16
dim3 threads(nThreadsx,nThreadsy,1);
dim3 grid((NX+nThreadsx-1)/nThreadsx,(NY+nThreadsy-1)/nThreadsy,1);
double* _nan_chk_buf_host = (double*)malloc(sizeof(double));
double* _nan_chk_probe_dev = nullptr;   // 稍后指向 rhoA_d


int main(int argc, char** argv){
    // 0) 选设备并打印 GPU 能力/显存，便于运行前容量与配置确认
    checkCudaErrors(cudaSetDevice(0));
    print_gpu_banner();

    // 1) 读参数：支持 ./app params.txt；未给参数则使用 RuntimeParams 默认值
    RuntimeParams P;
    if (argc >= 2) {
        P = load_params_txt(argv[1], P);
    } else if (std::filesystem::exists("params.txt")) {
        P = load_params_txt("params.txt", P);
    } else {
        if (const char* e = std::getenv("LBM_CKPT_DIR"); e && *e) P.ckpt_dir = e;
        if (const char* e = std::getenv("LBM_FILE_DIR"); e && *e) P.file_dir = e;
        P.source_note = "defaults + env(LBM_*_DIR)";
    }
    print_params_summary(P);

    // ── Huang & Wu (2016) SCMP dispatch ──
    // mcmp_huang_256 always runs SCMP; mcmp_sim always runs legacy MCMP.
#ifdef HUANG_256_BUILD
    {
        const char* params_path = (argc >= 2) ? argv[1] : "params.txt";
        run_scmp_huang(P, params_path);
    }
    return 0;
#endif

    // 2) 设备/主机容器：Host 侧用于输出与可视化，Device 侧用于计算
    Fluid_dev A_dev,B_dev; Mix_dev M_dev;
    Fluid_host AH,BH; Mix_host MH;
    allocate_all(A_dev,B_dev,M_dev);
    _nan_chk_probe_dev = A_dev.rho;


    // 3) 一次性下发：润湿查表与墙面图、__constant__ 常量、几何/材质图
    push_wettability_and_maps(P);   // Host 上构建并准备好墙/润湿映射
    init_device_variable();        //  PR-EOS 常数、MRT 矩阵等静态项（不随 run 改变）
    init_run_dirs_from_env();
    push_device_constants(P);      //  本次 run 的动态常量（驱动、ρ_init、τ、κ 等）到 __constant__

    Porous_host porous;
    if (P.geom_file.empty()) {
        // 解析几何：build_circle_array 生成颗粒阵列 → upload d_obs → mark_fluid_solid 填充 pointsflag
        build_and_upload_geometry(P, porous);
        mark_fluid_solid<<<grid, threads>>>(M_dev.pointsflag);
        CUDA_CHECK(cudaGetLastError());
        mark_boundary<<<grid, threads>>>(M_dev.pointsflag);
        CUDA_CHECK(cudaGetLastError());
        mark_ghost<<<grid, threads>>>(M_dev.pointsflag);
        CUDA_CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
    } else {
        // Tecplot 几何：从 .plt 文件读 flag 后再分类 ghost/边界
        build_and_upload_geometry_from_tecplot(P, M_dev);
        init_geometry(M_dev.pointsflag);
    }
    init_all(M_dev, A_dev, B_dev);      // A/B 两相与混合场初始化（ρ/ψ/fin 等）
    dbg_consts_once<<<1,1>>>();         // 可选：在设备侧打印/校验一次常量（调试用）
    cudaDeviceSynchronize();
     // 4) 监测器：准备稳态与局部锁死检测需要的缓冲与拓扑信息
    SteadyMonitor SM;
    SM.init_device_buffers();
    SM.prepare_domain(M_dev);            // 例如入口/出口截面掩码、BFS 缓冲等
    SM.init_limiter_monitor(M_dev.pointsflag);
    SM.limiter_log_every = SM.interval;     // 同频
    SM.limiter_log_thr   = 1e-3;            // 触发阈值（按需改）
    SM.limiter_stdout    = false;           // 禁止控制台打印
    SM.limiter_log_path  = P.file_dir + "/limiter_log.csv";

#ifdef HYDRATE_ENABLE
    // 5-H) 水合物模式初始化（仅 hydrate_enable 时有意义；alloc 始终执行以保持统一生命周期）
    Therm_dev TH_dev;  Conc_dev CN_dev;  VOP_dev VP_dev;
    HydrateHost HH;
    if (P.hydrate_enable) {
        init_device_variable_hydrate(P);
        alloc_therm(TH_dev);
        alloc_conc(CN_dev);
        alloc_vop(VP_dev);
        init_thermal_field(TH_dev, M_dev.pointsflag,
                           P.thermal_init_mode, P.T0_inlet, P.thermal_bc_side);
        init_conc_field(CN_dev, M_dev.pointsflag);
        init_vop(VP_dev, M_dev.pointsflag, P.Vh_init);
        printf("[hydrate] 水合物场初始化完成。\n");
    }
#endif

    // 5) 时间推进：把推进、监测、输出收敛前的所有循环交给 run_time_loop
    auto t1 = high_resolution_clock::now();

#ifdef HYDRATE_ENABLE
    RunResult R = [&]() -> RunResult {
        if (P.hydrate_enable) {
            // 阶段1：phase separation（无驱动，水合物场不参与）
            StageConfig eq;
            eq.tag = "eq"; eq.drive_scale = 0.0;
            eq.tol_rel = P.eq_tol_rel; eq.need_consec = P.eq_need_consec;
            eq.max_steps = P.eq_max_steps; eq.require_abs_quiet = false;
            eq.q_abs_eps = P.eq_q_abs_eps;
            RunResult R_eq = run_stage(A_dev, AH, B_dev, BH, M_dev, MH, SM, eq, P);

            // 阶段2：驱动流 + 水合物物理
            StageConfig flow;
            flow.tag = "flow"; flow.drive_scale = 1.0;
            flow.tol_rel = P.flow_tol_rel; flow.need_consec = P.flow_need_consec;
            flow.max_steps = P.flow_max_steps; flow.require_abs_quiet = false;
            flow.q_abs_eps = P.eq_q_abs_eps;
            RunResult Rf = run_stage_hydrate(A_dev, AH, B_dev, BH, M_dev, MH,
                                              TH_dev, CN_dev, VP_dev, HH,
                                              SM, flow, P);
            Rf.eq_steady = R_eq.steady; Rf.eq_steady_step = R_eq.steady_step;
            return Rf;
        }
        return run_equilibrate_then_flow(A_dev, AH, B_dev, BH, M_dev, MH, SM, P);
    }();
#else
    RunResult R = run_equilibrate_then_flow(A_dev, AH, B_dev, BH, M_dev, MH, SM, P);
#endif

    auto t2 = high_resolution_clock::now();
    printf("Total time = %.3f s\n", duration<double>(t2-t1).count());

    // 6) 写摘要，便于批量跑完自动抓取指标
    //write_run_summary(R, SM.interval, P);

    // 7) 释放资源并复位设备
    free_all(A_dev,B_dev,M_dev);
#ifdef HYDRATE_ENABLE
    if (P.hydrate_enable) {
        free_therm(TH_dev);
        free_conc(CN_dev);
        free_vop(VP_dev);
    }
#endif
    cudaDeviceReset();
    return 0;
}
