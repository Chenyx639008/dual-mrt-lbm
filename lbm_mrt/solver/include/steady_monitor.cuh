//steady_monitor.cuh
#pragma once
#include "LBM.h"
#include "unified_cuda_error_check.cuh"
#include <vector>
#include <unordered_map>

// ========== Limiter monitor: device globals (declarations) ==========
extern __device__ unsigned long long *g_ctA_any;
extern __device__ unsigned long long *g_ctA_full;
extern __device__ unsigned long long *g_ctB_any;
extern __device__ unsigned long long *g_ctB_full;
// 可选热度图（可为 nullptr）
extern __device__ unsigned int *g_hitA;
extern __device__ unsigned int *g_hitB;
// 流体格点总数（只读）
extern __device__ unsigned long long g_Nfluid;



struct LimiterStats {
    unsigned long long A_any=0, A_full=0, B_any=0, B_full=0;
    double rA_any=0, rA_full=0, rB_any=0, rB_full=0; // 占比
};



struct SteadyMonitor {
    // —— 稳态监测（保留，与你主程序已有流程一致）——
    int    interval    = 5000;
    int    need_consec = 3;
    double tol_rel     = 1e-3;
    double tol_abs     = 1e-9;

    int    consec_hit  = 0;
    bool   has_ref     = false;// 是否已有参考值
    double QA_ref = 0.0, QB_ref = 0.0, QT_ref = 0.0;

    // —— 设备端：仅保留体流量规约用缓存 + 占优掩码 —— 
    double *d_sumA = nullptr, *d_sumB = nullptr, *d_sumQT = nullptr;
    unsigned char *d_maskA_dom = nullptr, *d_maskB_dom = nullptr; // 互斥占优掩码

    // —— 锁死判据：只保留你需要的两个阈值 —— 
    // 1) 互斥占优阈值：sB = rhoB/(rhoA+rhoB) ≥ dom_ratio_thr → B 占优；≤1-dom_ratio_thr → A 占优
    double dom_ratio_thr   = 0.5;
    // 统计辅助
    int nFluid = -1;


    // ---- limiter 监控配置 ----
    // ---- limiter 监控配置（默认只写文件，不打控制台）----
    int         limiter_window = 5000;          // 统计窗口长度（步）
    int         limiter_log_every = 0;          // 写文件频率；0 表示跟随窗口
    bool        limiter_stdout = false;         // 是否向控制台打印
    bool        limiter_log_on_hit = false;     // 命中阈值时是否“即时”也写一行
    double      limiter_log_thr = 1e9;          // 命中阈值（若 limiter_log_on_hit=true 才有意义）
    std::string limiter_log_path = "data/file/limiter_log.csv";
    mutable bool limiter_csv_header_written = false;
    unsigned long long h_Nfluid   = 0;                 // 流体格点数（作为分母）


    // 设备端计数器（A/B × any/full）
    unsigned long long *d_ctA_any  = nullptr, *d_ctA_full = nullptr;
    unsigned long long *d_ctB_any  = nullptr, *d_ctB_full = nullptr;


    SteadyMonitor() = default;
    ~SteadyMonitor();
    SteadyMonitor(const SteadyMonitor&) = delete;
    SteadyMonitor& operator=(const SteadyMonitor&) = delete;

    // 稳态监测
    void   init_device_buffers();
    void   reset();
    void   compute_Q_GPU(const Fluid_dev& A_dev, const Fluid_dev& B_dev,
                         const Mix_dev& mix_dev, double& QA, double& QB,
                         double& QT) const;                     // ★ 改：增加 QT 出参
    bool   compare_and_update(double QA, double QB, double QT, int step);

    // ── SCMP single-phase flow rate (no B-phase, no dominance mask) ──
    void   compute_Q_scmp(const double* ux, const double* rho,
                          const int* pointsflag,
                          double& Q, int nTot) const;
    bool   compare_and_update_single(double Q, int step);

    // 一次性准备
    void   prepare_domain(const Mix_dev& mix_dev);

    void init_limiter_monitor(const int* d_pointsflag);
    void reset_limiter_counters(cudaStream_t s=0) const;
    LimiterStats fetch_limiter_stats_and_log(int step,
                                            double print_thr=-1.0,
                                            bool write_file=true) const;

private:
    // 计算 key：用“重心(像素级取整)+面积”生成一个稳健的64位键
    static uint64_t cluster_key(long long area, double cx, double cy);
};


