// sim_utils.h
#pragma once
#include <string>
#include "LBM.h"
#include "steady_monitor.cuh"
#ifdef HYDRATE_ENABLE
#include "hydrate.h"
#endif

// 统一的运行参数：把 params.txt 与本次 run 强相关的“动态常量”集中起来
struct RuntimeParams {
  // 几何/赋存形态
  int    init_eq = 1;            // 0=手动；1/2=使用预设组
  int    drive   = 1;            // 预留（与 drive_mode 概念不同，先占位）

  int    morph = 2;                    // 1=pore-fill, 2=coating, 3=mixed
  double r_obs = 20.0, l_gap = 20.0;
  double coat_thick = 6.0, r_mid = 6.0;

  // 初始/润湿
  double Sw = 0.3;
  unsigned long long water_seed = 1234567ULL;
  double thetaA_quartz_deg = 30.0, thetaA_hydrate_deg = 80.0;
  double GBw_quartz = 0, GBw_hydrate = 0;
  // —— 接触角 → GAw 的线性映射：GAw = m * (thetaA - c) ——
  double GAw_m = 1.0 / 456.69;   // 默认（旧公式）的斜率
  double GAw_c = 86.41;          // 默认（旧公式）的中心值 theta*


  // 驱动/物性（注意：tau_p_* 应 > 0.5）
  double Gx = 1e-5, Gy = 0.0; int drive_mode = 1;
  double rhoA_hi = 6.6293, rhoA_lo = 0.34127;
  double rhoB_hi = 0.3823,  rhoB_lo = 0.0001;
    // —— Sw 极端时（0 或 1）使用的密度覆盖 —— //
  // Sw=0 : A取 rhoA_in_gas_0, B取 rhoB_in_gas_0
  // Sw=1 : A取 rhoA_in_liq_1, B取 rhoB_in_liq_1
  double rhoA_ini_h_1 = 7.2243;
  double rhoA_ini_l_0 = 0.0;
  double rhoB_ini_l_1 = 0.0;
  double rhoB_ini_h_0 = 0.2363;


  double tau_p_a = 1.0, tau_p_b = 1;

  // 界面/相互作用新增
  double GAB   = 0.24;
  double GBA   = 0.24;
  double sigmaA= 0.11;
  double kappa = 0.6;

  // 断点控制
  int CP_EVERY = 50000;     // 每多少步存一次档；建议 eq 阶段 5000~20000，flow 阶段 10000+
  int CP_KEEP  = 2;     // 保留最近 N 份
  int CP_RESUME = 1;    // 尝试从最新档恢复（1=是，0=否）

  // ===== 总阀门（新增）=====
  bool ENABLE_CKPT       = true;  // 一键控制 checkpoint 的读/写

  // 运行目录（统一进入参数源，避免文件级全局状态）
  std::string ckpt_dir = "data/ckpt";
  std::string file_dir = "data/file";

  // 统一输出频率（避免散落硬编码）
  int OUTPUT_EVERY = static_cast<int>(NOUTPUT);

  // 两阶段稳态判据（避免 run_equilibrate_then_flow 内硬编码）
  double eq_tol_rel = 1e-4;
  int    eq_need_consec = 2;
  int    eq_max_steps = 200000;
  double eq_q_abs_eps = 1e-6;

  double flow_tol_rel = 1e-4;
  int    flow_need_consec = 3;
  int    flow_max_steps = static_cast<int>(NSTEPS);

  // 参数来源说明（用于“最终生效值与来源”打印）
  std::string source_note = "defaults";

  // 新增：几何文件名（若非空则从 Tecplot 读几何）
  std::string geom_file = "";          // 例如 "geometry_case001.plt"

#ifdef HYDRATE_ENABLE
  // ===== 水合物相变扩展参数 =====

  // 开关
  bool   hydrate_enable      = false;  // 运行时开关（compile + runtime 双重控制）
  int    hydrate_start_step  = 0;      // 从第几步开始激活相变（0=从一开始）

  // 热场（DDF 热 LBM）
  double T0_init             = 278.15; // 初始均匀温度 (K)
  double T0_inlet            = 285.0;  // 入口/顶部固定温度边界 (K)
  double lambda_fluid        = 0.6;    // 水相热导率 W/(m·K)
  double lambda_hydrate      = 0.49;   // 水合物热导率 W/(m·K)
  double lambda_solid        = 0.9;    // 石英颗粒热导率 W/(m·K)
  double rhocp_fluid         = 4.2e6;  // 水相 ρcp J/(m³·K)
  double rhocp_hydrate       = 2.1e6;  // 水合物 ρcp J/(m³·K)
  double rhocp_solid         = 2.0e6;  // 石英 ρcp J/(m³·K)

  // 浓度场（CST 质量传递）
  double D_mol_water         = 1.85e-9;// 甲烷在水中扩散系数 m²/s
  double Henry_KH            = 0.1;    // 无量纲 Henry 常数 (Cw = KH * Cg)
  double Cm_init             = 0.0;    // 初始溶解浓度（格子单位）

  // 反应动力学（Kim-Bishnoi）
  double k0_rxn              = 3.6e4;  // 前指数因子 mol/(m²·s·Pa)
  double Ea_rxn              = 9.75e4; // 活化能 J/mol
  double e1_peq              = 33.12;  // 平衡压力经验常数（Yang 2024 Eq.S29）
  double e2_peq              = -9005.5;// 平衡压力经验常数 (K)
  double latent_heat         = 4.3e4;  // 分解潜热 J/mol（吸热为正）

  // VOP 固相动态更新
  double Vm_hydrate          = 2.274e-5;// 水合物摩尔体积 m³/mol
  double Vh_init             = 1.0;    // 初始水合物体积分数 [0,1]
  double vop_terminate_frac  = 0.01;   // 水合物剩余分数低于此值时停止模拟

  // 物理单位换算基准
  double dx_phys             = 1.0e-5; // 格子物理尺寸 (m)，默认 10 μm
  double dt_phys             = 1.0e-6; // 物理时间步 (s)
#endif
};
// 运行阶段配置,控制加力与否
struct StageConfig {
    double drive_scale       = 1.0;    // 0: 关驱动；1: 开驱动
    double tol_rel           = 1e-3;   // compare_and_update 的相对阈值
    int    need_consec       = 3;      // 连续命中次数
    int    max_steps         = 100000; // 阶段内最大步数
    bool   require_abs_quiet = false;  // 是否要求 |Qmix| 很小（无驱动阶段建议 true）
    double q_abs_eps         = 1e-6;  // “绝对静止”阈值
    const char* tag          = "flow"; // 输出前缀标识：eq / flow

};
// —— 断点存取的头部（元数据） —— //
struct CheckpointHeader {
  char magic[8];     // "LBMCPv1"
  int  nx, ny, q;    // q=9 for D2Q9
  int  stage;        // 0=eq, 1=flow
  int  step;         // 保存时刻步数
  int  dtype_size;   // sizeof(double)
  unsigned long long params_hash; // 运行参数哈希（可选）
  unsigned long long geom_hash;   // 几何哈希（可选）
};

/* --------- I/O & 参数 --------- */
RuntimeParams load_params_txt(const std::string& path,
                              const RuntimeParams& defaults = RuntimeParams{});
void init_run_dirs_from_env();
void          print_gpu_banner();
void          print_params_summary(const RuntimeParams& p);
void          set_drive_scale(double s);

/* --------- 设备侧常量 + 润湿性 + 几何 --------- */
void push_wettability_and_maps(const RuntimeParams& p); // upload_wettability_table + alloc wall maps
void push_device_constants(const RuntimeParams& p);     // d_* + A_a_gpu/A_b_gpu（由 tau_p_* 组装）
void build_and_upload_geometry(const RuntimeParams& p, Porous_host& porous);
void build_and_upload_geometry_from_tecplot(const RuntimeParams& p, Mix_dev& M);
/* --------- 分配/释放 --------- */
void allocate_all(Fluid_dev& A, Fluid_dev& B, Mix_dev& M);
void free_all    (Fluid_dev& A, Fluid_dev& B, Mix_dev& M);



// 断点 //
bool save_checkpoint(const char* dir, const char* tag, int step,
                     const Fluid_dev& A, const Fluid_dev& B,
                     const Mix_dev&   M,
                     const RuntimeParams& P);

bool load_checkpoint(const char* dir, const char* tag, int& step_out,
                     Fluid_dev& A, Fluid_dev& B,
                     Mix_dev&   M,
                     const RuntimeParams& P);

void rotate_checkpoints(const char* dir, const char* tag, int keep);
/* --------- 运行循环 --------- */

struct RunResult {
  // 原有
  bool steady=false; int steady_step=-1, last_probe_step=-1;
  double QA=0.0, QB=0.0, Qmix=0.0;

  // 精简版：阶段1 & 断点最小信息
  bool eq_skipped=false;       // 找到 eq 存档后跳过阶段1
  bool eq_steady=false;        // 若跑过阶段1，是否判稳
  int  eq_steady_step=-1;      // 阶段1稳态步（若有）
  int  eq_ckpt_step=-1;        // 保存/用于基线的 eq 存档步（若有）

  bool flow_resumed=false;     // flow 阶段是否从断点恢复
  int  flow_resume_step=-1;    // 恢复自哪一步（若有）

  // 供内部传递用（不在摘要里输出也行）
  bool resumed=false;          // run_stage 内部：本阶段是否恢复
  int  resumed_step=-1;        // run_stage 内部：恢复步

#ifdef HYDRATE_ENABLE
  // ===== 水合物诊断 =====
  double hydrate_volume_frac = 1.0;   // Σ Vh / Σ Vh_init（1.0=未分解，0=全部分解）
  double Q_dissociation      = 0.0;   // Σ diss_rate（水合物表面节点求和）
  int    n_converted_total   = 0;     // 累计翻转节点数（hydrate→fluid）
#endif
};

RunResult run_time_loop(Fluid_dev& A, Fluid_host& AH,
                        Fluid_dev& B, Fluid_host& BH,
                        Mix_dev& M, Mix_host& MH,
                        SteadyMonitor& SM);

RunResult run_stage(Fluid_dev& A, Fluid_host& AH,
                    Fluid_dev& B, Fluid_host& BH,
                    Mix_dev&   MX, Mix_host&   MH,
                    SteadyMonitor& SM,
                    const StageConfig& cfg,
                    const RuntimeParams& P);

#ifdef HYDRATE_ENABLE
// 水合物宿主端缓冲（用于 VTK 输出）
struct HydrateHost {
    std::vector<double> T;          // 温度场 [NX*NY]
    std::vector<double> Cm;         // 溶解浓度 [NX*NY]
    std::vector<double> Vh;         // 水合物体积分数 [NX*NY]
    std::vector<double> diss_rate;  // 分解速率 [NX*NY]
    std::vector<double> pore_origin;// 诊断：1=分解释放孔隙，0=原生孔隙 [NX*NY]
    HydrateHost()
    : T(NX*NY, 0.0), Cm(NX*NY, 0.0),
      Vh(NX*NY, 0.0), diss_rate(NX*NY, 0.0), pore_origin(NX*NY, 0.0) {}
};

// 水合物扩展版 run_stage（含耦合物理循环）
RunResult run_stage_hydrate(Fluid_dev& A, Fluid_host& AH,
                            Fluid_dev& B, Fluid_host& BH,
                            Mix_dev&   MX, Mix_host&   MH,
                            Therm_dev& TH, Conc_dev& CN,
                            VOP_dev&   VP, HydrateHost& HH,
                            SteadyMonitor& SM,
                            const StageConfig& cfg,
                            const RuntimeParams& P);
#endif

RunResult run_equilibrate_then_flow(Fluid_dev& A, Fluid_host& AH,
                                    Fluid_dev& B, Fluid_host& BH,
                                    Mix_dev&   M, Mix_host&   MH,
                                    SteadyMonitor& SM,
                                    const RuntimeParams& P);


/* --------- 摘要输出 --------- */
void write_run_summary(const RunResult& R, int interval, const RuntimeParams& P);
