// sim_utils.cu
#include "../include/sim_utils.h"
#include "../include/unified_cuda_error_check.cuh"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cctype>
#include <sstream>
#include <map>
#include <cstdio>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/types.h>
using namespace std;

// 从环境变量注入（在 main 或初始化处调用一次）
void init_run_dirs_from_env() {
    // 为兼容保留该函数，目录逻辑已迁移到 RuntimeParams。
}



#define CK(call) do{ \
  cudaError_t _e=(call); \
  if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s failed @ %s:%d : %s\n", \
      #call,__FILE__,__LINE__,cudaGetErrorString(_e)); \
    exit(1); \
  } \
}while(0)

void set_drive_scale(double s){
  CK(cudaMemcpyToSymbol(d_drive_scale, &s, sizeof(double)));
}

/* ============ 读参工具：解析 key value 文本为 map ============
static map<string,double> parse_kv(const string& path){
  ifstream in(path);
  map<string,double> m; string k; double v;
  while (in>>k>>v) m[k]=v; return m;
}
*/

static std::map<std::string,double> parse_kv(const std::string& path){
    std::ifstream in(path);
    std::map<std::string,double> m;
    if(!in){ std::perror(("open "+path).c_str()); return m; }

    std::string line;

    auto notspace = [](unsigned char ch){ return !std::isspace(ch); };

    auto ltrim = [&](std::string& s){
        s.erase(s.begin(),
                std::find_if(s.begin(), s.end(),
                             [&](char ch){ return notspace(static_cast<unsigned char>(ch)); }));
    };
    auto rtrim = [&](std::string& s){
        s.erase(
            std::find_if(s.rbegin(), s.rend(),
                         [&](char ch){ return notspace(static_cast<unsigned char>(ch)); }).base(),
            s.end());
    };
    auto trim = [&](std::string& s){ ltrim(s); rtrim(s); };

    while (std::getline(in, line)) {
        // 去掉注释
        size_t p = line.find('#');  if (p!=std::string::npos) line.erase(p);
        p = line.find("//");        if (p!=std::string::npos) line.erase(p);
        trim(line);
        if (line.empty()) continue;

        // "key value" 或 "key=value"
        std::string k, vstr;
        p = line.find('=');
        if (p != std::string::npos) {
            k = line.substr(0, p);
            vstr = line.substr(p+1);
            trim(k); trim(vstr);
        } else {
            std::istringstream iss(line);
            if (!(iss >> k >> vstr)) continue;
        }
        try {
            m[k] = std::stod(vstr);
        } catch (...) {
            std::fprintf(stderr, "[params] skip invalid: %s = %s\n", k.c_str(), vstr.c_str());
        }
    }
    return m;
}



// 保证目录存在
static void ensure_dir(const char* dir){
    struct stat st{};
    if (stat(dir, &st) != 0) {
        #ifdef _WIN32
        _mkdir(dir);
        #else
        mkdir(dir, 0755);
        #endif
    }
}

static void write_all(std::ofstream& out, const void* buf, size_t n){
    out.write(reinterpret_cast<const char*>(buf), n);
    if (!out) { throw std::runtime_error("write failed"); }
}
static void read_all(std::ifstream& in, void* buf, size_t n){
    in.read(reinterpret_cast<char*>(buf), n);
    if (!in) { throw std::runtime_error("read failed"); }
}

static inline unsigned long long fnv1a64(const void* data, size_t n,
                                         unsigned long long h = 1469598103934665603ULL){
    const unsigned char* p = reinterpret_cast<const unsigned char*>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= static_cast<unsigned long long>(p[i]);
        h *= 1099511628211ULL;
    }
    return h;
}

static unsigned long long hash_runtime_params(const RuntimeParams& p){
    unsigned long long h = 1469598103934665603ULL;
    auto mix = [&](const auto& v){ h = fnv1a64(&v, sizeof(v), h); };
    mix(p.init_eq); mix(p.morph);
    mix(p.r_obs); mix(p.l_gap); mix(p.coat_thick); mix(p.r_mid);
    mix(p.Sw); mix(p.water_seed); mix(p.thetaA_quartz_deg); mix(p.thetaA_hydrate_deg);
    mix(p.GBw_quartz); mix(p.GBw_hydrate); mix(p.GAw_m); mix(p.GAw_c);
    mix(p.Gx); mix(p.Gy); mix(p.drive_mode);
    mix(p.rhoA_hi); mix(p.rhoA_lo); mix(p.rhoB_hi); mix(p.rhoB_lo);
    mix(p.rhoA_ini_h_1); mix(p.rhoA_ini_l_0); mix(p.rhoB_ini_l_1); mix(p.rhoB_ini_h_0);
    mix(p.tau_p_a); mix(p.tau_p_b); mix(p.GAB); mix(p.GBA); mix(p.sigmaA); mix(p.kappa);
    mix(p.CP_EVERY); mix(p.CP_KEEP); mix(p.CP_RESUME); mix(p.ENABLE_CKPT);
    mix(p.OUTPUT_EVERY); mix(p.eq_tol_rel); mix(p.eq_need_consec); mix(p.eq_max_steps); mix(p.eq_q_abs_eps);
    mix(p.flow_tol_rel); mix(p.flow_need_consec); mix(p.flow_max_steps);
    mix(p.epsilon_huang); mix(p.k2_huang);
    h = fnv1a64(p.geom_file.data(), p.geom_file.size(), h);
    return h;
}

static unsigned long long hash_device_geometry(const Mix_dev& M){
    std::vector<int> host_flag(NX*NY);
    CK(cudaMemcpy(host_flag.data(), M.pointsflag, sizeof(int)*NX*NY, cudaMemcpyDeviceToHost));
    return fnv1a64(host_flag.data(), host_flag.size()*sizeof(int));
}

struct FluidSnapshot {
    std::vector<double> rho, ux, uy, psi, pressure, Fx_mol, Fy_mol, Fx_ads, Fy_ads;
    std::vector<double> fin;
};

static void resize_snapshot(FluidSnapshot& s){
    const size_t n = static_cast<size_t>(NX) * NY;
    s.rho.resize(n); s.ux.resize(n); s.uy.resize(n); s.psi.resize(n); s.pressure.resize(n);
    s.Fx_mol.resize(n); s.Fy_mol.resize(n); s.Fx_ads.resize(n); s.Fy_ads.resize(n);
    s.fin.resize(n * Q);
}

static void copy_fluid_from_device(const Fluid_dev& F, FluidSnapshot& s){
    resize_snapshot(s);
    const size_t n = static_cast<size_t>(NX) * NY;
    CK(cudaMemcpy(s.rho.data(), F.rho, sizeof(double)*n, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(s.ux.data(), F.ux, sizeof(double)*n, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(s.uy.data(), F.uy, sizeof(double)*n, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(s.psi.data(), F.psi, sizeof(double)*n, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(s.pressure.data(), F.pressure, sizeof(double)*n, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(s.Fx_mol.data(), F.Fx_mol, sizeof(double)*n, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(s.Fy_mol.data(), F.Fy_mol, sizeof(double)*n, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(s.Fx_ads.data(), F.Fx_ads, sizeof(double)*n, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(s.Fy_ads.data(), F.Fy_ads, sizeof(double)*n, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(s.fin.data(), F.fin, sizeof(double)*n*Q, cudaMemcpyDeviceToHost));
}

static void copy_fluid_to_device(const FluidSnapshot& s, Fluid_dev& F){
    const size_t n = static_cast<size_t>(NX) * NY;
    CK(cudaMemcpy(F.rho, s.rho.data(), sizeof(double)*n, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(F.ux, s.ux.data(), sizeof(double)*n, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(F.uy, s.uy.data(), sizeof(double)*n, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(F.psi, s.psi.data(), sizeof(double)*n, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(F.pressure, s.pressure.data(), sizeof(double)*n, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(F.Fx_mol, s.Fx_mol.data(), sizeof(double)*n, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(F.Fy_mol, s.Fy_mol.data(), sizeof(double)*n, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(F.Fx_ads, s.Fx_ads.data(), sizeof(double)*n, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(F.Fy_ads, s.Fy_ads.data(), sizeof(double)*n, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(F.fin, s.fin.data(), sizeof(double)*n*Q, cudaMemcpyHostToDevice));
}

bool save_checkpoint(const char* dir, const char* tag, int step,
                     const Fluid_dev& A, const Fluid_dev& B,
                     const Mix_dev&   M,
                     const RuntimeParams& P)
{
    ensure_dir(dir);
    char path[256]; std::snprintf(path,sizeof(path),"%s/ckpt_%s_%08d.bin", dir, tag, step);
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;

    CheckpointHeader H{};
    std::memcpy(H.magic, "LBMCPv1", 7);
    H.nx = NX; H.ny = NY; H.q = 9; H.dtype_size = sizeof(double);
    H.stage = (std::string(tag)=="eq")?0:1;
    H.step  = step;
    H.params_hash = hash_runtime_params(P);
    H.geom_hash   = hash_device_geometry(M);

    write_all(out, &H, sizeof(H));

    FluidSnapshot sA, sB;
    copy_fluid_from_device(A, sA);
    copy_fluid_from_device(B, sB);

    auto write_snap = [&](const FluidSnapshot& s){
        write_all(out, s.rho.data(),      s.rho.size()*sizeof(double));
        write_all(out, s.ux.data(),       s.ux.size()*sizeof(double));
        write_all(out, s.uy.data(),       s.uy.size()*sizeof(double));
        write_all(out, s.psi.data(),      s.psi.size()*sizeof(double));
        write_all(out, s.pressure.data(), s.pressure.size()*sizeof(double));
        write_all(out, s.Fx_mol.data(),   s.Fx_mol.size()*sizeof(double));
        write_all(out, s.Fy_mol.data(),   s.Fy_mol.size()*sizeof(double));
        write_all(out, s.Fx_ads.data(),   s.Fx_ads.size()*sizeof(double));
        write_all(out, s.Fy_ads.data(),   s.Fy_ads.size()*sizeof(double));
        write_all(out, s.fin.data(),      s.fin.size()*sizeof(double));
    };
    write_snap(sA);
    write_snap(sB);
    out.close();

    // 滚动保留
    rotate_checkpoints(dir, tag, P.CP_KEEP);
    // 覆盖 latest
    char latest[256]; std::snprintf(latest,sizeof(latest), "%s/ckpt_latest_%s.bin", dir, tag);
    std::ofstream out2(latest, std::ios::binary); if(out2){
        std::ifstream in(path, std::ios::binary);
        out2 << in.rdbuf();
    }
    return true;
}

bool load_checkpoint(const char* dir, const char* tag, int& step_out,
                     Fluid_dev& A, Fluid_dev& B,
                     Mix_dev&   M,
                     const RuntimeParams& P)
{
    char latest[256]; std::snprintf(latest,sizeof(latest), "%s/ckpt_latest_%s.bin", dir, tag);
    std::ifstream in(latest, std::ios::binary);
    if (!in) return false;

    CheckpointHeader H{}; read_all(in, &H, sizeof(H));
    if (std::strncmp(H.magic,"LBMCPv1",7)!=0 || H.nx!=NX || H.ny!=NY || H.q!=9 || H.dtype_size!=(int)sizeof(double))
        return false;

    const auto expect_param = hash_runtime_params(P);
    const auto expect_geom  = hash_device_geometry(M);
    if (H.params_hash != expect_param || H.geom_hash != expect_geom) {
        std::fprintf(stderr, "[ckpt] reject %s: hash mismatch (param/geometric).\n", latest);
        return false;
    }

    FluidSnapshot sA, sB;
    resize_snapshot(sA);
    resize_snapshot(sB);
    auto read_snap = [&](FluidSnapshot& s){
        read_all(in, s.rho.data(),      s.rho.size()*sizeof(double));
        read_all(in, s.ux.data(),       s.ux.size()*sizeof(double));
        read_all(in, s.uy.data(),       s.uy.size()*sizeof(double));
        read_all(in, s.psi.data(),      s.psi.size()*sizeof(double));
        read_all(in, s.pressure.data(), s.pressure.size()*sizeof(double));
        read_all(in, s.Fx_mol.data(),   s.Fx_mol.size()*sizeof(double));
        read_all(in, s.Fy_mol.data(),   s.Fy_mol.size()*sizeof(double));
        read_all(in, s.Fx_ads.data(),   s.Fx_ads.size()*sizeof(double));
        read_all(in, s.Fy_ads.data(),   s.Fy_ads.size()*sizeof(double));
        read_all(in, s.fin.data(),      s.fin.size()*sizeof(double));
    };
    read_snap(sA);
    read_snap(sB);

    copy_fluid_to_device(sA, A);
    copy_fluid_to_device(sB, B);
    std::printf("[ckpt] try load: %s/ckpt_latest_%s.bin\n", dir, tag);
    step_out = H.step;
    return true;
}

void rotate_checkpoints(const char* /*dir*/, const char* /*tag*/, int /*keep*/) {
    // TODO: 以后按需实现老文件清理
}

/* ================= 参数读取 / 打印 ================= */
RuntimeParams load_params_txt(const string& path, const RuntimeParams& d){
    RuntimeParams r = d;
    r.source_note = "defaults";
    auto P = parse_kv(path);
        // 三个“getter”辅助，避免重复 if(count)
    auto get=[&](const char* k,double& x){ if(P.count(k)) x=P[k]; };
    auto geti=[&](const char* k,int& x){ if(P.count(k)) x=(int)P[k]; };
    auto getb =[&](const char* k,bool&   x){ if(P.count(k)) x=((int)P[k])!=0; };
    auto getu64=[&](const char* k,unsigned long long& x){ if(P.count(k)) x=(unsigned long long)P[k]; };
        // —— 逐字段覆盖默认值 —— //

    // 场景
    geti("init_eq", r.init_eq);

    // 几何
    geti("morph", r.morph); get("r_obs", r.r_obs); get("l_gap", r.l_gap);
    get("coat_thick", r.coat_thick); get("r_mid", r.r_mid);

    // 初始/润湿
    get("Sw", r.Sw); getu64("water_seed", r.water_seed);
    get("thetaA_quartz_deg", r.thetaA_quartz_deg);
    get("thetaA_hydrate_deg",r.thetaA_hydrate_deg);
    get("GBw_quartz", r.GBw_quartz); get("GBw_hydrate", r.GBw_hydrate);
    // 允许在 params.txt 里直接给出自定义仿射系数（可选）
    get("GAw_m", r.GAw_m);
    get("GAw_c", r.GAw_c);

    // 驱动
    get("Gx", r.Gx); get("Gy", r.Gy); geti("drive_mode", r.drive_mode);

    // 初始密度（旧键名延用）
    get("rhoA_ini_h", r.rhoA_hi); get("rhoA_ini_l", r.rhoA_lo);
    get("rhoB_ini_h", r.rhoB_hi); get("rhoB_ini_l", r.rhoB_lo);
    //极端情况下的密度分布
    get("rhoA_ini_h_1", r.rhoA_ini_h_1);
    get("rhoA_ini_l_0", r.rhoA_ini_l_0);
    get("rhoB_ini_l_1", r.rhoB_ini_l_1);
    get("rhoB_ini_h_0", r.rhoB_ini_h_0);
    // 黏性/松弛时间
    get("tau_p_a", r.tau_p_a); get("tau_p_b", r.tau_p_b);

    // 界面/相互作用新增（若 params.txt 里写了就能覆盖）
    get("GAB", r.GAB); get("GBA", r.GBA); get("sigmaA", r.sigmaA);
    get("kappa", r.kappa);

    // ── Huang & Wu (2016) SCMP 参数 ──
    geti("pp_mode", r.pp_mode);
    get("epsilon_huang", r.epsilon_huang);
    get("alpha_meq", r.alpha_meq);
    get("cs_a", r.cs_a); get("cs_b", r.cs_b); get("cs_R", r.cs_R);
    get("cs_T", r.cs_T); get("cs_G", r.cs_G);
    get("huang_R0", r.huang_R0); get("huang_xc", r.huang_xc);
    get("huang_yc", r.huang_yc); get("huang_W", r.huang_W);
    get("huang_rho_g", r.huang_rho_g); get("huang_rho_l", r.huang_rho_l);
    get("G_ads", r.G_ads_scmp);  // SCMP contact angle calibration
    get("theta_contact_deg", r.theta_contact_deg);  // ψ-based contact angle
    get("huang_psi_l_ref", r.huang_psi_l_ref); get("huang_psi_g_ref", r.huang_psi_g_ref);
    get("tau_huang", r.tau_huang); get("Lambda_huang", r.Lambda_huang);
    get("huang_u_max", r.huang_u_max); get("huang_psi_cut", r.huang_psi_cut);
    get("huang_tanh_factor", r.huang_tanh_factor); get("huang_rho_max_init", r.huang_rho_max_init);
    geti("huang_init_mode", r.huang_init_mode);

    geti("CP_EVERY", r.CP_EVERY);
    geti("CP_KEEP",  r.CP_KEEP);
    geti("CP_RESUME",r.CP_RESUME);
    geti("OUTPUT_EVERY", r.OUTPUT_EVERY);

    get("eq_tol_rel", r.eq_tol_rel);
    geti("eq_need_consec", r.eq_need_consec);
    geti("eq_max_steps", r.eq_max_steps);
    get("eq_q_abs_eps", r.eq_q_abs_eps);

    get("flow_tol_rel", r.flow_tol_rel);
    geti("flow_need_consec", r.flow_need_consec);
    geti("flow_max_steps", r.flow_max_steps);


    getb("ENABLE_CKPT",       r.ENABLE_CKPT);

#ifdef HYDRATE_ENABLE
    // 水合物相变参数
    getb("hydrate_enable",        r.hydrate_enable);
    geti("hydrate_start_step",    r.hydrate_start_step);
    get("T0_init",                r.T0_init);
    get("T0_inlet",               r.T0_inlet);
    geti("thermal_bc_side",       r.thermal_bc_side);
    geti("thermal_init_mode",     r.thermal_init_mode);
    get("lambda_fluid",           r.lambda_fluid);
    get("lambda_hydrate",         r.lambda_hydrate);
    get("lambda_solid",           r.lambda_solid);
    get("rhocp_fluid",            r.rhocp_fluid);
    get("rhocp_hydrate",          r.rhocp_hydrate);
    get("rhocp_solid",            r.rhocp_solid);
    get("D_mol_water",            r.D_mol_water);
    get("Henry_KH",               r.Henry_KH);
    get("Cm_init",                r.Cm_init);
    get("k0_rxn",                 r.k0_rxn);
    get("Ea_rxn",                 r.Ea_rxn);
    get("e1_peq",                 r.e1_peq);
    get("e2_peq",                 r.e2_peq);
    get("latent_heat",            r.latent_heat);
    get("Vm_hydrate",             r.Vm_hydrate);
    get("Vh_init",                r.Vh_init);
    get("vop_terminate_frac",     r.vop_terminate_frac);
    get("dx_phys",                r.dx_phys);
    get("dt_phys",                r.dt_phys);
#endif
    // ============ 应用 init_eq 预设（在所有普通键读完之后执行） ============
    // 约定：init_eq 为 1 或 2 时，以下组合将覆盖相关字段；
    //double rhoA_hi = 6.6293, rhoA_lo = 0.34127;
    //double rhoB_hi = 0.3823,  rhoB_lo = 0.0001;
    // 若想完全手动，请在 params.txt 里写 init_eq 0 或去掉该键。
    if (r.init_eq == 1) {
        r.rhoA_hi = 6.6293;  r.rhoA_lo = 0.5;
        r.rhoB_hi = 0.41931; r.rhoB_lo = 0.05;
        r.tau_p_a = 0.593418; r.tau_p_b = 0.515411;
        //r.tau_p_a = 1.2; r.tau_p_b = 1.2;
        r.GAB = 0.24; r.GBA = 0.24;
        r.sigmaA = 0.08;
        r.kappa  = 0.7698;
        r.GAw_m = 1.0 / 399.39;
        r.GAw_c = 85.29;
        r.rhoA_ini_h_1 = 7.5511;   r.rhoA_ini_l_0 = 0.0;
        r.rhoB_ini_l_1 = 0.0;      r.rhoB_ini_h_0 = 0.4290;
    } else if (r.init_eq == 2) {
        r.rhoA_hi = 6.6293;  r.rhoA_lo = 0.005;
        r.rhoB_hi = 0.236653; r.rhoB_lo = 0.014;
        r.tau_p_a = 0.608377; r.tau_p_b = 0.524639;
        r.GAB = 0.30; r.GBA = 0.30;
        r.sigmaA = 0.08;
        r.kappa  = 0.5337;
        r.GAw_m = 1.0 / 375.40;
        r.GAw_c = 85.54;
        r.rhoA_ini_h_1 = 7.2243;  r.rhoA_ini_l_0 = 0.0;
        r.rhoB_ini_l_1 = 0.0;     r.rhoB_ini_h_0 = 0.2363;
    }

    // 最低物理限：SRT/MRT 下 τ>0.5，避免负粘度/数值不稳定
    if (r.tau_p_a <= 0.5) r.tau_p_a = 0.5001;
    if (r.tau_p_b <= 0.5) r.tau_p_b = 0.5001;
    if (r.OUTPUT_EVERY <= 0) r.OUTPUT_EVERY = static_cast<int>(NOUTPUT);
    if (r.flow_max_steps <= 0) r.flow_max_steps = static_cast<int>(NSTEPS);
    if (r.eq_max_steps <= 0) r.eq_max_steps = 200000;
    // ========= 新增：单独再读一遍字符串参数（目前只需要 geom_file） =========
    {
        std::ifstream in(path);
        if (in) {
            std::string line;
            auto notspace = [](unsigned char ch){ return !std::isspace(ch); };
            auto ltrim = [&](std::string& s){
                s.erase(s.begin(), std::find_if(s.begin(), s.end(),
                    [&](char ch){ return notspace((unsigned char)ch); }));
            };
            auto rtrim = [&](std::string& s){
                s.erase(std::find_if(s.rbegin(), s.rend(),
                    [&](char ch){ return notspace((unsigned char)ch); }).base(), s.end());
            };
            auto trim = [&](std::string& s){ ltrim(s); rtrim(s); };

            while (std::getline(in, line)) {
                // 去掉注释
                size_t p = line.find('#');  if (p!=std::string::npos) line.erase(p);
                p = line.find("//");        if (p!=std::string::npos) line.erase(p);
                trim(line);
                if (line.empty()) continue;

                std::string k, vstr;
                p = line.find('=');
                if (p != std::string::npos) {
                    k = line.substr(0, p);
                    vstr = line.substr(p+1);
                    trim(k); trim(vstr);
                } else {
                    std::istringstream iss(line);
                    if (!(iss >> k >> vstr)) continue;
                }

                if (k == "geom_file") {
                    r.geom_file = vstr;   // 例如 "geometry_case001.plt"
                } else if (k == "ckpt_dir") {
                    r.ckpt_dir = vstr;
                } else if (k == "file_dir") {
                    r.file_dir = vstr;
                }
            }
        }
    }
    if (const char* e = std::getenv("LBM_CKPT_DIR"); e && *e) r.ckpt_dir = e;
    if (const char* e = std::getenv("LBM_FILE_DIR"); e && *e) r.file_dir = e;
    r.source_note = "defaults + " + path + " + env(LBM_*_DIR)";
    return r;
}
//打印GPU的参数
void print_gpu_banner(){
    cudaDeviceProp prop{}; checkCudaErrors(cudaGetDeviceProperties(&prop,0));
    size_t freeMem{}, totalMem{}; cudaMemGetInfo(&freeMem,&totalMem);
    const double MiB = 1024.0*1024.0;
    printf("[GPU] %s  (cc %d.%d, %d SMs)\n",prop.name,prop.major,prop.minor,prop.multiProcessorCount);
    printf(" Memory  %.1f / %.1f  MiB  free/total\n\n", freeMem/MiB,totalMem/MiB);
}

//运行适合打印信息
void print_params_summary(const RuntimeParams& p){
    // 备份/设置输出格式
    std::ios old_state(nullptr);
    old_state.copyfmt(std::cout);
    std::cout.setf(std::ios::scientific);
    std::cout << std::setprecision(6);

    // —— 派生量（用于可读性与排错）——
    const double denom  = (p.GAw_m != 0.0 ? 1.0 / p.GAw_m : 0.0);   // 旧公式里的分母
    //const double GAw_qz = GAw_from_theta(p.thetaA_quartz_deg,  p.GAw_m, p.GAw_c);
    //const double GAw_hy = GAw_from_theta(p.thetaA_hydrate_deg, p.GAw_m, p.GAw_c);
    const double nuA    = (p.tau_p_a - 0.5) / 3.0;                  // D2Q9: ν = (τ-0.5)/3
    const double nuB    = (p.tau_p_b - 0.5) / 3.0;

    std::cout
        << "[SCENE] init_eq=" << p.init_eq
        << " | morph=" << p.morph
        << " | drive_mode=" << p.drive_mode << "\n"

        << "[GEOM ] morph=" << p.morph
        << " | r_obs=" << p.r_obs
        << " l_gap=" << p.l_gap
        << " coat=" << p.coat_thick
        << " r_mid=" << p.r_mid << "\n"
        << " geom_file=" << (p.geom_file.empty() ? "(analytic)" : p.geom_file) << "\n"

        << "[DRIVE] G=(" << p.Gx << "," << p.Gy << ")\n"
        << "[INIT ] Sw=" << p.Sw
        << " seed=" << p.water_seed
        << " | rhoA_hi=" << p.rhoA_hi << " rhoA_lo=" << p.rhoA_lo
        << " | rhoB_hi=" << p.rhoB_hi << " rhoB_lo=" << p.rhoB_lo << "\n"

        << " | rhoA_ini_h_1=" << p.rhoA_ini_h_1 << " rhoA_ini_l_0=" << p.rhoA_ini_l_0
        << " | rhoB_ini_h_0=" << p.rhoB_ini_h_0 << " rhoB_ini_l_1=" << p.rhoB_ini_l_1 << "\n"


        << "[TAU  ] tauA=" << p.tau_p_a << " (nuA≈" << nuA << ")"
        << " | tauB=" << p.tau_p_b << " (nuB≈" << nuB << ")"
        << " | kappa=" << p.kappa << "\n"

        << "[COUPL] GAB=" << p.GAB << " GBA=" << p.GBA
        << " | sigmaA=" << p.sigmaA << "\n"

        << "[HUANG] pp_mode=" << p.pp_mode
        << " | epsilon=" << p.epsilon_huang
        << " k2=" << p.k2_huang
        << " (k1=" << -p.epsilon_huang/8.0 - p.k2_huang << ")\n"        << " alpha_meq=" << p.alpha_meq << "\n"
        << "         CS: a=" << p.cs_a << " b=" << p.cs_b
        << " R=" << p.cs_R << " T=" << p.cs_T << " G=" << p.cs_G << "\n"
        << "         G_ads=" << p.G_ads_scmp << " θ_contact=" << p.theta_contact_deg << "°"
        << " ψ_l=" << p.huang_psi_l_ref << " ψ_g=" << p.huang_psi_g_ref << "\n"
        << "         UMAX=" << p.huang_u_max << " psi_cut=" << p.huang_psi_cut
        << " init: mode=" << p.huang_init_mode
        << " R0=" << p.huang_R0 << " xc=" << p.huang_xc
        << " yc=" << p.huang_yc << " W=" << p.huang_W << "\n"

        << "[WETMF] GAw(theta) = m*(theta - c)"
        << " | m=" << p.GAw_m << " c=" << p.GAw_c
        << " (denom≈" << denom << ")\n"

        << "[WET  ] theta_qz=" << p.thetaA_quartz_deg  << "°"
        //<< " -> GAw_qz=" << GAw_qz
        //<< " | GBw_qz=" << p.GBw_quartz << "\n"
        << "        theta_hy=" << p.thetaA_hydrate_deg << "°"
        //<< " -> GAw_hy=" << GAw_hy
        << " | GBw_qz=" << p.GBw_quartz
        << " | GBw_hy=" << p.GBw_hydrate << "\n"

        << "[CKPT ] EVERY=" << p.CP_EVERY
        << " KEEP="  << p.CP_KEEP
        << " RESUME="<< p.CP_RESUME
        << " ENABLE="<< (p.ENABLE_CKPT ? "true" : "false")
        << " | dir=" << p.ckpt_dir << "\n"
        << "[OUT  ] EVERY=" << p.OUTPUT_EVERY
        << " | file_dir=" << p.file_dir << "\n"
        << "[EQ   ] tol=" << p.eq_tol_rel << " need=" << p.eq_need_consec
        << " max=" << p.eq_max_steps << " qeps=" << p.eq_q_abs_eps << "\n"
        << "[FLOW ] tol=" << p.flow_tol_rel << " need=" << p.flow_need_consec
        << " max=" << p.flow_max_steps << "\n"
        << "[SRC  ] " << p.source_note
        << std::endl;

    // 恢复格式
    std::cout.copyfmt(old_state);
}


extern double h_GAw_m, h_GAw_c;
/* ============ 润湿/常量/几何：一次性准备和下发 ============ */
void push_wettability_and_maps(const RuntimeParams& p){
    // Host 侧根据接触角/GBw 构建润湿查表（例如按材质ID → GA/GB/θ）
    h_GAw_m = p.GAw_m;
    h_GAw_c = p.GAw_c;
    upload_wettability_table_host(p.thetaA_quartz_deg, p.thetaA_hydrate_deg,
                                    p.GBw_quartz, p.GBw_hydrate);
    // 分配/准备墙面与润湿映射（Host 缓存：后续 upload_obstacles 时一并推给 GPU）
    alloc_wall_and_wettability_maps_host();
}

void push_device_constants(const RuntimeParams& p){
    // —— 把与本次 run 相关的“动态常量”装进 __constant__ —— //
    CK(cudaMemcpyToSymbol(d_water_satur, &p.Sw,sizeof(double)));
    CK(cudaMemcpyToSymbol(d_water_seed,  &p.water_seed,sizeof(unsigned long long)));
    CK(cudaMemcpyToSymbol(d_Gx, &p.Gx, sizeof(double)));
    CK(cudaMemcpyToSymbol(d_Gy, &p.Gy, sizeof(double)));
    CK(cudaMemcpyToSymbol(d_drive_mode, &p.drive_mode, sizeof(int)));
    CK(cudaMemcpyToSymbol(d_rhoA_ini_h, &p.rhoA_hi, sizeof(double)));
    CK(cudaMemcpyToSymbol(d_rhoA_ini_l, &p.rhoA_lo, sizeof(double)));
    CK(cudaMemcpyToSymbol(d_rhoB_ini_h, &p.rhoB_hi, sizeof(double)));
    CK(cudaMemcpyToSymbol(d_rhoB_ini_l, &p.rhoB_lo, sizeof(double)));

    CK(cudaMemcpyToSymbol(d_rhoA_ini_h_1, &p.rhoA_ini_h_1, sizeof(double)));
    CK(cudaMemcpyToSymbol(d_rhoA_ini_l_0, &p.rhoA_ini_l_0, sizeof(double)));
    CK(cudaMemcpyToSymbol(d_rhoB_ini_h_0, &p.rhoB_ini_h_0, sizeof(double)));
    CK(cudaMemcpyToSymbol(d_rhoB_ini_l_1, &p.rhoB_ini_l_1, sizeof(double)));



    CK(cudaMemcpyToSymbol(d_tau_p_a, &p.tau_p_a, sizeof(double)));
    CK(cudaMemcpyToSymbol(d_tau_p_b, &p.tau_p_b, sizeof(double)));
    CK(cudaMemcpyToSymbol(d_kappa,   &p.kappa,   sizeof(double)));


    CK(cudaMemcpyToSymbol(d_GAB,   &p.GAB,    sizeof(double)));
    CK(cudaMemcpyToSymbol(d_GBA,   &p.GBA,    sizeof(double)));
    CK(cudaMemcpyToSymbol(d_sigmaA,&p.sigmaA,sizeof(double)));

    // ── Huang & Wu (2016) SCMP ──
    // Compute k₁, k₂ from user-facing parameters ε = −8(k₁ + k₂).
    //   k₂ = user value (default 0);  k₁ = −ε/8 − k₂.
    // For normal operation (paper §6.3): keep k₂ = 0 and tune ε alone.
    // For paper Fig.5 σ⟂k₂ verification: set k₂ ≠ 0 to test σ independence.
    double k2_computed = p.k2_huang;
    double k1_computed = -p.epsilon_huang / 8.0 - k2_computed;
    CK(cudaMemcpyToSymbol(d_pp_mode,   &p.pp_mode,   sizeof(int)));
    CK(cudaMemcpyToSymbol(d_k1_huang,  &k1_computed, sizeof(double)));
    CK(cudaMemcpyToSymbol(d_k2_huang,  &k2_computed, sizeof(double)));
    CK(cudaMemcpyToSymbol(d_alpha_meq, &p.alpha_meq, sizeof(double)));
    CK(cudaMemcpyToSymbol(d_cs_a,      &p.cs_a,      sizeof(double)));
    CK(cudaMemcpyToSymbol(d_cs_b,      &p.cs_b,      sizeof(double)));
    CK(cudaMemcpyToSymbol(d_cs_R,      &p.cs_R,      sizeof(double)));
    CK(cudaMemcpyToSymbol(d_cs_T,      &p.cs_T,      sizeof(double)));
    CK(cudaMemcpyToSymbol(d_cs_G,      &p.cs_G,      sizeof(double)));
    CK(cudaMemcpyToSymbol(d_huang_R0,  &p.huang_R0,  sizeof(double)));
    CK(cudaMemcpyToSymbol(d_huang_xc,  &p.huang_xc,  sizeof(double)));
    CK(cudaMemcpyToSymbol(d_huang_yc,  &p.huang_yc,  sizeof(double)));
    CK(cudaMemcpyToSymbol(d_huang_W,   &p.huang_W,   sizeof(double)));
    CK(cudaMemcpyToSymbol(d_huang_rho_g, &p.huang_rho_g, sizeof(double)));
    CK(cudaMemcpyToSymbol(d_huang_rho_l, &p.huang_rho_l, sizeof(double)));
    CK(cudaMemcpyToSymbol(d_huang_init_mode, &p.huang_init_mode, sizeof(int)));
    CK(cudaMemcpyToSymbol(d_G_ads_scmp, &p.G_ads_scmp, sizeof(double)));
    CK(cudaMemcpyToSymbol(d_theta_contact_deg, &p.theta_contact_deg, sizeof(double)));
    { auto ti = [](double t) { double x = -1.78e-4*t*t*t + 4.95e-2*t*t - 3.02*t + 84.5; return x * M_PI / 180.0; };
      double ht[256]={0}; ht[1]=ti(p.thetaA_quartz_deg); ht[2]=ti(p.thetaA_hydrate_deg);
      CK(cudaMemcpyToSymbol(d_theta_by_mat_rad, ht, sizeof(ht))); }
    CK(cudaMemcpyToSymbol(d_huang_psi_l_ref, &p.huang_psi_l_ref, sizeof(double)));
    CK(cudaMemcpyToSymbol(d_huang_psi_g_ref, &p.huang_psi_g_ref, sizeof(double)));
    CK(cudaMemcpyToSymbol(d_tau_huang,  &p.tau_huang,  sizeof(double)));
    CK(cudaMemcpyToSymbol(d_Lambda_huang, &p.Lambda_huang, sizeof(double)));
    CK(cudaMemcpyToSymbol(d_huang_u_max, &p.huang_u_max, sizeof(double)));
    CK(cudaMemcpyToSymbol(d_huang_psi_cut, &p.huang_psi_cut, sizeof(double)));
    CK(cudaMemcpyToSymbol(d_huang_tanh_factor, &p.huang_tanh_factor, sizeof(double)));
    CK(cudaMemcpyToSymbol(d_huang_rho_max_init, &p.huang_rho_max_init, sizeof(double)));


    double A_a_host[9], A_b_host[9];
    if (p.pp_mode == 1) {
        // ── Huang & Wu (2016) paper: τ = tau_huang, s_q formula with Λ ──
        double tau = p.tau_huang;
        double Lambda = p.Lambda_huang;
        double s_paper = 1.0 / tau;  // s_p = s_e = s_ε = 1/τ
        double s_q_paper = 1.0 / (0.5 + Lambda / (tau - 0.5));  // Eq. from paper
        A_a_host[0] = 1.0;       // ρ: conserved
        A_a_host[1] = s_paper;   // e:  s_e = 1/τ
        A_a_host[2] = s_paper;   // ε:  s_ε = 1/τ
        A_a_host[3] = 1.0;       // j_x: conserved
        A_a_host[4] = s_q_paper; // q_x: s_q from formula
        A_a_host[5] = 1.0;       // j_y: conserved
        A_a_host[6] = s_q_paper; // q_y: s_q from formula
        A_a_host[7] = s_paper;   // p_xx: s_p = 1/τ
        A_a_host[8] = s_paper;   // p_xy: s_p = 1/τ
        // B-phase unused in SCMP but set to same for safety
        for (int i = 0; i < 9; ++i) A_b_host[i] = A_a_host[i];
    } else {
        // Legacy MCMP relaxation
        A_a_host[0] = 1.0; A_a_host[1] = 1/tau_e; A_a_host[2] = 1/tau_t;
        A_a_host[3] = 1.0; A_a_host[4] = 1/tau_q; A_a_host[5] = 1.0;
        A_a_host[6] = 1/tau_q; A_a_host[7] = 1/p.tau_p_a; A_a_host[8] = 1/p.tau_p_a;
        A_b_host[0] = 1.0; A_b_host[1] = 1/tau_e; A_b_host[2] = 1/tau_t;
        A_b_host[3] = 1.0; A_b_host[4] = 1/tau_q; A_b_host[5] = 1.0;
        A_b_host[6] = 1/tau_q; A_b_host[7] = 1/p.tau_p_b; A_b_host[8] = 1/p.tau_p_b;
    }
    CK(cudaMemcpyToSymbol(A_a_gpu, A_a_host, sizeof(A_a_host)));
    CK(cudaMemcpyToSymbol(A_b_gpu, A_b_host, sizeof(A_b_host)));
}


void build_and_upload_geometry(const RuntimeParams& p, Porous_host& porous){
    // 依据赋存形态与几何参数生成规则颗粒阵列：
    //  - pore-fill: 中点水合物颗粒1
    //  - coating  : 固体外包水合物壳2
    //  - mixed    : 两者混合3
    build_circle_array(porous, p.morph, p.r_obs, p.coat_thick, p.r_mid, p.l_gap);
    // 上传：写入 pointsflag/材质图/润湿映射到 GPU，全域几何就绪
    upload_obstacles(porous);

}
void build_and_upload_geometry_from_tecplot(const RuntimeParams& P,
                                            Mix_dev& M_dev)
{
    std::vector<int> host_flag;
    std::vector<unsigned char> hmd; read_tecplot_to_flag(P.geom_file, host_flag, hmd);  // geom_file 来自 params.txt

    // 把 host_flag 拷贝到 device 的 pointsflag
    checkCudaErrors(cudaMemcpy(M_dev.pointsflag,
                               host_flag.data(),
                               sizeof(int)*NX*NY,
                               cudaMemcpyHostToDevice));
}

/* ================= 分配/释放 ================= */
void allocate_all(Fluid_dev& A, Fluid_dev& B, Mix_dev& M){
  alloc_fluid(A); alloc_fluid(B); alloc_mix(M);
}
void free_all(Fluid_dev& A, Fluid_dev& B, Mix_dev& M){
  free_fluid(A); free_fluid(B); free_mix(M);
}

static inline void maybe_save_periodic_checkpoint(const RuntimeParams& P,
                                                  const char* tag,
                                                  int step,
                                                  const Fluid_dev& A,
                                                  const Fluid_dev& B,
                                                  const Mix_dev& M){
    if (P.ENABLE_CKPT && P.CP_EVERY > 0 && (step % P.CP_EVERY == 0)) {
        save_checkpoint(P.ckpt_dir.c_str(), tag, step, A, B, M, P);
    }
}

static inline void write_stage_output(int step,
                                      const StageConfig& cfg,
                                      const RuntimeParams& P,
                                      Fluid_dev& A, Fluid_host& AH,
                                      Fluid_dev& B, Fluid_host& BH,
                                      Mix_dev& MX, Mix_host& MH,
                                      const char* state){
    copy_and_check(A, AH, "A_");
    copy_and_check(B, BH, "B_");
    copy_back_mix(MX, MH);

    std::string prefix = std::string("outputdata_") + cfg.tag;
    std::string title  = std::string(cfg.tag) + " " + state;
    outputvtk(step, P.file_dir, prefix.c_str(), title.c_str(), AH, BH, MH);
}

static bool probe_and_check_steady(int step,
                                   const StageConfig& cfg,
                                   SteadyMonitor& SM,
                                   Fluid_dev& A,
                                   Fluid_dev& B,
                                   Mix_dev& MX,
                                   RunResult& R){
    if (step % SM.interval != 0) return false;
    double QA=0.0, QB=0.0 , QT=0.0;
    SM.compute_Q_GPU(A, B, MX, QA, QB, QT);
    R.QA = QA; R.QB = QB; R.Qmix = QT; R.last_probe_step = step;
    (void)SM.fetch_limiter_stats_and_log(step, /*print_thr=*/1e-3);

    const bool reached_rel = SM.compare_and_update(QA, QB, QT, step);
    const bool ok_abs = (!cfg.require_abs_quiet) || (std::fabs(R.Qmix) < cfg.q_abs_eps);
    if (reached_rel && ok_abs) {
        R.steady = true; R.steady_step = step;
        return true;
    }
    return false;
}

RunResult run_stage(Fluid_dev& A, Fluid_host& AH,
                    Fluid_dev& B, Fluid_host& BH,
                    Mix_dev&   MX, Mix_host&   MH,
                    SteadyMonitor& SM,
                    const StageConfig& cfg,
                    const RuntimeParams& P)
{
    RunResult R{};
    set_drive_scale(cfg.drive_scale);
    SM.tol_rel     = cfg.tol_rel;
    SM.need_consec = cfg.need_consec;
    SM.consec_hit  = 0;

    int resumed_step = -1;
    bool did_resume  = false;
    if (P.CP_RESUME && load_checkpoint(P.ckpt_dir.c_str(), cfg.tag, resumed_step, A, B, MX, P)) {
        std::printf("[ckpt] resumed %s at step %d\n", cfg.tag, resumed_step);
        did_resume = true;                    // ← 原来漏了
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    int start = (resumed_step >= 0) ? (resumed_step + 1) : 0;

    SM.limiter_window     = SM.interval;
    SM.limiter_log_every  = SM.limiter_window; // 写文件频率 == 窗口
    SM.limiter_stdout     = false;             // 不在控制台打印
    SM.limiter_log_on_hit = false;             // 不因“命中阈值”即时写
    SM.limiter_log_path = (P.file_dir + "/limiter_log.csv");
    for (int step = start; step < cfg.max_steps; ++step) {
        if (step % SM.limiter_window == 0) {
            SM.reset_limiter_counters();
        }
        evolution_all(A, B, MX);
        maybe_save_periodic_checkpoint(P, cfg.tag, step, A, B, MX);
        if (probe_and_check_steady(step, cfg, SM, A, B, MX, R)) {
            write_stage_output(step, cfg, P, A, AH, B, BH, MX, MH, "steady");
            break;
        }
        // —— 周期性过程输出 —— //
        if (P.OUTPUT_EVERY > 0 && step % P.OUTPUT_EVERY == 0) {
            write_stage_output(step, cfg, P, A, AH, B, BH, MX, MH, "running");
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::printf("[%s] time = %.3f s\n", cfg.tag,
                std::chrono::duration<double>(t2 - t1).count());
    R.resumed = did_resume;
    R.resumed_step = resumed_step;

    // 在 run_stage() 的最后，return 之前
    if (P.ENABLE_CKPT) {
    int step_to_save = (R.steady && R.steady_step >= 0) ? R.steady_step : R.last_probe_step;
    if (step_to_save < 0) step_to_save = 0;
    save_checkpoint(P.ckpt_dir.c_str(), cfg.tag, step_to_save, A, B, MX, P);
    }

    return R;
}


#ifdef HYDRATE_ENABLE
// ============================================================
// run_stage_hydrate：含水合物物理耦合的时间推进主循环
// ------------------------------------------------------------
// 耦合顺序（Yang 2024 Figure S2）：
//   Flow（evolution_all）→ Conc → LatentHeat → Thermal → VOP
// ============================================================

// 宿主端辅助：从设备拷贝水合物场到主机缓冲（用于 VTK 输出）
static void copy_hydrate_to_host(const Therm_dev& TH,
                                  const Conc_dev&  CN,
                                  const VOP_dev&   VP,
                                  HydrateHost&     HH)
{
    const size_t n = (size_t)NX * NY;
    CK(cudaMemcpy(HH.T.data(),         TH.T,         n*sizeof(double), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(HH.Cm.data(),        CN.Cm,        n*sizeof(double), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(HH.Vh.data(),        VP.Vh,        n*sizeof(double), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(HH.diss_rate.data(), VP.diss_rate, n*sizeof(double), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(HH.pore_origin.data(), VP.pore_origin, n*sizeof(double), cudaMemcpyDeviceToHost));
}

// 水合物诊断：计算 hydrate_volume_frac 和 Q_dissociation
static void update_hydrate_diagnostics(const VOP_dev& VP,
                                        double Vh_init_total,
                                        RunResult& R)
{
    const size_t n = (size_t)NX * NY;
    // GPU → host 求和（规模 300×300 = 90000，直接 host 累加足够快）
    std::vector<double> vh_h(n), dr_h(n);
    CK(cudaMemcpy(vh_h.data(), VP.Vh,        n*sizeof(double), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(dr_h.data(), VP.diss_rate, n*sizeof(double), cudaMemcpyDeviceToHost));
    double sum_vh = 0.0, sum_dr = 0.0;
    for (size_t i = 0; i < n; ++i) { sum_vh += vh_h[i]; sum_dr += dr_h[i]; }
    R.hydrate_volume_frac = (Vh_init_total > 0.0) ? sum_vh / Vh_init_total : 1.0;
    R.Q_dissociation      = sum_dr;
}

RunResult run_stage_hydrate(Fluid_dev& A, Fluid_host& AH,
                            Fluid_dev& B, Fluid_host& BH,
                            Mix_dev&   MX, Mix_host&   MH,
                            Therm_dev& TH, Conc_dev& CN,
                            VOP_dev&   VP, HydrateHost& HH,
                            SteadyMonitor& SM,
                            const StageConfig& cfg,
                            const RuntimeParams& P)
{
    RunResult R{};
    set_drive_scale(cfg.drive_scale);
    SM.tol_rel     = cfg.tol_rel;
    SM.need_consec = cfg.need_consec;
    SM.consec_hit  = 0;

    // 计算初始水合物总量（用于归一化 volume_frac）
    const size_t n_cells = (size_t)NX * NY;
    std::vector<double> vh0(n_cells);
    CK(cudaMemcpy(vh0.data(), VP.Vh, n_cells*sizeof(double), cudaMemcpyDeviceToHost));
    double Vh_init_total = 0.0;
    for (double v : vh0) Vh_init_total += v;

    SM.limiter_window    = SM.interval;
    SM.limiter_log_every = SM.limiter_window;
    SM.limiter_stdout    = false;
    SM.limiter_log_on_hit = false;
    SM.limiter_log_path  = (P.file_dir + "/limiter_log.csv");

    auto t1 = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < cfg.max_steps; ++step) {
        if (step % SM.limiter_window == 0) SM.reset_limiter_counters();

        // ── 流场演化（现有 MRT 伪势双相 LBM）──
        evolution_all(A, B, MX);

        // ── 水合物相变物理（Yang 2024 顺序）──
        if (P.hydrate_enable && step >= P.hydrate_start_step) {
            // 1. 浓度场（CST + Kang 反应边界 → diss_rate）
            step_conc(CN, TH, MX.ux, MX.uy, A.rho, B.rho, VP, MX.pointsflag);
            // 2. 潜热源项
            compute_latent_heat_source(VP, CN, TH, MX.pointsflag);
            // 3. 热场（含源项 S_latent）
            step_thermal(TH, MX.ux, MX.uy, VP.S_latent, MX.pointsflag, P.T0_inlet, P.thermal_bc_side);
            // 4. VOP 固相更新（返回翻转数；>0 时 ghost 已重建）
            int n_conv = step_vop(VP, TH, CN, A, B, MX);
            if (n_conv > 0) {
                R.n_converted_total += n_conv;
                // 翻转后重新上传润湿性（保持材质→GAw 映射一致）
                upload_wettability_table_host(P.thetaA_quartz_deg,
                                              P.thetaA_hydrate_deg,
                                              P.GBw_quartz,
                                              P.GBw_hydrate);
            }
        }

        maybe_save_periodic_checkpoint(P, cfg.tag, step, A, B, MX);

        // ── 稳态检测 + 输出 ──
        if (probe_and_check_steady(step, cfg, SM, A, B, MX, R)) {
            // 输出最终场
            copy_and_check(A, AH, "A_");
            copy_and_check(B, BH, "B_");
            copy_back_mix(MX, MH);
            std::string prefix = std::string("outputdata_") + cfg.tag;
            std::string title  = std::string(cfg.tag) + " steady";
            std::string vtk_path = P.file_dir + "/" + prefix
                + ([&]{ std::ostringstream o; o << std::setw(8) << std::setfill('0') << step; return o.str(); }())
                + ".vtk";
            outputvtk(step, P.file_dir, prefix.c_str(), title.c_str(), AH, BH, MH);
            if (P.hydrate_enable) {
                copy_hydrate_to_host(TH, CN, VP, HH);
                outputvtk_append_hydrate(vtk_path, HH.T, HH.Cm, HH.Vh, HH.diss_rate, HH.pore_origin);
                update_hydrate_diagnostics(VP, Vh_init_total, R);
            }
            break;
        }

        if (P.OUTPUT_EVERY > 0 && step % P.OUTPUT_EVERY == 0) {
            copy_and_check(A, AH, "A_");
            copy_and_check(B, BH, "B_");
            copy_back_mix(MX, MH);
            std::string prefix = std::string("outputdata_") + cfg.tag;
            std::string title  = std::string(cfg.tag) + " running";
            std::string vtk_path = P.file_dir + "/" + prefix
                + ([&]{ std::ostringstream o; o << std::setw(8) << std::setfill('0') << step; return o.str(); }())
                + ".vtk";
            outputvtk(step, P.file_dir, prefix.c_str(), title.c_str(), AH, BH, MH);
            if (P.hydrate_enable) {
                copy_hydrate_to_host(TH, CN, VP, HH);
                  outputvtk_append_hydrate(vtk_path, HH.T, HH.Cm, HH.Vh, HH.diss_rate, HH.pore_origin);
                update_hydrate_diagnostics(VP, Vh_init_total, R);
                printf("[hydrate step=%d] Vh_frac=%.4f  Q_diss=%.4e  n_conv_total=%d\n",
                       step, R.hydrate_volume_frac, R.Q_dissociation, R.n_converted_total);
            }
        }

        // 水合物耗尽终止条件
        if (P.hydrate_enable && step % SM.interval == 0) {
            update_hydrate_diagnostics(VP, Vh_init_total, R);
            if (R.hydrate_volume_frac < P.vop_terminate_frac) {
                printf("[VOP] 水合物分数 %.4f < terminate 阈值 %.4f，提前终止。\n",
                       R.hydrate_volume_frac, P.vop_terminate_frac);
                break;
            }
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    printf("[%s hydrate] time = %.3f s\n", cfg.tag,
           std::chrono::duration<double>(t2 - t1).count());

    if (P.ENABLE_CKPT) {
        int step_to_save = (R.steady && R.steady_step >= 0) ? R.steady_step : R.last_probe_step;
        if (step_to_save < 0) step_to_save = 0;
        save_checkpoint(P.ckpt_dir.c_str(), cfg.tag, step_to_save, A, B, MX, P);
    }

    return R;
}
#endif  // HYDRATE_ENABLE

RunResult run_equilibrate_then_flow(Fluid_dev& A, Fluid_host& AH,
                                    Fluid_dev& B, Fluid_host& BH,
                                    Mix_dev&   M, Mix_host&   MH,
                                    SteadyMonitor& SM,
                                    const RuntimeParams& P)
{
    RunResult R{}; // 汇总结果（最终会返回这一份）

    // —— 优先尝试用 eq 基线存档，能找到就跳过阶段1 —— //
    bool have_eq = false;
    int  step_eq = -1;
    if (P.CP_RESUME && load_checkpoint(P.ckpt_dir.c_str(), "eq", step_eq, A, B, M, P)) {
        std::printf("[ckpt] found equilibrated base → skip Stage-1 (resume step=%d)\n", step_eq);
        have_eq = true;
        R.eq_skipped  = true;
        R.eq_ckpt_step= step_eq;   // 这次使用的 eq 基线的步号
    }

    // —— 阶段1：无驱动相分离（若没找到 eq 基线才运行） —— //
    if (!have_eq) {
        StageConfig eq;
        eq.drive_scale       = 0.0;
        eq.tol_rel           = P.eq_tol_rel;
        eq.need_consec       = P.eq_need_consec;
        eq.max_steps         = std::max(P.eq_max_steps, SM.interval*20);
        eq.require_abs_quiet = true;
        eq.q_abs_eps         = P.eq_q_abs_eps;
        eq.tag               = "eq";

        // --- 用总阀门控制局部水锁 ---

        RunResult R_eq = run_stage(A, AH, B, BH, M, MH, SM, eq, P);
        // 记录阶段1关键信息（精简版）
        R.eq_steady      = R_eq.steady;
        R.eq_steady_step = R_eq.steady_step;

        // 不管是否严格命中稳态，落一份 eq 基线，供后续参数扫直接复用
        int ck = R_eq.steady ? R_eq.steady_step
                             : (R_eq.last_probe_step >= 0 ? R_eq.last_probe_step : 0);
        if (P.CP_EVERY > 0 || P.CP_RESUME) {           // 方案1：只要你开启过断点相关，就保存
            if (save_checkpoint(P.ckpt_dir.c_str(), "eq", ck, A, B, M, P)) {
                R.eq_ckpt_step = ck;
                std::printf("[ckpt] saved eq baseline at step %d\n", ck);
            } else {
                R.eq_ckpt_step = -1;
            }
        }
    }

    // —— 阶段2：加驱动正式流动 —— //
    StageConfig flow;
    flow.drive_scale       = 1.0;
    flow.tol_rel           = P.flow_tol_rel;
    flow.need_consec       = P.flow_need_consec;
    flow.max_steps         = P.flow_max_steps;
    flow.require_abs_quiet = false;
    flow.tag               = "flow";

    RunResult R_flow = run_stage(A, AH, B, BH, M, MH, SM, flow, P);

    // 把 flow 的“是否断点恢复”带出来，供摘要使用（精简版）
    R.flow_resumed     = R_flow.resumed;
    R.flow_resume_step = R_flow.resumed_step;

    // 把 flow 的最终稳态信息镜像到顶层，保持和旧版 write_run_summary 的兼容性
    R.steady          = R_flow.steady;
    R.steady_step     = R_flow.steady_step;
    R.last_probe_step = R_flow.last_probe_step;
    R.QA              = R_flow.QA;
    R.QB              = R_flow.QB;
    R.Qmix            = R_flow.Qmix;


    // 写摘要（只写核心几项）
    write_run_summary(R, SM.interval, P);

    return R;
}

/* ================= 摘要输出 ================= */
void write_run_summary(const RunResult& R, int interval, const RuntimeParams& P){
  std::ofstream f(P.file_dir + "/run_summary.txt");
  f.setf(std::ios::scientific);
  f << std::setprecision(16);

  // —— Flow（最终阶段）核心信息 —— //
  f << "QA " << R.QA << "\nQB " << R.QB << "\nQmix " << R.Qmix << "\n";
  f << "last_probe_step " << R.last_probe_step << "\n";
  f << "probe_interval "  << interval << "\n";
  if (R.steady) {
    f << "status steady\n" << "steady_step " << R.steady_step << "\n";
  } else {
    f << "status reached_NSTEPS_without_steady\n" << "NSTEPS " << NSTEPS << "\n";
  }

  // —— 精简的阶段1/断点信息 —— //
  f << "\n# eq\n";
  f << "eq_skipped "     << (R.eq_skipped ? 1 : 0) << "\n";
  f << "eq_steady_step " << R.eq_steady_step << "\n";
  f << "eq_ckpt_step "   << R.eq_ckpt_step   << "\n";

  f << "\n# restart\n";
  f << "flow_resumed "      << (R.flow_resumed ? 1 : 0) << "\n";
  f << "flow_resume_step "  << R.flow_resume_step << "\n";
}

/* =====================================================================
 * ── Huang & Wu (2016) SCMP run loop ──
 * Single-component multiphase on 256×256 periodic domain.
 * ===================================================================== */
#if defined(HUANG_256_BUILD) || defined(HUANG_POROUS_BUILD)
void run_scmp_huang(const RuntimeParams& P, const char* params_path)
{
    // ── Allocate single fluid device memory ──
    Fluid_dev F;
    alloc_fluid(F);

    // ── Allocate pointsflag (all fluid = 1) ──
    size_t N = size_t(NX) * NY;
    int* pointsflag_dev = nullptr;
    cudaMalloc(&pointsflag_dev, N * sizeof(int));
    cudaMemset(pointsflag_dev, 0, N * sizeof(int));

    // ── Load geometry from .plt ──
    std::vector<unsigned char> hm_geom;  // stored for wall_mat upload
    if (!P.geom_file.empty()) {
        std::vector<int> hf;
        read_tecplot_to_flag(P.geom_file, hf, hm_geom);
        checkCudaErrors(cudaMemcpy(pointsflag_dev, hf.data(), N*sizeof(int), cudaMemcpyHostToDevice));
        unsigned char* dm = nullptr;
        cudaMemcpyFromSymbol(&dm, d_wall_mat, sizeof(dm));
        if (dm) checkCudaErrors(cudaMemcpy(dm, hm_geom.data(), N, cudaMemcpyHostToDevice));
        int nf = 0; for (int v : hf) if (v > 0) nf++;
        printf("[scmp-porous] fluid nodes (from .plt): %d / %zu\n", nf, N);
    }

    // ── Init device constants (sets up e_gpu) ──
    init_device_variable();
    push_device_constants(P);
    dbg_consts_once<<<1,1>>>();
    cudaDeviceSynchronize();

    // ── Mark boundary & ghost (requires e_gpu from init_device_variable) ──
    if (!P.geom_file.empty()) {
        unsigned char* dm = nullptr;
        cudaMemcpyFromSymbol(&dm, d_wall_mat, sizeof(dm));
        scmp_init_geometry(pointsflag_dev, dm);
        printf("[scmp-porous] geometry loaded via scmp_init_geometry (%dx%d)\n", NX, NY);
    }

    init_all_scmp(F.rho, F.fin, F.fout, F.min, F.mout, pointsflag_dev);
    cudaDeviceSynchronize();

    // ── Host-side buffers for output ──
    std::vector<double> h_rho(N), h_ux(N), h_uy(N), h_pressure(N);
    std::vector<double> h_p_xx(N), h_p_yy(N);
    std::vector<double> h_Fx(N), h_Fy(N), h_psi(N);
    std::vector<int>    h_flag(N);

    // ── Output directory (respects file_dir from params or LBM_FILE_DIR env) ──
    std::string out_dir = P.file_dir + "/outputdata_scmp";
    namespace fs = std::filesystem;
    fs::create_directories(out_dir);

    // Write a copy of params.txt into the output directory for traceability
    {
        std::ifstream src(params_path);
        if (src) {
            std::ofstream dst(out_dir + "/params.txt");
            dst << src.rdbuf();
        }
    }

    printf("[scmp] output dir: %s\n", out_dir.c_str());
    printf("[scmp] Starting SCMP time loop: %d steps, output every %d\n",
           static_cast<int>(NSTEPS), P.OUTPUT_EVERY);

    unsigned char* wall_mat_dev = nullptr;
    cudaMemcpyFromSymbol(&wall_mat_dev, d_wall_mat, sizeof(wall_mat_dev));

    auto t_start = std::chrono::high_resolution_clock::now();

    // ── Time loop ──
    for (int step = 0; step < static_cast<int>(NSTEPS); ++step) {
        evolution_scmp(
            F.rho, F.ux, F.uy, F.psi, F.pressure,
            F.Fx_mol, F.Fy_mol,
            F.Fx_ads, F.Fy_ads,
            F.fin, F.fout, F.min, F.mout,
            F.S, F.C,
            F.p_xx, F.p_yy, F.p_xy,
            wall_mat_dev,
            pointsflag_dev);

        // Periodic output
        if (P.OUTPUT_EVERY > 0 && step % P.OUTPUT_EVERY == 0) {
            cudaMemcpy(h_rho.data(),      F.rho,      N*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_ux.data(),       F.ux,       N*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_uy.data(),       F.uy,       N*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_pressure.data(), F.pressure, N*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_p_xx.data(),     F.p_xx,     N*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_p_yy.data(),     F.p_yy,     N*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_Fx.data(),     F.Fx_mol,   N*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_Fy.data(),     F.Fy_mol,   N*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_psi.data(),    F.psi,      N*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_flag.data(),     pointsflag_dev, N*sizeof(int), cudaMemcpyDeviceToHost);

            outputvtk_scmp(step, out_dir, "flow", "scmp_step", h_rho, h_ux, h_uy, h_pressure, h_p_xx, h_p_yy, h_Fx, h_Fy, h_psi, h_flag);
            printf("[scmp] step %d output written\n", step);
        }
    }

    // Final output
    cudaMemcpy(h_rho.data(),      F.rho,      N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ux.data(),       F.ux,       N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_uy.data(),       F.uy,       N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pressure.data(), F.pressure, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_p_xx.data(),     F.p_xx,     N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_p_yy.data(),     F.p_yy,     N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Fx.data(),     F.Fx_mol,   N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Fy.data(),     F.Fy_mol,   N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_psi.data(),    F.psi,      N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_flag.data(),     pointsflag_dev, N*sizeof(int), cudaMemcpyDeviceToHost);

    int final_step = static_cast<int>(NSTEPS);
    outputvtk_scmp(final_step, out_dir, "flow", "scmp_final", h_rho, h_ux, h_uy, h_pressure, h_p_xx, h_p_yy, h_Fx, h_Fy, h_psi, h_flag);

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    printf("[scmp] Done. %d steps in %.1f s (%.1f MLUPS)\n",
           final_step, elapsed, (final_step * NX * NY / elapsed / 1e6));

    // ── Write run summary ──
    {
        std::ofstream summary(out_dir + "/run_summary.txt");
        summary.setf(std::ios::scientific);
        summary << std::setprecision(6);
        summary << "NSTEPS " << final_step << "\n";
        summary << "elapsed_s " << elapsed << "\n";
        summary << "MLUPS " << (final_step * NX * NY / elapsed / 1e6) << "\n";
        summary << "pp_mode " << P.pp_mode << "\n";
        summary << "epsilon_huang " << P.epsilon_huang << "\n";
        summary << "k2_huang " << P.k2_huang << "\n";
        summary << "k1_computed " << (-P.epsilon_huang/8.0 - P.k2_huang) << "\n";
        summary << "cs_T " << P.cs_T << "\n";
        summary << "cs_a " << P.cs_a << "\n";
    }

    // ── Cleanup ──
    free_fluid(F);
    cudaFree(pointsflag_dev);
}
#endif // HUANG_256_BUILD
