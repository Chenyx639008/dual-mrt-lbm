#include "../include/steady_monitor.cuh"
#include <vector>
#include <queue>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <cmath>
#include <cstring>


#define CK(call) do{ \
  cudaError_t _e=(call); \
  if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s failed @ %s:%d : %s\n", \
      #call,__FILE__,__LINE__,cudaGetErrorString(_e)); \
    exit(1); \
  } \
}while(0)


// ========== GPU kernels ==========

// 1) 体流量规约（与你现有保持一致）

// ── SCMP single-phase flow rate reduction ─────────────────────
__global__ void reduce_Q_scmp_kernel(
    const double* __restrict__ ux,
    const int*    __restrict__ flag,
    double*       sumQ,
    int nTot)
{
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int sth = blockDim.x * gridDim.x;
    double local = 0.0;
    for (; i < nTot; i += sth) {
        if (flag[i] > 0) {
            local += ux[i];
        }
    }
    atomicAdd(sumQ, local);
}

__global__ void reduce_flow_Q_dom_kernel(
    const double* __restrict__ ux_A,
    const double* __restrict__ rho_A,
    const double* __restrict__ ux_B,
    const double* __restrict__ rho_B,
    const int*    __restrict__ flag,
    double rhoA_thr, double rhoB_thr,
    double*       sumA,
    double*       sumB,
    int nTot)
{
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int sth = blockDim.x * gridDim.x;
    double localA = 0.0, localB = 0.0;
    for (; i < nTot; i += sth) {
        if (flag[i] > 0) {
            if (rho_A[i] >= rhoA_thr) localA += ux_A[i];
            if (rho_B[i] >= rhoB_thr) localB += ux_B[i];
        }
    }
    atomicAdd(sumA, localA);
    atomicAdd(sumB, localB);
}

__global__ void reduce_flow_Q_dom_kernel_mask(
    const double* __restrict__ ux_A,
    const double* __restrict__ ux_B,
    const unsigned char* __restrict__ maskA_dom,
    const unsigned char* __restrict__ maskB_dom,
    double* sumA,
    double* sumB,
    int nTot)
{
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int sth = blockDim.x * gridDim.x;
    double localA = 0.0, localB = 0.0;
    for (; i < nTot; i += sth) {
        if (maskA_dom[i]) localA += ux_A[i];
        if (maskB_dom[i]) localB += ux_B[i];
    }
    atomicAdd(sumA, localA);
    atomicAdd(sumB, localB);
}

// === 总质量通量（Qtot）规约：∑(rhoA*uxA + rhoB*uxB) ===
__global__ void reduce_Qtot_kernel(
    const double* __restrict__ ux_A,
    const double* __restrict__ rho_A,
    const double* __restrict__ ux_B,
    const double* __restrict__ rho_B,
    const int*    __restrict__ flag,
    double*       sumQT,
    int nTot)
{
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int sth = blockDim.x * gridDim.x;
    double local = 0.0;
    for (; i < nTot; i += sth) {
        if (flag[i] > 0) {
            double rTot = fmax(rho_A[i] + rho_B[i], 1e-12);
            local += (rho_A[i]*ux_A[i] + rho_B[i]*ux_B[i]) / rTot;
        }
    }
    atomicAdd(sumQT, local);
}





// 2) 写“互斥占优掩码”
//    sB = rhoB/(rhoA+rhoB)； sB >= thr → B_dom=1； sB <= 1-thr → A_dom=1；其他置0
__global__ void dominant_mask_kernel(
    const double* __restrict__ rho_A,
    const double* __restrict__ rho_B,
    const int*    __restrict__ flag,
    double thr,                    // dom_ratio_thr
    unsigned char* maskA_dom,
    unsigned char* maskB_dom,
    int nTot)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int s = blockDim.x * gridDim.x;
    for (; i < nTot; i += s) {
        unsigned char a=0,b=0;
        if (flag[i] > 0) {
            double ra = rho_A[i], rb = rho_B[i];
            double denom = fabs(ra)+fabs(rb);
            if (denom > 0.0) {
                double sB = rb / denom;
                if (sB >= thr)      { b=1; }
                else if (sB <= 1.0 - thr) { a=1; }
            }
        }
        maskA_dom[i] = a;
        maskB_dom[i] = b;
    }
}

// 统计流体节点数（一次性）
__global__ void count_fluid_kernel(const int* __restrict__ flag, int* out, int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int s = blockDim.x * gridDim.x;
    int loc = 0;
    for (; i < n; i += s) if (flag[i] > 0) loc++;
    atomicAdd(out, loc);
}

// ========== SteadyMonitor impl ==========

SteadyMonitor::~SteadyMonitor() {
    if (d_sumA) cudaFree(d_sumA), d_sumA=nullptr;
    if (d_sumB) cudaFree(d_sumB), d_sumB=nullptr;
    if (d_sumQT) cudaFree(d_sumQT), d_sumQT=nullptr;
    if (d_maskA_dom) cudaFree(d_maskA_dom), d_maskA_dom=nullptr;
    if (d_maskB_dom) cudaFree(d_maskB_dom), d_maskB_dom=nullptr;
}

void SteadyMonitor::init_device_buffers() {
    if (!d_sumA) checkCudaErrors(cudaMalloc(&d_sumA, sizeof(double)));
    if (!d_sumB) checkCudaErrors(cudaMalloc(&d_sumB, sizeof(double)));
    if (!d_sumQT) checkCudaErrors(cudaMalloc(&d_sumQT, sizeof(double)));
}

void SteadyMonitor::reset() {
    consec_hit = 0; has_ref = false;
    QA_ref = QB_ref = 0.0;
}

void SteadyMonitor::prepare_domain(const Mix_dev& mix_dev) {
    if (nFluid >= 0) return;
    int *d_tmp = nullptr;
    checkCudaErrors(cudaMalloc(&d_tmp, sizeof(int)));
    checkCudaErrors(cudaMemset(d_tmp, 0, sizeof(int)));
    int n = NX*NY, BS=256, GS=(n+BS-1)/BS;
    count_fluid_kernel<<<GS,BS>>>(mix_dev.pointsflag, d_tmp, n);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(&nFluid, d_tmp, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_tmp);

    // 分配互斥占优掩码
    checkCudaErrors(cudaMalloc(&d_maskA_dom, sizeof(unsigned char)*n));
    checkCudaErrors(cudaMalloc(&d_maskB_dom, sizeof(unsigned char)*n));

}

//static inline double lerp(double a,double b,double t){ return a + t*(b-a); }

void SteadyMonitor::compute_Q_GPU(const Fluid_dev& A_dev,
                                  const Fluid_dev& B_dev,
                                  const Mix_dev&   mix_dev,
                                  double& QA, double& QB,
                                  double& QT) const // 新增

{
    // 先更新互斥占优掩码
    const int nTot = NX*NY;
    const int BS = 256, GS = (nTot + BS - 1) / BS;
    double thr = dom_ratio_thr;  // 建议作为 SteadyMonitor 的成员变量
    dominant_mask_kernel<<<GS,BS>>>(
        A_dev.rho, B_dev.rho, mix_dev.pointsflag,
        thr, d_maskA_dom, d_maskB_dom, nTot);
    checkCudaErrors(cudaDeviceSynchronize());

    // 清零累计和
    checkCudaErrors(cudaMemset(d_sumA, 0, sizeof(double)));
    checkCudaErrors(cudaMemset(d_sumB, 0, sizeof(double)));
    checkCudaErrors(cudaMemset(d_sumQT, 0, sizeof(double)));

    // 调用互斥掩码规约
    reduce_flow_Q_dom_kernel_mask<<<GS,BS>>>(
        A_dev.ux, B_dev.ux,
        d_maskA_dom, d_maskB_dom,
        d_sumA, d_sumB, nTot);
    checkCudaErrors(cudaDeviceSynchronize());

    reduce_Qtot_kernel<<<GS,BS>>>(
        A_dev.ux, A_dev.rho, B_dev.ux, B_dev.rho,
        mix_dev.pointsflag, d_sumQT, nTot);
    checkCudaErrors(cudaDeviceSynchronize());


    // 拷回并归一化
    double SA=0.0, SB=0.0, SQT=0.0;
    checkCudaErrors(cudaMemcpy(&SA, d_sumA, sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&SB, d_sumB, sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&SQT, d_sumQT, sizeof(double), cudaMemcpyDeviceToHost));
    QA = SA / (double)NX;
    QB = SB / (double)NX;
    QT = SQT / (double)NX;    // 通过出参返回总流量
}

bool SteadyMonitor::compare_and_update(double QA, double QB, double QT, int step) {
    auto rel = [&](double cur, double ref){
        if (fabs(ref) <= tol_abs && fabs(cur) <= tol_abs) return 0.0;
        double denom = (fabs(ref) > tol_abs) ? fabs(ref) : 1.0;
        return fabs(cur - ref) / denom;
    };
    if (!has_ref) {
        QA_ref = QA; QB_ref = QB; QT_ref = QT;
        has_ref = true;
        consec_hit = 0;
        printf("[steady] step=%d  INIT  QA=%.4e QB=%.4e QT=%.4e\n",
               step, QA, QB, QT);
        return false;
    }
    double rA = rel(QA, QA_ref);
    double rB = rel(QB, QB_ref);
    double rT = rel(QT, QT_ref);

    // 打印三者的当前值 + 相对变化
    printf("[steady] step=%d  QA=%.4e QB=%.4e QT=%.4e  dA=%.3e dB=%.3e dT=%.3e  hits=%d/%d\n",
           step, QA, QB, QT, rA, rB, rT, consec_hit, need_consec);

    // 参考值滚动更新（便于下一步继续计算 dA/dB/dT）
    QA_ref = QA; QB_ref = QB; QT_ref = QT;
    // —— 判稳只看总流量 QT —— //
    const bool passT = (rT < tol_rel);
    // 连续命中判据（更稳健）
    consec_hit = passT ? (consec_hit + 1) : 0;
    return (consec_hit >= need_consec);

    // 如果你希望“一次命中就算稳”，把上面两行换成：
    // return passT;
}

// --- SCMP single-phase flow rate computation ---
void SteadyMonitor::compute_Q_scmp(
    const double* ux, const double* rho,
    const int* pointsflag,
    double& Q, int nTot) const
{
    checkCudaErrors(cudaMemset(d_sumA, 0, sizeof(double)));
    const int BS = 256, GS = (nTot + BS - 1) / BS;
    reduce_Q_scmp_kernel<<<GS, BS>>>(ux, pointsflag, d_sumA, nTot);
    checkCudaErrors(cudaDeviceSynchronize());
    double sum;
    checkCudaErrors(cudaMemcpy(&sum, d_sumA, sizeof(double), cudaMemcpyDeviceToHost));
    Q = sum / (double)NX;
}

bool SteadyMonitor::compare_and_update_single(double Q, int step) {
    auto rel = [&](double cur, double ref){
        if (fabs(ref) <= tol_abs && fabs(cur) <= tol_abs) return 0.0;
        double denom = (fabs(ref) > tol_abs) ? fabs(ref) : 1.0;
        return fabs(cur - ref) / denom;
    };
    if (!has_ref) {
        QA_ref = Q; has_ref = true; consec_hit = 0;
        printf("[steady-scmp] step=%d  INIT  Q=%.4e\n", step, Q);
        return false;
    }
    double r = rel(Q, QA_ref);
    printf("[steady-scmp] step=%d  Q=%.4e  dQ=%.3e  hits=%d/%d\n",
           step, Q, r, consec_hit, need_consec);
    QA_ref = Q;
    if (r < tol_rel) {
        consec_hit++;
        if (consec_hit >= need_consec) {
            printf("[steady-scmp] converged at step %d  Q=%.4e\n", step, Q);
            return true;
        }
    } else { consec_hit = 0; }
    return false;
}

// ========== device globals (definitions) ==========
__device__ unsigned long long *g_ctA_any = nullptr;
__device__ unsigned long long *g_ctA_full= nullptr;
__device__ unsigned long long *g_ctB_any = nullptr;
__device__ unsigned long long *g_ctB_full= nullptr;
__device__ unsigned int *g_hitA = nullptr;
__device__ unsigned int *g_hitB = nullptr;
__device__ unsigned long long g_Nfluid = 0ULL;

__global__ void count_fluid_nodes_kernel(const int* __restrict__ flag,
                                         unsigned long long* __restrict__ out){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x>=NX || y>=NY) return;
    int idx = NX*y + x;
    if (flag[idx] >= 0) atomicAdd(out, 1ULL);
}

void SteadyMonitor::init_limiter_monitor(const int* d_pointsflag){
    // 1) 分配四个计数器（只分配一次）
    if (!d_ctA_any){
        CK(cudaMalloc(&d_ctA_any , sizeof(unsigned long long)));
        CK(cudaMalloc(&d_ctA_full, sizeof(unsigned long long)));
        CK(cudaMalloc(&d_ctB_any , sizeof(unsigned long long)));
        CK(cudaMalloc(&d_ctB_full, sizeof(unsigned long long)));
        // 可选热度图（按需打开）
        // CK(cudaMalloc(&d_hitA, NX*NY*sizeof(unsigned int)));
        // CK(cudaMalloc(&d_hitB, NX*NY*sizeof(unsigned int)));
    }

    // 2) 统计一次 Nfluid
    unsigned long long *d_tmp=nullptr;
    CK(cudaMalloc(&d_tmp, sizeof(unsigned long long)));
    CK(cudaMemset(d_tmp, 0, sizeof(unsigned long long)));
    dim3 th(32,8), gr((NX+th.x-1)/th.x, (NY+th.y-1)/th.y);
    count_fluid_nodes_kernel<<<gr,th>>>(d_pointsflag, d_tmp);
    CK(cudaGetLastError());
    CK(cudaMemcpy(&h_Nfluid, d_tmp, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CK(cudaFree(d_tmp));
    if (h_Nfluid==0) {
        fprintf(stderr, "[limiter] WARNING: h_Nfluid==0\n");
    }

    // 3) 把设备指针“发布”到 device globals（全局可见）
    CK(cudaMemcpyToSymbol(g_ctA_any , &d_ctA_any , sizeof(d_ctA_any )));
    CK(cudaMemcpyToSymbol(g_ctA_full, &d_ctA_full, sizeof(d_ctA_full)));
    CK(cudaMemcpyToSymbol(g_ctB_any , &d_ctB_any , sizeof(d_ctB_any )));
    CK(cudaMemcpyToSymbol(g_ctB_full, &d_ctB_full, sizeof(d_ctB_full)));
    //CK(cudaMemcpyToSymbol(g_hitA    , &d_hitA    , sizeof(d_hitA    )));
    //CK(cudaMemcpyToSymbol(g_hitB    , &d_hitB    , sizeof(d_hitB    )));
    CK(cudaMemcpyToSymbol(g_Nfluid  , &h_Nfluid  , sizeof(h_Nfluid  )));
}

void SteadyMonitor::reset_limiter_counters(cudaStream_t s) const {
    (void)s;  // 如果你继续用 cudaMemset 而不是 Async，避免未使用参数告警
    const size_t S = sizeof(unsigned long long);
    CK(cudaMemset(d_ctA_any , 0, S));
    CK(cudaMemset(d_ctA_full, 0, S));
    CK(cudaMemset(d_ctB_any , 0, S));
    CK(cudaMemset(d_ctB_full, 0, S));
    // if (d_hitA) CK(cudaMemset(d_hitA, 0, NX*NY*sizeof(unsigned int)));
    // if (d_hitB) CK(cudaMemset(d_hitB, 0, NX*NY*sizeof(unsigned int)));
}
static void append_limiter_csv_line(const std::string& path,
                                    int step, const LimiterStats& ls,
                                    unsigned long long Nfluid,
                                    bool& header_written)
{
    // 首次或空文件 → 写表头
    if (!header_written) {
        bool empty = true;
        { std::ifstream fin(path, std::ios::binary);
          if (fin) { fin.seekg(0,std::ios::end); empty = (fin.tellg()==0); } }
        std::ofstream out(path, std::ios::app);
        if (empty) out << "step,A_any,A_full,B_any,B_full,rA_any,rA_full,rB_any,rB_full,Nfluid\n";
        header_written = true;
    }
    std::ofstream out(path, std::ios::app);
    out.setf(std::ios::scientific);
    out << step << ","
        << ls.A_any  << "," << ls.A_full << ","
        << ls.B_any  << "," << ls.B_full << ","
        << ls.rA_any << "," << ls.rA_full << ","
        << ls.rB_any << "," << ls.rB_full << ","
        << Nfluid    << "\n";
}

LimiterStats SteadyMonitor::fetch_limiter_stats_and_log(int step,
                                                        double print_thr,
                                                        bool write_file) const
{
    LimiterStats ls{};
    CK(cudaMemcpy(&ls.A_any , d_ctA_any , sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&ls.A_full, d_ctA_full, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&ls.B_any , d_ctB_any , sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&ls.B_full, d_ctB_full, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    const double N = (h_Nfluid>0) ? (double)h_Nfluid : (double)(NX*NY);
    ls.rA_any  = ls.A_any  / N;  ls.rA_full = ls.A_full / N;
    ls.rB_any  = ls.B_any  / N;  ls.rB_full = ls.B_full / N;

    const double thr = (print_thr >= 0.0 ? print_thr : limiter_log_thr);
    const bool hit   = (ls.rA_any>thr || ls.rB_any>thr || ls.A_full>0 || ls.B_full>0);

    // 控制台：按开关
    if (limiter_stdout && hit){
        printf("[limiter @%d] anyA=%.3e fullA=%.3e | anyB=%.3e fullB=%.3e (Nfluid=%llu)\n",
               step, ls.rA_any, ls.rA_full, ls.rB_any, ls.rB_full,
               (unsigned long long)h_Nfluid);
    }

    // 文件：按频率/或命中（命中是否启用由 limiter_log_on_hit 决定）
    const int every = (limiter_log_every > 0 ? limiter_log_every : limiter_window);
    const bool write_by_freq = (every > 0) && ((step % every) == 0);
    const bool write_by_hit  = limiter_log_on_hit && hit;

    if (write_file && (write_by_freq || write_by_hit)) {
        append_limiter_csv_line(limiter_log_path, step, ls, h_Nfluid,
                                limiter_csv_header_written);
    }
    return ls;
}
