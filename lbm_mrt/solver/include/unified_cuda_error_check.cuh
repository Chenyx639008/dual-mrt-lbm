/************************************************************
*  unified_cuda_error_check.cuh
*  ---------------------------------------------------------
*  同时支持  ⬇                                             
*      checkCudaErrors(...) / getLastCudaError(...)  ← 教材
*      CUDA_CHECK(...)      / KCALL(...)             ← 旧版
**********************************************************/
#pragma once
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include "LBM.h"


/* ------------ 0. 内部实现函数 ------------ */
inline void __checkCudaErrors(cudaError_t err,
                              const char* func,
                              const char* file,
                              int         line)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "CUDA error at %s(%d) in \"%s\": [%d] %s\n",
                file, line, func, int(err), cudaGetErrorString(err));
        std::exit(-1);
    }
}

// 检查数组中是否存在 NaN、Inf 等非法值
inline void check_nan_all(const double* arr, const char* name) {
    for (int i = 0; i < NX * NY; ++i) {
        if (!std::isfinite(arr[i])) {
            std::cout << "[NaN detected] Array: " << name
                      << ", Index: " << i
                      << ", Value: " << arr[i]
                      << ", (x,y)=(" << i % NX << "," << i / NX << ")\n";
            break; // 只报第一个
        }
    }
}


inline void __getLastCudaError(const char* msg,
                               const char* file,
                               int         line)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "CUDA error at %s(%d) after %s: [%d] %s\n",
                file, line, msg, int(err), cudaGetErrorString(err));
        std::exit(-1);
    }
}

/* ------------ 1. 公共宏接口 ------------ */
/* ——教材版—— */
#define checkCudaErrors(err)  __checkCudaErrors((err),#err,__FILE__,__LINE__)
#define getLastCudaError(msg) __getLastCudaError((msg),__FILE__,__LINE__)

/* ——旧版别名—— */
#define CUDA_CHECK(err)       checkCudaErrors(err)

/* ------------ 2. NaN 调试辅助（可选） ------------ */
#ifndef DISABLE_NAN_CHECK
inline bool nan_guard(double* dptr_host,       // 只需 1 个双精度缓冲
                      const double* dptr_dev,  // 设备地址
                      const char*   tag)
{
    checkCudaErrors(cudaMemcpy(dptr_host, dptr_dev,
                               sizeof(double), cudaMemcpyDeviceToHost));
    if (!std::isfinite(dptr_host[0]))
    {
        printf("[NaN detected] after %s  value=%e\n", tag, dptr_host[0]);
        return true;
    }
    return false;
}
#endif

/* ------------ 3. Kernel 一键包装 ------------ */
/*  依赖全局 dim3 grid, threads；如名称不同请自行替换           */
/*  NaN 检测默认开启，若不需要可在编译前 #define DISABLE_NAN_CHECK */
/* ---------- 预处理条件，放在宏外面 ---------- */
#ifdef DISABLE_NAN_CHECK
    #define _NAN_GUARD(stmt)  /* 空宏，什么都不做 */
#else
    #define _NAN_GUARD(stmt)  do{ stmt; }while(0)
#endif

/* ---------- 核函数包装宏 ---------- 
#define KCALL(kernel_call, kernel_name_str)                    \
    do {                                                       \
        kernel_call;                                           \
        getLastCudaError(kernel_name_str);                     \
        checkCudaErrors(cudaDeviceSynchronize());              \
        _NAN_GUARD(                                            \
            extern double* _nan_chk_buf_host;                  \
            extern double* _nan_chk_probe_dev;                 \
            if (nan_guard(_nan_chk_buf_host,                   \
                          _nan_chk_probe_dev,                  \
                          kernel_name_str)) return;            \
        );                                                    \
    } while (0)
*/

#define KCALL(kernel, args)                                   \
    do {                                                      \
        kernel<<<grid, threads>>>args;                        \
        getLastCudaError(#kernel " launch");                  \
        checkCudaErrors(cudaDeviceSynchronize());             \
        _NAN_GUARD(                                           \
            extern double* _nan_chk_buf_host;                 \
            extern double* _nan_chk_probe_dev;                \
            if (nan_guard(_nan_chk_buf_host,                  \
                          _nan_chk_probe_dev,                 \
                          #kernel)) return;                   \
        );                                                    \
    } while (0)

