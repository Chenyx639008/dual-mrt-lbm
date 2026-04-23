#! comment: This script compiles the CUDA code for the LBM simulation.
#!/usr/bin/env bash
set -e  # 出错立即退出

###############################################################################
# 根据你的 GPU 算力把 sm_XY 改成对应值：

###############################################################################
#-Xptxas -v 看常量内存

# ===========================================================================
# 构建目标选择：
#   bash compile.sh           → flow-only 原版  (mcmp_sim)
#   bash compile.sh hydrate   → 水合物完整版    (mcmp_sim_hydrate)
# ===========================================================================

TARGET=${1:-””}

if [ “$TARGET” = “hydrate” ]; then
  nvcc -std=c++17 -O3 -rdc=true -lineinfo \
    -DHYDRATE_ENABLE \
    -gencode arch=compute_120,code=sm_120 \
    -gencode arch=compute_120,code=compute_120 \
    main.cu LBM.cu steady_monitor.cu sim_utils.cu \
    hydrate.cu hydrate_vop.cu \
    -o mcmp_sim_hydrate
else
  nvcc -std=c++17 -O3 -rdc=true -lineinfo \
    -gencode arch=compute_120,code=sm_120 \
    -gencode arch=compute_120,code=compute_120 \
    main.cu LBM.cu steady_monitor.cu sim_utils.cu -o mcmp_sim
fi



#没有 -rdc=true，编译器不会做”设备链接”。当你的 __constant__/__device__
#符号定义在 LBM.cu，但在另一份 sim_utils.cu 里用 cudaMemcpyToSymbol 跨文件访问它时，
#就会在运行时出现 invalid device symbol