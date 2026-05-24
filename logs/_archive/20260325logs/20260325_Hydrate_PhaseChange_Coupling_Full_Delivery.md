 ▐▛███▜▌   Claude Code v2.1.81
▝▜█████▛▘  Sonnet 4.6 · API Usage Billing
  ▘▘ ▝▝    ~/projects/lbm_twoflow/complex_8.23/lbm_template

✻ Conversation compacted (ctrl+o for history)

  ⎿  Read hydrate.cu (517 lines)
  ⎿  Read hydrate.h (237 lines)
  ⎿  Read hydrate_vop.cu (24 lines)
  ⎿  Read compile.sh (38 lines)
  ⎿  Read LBM.h (342 lines)
  ⎿  Plan file referenced (~/.claude/plans/pure-moseying-ritchie.md)

  Searched for 1 pattern, read 1 file (ctrl+o to expand)

● Good, I have enough context. Now I'll implement Phase 2. The plan:

  1. Add hydrate param parsing to sim_utils.cu load_params_txt
  2. Replace stubs in hydrate.cu with full concentration field implementation

  Let me start with both in parallel:

● Now add hydrate param parsing to sim_utils.cu after the existing
  getb("ENABLE_CKPT", ...) line:

● Update(sim_utils.cu)
  ⎿  Added 27 lines
      353                                                                
      354                                                                
      355      getb("ENABLE_CKPT",       r.ENABLE_CKPT);                 
      356 +                                                              
      357 +#ifdef HYDRATE_ENABLE                                         
      358 +    // 水合物相变参数                                         
      359 +    getb("hydrate_enable",        r.hydrate_enable);          
      360 +    geti("hydrate_start_step",    r.hydrate_start_step);      
      361 +    get("T0_init",                r.T0_init);                 
      362 +    get("T0_inlet",               r.T0_inlet);                
      363 +    get("lambda_fluid",           r.lambda_fluid);            
      364 +    get("lambda_hydrate",         r.lambda_hydrate);          
      365 +    get("lambda_solid",           r.lambda_solid);            
      366 +    get("rhocp_fluid",            r.rhocp_fluid);             
      367 +    get("rhocp_hydrate",          r.rhocp_hydrate);           
      368 +    get("rhocp_solid",            r.rhocp_solid);             
      369 +    get("D_mol_water",            r.D_mol_water);             
      370 +    get("Henry_KH",               r.Henry_KH);                
      371 +    get("Cm_init",                r.Cm_init);                 
      372 +    get("k0_rxn",                 r.k0_rxn);                  
      373 +    get("Ea_rxn",                 r.Ea_rxn);                  
      374 +    get("e1_peq",                 r.e1_peq);                  
      375 +    get("e2_peq",                 r.e2_peq);                  
      376 +    get("latent_heat",            r.latent_heat);             
      377 +    get("Vm_hydrate",             r.Vm_hydrate);              
      378 +    get("Vh_init",                r.Vh_init);                 
      379 +    get("vop_terminate_frac",     r.vop_terminate_frac);      
      380 +    get("dx_phys",                r.dx_phys);                 
      381 +    get("dt_phys",                r.dt_phys);                 
      382 +#endif                                                        
      383      // ============ 应用 init_eq                              
           预设（在所有普通键读完之后执行） ============                 
      384      // 约定：init_eq 为 1 或 2 时，以下组合将覆盖相关字段；   
      385      //double rhoA_hi = 6.6293, rhoA_lo = 0.34127;             

● Now implement the full concentration field in hydrate.cu. I'll replace the
  stub sections for Phase 2 (§11) and also fill in the
  compute_latent_heat_source (§10):

● Update(hydrate.cu)
  ⎿  Added 274 lines, removed 8 lines
      452  }                                                             
      453                                                                
      454  //                                                            
           ============================================================  
      455 -// §11  浓度场（Phase 2 桩）                                  
      455 +// §11  浓度场（Phase 2 实现）                                
      456  //                                                            
           ============================================================  
      457 +                                                              
      458 +//                                                            
          +------------------------------------------------------------  
      459 +// §11.1  宏观浓度更新                                        
      460 +//                                                            
          +------------------------------------------------------------  
      461 +__global__ void kernel_update_Cm(const double* g_in, double*  
          +Cm,                                                           
      462 +                                   const int* pointsflag)     
      463 +{                                                             
      464 +    int x = blockIdx.x * blockDim.x + threadIdx.x;            
      465 +    int y = blockIdx.y * blockDim.y + threadIdx.y;            
      466 +    if (x >= NX || y >= NY) return;                           
      467 +                                                              
      468 +    const size_t s = idx_scalar(x, y);                        
      469 +    double sum = 0.0;                                         
      470 +    for (int k = 0; k < Q5; ++k)                              
      471 +        sum += g_in[idx5(k, x, y)];                           
      472 +    Cm[s] = sum;                                              
      473 +}                                                             
      474 +                                                              
      475 +//                                                            
          +------------------------------------------------------------  
      476 +// §11.2  初始化浓度场（均匀 Cm_init）                        
      477 +//                                                            
          +------------------------------------------------------------  
      478 +__global__ void kernel_init_conc(double* g_in, double* Cm,    
      479 +                                   const int* pointsflag)     
      480 +{                                                             
      481 +    int x = blockIdx.x * blockDim.x + threadIdx.x;            
      482 +    int y = blockIdx.y * blockDim.y + threadIdx.y;            
      483 +    if (x >= NX || y >= NY) return;                           
      484 +                                                              
      485 +    const size_t s  = idx_scalar(x, y);                       
      486 +    const double C0 = d_Cm_init;                              
      487 +                                                              
      488 +    Cm[s] = C0;                                               
      489 +    for (int k = 0; k < Q5; ++k)                              
      490 +        g_in[idx5(k, x, y)] = w5_gpu[k] * C0;  // g_eq(Cm,    
          +u=0)                                                          
      491 +}                                                             
      492 +                                                              
      493 +//                                                            
          +------------------------------------------------------------  
      494 +// §11.3  D2Q5 MRT 碰撞核函数（浓度场）                       
      495 +//                                                            
      496 +// 与热场碰撞相同结构，只是：                                 
      497 +//   ωD = 1/(0.5 + D_latt/cs2_5)  （均匀扩散，流体节点）      
      498 +//   m_eq = [Cm, Cm·ux, Cm·uy, 0.75*Cm, 0]                    
      499 +//   CST 源项（气-水界面）：在守恒量 m0 加入松弛到 Henry 平衡 
      500 +//     气-水界面判据：rhoB/(rhoA+rhoB) > theta_iface（0.3）   
      501 +//     源项形式（对流-扩散方程 BGK 型）：                     
      502 +//       S_cst = -omega_cst * (Cm - Ceq_latt)                 
          +（调整后直接加到 m0）                                         
      503 +//       Ceq_latt = KH * Cg = KH * exp(e1 + e2/T)  [格子单位] 
      504 +//     其中 omega_cst 可取 1（强制平衡）或小值（弱松弛）；    
      505 +//     此处取 omega_cst = 1（在 CST                           
          +界面节点一步直接平衡，Yang 2024 §S3）                         
      506 +//                                                            
          +------------------------------------------------------------  
      507 +__global__ void kernel_collide_conc(                          
      508 +    const double* __restrict__ g_in,                          
      509 +          double* __restrict__ g_out,                         
      510 +    const double* __restrict__ Cm,                            
      511 +    const double* __restrict__ T,                             
      512 +    const double* __restrict__ ux_mix,                        
      513 +    const double* __restrict__ uy_mix,                        
      514 +    const double* __restrict__ rho_A,                         
      515 +    const double* __restrict__ rho_B,                         
      516 +    const int*    __restrict__ pointsflag)                    
      517 +{                                                             
      518 +    int x = blockIdx.x * blockDim.x + threadIdx.x;            
      519 +    int y = blockIdx.y * blockDim.y + threadIdx.y;            
      520 +    if (x >= NX || y >= NY) return;                           
      521 +                                                              
      522 +    const size_t s  = idx_scalar(x, y);                       
      523 +    const int    fl = pointsflag[s];                          
      524 +                                                              
      525 +    // ghost 节点：仅复制（交由边界核函数处理）               
      526 +    if (fl == -1) {                                           
      527 +        for (int k = 0; k < Q5; ++k)                          
      528 +            g_out[idx5(k, x, y)] = g_in[idx5(k, x, y)];       
      529 +        return;                                               
      530 +    }                                                         
      531 +                                                              
      532 +    // 弛豫率（仅流体区域有意义；固/水合物内部可不参与，但复  
          +制确保数组有效）                                              
      533 +    const double omegaD = 1.0 / (0.5 + d_D_latt / cs2_5);     
      534 +                                                              
      535 +    double g[Q5];                                             
      536 +    for (int k = 0; k < Q5; ++k)                              
      537 +        g[k] = g_in[idx5(k, x, y)];                           
      538 +                                                              
      539 +    const double Ci  = Cm[s];                                 
      540 +    const double uxi = ux_mix[s];                             
      541 +    const double uyi = uy_mix[s];                             
      542 +                                                              
      543 +    // 变换到矩空间（与热场完全相同的 M5）                    
      544 +    double m0 = g[0] + g[1] + g[2] + g[3] + g[4];             
      545 +    double m1 = g[1] - g[2];                                  
      546 +    double m2 = g[3] - g[4];                                  
      547 +    double m3 = -4.0*g[0] + g[1] + g[2] + g[3] + g[4];        
      548 +    double m4 = g[1] + g[2] - g[3] - g[4];                    
      549 +                                                              
      550 +    // 平衡矩                                                 
      551 +    double meq0 = Ci;                                         
      552 +    double meq1 = Ci * uxi;                                   
      553 +    double meq2 = Ci * uyi;                                   
      554 +    double meq3 = 0.75 * Ci;                                  
      555 +    double meq4 = 0.0;                                        
      556 +                                                              
      557 +    // CST 源项（Henry 平衡，仅在气-水界面流体节点）          
      558 +    //   Ceq_latt = KH * exp(e1_peq + e2_peq / T)             
      559 +    //   若 rhoB/(rhoA+rhoB) > 0.3，视为界面节点，强制松弛到  
          +Ceq                                                           
      560 +    double src_cst = 0.0;                                     
      561 +    if (fl == 1) {                                            
      562 +        const double rA = rho_A[s];                           
      563 +        const double rB = rho_B[s];                           
      564 +        const double rT = rA + rB;                            
      565 +        if (rT > 1e-12 && rB / rT > 0.3) {                    
      566 +            const double Ti    = T[s];                        
      567 +            const double Ceq   = d_Henry_KH * exp(d_e1_peq +  
          +d_e2_peq / Ti);                                               
      568 +            // omega_cst = 1 → 一步全松弛                     
      569 +            src_cst = Ceq - Ci;  // 直接修正 m0（守恒量）     
      570 +        }                                                     
      571 +    }                                                         
      572 +                                                              
      573 +    // MRT 弛豫                                               
      574 +    double mo0 = m0 - 1.0    * (m0 - meq0) + src_cst;         
      575 +    double mo1 = m1 - omegaD * (m1 - meq1);                   
      576 +    double mo2 = m2 - omegaD * (m2 - meq2);                   
      577 +    double mo3 = m3 - 1.0    * (m3 - meq3);                   
      578 +    double mo4 = m4 - 1.0    * (m4 - meq4);                   
      579 +                                                              
      580 +    // 逆变换（Minv5，与热场相同）                            
      581 +    const double inv5  = 1.0/5.0;                             
      582 +    const double inv20 = 1.0/20.0;                            
      583 +    const double half  = 0.5;                                 
      584 +    const double qtr   = 0.25;                                
      585 +                                                              
      586 +    g_out[idx5(0, x, y)] = inv5*mo0                    -      
          +inv5*mo3;                                                     
      587 +    g_out[idx5(1, x, y)] = inv5*mo0 + half*mo1         +      
          +inv20*mo3 + qtr*mo4;                                          
      588 +    g_out[idx5(2, x, y)] = inv5*mo0 - half*mo1         +      
          +inv20*mo3 + qtr*mo4;                                          
      589 +    g_out[idx5(3, x, y)] = inv5*mo0            + half*mo2 +   
          +inv20*mo3 - qtr*mo4;                                          
      590 +    g_out[idx5(4, x, y)] = inv5*mo0            - half*mo2 +   
          +inv20*mo3 - qtr*mo4;                                          
      591 +}                                                             
      592 +                                                              
      593 +//                                                            
          +------------------------------------------------------------  
      594 +// §11.4  D2Q5 流步核函数（浓度场）                           
      595 +//                                                            
          +------------------------------------------------------------  
      596 +__global__ void kernel_stream_conc(                           
      597 +          double* __restrict__ g_in,                          
      598 +    const double* __restrict__ g_out,                         
      599 +    const int*    __restrict__ pointsflag)                    
      600 +{                                                             
      601 +    int x = blockIdx.x * blockDim.x + threadIdx.x;            
      602 +    int y = blockIdx.y * blockDim.y + threadIdx.y;            
      603 +    if (x >= NX || y >= NY) return;                           
      604 +                                                              
      605 +    for (int k = 0; k < Q5; ++k) {                            
      606 +        int xp = (x - e5_gpu[k][0] + NX) % NX;                
      607 +        int yp = (y - e5_gpu[k][1] + NY) % NY;                
      608 +        g_in[idx5(k, x, y)] = g_out[idx5(k, xp, yp)];         
      609 +    }                                                         
      610 +}                                                             
      611 +                                                              
      612 +//                                                            
          +------------------------------------------------------------  
      613 +// §11.5  Kang 方案反应边界 + 出入口边界（浓度场）            
      614 +//                                                            
      615 +// 处理三类 ghost/boundary：                                  
      616 +//   y==0  (入口 ghost, fl==-1)：Cm = Cm_init  → g_eq         
      617 +//   y==NY-1 (出口 ghost, fl==-1)：全展开（copy from y=NY-2） 
      618 +//   水合物面 ghost (fl==-1, mat==2)：Kang 方案 Cm_bc         
      619 +//     Cm_bc = (D_latt * Cm_nbr + k_r * Csat * dx_latt)       
      620 +//             / (D_latt + k_r * dx_latt)                     
      621 +//     其中 Csat = exp(e1+e2/T)（无量纲"溶解平衡" Henry 侧）  
      622 +//     dx_latt = 1                                            
      623 +//   同时将反应速率写入 diss_rate（格子单位                   
          +mol/(格子²·格子时间)）                                        
      624 +//     diss_rate = k_r * (1 - Cm_nbr/Csat)   （当 Cm<Csat     
          +时才分解）                                                    
      625 +//                                                            
          +------------------------------------------------------------  
      626 +__global__ void kernel_boundary_conc_reaction(                
      627 +          double* __restrict__ g_in,                          
      628 +    const double* __restrict__ g_out,                         
      629 +    const double* __restrict__ Cm,                            
      630 +    const double* __restrict__ T,                             
      631 +          double* __restrict__ diss_rate,                     
      632 +    const int*    __restrict__ pointsflag)                    
      633 +{                                                             
      634 +    int x = blockIdx.x * blockDim.x + threadIdx.x;            
      635 +    int y = blockIdx.y * blockDim.y + threadIdx.y;            
      636 +    if (x >= NX || y >= NY) return;                           
      637 +                                                              
      638 +    const size_t s  = idx_scalar(x, y);                       
      639 +    const int    fl = pointsflag[s];                          
      640 +                                                              
      641 +    if (fl != -1) return;  // 只处理 ghost 节点               
      642 +                                                              
      643 +    // ---- 入口：y==0，Dirichlet Cm_init ----                
      644 +    if (y == 0) {                                             
      645 +        for (int k = 0; k < Q5; ++k)                          
      646 +            g_in[idx5(k, x, y)] = w5_gpu[k] * d_Cm_init;      
      647 +        return;                                               
      648 +    }                                                         
      649 +                                                              
      650 +    // ---- 出口：y==NY-1，全展开 ----                        
      651 +    if (y == NY - 1) {                                        
      652 +        for (int k = 0; k < Q5; ++k)                          
      653 +            g_in[idx5(k, x, y)] = g_in[idx5(k, x, NY - 2)];   
      654 +        return;                                               
      655 +    }                                                         
      656 +                                                              
      657 +    // ---- 水合物/固体 ghost（内部边界）----                 
      658 +    const unsigned char mat = d_wall_mat[s];                  
      659 +    if (mat == 0) return;  //                                 
          +普通固体（石英）：绝质边界（全反弹）                          
      660 +                                                              
      661 +    if (mat == 2) {                                           
      662 +        // 水合物面 → Kang 方案                               
      663 +        // 找最近流体邻居（沿 +y 方向优先，然后其他方向）     
      664 +        // 简化实现：取邻居中第一个流体节点的 Cm 和 T         
      665 +        double Cm_nbr = d_Cm_init;                            
      666 +        double T_nbr  = d_T0_init;                            
      667 +        bool   found  = false;                                
      668 +        for (int k = 1; k < Q5 && !found; ++k) {  // k=0      
          +静止跳过                                                      
      669 +            int xn = x + e5_gpu[k][0];                        
      670 +            int yn = y + e5_gpu[k][1];                        
      671 +            if (xn < 0 || xn >= NX || yn < 0 || yn >= NY)     
          +continue;                                                     
      672 +            size_t sn = idx_scalar(xn, yn);                   
      673 +            if (pointsflag[sn] == 1) {                        
      674 +                Cm_nbr = Cm[sn];                              
      675 +                T_nbr  = T[sn];                               
      676 +                found  = true;                                
      677 +            }                                                 
      678 +        }                                                     
      679 +                                                              
      680 +        const double Ti    = T_nbr;                           
      681 +        // 平衡溶解浓度（无单位校正，直接用格子温度 K）       
      682 +        const double Csat  = __expf((float)(d_e1_peq +        
          +d_e2_peq / Ti));                                              
      683 +        // Kim-Bishnoi 速率（格子单位）                       
      684 +        const double k_r   = d_k0_latt * exp(-d_Ea_over_R /   
          +Ti)                                                           
      685 +                             * fmax(0.0, 1.0 - Cm_nbr / (Csat 
          + + 1e-30));                                                   
      686 +        // Kang BC 浓度                                       
      687 +        // Cm_bc = (D·Cm_nbr + k_r·Csat) / (D + k_r)          
          +(dx_latt=1)                                                   
      688 +        const double D     = d_D_latt;                        
      689 +        const double Cm_bc = (D * Cm_nbr + k_r * Csat) / (D + 
          + k_r + 1e-30);                                                
      690 +                                                              
      691 +        // 写分解速率到 diss_rate                             
      692 +        diss_rate[s] = k_r * fmax(0.0, 1.0 - Cm_nbr / (Csat + 
          + 1e-30));                                                     
      693 +                                                              
      694 +        // 将 g_in 设为均衡分布（对应 Cm_bc, u=0）            
      695 +        for (int k = 0; k < Q5; ++k)                          
      696 +            g_in[idx5(k, x, y)] = w5_gpu[k] * Cm_bc;          
      697 +    } else {                                                  
      698 +        //                                                    
          +普通固体（石英等）：全反弹，不参与反应，diss_rate=0           
      699 +        for (int k = 0; k < Q5; ++k)                          
      700 +            g_in[idx5(k, x, y)] = g_out[idx5(opp5_gpu[k], x,  
          +y)];                                                          
      701 +        diss_rate[s] = 0.0;                                   
      702 +    }                                                         
      703 +}                                                             
      704 +                                                              
      705 +//                                                            
          +------------------------------------------------------------  
      706 +// §11.6  宿主函数：每步浓度场演化                            
      707 +//                                                            
          +------------------------------------------------------------  
      708  void init_conc_field(Conc_dev& CN, const int* pointsflag)     
      709  {                                                             
      710 -    // Phase 2 实现                                           
      711 -    CUDA_CHECK(cudaMemset(CN.g_in,  0, mem_size_D2Q5));       
      712 -    CUDA_CHECK(cudaMemset(CN.g_out, 0, mem_size_D2Q5));       
      713 -    CUDA_CHECK(cudaMemset(CN.Cm,    0, mem_size_scalar));     
      710 +    kernel_init_conc<<<grid, threads>>>(CN.g_in, CN.Cm,       
          +pointsflag);                                                  
      711 +    CUDA_CHECK(cudaGetLastError());                           
      712 +    CUDA_CHECK(cudaDeviceSynchronize());                      
      713  }                                                             
      714                                                                
      715  void step_conc(Conc_dev& CN, const Therm_dev& TH,             
     ...
      467                 const double* rho_A, const double* rho_B,      
      468                 VOP_dev& VP, const int* pointsflag)            
      469  {                                                             
      470 -    // Phase 2 实现：碰撞→流→Kang 边界→更新 Cm→写 diss_rate   
      471 -    (void)CN; (void)TH; (void)ux_mix; (void)uy_mix;           
      472 -    (void)rho_A; (void)rho_B; (void)VP; (void)pointsflag;     
      470 +    // 1) 宏观浓度更新                                        
      471 +    kernel_update_Cm<<<grid, threads>>>(CN.g_in, CN.Cm,       
          +pointsflag);                                                  
      472 +    CUDA_CHECK(cudaGetLastError());                           
      473 +                                                              
      474 +    // 2) MRT 碰撞（含 CST Henry 源项）                       
      475 +    kernel_collide_conc<<<grid, threads>>>(                   
      476 +        CN.g_in, CN.g_out, CN.Cm, TH.T,                       
      477 +        ux_mix, uy_mix, rho_A, rho_B, pointsflag);            
      478 +    CUDA_CHECK(cudaGetLastError());                           
      479 +                                                              
      480 +    // 3) 流步                                                
      481 +    kernel_stream_conc<<<grid, threads>>>(CN.g_in, CN.g_out,  
          +pointsflag);                                                  
      482 +    CUDA_CHECK(cudaGetLastError());                           
      483 +                                                              
      484 +    // 4) Kang 反应边界 + 入/出口边界（同时计算 diss_rate）   
      485 +    kernel_boundary_conc_reaction<<<grid, threads>>>(         
      486 +        CN.g_in, CN.g_out, CN.Cm, TH.T,                       
      487 +        VP.diss_rate, pointsflag);                            
      488 +    CUDA_CHECK(cudaGetLastError());                           
      489  }                                                             
      490                                                                
      491  //                                                            
           ============================================================  

● Now implement compute_latent_heat_source (§10) — replace the memset stub:

● Update(hydrate.cu)
  ⎿  Added 39 lines, removed 4 lines
      439  }                                                             
      440                                                                
      441  //                                                            
           ============================================================  
      442 -// §10  潜热源项（Phase 4，桩 —— Phase 4 实现时填充）         
      442 +// §10  潜热源项（Phase 4 实现）                              
      443  //                                                            
           ============================================================  
      444 +// S_latent[idx] = -ΔH_latt * diss_rate[idx]                  
      445 +//   diss_rate 已由 kernel_boundary_conc_reaction 在水合物    
          +ghost 节点写入                                                
      446 +//   ΔH_latt = latent_heat / (rhocp_fluid * dx)  [K·格子]     
          +（在 init_device_variable_hydrate 中计算）                    
      447 +// 仅在水合物面 ghost                                         
          +节点（mat==2，fl==-1）非零；流体节点接收该源项                
      448 +                                                              
      449 +__global__ void kernel_compute_latent_heat(                   
      450 +          double* __restrict__ S_latent,                      
      451 +    const double* __restrict__ diss_rate,                     
      452 +    const int*    __restrict__ pointsflag)                    
      453 +{                                                             
      454 +    int x = blockIdx.x * blockDim.x + threadIdx.x;            
      455 +    int y = blockIdx.y * blockDim.y + threadIdx.y;            
      456 +    if (x >= NX || y >= NY) return;                           
      457 +                                                              
      458 +    const size_t s = idx_scalar(x, y);                        
      459 +    // 仅流体节点接收潜热：遍历邻居 ghost 节点累加            
      460 +    if (pointsflag[s] != 1) {                                 
      461 +        S_latent[s] = 0.0;                                    
      462 +        return;                                               
      463 +    }                                                         
      464 +                                                              
      465 +    double src = 0.0;                                         
      466 +    for (int k = 1; k < Q5; ++k) {                            
      467 +        int xn = x + e5_gpu[k][0];                            
      468 +        int yn = y + e5_gpu[k][1];                            
      469 +        if (xn < 0 || xn >= NX || yn < 0 || yn >= NY)         
          +continue;                                                     
      470 +        const size_t sn = idx_scalar(xn, yn);                 
      471 +        if (pointsflag[sn] == -1 && d_wall_mat[sn] == 2) {    
      472 +            // 相邻水合物 ghost 节点的分解速率贡献潜热        
      473 +            src -= d_latent_H_latt * diss_rate[sn];           
      474 +        }                                                     
      475 +    }                                                         
      476 +    S_latent[s] = src;                                        
      477 +}                                                             
      478 +                                                              
      479  void compute_latent_heat_source(VOP_dev& VP,                  
      480                                  const Conc_dev& CN,           
      481                                  const Therm_dev& TH,          
      482                                  const int* pointsflag)        
      483  {                                                             
      484 -    // Phase 4 实现：S_latent[idx] = -ΔH * diss_rate[idx] *   
          -(1 - Cm/Csat) / rhocp                                         
      485 -    // 暂时清零                                               
      486 -    CUDA_CHECK(cudaMemset(VP.S_latent, 0, mem_size_scalar));  
      484 +    kernel_compute_latent_heat<<<grid, threads>>>(            
      485 +        VP.S_latent, VP.diss_rate, pointsflag);               
      486 +    CUDA_CHECK(cudaGetLastError());                           
      487  }                                                             
      488                                                                
      489  //                                                            
           ============================================================  

● Now compile and check for errors:

● Bash(bash compile.sh hydrate 2>&1 | tail -40)
  ⎿  (No output)
  ⎿  (timeout 2m)

● Clean build. Let me verify the binary exists and check the new symbols:

● Bash(ls -lh mcmp_sim_hydrate && nm mcmp_sim_hydrate | grep -E
      "kernel_collide_conc|kernel_boundary_conc|kernel_compute_latent|kernel_upd
      ate_Cm|kernel_init_conc|kerne…)
  ⎿  -rwx------ 1 server server 2.7M  3月 25 19:19 mcmp_sim_hydrate
     000000000002c9e0 T _Z16kernel_init_concPdS_PKi
     000000000002c900 T _Z16kernel_update_CmPKdPdPKi
     … +15 lines (ctrl+o to expand)

● All 6 new Phase 2 kernels compiled successfully. Let me also verify flow-only
  still compiles clean:

● Bash(bash compile.sh 2>&1 | tail -5 && ls -lh mcmp_sim)
  ⎿  -rwx------ 1 server server 2.5M  3月 25 19:19 mcmp_sim
  ⎿  (timeout 2m)

● Both builds pass. Let me do a final review of the concentration field code to
  check for any issues:

  Read 1 file (ctrl+o to expand)

● Phase 2 is complete. Here's what was implemented:

  sim_utils.cu — added hydrate param parsing to load_params_txt (all 24 params
  inside #ifdef HYDRATE_ENABLE).

  hydrate.cu — 6 new kernels replacing the stubs:

  ┌───────────────────────────────┬──────────────────────────────────────────┐
  │             函数              │                   功能                   │
  ├───────────────────────────────┼──────────────────────────────────────────┤
  │ kernel_init_conc              │ 均匀初始化 g_in = w5[k]·Cm_init          │
  ├───────────────────────────────┼──────────────────────────────────────────┤
  │ kernel_update_Cm              │ Cm = Σ g_in[k]                           │
  ├───────────────────────────────┼──────────────────────────────────────────┤
  │                               │ D2Q5 MRT 碰撞；CST                       │
  │ kernel_collide_conc           │ 源项：气-水界面（rhoB/rhoT >             │
  │                               │ 0.3）强制拉向 Ceq = KH·exp(e1+e2/T)      │
  ├───────────────────────────────┼──────────────────────────────────────────┤
  │ kernel_stream_conc            │ 标准 D2Q5 上游流步                       │
  ├───────────────────────────────┼──────────────────────────────────────────┤
  │                               │ 入口 Dirichlet                           │
  │ kernel_boundary_conc_reaction │ (Cm_init)；出口全展开；水合物面 Kang     │
  │                               │ BC：Cm_bc = (D·Cm_nbr + k_r·Csat)/(D +   │
  │                               │ k_r)，同时写入 VP.diss_rate              │
  ├───────────────────────────────┼──────────────────────────────────────────┤
  │ kernel_compute_latent_heat    │ S_latent[fluid] = -ΔH_latt · Σ           │
  │                               │ diss_rate(邻居水合物ghost)               │
  └───────────────────────────────┴──────────────────────────────────────────┘

  两个编译目标均干净通过。Phase 3（VOP 固相动态更新）可按需开始。

✻ Baked for 13m 1s
❯ Phase 3（VOP 固相动态更新）开始                                               

● Let me read the existing geometry/ghost setup functions in LBM.cu before
  writing VOP code:

  Searched for 1 pattern, read 1 file (ctrl+o to expand)

● Now I understand the geometry rebuild chain: mark_boundary → mark_ghost. I
  also see that init_geometry calls init_wall_mat_from_flag first. Let me check
  that function and verify the f_eq pattern for node re-initialization:

  Searched for 1 pattern, read 1 file (ctrl+o to expand)

● Now I have everything I need. Let me implement Phase 3 in hydrate_vop.cu:

● Write(hydrate_vop.cu)
  ⎿  Added 225 lines, removed 7 lines
      1  // hydrate_vop.cu                                               
      2  // VOP 固相动态更新模块（Phase 3）                              
      3 -// 当前为桩文件；Phase 3 实现将填充此处的核函数                 
      3 +// 实现：Vh 体积分数更新、节点翻转、new-fluid                   
        +节点重初始化、ghost 层重建                                      
      4  //                                                              
      5 -// 此文件与 hydrate.cu 分离，是为了保持单文件长度可控。         
      5  // 编译时必须有 -DHYDRATE_ENABLE                                
      6                                                                  
      7  #ifdef HYDRATE_ENABLE                                           
     ...
       12  #include "sim_utils.h"                                        
       13  #include "unified_cuda_error_check.cuh"                       
       14  #include <cstdio>                                             
       15 +#include <cstring>                                            
       16                                                                
       17 -// Phase 3 核函数将在此实现：                                 
       18 -//   kernel_update_vop(...)                                   
       19 -//   kernel_apply_vop_conversion(...)                         
       20 -//   kernel_reinit_new_fluid_nodes(...)                       
       17 +// 与 LBM.cu／main.cu 共享的全局线程块                        
       18 +extern dim3 grid, threads;                                    
       19                                                                
       20 -// 暂无额外符号定义（alloc_vop/free_vop/init_vop/step_vop     
          -已在 hydrate.cu 中）                                          
       20 +// LBM.cu 声明的设备端几何函数（由 init_geometry              
          +调用序列中使用）                                              
       21 +extern __global__ void mark_boundary(int* pointsflag);        
       22 +extern __global__ void mark_ghost(int* pointsflag);           
       23 +extern __global__ void init_wall_mat_from_flag(int*           
          +pointsflag);                                                  
       24 +// 润湿性表重建（声明在 LBM.h）                               
       25 +extern void upload_wettability_table_host(double, double,     
          +double, double);                                              
       26                                                                
       27 +// D2Q9 feq（声明在 LBM.cu，已对外可见）                      
       28 +extern __device__ double feq_gpu(int k, double rho, const     
          +double u[2]);                                                 
       29 +                                                              
       30 +//                                                            
          +============================================================  
       31 +// § VOP-1  更新水合物体积分数 Vh                             
       32 +//                                                            
          +------------------------------------------------------------  
       33 +// Vh[idx] -= Vm_latt * diss_rate[idx]                        
       34 +//   diss_rate 已在 kernel_boundary_conc_reaction             
          +中写入（格子单位）                                            
       35 +// 当 Vh <= 0 时：                                            
       36 +//   new_fluid_flag[idx] = 1                                  
       37 +//   Vh[idx]             = 0  （夹紧，防止负值）              
       38 +// 仅对 flag == -3（水合物节点）或 flag == -1 &&              
          +mat==2（水合物 ghost）操作                                    
       39 +//                                                            
          +============================================================  
       40 +__global__ void kernel_update_vop(                            
       41 +          double* __restrict__ Vh,                            
       42 +          int*    __restrict__ new_fluid_flag,                
       43 +    const double* __restrict__ diss_rate,                     
       44 +    const int*    __restrict__ pointsflag,                    
       45 +    double Vm_latt)                                           
       46 +{                                                             
       47 +    int x = blockIdx.x * blockDim.x + threadIdx.x;            
       48 +    int y = blockIdx.y * blockDim.y + threadIdx.y;            
       49 +    if (x >= NX || y >= NY) return;                           
       50 +                                                              
       51 +    const size_t s  = (size_t)NX * y + x;                     
       52 +    const int    fl = pointsflag[s];                          
       53 +                                                              
       54 +    // 只处理水合物内节点（flag==-3）                         
       55 +    if (fl != -3) return;                                     
       56 +                                                              
       57 +    double v = Vh[s] - Vm_latt * diss_rate[s];                
       58 +    if (v <= 0.0) {                                           
       59 +        Vh[s]              = 0.0;                             
       60 +        new_fluid_flag[s]  = 1;                               
       61 +    } else {                                                  
       62 +        Vh[s] = v;                                            
       63 +    }                                                         
       64 +}                                                             
       65 +                                                              
       66 +//                                                            
          +============================================================  
       67 +// § VOP-2  应用节点翻转：-3 → 1（流体），清除 d_wall_mat     
       68 +//                                                            
          +------------------------------------------------------------  
       69 +// 使用 atomicAdd 统计翻转数写入 n_conv[0]                    
       70 +// 翻转后将 pointsflag 设为 1，d_wall_mat 清为 0，Vh 保留 0   
       71 +//                                                            
          +============================================================  
       72 +__global__ void kernel_apply_vop_conversion(                  
       73 +          int*    __restrict__ pointsflag,                    
       74 +    const int*    __restrict__ new_fluid_flag,                
       75 +          int*    __restrict__ n_conv)                        
       76 +{                                                             
       77 +    int x = blockIdx.x * blockDim.x + threadIdx.x;            
       78 +    int y = blockIdx.y * blockDim.y + threadIdx.y;            
       79 +    if (x >= NX || y >= NY) return;                           
       80 +                                                              
       81 +    const size_t s = (size_t)NX * y + x;                      
       82 +    if (!new_fluid_flag[s]) return;                           
       83 +                                                              
       84 +    // flag 从 -3 翻到 1（流体）                              
       85 +    pointsflag[s]  = 1;                                       
       86 +    d_wall_mat[s]  = 0;                                       
       87 +                                                              
       88 +    atomicAdd(n_conv, 1);                                     
       89 +}                                                             
       90 +                                                              
       91 +//                                                            
          +============================================================  
       92 +// § VOP-3  重初始化新流体节点的分布函数                      
       93 +//                                                            
          +------------------------------------------------------------  
       94 +// 对 new_fluid_flag[s]==1 的节点：                           
       95 +//   rho_A/rho_B =                                            
          +邻居流体节点均值（若无流体邻居，用全局低密度下限）            
       96 +//   fin_A/fin_B = feq(rho_mean, u=0)                         
       97 +//   fout_A/fout_B = fin（避免异常反弹）                      
       98 +//   h_in  = h_eq(T_mean, u=0) = w5[k]*T_mean                 
       99 +//   g_in  = g_eq(Cm_mean,u=0) = w5[k]*Cm_mean                
      100 +//                                                            
          +============================================================  
      101 +__global__ void kernel_reinit_new_fluid(                      
      102 +    const int*    __restrict__ new_fluid_flag,                
      103 +    const int*    __restrict__ pointsflag,                    
      104 +          double* __restrict__ fin_A,                         
      105 +          double* __restrict__ fout_A,                        
      106 +          double* __restrict__ rho_A,                         
      107 +          double* __restrict__ fin_B,                         
      108 +          double* __restrict__ fout_B,                        
      109 +          double* __restrict__ rho_B,                         
      110 +          double* __restrict__ h_in,                          
      111 +          double* __restrict__ T,                             
      112 +          double* __restrict__ g_in,                          
      113 +          double* __restrict__ Cm)                            
      114 +{                                                             
      115 +    int x = blockIdx.x * blockDim.x + threadIdx.x;            
      116 +    int y = blockIdx.y * blockDim.y + threadIdx.y;            
      117 +    if (x >= NX || y >= NY) return;                           
      118 +                                                              
      119 +    const size_t s = (size_t)NX * y + x;                      
      120 +    if (!new_fluid_flag[s]) return;                           
      121 +                                                              
      122 +    // 外部方向向量（D2Q9）                                   
      123 +    extern __constant__ int    e_gpu[9][2];                   
      124 +    extern __constant__ double w_gpu[9];                      
      125 +                                                              
      126 +    // 采样邻居流体节点（D2Q9 邻居）                          
      127 +    double sumrA = 0.0, sumrB = 0.0, sumT = 0.0, sumCm = 0.0; 
      128 +    int    cnt   = 0;                                         
      129 +                                                              
      130 +    for (int k = 1; k < Q; ++k) {   // k=0 静止跳过           
      131 +        int xn = x + e_gpu[k][0];                             
      132 +        int yn = y + e_gpu[k][1];                             
      133 +        if (xn < 0 || xn >= NX || yn < 0 || yn >= NY)         
          +continue;                                                     
      134 +        const size_t sn = (size_t)NX * yn + xn;               
      135 +        if (pointsflag[sn] == 1 || pointsflag[sn] == 0) {     
      136 +            sumrA += rho_A[sn];                               
      137 +            sumrB += rho_B[sn];                               
      138 +            sumT  += T[sn];                                   
      139 +            sumCm += Cm[sn];                                  
      140 +            ++cnt;                                            
      141 +        }                                                     
      142 +    }                                                         
      143 +                                                              
      144 +    // 若没找到流体邻居，用保守的低密度默认值                 
      145 +    const double rA_new = (cnt > 0) ? sumrA / cnt : 0.05;     
      146 +    const double rB_new = (cnt > 0) ? sumrB / cnt : 0.05;     
      147 +    const double T_new  = (cnt > 0) ? sumT  / cnt :           
          +d_T0_init;                                                    
      148 +    const double Cm_new = (cnt > 0) ? sumCm / cnt :           
          +d_Cm_init;                                                    
      149 +                                                              
      150 +    rho_A[s] = rA_new;                                        
      151 +    rho_B[s] = rB_new;                                        
      152 +    T[s]     = T_new;                                         
      153 +    Cm[s]    = Cm_new;                                        
      154 +                                                              
      155 +    // D2Q9 分布函数（feq, u=0）                              
      156 +    const double u0[2] = {0.0, 0.0};                          
      157 +    const size_t off9  = (size_t)NX * NY;                     
      158 +    for (int k = 0; k < Q; ++k) {                             
      159 +        double fA = feq_gpu(k, rA_new, u0);                   
      160 +        double fB = feq_gpu(k, rB_new, u0);                   
      161 +        fin_A [(size_t)NX * NY * k + s] = fout_A[(size_t)NX * 
          + NY * k + s] = fA;                                            
      162 +        fin_B [(size_t)NX * NY * k + s] = fout_B[(size_t)NX * 
          + NY * k + s] = fB;                                            
      163 +    }                                                         
      164 +                                                              
      165 +    // D2Q5 分布函数（h_eq = w5[k]*T, g_eq = w5[k]*Cm, u=0）  
      166 +    for (int k = 0; k < Q5; ++k) {                            
      167 +        h_in[(size_t)NX * ((size_t)NY * k + y) + x] =         
          +w5_gpu[k] * T_new;                                            
      168 +        g_in[(size_t)NX * ((size_t)NY * k + y) + x] =         
          +w5_gpu[k] * Cm_new;                                           
      169 +    }                                                         
      170 +}                                                             
      171 +                                                              
      172 +//                                                            
          +============================================================  
      173 +// § VOP-4  清除 new_fluid_flag（每步用完后重置）             
      174 +//                                                            
          +============================================================  
      175 +__global__ void kernel_clear_new_fluid_flag(int* __restrict__ 
          + new_fluid_flag)                                              
      176 +{                                                             
      177 +    int x = blockIdx.x * blockDim.x + threadIdx.x;            
      178 +    int y = blockIdx.y * blockDim.y + threadIdx.y;            
      179 +    if (x >= NX || y >= NY) return;                           
      180 +    new_fluid_flag[(size_t)NX * y + x] = 0;                   
      181 +}                                                             
      182 +                                                              
      183 +//                                                            
          +============================================================  
      184 +// § VOP-5  宿主函数：完整 VOP 步                             
      185 +//                                                            
          +------------------------------------------------------------  
      186 +// 返回：本步翻转节点数（>0 时主循环应重建 ghost 层）         
      187 +//                                                            
          +============================================================  
      188 +int step_vop(VOP_dev& VP,                                     
      189 +             Therm_dev& TH, Conc_dev& CN,                     
      190 +             Fluid_dev& A, Fluid_dev& B, Mix_dev& MX)         
      191 +{                                                             
      192 +    // 1) 更新 Vh，检测翻转                                   
      193 +    kernel_update_vop<<<grid, threads>>>(                     
      194 +        VP.Vh, VP.new_fluid_flag,                             
      195 +        VP.diss_rate, MX.pointsflag,                          
      196 +        d_Vm_latt_host);                                      
      197 +    CUDA_CHECK(cudaGetLastError());                           
      198 +                                                              
      199 +    // 2) 翻转标记节点（-3 → 1），统计翻转数                  
      200 +    int* d_n_conv = nullptr;                                  
      201 +    CUDA_CHECK(cudaMalloc(&d_n_conv, sizeof(int)));           
      202 +    CUDA_CHECK(cudaMemset(d_n_conv, 0, sizeof(int)));         
      203 +                                                              
      204 +    kernel_apply_vop_conversion<<<grid, threads>>>(           
      205 +        MX.pointsflag, VP.new_fluid_flag, d_n_conv);          
      206 +    CUDA_CHECK(cudaGetLastError());                           
      207 +    CUDA_CHECK(cudaDeviceSynchronize());                      
      208 +                                                              
      209 +    int n_conv = 0;                                           
      210 +    CUDA_CHECK(cudaMemcpy(&n_conv, d_n_conv, sizeof(int),     
          +cudaMemcpyDeviceToHost));                                     
      211 +    cudaFree(d_n_conv);                                       
      212 +                                                              
      213 +    if (n_conv > 0) {                                         
      214 +        // 3) 重初始化新流体节点                              
      215 +        kernel_reinit_new_fluid<<<grid, threads>>>(           
      216 +            VP.new_fluid_flag, MX.pointsflag,                 
      217 +            A.fin, A.fout, A.rho,                             
      218 +            B.fin, B.fout, B.rho,                             
      219 +            TH.h_in, TH.T,                                    
      220 +            CN.g_in, CN.Cm);                                  
      221 +        CUDA_CHECK(cudaGetLastError());                       
      222 +                                                              
      223 +        // 4) 重建几何：壁面材质 → 边界层 → ghost 层          
      224 +        init_wall_mat_from_flag<<<grid,                       
          +threads>>>(MX.pointsflag);                                    
      225 +        CUDA_CHECK(cudaGetLastError());                       
      226 +        mark_boundary<<<grid, threads>>>(MX.pointsflag);      
      227 +        CUDA_CHECK(cudaGetLastError());                       
      228 +        mark_ghost<<<grid, threads>>>(MX.pointsflag);         
      229 +        CUDA_CHECK(cudaGetLastError());                       
      230 +        CUDA_CHECK(cudaDeviceSynchronize());                  
      231 +                                                              
      232 +        printf("[VOP] 本步翻转 %d 个水合物节点 → 流体\n",     
          +n_conv);                                                      
      233 +    }                                                         
      234 +                                                              
      235 +    // 5) 清除 flag，为下一步准备                             
      236 +    kernel_clear_new_fluid_flag<<<grid,                       
          +threads>>>(VP.new_fluid_flag);                                
      237 +    CUDA_CHECK(cudaGetLastError());                           
      238 +                                                              
      239 +    return n_conv;                                            
      240 +}                                                             
      241 +                                                              
      242  #endif  // HYDRATE_ENABLE                                     

● There's a problem: kernel_reinit_new_fluid uses extern __constant__ inside
  device code which won't work. Also d_Vm_latt_host doesn't exist — I need to
  pass Vm_latt differently since __constant__ can't be read on the host. Let me
  check what device constants are available and fix the approach:

  Searched for 1 pattern (ctrl+o to expand)

● d_Vm_latt is a __constant__ — accessible directly in device code but not on
  host. I'll rework two things:
  1. In kernel_update_vop, use d_Vm_latt directly (no host parameter needed)
  2. Remove the extern __constant__ declarations inside the kernel (device
  constants are already visible in device scope when linked with -rdc=true)

● Write(hydrate_vop.cu)
  ⎿  Added 34 lines, removed 57 lines
      11  #include "sim_utils.h"                                         
      12  #include "unified_cuda_error_check.cuh"                        
      13  #include <cstdio>                                              
      14 -#include <cstring>                                             
      14                                                                 
      15 -// 与 LBM.cu／main.cu 共享的全局线程块                         
      15 +//                                                             
         +============================================================   
      16 +// 与 LBM.cu / main.cu 共享的全局线程块                        
      17  extern dim3 grid, threads;                                     
      18                                                                 
      19 -// LBM.cu 声明的设备端几何函数（由 init_geometry               
         -调用序列中使用）                                               
      19 +// LBM.cu 内部核函数（geometry rebuild，通过 rdc=true          
         +跨文件可见）                                                   
      20  extern __global__ void mark_boundary(int* pointsflag);         
      21  extern __global__ void mark_ghost(int* pointsflag);            
      22  extern __global__ void init_wall_mat_from_flag(int*            
          pointsflag);                                                   
      23 -// 润湿性表重建（声明在 LBM.h）                                
      24 -extern void upload_wettability_table_host(double, double,      
         -double, double);                                               
      23                                                                 
      24 -// D2Q9 feq（声明在 LBM.cu，已对外可见）                       
      25 -extern __device__ double feq_gpu(int k, double rho, const      
         -double u[2]);                                                  
      26 -                                                               
      24  //                                                             
          ============================================================   
      25  // § VOP-1  更新水合物体积分数 Vh                              
      26  //                                                             
          ------------------------------------------------------------   
      27 -// Vh[idx] -= Vm_latt * diss_rate[idx]                         
      28 -//   diss_rate 已在 kernel_boundary_conc_reaction 中           
         -写入（格子单位）                                               
      29 -// 当 Vh <= 0 时：                                             
      30 -//   new_fluid_flag[idx] = 1                                   
      31 -//   Vh[idx]             = 0  （夹紧，防止负值）               
      32 -// 仅对 flag == -3（水合物节点）或 flag == -1 &&               
         -mat==2（水合物 ghost）操作                                     
      27 +// Vh[idx] -= d_Vm_latt * diss_rate[idx]                       
      28 +//   diss_rate 由 kernel_boundary_conc_reaction                
         +写入（格子单位）                                               
      29 +// Vh <= 0 时标记翻转；只处理 flag == -3 的水合物内部节点      
      30  //                                                             
          ============================================================   
      31  __global__ void kernel_update_vop(                             
      32            double* __restrict__ Vh,                             
      33            int*    __restrict__ new_fluid_flag,                 
      34      const double* __restrict__ diss_rate,                      
      35 -    const int*    __restrict__ pointsflag,                     
      36 -    double Vm_latt)                                            
      35 +    const int*    __restrict__ pointsflag)                     
      36  {                                                              
      37      int x = blockIdx.x * blockDim.x + threadIdx.x;             
      38      int y = blockIdx.y * blockDim.y + threadIdx.y;             
      39      if (x >= NX || y >= NY) return;                            
      40                                                                 
      41      const size_t s  = (size_t)NX * y + x;                      
      42 -    const int    fl = pointsflag[s];                           
      42 +    if (pointsflag[s] != -3) return;                           
      43                                                                 
      44 -    // 只处理水合物内节点（flag==-3）                          
      45 -    if (fl != -3) return;                                      
      46 -                                                               
      47 -    double v = Vh[s] - Vm_latt * diss_rate[s];                 
      44 +    double v = Vh[s] - d_Vm_latt * diss_rate[s];               
      45      if (v <= 0.0) {                                            
      46 -        Vh[s]              = 0.0;                              
      47 -        new_fluid_flag[s]  = 1;                                
      46 +        Vh[s]             = 0.0;                               
      47 +        new_fluid_flag[s] = 1;                                 
      48      } else {                                                   
      49          Vh[s] = v;                                             
      50      }                                                          
      51  }                                                              
      52                                                                 
      53  //                                                             
          ============================================================   
      54 -// § VOP-2  应用节点翻转：-3 → 1（流体），清除 d_wall_mat      
      54 +// § VOP-2  应用节点翻转：-3 → 1，清除 d_wall_mat              
      55  //                                                             
          ------------------------------------------------------------   
      56 -// 使用 atomicAdd 统计翻转数写入 n_conv[0]                     
      57 -// 翻转后将 pointsflag 设为 1，d_wall_mat 清为 0，Vh 保留 0    
      56 +// atomicAdd 统计翻转数写入 n_conv[0]                          
      57  //                                                             
          ============================================================   
      58  __global__ void kernel_apply_vop_conversion(                   
      59            int*    __restrict__ pointsflag,                     
     ...
      80      const size_t s = (size_t)NX * y + x;                       
      81      if (!new_fluid_flag[s]) return;                            
      82                                                                 
      83 -    // flag 从 -3 翻到 1（流体）                               
      84 -    pointsflag[s]  = 1;                                        
      85 -    d_wall_mat[s]  = 0;                                        
      86 -                                                               
      83 +    pointsflag[s] = 1;                                         
      84 +    d_wall_mat[s] = 0;                                         
      85      atomicAdd(n_conv, 1);                                      
      86  }                                                              
      87                                                                 
      88  //                                                             
          ============================================================   
      89  // § VOP-3  重初始化新流体节点的分布函数                       
      90  //                                                             
          ------------------------------------------------------------   
      91 -// 对 new_fluid_flag[s]==1 的节点：                            
      92 -//   rho_A/rho_B =                                             
         -邻居流体节点均值（若无流体邻居，用全局低密度下限）             
      93 -//   fin_A/fin_B = feq(rho_mean, u=0)                          
      94 -//   fout_A/fout_B = fin（避免异常反弹）                       
      95 -//   h_in  = h_eq(T_mean, u=0) = w5[k]*T_mean                  
      96 -//   g_in  = g_eq(Cm_mean,u=0) = w5[k]*Cm_mean                 
      91 +// rho_A/rho_B = D2Q9 邻居流体节点均值                         
      92 +// fin/fout    = feq(rho_mean, u=0)                            
      93 +// h_in[k]     = w5[k] * T_mean                                
      94 +// g_in[k]     = w5[k] * Cm_mean                               
      95  //                                                             
          ============================================================   
      96  __global__ void kernel_reinit_new_fluid(                       
      97      const int*    __restrict__ new_fluid_flag,                 
     ...
      118      const size_t s = (size_t)NX * y + x;                      
      119      if (!new_fluid_flag[s]) return;                           
      120                                                                
      121 -    // 外部方向向量（D2Q9）                                   
      122 -    extern __constant__ int    e_gpu[9][2];                   
      123 -    extern __constant__ double w_gpu[9];                      
      124 -                                                              
      125 -    // 采样邻居流体节点（D2Q9 邻居）                          
      121 +    // 采样 D2Q9 邻居（仅流体 flag==1 或边界 flag==0）        
      122      double sumrA = 0.0, sumrB = 0.0, sumT = 0.0, sumCm = 0.0; 
      123      int    cnt   = 0;                                         
      124                                                                
      125 -    for (int k = 1; k < Q; ++k) {   // k=0 静止跳过           
      125 +    for (int k = 1; k < Q; ++k) {                             
      126          int xn = x + e_gpu[k][0];                             
      127          int yn = y + e_gpu[k][1];                             
      128          if (xn < 0 || xn >= NX || yn < 0 || yn >= NY)         
           continue;                                                     
      129          const size_t sn = (size_t)NX * yn + xn;               
      130 -        if (pointsflag[sn] == 1 || pointsflag[sn] == 0) {     
      130 +        const int    fn = pointsflag[sn];                     
      131 +        if (fn == 1 || fn == 0) {                             
      132              sumrA += rho_A[sn];                               
      133              sumrB += rho_B[sn];                               
      134              sumT  += T[sn];                                   
     ...
      140          }                                                     
      141      }                                                         
      142                                                                
      143 -    // 若没找到流体邻居，用保守的低密度默认值                 
      143      const double rA_new = (cnt > 0) ? sumrA / cnt : 0.05;     
      144      const double rB_new = (cnt > 0) ? sumrB / cnt : 0.05;     
      145      const double T_new  = (cnt > 0) ? sumT  / cnt :           
           d_T0_init;                                                    
     ...
      151      T[s]     = T_new;                                         
      152      Cm[s]    = Cm_new;                                        
      153                                                                
      154 -    // D2Q9 分布函数（feq, u=0）                              
      154 +    // D2Q9 分布函数：feq(rho, u=0)                           
      155      const double u0[2] = {0.0, 0.0};                          
      156 -    const size_t off9  = (size_t)NX * NY;                     
      156      for (int k = 0; k < Q; ++k) {                             
      157          double fA = feq_gpu(k, rA_new, u0);                   
      158          double fB = feq_gpu(k, rB_new, u0);                   
     ...
      161          fin_B [(size_t)NX * NY * k + s] = fout_B[(size_t)NX * 
            NY * k + s] = fB;                                            
      162      }                                                         
      163                                                                
      164 -    // D2Q5 分布函数（h_eq = w5[k]*T, g_eq = w5[k]*Cm, u=0）  
      164 +    // D2Q5 分布函数：h_eq = w5[k]*T, g_eq = w5[k]*Cm (u=0)   
      165      for (int k = 0; k < Q5; ++k) {                            
      166          h_in[(size_t)NX * ((size_t)NY * k + y) + x] =         
           w5_gpu[k] * T_new;                                            
      167          g_in[(size_t)NX * ((size_t)NY * k + y) + x] =         
           w5_gpu[k] * Cm_new;                                           
     ...
      169  }                                                             
      170                                                                
      171  //                                                            
           ============================================================  
      172 -// § VOP-4  清除 new_fluid_flag（每步用完后重置）             
      172 +// § VOP-4  清除 new_fluid_flag（每步末尾重置）               
      173  //                                                            
           ============================================================  
      174  __global__ void kernel_clear_new_fluid_flag(int* __restrict__ 
            new_fluid_flag)                                              
      175  {                                                             
     ...
      182  //                                                            
           ============================================================  
      183  // § VOP-5  宿主函数：完整 VOP 步                             
      184  //                                                            
           ------------------------------------------------------------  
      185 -// 返回：本步翻转节点数（>0 时主循环应重建 ghost 层）         
      185 +// 顺序：Vh 更新 → 翻转 → 新节点重初 → ghost 重建 → 清 flag   
      186 +// 返回：本步翻转节点数（>0                                   
          +时主循环应做额外处理如润湿性上传）                            
      187  //                                                            
           ============================================================  
      188  int step_vop(VOP_dev& VP,                                     
      189               Therm_dev& TH, Conc_dev& CN,                     
      190               Fluid_dev& A, Fluid_dev& B, Mix_dev& MX)         
      191  {                                                             
      192 -    // 1) 更新 Vh，检测翻转                                   
      192 +    // 1) 更新 Vh，标记翻转节点                               
      193      kernel_update_vop<<<grid, threads>>>(                     
      194 -        VP.Vh, VP.new_fluid_flag,                             
      195 -        VP.diss_rate, MX.pointsflag,                          
      196 -        d_Vm_latt_host);                                      
      194 +        VP.Vh, VP.new_fluid_flag,                             
          + VP.diss_rate, MX.pointsflag);                                
      195      CUDA_CHECK(cudaGetLastError());                           
      196                                                                
      197 -    // 2) 翻转标记节点（-3 → 1），统计翻转数                  
      197 +    // 2) 统计 + 执行翻转                                     
      198      int* d_n_conv = nullptr;                                  
      199      CUDA_CHECK(cudaMalloc(&d_n_conv, sizeof(int)));           
      200      CUDA_CHECK(cudaMemset(d_n_conv, 0, sizeof(int)));         
     ...
      210      cudaFree(d_n_conv);                                       
      211                                                                
      212      if (n_conv > 0) {                                         
      213 -        // 3) 重初始化新流体节点                              
      213 +        // 3) 新流体节点重初始化                              
      214          kernel_reinit_new_fluid<<<grid, threads>>>(           
      215              VP.new_fluid_flag, MX.pointsflag,                 
      216              A.fin, A.fout, A.rho,                             
     ...
      219              CN.g_in, CN.Cm);                                  
      220          CUDA_CHECK(cudaGetLastError());                       
      221                                                                
      222 -        // 4) 重建几何：壁面材质 → 边界层 → ghost 层          
      222 +        // 4) 全域 ghost 层重建：壁面材质 → 边界 → ghost      
      223          init_wall_mat_from_flag<<<grid,                       
           threads>>>(MX.pointsflag);                                    
      224          CUDA_CHECK(cudaGetLastError());                       
      225          mark_boundary<<<grid, threads>>>(MX.pointsflag);      
     ...
      231          printf("[VOP] 本步翻转 %d 个水合物节点 → 流体\n",     
           n_conv);                                                      
      232      }                                                         
      233                                                                
      234 -    // 5) 清除 flag，为下一步准备                             
      234 +    // 5) 清 flag，为下步准备                                 
      235      kernel_clear_new_fluid_flag<<<grid,                       
           threads>>>(VP.new_fluid_flag);                                
      236      CUDA_CHECK(cudaGetLastError());                           
      237                                                                

● Now remove the step_vop stub from hydrate.cu (it's now defined in
  hydrate_vop.cu):

● Update(hydrate.cu)
  ⎿  Added 1 line, removed 8 lines
      785                                                                
           CUDA_CHECK(cudaMemset(VP.new_fluid_flag,0,mem_size_flag));    
      786  }                                                             
      787                                                                
      788 -int step_vop(VOP_dev& VP,                                     
      789 -             Therm_dev& TH, Conc_dev& CN,                     
      790 -             Fluid_dev& A, Fluid_dev& B, Mix_dev& MX)         
      791 -{                                                             
      792 -    // Phase 3 实现：更新 Vh，翻转节点，返回翻转数            
      793 -    (void)VP; (void)TH; (void)CN; (void)A; (void)B; (void)MX; 
      794 -    return 0;                                                 
      795 -}                                                             
      788 +// step_vop 实现在 hydrate_vop.cu（Phase 3）                  
      789                                                                
      790  //                                                            
           ============================================================  
      791  // §13  全耦合入口（Phase 5 桩）                              

● Also need to check feq_gpu — it's __device__ in LBM.cu, need to declare it in
  hydrate_vop.cu:

  Searched for 1 pattern (ctrl+o to expand)

● feq_gpu is __device__ __forceinline__ defined in LBM.cu. With -rdc=true it's
  accessible across translation units. The extern approach won't work for
  __forceinline__ device functions — they need to be visible at the point of
  use. Let me add a local inline equivalent in hydrate_vop.cu to avoid the
  cross-file __forceinline__ issue:

● Update(hydrate_vop.cu)
  ⎿  Added 10 lines
      21  extern __global__ void mark_ghost(int* pointsflag);            
      22  extern __global__ void init_wall_mat_from_flag(int*            
          pointsflag);                                                   
      23                                                                 
      24 +// D2Q9 feq（本文件本地 inline 定义，与 LBM.cu 中完全相同）    
      25 +// 避免跨文件引用 __forceinline__ device 函数的链接问题        
      26 +__device__ __forceinline__ double feq_vop(int k, double rho,   
         +const double u[2])                                             
      27 +{                                                              
      28 +    const double eu  = e_gpu[k][0] * u[0] + e_gpu[k][1] *      
         +u[1];                                                          
      29 +    const double uv  = u[0] * u[0] + u[1] * u[1];              
      30 +    const double cs2 = 1.0 / 3.0;   // cs2_gpu 值（与 LBM.cu   
         +一致）                                                         
      31 +    return w_gpu[k] * rho * (1.0 + eu / cs2 + eu * eu / (2.0 * 
         + cs2 * cs2) - uv / (2.0 * cs2));                               
      32 +}                                                              
      33 +                                                               
      34  //                                                             
          ============================================================   
      35  // § VOP-1  更新水合物体积分数 Vh                              
      36  //                                                             
          ------------------------------------------------------------   

● Now replace feq_gpu with feq_vop in the reinit kernel:

● Update(hydrate_vop.cu)
  ⎿  Added 2 lines, removed 2 lines
      143      // D2Q9 分布函数：feq(rho, u=0)                           
      144      const double u0[2] = {0.0, 0.0};                          
      145      for (int k = 0; k < Q; ++k) {                             
      146 -        double fA = feq_gpu(k, rA_new, u0);                   
      147 -        double fB = feq_gpu(k, rB_new, u0);                   
      146 +        double fA = feq_vop(k, rA_new, u0);                   
      147 +        double fB = feq_vop(k, rB_new, u0);                   
      148          fin_A [(size_t)NX * NY * k + s] = fout_A[(size_t)NX * 
            NY * k + s] = fA;                                            
      149          fin_B [(size_t)NX * NY * k + s] = fout_B[(size_t)NX * 
            NY * k + s] = fB;                                            
      150      }                                                         

● Also remove the extern __global__ for feq_gpu that I added earlier (no longer
  needed):

  Searched for 1 pattern (ctrl+o to expand)

● Good. Now compile:

● Bash(bash compile.sh hydrate 2>&1)
  ⎿  Error: Exit code 2
     hydrate_vop.cu(28): error: identifier "e_gpu" is undefined
           const double eu = e_gpu[k][0] * u[0] + e_gpu[k][1] * u[1];
                             ^

     hydrate_vop.cu(31): error: identifier "w_gpu" is undefined
           return w_gpu[k] * rho * (1.0 + eu / cs2 + eu * eu / (2.0 * cs2 * cs2)
      - uv / (2.0 * cs2));
                  ^

     hydrate_vop.cu(119): error: identifier "e_gpu" is undefined

     … +4 lines (ctrl+o to see all)

● e_gpu and w_gpu are __constant__ arrays defined in LBM.cu under
  LBM_DEFINE_GLOBALS. They need extern declarations in hydrate_vop.cu. Let me
  check their exact declarations:

  Searched for 1 pattern (ctrl+o to expand)

● These are defined directly in LBM.cu without a LBM_DEFINE_GLOBALS guard. With
  -rdc=true, they're accessible via extern. Add extern declarations to
  hydrate_vop.cu:

● Update(hydrate_vop.cu)
  ⎿  Added 4 lines
      16  // 与 LBM.cu / main.cu 共享的全局线程块                        
      17  extern dim3 grid, threads;                                     
      18                                                                 
      19 +// LBM.cu 中定义的 D2Q9 设备常量（通过 -rdc=true 跨文件可见）  
      20 +extern __device__ __constant__ int    e_gpu[9][2];             
      21 +extern __device__ __constant__ double w_gpu[9];                
      22 +                                                               
      23  // LBM.cu 内部核函数（geometry rebuild，通过 rdc=true          
          跨文件可见）                                                   
      24  extern __global__ void mark_boundary(int* pointsflag);         
      25  extern __global__ void mark_ghost(int* pointsflag);            
● Bash(bash compile.sh hydrate 2>&1)
  ⎿  (No output)
  ⎿  (timeout 2m)

● Bash(ls -lh mcmp_sim_hydrate && nm mcmp_sim_hydrate | grep -E "kernel_update_v
      op|kernel_apply_vop|kernel_reinit|kernel_clear_new|step_vop" | head -20)
  ⎿  -rwx------ 1 server server 2.8M  3月 25 19:29 mcmp_sim_hydrate
     000000000000d540 t _GLOBAL__sub_I__Z8step_vopR7VOP_devR9Therm_devR8Con
     c_devR9Fluid_devS6_R7Mix_dev
     … +14 lines (ctrl+o to expand)

● All Phase 3 symbols present. Verify flow-only still clean:

● Bash(bash compile.sh 2>&1 | tail -3 && ls -lh mcmp_sim)
  ⎿  -rwx------ 1 server server 2.5M  3月 25 19:30 mcmp_sim
  ⎿  (timeout 2m)

● Phase 3 完成，编译干净。实现摘要：

  hydrate_vop.cu — 5 个核函数 + 1 个宿主函数：

  ┌─────────────────────────────┬────────────────────────────────────────────┐
  │            函数             │                    功能                    │
  ├─────────────────────────────┼────────────────────────────────────────────┤
  │ kernel_update_vop           │ Vh -= d_Vm_latt * diss_rate，Vh≤0 时设     │
  │                             │ new_fluid_flag=1                           │
  ├─────────────────────────────┼────────────────────────────────────────────┤
  │ kernel_apply_vop_conversion │ flag -3→1，清 d_wall_mat，atomicAdd        │
  │                             │ 统计翻转数                                 │
  ├─────────────────────────────┼────────────────────────────────────────────┤
  │                             │ D2Q9 邻居均值初始化 rho_A/B，写 fin/fout = │
  │ kernel_reinit_new_fluid     │  feq(u=0)；D2Q5 写 h_in = w5·T，g_in =     │
  │                             │ w5·Cm                                      │
  ├─────────────────────────────┼────────────────────────────────────────────┤
  │ kernel_clear_new_fluid_flag │ 重置 new_fluid_flag 为 0                   │
  ├─────────────────────────────┼────────────────────────────────────────────┤
  │                             │ 主机端编排：Vh 更新 → 翻转 → 重初始化 →    │
  │ step_vop                    │ init_wall_mat_from_flag + mark_boundary +  │
  │                             │ mark_ghost 全域 ghost 重建 → 清            │
  │                             │ flag；返回翻转数                           │
  └─────────────────────────────┴────────────────────────────────────────────┘

  Phase 4（潜热源项）已在 Phase 2
  末尾同步实现（kernel_compute_latent_heat）。下一步可直接进入 Phase 
  5（全耦合集成：run_stage 调用链接入、checkpoint 扩展、VTK 输出扩展）。

✻ Sautéed for 5m 42s

❯ 直接进入 Phase 4和5，请你继续进行下一步                                       

● Let me read the relevant sections before writing any code:

  Searched for 2 patterns, read 2 files (ctrl+o to expand)

● Now let me check outputvtk/outputdat signatures, sim_utils.h RunResult and
  function declarations, and the write_stage_output function to understand the
  full picture:

  Searched for 1 pattern, read 2 files (ctrl+o to expand)

● Now let me check how main.cu uses run_stage and run_equilibrate_then_flow to
  understand what objects are available:

  Searched for 1 pattern, read 1 file (ctrl+o to expand)

● Now I have a complete picture. The hydrate objects need to live in main.cu and
   be passed through to run_stage. The cleanest approach is:

  1. main.cu — allocate/init/free hydrate structs under #ifdef HYDRATE_ENABLE
  2. sim_utils.h — extend run_stage signature to optionally take hydrate
  pointers (or use a single HydrateDev* wrapper)
  3. sim_utils.cu — integrate step_hydrate_physics in the loop + extend
  write_stage_output for T/Cm/Vh VTK output
  4. LBM.cu — extend outputvtk signature to accept optional hydrate host arrays

  The cleanest no-break-existing-API approach: add an overload of run_stage that
   takes hydrate objects, and add a HydrateHost struct for VTK output data. Let
  me implement this:

● Step 1: Add HydrateHost struct and extend run_stage signature in sim_utils.h

  Read 1 file (ctrl+o to expand)

● Update(sim_utils.h)
  ⎿  Added 23 lines
      211                      const StageConfig& cfg,                   
      212                      const RuntimeParams& P);                  
      213                                                                
      214 +#ifdef HYDRATE_ENABLE                                         
      215 +// 水合物宿主端缓冲（用于 VTK 输出）                          
      216 +struct HydrateHost {                                          
      217 +    std::vector<double> T;          // 温度场 [NX*NY]         
      218 +    std::vector<double> Cm;         // 溶解浓度 [NX*NY]       
      219 +    std::vector<double> Vh;         // 水合物体积分数 [NX*NY] 
      220 +    std::vector<double> diss_rate;  // 分解速率 [NX*NY]       
      221 +    HydrateHost()                                             
      222 +    : T(NX*NY, 0.0), Cm(NX*NY, 0.0),                          
      223 +      Vh(NX*NY, 0.0), diss_rate(NX*NY, 0.0) {}                
      224 +};                                                            
      225 +                                                              
      226 +// 水合物扩展版 run_stage（含耦合物理循环）                   
      227 +RunResult run_stage_hydrate(Fluid_dev& A, Fluid_host& AH,     
      228 +                            Fluid_dev& B, Fluid_host& BH,     
      229 +                            Mix_dev&   MX, Mix_host&   MH,    
      230 +                            Therm_dev& TH, Conc_dev& CN,      
      231 +                            VOP_dev&   VP, HydrateHost& HH,   
      232 +                            SteadyMonitor& SM,                
      233 +                            const StageConfig& cfg,           
      234 +                            const RuntimeParams& P);          
      235 +#endif                                                        
      236 +                                                              
      237  RunResult run_equilibrate_then_flow(Fluid_dev& A, Fluid_host& 
            AH,                                                          
      238                                      Fluid_dev& B, Fluid_host& 
            BH,                                                          
      239                                      Mix_dev&   M, Mix_host&   
            MH,                                                          

● Step 2: Extend outputvtk to accept hydrate arrays in LBM.h

  Read 1 file (ctrl+o to expand)

● HydrateHost is defined in sim_utils.h, but LBM.h is included before it. I'll
  keep VTK extension in sim_utils.cu using a separate write_hydrate_vtk_fields
  approach — inject after main VTK writing. Actually the cleanest way is to add
  a forwarded inline implementation in LBM.cu after the existing outputvtk. Let
  me add the overload to LBM.h under #ifdef HYDRATE_ENABLE at the bottom where
  it's safe:

● Update(LBM.h)
  ⎿  Added 8 lines
      338  // 放在 LBM.h 末尾，确保 Fluid_dev、Mix_dev                   
           等已经定义后再包含                                            
      339  #ifdef HYDRATE_ENABLE                                         
      340  #include "hydrate.h"                                          
      341 +                                                              
      342 +// hydrate 场 VTK 追加写（向已打开的 .vtk 文件末尾写          
          +T/Cm/Vh/diss_rate）                                           
      343 +// 由 sim_utils.cu 中 write_stage_output_hydrate 调用         
      344 +void outputvtk_append_hydrate(const std::string& vtk_path,    
      345 +                               const std::vector<double>& T,  
      346 +                               const std::vector<double>& Cm, 
      347 +                               const std::vector<double>& Vh, 
      348 +                               const std::vector<double>&     
          +diss_rate);                                                   
      349  #endif                                                        

● Step 3: Implement outputvtk_append_hydrate in LBM.cu

  Read 1 file (ctrl+o to expand)

● Update(LBM.cu)
  ⎿  Added 23 lines
      2007      std::cout << "[output] vtk step " << step << "           
            completed\n";                                                
      2008  }                                                            
      2009                                                               
      2010 +#ifdef HYDRATE_ENABLE                                        
      2011 +// 向已有 .vtk 文件末尾追加水合物场（以 append 模式打开）    
      2012 +__host__ void outputvtk_append_hydrate(const std::string&    
           +vtk_path,                                                    
      2013 +                                        const                
           +std::vector<double>& T,                                      
      2014 +                                        const                
           +std::vector<double>& Cm,                                     
      2015 +                                        const                
           +std::vector<double>& Vh,                                     
      2016 +                                        const                
           +std::vector<double>& diss_rate)                              
      2017 +{                                                            
      2018 +    std::ofstream vtk(vtk_path, std::ios::binary |           
           +std::ios::app);                                              
      2019 +    if (!vtk) return;                                        
      2020 +                                                             
      2021 +    auto write_scalar = [&](const std::string& name, const   
           +std::vector<double>& vec)                                    
      2022 +    {                                                        
      2023 +        vtk << "\nSCALARS " << name << "                     
           +double\nLOOKUP_TABLE default\n";                             
      2024 +        for (double v : vec) { double t = v; SwapEnd(t);     
           +vtk.write(reinterpret_cast<char*>(&t), sizeof(double)); }    
      2025 +    };                                                       
      2026 +    write_scalar("temperature",  T);                         
      2027 +    write_scalar("concentration", Cm);                       
      2028 +    write_scalar("hydrate_Vh",   Vh);                        
      2029 +    write_scalar("diss_rate",    diss_rate);                 
      2030 +}                                                            
      2031 +#endif  // HYDRATE_ENABLE                                    
      2032 +                                                             
      2033  /***********  大小端翻转工具保持不变 ***********/            
      2034                                                               
      2035  __host__ void SwapEnd(double& v)                             

● Step 4: Implement run_stage_hydrate in sim_utils.cu — add after the existing
  run_stage:

  Read 1 file (ctrl+o to expand)
● Update(sim_utils.cu)
  ⎿  Added 159 lines
      737  }                                                             
      738                                                                
      739                                                                
      740 +#ifdef HYDRATE_ENABLE                                         
      741 +//                                                            
          +============================================================  
      742 +// run_stage_hydrate：含水合物物理耦合的时间推进主循环        
      743 +//                                                            
          +------------------------------------------------------------  
      744 +// 耦合顺序（Yang 2024 Figure S2）：                          
      745 +//   Flow（evolution_all）→ Conc → LatentHeat → Thermal → VOP 
      746 +//                                                            
          +============================================================  
      747 +                                                              
      748 +// 宿主端辅助：从设备拷贝水合物场到主机缓冲（用于 VTK 输出）  
      749 +static void copy_hydrate_to_host(const Therm_dev& TH,         
      750 +                                  const Conc_dev&  CN,        
      751 +                                  const VOP_dev&   VP,        
      752 +                                  HydrateHost&     HH)        
      753 +{                                                             
      754 +    const size_t n = (size_t)NX * NY;                         
      755 +    CK(cudaMemcpy(HH.T.data(),         TH.T,                  
          +n*sizeof(double), cudaMemcpyDeviceToHost));                   
      756 +    CK(cudaMemcpy(HH.Cm.data(),        CN.Cm,                 
          +n*sizeof(double), cudaMemcpyDeviceToHost));                   
      757 +    CK(cudaMemcpy(HH.Vh.data(),        VP.Vh,                 
          +n*sizeof(double), cudaMemcpyDeviceToHost));                   
      758 +    CK(cudaMemcpy(HH.diss_rate.data(), VP.diss_rate,          
          +n*sizeof(double), cudaMemcpyDeviceToHost));                   
      759 +}                                                             
      760 +                                                              
      761 +// 水合物诊断：计算 hydrate_volume_frac 和 Q_dissociation     
      762 +static void update_hydrate_diagnostics(const VOP_dev& VP,     
      763 +                                        double Vh_init_total, 
      764 +                                        RunResult& R)         
      765 +{                                                             
      766 +    const size_t n = (size_t)NX * NY;                         
      767 +    // GPU → host 求和（规模 300×300 = 90000，直接 host       
          +累加足够快）                                                  
      768 +    std::vector<double> vh_h(n), dr_h(n);                     
      769 +    CK(cudaMemcpy(vh_h.data(), VP.Vh,                         
          +n*sizeof(double), cudaMemcpyDeviceToHost));                   
      770 +    CK(cudaMemcpy(dr_h.data(), VP.diss_rate,                  
          +n*sizeof(double), cudaMemcpyDeviceToHost));                   
      771 +    double sum_vh = 0.0, sum_dr = 0.0;                        
      772 +    for (size_t i = 0; i < n; ++i) { sum_vh += vh_h[i];       
          +sum_dr += dr_h[i]; }                                          
      773 +    R.hydrate_volume_frac = (Vh_init_total > 0.0) ? sum_vh /  
          +Vh_init_total : 1.0;                                          
      774 +    R.Q_dissociation      = sum_dr;                           
      775 +}                                                             
      776 +                                                              
      777 +RunResult run_stage_hydrate(Fluid_dev& A, Fluid_host& AH,     
      778 +                            Fluid_dev& B, Fluid_host& BH,     
      779 +                            Mix_dev&   MX, Mix_host&   MH,    
      780 +                            Therm_dev& TH, Conc_dev& CN,      
      781 +                            VOP_dev&   VP, HydrateHost& HH,   
      782 +                            SteadyMonitor& SM,                
      783 +                            const StageConfig& cfg,           
      784 +                            const RuntimeParams& P)           
      785 +{                                                             
      786 +    RunResult R{};                                            
      787 +    set_drive_scale(cfg.drive_scale);                         
      788 +    SM.tol_rel     = cfg.tol_rel;                             
      789 +    SM.need_consec = cfg.need_consec;                         
      790 +    SM.consec_hit  = 0;                                       
      791 +                                                              
      792 +    // 计算初始水合物总量（用于归一化 volume_frac）           
      793 +    const size_t n_cells = (size_t)NX * NY;                   
      794 +    std::vector<double> vh0(n_cells);                         
      795 +    CK(cudaMemcpy(vh0.data(), VP.Vh, n_cells*sizeof(double),  
          +cudaMemcpyDeviceToHost));                                     
      796 +    double Vh_init_total = 0.0;                               
      797 +    for (double v : vh0) Vh_init_total += v;                  
      798 +                                                              
      799 +    SM.limiter_window    = SM.interval;                       
      800 +    SM.limiter_log_every = SM.limiter_window;                 
      801 +    SM.limiter_stdout    = false;                             
      802 +    SM.limiter_log_on_hit = false;                            
      803 +    SM.limiter_log_path  = (P.file_dir + "/limiter_log.csv"); 
      804 +                                                              
      805 +    auto t1 = std::chrono::high_resolution_clock::now();      
      806 +                                                              
      807 +    for (int step = 0; step < cfg.max_steps; ++step) {        
      808 +        if (step % SM.limiter_window == 0)                    
          +SM.reset_limiter_counters();                                  
      809 +                                                              
      810 +        // ── 流场演化（现有 MRT 伪势双相 LBM）──             
      811 +        evolution_all(A, B, MX);                              
      812 +                                                              
      813 +        // ── 水合物相变物理（Yang 2024 顺序）──              
      814 +        if (P.hydrate_enable && step >= P.hydrate_start_step) 
          + {                                                            
      815 +            // 1. 浓度场（CST + Kang 反应边界 → diss_rate）   
      816 +            step_conc(CN, TH, MX.ux, MX.uy, A.rho, B.rho, VP, 
          + MX.pointsflag);                                              
      817 +            // 2. 潜热源项                                    
      818 +            compute_latent_heat_source(VP, CN, TH,            
          +MX.pointsflag);                                               
      819 +            // 3. 热场（含源项 S_latent）                     
      820 +            step_thermal(TH, MX.ux, MX.uy, VP.S_latent,       
          +MX.pointsflag, P.T0_inlet);                                   
      821 +            // 4. VOP 固相更新（返回翻转数；>0 时 ghost       
          +已重建）                                                      
      822 +            int n_conv = step_vop(VP, TH, CN, A, B, MX);      
      823 +            if (n_conv > 0) {                                 
      824 +                R.n_converted_total += n_conv;                
      825 +                // 翻转后重新上传润湿性（保持材质→GAw         
          +映射一致）                                                    
      826 +                                                              
          +upload_wettability_table_host(P.thetaA_quartz_deg,            
      827 +                                                              
          +P.thetaA_hydrate_deg,                                         
      828 +                                              P.GBw_quartz,   
      829 +                                              P.GBw_hydrate); 
      830 +            }                                                 
      831 +        }                                                     
      832 +                                                              
      833 +        maybe_save_periodic_checkpoint(P, cfg.tag, step, A,   
          +B, MX);                                                       
      834 +                                                              
      835 +        // ── 稳态检测 + 输出 ──                              
      836 +        if (probe_and_check_steady(step, cfg, SM, A, B, MX,   
          +R)) {                                                         
      837 +            // 输出最终场                                     
      838 +            copy_and_check(A, AH, "A_");                      
      839 +            copy_and_check(B, BH, "B_");                      
      840 +            copy_back_mix(MX, MH);                            
      841 +            std::string prefix = std::string("outputdata_") + 
          + cfg.tag;                                                     
      842 +            std::string title  = std::string(cfg.tag) + "     
          +steady";                                                      
      843 +            std::string vtk_path = P.file_dir + "/" + prefix  
      844 +                + ([&]{ std::ostringstream o; o <<            
          +std::setw(8) << std::setfill('0') << step; return o.str();    
          +}())                                                          
      845 +                + ".vtk";                                     
      846 +            outputvtk(step, P.file_dir, prefix.c_str(),       
          +title.c_str(), AH, BH, MH);                                   
      847 +            if (P.hydrate_enable) {                           
      848 +                copy_hydrate_to_host(TH, CN, VP, HH);         
      849 +                outputvtk_append_hydrate(vtk_path, HH.T,      
          +HH.Cm, HH.Vh, HH.diss_rate);                                  
      850 +                update_hydrate_diagnostics(VP, Vh_init_total, 
          + R);                                                          
      851 +            }                                                 
      852 +            break;                                            
      853 +        }                                                     
      854 +                                                              
      855 +        if (P.OUTPUT_EVERY > 0 && step % P.OUTPUT_EVERY == 0) 
          + {                                                            
      856 +            copy_and_check(A, AH, "A_");                      
      857 +            copy_and_check(B, BH, "B_");                      
      858 +            copy_back_mix(MX, MH);                            
      859 +            std::string prefix = std::string("outputdata_") + 
          + cfg.tag;                                                     
      860 +            std::string title  = std::string(cfg.tag) + "     
          +running";                                                     
      861 +            std::string vtk_path = P.file_dir + "/" + prefix  
      862 +                + ([&]{ std::ostringstream o; o <<            
          +std::setw(8) << std::setfill('0') << step; return o.str();    
          +}())                                                          
      863 +                + ".vtk";                                     
      864 +            outputvtk(step, P.file_dir, prefix.c_str(),       
          +title.c_str(), AH, BH, MH);                                   
      865 +            if (P.hydrate_enable) {                           
      866 +                copy_hydrate_to_host(TH, CN, VP, HH);         
      867 +                outputvtk_append_hydrate(vtk_path, HH.T,      
          +HH.Cm, HH.Vh, HH.diss_rate);                                  
      868 +                update_hydrate_diagnostics(VP, Vh_init_total, 
          + R);                                                          
      869 +                printf("[hydrate step=%d] Vh_frac=%.4f        
          +Q_diss=%.4e  n_conv_total=%d\n",                              
      870 +                       step, R.hydrate_volume_frac,           
          +R.Q_dissociation, R.n_converted_total);                       
      871 +            }                                                 
      872 +        }                                                     
      873 +                                                              
      874 +        // 水合物耗尽终止条件                                 
      875 +        if (P.hydrate_enable && step % SM.interval == 0) {    
      876 +            update_hydrate_diagnostics(VP, Vh_init_total, R); 
      877 +            if (R.hydrate_volume_frac < P.vop_terminate_frac) 
          + {                                                            
      878 +                printf("[VOP] 水合物分数 %.4f < terminate     
          +阈值 %.4f，提前终止。\n",                                     
      879 +                       R.hydrate_volume_frac,                 
          +P.vop_terminate_frac);                                        
      880 +                break;                                        
      881 +            }                                                 
      882 +        }                                                     
      883 +    }                                                         
      884 +                                                              
      885 +    auto t2 = std::chrono::high_resolution_clock::now();      
      886 +    printf("[%s hydrate] time = %.3f s\n", cfg.tag,           
      887 +           std::chrono::duration<double>(t2 - t1).count());   
      888 +                                                              
      889 +    if (P.ENABLE_CKPT) {                                      
      890 +        int step_to_save = (R.steady && R.steady_step >= 0) ? 
          + R.steady_step : R.last_probe_step;                           
      891 +        if (step_to_save < 0) step_to_save = 0;               
      892 +        save_checkpoint(P.ckpt_dir.c_str(), cfg.tag,          
          +step_to_save, A, B, MX, P);                                   
      893 +    }                                                         
      894 +                                                              
      895 +    return R;                                                 
      896 +}                                                             
      897 +#endif  // HYDRATE_ENABLE                                     
      898 +                                                              
      899  RunResult run_equilibrate_then_flow(Fluid_dev& A, Fluid_host& 
            AH,                                                          
      900                                      Fluid_dev& B, Fluid_host& 
            BH,                                                          
      901                                      Mix_dev&   M, Mix_host&   
            MH,                                                          

● Step 5: Extend main.cu to allocate/init/free hydrate objects and call 
  run_stage_hydrate

● Update(main.cu)
  ⎿  Added 59 lines
       72      SM.limiter_stdout    = false;           // 禁止控制台打印 
       73      SM.limiter_log_path  = P.file_dir + "/limiter_log.csv";   
       74                                                                
       75 +#ifdef HYDRATE_ENABLE                                         
       76 +    // 5-H) 水合物模式初始化（仅 hydrate_enable               
          +时有意义；alloc 始终执行以保持统一生命周期）                  
       77 +    Therm_dev TH_dev;  Conc_dev CN_dev;  VOP_dev VP_dev;      
       78 +    HydrateHost HH;                                           
       79 +    if (P.hydrate_enable) {                                   
       80 +        init_device_variable_hydrate(P);                      
       81 +        alloc_therm(TH_dev);                                  
       82 +        alloc_conc(CN_dev);                                   
       83 +        alloc_vop(VP_dev);                                    
       84 +        init_thermal_field(TH_dev, M_dev.pointsflag);         
       85 +        init_conc_field(CN_dev, M_dev.pointsflag);            
       86 +        init_vop(VP_dev, M_dev.pointsflag);                   
       87 +        printf("[hydrate] 水合物场初始化完成。\n");           
       88 +    }                                                         
       89 +#endif                                                        
       90 +                                                              
       91      // 5) 时间推进：把推进、监测、输出收敛前的所有循环交给    
           run_time_loop                                                 
       92      auto t1 = high_resolution_clock::now();                   
       93 +                                                              
       94 +#ifdef HYDRATE_ENABLE                                         
       95 +    RunResult R;                                              
       96 +    if (P.hydrate_enable) {                                   
       97 +        // 流场先跑平衡（不带水合物），再切换到带水合物的流段 
       98 +        // 阶段1：phase separation（无驱动）                  
       99 +        StageConfig eq;                                       
      100 +        eq.tag            = "eq";                             
      101 +        eq.drive_scale    = 0.0;                              
      102 +        eq.tol_rel        = P.eq_tol_rel;                     
      103 +        eq.need_consec    = P.eq_need_consec;                 
      104 +        eq.max_steps      = P.eq_max_steps;                   
      105 +        eq.require_abs_quiet = false;                         
      106 +        eq.q_abs_eps      = P.eq_q_abs_eps;                   
      107 +        RunResult R_eq = run_stage(A_dev, AH, B_dev, BH,      
          +M_dev, MH, SM, eq, P);                                        
      108 +                                                              
      109 +        // 阶段2：驱动流 + 水合物物理                         
      110 +        StageConfig flow;                                     
      111 +        flow.tag           = "flow";                          
      112 +        flow.drive_scale   = 1.0;                             
      113 +        flow.tol_rel       = P.flow_tol_rel;                  
      114 +        flow.need_consec   = P.flow_need_consec;              
      115 +        flow.max_steps     = P.flow_max_steps;                
      116 +        flow.require_abs_quiet = false;                       
      117 +        flow.q_abs_eps     = P.eq_q_abs_eps;                  
      118 +        R = run_stage_hydrate(A_dev, AH, B_dev, BH, M_dev,    
          +MH,                                                           
      119 +                               TH_dev, CN_dev, VP_dev, HH,    
      120 +                               SM, flow, P);                  
      121 +        R.eq_steady      = R_eq.steady;                       
      122 +        R.eq_steady_step = R_eq.steady_step;                  
      123 +    } else {                                                  
      124 +        R = run_equilibrate_then_flow(A_dev, AH, B_dev, BH,   
          +M_dev, MH, SM, P);                                            
      125 +    }                                                         
      126 +#else                                                         
      127      RunResult R = run_equilibrate_then_flow(A_dev, AH, B_dev, 
            BH, M_dev, MH, SM, P);                                       
      128 +#endif                                                        
      129 +                                                              
      130      auto t2 = high_resolution_clock::now();                   
      131      printf("Total time = %.3f s\n",                           
           duration<double>(t2-t1).count());                             
      132                                                                
     ...
      83                                                                 
      84      // 7) 释放资源并复位设备                                   
      85      free_all(A_dev,B_dev,M_dev);                               
      86 +#ifdef HYDRATE_ENABLE                                          
      87 +    if (P.hydrate_enable) {                                    
      88 +        free_therm(TH_dev);                                    
      89 +        free_conc(CN_dev);                                     
      90 +        free_vop(VP_dev);                                      
      91 +    }                                                          
      92 +#endif                                                         
      93      cudaDeviceReset();                                         
      94      return 0;                                                  
      95  }                                                              

● Now I need to add #include <iomanip> (for setw/setfill) and #include <sstream>
   to sim_utils.cu — check if they're already there:

  Read 1 file (ctrl+o to expand)

● <sstream> is there but not <iomanip>. Add it:

● Update(sim_utils.cu)
  ⎿  Added 1 line
      3  #include "unified_cuda_error_check.cuh"                         
      4  #include <fstream>                                              
      5  #include <iostream>                                             
      6 +#include <iomanip>                                              
      7  #include <algorithm>                                            
      8  #include <cctype>                                               
      9  #include <sstream>                                              

● Also need <vector> in sim_utils.cu for copy_hydrate_to_host:

  Searched for 1 pattern (ctrl+o to expand)

● Update(sim_utils.cu)
  ⎿  Added 1 line
       4  #include <fstream>                                             
       5  #include <iostream>                                            
       6  #include <iomanip>                                             
       7 +#include <vector>                                              
       8  #include <algorithm>                                           
       9  #include <cctype>                                              
      10  #include <sstream>                                             

● Now compile both targets:

● Bash(bash compile.sh hydrate 2>&1)
  ⎿  main.cu(95): warning #550-D: variable "R" was set but never used
           RunResult R;
                     ^
     … +3 lines (ctrl+o to expand)
  ⎿  (timeout 3m)

● Only one warning — the RunResult R declaration in main.cu is initialized
  inside the if/else block and the compiler sees it as potentially unset. Fix
  with direct initialization:

● Update(main.cu)
  ⎿  Added 1 line, removed 1 line
      92      auto t1 = high_resolution_clock::now();                    
      93                                                                 
      94  #ifdef HYDRATE_ENABLE                                          
      95 -    RunResult R;                                               
      95 +    RunResult R{};                                             
      96      if (P.hydrate_enable) {                                    
      97          // 流场先跑平衡（不带水合物），再切换到带水合物的流段  
      98          // 阶段1：phase separation（无驱动）                   

● Bash(bash compile.sh hydrate 2>&1 && bash compile.sh 2>&1 | tail -3)
  ⎿  main.cu(95): warning #550-D: variable "R" was set but never used
           RunResult R{};
                     ^
     … +3 lines (ctrl+o to expand)
  ⎿  (timeout 3m)

● The warning persists because nvcc sees R{} as unused when P.hydrate_enable is
  false (the else branch assigns a different R). This is a structural issue —
  let me fix the pattern to avoid double declaration:

  Read 1 file (ctrl+o to expand)

● The issue: nvcc is warning because when P.hydrate_enable==false the R{}
  default-constructed value is overwritten by run_equilibrate_then_flow. This is
   benign — suppress it with (void)R or just restructure slightly. Simplest fix:
   remove the default init and always assign inside the if/else:

● Update(main.cu)
  ⎿  Added 23 lines, removed 30 lines
       92      auto t1 = high_resolution_clock::now();                   
       93                                                                
       94  #ifdef HYDRATE_ENABLE                                         
       95 -    RunResult R{};                                            
       96 -    if (P.hydrate_enable) {                                   
       97 -        // 流场先跑平衡（不带水合物），再切换到带水合物的流段 
       98 -        // 阶段1：phase separation（无驱动）                  
       99 -        StageConfig eq;                                       
      100 -        eq.tag            = "eq";                             
      101 -        eq.drive_scale    = 0.0;                              
      102 -        eq.tol_rel        = P.eq_tol_rel;                     
      103 -        eq.need_consec    = P.eq_need_consec;                 
      104 -        eq.max_steps      = P.eq_max_steps;                   
      105 -        eq.require_abs_quiet = false;                         
      106 -        eq.q_abs_eps      = P.eq_q_abs_eps;                   
      107 -        RunResult R_eq = run_stage(A_dev, AH, B_dev, BH,      
          -M_dev, MH, SM, eq, P);                                        
       95 +    RunResult R = [&]() -> RunResult {                        
       96 +        if (P.hydrate_enable) {                               
       97 +            // 阶段1：phase                                   
          +separation（无驱动，水合物场不参与）                          
       98 +            StageConfig eq;                                   
       99 +            eq.tag = "eq"; eq.drive_scale = 0.0;              
      100 +            eq.tol_rel = P.eq_tol_rel; eq.need_consec =       
          +P.eq_need_consec;                                             
      101 +            eq.max_steps = P.eq_max_steps;                    
          +eq.require_abs_quiet = false;                                 
      102 +            eq.q_abs_eps = P.eq_q_abs_eps;                    
      103 +            RunResult R_eq = run_stage(A_dev, AH, B_dev, BH,  
          +M_dev, MH, SM, eq, P);                                        
      104                                                                
      105 -        // 阶段2：驱动流 + 水合物物理                         
      106 -        StageConfig flow;                                     
      107 -        flow.tag           = "flow";                          
      108 -        flow.drive_scale   = 1.0;                             
      109 -        flow.tol_rel       = P.flow_tol_rel;                  
      110 -        flow.need_consec   = P.flow_need_consec;              
      111 -        flow.max_steps     = P.flow_max_steps;                
      112 -        flow.require_abs_quiet = false;                       
      113 -        flow.q_abs_eps     = P.eq_q_abs_eps;                  
      114 -        R = run_stage_hydrate(A_dev, AH, B_dev, BH, M_dev,    
          -MH,                                                           
      115 -                               TH_dev, CN_dev, VP_dev, HH,    
      116 -                               SM, flow, P);                  
      117 -        R.eq_steady      = R_eq.steady;                       
      118 -        R.eq_steady_step = R_eq.steady_step;                  
      119 -    } else {                                                  
      120 -        R = run_equilibrate_then_flow(A_dev, AH, B_dev, BH,   
          -M_dev, MH, SM, P);                                            
      121 -    }                                                         
      105 +            // 阶段2：驱动流 + 水合物物理                     
      106 +            StageConfig flow;                                 
      107 +            flow.tag = "flow"; flow.drive_scale = 1.0;        
      108 +            flow.tol_rel = P.flow_tol_rel; flow.need_consec = 
          + P.flow_need_consec;                                          
      109 +            flow.max_steps = P.flow_max_steps;                
          +flow.require_abs_quiet = false;                               
      110 +            flow.q_abs_eps = P.eq_q_abs_eps;                  
      111 +            RunResult Rf = run_stage_hydrate(A_dev, AH,       
          +B_dev, BH, M_dev, MH,                                         
      112 +                                              TH_dev, CN_dev, 
          + VP_dev, HH,                                                  
      113 +                                              SM, flow, P);   
      114 +            Rf.eq_steady = R_eq.steady; Rf.eq_steady_step =   
          +R_eq.steady_step;                                             
      115 +            return Rf;                                        
      116 +        }                                                     
      117 +        return run_equilibrate_then_flow(A_dev, AH, B_dev,    
          +BH, M_dev, MH, SM, P);                                        
      118 +    }();                                                      
      119  #else                                                         
      120      RunResult R = run_equilibrate_then_flow(A_dev, AH, B_dev, 
            BH, M_dev, MH, SM, P);                                       
      121  #endif                                                        

● Bash(bash compile.sh hydrate 2>&1 && echo "=== flow-only ===" && bash
      compile.sh 2>&1 | tail -3)
  ⎿  === flow-only ===
  ⎿  (timeout 3m)

● Both compile clean with zero warnings. Verify sizes and key symbols:

● Bash(ls -lh mcmp_sim mcmp_sim_hydrate && nm mcmp_sim_hydrate | grep -E
      "run_stage_hydrate|outputvtk_append|copy_hydrate|update_hydrate_diag" |
      head -10)
  ⎿  -rwx------ 1 server server 2.5M  3月 25 19:59 mcmp_sim
     -rwx------ 1 server server 2.8M  3月 25 19:58 mcmp_sim_hydrate
     0000000000026700 T _Z17run_stage_hydrateR9Fluid_devR10Fluid_hostS0_S2_
     … +12 lines (ctrl+o to expand)

● Phase 4 + Phase 5 完成，两个编译目标零警告零错误。完整交付内容：

  ---
  Phase 4（潜热源项） — 已在 Phase 2 末尾实现的 kernel_compute_latent_heat
  正式接入耦合循环。

  Phase 5（全耦合集成） — 本轮新增：

  ┌──────────────┬───────────────────────────────────────────────────────────┐
  │     文件     │                           改动                            │
  ├──────────────┼───────────────────────────────────────────────────────────┤
  │              │ run_stage_hydrate()：每步循环顺序为 evolution_all →       │
  │ sim_utils.cu │ step_conc → compute_latent_heat_source → step_thermal →   │
  │              │ step_vop；周期输出+稳态输出均调用                         │
  │              │ outputvtk_append_hydrate；水合物饱和度终止检测            │
  ├──────────────┼───────────────────────────────────────────────────────────┤
  │ sim_utils.h  │ HydrateHost 结构体（T/Cm/Vh/diss_rate                     │
  │              │ 宿主缓冲）；run_stage_hydrate 函数声明                    │
  ├──────────────┼───────────────────────────────────────────────────────────┤
  │              │ outputvtk_append_hydrate()：以追加模式向已生成的 .vtk     │
  │ LBM.cu       │ 文件末尾写入                                              │
  │              │ temperature/concentration/hydrate_Vh/diss_rate 四个标量场 │
  ├──────────────┼───────────────────────────────────────────────────────────┤
  │ LBM.h        │ outputvtk_append_hydrate 声明（#ifdef HYDRATE_ENABLE）    │
  ├──────────────┼───────────────────────────────────────────────────────────┤
  │              │ #ifdef HYDRATE_ENABLE 块：分配/初始化                     │
  │ main.cu      │ Therm_dev/Conc_dev/VOP_dev/HydrateHost；调用              │
  │              │ run_stage_hydrate；释放资源                               │
  └──────────────┴───────────────────────────────────────────────────────────┘

  运行方式：
  # 编译
  bash compile.sh hydrate

  # 运行（水合物模式：params.txt 中 hydrate_enable 1）
  ./mcmp_sim_hydrate params.txt

  Phase 6（基准验证参数文件）可随需展开。

✻ Brewed for 9m 10s

❯ 现在请进入 Phase 6（基准验证）                                                

● Let me read params.txt and check available geometry options first:

  Searched for 2 patterns, read 2 files (ctrl+o to expand)
● Now I have everything I need. Let me create all benchmark files and
  Research.md in parallel:

● Write(benchmark_diffusion.txt)
  ⎿  Wrote 87 lines to benchmark_diffusion.txt
     # ================================================================
     # BM-1: 纯热扩散基准（Phase 1 验证）
     # ----------------------------------------------------------------
     # 目标：空域（无障碍），零速度驱动，固定入口/出口温度边界。
     # 预期：稳态 T(y) 应为线性剖面，L2 误差 < 1e-4。
     #
     # 验证方法：
     #   1) 运行至稳态（flow_max_steps 步或者 eq_tol_rel 收敛）
     #   2) 读取 outputdata_flow*.vtk 中 temperature 场
     #   3) 对每列 x 取均值，与 T_lin(y) = T0_init + (T0_inlet -
     T0_init)*y/(NY-1) 比较
     … +77 lines (ctrl+o to expand)

● Write(benchmark_conjugate_heat.txt)
  ⎿  Wrote 88 lines to benchmark_conjugate_heat.txt
     # ================================================================
     # BM-2: 共轭热传递基准（Phase 1 验证）
     # ----------------------------------------------------------------
     # 目标：含石英固体颗粒圆柱阵（无水合物），验证固-液界面
     #       温度梯度跳变是否满足共轭条件：λ_fluid * ∇T_f = λ_solid * ∇T_s
     #
     # 验证方法（Zhang 2019 §4.2 类比）：
     #   1) 运行至热场稳态
     #   2) 取穿过颗粒中心的水平剖面 T(x, y=NY/2)
     #   3) 在固-液界面两侧各取 3 格，拟合线性梯度
     … +78 lines (ctrl+o to expand)

● Write(benchmark_reactive.txt)
  ⎿  Wrote 98 lines to benchmark_reactive.txt
     # ================================================================
     # BM-3: 反应-扩散基准（Phase 2 + Phase 4 验证）
     # ----------------------------------------------------------------
     # 目标：一维反应-扩散稳态剖面验证。
     #       平板水合物（上半域，y > NY/2），水相（下半域），零对流。
     #
     # 解析解（稳态，1D，反应边界在 y = NY/2）：
     #   Cm(y) = Csat * (1 - exp(-(NY/2 - y) * sqrt(k_r_eff / D_latt)))
     #   其中 k_r_eff = k0_rxn_latt * exp(-Ea/RT) * (1 - Cm/Csat)
     #   近似（线性化）：Cm(y) ≈ Cm∞ * (1-exp(-y/L_r)), L_r = sqrt(D/k_r)
     … +88 lines (ctrl+o to expand)

● Write(benchmark_vop.txt)
  ⎿  Wrote 96 lines to benchmark_vop.txt
     # ================================================================
     # BM-4: VOP 质量守恒基准（Phase 3 验证）
     # ----------------------------------------------------------------
     # 目标：单个水合物圆柱，恒定小反应速率（k_r=const），
     #       验证：Σ Vh + Σ (Cm * Vm_latt) ≈ const（±0.5%/1000步）
     #
     # 验证步骤：
     #   1) 记录初始 Σ Vh_0 = 水合物节点数（每节点 Vh=1）
     #   2) 每 1000 步记录一次 Σ Vh 和 Σ Cm
     #   3) 计算 total_mass(t) = Σ Vh + Σ (Cm * Vm_latt / V_cell)
     … +86 lines (ctrl+o to expand)

● Write(benchmark_full_coupling.txt)
  ⎿  Wrote 98 lines to benchmark_full_coupling.txt
     # ================================================================
     # BM-5: 全耦合趋势基准（Phase 5 验证）
     # ----------------------------------------------------------------
     # 目标：仿 Yang 2024 / Zhang 2019 典型条件，验证五个定性趋势：
     #   (1) Vh 单调递减（水合物持续分解）
     #   (2) 水合物面附近 T 下降（吸热分解）
     #   (3) 水相中 Cm 梯度从水合物面向远场衰减
     #   (4) 节点翻转后流场无大速度冲击（re-init 效果）
     #   (5) 气-水界面处 Cm 趋向 Henry 平衡值
     #
     … +88 lines (ctrl+o to expand)

● Now create the Python validation script and Research.md in parallel:

● Write(validate_benchmarks.py)
  ⎿  Wrote 280 lines to validate_benchmarks.py
     #!/usr/bin/env python3
     """
     validate_benchmarks.py
     ======================
     自动验证水合物分解 LBM 模拟的基准测试结果。

     用法：
         python3 validate_benchmarks.py --bm BM1 --dir results/
         python3 validate_benchmarks.py --all --dir results/

     … +270 lines (ctrl+o to expand)

● Now Research.md:

● Write(Research.md)
  ⎿  Wrote 238 lines to Research.md
     # 甲烷水合物分解多物理场 LBM 数值模型文档

     > 参考文献：
     > **Yang 2024** — Yang et al., *Three-dimensional pore-scale study of 
     methane hydrate dissociation mechanisms based on micro-CT images*
     (2024)
     > **Zhang 2019** — Zhang et al., *Pore-Scale Investigation of Methane 
     Hydrate Dissociation Using the Lattice Boltzmann Method* (2019)

     ---

     ## 1. 物理过程概述

     … +228 lines (ctrl+o to expand)

● Verify all files exist and the hydrate binary still compiles clean:

● Bash(ls -lh benchmark_*.txt validate_benchmarks.py Research.md && bash
      compile.sh hydrate 2>&1 | tail -3)
  ⎿  -rw------- 1 server server 2.6K  3月 25 20:12 benchmark_conjugate_heat
     .txt
     -rw------- 1 server server 3.0K  3月 25 20:12 benchmark_diffusion.txt
     … +6 lines (ctrl+o to expand)
  ⎿  (timeout 2m)

● Phase 6 全部完成。交付内容：

  ---
  5个基准参数文件：

  ┌──────────────────────┬───────────────┬──────────────────────────────────┐
  │         文件         │   测试内容    │             关键设置             │
  ├──────────────────────┼───────────────┼──────────────────────────────────┤
  │ benchmark_diffusion. │ 纯热扩散（BM- │ k0_rxn=0，latent_heat=0，零驱动  │
  │ txt                  │ 1）           │ ，空域                           │
  ├──────────────────────┼───────────────┼──────────────────────────────────┤
  │ benchmark_conjugate_ │ 共轭热传递（B │ 石英圆柱阵，coat_thick=0，无反应 │
  │ heat.txt             │ M-2）         │ ，λ比=1.5验证                    │
  ├──────────────────────┼───────────────┼──────────────────────────────────┤
  │ benchmark_reactive.t │ 反应-扩散（BM │ 等温近似（Ea_rxn=0），小         │
  │ xt                   │ -3）          │ k0_rxn=1.0，零驱动               │
  ├──────────────────────┼───────────────┼──────────────────────────────────┤
  │ benchmark_vop.txt    │ VOP 质量守恒  │ k0_rxn=0.01，OUTPUT_EVERY=1000， │
  │                      │ （BM-4）      │ 终止于50%                        │
  ├──────────────────────┼───────────────┼──────────────────────────────────┤
  │ benchmark_full_coupl │ 全耦合趋势（B │ coating 几何，T_inlet=283K >     │
  │ ing.txt              │ M-5）         │ 分解温度，完整 Kim-Bishnoi       │
  └──────────────────────┴───────────────┴──────────────────────────────────┘

  验证脚本 validate_benchmarks.py：读 VTK、自动计算 L2 误差 / λ 梯度比 /
  质量守恒漂移，--bm BM1..BM5 或 --all。

  Research.md：12节完整文档，涵盖控制方程、D2Q5 MRT
  矩阵推导、格子单位换算表、基准测试概览、实验验证流程、已知局限。

  运行命令：
  bash compile.sh hydrate
  ./mcmp_sim_hydrate benchmark_diffusion.txt
  python3 validate_benchmarks.py --bm BM1 --dir results/

✻ Churned for 6m 25s