What's New
------------

 * **Support AVX512 level-3/1f/1 kernels on SkylakeX** 
 * **Unified GEMM/GEMMSUP for both small and large problem sizes** 

Key Features
------------

 * **Edges self-adaptive**: Leverages macro templates to make it much easier to code up the numerous edge case kernels that arise for the SkylakeX's larger microtile;

 * **From inline ASM to instrinsics**: In order to accommodate the complex process logic caused by edges self-adaptive, coded the kernels with the instruction instead of inline ASM. Additional performance losses are avoided by following some necessary compiler efficiency best practices;

 * **Unified GEMMSUP into GEMM for both small and large problem sizes**  by introducing matrix-shape conditioned cntl tree. Additional configure --enable-fip --disable-fip (default) were applied to enable/disable this feature. Packing is still work along (instead of fused in kernel) in the unified processs.

Performance Benchmark
------------

----------------------------------------------------------------------------------------
      Run on (32 X 2500 MHz CPU s)
      CPU Caches:
      L1 Data 32 KiB (x16)
      L1 Instruction 32 KiB (x16)
      L2 Unified 1024 KiB (x16)
      L3 Unified 36608 KiB (x1)
      Load Average: 3.43, 2.25, 1.76
----------------------------------------------------------------------------------------      
      Benchmark (Unified GEMM)         Time             CPU   Iterations UserCounters...
    BM_blis_gemm<double>/32         2642 ns         2642 ns       267571 Threads=1 col_major=1 items_per_second=12.405G/s
    BM_blis_gemm<double>/64         9373 ns         9372 ns        74638 Threads=1 col_major=1 items_per_second=27.9703G/s
    BM_blis_gemm<double>/128       62268 ns        62266 ns        11260 Threads=1 col_major=1 items_per_second=33.6807G/s
    BM_blis_gemm<double>/256      548689 ns       548668 ns         1274 Threads=1 col_major=1 items_per_second=30.5781G/s
    BM_blis_gemm<double>/512     4194179 ns      4194023 ns          167 Threads=1 col_major=1 items_per_second=32.0021G/s
    BM_blis_gemm<double>/1024   32267018 ns     32265855 ns           22 Threads=1 col_major=1 items_per_second=33.278G/s
    BM_blis_gemm<double>/2048  256573386 ns    256563719 ns            3 Threads=1 col_major=1 items_per_second=33.4807G/s
    BM_blis_gemm<double>/4096 2058634680 ns   2058546217 ns            1 Threads=1 col_major=1 items_per_second=33.3825G/s
    BM_blis_gemm<double>/8192 1.6417e+10 ns   1.6416e+10 ns            1 Threads=1 col_major=1 items_per_second=33.4882G/s
    BM_blis_gemm<float>/32          2002 ns         2002 ns       349674 Threads=1 col_major=1 items_per_second=16.3675G/s
    BM_blis_gemm<float>/64          5708 ns         5708 ns       123029 Threads=1 col_major=1 items_per_second=45.9248G/s
    BM_blis_gemm<float>/128        31549 ns        31548 ns        22184 Threads=1 col_major=1 items_per_second=66.4745G/s
    BM_blis_gemm<float>/256       243990 ns       243981 ns         2871 Threads=1 col_major=1 items_per_second=68.7644G/s
    BM_blis_gemm<float>/512      2078766 ns      2078688 ns          337 Threads=1 col_major=1 items_per_second=64.5685G/s
    BM_blis_gemm<float>/1024    15968272 ns     15967672 ns           44 Threads=1 col_major=1 items_per_second=67.2447G/s
    BM_blis_gemm<float>/2048   124904741 ns    124899993 ns            6 Threads=1 col_major=1 items_per_second=68.7745G/s
    BM_blis_gemm<float>/4096  1004359953 ns   1004311338 ns            1 Threads=1 col_major=1 items_per_second=68.4245G/s
    BM_blis_gemm<float>/8192  7995180868 ns   7994852260 ns            1 Threads=1 col_major=1 items_per_second=68.7637G/s
      Benchmark (Seperated SUP)        Time             CPU   Iterations UserCounters...
    BM_blis_gemm<double>/32         2544 ns         2544 ns       275095 Threads=1 col_major=1 items_per_second=12.8818G/s
    BM_blis_gemm<double>/64         9112 ns         9112 ns        76765 Threads=1 col_major=1 items_per_second=28.7701G/s
    BM_blis_gemm<double>/128       61274 ns        61271 ns        11422 Threads=1 col_major=1 items_per_second=34.2275G/s
    BM_blis_gemm<double>/256      546364 ns       546340 ns         1283 Threads=1 col_major=1 items_per_second=30.7084G/s
    BM_blis_gemm<double>/512     4195056 ns      4194863 ns          167 Threads=1 col_major=1 items_per_second=31.9957G/s
    BM_blis_gemm<double>/1024   32359881 ns     32358664 ns           22 Threads=1 col_major=1 items_per_second=33.1825G/s
    BM_blis_gemm<double>/2048  256216233 ns    256201163 ns            3 Threads=1 col_major=1 items_per_second=33.5281G/s
    BM_blis_gemm<double>/4096 2095371202 ns   2095280935 ns            1 Threads=1 col_major=1 items_per_second=32.7973G/s
    BM_blis_gemm<double>/8192 1.6393e+10 ns   1.6392e+10 ns            1 Threads=1 col_major=1 items_per_second=33.538G/s
    BM_blis_gemm<float>/32          1987 ns         1987 ns       359147 Threads=1 col_major=1 items_per_second=16.4939G/s
    BM_blis_gemm<float>/64          5514 ns         5514 ns       125913 Threads=1 col_major=1 items_per_second=47.5443G/s
    BM_blis_gemm<float>/128        31023 ns        31021 ns        22572 Threads=1 col_major=1 items_per_second=67.6046G/s
    BM_blis_gemm<float>/256       244843 ns       244834 ns         2859 Threads=1 col_major=1 items_per_second=68.5249G/s
    BM_blis_gemm<float>/512      2071280 ns      2071202 ns          338 Threads=1 col_major=1 items_per_second=64.8018G/s
    BM_blis_gemm<float>/1024    15943468 ns     15942870 ns           44 Threads=1 col_major=1 items_per_second=67.3493G/s
    BM_blis_gemm<float>/2048   124711064 ns    124703512 ns            6 Threads=1 col_major=1 items_per_second=68.8829G/s
    BM_blis_gemm<float>/4096  1000378005 ns   1000333945 ns            1 Threads=1 col_major=1 items_per_second=68.6965G/s
    BM_blis_gemm<float>/8192  7945549913 ns   7945214561 ns            1 Threads=1 col_major=1 items_per_second=69.1933G/s
