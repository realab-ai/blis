What's New
------------

 * **Support AVX512 GEMM and GEMMSUP kernels on SKX.** 

Key Features
------------

 * **Edges self-adaptive**: Instead of define different kernels for different edge cases, all the new kernels are self-adaptive with the edge cases:
   * cv_d24x8 (used by GEMM and GEMMSUP): column-vector double kernel at size 24m x 8n;
   * rd_d24x8 (used by GEMMSUP): row-dot double kernel at size 24m x 8n;
   * cv_s48x8 (used by GEMM and GEMMSUP): column-vector float kernel at size 48m x 8n;
   * rd_s48x8 (used by GEMMSUP): row-dot float kernel at size 48m x 8n;

    Obviously, such adaptive features can help remove a large number of different GEMMSUP kernel definitions. And one step further, GEMM kernels with edge adaptive feature are special cases of GEMMSUP kernels which will ease the maintenance of GEMM/GEMMSUP kernels.

 * **From inline ASM to instrinsics**: In order to accommodate the complex process logic caused by edges self-adaptive, coded the kernels with the instruction instead of inline ASM. Additional performance losses are avoided by following some necessary compiler efficiency best practices:

 * **No change on the baseline codes from BLIS 0.9.0 except:** 
   * Modified the skx config to enable GEMMSUP;
   * Added configure --enable-diagnosis (default disabled) to display some formated debug informtion;
   * Added configure --enable-fip (defaullt disabled) for the coming Fusing in Packing feature development.

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
      Benchmark                        Time             CPU   Iterations UserCounters...
    BM_blis_gemm<double>/32         2511 ns         2511 ns       278633 Threads=1 col_major=1 items_per_second=13.0488G/s
    BM_blis_gemm<double>/64         9231 ns         9230 ns        75832 Threads=1 col_major=1 items_per_second=28.4G/s
    BM_blis_gemm<double>/128       63031 ns        63029 ns        11026 Threads=1 col_major=1 items_per_second=33.273G/s
    BM_blis_gemm<double>/256      548068 ns       548036 ns         1275 Threads=1 col_major=1 items_per_second=30.6133G/s
    BM_blis_gemm<double>/512     4191055 ns      4190899 ns          167 Threads=1 col_major=1 items_per_second=32.026G/s
    BM_blis_gemm<double>/1024   32175549 ns     32174341 ns           22 Threads=1 col_major=1 items_per_second=33.3726G/s
    BM_blis_gemm<double>/2048  255924702 ns    255914868 ns            3 Threads=1 col_major=1 items_per_second=33.5656G/s
    BM_blis_gemm<double>/4096 2053021241 ns   2052933870 ns            1 Threads=1 col_major=1 items_per_second=33.4738G/s
    BM_blis_gemm<double>/8192 1.6349e+10 ns   1.6348e+10 ns            1 Threads=1 col_major=1 items_per_second=33.6286G/s
    BM_blis_gemm<float>/32          1945 ns         1945 ns       359971 Threads=1 col_major=1 items_per_second=16.8508G/s
    BM_blis_gemm<float>/64          5539 ns         5539 ns       126389 Threads=1 col_major=1 items_per_second=47.3265G/s
    BM_blis_gemm<float>/128        30473 ns        30472 ns        22998 Threads=1 col_major=1 items_per_second=68.8216G/s
    BM_blis_gemm<float>/256       239055 ns       239043 ns         2929 Threads=1 col_major=1 items_per_second=70.1848G/s
    BM_blis_gemm<float>/512      2085223 ns      2085130 ns          337 Threads=1 col_major=1 items_per_second=64.369G/s
    BM_blis_gemm<float>/1024    15989415 ns     15988800 ns           44 Threads=1 col_major=1 items_per_second=67.1559G/s
    BM_blis_gemm<float>/2048   126683945 ns    126679210 ns            6 Threads=1 col_major=1 items_per_second=67.8086G/s
    BM_blis_gemm<float>/4096  1006389685 ns   1006335446 ns            1 Threads=1 col_major=1 items_per_second=68.2868G/s
    BM_blis_gemm<float>/8192  8001990732 ns   8001655669 ns            1 Threads=1 col_major=1 items_per_second=68.7053G/s
      Benchmark                        Time             CPU   Iterations UserCounters...
    BM_blis_gemm<double>/32        12386 ns        12385 ns        56228 Threads=8 col_major=1 items_per_second=2.64574G/s
    BM_blis_gemm<double>/64        14851 ns        14851 ns        57510 Threads=8 col_major=1 items_per_second=17.6522G/s
    BM_blis_gemm<double>/128       25368 ns        25367 ns        31970 Threads=8 col_major=1 items_per_second=82.6714G/s
    BM_blis_gemm<double>/256      137292 ns       136813 ns         5162 Threads=8 col_major=1 items_per_second=122.629G/s
    BM_blis_gemm<double>/512      730084 ns       729411 ns          957 Threads=8 col_major=1 items_per_second=184.008G/s
    BM_blis_gemm<double>/1024    4760274 ns      4759702 ns          144 Threads=8 col_major=1 items_per_second=225.59G/s
    BM_blis_gemm<double>/2048   36163451 ns     36161616 ns           19 Threads=8 col_major=1 items_per_second=237.543G/s
    BM_blis_gemm<double>/4096  294513791 ns    294499274 ns            3 Threads=8 col_major=1 items_per_second=233.343G/s
    BM_blis_gemm<double>/8192 2216791216 ns   2216686952 ns            1 Threads=8 col_major=1 items_per_second=248.008G/s
    BM_blis_gemm<float>/32         10619 ns        10619 ns        66284 Threads=8 col_major=1 items_per_second=3.08585G/s
    BM_blis_gemm<float>/64         11482 ns        11481 ns        60182 Threads=8 col_major=1 items_per_second=22.8327G/s
    BM_blis_gemm<float>/128        18595 ns        18594 ns        38437 Threads=8 col_major=1 items_per_second=112.785G/s
    BM_blis_gemm<float>/256        41770 ns        41769 ns        16431 Threads=8 col_major=1 items_per_second=401.668G/s
    BM_blis_gemm<float>/512       380200 ns       379768 ns         1850 Threads=8 col_major=1 items_per_second=353.421G/s
    BM_blis_gemm<float>/1024     2461177 ns      2460580 ns          276 Threads=8 col_major=1 items_per_second=436.378G/s
    BM_blis_gemm<float>/2048    17845051 ns     17843228 ns           40 Threads=8 col_major=1 items_per_second=481.411G/s
    BM_blis_gemm<float>/4096   137853471 ns    137845461 ns            5 Threads=8 col_major=1 items_per_second=498.525G/s
    BM_blis_gemm<float>/8192  1026327297 ns   1026266145 ns            1 Threads=8 col_major=1 items_per_second=535.685G/s
