/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/
// -- level-3 ---------------------------------------------------------
GEMM_UKR_PROT( float ,    s, gemm_skx_asm_32x12_l2 )
GEMM_UKR_PROT( float ,    s, gemm_skx_asm_12x32_l2 )

GEMM_UKR_PROT( double,    d, gemm_skx_asm_16x12_l2 )
GEMM_UKR_PROT( double,    d, gemm_skx_asm_16x14 )

// gemm_cv
GEMM_UKR_PROT( float,     s, gemm_cv_skx_int_48x8 )
GEMM_UKR_PROT( double,    d, gemm_cv_skx_int_24x8 )
// gemmsup_cv
GEMMSUP_KER_PROT( float,  s, gemmsup_cv_skx_int_48x8 )
GEMMSUP_KER_PROT( double, d, gemmsup_cv_skx_int_24x8 )
// gemmsup_rd
GEMMSUP_KER_PROT( float,  s, gemmsup_rd_skx_int_2x8 )
GEMMSUP_KER_PROT( double, d, gemmsup_rd_skx_int_2x8 )

// -- level-1f --------------------------------------------------------
AXPYF_KER_PROT( float,    s, axpyf_skx_int_128 )
AXPYF_KER_PROT( double,   d, axpyf_skx_int_64 )

DOTXF_KER_PROT( float,    s, dotxf_skx_int_8 )
DOTXF_KER_PROT( double,   d, dotxf_skx_int_8 )

// -- level-1 ---------------------------------------------------------
AMAXV_KER_PROT(  float,   s, amaxv_skx_int_16 )
AMAXV_KER_PROT( double,   d, amaxv_skx_int_8 )

AXPYV_KER_PROT(  float,   s, axpyv_skx_int_128 )
AXPYV_KER_PROT( double,   d, axpyv_skx_int_64 )

DOTXV_KER_PROT(  float,   s, dotxv_skx_int_128 )
DOTXV_KER_PROT( double,   d, dotxv_skx_int_64 )

DOTV_KER_PROT(  float,    s, dotv_skx_int_128 )
DOTV_KER_PROT( double,    d, dotv_skx_int_64 )

SCALV_KER_PROT(  float,   s, scalv_skx_int_128 )
SCALV_KER_PROT( double,   d, scalv_skx_int_64 )

SETV_KER_PROT(  float,    s, setv_skx_int_128 )
SETV_KER_PROT( double,    d, setv_skx_int_64 )

SWAPV_KER_PROT(  float,   s, swapv_skx_int_128 )
SWAPV_KER_PROT( double,   d, swapv_skx_int_64 )

COPYV_KER_PROT(  float,   s, copyv_skx_int_128 )
COPYV_KER_PROT( double,   d, copyv_skx_int_64 )
// -- end -------------------------------------------------------------
