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
   AS IS AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY
   OF TEXAS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
   OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"
#include <immintrin.h>
#include "bli_avx512_intrin_macros.h"

#define SKX_MR 3
#define SKX_NR 8

// --------------------------------------------------------------------
// Update C
// --------------------------------------------------------------------
#define UPDATE_C_ROW(midx, n_unroll, n_mask, co, ldc, s_c) \
{ \
	preduceadd_8x8d(zmm[c_regs[0*3+midx]], \
			        zmm[c_regs[1*3+midx]], \
			        zmm[c_regs[2*3+midx]], \
			        zmm[c_regs[3*3+midx]], \
			        zmm[c_regs[4*3+midx]], \
			        zmm[c_regs[5*3+midx]], \
			        zmm[c_regs[6*3+midx]], \
			        zmm[c_regs[7*3+midx]], \
			        zmm[c_store_regs[midx]]); \
	zmm[c_regs[0*3+midx]] = pzero_8d;  \
	zmm[c_regs[1*3+midx]] = pzero_8d;  \
	zmm[c_regs[2*3+midx]] = pzero_8d;  \
	zmm[c_regs[3*3+midx]] = pzero_8d;  \
	zmm[c_regs[4*3+midx]] = pzero_8d;  \
	zmm[c_regs[5*3+midx]] = pzero_8d;  \
	zmm[c_regs[6*3+midx]] = pzero_8d;  \
	zmm[c_regs[7*3+midx]] = pzero_8d;  \
	if (!is_alpha1) { \
		zmm[c_store_regs[midx]] = pmul_8d(zmm[c_store_regs[midx]], zmm[alpha_load_reg]); \
	} \
	if (!is_beta0) { \
		pmzloadu_8d(co, zmm[c_load_regs[midx]], n_mask); \
		zmm[c_store_regs[midx]] = pmfmadd_8d( zmm[beta_load_reg], \
			                                  zmm[c_load_regs[midx]], \
			                                  zmm[c_store_regs[midx]], \
			                                  n_mask); \
	} \
	if (1==s_c) { \
		pmstoreu_8d(co, zmm[c_store_regs[midx]], n_mask); \
	} \
	else { \
		pmstores_8d(co, zmm[c_store_regs[midx]], s_c, n_mask); \
	} \
	co += ldc*BLIS_SIZEOF_D; \
}

#define UPDATE_C(m_unroll, n_unroll, n_mask, co) \
{ \
	zmm[alpha_load_reg] = pload1_8d(alpha); \
	zmm[ beta_load_reg] = pload1_8d(beta); \
	m_packloop(SKX_MR, m_unroll, UPDATE_C_ROW, n_unroll, n_mask, co, rs_c, cs_c); \
	co -= m_unroll*rs_c*BLIS_SIZEOF_D; \
}

// --------------------------------------------------------------------
// Accumulate C
// --------------------------------------------------------------------
#define ACCUMULATE_C(midx, nidx) \
{ \
	zmm[c_regs[nidx*3+midx]] = pfmadd_8d(   zmm[a_regs[midx]], \
			                                zmm[b_regs[0]], \
			                                zmm[c_regs[nidx*3+midx]]); \
}
#define CD_KER_UN(nidx, m_unroll, k_mask, bo) \
{ \
	pmzloadu_8d(bo+nidx*cs_b*BLIS_SIZEOF_D, zmm[b_regs[0]], k_mask); \
	m_packloop(SKX_MR, m_unroll, ACCUMULATE_C, nidx); \
}

#define VLOAD_A(midx, mask, ao) \
{ \
	pmzloadu_8d(ao+ midx*rs_a*BLIS_SIZEOF_D, zmm[a_regs[midx]], mask); \
}
#define CD_KER_PK(k_unroll, m_unroll, n_unroll, ao, bo) \
{ \
	__mmask8 k_mask = edge_mask8(k_unroll); \
	m_packloop(SKX_MR, m_unroll, VLOAD_A, k_mask, ao); \
	n_packloop(SKX_NR, n_unroll, CD_KER_UN, m_unroll, k_mask, bo); \
	ao += k_unroll*cs_a*BLIS_SIZEOF_D; \
	bo += k_unroll*rs_b*BLIS_SIZEOF_D; \
}

#define KLOOP(m_unroll, n_unroll, ao, bo, co) \
{ \
	k_alignloop(k, CD_KER_PK, m_unroll, n_unroll, ao, bo); \
}

#define EDGE_B(n_unroll, m_unroll, ao, bo, co) \
{ \
	__mmask8 n_mask = edge_mask8(n_unroll); \
	KLOOP(m_unroll, n_unroll, ao, bo, co); \
	UPDATE_C(m_unroll, n_unroll, n_mask, co); \
	ao -= k*cs_a*BLIS_SIZEOF_D; \
	bo -= k*rs_b*BLIS_SIZEOF_D; \
	bo += n_unroll*cs_b*BLIS_SIZEOF_D; \
	co += n_unroll*cs_c*BLIS_SIZEOF_D; \
}

#define EDGE_A(m_unroll, ao, bo, co) \
{ \
	n_powedges(SKX_NR, n, EDGE_B, m_unroll, ao, bo, co); \
	ao += m_unroll*rs_a*BLIS_SIZEOF_D; \
	bo  = b; \
	co += m_unroll*rs_c*BLIS_SIZEOF_D - n*cs_c*BLIS_SIZEOF_D; \
}

#define ZERO_ZMMS(midx, from) \
{ \
	zmm[from+midx] = pzero_8d; \
}

// --------------------------------------------------------------------
// CRC:
//     --------        --------       | | | | | | | |
//     --------        --------       | | | | | | | |
//     --------   +=   --------  ...  | | | | | | | |
//     --------        --------       | | | | | | | |
//     --------        --------              :
//     --------        --------              :
//
// Assumptions:
// - A is row-stored;
// - B is column-stored;
// Therefore, this (r)ow-preferential microkernel is well-suited
// for a dot-product-based accumulation that performs vector loads
// from both A and B.
// --------------------------------------------------------------------

// --------------------------------------------------------------------
// Compute double SUPMM row-dot (rd) kernel with max unroll support of:
//     max_a_unroll: 3, 2, 1
//     max_b_unroll: 8, 4, 2, 1
// --------------------------------------------------------------------
void bli_dgemmsup_rd_skx_intrin_24x8
	 (
			conj_t      conja,
			conj_t      conjb,
			 dim_t      m0,
			 dim_t      n0,
			 dim_t      k0,
	   const void*      alpha,
	   const void*      a, inc_t rs_a0, inc_t cs_a0,
	   const void*      b, inc_t rs_b0, inc_t cs_b0,
	   const void*      beta,
			 void*      c, inc_t rs_c0, inc_t cs_c0,
		auxinfo_t*      data,
	 const cntx_t*      cntx
	 )
{
	(void)data;
	(void)cntx;

	const int64_t m    = m0;
	const int64_t n    = n0;
	const int64_t k    = k0;
	const int64_t rs_c = rs_c0;
	const int64_t cs_c = cs_c0;
	const int64_t rs_a = rs_a0;
	const int64_t cs_a = cs_a0;
	const int64_t rs_b = rs_b0;
	const int64_t cs_b = cs_b0;

	const bool is_alpha1     = *((double *)alpha)==1? true: false;
	const bool is_beta0      = *((double *)beta )==0? true: false;
	const int a_regs[]       = { 0,  1,  2};
	const int b_regs[]       = { 3,  4};
	const int c_regs[]       = { 8, 16, 24,  9, 17, 25, 10, 18, 26, 11, 19, 27, 
			                    12, 20, 28, 13, 21, 29, 14, 22, 30, 15, 23, 31};
	const int c_load_regs[]  = { 0,  1,  2};
	const int c_store_regs[] = { 5,  6,  7};
	const int alpha_load_reg = 3;
	const int beta_load_reg  = 4;

	Packet8d zmm[32];
	m_packloop(24, 24, ZERO_ZMMS, 8);
	
	const void *ao = a;
	const void *bo = b;
		  void *co = c;
	if (bli_info_get_enable_diagnosis())
	{
		printf("KER m    n    k alpha beta cs_c rs_c cs_a rs_a cs_b rs_b\n");
		printf("%5d%5d%5d%5.1f%5.1f%5d%5d%5d%5d%5d%5d\n", 
			    (int)m, (int)n, (int)k, *((double *)alpha), *((double *)beta), 
			    (int)cs_c, (int)rs_c, (int)cs_a, (int)rs_a, (int)cs_b, (int)rs_b);
	}
	m_powedges(SKX_MR, m, EDGE_A, ao, bo, co);
}
