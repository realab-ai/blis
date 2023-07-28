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

#define SKX_MR 48
#define SKX_NR 8
#define A_L1_PREFETCH_DIST 4           // in units of k iterations
#define B_L1_PREFETCH_DIST 4

// --------------------------------------------------------------------
// Update C
// --------------------------------------------------------------------
#define ALPHA_C_COL(mvidx, nidx) \
{ \
	zmm[c_regs[nidx*3+mvidx]] = \
	pmul_16f(zmm[c_regs[nidx*3+mvidx]], zmm[alpha_load_reg]); \
}
#define ALPHA_C_ROW(nvidx, idx_r, stack) \
{ \
	zmm[c_regs[idx_r*3+stack+nvidx*24]] = \
	pmul_16f(zmm[c_regs[idx_r*3+stack+nvidx*24]], zmm[alpha_load_reg]); \
}
#define BETA_C_COL(mvidx, nidx, co, mask) \
{ \
	pmzloadu_16f(co+ mvidx*16*BLIS_SIZEOF_S, zmm[c_load_regs[mvidx]], mask); \
	zmm[c_regs[nidx*3+mvidx]] = \
		pmfmadd_16f(zmm[beta_load_reg], \
			        zmm[c_load_regs[mvidx]], \
			        zmm[c_regs[nidx*3+mvidx]], \
			        mask); \
}
#define BETA_C_SCOL(mvidx, nidx, co, s_c, mask) \
{ \
	pmzloads_16f(co+ mvidx*16*s_c*BLIS_SIZEOF_S, zmm[c_load_regs[mvidx]], s_c, mask); \
	zmm[c_regs[nidx*3+mvidx]] = \
		pmfmadd_16f(zmm[beta_load_reg], \
			        zmm[c_load_regs[mvidx]], \
			        zmm[c_regs[nidx*3+mvidx]], \
			        mask); \
}
#define BETA_C_ROW(nvidx, idx_r, stack, co, mask) \
{ \
	pmzloadu_16f(co, zmm[c_load_regs[nvidx]], mask); \
	zmm[c_regs[idx_r*3+stack+nvidx*24]] = \
		pfmadd_16f( zmm[beta_load_reg], \
			        zmm[c_load_regs[nvidx]], \
			        zmm[c_regs[idx_r*3+stack+nvidx*24]]); \
}
#define STORE_C_COL(mvidx, nidx, co, mask) \
{ \
	pmstoreu_16f(co+ mvidx*16*BLIS_SIZEOF_S, zmm[c_regs[nidx*3+mvidx]], mask); \
	zmm[c_regs[nidx*3+mvidx]] = pzero_16f; \
}
#define STORE_C_SCOL(mvidx, nidx, co, s_c, mask) \
{ \
	pmstores_16f(co+ mvidx*16*s_c*BLIS_SIZEOF_S, zmm[c_regs[nidx*3+mvidx]], s_c, mask ); \
	zmm[c_regs[nidx*3+mvidx]] = pzero_16f; \
}
#define STORE_CL_ROW(nvidx, idx_r, stack, co, mask) \
{ \
	pmstoreu_16f(co, zmm[c_regs[idx_r*3+stack+nvidx*24]], mask); \
	zmm[c_regs[idx_r*3+stack+nvidx*24]] = \
		_mm512_shuffle_f32x4(   zmm[c_regs[idx_r*3+stack+nvidx*24]], \
			                    zmm[c_regs[idx_r*3+stack+nvidx*24]], \
			                    0x1E); \
}
#define STORE_CH_ROW(nvidx, idx_r, stack, co, mask) \
{ \
	pmstoreu_16f(co, zmm[c_regs[idx_r*3+stack+nvidx*24]], mask); \
	zmm[c_regs[idx_r*3+stack+nvidx*24]] = pzero_16f; \
}
#define UPDATE_C_COL(nidx, vecs, mask, co, ldc, s_c) \
{ \
	if (!is_alpha1) { \
		m_packloop(3, vecs, ALPHA_C_COL, nidx); \
	} \
	if (1==s_c) { \
		if (!is_beta0) { \
			m_packloop(3, vecs, BETA_C_COL, nidx, co, mask); \
		} \
		m_packloop(3, vecs, STORE_C_COL, nidx, co, mask ); \
	} \
	else {\
		if (!is_beta0) { \
			m_packloop(3, vecs, BETA_C_SCOL, nidx, co, s_c, mask); \
		} \
		m_packloop(3, vecs, STORE_C_SCOL, nidx, co, s_c, mask ); \
	} \
	co += ldc*BLIS_SIZEOF_S; \
}
#define UPDATE_C_ROW(midx, vecs, mask, co, ldc, s_c) \
{ \
	int idx_r = (midx%16)/2; \
	int stack = midx/16; \
	if (!is_alpha1 && (midx)%2==0) { \
		n_packloop(1, vecs, ALPHA_C_ROW, idx_r, stack); \
	} \
	if (!is_beta0) { \
		n_packloop(1, vecs, BETA_C_ROW, idx_r, stack, co, mask); \
	} \
	if ((midx)%2 == 0) {\
		n_packloop(1, vecs, STORE_CL_ROW, idx_r, stack, co, mask); \
	} \
	else { \
		n_packloop(1, vecs, STORE_CH_ROW, idx_r, stack, co, mask); \
	} \
	co += ldc*BLIS_SIZEOF_S; \
}

#define TRANSPOSE_C_BLK(mvidx, m_mask, n_mask) \
{ \
	ptranspose_8x16f(   zmm[c_regs[ 0+mvidx ]], zmm[c_regs[ 3+mvidx ]], \
			            zmm[c_regs[ 6+mvidx ]], zmm[c_regs[ 9+mvidx ]], \
			            zmm[c_regs[12+mvidx]], zmm[c_regs[15+mvidx]], \
			            zmm[c_regs[18+mvidx]], zmm[c_regs[21+mvidx]]); \
}
#define UPDATE_C(m_unroll, m_vecs, m_mask, n_unroll, n_vecs, n_mask, co) \
{ \
	zmm[alpha_load_reg] = pload1_16f(alpha); \
	zmm[ beta_load_reg] = pload1_16f(beta); \
	if (1==cs_c) {\
		m_packloop(3, m_vecs, TRANSPOSE_C_BLK, m_mask, n_mask); \
		m_packloop(48, m_unroll, UPDATE_C_ROW, n_vecs, n_mask, co, rs_c, cs_c); \
		co -= m_unroll*rs_c*BLIS_SIZEOF_S; \
	} \
	else { \
		n_packloop(48, n_unroll, UPDATE_C_COL, m_vecs, m_mask, co, cs_c, rs_c); \
		co -= n_unroll*cs_c*BLIS_SIZEOF_S; \
	} \
}

// --------------------------------------------------------------------
// Accumulate C
// --------------------------------------------------------------------
#define ACCUMULATE_C(mvidx, nidx) \
{ \
	zmm[c_regs[nidx*3+mvidx]] = \
		pfmadd_16f( zmm[a_regs[mvidx]], \
			        zmm[b_regs[0]], \
			        zmm[c_regs[nidx*3+mvidx]]); \
}
#define CV_KER_UN(nidx, m_vecs) \
{ \
	zmm[b_regs[0]] = pload1_16f(bo+nidx*BLIS_SIZEOF_S*cs_b); \
	m_packloop(3, m_vecs, ACCUMULATE_C, nidx); \
}

#define VLOAD_A(mvidx, mask, ao) \
{ \
	pmzloadu_16f(ao+ mvidx*16*BLIS_SIZEOF_S, zmm[a_regs[mvidx]], mask); \
	if(is_prefetch && 1==rs_a) \
		prefetch_16f_at(A_L1_PREFETCH_DIST, ao+mvidx*16*BLIS_SIZEOF_S, cs_a); \
}
#define CV_KER_UK(m_unroll, m_vecs, m_mask, n_unroll, n_vecs, n_mask, ao, bo) \
{ \
	m_packloop(3, div_up(m_unroll, 16), VLOAD_A, m_mask, ao); \
	n_packloop(8, n_unroll, CV_KER_UN, m_vecs); \
	if(is_prefetch && 1==cs_b) \
		prefetch_16f_at(B_L1_PREFETCH_DIST, bo, rs_b); \
	ao += cs_a*BLIS_SIZEOF_S; \
	bo += rs_b*BLIS_SIZEOF_S; \
}

#define KLOOP(m_unroll, m_vecs, m_mask, n_unroll, n_vecs, n_mask, ao, bo, co) \
{ \
	for (int p=0; p<k; p++) { \
		CV_KER_UK(m_unroll, m_vecs, m_mask, n_unroll, n_vecs, n_mask, ao, bo); \
	} \
}

#define EDGE_B(n_unroll, m_unroll, m_vecs, m_mask, ao, bo, co) \
{ \
	int n_vecs = div_up(n_unroll, 16); \
	__mmask16 n_mask = edge_mask16(n_unroll); \
	KLOOP(m_unroll, m_vecs, m_mask, n_unroll, n_vecs, n_mask, ao, bo, co); \
	UPDATE_C(m_unroll, m_vecs, m_mask, n_unroll, n_vecs, n_mask, co); \
	ao -= k*cs_a*BLIS_SIZEOF_S; \
	bo -= k*rs_b*BLIS_SIZEOF_S; \
	bo += n_unroll*cs_b*BLIS_SIZEOF_S; \
	co += n_unroll*cs_c*BLIS_SIZEOF_S; \
}

#define EDGE_A(m_unroll, ao, bo, co) \
{ \
	int m_vecs = div_up(m_unroll, 16); \
	__mmask16 m_mask = edge_mask16(m_unroll); \
	if (is_prefetch && 1==rs_c) \
		m_packloop(3, m_vecs, prefetch_x16f_at,  8, co, cs_c, 16); \
	if (is_prefetch && 1==cs_c) \
		m_packloop(SKX_MR, m_unroll, prefetch_16f_at, co, rs_c); \
	n_powedges(SKX_NR, n, EDGE_B, m_unroll, m_vecs, m_mask, ao, bo, co); \
	ao += m_unroll*rs_a*BLIS_SIZEOF_S; \
	bo  = b; \
	co += m_unroll*rs_c*BLIS_SIZEOF_S - n*cs_c*BLIS_SIZEOF_S; \
}

#define ZERO_ZMMS(midx, from) \
{ \
	zmm[from+midx] = pzero_16f; \
}

// --------------------------------------------------------------------
// CCR:
//     | | | | | | | |        | | | | | | | |       --------------
//     | | | | | | | |        | | | | | | | |       --------------
//     | | | | | | | |   +=   | | | | | | | |  ...  --------------
//     | | | | | | | |        | | | | | | | |       --------------
//     | | | | | | | |        | | | | | | | |              :
//     | | | | | | | |        | | | | | | | |              :
// CCC:
//     | | | | | | | |        | | | | | | | |       | | | | | | | |
//     | | | | | | | |        | | | | | | | |       | | | | | | | |
//     | | | | | | | |   +=   | | | | | | | |  ...  | | | | | | | |
//     | | | | | | | |        | | | | | | | |       | | | | | | | |
//     | | | | | | | |        | | | | | | | |              :
//     | | | | | | | |        | | | | | | | |              :
// Assumptions:
// - A is column-stored;
// - B is row- or column-stored;
// Therefore, this (c)olummn-preferential kernel is well-suited for contiguous
// (v)ector loads on A and single-element broadcasts from B.
// --------------------------------------------------------------------

// --------------------------------------------------------------------
// Compute float GEMM col-vector (cv) kernel with max unroll support of:
//     max_a_unroll: 48, 32, 16, 8, 4, 2, 1
//     max_b_unroll: 8, 4, 2, 1
// --------------------------------------------------------------------
void bli_sgemm_cv_skx_intrin_48x8
	 (
			 dim_t      m,
			 dim_t      n,
			 dim_t      k_,
	   const void*      alpha,
	   const void*      a,
	   const void*      b,
	   const void*      beta,
			 void*      c, inc_t rs_c_, inc_t cs_c_,
			 auxinfo_t* data,
	   const cntx_t*    cntx
	 )
{
	(void)data;
	(void)cntx;

	const int64_t k    = k_;
	const int64_t rs_c = rs_c_;
	const int64_t cs_c = cs_c_;
	const int64_t rs_a = 1;
	const int64_t cs_a = SKX_MR;
	const int64_t rs_b = SKX_NR;
	const int64_t cs_b = 1;

	const bool is_alpha1     = *((float *)alpha)==1? true: false;
	const bool is_beta0      = *((float *)beta )==0? true: false;
	const bool is_prefetch   = true;
	const int a_regs[]       = { 0,  1,  2};
	const int b_regs[]       = { 3,  4};
	const int c_regs[]       = { 8, 16, 24,  9, 17, 25, 10, 18, 26, 11, 19, 27, 
			                    12, 20, 28, 13, 21, 29, 14, 22, 30, 15, 23, 31};
	const int c_load_regs[]  = { 0,  1,  2};
	const int alpha_load_reg = 3;
	const int beta_load_reg  = 4;

	Packet16f zmm[32];
	m_packloop(24, 24, ZERO_ZMMS, 8);

	const void *ao = a;
	const void *bo = b;
	void *co = c;
	if (bli_info_get_enable_diagnosis())
	{
		printf("KER m    n    k alpha beta cs_c rs_c cs_a rs_a cs_b rs_b\n");
		printf("%5d%5d%5d%5.1f%5.1f%5d%5d%5d%5d%5d%5d\n",
			    (int)m, (int)n, (int)k, *((float *)alpha), *((float *)beta),
			    (int)cs_c, (int)rs_c, (int)cs_a, (int)rs_a, (int)cs_b, (int)rs_b);
	}
	m_powedges(SKX_MR, m, EDGE_A, ao, bo, co);
}

// --------------------------------------------------------------------
// Compute float SUPMM col-vector (cv) kernel with max unroll support of:
//     max_a_unroll: 48, 32, 16, 8, 4, 2, 1
//     max_b_unroll: 8, 4, 2, 1
// --------------------------------------------------------------------
void bli_sgemmsup_cv_skx_intrin_48x8
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

	const bool is_alpha1     = *((float *)alpha)==1? true: false;
	const bool is_beta0      = *((float *)beta )==0? true: false;
	const bool is_prefetch   = false;
	const int a_regs[]       = { 0,  1,  2};
	const int b_regs[]       = { 3,  4};
	const int c_regs[]       = { 8, 16, 24,  9, 17, 25, 10, 18, 26, 11, 19, 27, 
			                    12, 20, 28, 13, 21, 29, 14, 22, 30, 15, 23, 31};
	const int c_load_regs[]  = { 0,  1,  2};
	const int alpha_load_reg = 3;
	const int beta_load_reg  = 4;

	Packet16f zmm[32];
	m_packloop(24, 24, ZERO_ZMMS, 8);

	const void *ao = a;
	const void *bo = b;
		  void *co = c;
	if (bli_info_get_enable_diagnosis())
	{
		printf("KER m    n    k alpha beta cs_c rs_c cs_a rs_a cs_b rs_b\n");
		printf("%5d%5d%5d%5.1f%5.1f%5d%5d%5d%5d%5d%5d\n",
			    (int)m, (int)n, (int)k, *((float *)alpha), *((float *)beta),
			    (int)cs_c, (int)rs_c, (int)cs_a, (int)rs_a, (int)cs_b, (int)rs_b);
	}
	m_powedges(SKX_MR, m, EDGE_A, ao, bo, co);
}

