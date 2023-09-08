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

#define SKX_MR 24
#define SKX_NR 8
#define A_L1_PREFETCH_DIST 4           // in units of k iterations
#define B_L1_PREFETCH_DIST 4

// --------------------------------------------------------------------
// Update C
// --------------------------------------------------------------------
#define ALPHA_C_COL(mvidx, nidx) \
{ \
	zmm[c_regs[nidx*3+mvidx]] = \
		pmul_8d(zmm[c_regs[nidx*3+mvidx]], zmm[alpha_load_reg]); \
}
#define ALPHA_C_ROW(nvidx, midx) \
{ \
	zmm[c_regs[(midx%8)*3+midx/8+nvidx*24]] = \
		pmul_8d(zmm[c_regs[(midx%8)*3+midx/8+nvidx*24]], zmm[alpha_load_reg]); \
}

#define BETA_C_COL(mvidx, nidx, co, mask) \
{ \
	pmzloadu_8d(co+ mvidx*8*BLIS_SIZEOF_D, zmm[c_load_regs[mvidx]], mask); \
	zmm[c_regs[nidx*3+mvidx]] = pmfmadd_8d( zmm[beta_load_reg], \
			                                zmm[c_load_regs[mvidx]], \
			                                zmm[c_regs[nidx*3+mvidx]], \
			                                mask ); \
}
#define BETA_C_SCOL(mvidx, nidx, co, s_c, mask) \
{ \
	pmzloads_8d(co+ mvidx*8*s_c*BLIS_SIZEOF_D, zmm[c_load_regs[mvidx]], s_c, mask); \
	zmm[c_regs[nidx*3+mvidx]] = pmfmadd_8d( zmm[beta_load_reg], \
			                                zmm[c_load_regs[mvidx]], \
			                                zmm[c_regs[nidx*3+mvidx]], \
		                                    mask ); \
}
#define BETA_C_ROW(nvidx, midx, co, mask) \
{ \
	pmzloadu_8d(co+ nvidx*8*BLIS_SIZEOF_D, zmm[c_load_regs[nvidx]], mask); \
	zmm[c_regs[(midx%8)*3+midx/8+nvidx*24]] = \
		pmfmadd_8d( zmm[beta_load_reg], \
		            zmm[c_load_regs[nvidx]], \
		            zmm[c_regs[(midx%8)*3+midx/8+nvidx*24]], \
		            mask); \
}

#define STORE_C_COL(mvidx, nidx, co, mask) \
{ \
	pmstoreu_8d(co+ mvidx*8*BLIS_SIZEOF_D, zmm[c_regs[nidx*3+mvidx]], mask); \
	zmm[c_regs[nidx*3+mvidx]] = pzero_8d; \
}
#define STORE_C_SCOL(mvidx, nidx, co, s_c, mask) \
{ \
	pmstores_8d(co+ mvidx*8*s_c*BLIS_SIZEOF_D, zmm[c_regs[nidx*3+mvidx]], s_c, mask ); \
	zmm[c_regs[nidx*3+mvidx]] = pzero_8d; \
}
#define STORE_C_ROW(nvidx, midx, co, mask) \
{ \
	pmstoreu_8d(co, zmm[c_regs[(midx%8)*3+midx/8+nvidx*24]], mask); \
	zmm[c_regs[(midx%8)*3+midx/8+nvidx*24]] = pzero_8d; \
}

#define UPDATE_C_COL(nidx, vecs, mask, co, ldc, s_c) \
{ \
	ALPHA_C_COL ( 0, nidx ); \
	ALPHA_C_COL ( 1, nidx ); \
	ALPHA_C_COL ( 2, nidx ); \
	if (1==s_c) { \
		if (!is_beta0) { \
			if ( 1 <= vecs ) BETA_C_COL ( 0, nidx, co, mask ); \
			if ( 2 <= vecs ) BETA_C_COL ( 1, nidx, co, mask ); \
			if ( 3 <= vecs ) BETA_C_COL ( 2, nidx, co, mask ); \
		} \
		if ( 1 <= vecs ) STORE_C_COL ( 0, nidx, co, mask ); \
		if ( 2 <= vecs ) STORE_C_COL ( 1, nidx, co, mask ); \
		if ( 3 <= vecs ) STORE_C_COL ( 2, nidx, co, mask ); \
	} \
	else { \
		if (!is_beta0) { \
			if ( 1 <= vecs ) BETA_C_SCOL ( 0, nidx, co, s_c, mask ); \
			if ( 2 <= vecs ) BETA_C_SCOL ( 1, nidx, co, s_c, mask ); \
			if ( 3 <= vecs ) BETA_C_SCOL ( 2, nidx, co, s_c, mask ); \
		} \
		if ( 1 <= vecs ) STORE_C_SCOL ( 0, nidx, co, s_c, mask ); \
		if ( 2 <= vecs ) STORE_C_SCOL ( 1, nidx, co, s_c, mask ); \
		if ( 3 <= vecs ) STORE_C_SCOL ( 2, nidx, co, s_c, mask ); \
	} \
	co += ldc*BLIS_SIZEOF_D; \
}
#define UPDATE_C_ROW(midx, mask, co, ldc, s_c) \
{ \
	ALPHA_C_ROW( 0, midx ); \
	if (!is_beta0) { \
		BETA_C_ROW ( 0, midx, co, mask ); \
	} \
	STORE_C_ROW ( 0, midx, co, mask ); \
	co += ldc*BLIS_SIZEOF_D; \
}

#define TRANSPOSE_C_BLK(mvidx, m_mask, n_mask) \
{ \
	ptranspose_8x8d(zmm[c_regs[ 0+mvidx]], zmm[c_regs[ 3+mvidx]], \
		            zmm[c_regs[ 6+mvidx]], zmm[c_regs[ 9+mvidx]], \
		            zmm[c_regs[12+mvidx]], zmm[c_regs[15+mvidx]], \
		            zmm[c_regs[18+mvidx]], zmm[c_regs[21+mvidx]]); \
}
#define UPDATE_C(m_unroll, m_vecs, m_mask, n_unroll, n_mask, co) \
{ \
	zmm[alpha_load_reg] = pload1_8d(alpha); \
	zmm[ beta_load_reg] = pload1_8d(beta); \
	if (1==cs_c) {\
		if ( 1 <= m_vecs ) TRANSPOSE_C_BLK ( 0, m_mask, n_mask ); \
		if ( 2 <= m_vecs ) TRANSPOSE_C_BLK ( 1, m_mask, n_mask ); \
		if ( 3 <= m_vecs ) TRANSPOSE_C_BLK ( 2, m_mask, n_mask ); \
		m_packloop( SKX_MR, m_unroll, UPDATE_C_ROW, n_mask, co, rs_c, cs_c ); \
		co -= m_unroll*rs_c*BLIS_SIZEOF_D; \
	} \
	else {\
		n_packloop( SKX_NR, n_unroll, UPDATE_C_COL, m_vecs, m_mask, co, cs_c, rs_c ); \
		co -= n_unroll*cs_c*BLIS_SIZEOF_D; \
	} \
}

// --------------------------------------------------------------------
// Accumulate C
// --------------------------------------------------------------------
#define ACCUMULATE_C( mvidx, nidx, kidx, k_align, ao ) \
{ \
	zmm[c_regs[nidx*3+mvidx]] = pfmadd_8d( zmm[a_regs[mvidx*k_align+kidx]], \
			                               zmm[b_regs[kidx]], \
			                               zmm[c_regs[nidx*3+mvidx]]); \
}
#define CV_KER_UN(nidx, kidx, m_vecs, k_align, ao, bo ) \
{ \
	zmm[b_regs[kidx]] = pload1_8d( bo+(kidx*rs_b+nidx*cs_b)*BLIS_SIZEOF_D ); \
	if ( 1 <= m_vecs ) ACCUMULATE_C ( 0, nidx, kidx, k_align, ao ); \
	if ( 2 <= m_vecs ) ACCUMULATE_C ( 1, nidx, kidx, k_align, ao ); \
	if ( 3 <= m_vecs ) ACCUMULATE_C ( 2, nidx, kidx, k_align, ao ); \
}

#define VLOAD_A(mvidx, kidx, mask, k_align, ao, is_prefetch ) \
{ \
	if ( 1==rs_a ) { \
		pmzloadu_8d(ao+(kidx*cs_a+mvidx*8)*BLIS_SIZEOF_D, zmm[a_regs[mvidx*k_align+kidx]], mask); \
		if ( is_prefetch && 1==rs_a ) \
			prefetch_8d_at( A_L1_PREFETCH_DIST+kidx, ao+mvidx*8*BLIS_SIZEOF_D, cs_a, _MM_HINT_T0 ); \
	} \
	else { \
		pmzloads_8d(ao+(kidx*cs_a+mvidx*8*rs_a)*BLIS_SIZEOF_D, zmm[a_regs[mvidx*k_align+kidx]], rs_a, mask); \
	} \
}
#define CV_KER_UK( kidx, m_unroll, m_vecs, m_mask, n_unroll, n_mask, k_align, ao, bo, is_prefetch ) \
{ \
	if ( 1 <= m_vecs ) VLOAD_A ( 0, kidx, m_mask, k_align, ao, is_prefetch ); \
	if ( 2 <= m_vecs ) VLOAD_A ( 1, kidx, m_mask, k_align, ao, is_prefetch ); \
	if ( 3 <= m_vecs ) VLOAD_A ( 2, kidx, m_mask, k_align, ao, is_prefetch ); \
	n_packloop( SKX_NR, n_unroll, CV_KER_UN, kidx, m_vecs, k_align, ao, bo ); \
	if( is_prefetch && 1==cs_b ) \
		prefetch_16f_at( B_L1_PREFETCH_DIST+kidx, bo, rs_b, _MM_HINT_T0 ); \
}

#define CV_KER_PK( k_unroll, m_unroll, m_vecs, m_mask, n_unroll, n_mask, k_align, ao, bo, is_prefetch ) \
{ \
	if ( 1 <= k_unroll ) CV_KER_UK( 0, m_unroll, m_vecs, m_mask, n_unroll, n_mask, k_align, ao, bo, is_prefetch ); \
	if ( 2 <= k_unroll ) CV_KER_UK( 1, m_unroll, m_vecs, m_mask, n_unroll, n_mask, k_align, ao, bo, is_prefetch ); \
	ao += k_unroll*cs_a*BLIS_SIZEOF_D; \
	bo += k_unroll*rs_b*BLIS_SIZEOF_D; \
}

#define EDGE_B(n_unroll, m_unroll, m_vecs, m_mask, k_align, ao, bo, co, is_prefetch) \
{ \
	__mmask8 n_mask = edge_mask8(n_unroll); \
	k_alignedges( k_align, k, CV_KER_PK, m_unroll, m_vecs, m_mask, n_unroll, n_mask, k_align, ao, bo, is_prefetch ); \
	UPDATE_C( m_unroll, m_vecs, m_mask, n_unroll, n_mask, co ); \
	ao -= k*cs_a*BLIS_SIZEOF_D; \
	bo -= k*rs_b*BLIS_SIZEOF_D; \
	bo += n_unroll*cs_b*BLIS_SIZEOF_D; \
	co += n_unroll*cs_c*BLIS_SIZEOF_D; \
}

#define EDGE_A(m_unroll, k_align, ao, bo, co, is_prefetch) \
{ \
	int m_vecs = div_up(m_unroll, 8); \
	__mmask8 m_mask = edge_mask8(m_unroll); \
	if ( is_a_packed && 1==rs_c ) { \
		if ( 1 <= m_vecs ) prefetch_x8d_at ( 0, 8, co, cs_c, 8, _MM_HINT_T0 ); \
		if ( 2 <= m_vecs ) prefetch_x8d_at ( 1, 8, co, cs_c, 8, _MM_HINT_T0 ); \
		if ( 3 <= m_vecs ) prefetch_x8d_at ( 2, 8, co, cs_c, 8, _MM_HINT_T0 ); \
	} \
	if ( is_prefetch && 1==cs_c ) \
		m_packloop( SKX_MR, m_unroll, prefetch_8d_at, co, rs_c, _MM_HINT_T0 ); \
	n_powedges(SKX_NR, n, EDGE_B, m_unroll, m_vecs, m_mask, k_align, ao, bo, co, is_prefetch); \
	ao += m_unroll*rs_a*BLIS_SIZEOF_D; \
	bo -= n*cs_b*BLIS_SIZEOF_D; \
	co += m_unroll*rs_c*BLIS_SIZEOF_D - n*cs_c*BLIS_SIZEOF_D; \
}

#define ZERO_ZMMS(midx, from) \
{ \
	zmm[from+midx] = pzero_8d; \
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
// Compute double GEMM col-vector (cv) kernel with max unroll support of:
//     max_a_unroll: 24, 16, 8, 4, 2, 1
//     max_b_unroll: 8, 4, 2, 1
// --------------------------------------------------------------------
void bli_dgemm_cv_skx_int_24x8
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

	const bool is_beta0      = *((double *)beta )==0? true: false;
	const bool is_a_packed   = true;
	const bool is_b_packed   = true;
	const int a_regs[]       = { 0,  1,  2,  3,  4,  5 };
	const int b_regs[]       = { 6,  7 };
	const int c_regs[]       = { 8, 16, 24,  9, 17, 25, 10, 18, 26, 11, 19, 27, 
		                        12, 20, 28, 13, 21, 29, 14, 22, 30, 15, 23, 31};
	const int c_load_regs[]  = { 0,  1,  2 };
	const int alpha_load_reg = 3;
	const int beta_load_reg  = 4;

	Packet8d zmm[32];
	m_packloop(24, 24, ZERO_ZMMS, 8);

	const void *ao = a;
	const void *bo = b;
	void *co = c;
	if (bli_info_get_enable_diagnosis())
	{
		printf("PCV m    n    k alpha beta cs_c rs_c cs_a rs_a cs_b rs_b\n");
		printf("%s%s%3d%5d%5d%5.1f%5.1f%5d%5d%5d%5d%5d%5d\n", 
				is_a_packed ?" ":"A", is_b_packed?" ":"B",
				(int)m, (int)n, (int)k, *((double *)alpha), *((double *)beta), 
				(int)cs_c, (int)rs_c, (int)cs_a, (int)rs_a, (int)cs_b, (int)rs_b);
	}
	m_powedges(SKX_MR, m, EDGE_A, 1, ao, bo, co, true);
}

// --------------------------------------------------------------------
// Compute double SUPMM col-vector (cv) kernel with max unroll support of:
//     max_a_unroll: 24, 16, 8, 4, 2, 1
//     max_b_unroll: 8, 4, 2, 1
// --------------------------------------------------------------------
void bli_dgemmsup_cv_skx_int_24x8
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

	const bool is_beta0      = *((double *)beta )==0 ? true : false;
	const bool is_a_packed   = (cs_a==SKX_MR && rs_a==1) ? true : false;
	const bool is_b_packed   = (rs_b==SKX_NR && cs_b==1) ? true : false;
	const int a_regs[]       = { 0,  1,  2,  3,  4,  5 };
	const int b_regs[]       = { 6,  7};
	const int c_regs[]       = { 8, 16, 24,  9, 17, 25, 10, 18, 26, 11, 19, 27, 
		                        12, 20, 28, 13, 21, 29, 14, 22, 30, 15, 23, 31};
	const int c_load_regs[]  = { 0,  1,  2};
	const int alpha_load_reg = 3;
	const int beta_load_reg  = 4;

	Packet8d zmm[32];
	m_packloop(24, 24, ZERO_ZMMS, 8);

	const void *ao = a;
	const void *bo = b;
		  void *co = c;
	if(bli_info_get_enable_diagnosis())
	{
		printf("UCV m    n    k alpha beta cs_c rs_c cs_a rs_a cs_b rs_b\n");
		printf("%s%s%3d%5d%5d%5.1f%5.1f%5d%5d%5d%5d%5d%5d\n", 
				is_a_packed ?" ":"A", is_b_packed?" ":"B",
				(int)m, (int)n, (int)k, *((double *)alpha), *((double *)beta), 
				(int)cs_c, (int)rs_c, (int)cs_a, (int)rs_a, (int)cs_b, (int)rs_b);
	}
	if ( is_a_packed )
		m_powedges(SKX_MR, m, EDGE_A, 1, ao, bo, co, true);
	else
		m_powedges(SKX_MR, m, EDGE_A, 1, ao, bo, co, false);

}

