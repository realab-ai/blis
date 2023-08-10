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

#define SKX_MR 64
// --------------------------------------------------------------------
// Update C
// --------------------------------------------------------------------
#define UPDATE_C_COL( mvidx, m_mask, k_align, co ) \
{ \
	if ( 2 <= k_align ) zmm[c_regs[mvidx*k_align]] = padd_8d(zmm[c_regs[mvidx*k_align]], zmm[c_regs[mvidx*k_align+1]]); \
	pmzloadu_8d(co+mvidx*8*BLIS_SIZEOF_D, zmm[c_load_regs[mvidx*k_align]], m_mask); \
	zmm[c_regs[mvidx*k_align]] = pmfmadd_8d( zmm[alpha_load_reg], \
						                     zmm[c_regs[mvidx*k_align]], \
						                     zmm[c_load_regs[mvidx*k_align]], \
						                     m_mask); \
	pmstoreu_8d(co+mvidx*8*BLIS_SIZEOF_D, zmm[c_regs[mvidx*k_align]], m_mask); \
}
#define UPDATE_C_SCOL( mvidx, m_mask, s_c, k_align, co ) \
{ \
	if ( 2 <= k_align ) zmm[c_regs[mvidx*k_align]] = padd_8d(zmm[c_regs[mvidx*k_align]], zmm[c_regs[mvidx*k_align+1]]); \
	pmzloads_8d(co+mvidx*8*s_c*BLIS_SIZEOF_D, zmm[c_load_regs[mvidx*k_align]], s_c, m_mask); \
	zmm[c_regs[mvidx*k_align]] = pmfmadd_8d( zmm[alpha_load_reg], \
						                     zmm[c_regs[mvidx*k_align]], \
						                     zmm[c_load_regs[mvidx*k_align]], \
						                     m_mask); \
	pmstores_8d(co+mvidx*8*s_c*BLIS_SIZEOF_D, zmm[c_regs[mvidx*k_align]], s_c, m_mask); \
}

#define UPDATE_C( m_unroll, m_vecs, m_mask, k_align, co) \
{ \
	if ( 1 == rs_c ) { \
		m_packloop( (8/k_align), m_vecs, UPDATE_C_COL, m_mask, k_align, co ); \
	} \
	else { \
		m_packloop( (8/k_align), m_vecs, UPDATE_C_SCOL, m_mask, rs_c, k_align, co ); \
	} \
}

// --------------------------------------------------------------------
// Accumulate C
// --------------------------------------------------------------------
#define ACCUMULATE_C(mvidx, mask, kidx, k_align) \
{ \
	zmm[c_regs[mvidx*k_align+kidx]] = pfmadd_8d( zmm[a_regs[mvidx*k_align+kidx]], \
						                         zmm[b_regs[kidx]], \
						                         zmm[c_regs[mvidx*k_align+kidx]] ); \
}
#define VLOAD_A(mvidx, mask, kidx, k_align, ao) \
{ \
	if ( 1==rs_a  ) { \
		pmzloadu_8d(ao+(kidx*cs_a+mvidx*8)*BLIS_SIZEOF_D, zmm[a_regs[mvidx*k_align+kidx]], mask); \
	} \
	else { \
		pmzloads_8d(ao+(kidx*cs_a+mvidx*8*rs_a)*BLIS_SIZEOF_D, zmm[a_regs[mvidx*k_align+kidx]], rs_a, mask); \
	} \
}

#define AX_KER_UK( kidx, m_unroll, m_vecs, m_mask, k_align, ao, bo ) \
{ \
	m_packloop( (8/k_align), m_vecs, VLOAD_A, m_mask, kidx, k_align, ao ); \
	zmm[b_regs[kidx]] = pload1_8d( bo+kidx*rs_b*BLIS_SIZEOF_D ); \
	m_packloop( (8/k_align), m_vecs, ACCUMULATE_C, m_mask, kidx, k_align ); \
}

#define AX_KER_PK( k_unroll, m_unroll, m_vecs, m_mask, k_align, ao, bo, co) \
{ \
	if ( 1 <= k_unroll ) AX_KER_UK( 0, m_unroll, m_vecs, m_mask, k_align, ao, bo ); \
	if ( 2 <= k_unroll ) AX_KER_UK( 1, m_unroll, m_vecs, m_mask, k_align, ao, bo ); \
	ao += k_unroll*cs_a*BLIS_SIZEOF_D; \
	bo += k_unroll*rs_b*BLIS_SIZEOF_D; \
}

#define ZERO_ZMMS(midx, from) \
{ \
	zmm[from+midx] = pzero_8d; \
}

#define EDGE_A(m_unroll, k_align, ao, bo, co) \
{ \
	int m_vecs = div_up(m_unroll, 8); \
	__mmask8 m_mask = edge_mask8(m_unroll); \
	k_alignedges( k_align, k, AX_KER_PK, m_unroll, m_vecs, m_mask, k_align, ao, bo, co ); \
	UPDATE_C( m_unroll, m_vecs, m_mask, k_align, co ); \
	m_packloop( 8, 8, ZERO_ZMMS, 24 ); \
	ao -= k*cs_a*BLIS_SIZEOF_D; \
	bo -= k*rs_b*BLIS_SIZEOF_D; \
	ao += m_unroll*rs_a*BLIS_SIZEOF_D; \
	co += m_unroll*rs_c*BLIS_SIZEOF_D; \
}

// --------------------------------------------------------------------
// JCJ:
//     |         | | | |       |
//     |         | | | |       |
//     |    +=   | | | |  ...  |
//     |         | | | |       |
//     |         | | | |       :
//     |         | | | |       :
// Assumptions:
// - C is column-stored vector, A is column-stored, B is column-stored vector;
// Therefore, this (c)olummn-preferential kernel is well-suited for contiguous
// (v)ector loads on A and single-element broadcasts from B.
// --------------------------------------------------------------------

// --------------------------------------------------------------------
// Compute double axpyf kernel with max unroll support of:
//     max_m_unroll: 64, 32, 16, 8, 4, 2, 1
//     max_k_unroll:  2,  1
// --------------------------------------------------------------------
void bli_daxpyf_skx_int_64
	 (
		    conj_t      conja,
		    conj_t      conjx,
		     dim_t      m,
		     dim_t      b_n,
	   const void*      alpha0,
	   const void*      a0, inc_t inca, inc_t lda,
	   const void*      x0, inc_t incx,
	         void*      y0, inc_t incy,
	   const cntx_t*    cntx
	 )
{
	(void)cntx;

	const int64_t k    = b_n;
	const int64_t rs_a = inca;
	const int64_t cs_a = lda;
	const int64_t rs_b = incx;
	const int64_t cs_b = 0;
	const int64_t rs_c = incy;
	const int64_t cs_c = 0;

	const void* alpha        = alpha0;
	const int a_regs[]       = {  8,  9, 10, 11, 12, 13, 14, 15 };
	const int b_regs[]       = { 16, 17 };
	const int c_regs[]       = { 24, 25, 26, 27, 28, 29, 30, 31 };
	const int c_load_regs[]  = {  8,  9, 10, 11, 12, 13, 14, 15 };
	const int alpha_load_reg = 4;

	Packet8d zmm[32];
	m_packloop( 8, 8, ZERO_ZMMS, 24 );

	zmm[alpha_load_reg] = pload1_8d(alpha);

	const void *ao = a0;
	const void *bo = x0;
		  void *co = y0;
	if(bli_info_get_enable_diagnosis())
	{
		printf("KER m    n    k alpha beta cs_c rs_c cs_a rs_a cs_b rs_b\n");
		printf("%5d%5d%5d%5.1f%5.1f%5d%5d%5d%5d%5d%5d\n", 
				(int)m, 1, (int)k, *((double *)alpha), 1.0, 
				(int)cs_c, (int)rs_c, (int)cs_a, (int)rs_a, (int)cs_b, (int)rs_b);
	}
	if ( m <= (SKX_MR/2) ) { m_powedges( (SKX_MR/2), m, EDGE_A, 2, ao, bo, co ); }
	else                   { m_powedges( (SKX_MR/1), m, EDGE_A, 1, ao, bo, co ); }
}

