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
#define SKX_KA 16

// --------------------------------------------------------------------
// Update C
// --------------------------------------------------------------------
#define UPDATE_C_ROW(midx, n_unroll, n_mask, co, ldc, s_c) \
{ \
	preduceadd_8x16f( zmm[c_regs[0*2+midx]], zmm[c_regs[1*2+midx]], \
			          zmm[c_regs[2*2+midx]], zmm[c_regs[3*2+midx]], \
			          zmm[c_regs[4*2+midx]], zmm[c_regs[5*2+midx]], \
			          zmm[c_regs[6*2+midx]], zmm[c_regs[7*2+midx]], \
			          zmm[c_store_regs[midx]]); \
	zmm[c_regs[0*2+midx]] = pzero_16f;  zmm[c_regs[1*2+midx]] = pzero_16f;  \
	zmm[c_regs[2*2+midx]] = pzero_16f;  zmm[c_regs[3*2+midx]] = pzero_16f;  \
	zmm[c_regs[4*2+midx]] = pzero_16f;  zmm[c_regs[5*2+midx]] = pzero_16f;  \
	zmm[c_regs[6*2+midx]] = pzero_16f;  zmm[c_regs[7*2+midx]] = pzero_16f;  \
	zmm[c_store_regs[midx]] = pmul_16f(zmm[c_store_regs[midx]], zmm[alpha_load_reg]); \
	if (!is_beta0) { \
		pmzloadu_16f(co, zmm[c_load_regs[midx]], n_mask); \
		zmm[c_store_regs[midx]] = pmfmadd_16f( zmm[beta_load_reg], \
			                                   zmm[c_load_regs[midx]], \
			                                   zmm[c_store_regs[midx]], \
			                                   n_mask); \
	} \
	if (1==s_c) { \
		pmstoreu_16f(co, zmm[c_store_regs[midx]], n_mask); \
	} \
	else { \
		pmstores_16f(co, zmm[c_store_regs[midx]], s_c, n_mask); \
	} \
	co += ldc*BLIS_SIZEOF_S; \
}

#define UPDATE_C(m_unroll, n_unroll, n_mask, co) \
{ \
	if ( 0 < m_unroll ) UPDATE_C_ROW( 0, n_unroll, n_mask, co, rs_c, cs_c ); \
	if ( 1 < m_unroll ) UPDATE_C_ROW( 1, n_unroll, n_mask, co, rs_c, cs_c ); \
	co -= m_unroll*rs_c*BLIS_SIZEOF_S; \
}

// --------------------------------------------------------------------
// Accumulate C
// --------------------------------------------------------------------
#define CD_KER_UMUN( nidx, midx, k_mask, bo ) \
{ \
	pmzloadu_16f( bo+nidx*cs_b*BLIS_SIZEOF_S, zmm[b_regs[nidx]], k_mask ); \
	zmm[c_regs[nidx*2+midx]] = pfmadd_16f( zmm[a_regs[midx]], \
			                               zmm[b_regs[nidx]], \
				                           zmm[c_regs[nidx*2+midx]]); \
}

#define CD_KER_UM( midx, n_unroll, k_mask, ao, bo ) \
{ \
	pmzloadu_16f( ao+midx*rs_a*BLIS_SIZEOF_S, zmm[a_regs[midx]], k_mask ); \
	n_packloop( SKX_NR, n_unroll, CD_KER_UMUN, midx, k_mask, bo); \
}
#define CD_KER_PK(k_unroll, m_unroll, n_unroll, ao, bo) \
{ \
	__mmask16 k_mask = edge_mask16(k_unroll); \
	if ( 0 < m_unroll ) CD_KER_UM( 0, n_unroll, k_mask, ao, bo ); \
	if ( 1 < m_unroll ) CD_KER_UM( 1, n_unroll, k_mask, ao, bo ); \
	ao += k_unroll*cs_a*BLIS_SIZEOF_S; \
	bo += k_unroll*rs_b*BLIS_SIZEOF_S; \
}

#define EDGE_B(n_unroll, m_unroll, ao, bo, co) \
{ \
	__mmask8 n_mask = edge_mask8(n_unroll); \
	k_alignedges( SKX_KA, k, CD_KER_PK, m_unroll, n_unroll, ao, bo ); \
	UPDATE_C(m_unroll, n_unroll, n_mask, co); \
	ao -= k*cs_a*BLIS_SIZEOF_S; \
	bo -= k*rs_b*BLIS_SIZEOF_S; \
	bo += n_unroll*cs_b*BLIS_SIZEOF_S; \
	co += n_unroll*cs_c*BLIS_SIZEOF_S; \
}

#define EDGE_A(m_unroll, ao, bo, co) \
{ \
	n_powedges(SKX_NR, n, EDGE_B, m_unroll, ao, bo, co); \
	bo -= n*cs_b*BLIS_SIZEOF_S; \
	co -= n*cs_c*BLIS_SIZEOF_S; \
	ao += m_unroll*rs_a*BLIS_SIZEOF_S; \
	co += m_unroll*rs_c*BLIS_SIZEOF_S; \
}

#define ZERO_ZMMS(midx, from) \
{ \
	zmm[from+midx] = pzero_16f; \
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
// Compute float SUPMM row-dot (rd) kernel with max unroll support of:
//     max_a_unroll: 2, 1
//     max_b_unroll: 8, 4, 2, 1
// --------------------------------------------------------------------
void bli_sgemmsup_rd_skx_int_2x8
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

	const bool is_beta0      = *((float *)beta )==0? true: false;
	const int a_regs[]       = {  0,  1 };
	const int b_regs[]       = {  8,  9, 10, 11, 12, 13, 14, 15 };
	const int c_regs[]       = { 16, 17, 18, 19, 20, 21, 22, 23,
			                     24, 25, 26, 27, 28, 29, 30, 31 };
	const int c_load_regs[]  = {  0,  1 };
	const int c_store_regs[] = {  2,  3 };
	const int alpha_load_reg = 4;
	const int beta_load_reg  = 5;

	Packet16f zmm[32];
	m_packloop( 16, 16, ZERO_ZMMS, 16 );
	zmm[alpha_load_reg] = pload1_16f(alpha);
	zmm[ beta_load_reg] = pload1_16f(beta);

	const void *ao = a;
	const void *bo = b;
		  void *co = c;
	if (bli_info_get_enable_diagnosis())
	{
		printf("URD m    n    k alpha beta cs_c rs_c cs_a rs_a cs_b rs_b\n");
		printf("%s%s%3d%5d%5d%5.1f%5.1f%5d%5d%5d%5d%5d%5d\n", 
				(cs_a==SKX_MR && rs_a==1)?" ":"A", (rs_b==SKX_NR && cs_b==1)?" ":"B",
			    (int)m, (int)n, (int)k, *((float *)alpha), *((float *)beta), 
			    (int)cs_c, (int)rs_c, (int)cs_a, (int)rs_a, (int)cs_b, (int)rs_b);
	}
	m_powedges( 2, m, EDGE_A, ao, bo, co );
}
