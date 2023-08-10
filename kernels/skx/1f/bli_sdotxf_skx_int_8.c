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

#define SKX_MR   8
#define SKX_KA  16
// --------------------------------------------------------------------
// Update C
// --------------------------------------------------------------------
#define UPDATE_C( m_unroll, m_mask, co) \
{ \
	preduceadd_8x16f( zmm[c_regs[0]], \
			          zmm[c_regs[1]], \
			          zmm[c_regs[2]], \
			          zmm[c_regs[3]], \
		    	      zmm[c_regs[4]], \
		        	  zmm[c_regs[5]], \
			          zmm[c_regs[6]], \
					  zmm[c_regs[7]], \
					  zmm[c_regs[0]] ); \
	zmm[c_regs[0]] = pmul_16f(zmm[c_regs[0]], zmm[alpha_load_reg]); \
	if ( 1==rs_c ) { \
		if (!is_beta0) { \
			pmzloadu_16f(co, zmm[c_load_reg], m_mask); \
			zmm[c_regs[0]] = pmfmadd_16f( zmm[beta_load_reg], \
								          zmm[c_load_reg], \
								          zmm[c_regs[0]], \
								          m_mask); \
		} \
		pmstoreu_16f(co, zmm[c_regs[0]], m_mask); \
	} \
	else { \
		if (!is_beta0) { \
			pmzloads_16f(co, zmm[c_load_reg], rs_c, m_mask); \
			zmm[c_regs[0]] = pmfmadd_16f( zmm[beta_load_reg], \
								          zmm[c_load_reg], \
								          zmm[c_regs[0]], \
								          m_mask); \
		} \
		pmstores_16f(co, zmm[c_regs[0]], rs_c, m_mask); \
	} \
}

// --------------------------------------------------------------------
// Accumulate C
// --------------------------------------------------------------------
#define ACCUMULATE_C( midx, k_mask ) \
{ \
	zmm[c_regs[midx]] = pfmadd_16f( zmm[a_regs[midx]], \
				                    zmm[b_reg], \
				                    zmm[c_regs[midx]] ); \
}
#define VLOAD_A(midx, k_mask, ao) \
{ \
	if ( 1==cs_a ) \
		pmzloadu_16f( ao+midx*rs_a*BLIS_SIZEOF_S, \
				      zmm[a_regs[midx]], \
				      k_mask ); \
	else \
		pmzloads_16f( ao+midx*rs_a*BLIS_SIZEOF_S, \
				      zmm[a_regs[midx]], \
				      cs_a, \
				      k_mask ); \
}

#define DX_KER_PK( k_unroll, m_unroll, m_mask, ao, bo, co) \
{ \
	__mmask16 k_mask = edge_mask16(k_unroll); \
	n_packloop( SKX_MR, m_unroll, VLOAD_A, k_mask, ao ); \
	if ( 1==rs_b  ) \
		pmzloadu_16f(bo, zmm[b_reg], k_mask); \
	else \
		pmzloads_16f(bo, zmm[b_reg], rs_b, k_mask); \
	n_packloop( SKX_MR, m_unroll, ACCUMULATE_C, k_mask ); \
	ao += k_unroll*cs_a*BLIS_SIZEOF_S; \
	bo += k_unroll*rs_b*BLIS_SIZEOF_S; \
}

#define ZERO_ZMMS(midx, from) \
{ \
	zmm[from+midx] = pzero_16f; \
}

#define EDGE_A(m_unroll, ao, bo, co) \
{ \
	__mmask16 m_mask = edge_mask16(m_unroll); \
	k_alignedges( SKX_KA, k, DX_KER_PK, m_unroll, m_mask, ao, bo, co ); \
	UPDATE_C( m_unroll, m_mask, co ); \
	n_packloop( 8, 8, ZERO_ZMMS, 24 ); \
	ao -= k*cs_a*BLIS_SIZEOF_S; \
	bo -= k*rs_b*BLIS_SIZEOF_S; \
	ao += m_unroll*rs_a*BLIS_SIZEOF_S; \
	co += m_unroll*rs_c*BLIS_SIZEOF_S; \
}

// --------------------------------------------------------------------
// JCJ:
//     |         --------       |
//     |         --------       |
//     |    +=   --------  ...  |
//     |         --------       |
//     |         --------       :
//     |         --------       :
// Assumptions:
// - y is column-stored vector, A is row-stored, x is column-stored vector;
// Therefore, this row-preferential kernel is well-suited for dot-production
// loads on A and x.
// --------------------------------------------------------------------

// --------------------------------------------------------------------
// Compute float dotxf kernel with max unroll support of:
//     max_m_unroll:  8,  4,  2,  1
//     max_k_align:  16
// --------------------------------------------------------------------
void bli_sdotxf_skx_int_8
	 (
		    conj_t      conjat,
		    conj_t      conjx,
		     dim_t      m0,
		     dim_t      b_n,
	   const void*      alpha0,
	   const void*      a0, inc_t inca, inc_t lda,
	   const void*      x0, inc_t incx,
	   const void*      beta0,
	         void*      y0, inc_t incy,
	   const cntx_t*    cntx
	 )
{
	(void)cntx;
	
	const int64_t m    = b_n;
	const int64_t k    = m0;
	const int64_t rs_a = lda;
	const int64_t cs_a = inca;
	const int64_t rs_b = incx;
	const int64_t cs_b = 0;
	const int64_t rs_c = incy;
	const int64_t cs_c = 0;

	// If the b_n dimension is zero, y is empty and there is no computation.
	if ( bli_zero_dim1( b_n ) ) return;

	// If the m dimension is zero, or if alpha is zero, the computation
	// simplifies to updating y.
	if ( bli_zero_dim1( m0 ) || PASTEMAC(s,eq0)( *((float*)alpha0 ) ) )
	{
		scalv_ker_ft f = bli_cntx_get_ukr_dt( BLIS_FLOAT, BLIS_SCALV_KER, cntx );
		f
		(
			BLIS_NO_CONJUGATE,
			b_n,
			(float*)beta0,
			(float*)y0, incy,
			cntx
		);
		return;
	}


	const void* alpha        = alpha0;
	const void* beta         = beta0;
	const bool is_beta0      = *((float *)beta )==0? true: false;
	const int a_regs[]       = {  8,  9, 10, 11, 12, 13, 14, 15 };
	const int c_regs[]       = { 24, 25, 26, 27, 28, 29, 30, 31 };
	const int b_reg          = 0;
	const int c_load_reg     = 1;
	const int alpha_load_reg = 4;
	const int beta_load_reg  = 5;
	
	Packet16f zmm[32];
	n_packloop( 8, 8, ZERO_ZMMS, 24);

	zmm[alpha_load_reg] = pload1_16f(alpha);
	zmm[ beta_load_reg] = pload1_16f(beta);

	const void *ao = a0;
	const void *bo = x0;
		  void *co = y0;
	if(bli_info_get_enable_diagnosis())
	{
		printf("KER m    n    k alpha beta cs_c rs_c cs_a rs_a cs_b rs_b\n");
		printf("%5d%5d%5d%5.1f%5.1f%5d%5d%5d%5d%5d%5d\n", 
				(int)m, 1, (int)k, *((float*)alpha), *((float*)beta), 
				(int)cs_c, (int)rs_c, (int)cs_a, (int)rs_a, (int)cs_b, (int)rs_b);
	}
	n_powedges( SKX_MR, m, EDGE_A, ao, bo, co );
}

