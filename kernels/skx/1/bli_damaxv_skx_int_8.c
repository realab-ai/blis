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

#define SKX_MR 8

#define AMAX_KER( nidx, n_mask, zmm_x, zmm_mx, zmm_idx, zmm_midx, ao ) \
{ \
	if ( 1 == rs_a ) { \
		pmzloadu_8l(ao+nidx*8*BLIS_SIZEOF_D, zmm_x, n_mask); \
	} \
	else { \
		pmzloads_8l(ao+nidx*8*rs_a*BLIS_SIZEOF_D, zmm_x, rs_a, n_mask); \
	} \
	zmm_x = _mm512_and_si512( zmm_x, absmask ); \
	__mmask8 gt_mask  = _mm512_cmpgt_epu64_mask(zmm_x, zmm_mx); \
	zmm_midx = _mm512_mask_blend_epi64( gt_mask, zmm_midx, zmm_idx ); \
	zmm_mx   = _mm512_max_epu64( zmm_mx, zmm_x ); \
	zmm_idx  = _mm512_add_epi64( zmm_idx, zmm_8 ); \
}

#define EDGE_A( n_unroll, zmm_x, zmm_mx, zmm_idx, zmm_midx, ao ) \
{ \
	__mmask8 n_mask = edge_mask8( n_unroll ); \
	AMAX_KER( 0, n_mask, zmm_x, zmm_mx, zmm_idx, zmm_midx, ao ); \
	ao += n_unroll*rs_a*BLIS_SIZEOF_D; \
}

// --------------------------------------------------------------------
// y := maxidx(abs(x))
// Given a vector of length n, return the zero-based index of the element 
// of vector x that contains the largest absolute value
// Compute amaxv kernel with max unroll support of:
//     max_m_unroll: 8, 4, 2, 1
// --------------------------------------------------------------------
void bli_damaxv_skx_int_8
	 (
		     dim_t      n,
	   const void*      x0, inc_t incx,
	         dim_t*     index,
	   const cntx_t*    cntx
	 )
{
	(void)cntx;

	const int64_t rs_a       = incx;
	const dim_t* zero_i      = PASTEMAC(i,0);
	// If the vector length is zero, return early. This directly emulates
	// the behavior of netlib BLAS's i?amax() routines.
	if ( bli_zero_dim1( n ) )
	{
		PASTEMAC(i,copys)( *zero_i, *index );
		return;
	}

	const Packet8l absmask = pset1_8l(0x7FFFFFFFFFFFFFFF);
	Packet8l zmm_idx       = _mm512_set_epi64( 7, 6, 5, 4, 3, 2, 1, 0 );
	Packet8l zmm_midx      = zmm_idx;
	Packet8l zmm_x         = pzero_8l;
	Packet8l zmm_mx        = pzero_8l;
	Packet8l zmm_8         = _mm512_set1_epi64( 8 );
	int64_t max_idx        = 0;
	const void *ao         = x0;

	m_powedges( SKX_MR, n, EDGE_A, zmm_x, zmm_mx, zmm_idx, zmm_midx, ao );

	pargmax_8d( (Packet8d)zmm_mx, zmm_midx, max_idx );
	*index = max_idx;
}

