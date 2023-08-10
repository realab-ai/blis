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

#define SKX_MR 128

#define UPDATE_AC( nidx, n_mask, ao, co ) \
{ \
	if ( 1 == rs_a ) { \
		pmstoreu_16f(ao+nidx*16*BLIS_SIZEOF_S, zmm[c_regs[nidx]], n_mask); \
	} \
	else { \
		pmstores_16f(ao+nidx*16*rs_a*BLIS_SIZEOF_S, zmm[c_regs[nidx]], rs_a, n_mask); \
	} \
	if ( 1 == rs_c ) { \
		pmstoreu_16f(co+nidx*16*BLIS_SIZEOF_S, zmm[a_regs[nidx]], n_mask); \
	} \
	else { \
		pmstores_16f(co+nidx*16*rs_c*BLIS_SIZEOF_S, zmm[a_regs[nidx]], rs_c, n_mask); \
	} \
}

#define LOAD_AC( nidx, n_mask, ao, co ) \
{ \
	if ( 1 == rs_a ) { \
		pmzloadu_16f(ao+nidx*16*BLIS_SIZEOF_S, zmm[a_regs[nidx]], n_mask); \
	} \
	else { \
		pmzloads_16f(ao+nidx*16*rs_a*BLIS_SIZEOF_S, zmm[a_regs[nidx]], rs_a, n_mask); \
	} \
	if ( 1 == rs_c ) { \
		pmzloadu_16f(co+nidx*16*BLIS_SIZEOF_S, zmm[c_regs[nidx]], n_mask); \
	} \
	else { \
		pmzloads_16f(co+nidx*16*rs_c*BLIS_SIZEOF_S, zmm[c_regs[nidx]], rs_c, n_mask); \
	} \
}

#define EDGE_A( n_unroll, ao, co ) \
{ \
	int n_vecs = div_up( n_unroll, 16 ); \
	__mmask16 n_mask = edge_mask16( n_unroll ); \
	n_packloop( (SKX_MR/16), n_vecs, LOAD_AC, n_mask, ao, co ); \
	n_packloop( (SKX_MR/16), n_vecs, UPDATE_AC, n_mask, ao, co ); \
	ao += n_unroll*rs_a*BLIS_SIZEOF_S; \
	co += n_unroll*rs_c*BLIS_SIZEOF_S; \
}

// --------------------------------------------------------------------
// Swap corresponding elements of two n-length vectors x and y.
// Compute float swapv kernel with max unroll support of:
//     max_m_unroll: 128, 64, 32, 16, 8, 4, 2, 1
// --------------------------------------------------------------------
void bli_sswapv_skx_int_128
	 (
		     dim_t      n,
	         void*      x0, inc_t incx,
			 void*      y0, inc_t incy,
	   const cntx_t*    cntx
	 )
{
	(void)cntx;

	const int64_t rs_a       = incx;
	const int64_t rs_c       = incy;
	const int a_regs[]       = { 16, 17, 18, 19, 20, 21, 22, 23 };
	const int c_regs[]       = { 24, 25, 26, 27, 28, 29, 30, 31 };

	Packet16f zmm[32];

	void *ao = x0;
	void *co = y0;
	m_powedges( SKX_MR, n, EDGE_A, ao, co );
}

