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

#define ZERO_ZMMS(midx, from) \
{ \
	zmm[from+midx] = pzero_16f; \
}

#define ACCUMULATE_C( nidx, n_mask ) \
{ \
	zmm[c_regs[nidx]] = pfmadd_16f( zmm[a_regs[nidx]], \
 			                        zmm[b_regs[nidx]], \
			                        zmm[c_regs[nidx]] ); \
}
#define VLOAD_A( nidx, n_mask, ao ) \
{ \
	if ( 1 == rs_a ) { \
		pmzloadu_16f(ao+nidx*16*BLIS_SIZEOF_S, zmm[a_regs[nidx]], n_mask); \
	} \
	else { \
		pmzloads_16f(ao+nidx*16*rs_a*BLIS_SIZEOF_S, zmm[a_regs[nidx]], rs_a, n_mask); \
	} \
}
#define VLOAD_B( nidx, n_mask, bo ) \
{ \
	if ( 1 == rs_b ) { \
		pmzloadu_16f(bo+nidx*16*BLIS_SIZEOF_S, zmm[b_regs[nidx]], n_mask); \
	} \
	else { \
		pmzloads_16f(bo+nidx*16*rs_b*BLIS_SIZEOF_S, zmm[b_regs[nidx]], rs_b, n_mask); \
	} \
}
#define EDGE_A( n_unroll, ao, bo ) \
{ \
	int n_vecs = div_up( n_unroll, 16 ); \
	__mmask16 n_mask = edge_mask16( n_unroll ); \
	n_packloop( (SKX_MR/16), n_vecs, VLOAD_A, n_mask, ao ); \
	n_packloop( (SKX_MR/16), n_vecs, VLOAD_B, n_mask, bo ); \
	n_packloop( (SKX_MR/16), n_vecs, ACCUMULATE_C, n_mask ); \
	ao += n_unroll*rs_a*BLIS_SIZEOF_S; \
	bo += n_unroll*rs_b*BLIS_SIZEOF_S; \
}

// --------------------------------------------------------------------
// rho := conj?(x)^T * conj?(y)
// where x and y are vectors of length n, and alpha, beta, and rho are scalars.
// Compute float dotv kernel with max unroll support of:
//     max_m_unroll: 128, 64, 32, 16, 8, 4, 2, 1
// --------------------------------------------------------------------
void bli_sdotv_skx_int_128
	 (
		    conj_t      conjx,
			conj_t      conjy,
		     dim_t      n,
	   const void*      x0, inc_t incx,
	   const void*      y0, inc_t incy,
	         void*      rho0,
	   const cntx_t*    cntx
	 )
{
	(void)cntx;

	const int64_t rs_a       = incx;
	const int64_t rs_b       = incy;
	const void* rho          = rho0;
	const int a_regs[]       = {  8,  9, 10, 11, 12, 13, 14, 15 };
	const int b_regs[]       = { 16, 17, 18, 19, 20, 21, 22, 23 };
	const int c_regs[]       = { 24, 25, 26, 27, 28, 29, 30, 31 };

	// If the vector dimension is zero, set rho to zero and return early.
	if ( bli_zero_dim1( n ) )
	{
		PASTEMAC(s,set0s)( *((float*)rho) );
		return;
	}

	float rho_l;
	Packet16f zmm[32];
	n_packloop( 8, 8, ZERO_ZMMS, 24 );

	const void *ao = x0;
	const void *bo = y0;

	m_powedges( SKX_MR, n, EDGE_A, ao, bo );

	// Accumulate the final result into the output variable.
	zmm[c_regs[0]] = padd_16f( zmm[c_regs[0]], zmm[c_regs[1]] );
	zmm[c_regs[2]] = padd_16f( zmm[c_regs[2]], zmm[c_regs[3]] );
	zmm[c_regs[4]] = padd_16f( zmm[c_regs[4]], zmm[c_regs[5]] );
	zmm[c_regs[6]] = padd_16f( zmm[c_regs[6]], zmm[c_regs[7]] );
	zmm[c_regs[0]] = padd_16f( zmm[c_regs[0]], zmm[c_regs[2]] );
	zmm[c_regs[4]] = padd_16f( zmm[c_regs[4]], zmm[c_regs[6]] );
	zmm[c_regs[0]] = padd_16f( zmm[c_regs[0]], zmm[c_regs[4]] );

	rho_l = _mm512_reduce_add_ps( zmm[c_regs[0]] );

	PASTEMAC(s,copys)( rho_l, *((float*)rho) );
}

