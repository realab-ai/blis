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

#define UPDATE_C( nidx, n_mask, co ) \
{ \
	if ( 1 == rs_c ) { \
		pmzloadu_8d(co+nidx*8*BLIS_SIZEOF_D, zmm[c_regs[nidx]], n_mask); \
		zmm[c_regs[nidx]] = pmmul_8d( zmm[alpha_load_reg], \
									  zmm[c_regs[nidx]], \
				                      zmm[c_regs[nidx]], \
									  n_mask ); \
		pmstoreu_8d(co+nidx*8*BLIS_SIZEOF_D, zmm[c_regs[nidx]], n_mask); \
	} \
	else { \
		pmzloads_8d(co+nidx*8*rs_c*BLIS_SIZEOF_D, zmm[c_regs[nidx]], rs_c, n_mask); \
		zmm[c_regs[nidx]] = pmmul_8d( zmm[alpha_load_reg], \
									  zmm[c_regs[nidx]], \
				                      zmm[c_regs[nidx]], \
									  n_mask ); \
		pmstores_8d(co+nidx*8*rs_c*BLIS_SIZEOF_D, zmm[c_regs[nidx]], rs_c, n_mask); \
	} \
}

#define EDGE_A( n_unroll, co ) \
{ \
	int n_vecs = div_up( n_unroll, 8 ); \
	__mmask8 n_mask = edge_mask8( n_unroll ); \
	n_packloop( (SKX_MR/8), n_vecs, UPDATE_C, n_mask, co ); \
	co += n_unroll*rs_c*BLIS_SIZEOF_D; \
}

// --------------------------------------------------------------------
// x := conj?(alpha) * x
// Given an n-length vector x, scale all elements with the scalar alpha. 
// Compute double scalv kernel with max unroll support of:
//     max_m_unroll: 64, 32, 16, 8, 4, 2, 1
// --------------------------------------------------------------------
void bli_dscalv_skx_int_64
	 (
		    conj_t      conjalpha,
		     dim_t      n,
	   const void*      alpha0,
	         void*      x0, inc_t incx,
	   const cntx_t*    cntx
	 )
{
	(void)cntx;

	const int64_t rs_c       = incx;
	const void* alpha        = alpha0;
	const int c_regs[]       = { 24, 25, 26, 27, 28, 29, 30, 31 };
	const int alpha_load_reg = 4;

	// If the vector dimension is zero, or if alpha is unit, return early.
	if ( bli_zero_dim1( n ) || PASTEMAC(d, eq1)( *((double*)alpha) ) ) 
		return;

	Packet8d zmm[32];
	zmm[alpha_load_reg] = pload1_8d(alpha);

	void *co = x0;
	m_powedges( SKX_MR, n, EDGE_A, co );
}

