/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2018, The University of Texas at Austin
   Copyright (C) 2019, Advanced Micro Devices, Inc.

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
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#ifndef BLIS_AVX512_INTRIN_MACROS_H
#define BLIS_AVX512_INTRIN_MACROS_H

// --------------------------------------------------------------------
// Type definitions
// --------------------------------------------------------------------
typedef __m64   Packet2f;
typedef __m64   Packet1d;
typedef __m128  Packet4f;
typedef __m128d Packet2d;
typedef __m128i Packet4i;
typedef __m256  Packet8f;
typedef __m256d Packet4d;
typedef __m256i Packet8i;
typedef __m512  Packet16f;
typedef __m512d Packet8d;
typedef __m512i Packet16i;
typedef __m512i Packet8l;

typedef const char * SsePrefetchPtrType;

// --------------------------------------------------------------------
// Initial value set
// --------------------------------------------------------------------
#define pzero_16f _mm512_setzero_ps()
#define pzero_8d  _mm512_setzero_pd()
#define pzero_16i _mm512_setzero_epi32()
#define pzero_8l  _mm512_setzero_si512()

#define pset1_16f( from ) _mm512_set1_ps( from )
#define pset1_8d(  from ) _mm512_set1_pd( from )
#define pset1_16i( from ) _mm512_set1_epi32( from )
#define pset1_8l(  from ) _mm512_set1_epi64( from )

#define edge_mask8(I_)  (( 8 <= (I_))? 0x00FF : ((1<<(I_))-1))
#define edge_mask16(I_) ((16 <= (I_))? 0xFFFF : ((1<<(I_))-1))

// --------------------------------------------------------------------
// Vector + - * /
// --------------------------------------------------------------------
#define padd_16f(a,b) \
	_mm512_add_ps((Packet16f)(a), (Packet16f)(b))
#define padd_8d(a,b) \
	_mm512_add_pd((Packet8d )(a), (Packet8d) (b))
#define psub_16f(a,b) \
	_mm512_sub_ps((Packet16f)(a), (Packet16f)(b))
#define psub_8d(a,b) \
	_mm512_sub_pd((Packet8d )(a), (Packet8d) (b))
#define pmul_16f(a,b) \
	_mm512_mul_ps((Packet16f)(a), (Packet16f)(b))
#define pmul_8d(a,b) \
	_mm512_mul_pd((Packet8d )(a), (Packet8d) (b))
#define pdiv_16f(a,b) \
	_mm512_div_ps((Packet16f)(a), (Packet16f)(b))
#define pdiv_8d(a,b) \
	_mm512_div_pd((Packet8d )(a), (Packet8d) (b))

#define pfmadd_16f(a,b,c) \
	_mm512_fmadd_ps((Packet16f)(a), (Packet16f)(b), (Packet16f)(c))
#define pfmadd_8d(a,b,c)  \
	_mm512_fmadd_pd((Packet8d )(a), (Packet8d) (b), (Packet8d) (c))
#define pfmsub_16f(a,b,c) \
	_mm512_fmsub_ps((Packet16f)(a), (Packet16f)(b), (Packet16f)(c))
#define pfmsub_8d(a,b,c)  \
	_mm512_fmsub_pd((Packet8d )(a), (Packet8d) (b), (Packet8d) (c))
#define pfnmadd_16f(a,b,c) \
	_mm512_fnmadd_ps((Packet16f)(a), (Packet16f)(b), (Packet16f)(c))
#define pfnmadd_8d(a,b,c)  \
	_mm512_fnmadd_pd((Packet8d )(a), (Packet8d) (b), (Packet8d) (c))
#define pfnmsub_16f(a,b,c) \
	_mm512_fnmsub_ps((Packet16f)(a), (Packet16f)(b), (Packet16f)(c))
#define pfnmsub_8d(a,b,c)  \
	_mm512_fnmsub_pd((Packet8d )(a), (Packet8d) (b), (Packet8d) (c))

// mask version
#define pmadd_16f(a,b,c,mask) \
	_mm512_mask_add_ps((Packet16f)(c), (__mmask16)(mask), (Packet16f)(a), (Packet16f)(b))
#define pmadd_8d(a,b,c,mask) \
	_mm512_mask_add_pd((Packet8d)(c), (__mmask8)(mask), (Packet8d )(a), (Packet8d) (b))
#define pmsub_16f(a,b,c,mask) \
	_mm512_mask_sub_ps(((Packet16f)(c), (__mmask16)(mask), Packet16f)(a), (Packet16f)(b))
#define pmsub_8d(a,b,c,mask) \
	_mm512_mask_sub_pd((Packet8d)(c), (__mmask8)(mask), (Packet8d )(a), (Packet8d) (b))
#define pmmul_16f(a,b,c,mask) \
	_mm512_mask_mul_ps((Packet16f)(c), (__mmask16)(mask), (Packet16f)(a), (Packet16f)(b))
#define pmmul_8d(a,b,c,mask) \
	_mm512_mask_mul_pd((Packet8d)(c), (__mmask8)(mask), (Packet8d )(a), (Packet8d) (b))
#define pmdiv_16f(a,b,c,mask) \
	_mm512_mask_div_ps((Packet16f)(c), (__mmask16)(mask), (Packet16f)(a), (Packet16f)(b))
#define pmdiv_8d(a,b,c,mask) \
	_mm512_mask_div_pd((Packet8d)(c), (__mmask8)(mask), (Packet8d )(a), (Packet8d) (b))

#define pmfmadd_16f(a,b,c,mask) \
	_mm512_mask_fmadd_ps((Packet16f)(a), (__mmask16)(mask), (Packet16f)(b), (Packet16f)(c))
#define pmfmadd_8d(a,b,c,mask)  \
	_mm512_mask_fmadd_pd((Packet8d )(a), (__mmask8)(mask), (Packet8d) (b), (Packet8d) (c))
#define pmfmsub_16f(a,b,c,mask) \
	_mm512_mask_fmsub_ps((Packet16f)(a), (__mmask16)(mask), (Packet16f)(b), (Packet16f)(c))
#define pmfmsub_8d(a,b,c,mask)  \
	_mm512_mask_fmsub_pd((Packet8d )(a), (__mmask8)(mask), (Packet8d) (b), (Packet8d) (c))
#define pmfnmadd_16f(a,b,c,mask) \
	_mm512_mask_fnmadd_ps((Packet16f)(a), (__mmask16)(mask), (Packet16f)(b), (Packet16f)(c))
#define pmfnmadd_8d(a,b,c,mask)  \
	_mm512_mask_fnmadd_pd((Packet8d )(a), (__mmask8)(mask), (Packet8d) (b), (Packet8d) (c))
#define pmfnmsub_16f(a,b,c,mask) \
	_mm512_mask_fnmsub_ps((Packet16f)(a), (__mmask16)(mask), (Packet16f)(b), (Packet16f)(c))
#define pmfnmsub_8d(a,b,c,mask)  \
	_mm512_mask_fnmsub_pd((Packet8d )(a), (__mmask8)(mask), (Packet8d) (b), (Packet8d) (c))

// --------------------------------------------------------------------
// Load from mem
// --------------------------------------------------------------------
// Aligned load
#define pload_16f( from, to ) to = _mm512_load_ps( from )
#define pload_8d(  from, to ) to = _mm512_load_pd( from )
#define pload_16i( from, to ) to = _mm512_load_epi32( from )
#define pload_8l(  from, to ) to = _mm512_load_epi64( from )
// Align-masked load
#define pmload_16f(  from, to, mask ) to = _mm512_mask_load_ps( to, mask, from )
#define pmload_8d(   from, to, mask ) to = _mm512_mask_load_pd( to, mask, from )
#define pmload_16i(  from, to, mask ) to = _mm512_mask_load_epi32( to, mask, from )
#define pmload_8l(   from, to, mask ) to = _mm512_mask_load_epi64( to, mask, from )
// Align-mask-zeroed lad
#define pmzload_16f( from, to, mask ) to = _mm512_maskz_load_ps( mask, from )
#define pmzload_8d(  from, to, mask ) to = _mm512_maskz_load_pd( mask, from )
#define pmzload_16i( from, to, mask ) to = _mm512_maskz_load_epi32( mask, from )
#define pmzload_8l(  from, to, mask ) to = _mm512_maskz_load_epi64( mask, from )

// Unaligned load
#define ploadu_16f( from, to ) to = _mm512_loadu_ps( from )
#define ploadu_8d(  from, to ) to = _mm512_loadu_pd( from )
#define ploadu_16i( from, to ) to = _mm512_loadu_epi32( from )
#define ploadu_8l(  from, to ) to = _mm512_loadu_epi64( from )
// Unalign-masked load
#define pmloadu_16f(  from, to, mask ) to = _mm512_mask_loadu_ps( to, mask, from )
#define pmloadu_8d(   from, to, mask ) to = _mm512_mask_loadu_pd( to, mask, from )
#define pmloadu_16i(  from, to, mask ) to = _mm512_mask_loadu_epi32( to, mask, from )
#define pmloadu_8l(   from, to, mask ) to = _mm512_mask_loadu_epi64( to, mask, from )
// Unalign-mask-zeroed load
#define pmzloadu_16f( from, to, mask ) to = _mm512_maskz_loadu_ps( mask, from )
#define pmzloadu_8d(  from, to, mask ) to = _mm512_maskz_loadu_pd( mask, from )
#define pmzloadu_16i( from, to, mask ) to = _mm512_maskz_loadu_epi32( mask, from )
#define pmzloadu_8l(  from, to, mask ) to = _mm512_maskz_loadu_epi64( mask, from )

// strided load
#define ploads_16f(from, to, stride) \
{ \
	Packet16i stride_vector = _mm512_set1_epi32(stride); \
	Packet16i stride_multiplier = _mm512_set_epi32( 15, 14, 13, 12, 11, 10, 9, 8, \
			                                         7,  6,  5,  4,  3,  2, 1, 0 ); \
	Packet16i indices = _mm512_mullo_epi32(stride_vector, stride_multiplier); \
	to = _mm512_i32gather_ps(indices, from, 4); \
}
#define ploads_8d(from, to, stride) \
{ \
	Packet8i stride_vector = _mm256_set1_epi32(stride); \
	Packet8i stride_multiplier = _mm256_set_epi32( 7, 6, 5, 4, 3, 2, 1, 0 ); \
	Packet8i indices = _mm256_mullo_epi32(stride_vector, stride_multiplier); \
	to = _mm512_i32gather_pd( indices, from, 8 ); \
}
#define ploads_16i(from, to, stride) \
{ \
	Packet16i stride_vector = _mm512_set1_epi32(stride); \
	Packet16i stride_multiplier = _mm512_set_epi32( 15, 14, 13, 12, 11, 10, 9, 8, \
			                                         7,  6,  5,  4,  3,  2, 1, 0 ); \
	Packet16i indices = _mm512_mullo_epi32(stride_vector, stride_multiplier); \
	to = _mm512_i32gather_epi32(indices, from, 4); \
}
#define ploads_8l(from, to, stride) \
{ \
	Packet8i stride_vector = _mm256_set1_epi32(stride); \
	Packet8i stride_multiplier = _mm256_set_epi32( 7, 6, 5, 4, 3, 2, 1, 0 ); \
	Packet8i indices = _mm256_mullo_epi32(stride_vector, stride_multiplier); \
	to = _mm512_i32gather_epi64( indices, from, 8 ); \
}
// stride-masked load
#define pmloads_16f(from, to, stride, mask) \
{ \
	Packet16i stride_vector = _mm512_set1_epi32(stride); \
	Packet16i stride_multiplier = _mm512_set_epi32( 15, 14, 13, 12, 11, 10, 9, 8, \
			                                         7,  6,  5,  4,  3,  2, 1, 0 ); \
	Packet16i indices = _mm512_mullo_epi32(stride_vector, stride_multiplier); \
	to = _mm512_mask_i32gather_ps( to, mask, indices, from, 4 ); \
}
#define pmloads_8d(from, to, stride, mask) \
{ \
	Packet8i stride_vector = _mm256_set1_epi32(stride); \
	Packet8i stride_multiplier = _mm256_set_epi32( 7, 6, 5, 4, 3, 2, 1, 0 ); \
	Packet8i indices = _mm256_mullo_epi32( stride_vector, stride_multiplier ); \
	to = _mm512_mask_i32gather_pd( to, mask, indices, from, 8 ); \
}
#define pmloads_16i(from, to, stride, mask) \
{ \
	Packet16i stride_vector = _mm512_set1_epi32(stride); \
	Packet16i stride_multiplier = _mm512_set_epi32( 15, 14, 13, 12, 11, 10, 9, 8, \
			                                         7,  6,  5,  4,  3,  2, 1, 0 ); \
	Packet16i indices = _mm512_mullo_epi32(stride_vector, stride_multiplier); \
	to = _mm512_mask_i32gather_epi32( to, mask, indices, from, 4 ); \
}
#define pmloads_8l(from, to, stride, mask) \
{ \
	Packet8i stride_vector = _mm256_set1_epi32(stride); \
	Packet8i stride_multiplier = _mm256_set_epi32( 7, 6, 5, 4, 3, 2, 1, 0 ); \
	Packet8i indices = _mm256_mullo_epi32( stride_vector, stride_multiplier ); \
	to = _mm512_mask_i32gather_epi64( to, mask, indices, from, 8 ); \
}
// stride-mask-zeroed load
#define pmzloads_16f(from, to, stride, mask) \
{ \
	Packet16i stride_vector = _mm512_set1_epi32(stride); \
	Packet16i stride_multiplier = _mm512_set_epi32( 15, 14, 13, 12, 11, 10, 9, 8, \
			                                         7,  6,  5,  4,  3,  2, 1, 0 ); \
	Packet16i indices = _mm512_mullo_epi32(stride_vector, stride_multiplier); \
	to = _mm512_setzero_ps(); \
	to = _mm512_mask_i32gather_ps( to, mask, indices, from, 4 ); \
}
#define pmzloads_8d(from, to, stride, mask) \
{ \
	Packet8i stride_vector = _mm256_set1_epi32( stride ); \
	Packet8i stride_multiplier = _mm256_set_epi32( 7, 6, 5, 4, 3, 2, 1, 0 ); \
	Packet8i indices = _mm256_mullo_epi32( stride_vector, stride_multiplier ); \
	to = _mm512_setzero_pd(); \
	to = _mm512_mask_i32gather_pd( to, mask, indices, from, 8 ); \
}
#define pmzloads_16i(from, to, stride, mask) \
{ \
	Packet16i stride_vector = _mm512_set1_epi32(stride); \
	Packet16i stride_multiplier = _mm512_set_epi32( 15, 14, 13, 12, 11, 10, 9, 8, \
			                                         7,  6,  5,  4,  3,  2, 1, 0 ); \
	Packet16i indices = _mm512_mullo_epi32(stride_vector, stride_multiplier); \
	to = _mm512_setzero_epi32(); \
	to = _mm512_mask_i32gather_epi32( to, mask, indices, from, 4 ); \
}
#define pmzloads_8l(from, to, stride, mask) \
{ \
	Packet8i stride_vector = _mm256_set1_epi32( stride ); \
	Packet8i stride_multiplier = _mm256_set_epi32( 7, 6, 5, 4, 3, 2, 1, 0 ); \
	Packet8i indices = _mm256_mullo_epi32( stride_vector, stride_multiplier ); \
	to = _mm512_setzero_si512(); \
	to = _mm512_mask_i32gather_epi64( to, mask, indices, from, 8 ); \
}


// load1
#define pload1_16f( from ) _mm512_set1_ps(*(( float*)(from)))
#define pload1_8d(  from ) _mm512_set1_pd(*((double*)(from)))
#define pload1_16i( from ) _mm512_set1_epi32(*((__int32*)(from)))
#define pload1_8l(  from ) _mm512_set1_epi64(*((__int64*)(from)))

// --------------------------------------------------------------------
// Store to mem
// --------------------------------------------------------------------
// Aligned store
#define pstore_16f( to, from ) _mm512_store_ps( to, from )
#define pstore_8d(  to, from ) _mm512_store_pd( to, from )
#define pstore_16i( to, from ) _mm512_store_epi32( to, from )
#define pstore_8l(  to, from ) _mm512_store_epi64( to, from )
// Align-masked store
#define pmstore_16f( to, from, mask ) _mm512_mask_store_ps( to, mask, from )
#define pmstore_8d(  to, from, mask ) _mm512_mask_store_pd( to, mask, from )
#define pmstore_16i( to, from, mask ) _mm512_mask_store_epi32( to, mask, from )
#define pmstore_8l(  to, from, mask ) _mm512_mask_store_epi64( to, mask, from )

// Unaligned store
#define pstoreu_16f( to, from ) _mm512_storeu_ps( to, from )
#define pstoreu_8d(  to, from ) _mm512_storeu_pd( to, from )
#define pstoreu_16i( to, from ) _mm512_storeu_epi32( to, from )
#define pstoreu_8l(  to, from ) _mm512_storeu_epi64( to, from )
// Unalign-masked store
#define pmstoreu_16f( to, from, mask ) _mm512_mask_storeu_ps( to, mask, from )
#define pmstoreu_8d(  to, from, mask ) _mm512_mask_storeu_pd( to, mask, from )
#define pmstoreu_16i( to, from, mask ) _mm512_mask_storeu_epi32( to, mask, from )
#define pmstoreu_8l(  to, from, mask ) _mm512_mask_storeu_epi64( to, mask, from )

// strided (masked) store
#define pstores_16f(to, from, stride) \
{ \
	Packet16i stride_vector = _mm512_set1_epi32(stride); \
	Packet16i stride_multiplier = \
		_mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0); \
	Packet16i indices = _mm512_mullo_epi32(stride_vector, stride_multiplier); \
	_mm512_i32scatter_ps(to, indices, from, 4); \
}
#define pstores_8d(to, from, stride) \
{ \
	Packet8i stride_vector = _mm256_set1_epi32(stride); \
	Packet8i stride_multiplier = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0); \
	Packet8i indices = _mm256_mullo_epi32(stride_vector, stride_multiplier); \
	_mm512_i32scatter_pd(to, indices, from, 8); \
}
#define pmstores_16f(to, from, stride, mask) \
{ \
	Packet16i stride_vector = _mm512_set1_epi32(stride); \
	Packet16i stride_multiplier = \
		_mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0); \
	Packet16i indices = _mm512_mullo_epi32(stride_vector, stride_multiplier); \
	_mm512_mask_i32scatter_ps(to, mask, indices, from, 4); \
}
#define pmstores_8d(to, from, stride, mask) \
{ \
	Packet8i stride_vector = _mm256_set1_epi32(stride); \
	Packet8i stride_multiplier = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0); \
	Packet8i indices = _mm256_mullo_epi32(stride_vector, stride_multiplier); \
	_mm512_mask_i32scatter_pd(to, mask, indices, from, 8); \
}

// --------------------------------------------------------------------
// Prefetch
// --------------------------------------------------------------------
#define prefetch(addr, hint) _mm_prefetch( (SsePrefetchPtrType)(addr), hint )

// prefetch 1 cacheline
#define prefetch_8d_at(idx, addr, s_i, hint) \
{ \
	prefetch(addr + (idx)*s_i*8, hint); \
}
#define prefetch_16f_at(idx, addr, s_i, hint) \
{ \
	prefetch(addr + (idx)*s_i*16, hint); \
}
// prefetch 8 cachelines
#define prefetch_8x8d(addr, ld_x, hint) \
{ \
	prefetch(addr + 0*ld_x*8, hint); \
	prefetch(addr + 1*ld_x*8, hint); \
	prefetch(addr + 2*ld_x*8, hint); \
	prefetch(addr + 3*ld_x*8, hint); \
	prefetch(addr + 4*ld_x*8, hint); \
	prefetch(addr + 5*ld_x*8, hint); \
	prefetch(addr + 6*ld_x*8, hint); \
	prefetch(addr + 7*ld_x*8, hint); \
}
#define prefetch_8x16f(addr, ld_x, hint) \
{ \
	prefetch(addr + 0*ld_x*4, hint); \
	prefetch(addr + 1*ld_x*4, hint); \
	prefetch(addr + 2*ld_x*4, hint); \
	prefetch(addr + 3*ld_x*4, hint); \
	prefetch(addr + 4*ld_x*4, hint); \
	prefetch(addr + 5*ld_x*4, hint); \
	prefetch(addr + 6*ld_x*4, hint); \
	prefetch(addr + 7*ld_x*4, hint); \
}

// prefetch x cachelines
#define prefetch_x8d(x, addr, ld_x, hint) \
{ \
	int x_i = x; \
	if (8 < x_i) { \
		for (; x_i > 8; x_i-=8) { \
			prefetch_8x8d(addr+(x-x_i)*ld_x*8, ld_x, hint); \
		} \
	} \
	if (8 == x_i) { \
		prefetch_8x8d(addr+(x-x_i)*ld_x*8, ld_x, hint); \
	} \
	else { \
		for (; x_i > 0; x_i--) { \
			prefetch(addr + (x-x_i)*ld_x*8, hint); \
		}\
	} \
}
#define prefetch_x8d_at(idx, x, addr, ld_x, s_i, hint) { \
	prefetch_x8d(x, addr+(idx)*s_i*8, ld_x, hint); \
}

#define prefetch_x16f(x, addr, ld_x, hint) \
{ \
	int x_i = x; \
	if (8 < x) { \
		for (; x_i > 8; x_i-=8) { \
			prefetch_8x16f(addr+(x-x_i)*ld_x*4, ld_x, hint); \
		} \
	} \
	if (8==x_i) { \
		prefetch_8x16f(addr+(x-x_i)*ld_x*4, ld_x, hint); \
	} \
	else { \
		for (; x_i >0; x_i--) { \
			prefetch(addr + (x-x_i)*ld_x*4, hint); \
		} \
	} \
}
#define prefetch_x16f_at(idx, x, addr, ld_x, s_i, hint) { \
	prefetch_x16f(x, addr+(idx)*s_i*4, ld_x, hint); \
}

// --------------------------------------------------------------------
// Inner register transpose
// --------------------------------------------------------------------
#define ptranspose_8x8d(zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7) \
{ \
	Packet8d T0 = _mm512_shuffle_pd(zmm0, zmm1, 0x00); \
	Packet8d T1 = _mm512_shuffle_pd(zmm0, zmm1, 0xFF); \
	Packet8d T2 = _mm512_shuffle_pd(zmm2, zmm3, 0x00); \
	Packet8d T3 = _mm512_shuffle_pd(zmm2, zmm3, 0xFF); \
	Packet8d T4 = _mm512_shuffle_pd(zmm4, zmm5, 0x00); \
	Packet8d T5 = _mm512_shuffle_pd(zmm4, zmm5, 0xFF); \
	Packet8d T6 = _mm512_shuffle_pd(zmm6, zmm7, 0x00); \
	Packet8d T7 = _mm512_shuffle_pd(zmm6, zmm7, 0xFF); \
	\
	zmm0 = _mm512_permutex_pd(T2, 0x4E); \
	zmm0 = _mm512_mask_blend_pd(0xCC, T0, zmm0); \
	zmm1 = _mm512_permutex_pd(T3, 0x4E); \
	zmm1 = _mm512_mask_blend_pd(0xCC, T1, zmm1); \
	zmm2 = _mm512_permutex_pd(T0, 0x4E); \
	zmm2 = _mm512_mask_blend_pd(0xCC, zmm2, T2); \
	zmm3 = _mm512_permutex_pd(T1, 0x4E); \
	zmm3 = _mm512_mask_blend_pd(0xCC, zmm3, T3); \
	zmm4 = _mm512_permutex_pd(T6, 0x4E); \
	zmm4 = _mm512_mask_blend_pd(0xCC, T4, zmm4); \
	zmm5 = _mm512_permutex_pd(T7, 0x4E); \
	zmm5 = _mm512_mask_blend_pd(0xCC, T5, zmm5); \
	zmm6 = _mm512_permutex_pd(T4, 0x4E); \
	zmm6 = _mm512_mask_blend_pd(0xCC, zmm6, T6); \
	zmm7 = _mm512_permutex_pd(T5, 0x4E); \
	zmm7 = _mm512_mask_blend_pd(0xCC, zmm7, T7); \
	\
	T0 = _mm512_shuffle_f64x2(zmm0, zmm4, 0x44); \
	T4 = _mm512_shuffle_f64x2(zmm0, zmm4, 0xEE); \
	T1 = _mm512_shuffle_f64x2(zmm1, zmm5, 0x44); \
	T5 = _mm512_shuffle_f64x2(zmm1, zmm5, 0xEE); \
	T2 = _mm512_shuffle_f64x2(zmm2, zmm6, 0x44); \
	T6 = _mm512_shuffle_f64x2(zmm2, zmm6, 0xEE); \
	T3 = _mm512_shuffle_f64x2(zmm3, zmm7, 0x44); \
	T7 = _mm512_shuffle_f64x2(zmm3, zmm7, 0xEE); \
	\
	zmm0 = T0; \
	zmm1 = T1; \
	zmm2 = T2; \
	zmm3 = T3; \
	zmm4 = T4; \
	zmm5 = T5; \
	zmm6 = T6; \
	zmm7 = T7; \
}
#define ptranspose_8x16f(zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7) \
{ \
	Packet16f T0 = _mm512_unpacklo_ps(zmm0, zmm1); \
	Packet16f T1 = _mm512_unpackhi_ps(zmm0, zmm1); \
	Packet16f T2 = _mm512_unpacklo_ps(zmm2, zmm3); \
	Packet16f T3 = _mm512_unpackhi_ps(zmm2, zmm3); \
	Packet16f T4 = _mm512_unpacklo_ps(zmm4, zmm5); \
	Packet16f T5 = _mm512_unpackhi_ps(zmm4, zmm5); \
	Packet16f T6 = _mm512_unpacklo_ps(zmm6, zmm7); \
	Packet16f T7 = _mm512_unpackhi_ps(zmm6, zmm7); \
	\
	zmm0 = _mm512_shuffle_ps(T0, T2, 0x44); \
	zmm1 = _mm512_shuffle_ps(T0, T2, 0xEE); \
	zmm2 = _mm512_shuffle_ps(T1, T3, 0x44); \
	zmm3 = _mm512_shuffle_ps(T1, T3, 0xEE); \
	zmm4 = _mm512_shuffle_ps(T4, T6, 0x44); \
	zmm5 = _mm512_shuffle_ps(T4, T6, 0xEE); \
	zmm6 = _mm512_shuffle_ps(T5, T7, 0x44); \
	zmm7 = _mm512_shuffle_ps(T5, T7, 0xEE); \
	\
	T0 = _mm512_shuffle_f32x4(zmm0, zmm4, 0x88); \
	T1 = _mm512_shuffle_f32x4(zmm1, zmm5, 0x88); \
	T2 = _mm512_shuffle_f32x4(zmm2, zmm6, 0x88); \
	T3 = _mm512_shuffle_f32x4(zmm3, zmm7, 0x88); \
	T4 = _mm512_shuffle_f32x4(zmm0, zmm4, 0xDD); \
	T5 = _mm512_shuffle_f32x4(zmm1, zmm5, 0xDD); \
	T6 = _mm512_shuffle_f32x4(zmm2, zmm6, 0xDD); \
	T7 = _mm512_shuffle_f32x4(zmm3, zmm7, 0xDD); \
	\
	zmm0 = _mm512_shuffle_f32x4(T0, T1, 0x88); \
	zmm1 = _mm512_shuffle_f32x4(T2, T3, 0x88); \
	zmm2 = _mm512_shuffle_f32x4(T4, T5, 0x88); \
	zmm3 = _mm512_shuffle_f32x4(T6, T7, 0x88); \
	zmm4 = _mm512_shuffle_f32x4(T0, T1, 0xDD); \
	zmm5 = _mm512_shuffle_f32x4(T2, T3, 0xDD); \
	zmm6 = _mm512_shuffle_f32x4(T4, T5, 0xDD); \
	zmm7 = _mm512_shuffle_f32x4(T6, T7, 0xDD); \
}

// --------------------------------------------------------------------
// reduce add
// --------------------------------------------------------------------
#define preduceadd_8x8d(zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7, to) \
{ \
	Packet8d T0 = _mm512_shuffle_pd(zmm0, zmm1, 0x00); \
	Packet8d T1 = _mm512_shuffle_pd(zmm0, zmm1, 0xFF); \
	Packet8d T2 = _mm512_shuffle_pd(zmm2, zmm3, 0x00); \
	Packet8d T3 = _mm512_shuffle_pd(zmm2, zmm3, 0xFF); \
	Packet8d T4 = _mm512_shuffle_pd(zmm4, zmm5, 0x00); \
	Packet8d T5 = _mm512_shuffle_pd(zmm4, zmm5, 0xFF); \
	Packet8d T6 = _mm512_shuffle_pd(zmm6, zmm7, 0x00); \
	Packet8d T7 = _mm512_shuffle_pd(zmm6, zmm7, 0xFF); \
	T0 = _mm512_add_pd(T0, T1); \
	T2 = _mm512_add_pd(T2, T3); \
	T4 = _mm512_add_pd(T4, T5); \
	T6 = _mm512_add_pd(T6, T7); \
	\
	zmm0 = _mm512_permutex_pd(T2, 0x4E); \
	zmm0 = _mm512_mask_blend_pd(0xCC, T0, zmm0); \
	zmm2 = _mm512_permutex_pd(T0, 0x4E); \
	zmm2 = _mm512_mask_blend_pd(0xCC, zmm2, T2); \
	zmm4 = _mm512_permutex_pd(T6, 0x4E); \
	zmm4 = _mm512_mask_blend_pd(0xCC, T4, zmm4); \
	zmm6 = _mm512_permutex_pd(T4, 0x4E); \
	zmm6 = _mm512_mask_blend_pd(0xCC, zmm6, T6); \
	zmm0 = _mm512_add_pd(zmm0, zmm2); \
	zmm4 = _mm512_add_pd(zmm4, zmm6); \
	\
	T0 = _mm512_shuffle_f64x2(zmm0, zmm4, 0x44); \
	T4 = _mm512_shuffle_f64x2(zmm0, zmm4, 0xEE); \
	to = _mm512_add_pd(T0, T4); \
}
#define preduceadd_4x8d(zmm0, zmm1, zmm2, zmm3, to) \
{ \
	Packet8d T0 = _mm512_shuffle_pd(zmm0, zmm1, 0x00); \
	Packet8d T1 = _mm512_shuffle_pd(zmm0, zmm1, 0xFF); \
	Packet8d T2 = _mm512_shuffle_pd(zmm2, zmm3, 0x00); \
	Packet8d T3 = _mm512_shuffle_pd(zmm2, zmm3, 0xFF); \
	T0 = _mm512_add_pd(T0, T1); \
	T2 = _mm512_add_pd(T2, T3); \
	\
	zmm0 = _mm512_permutex_pd(T2, 0x4E); \
	zmm0 = _mm512_mask_blend_pd(0xCC, T0, zmm0); \
	zmm2 = _mm512_permutex_pd(T0, 0x4E); \
	zmm2 = _mm512_mask_blend_pd(0xCC, zmm2, T2); \
	zmm0 = _mm512_add_pd(zmm0, zmm2); \
	\
	T0 = _mm512_shuffle_f64x2(zmm0, pzero_8d, 0x44); \
	T1 = _mm512_shuffle_f64x2(zmm0, pzero_8d, 0xEE); \
	to = _mm512_add_pd(T0, T1); \
}
#define preduceadd_8x16f(zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7, to) \
{ \
	Packet16f T0 = _mm512_unpacklo_ps(zmm0, zmm1); \
	Packet16f T1 = _mm512_unpackhi_ps(zmm0, zmm1); \
	Packet16f T2 = _mm512_unpacklo_ps(zmm2, zmm3); \
	Packet16f T3 = _mm512_unpackhi_ps(zmm2, zmm3); \
	Packet16f T4 = _mm512_unpacklo_ps(zmm4, zmm5); \
	Packet16f T5 = _mm512_unpackhi_ps(zmm4, zmm5); \
	Packet16f T6 = _mm512_unpacklo_ps(zmm6, zmm7); \
	Packet16f T7 = _mm512_unpackhi_ps(zmm6, zmm7); \
	T0 = _mm512_add_ps(T0, T1); \
	T2 = _mm512_add_ps(T2, T3); \
	T4 = _mm512_add_ps(T4, T5); \
	T6 = _mm512_add_ps(T6, T7); \
	\
	zmm0 = _mm512_shuffle_ps(T0, T2, 0x44); \
	zmm1 = _mm512_shuffle_ps(T0, T2, 0xEE); \
	zmm4 = _mm512_shuffle_ps(T4, T6, 0x44); \
	zmm5 = _mm512_shuffle_ps(T4, T6, 0xEE); \
	zmm0 = _mm512_add_ps(zmm0, zmm1); \
	zmm4 = _mm512_add_ps(zmm4, zmm5); \
	\
	T0 = _mm512_shuffle_f32x4(zmm0, zmm4, 0x88); \
	T4 = _mm512_shuffle_f32x4(zmm0, zmm4, 0xDD); \
	T0 = _mm512_add_ps(T0, T4); \
	\
	zmm0 = _mm512_shuffle_f32x4(T0, T0, 0x88); \
	zmm4 = _mm512_shuffle_f32x4(T0, T0, 0xDD); \
	to = _mm512_maskz_add_ps(0x00FF, zmm0, zmm4); \
}

// --------------------------------------------------------------------
// argmax
// --------------------------------------------------------------------
#define pargmax_8d( zmm_x, zmm_arg, max_arg ) \
{ \
	__m256d ymm_x0, ymm_x1; \
	__m128d xmm_x0, xmm_x1; \
	__m256i ymm_arg; \
	__m128i xmm_arg; \
	__mmask8 lt_mask; \
	ymm_x0    = _mm512_castpd512_pd256( zmm_x ); \
	ymm_x1    = _mm512_extractf64x4_pd( zmm_x, 1 ); \
	ymm_arg   = _mm512_castsi512_si256( zmm_arg ); \
	lt_mask   = _mm256_cmplt_epu64_mask( _mm256_castpd_si256(ymm_x0), _mm256_castpd_si256(ymm_x1) ); \
	ymm_x0    = _mm512_mask_extractf64x4_pd( ymm_x0, lt_mask, zmm_x, 1 ); \
	ymm_arg   = _mm512_mask_extracti64x4_epi64( ymm_arg, lt_mask, zmm_arg, 1 ); \
	\
	xmm_x0    = _mm256_castpd256_pd128( ymm_x0 ); \
	xmm_x1    = _mm256_extractf64x2_pd( ymm_x0, 1 ); \
	xmm_arg   = _mm256_castsi256_si128( ymm_arg ); \
	lt_mask   = _mm_cmplt_epu64_mask( _mm_castpd_si128(xmm_x0), _mm_castpd_si128(xmm_x1) ); \
	xmm_x0    = _mm256_mask_extractf64x2_pd( xmm_x0, lt_mask, ymm_x0, 1 ); \
	xmm_arg   = _mm256_mask_extracti64x2_epi64( xmm_arg, lt_mask, ymm_arg, 1 ); \
	\
	xmm_x1    = _mm_permute_pd( xmm_x0, 0x01 ); \
	lt_mask   = _mm_cmplt_epu64_mask( _mm_castpd_si128(xmm_x0), _mm_castpd_si128(xmm_x1) ); \
	if ( lt_mask&0x01 ) \
		max_arg = _mm_extract_epi64( xmm_arg, 1 ); \
	else \
		max_arg = _mm_extract_epi64( xmm_arg, 0 ); \
}
#define pargmax_16f( zmm_x, zmm_arg, max_arg ) \
{ \
	__m256 ymm_x0, ymm_x1; \
	__m128 xmm_x0, xmm_x1; \
	__m256i ymm_arg; \
	__m128i xmm_arg; \
	__mmask16 lt16; \
	__mmask8  lt8; \
	ymm_x0    = _mm512_castps512_ps256( zmm_x ); \
	ymm_x1    = _mm512_extractf32x8_ps( zmm_x, 1 ); \
	ymm_arg   = _mm512_castsi512_si256( zmm_arg ); \
	lt16      = _mm256_cmplt_epu32_mask( _mm256_castps_si256(ymm_x0), _mm256_castps_si256(ymm_x1) ); \
	ymm_x0    = _mm512_mask_extractf32x8_ps( ymm_x0, lt16, zmm_x, 1 ); \
	ymm_arg   = _mm512_mask_extracti32x8_epi32( ymm_arg, lt16, zmm_arg, 1 ); \
	\
	xmm_x0    = _mm256_castps256_ps128( ymm_x0 ); \
	xmm_x1    = _mm256_extractf32x4_ps( ymm_x0, 1 ); \
	xmm_arg   = _mm256_castsi256_si128( ymm_arg ); \
	lt8       = _mm_cmplt_epu32_mask( _mm_castps_si128(xmm_x0), _mm_castps_si128(xmm_x1) ); \
	xmm_x0    = _mm256_mask_extractf32x4_ps( xmm_x0, lt8, ymm_x0, 1 ); \
	xmm_arg   = _mm256_mask_extracti32x4_epi32( xmm_arg, lt8, ymm_arg, 1 ); \
	\
	xmm_x1    = _mm_permute_ps( xmm_x0, 0x0E ); \
	lt8       = _mm_cmplt_epu32_mask( _mm_castps_si128(xmm_x0), _mm_castps_si128(xmm_x1) ); \
	xmm_x0    = _mm_mask_permute_ps( xmm_x0, lt8, xmm_x0, 0x0E ); \
	xmm_arg   = _mm_mask_shuffle_epi32( xmm_arg, lt8, xmm_arg, 0x0E ); \
	\
	xmm_x1    = _mm_permute_ps( xmm_x0, 0x01 ); \
	lt8       = _mm_cmplt_epu32_mask( _mm_castps_si128(xmm_x0), _mm_castps_si128(xmm_x1) ); \
	xmm_arg   = _mm_mask_shuffle_epi32( xmm_arg, lt8, xmm_arg, 0x01 ); \
	max_arg   = _mm_cvtsi128_si32( xmm_arg ); \
}

// --------------------------------------------------------------------
// process control
// --------------------------------------------------------------------
#define div_up(a, b) ((a-1)/(b)+1)

// handle m-dim edges <= 128
#define m_powedges(MR, x, EdgeFunc, ...) \
{ \
	int x_i = x; \
	if ( MR < x_i) { \
		for ( ; x_i > MR; x_i-= MR ) { \
			EdgeFunc( MR, __VA_ARGS__ ); \
		} \
	} \
	\
	if ( MR == x_i ) { \
		EdgeFunc( MR, __VA_ARGS__ ); \
	} \
	else { \
		if (  64 <= MR && x_i& 64 ) EdgeFunc(  64, __VA_ARGS__ ); \
		if (  32 <= MR && x_i& 32 ) EdgeFunc(  32, __VA_ARGS__ ); \
		if (  16 <= MR && x_i& 16 ) EdgeFunc(  16, __VA_ARGS__ ); \
		if (   8 <= MR && x_i&  8 ) EdgeFunc(   8, __VA_ARGS__ ); \
		if (   4 <= MR && x_i&  4 ) EdgeFunc(   4, __VA_ARGS__ ); \
		if (   2 <= MR && x_i&  2 ) EdgeFunc(   2, __VA_ARGS__ ); \
		if (   1 <= MR && x_i&  1 ) EdgeFunc(   1, __VA_ARGS__ ); \
	}\
}
// handle n-dim edges <= 16
#define n_powedges(NR, x, EdgeFunc, ...) \
{ \
	int x_i = x; \
	if ( NR < x_i ) { \
		for (; x_i > NR; x_i-=NR) { \
			EdgeFunc(NR, __VA_ARGS__); \
		} \
	} \
	\
	if ( NR==x_i ) { \
		EdgeFunc(NR, __VA_ARGS__); \
	} \
	else { \
		if ( 8 <= NR && x_i& 8 ) EdgeFunc( 8, __VA_ARGS__ ); \
		if ( 4 <= NR && x_i& 4 ) EdgeFunc( 4, __VA_ARGS__ ); \
		if ( 2 <= NR && x_i& 2 ) EdgeFunc( 2, __VA_ARGS__ ); \
		if ( 1 <= NR && x_i& 1 ) EdgeFunc( 1, __VA_ARGS__ ); \
	}\
}
// handle k-dim edges <= 8
#define k_alignedges(KA, x, IterFunc, ...) \
{ \
	int x_i = x; \
	if ( KA < x_i ) { \
		for ( ; x_i > KA; x_i-=KA ) { \
			IterFunc( KA,  __VA_ARGS__ ); \
		} \
	} \
	if ( KA == x_i )  { \
		IterFunc( KA, __VA_ARGS__ ); \
	} \
	else { \
		IterFunc( x_i, __VA_ARGS__); \
	} \
}

#define m_packloop(Pack, x, IterFunc, ...) \
{ \
	int mp_ = 0; \
	if (64 <= Pack && x&64) { \
		IterFunc((mp_+ 0), __VA_ARGS__); IterFunc((mp_+ 1), __VA_ARGS__); \
		IterFunc((mp_+ 2), __VA_ARGS__); IterFunc((mp_+ 3), __VA_ARGS__); \
		IterFunc((mp_+ 4), __VA_ARGS__); IterFunc((mp_+ 5), __VA_ARGS__); \
		IterFunc((mp_+ 6), __VA_ARGS__); IterFunc((mp_+ 7), __VA_ARGS__); \
		IterFunc((mp_+ 8), __VA_ARGS__); IterFunc((mp_+ 9), __VA_ARGS__); \
		IterFunc((mp_+10), __VA_ARGS__); IterFunc((mp_+11), __VA_ARGS__); \
		IterFunc((mp_+12), __VA_ARGS__); IterFunc((mp_+13), __VA_ARGS__); \
		IterFunc((mp_+14), __VA_ARGS__); IterFunc((mp_+15), __VA_ARGS__); \
		IterFunc((mp_+16), __VA_ARGS__); IterFunc((mp_+17), __VA_ARGS__); \
		IterFunc((mp_+18), __VA_ARGS__); IterFunc((mp_+19), __VA_ARGS__); \
		IterFunc((mp_+20), __VA_ARGS__); IterFunc((mp_+21), __VA_ARGS__); \
		IterFunc((mp_+22), __VA_ARGS__); IterFunc((mp_+23), __VA_ARGS__); \
		IterFunc((mp_+24), __VA_ARGS__); IterFunc((mp_+25), __VA_ARGS__); \
		IterFunc((mp_+26), __VA_ARGS__); IterFunc((mp_+27), __VA_ARGS__); \
		IterFunc((mp_+28), __VA_ARGS__); IterFunc((mp_+29), __VA_ARGS__); \
		IterFunc((mp_+30), __VA_ARGS__); IterFunc((mp_+31), __VA_ARGS__); \
		IterFunc((mp_+32), __VA_ARGS__); IterFunc((mp_+33), __VA_ARGS__); \
		IterFunc((mp_+34), __VA_ARGS__); IterFunc((mp_+35), __VA_ARGS__); \
		IterFunc((mp_+36), __VA_ARGS__); IterFunc((mp_+37), __VA_ARGS__); \
		IterFunc((mp_+38), __VA_ARGS__); IterFunc((mp_+39), __VA_ARGS__); \
		IterFunc((mp_+40), __VA_ARGS__); IterFunc((mp_+41), __VA_ARGS__); \
		IterFunc((mp_+42), __VA_ARGS__); IterFunc((mp_+43), __VA_ARGS__); \
		IterFunc((mp_+44), __VA_ARGS__); IterFunc((mp_+45), __VA_ARGS__); \
		IterFunc((mp_+46), __VA_ARGS__); IterFunc((mp_+47), __VA_ARGS__); \
		IterFunc((mp_+48), __VA_ARGS__); IterFunc((mp_+49), __VA_ARGS__); \
		IterFunc((mp_+50), __VA_ARGS__); IterFunc((mp_+51), __VA_ARGS__); \
		IterFunc((mp_+52), __VA_ARGS__); IterFunc((mp_+53), __VA_ARGS__); \
		IterFunc((mp_+54), __VA_ARGS__); IterFunc((mp_+55), __VA_ARGS__); \
		IterFunc((mp_+56), __VA_ARGS__); IterFunc((mp_+57), __VA_ARGS__); \
		IterFunc((mp_+58), __VA_ARGS__); IterFunc((mp_+59), __VA_ARGS__); \
		IterFunc((mp_+60), __VA_ARGS__); IterFunc((mp_+61), __VA_ARGS__); \
		IterFunc((mp_+62), __VA_ARGS__); IterFunc((mp_+63), __VA_ARGS__); \
		mp_ += 64; \
	} \
	if (32 <= Pack && x&32) { \
		IterFunc((mp_+ 0), __VA_ARGS__); IterFunc((mp_+ 1), __VA_ARGS__); \
		IterFunc((mp_+ 2), __VA_ARGS__); IterFunc((mp_+ 3), __VA_ARGS__); \
		IterFunc((mp_+ 4), __VA_ARGS__); IterFunc((mp_+ 5), __VA_ARGS__); \
		IterFunc((mp_+ 6), __VA_ARGS__); IterFunc((mp_+ 7), __VA_ARGS__); \
		IterFunc((mp_+ 8), __VA_ARGS__); IterFunc((mp_+ 9), __VA_ARGS__); \
		IterFunc((mp_+10), __VA_ARGS__); IterFunc((mp_+11), __VA_ARGS__); \
		IterFunc((mp_+12), __VA_ARGS__); IterFunc((mp_+13), __VA_ARGS__); \
		IterFunc((mp_+14), __VA_ARGS__); IterFunc((mp_+15), __VA_ARGS__); \
		IterFunc((mp_+16), __VA_ARGS__); IterFunc((mp_+17), __VA_ARGS__); \
		IterFunc((mp_+18), __VA_ARGS__); IterFunc((mp_+19), __VA_ARGS__); \
		IterFunc((mp_+20), __VA_ARGS__); IterFunc((mp_+21), __VA_ARGS__); \
		IterFunc((mp_+22), __VA_ARGS__); IterFunc((mp_+23), __VA_ARGS__); \
		IterFunc((mp_+24), __VA_ARGS__); IterFunc((mp_+25), __VA_ARGS__); \
		IterFunc((mp_+26), __VA_ARGS__); IterFunc((mp_+27), __VA_ARGS__); \
		IterFunc((mp_+28), __VA_ARGS__); IterFunc((mp_+29), __VA_ARGS__); \
		IterFunc((mp_+30), __VA_ARGS__); IterFunc((mp_+31), __VA_ARGS__); \
		mp_ += 32; \
	} \
	if (16 <= Pack && x&16) { \
		IterFunc((mp_+ 0), __VA_ARGS__); IterFunc((mp_+ 1), __VA_ARGS__); \
		IterFunc((mp_+ 2), __VA_ARGS__); IterFunc((mp_+ 3), __VA_ARGS__); \
		IterFunc((mp_+ 4), __VA_ARGS__); IterFunc((mp_+ 5), __VA_ARGS__); \
		IterFunc((mp_+ 6), __VA_ARGS__); IterFunc((mp_+ 7), __VA_ARGS__); \
		IterFunc((mp_+ 8), __VA_ARGS__); IterFunc((mp_+ 9), __VA_ARGS__); \
		IterFunc((mp_+10), __VA_ARGS__); IterFunc((mp_+11), __VA_ARGS__); \
		IterFunc((mp_+12), __VA_ARGS__); IterFunc((mp_+13), __VA_ARGS__); \
		IterFunc((mp_+14), __VA_ARGS__); IterFunc((mp_+15), __VA_ARGS__); \
		mp_ += 16; \
	} \
	if ( 8 <= Pack && x& 8) {\
		IterFunc((mp_+ 0), __VA_ARGS__); \
		IterFunc((mp_+ 1), __VA_ARGS__); \
		IterFunc((mp_+ 2), __VA_ARGS__); \
		IterFunc((mp_+ 3), __VA_ARGS__); \
		IterFunc((mp_+ 4), __VA_ARGS__); \
		IterFunc((mp_+ 5), __VA_ARGS__); \
		IterFunc((mp_+ 6), __VA_ARGS__); \
		IterFunc((mp_+ 7), __VA_ARGS__); \
		mp_ += 8; \
	} \
	if ( 4 <= Pack && x& 4) { \
		IterFunc((mp_+ 0), __VA_ARGS__); \
		IterFunc((mp_+ 1), __VA_ARGS__); \
		IterFunc((mp_+ 2), __VA_ARGS__); \
		IterFunc((mp_+ 3), __VA_ARGS__); \
		mp_ += 4; \
	} \
	if ( 2 <= Pack && x& 2) { \
		IterFunc((mp_+ 0), __VA_ARGS__); \
		IterFunc((mp_+ 1), __VA_ARGS__); \
		mp_ += 2; \
	} \
	if ( 1 <= Pack && x& 1) { \
		IterFunc((mp_+ 0), __VA_ARGS__); \
		mp_ += 1; \
	} \
}
#define n_packloop(Pack, x, IterFunc, ...) \
{ \
	int np_ = 0; \
	if (16 <= Pack && x&16) { \
		IterFunc((np_+ 0), __VA_ARGS__); IterFunc((np_+ 1), __VA_ARGS__); \
		IterFunc((np_+ 2), __VA_ARGS__); IterFunc((np_+ 3), __VA_ARGS__); \
		IterFunc((np_+ 4), __VA_ARGS__); IterFunc((np_+ 5), __VA_ARGS__); \
		IterFunc((np_+ 6), __VA_ARGS__); IterFunc((np_+ 7), __VA_ARGS__); \
		IterFunc((np_+ 8), __VA_ARGS__); IterFunc((np_+ 9), __VA_ARGS__); \
		IterFunc((np_+10), __VA_ARGS__); IterFunc((np_+11), __VA_ARGS__); \
		IterFunc((np_+12), __VA_ARGS__); IterFunc((np_+13), __VA_ARGS__); \
		IterFunc((np_+14), __VA_ARGS__); IterFunc((np_+15), __VA_ARGS__); \
		np_ += 16; \
	} \
	if ( 8 <= Pack && x& 8) {\
		IterFunc((np_+ 0), __VA_ARGS__); \
		IterFunc((np_+ 1), __VA_ARGS__); \
		IterFunc((np_+ 2), __VA_ARGS__); \
		IterFunc((np_+ 3), __VA_ARGS__); \
		IterFunc((np_+ 4), __VA_ARGS__); \
		IterFunc((np_+ 5), __VA_ARGS__); \
		IterFunc((np_+ 6), __VA_ARGS__); \
		IterFunc((np_+ 7), __VA_ARGS__); \
		np_ += 8; \
	} \
	if ( 4 <= Pack && x& 4) { \
		IterFunc((np_+ 0), __VA_ARGS__); \
		IterFunc((np_+ 1), __VA_ARGS__); \
		IterFunc((np_+ 2), __VA_ARGS__); \
		IterFunc((np_+ 3), __VA_ARGS__); \
		np_ += 4; \
	} \
	if ( 2 <= Pack && x& 2) { \
		IterFunc((np_+ 0), __VA_ARGS__); \
		IterFunc((np_+ 1), __VA_ARGS__); \
		np_ += 2; \
	} \
	if ( 1 <= Pack && x& 1) { \
		IterFunc((np_+ 0), __VA_ARGS__); \
		np_ += 1; \
	} \
}

#define k_packloop(Pack, x, IterFunc, ...) \
{ \
	int kp_ = 0; \
	if (16 <= Pack && x&16) { \
		IterFunc((kp_+ 0), __VA_ARGS__); IterFunc((kp_+ 1), __VA_ARGS__); \
		IterFunc((kp_+ 2), __VA_ARGS__); IterFunc((kp_+ 3), __VA_ARGS__); \
		IterFunc((kp_+ 4), __VA_ARGS__); IterFunc((kp_+ 5), __VA_ARGS__); \
		IterFunc((kp_+ 6), __VA_ARGS__); IterFunc((kp_+ 7), __VA_ARGS__); \
		IterFunc((kp_+ 8), __VA_ARGS__); IterFunc((kp_+ 9), __VA_ARGS__); \
		IterFunc((kp_+10), __VA_ARGS__); IterFunc((kp_+11), __VA_ARGS__); \
		IterFunc((kp_+12), __VA_ARGS__); IterFunc((kp_+13), __VA_ARGS__); \
		IterFunc((kp_+14), __VA_ARGS__); IterFunc((kp_+15), __VA_ARGS__); \
		kp_ += 16; \
	} \
	if ( 8 <= Pack && x& 8) {\
		IterFunc((kp_+ 0), __VA_ARGS__); \
		IterFunc((kp_+ 1), __VA_ARGS__); \
		IterFunc((kp_+ 2), __VA_ARGS__); \
		IterFunc((kp_+ 3), __VA_ARGS__); \
		IterFunc((kp_+ 4), __VA_ARGS__); \
		IterFunc((kp_+ 5), __VA_ARGS__); \
		IterFunc((kp_+ 6), __VA_ARGS__); \
		IterFunc((kp_+ 7), __VA_ARGS__); \
		kp_ += 8; \
	} \
	if ( 4 <= Pack && x& 4) { \
		IterFunc((kp_+ 0), __VA_ARGS__); \
		IterFunc((kp_+ 1), __VA_ARGS__); \
		IterFunc((kp_+ 2), __VA_ARGS__); \
		IterFunc((kp_+ 3), __VA_ARGS__); \
		kp_ += 4; \
	} \
	if ( 2 <= Pack && x& 2) { \
		IterFunc((kp_+ 0), __VA_ARGS__); \
		IterFunc((kp_+ 1), __VA_ARGS__); \
		kp_ += 2; \
	} \
	if ( 1 <= Pack && x& 1) { \
		IterFunc((kp_+ 0), __VA_ARGS__); \
		kp_ += 1; \
	} \
}
// --------------------------------------------------------------------
// misc.
// --------------------------------------------------------------------
// zmm
#define print_8d(fmt, a) \
{ \
	double mem[8]; \
	pstoreu_8d(mem, (Packet8d)(a)); \
	for (int i = 0; i < 8; i++) printf(fmt, mem[i]); \
	printf("\n"); \
}
#define print_8l(fmt, a) \
{ \
	__int64_t mem[8]; \
	pstoreu_8l( mem, a ); \
	for (int i = 0; i < 8; i++) printf(fmt, (int)mem[i]); \
	printf("\n"); \
}
#define print_16f(fmt, a) \
{ \
	float mem[16]; \
	pstoreu_16f(mem, (Packet16f)(a)); \
	for (int i = 0; i < 16; i++) printf(fmt, mem[i]); \
	printf("\n"); \
}
#define print_16i(fmt, a) \
{ \
	__int32_t mem[16]; \
	pstoreu_16i(mem, (Packet16i)(a)); \
	for (int i = 0; i < 16; i++) printf(fmt, (int)mem[i]); \
	printf("\n"); \
}

// ymm
#define print_4d(fmt, a) \
{ \
	double mem[4]; \
	_mm256_store_pd( mem, a ); \
	for (int i = 0; i < 4; i++) printf(fmt, mem[i]); \
	printf("\n"); \
}
#define print_4l(fmt, a) \
{ \
	__int64_t mem[4]; \
	_mm256_store_epi64( mem, a ); \
	for (int i = 0; i < 4; i++) printf(fmt, (int)mem[i]); \
	printf("\n"); \
}
#define print_8f(fmt, a) \
{ \
	float mem[8]; \
	_mm256_store_ps(mem, (Packet8f)(a)); \
	for (int i = 0; i < 8; i++) printf(fmt, mem[i]); \
	printf("\n"); \
}
#define print_8i(fmt, a) \
{ \
	__int32_t mem[8]; \
	_mm256_store_epi32(mem, (Packet8i)(a)); \
	for (int i = 0; i < 8; i++) printf(fmt, (int)mem[i]); \
	printf("\n"); \
}

// xmm
#define print_2d(fmt, a) \
{ \
	double mem[2]; \
	_mm_store_pd( mem, a ); \
	for (int i = 0; i < 2; i++) printf(fmt, mem[i]); \
	printf("\n"); \
}
#define print_2l(fmt, a) \
{ \
	__int64_t mem[2]; \
	_mm_store_epi64( mem, a ); \
	for (int i = 0; i < 2; i++) printf(fmt, (int)mem[i]); \
	printf("\n"); \
}
#define print_4f(fmt, a) \
{ \
	float mem[4]; \
	_mm_store_ps(mem, (Packet4f)(a)); \
	for (int i = 0; i < 4; i++) printf(fmt, mem[i]); \
	printf("\n"); \
}
#define print_4i(fmt, a) \
{ \
	__int32_t mem[4]; \
	_mm_store_epi32(mem, (Packet4i)(a)); \
	for (int i = 0; i < 4; i++) printf(fmt, (int)mem[i]); \
	printf("\n"); \
}

#endif
