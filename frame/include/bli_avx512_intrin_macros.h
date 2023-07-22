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

typedef const char * SsePrefetchPtrType;

// --------------------------------------------------------------------
// Initial value set
// --------------------------------------------------------------------
#define pzero_16f (Packet16f)_mm512_setzero_ps()
#define pzero_8d  (Packet8d )_mm512_setzero_pd()

#define pset1_16f(from) (Packet16f)_mm512_set1_ps((float )(from))
#define pset1_8d(from)  (Packet8d )_mm512_set1_pd((double)(from))
#define pset1_16i(from) (Packet16i)_mm512_set1_epi32((int)(from))

#define edge_mask8(I_)  (( 8 <= I_)? 0x00FF : ((1<<(I_))-1))
#define edge_mask16(I_) ((16 <= I_)? 0xFFFF : ((1<<(I_))-1))

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
// Aligned (masked) load
#define pload_16f(from, to) \
	to = _mm512_load_ps(( float*)(from))
#define pload_8d(from, to) \
	to = _mm512_load_pd((double*)(from))
#define pmload_16f(from, to, mask) \
	to = _mm512_mask_load_ps((Packet16f)(to), (__mmask16)(mask), ( float*)(from))
#define pmload_8d(from, mask) \
	to = _mm512_mask_load_pd((Packet8d )(to), (__mmask8 )(mask), (double*)(from))
#define pmzload_16f(from, to, mask) \
	to = _mm512_maskz_load_ps((__mmask16)(mask), ( float*)(from))
#define pmzload_8d(from, mask) \
	to = _mm512_maskz_load_pd((__mmask8 )(mask), (double*)(from))

// Unaligned (masked) load
#define ploadu_16f(from, to) \
	to = _mm512_loadu_ps(( float*)(from))
#define ploadu_8d(from, to) \
	to = _mm512_loadu_pd((double*)(from))
#define pmloadu_16f(from, to, mask) \
	to = _mm512_mask_loadu_ps((Packet16f)(to), (__mmask16)(mask), ( float*)(from))
#define pmloadu_8d(from, to, mask) \
	to = _mm512_mask_loadu_pd((Packet8d )(to), (__mmask8 )(mask), (double*)(from))
#define pmzloadu_16f(from, to, mask) \
	to = _mm512_maskz_loadu_ps((__mmask16)(mask), ( float*)(from))
#define pmzloadu_8d(from, to, mask) \
	to = _mm512_maskz_loadu_pd((__mmask8 )(mask), (double*)(from))

// strided (masked) load
#define ploads_16f(from, to, stride) \
{ \
	Packet16i stride_vector = _mm512_set1_epi32(stride); \
	Packet16i stride_multiplier = \
		_mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0); \
	Packet16i indices = _mm512_mullo_epi32(stride_vector, stride_multiplier); \
	to = _mm512_i32gather_ps(indices, from, 4); \
}
#define ploads_8d(from, to, stride) \
{ \
	Packet8i stride_vector = _mm256_set1_epi32(stride); \
	Packet8i stride_multiplier = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0); \
	Packet8i indices = _mm256_mullo_epi32(stride_vector, stride_multiplier); \
	to = _mm512_i32gather_pd(indices, from, 8); \
}
#define pmloads_16f(from, to, stride, mask) \
{ \
	Packet16i stride_vector = _mm512_set1_epi32(stride); \
	Packet16i stride_multiplier = \
		_mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0); \
	Packet16i indices = _mm512_mullo_epi32(stride_vector, stride_multiplier); \
	to = _mm512_mask_i32gather_ps(to, mask, indices, from, 4); \
}
#define pmloads_8d(from, to, stride, mask) \
{ \
	Packet8i stride_vector = _mm256_set1_epi32(stride); \
	Packet8i stride_multiplier = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0); \
	Packet8i indices = _mm256_mullo_epi32(stride_vector, stride_multiplier); \
	to = _mm512_mask_i32gather_pd(to, mask, indices, from, 8); \
}
#define pmzloads_16f(from, to, stride, mask) \
{ \
	Packet16i stride_vector = _mm512_set1_epi32(stride); \
	Packet16i stride_multiplier = \
		_mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0); \
	Packet16i indices = _mm512_mullo_epi32(stride_vector, stride_multiplier); \
	to = _mm512_setzero_ps(); \
	to = _mm512_mask_i32gather_ps(to, mask, indices, from, 4); \
}
#define pmzloads_8d(from, to, stride, mask) \
{ \
	Packet8i stride_vector = _mm256_set1_epi32(stride); \
	Packet8i stride_multiplier = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0); \
	Packet8i indices = _mm256_mullo_epi32(stride_vector, stride_multiplier); \
	to = _mm512_setzero_pd(); \
	to = _mm512_mask_i32gather_pd(to, mask, indices, from, 8); \
}


// load1
#define pload1_16f(from) (Packet16f) _mm512_set1_ps(*(( float* )(from)))
#define pload1_8d(from)  (Packet8d ) _mm512_set1_pd(*((double*)(from)))

// --------------------------------------------------------------------
// Store to mem
// --------------------------------------------------------------------
// Aligned (masked) store
#define pstore_16f(to, from) \
	_mm512_store_ps(( float*)(to), (Packet16f)(from))
#define pstore_8d(to, from) \
	_mm512_store_pd((double*)(to), (Packet8d )(from))
#define pmstore_16f(to, from, mask) \
	_mm512_mask_store_ps(( float*)(to), (__mmask16)(mask), (Packet16f)(from))
#define pmstore_8d(to, from, mask) \
	_mm512_mask_store_pd((double*)(to), (__mmask8 )(mask), (Packet8d )(from))

// Unaligned (masked) store
#define pstoreu_16f(to, from) \
	_mm512_storeu_ps(( float*)(to), (Packet16f)(from))
#define pstoreu_8d(to, from) \
	_mm512_storeu_pd((double*)(to), (Packet8d )(from))
#define pmstoreu_16f(to, from, mask) \
	_mm512_mask_storeu_ps(( float*)(to), (__mmask16)(mask), (Packet16f)(from))
#define pmstoreu_8d(to, from, mask) \
	_mm512_mask_storeu_pd((double*)(to), (__mmask8 )(mask), (Packet8d )(from))

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
#define prefetch(addr) _mm_prefetch((SsePrefetchPtrType)(addr), _MM_HINT_T0)

// prefetch 1 cacheline
#define prefetch_8d_at(idx, addr, s_i) \
{ \
	prefetch(addr + (idx)*s_i*8); \
}
#define prefetch_16f_at(idx, addr, s_i) \
{ \
	prefetch(addr + (idx)*s_i*16); \
}
// prefetch 8 cachelines
#define prefetch_8x8d(addr, ld_x) \
{ \
	prefetch(addr + 0*ld_x*8); \
	prefetch(addr + 1*ld_x*8); \
	prefetch(addr + 2*ld_x*8); \
	prefetch(addr + 3*ld_x*8); \
	prefetch(addr + 4*ld_x*8); \
	prefetch(addr + 5*ld_x*8); \
	prefetch(addr + 6*ld_x*8); \
	prefetch(addr + 7*ld_x*8); \
}
#define prefetch_8x16f(addr, ld_x) \
{ \
	prefetch(addr + 0*ld_x*4); \
	prefetch(addr + 1*ld_x*4); \
	prefetch(addr + 2*ld_x*4); \
	prefetch(addr + 3*ld_x*4); \
	prefetch(addr + 4*ld_x*4); \
	prefetch(addr + 5*ld_x*4); \
	prefetch(addr + 6*ld_x*4); \
	prefetch(addr + 7*ld_x*4); \
}

// prefetch x cachelines
#define prefetch_x8d(x, addr, ld_x) \
{ \
	int x_i = x; \
	if (8 < x_i) { \
		for (; x_i > 8; x_i-=8) { \
			prefetch_8x8d(addr+(x-x_i)*ld_x*8, ld_x); \
		} \
	} \
	if (8 == x_i) { \
		prefetch_8x8d(addr+(x-x_i)*ld_x*8, ld_x); \
	} \
	else { \
		for (; x_i > 0; x_i--) { \
			prefetch(addr + (x-x_i)*ld_x*8); \
		}\
	} \
}
#define prefetch_x8d_at(idx, x, addr, ld_x, s_i) { \
	prefetch_x8d(x, addr+(idx)*s_i*8, ld_x); \
}

#define prefetch_x16f(x, addr, ld_x) \
{ \
	int x_i = x; \
	if (8 < x) { \
		for (; x_i > 8; x_i-=8) { \
			prefetch_8x16f(addr+(x-x_i)*ld_x*4, ld_x); \
		} \
	} \
	if (8==x_i) { \
		prefetch_8x16f(addr+(x-x_i)*ld_x*4, ld_x); \
	} \
	else { \
		for (; x_i >0; x_i--) { \
			prefetch(addr + (x-x_i)*ld_x*4); \
		} \
	} \
}
#define prefetch_x16f_at(idx, x, addr, ld_x, s_i) { \
	prefetch_x16f(x, addr+(idx)*s_i*4, ld_x); \
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
// process control
// --------------------------------------------------------------------
#define div_up(a, b) ((a-1)/(b)+1)

#define m_powedges_8d(MR, x, EdgeFunc, ...) \
{ \
	int x_i = x; \
	if (MR < x_i) { \
		for (; x_i > MR; x_i-= MR) { \
			EdgeFunc(MR, __VA_ARGS__); \
		} \
	} \
	if (MR==x_i) { \
		EdgeFunc(MR, __VA_ARGS__); \
	} \
	else { \
		if (16 <= MR && x_i&16) EdgeFunc(16, __VA_ARGS__); \
		if ( 8 <= MR && x_i& 8) EdgeFunc( 8, __VA_ARGS__); \
		if ( 4 <= MR && x_i& 4) EdgeFunc( 4, __VA_ARGS__); \
		if ( 2 <= MR && x_i& 2) EdgeFunc( 2, __VA_ARGS__); \
		if ( 1 <= MR && x_i& 1) EdgeFunc( 1, __VA_ARGS__); \
	}\
}
#define m_powedges_16f(MR, x, EdgeFunc, ...) \
{ \
	int x_i = x; \
	if (MR < x_i) { \
		for (; x_i > MR; x_i-=MR) { \
			EdgeFunc(MR, __VA_ARGS__); \
		} \
	} \
	if (MR==x_i) { \
		EdgeFunc(MR, __VA_ARGS__); \
	} \
	else { \
		if (32 <= MR && x_i&32) EdgeFunc(32, __VA_ARGS__); \
		if (16 <= MR && x_i&16) EdgeFunc(16, __VA_ARGS__); \
		if ( 8 <= MR && x_i& 8) EdgeFunc( 8, __VA_ARGS__); \
		if ( 4 <= MR && x_i& 4) EdgeFunc( 4, __VA_ARGS__); \
		if ( 2 <= MR && x_i& 2) EdgeFunc( 2, __VA_ARGS__); \
		if ( 1 <= MR && x_i& 1) EdgeFunc( 1, __VA_ARGS__); \
	}\
}
#define n_powedges_8d(NR, x, EdgeFunc, ...) \
{ \
	int x_i = x; \
	if (NR < x_i) { \
		for (; x_i > NR; x_i-=NR) { \
			EdgeFunc(NR, __VA_ARGS__); \
		} \
	} \
	if (NR==x_i) { \
		EdgeFunc(NR, __VA_ARGS__); \
	} \
	else { \
		if ( 4 <= NR && x_i& 4) EdgeFunc( 4, __VA_ARGS__); \
		if ( 2 <= NR && x_i& 2) EdgeFunc( 2, __VA_ARGS__); \
		if ( 1 <= NR && x_i& 1) EdgeFunc( 1, __VA_ARGS__); \
	}\
}
#define n_powedges_16f(NR, x, EdgeFunc, ...) \
{ \
	int x_i = x; \
	if (NR < x_i) { \
		for (; x_i > NR; x_i-=NR) { \
			EdgeFunc(NR, __VA_ARGS__); \
		} \
	} \
	if (NR==x_i) { \
		EdgeFunc(NR, __VA_ARGS__); \
	} \
	else { \
		if ( 4 <= NR && x_i& 4) EdgeFunc( 4, __VA_ARGS__); \
		if ( 2 <= NR && x_i& 2) EdgeFunc( 2, __VA_ARGS__); \
		if ( 1 <= NR && x_i& 1) EdgeFunc( 1, __VA_ARGS__); \
	}\
}
#define k_alignloop(x, IterFunc, ...) \
{ \
	int x_i = x; \
	if (8 < x_i) { \
		for (; x_i > 8; x_i-=8) { \
			IterFunc(8,  __VA_ARGS__); \
		} \
	} \
	if (8==x_i) { \
		IterFunc(8,  __VA_ARGS__); \
	} \
	else { \
		IterFunc(x_i, __VA_ARGS__); \
	} \
}

#define m_packloop(Pack, x, IterFunc, ...) \
{ \
	int mp_ = 0; \
	if (32 <= Pack && x&32) { \
		IterFunc((mp_+ 0), __VA_ARGS__); \
		IterFunc((mp_+ 1), __VA_ARGS__); \
		IterFunc((mp_+ 2), __VA_ARGS__); \
		IterFunc((mp_+ 3), __VA_ARGS__); \
		IterFunc((mp_+ 4), __VA_ARGS__); \
		IterFunc((mp_+ 5), __VA_ARGS__); \
		IterFunc((mp_+ 6), __VA_ARGS__); \
		IterFunc((mp_+ 7), __VA_ARGS__); \
		IterFunc((mp_+ 8), __VA_ARGS__); \
		IterFunc((mp_+ 9), __VA_ARGS__); \
		IterFunc((mp_+10), __VA_ARGS__); \
		IterFunc((mp_+11), __VA_ARGS__); \
		IterFunc((mp_+12), __VA_ARGS__); \
		IterFunc((mp_+13), __VA_ARGS__); \
		IterFunc((mp_+14), __VA_ARGS__); \
		IterFunc((mp_+15), __VA_ARGS__); \
		IterFunc((mp_+16), __VA_ARGS__); \
		IterFunc((mp_+17), __VA_ARGS__); \
		IterFunc((mp_+18), __VA_ARGS__); \
		IterFunc((mp_+19), __VA_ARGS__); \
		IterFunc((mp_+20), __VA_ARGS__); \
		IterFunc((mp_+21), __VA_ARGS__); \
		IterFunc((mp_+22), __VA_ARGS__); \
		IterFunc((mp_+23), __VA_ARGS__); \
		IterFunc((mp_+24), __VA_ARGS__); \
		IterFunc((mp_+25), __VA_ARGS__); \
		IterFunc((mp_+26), __VA_ARGS__); \
		IterFunc((mp_+27), __VA_ARGS__); \
		IterFunc((mp_+28), __VA_ARGS__); \
		IterFunc((mp_+29), __VA_ARGS__); \
		IterFunc((mp_+30), __VA_ARGS__); \
		IterFunc((mp_+31), __VA_ARGS__); \
		mp_ += 32; \
	} \
	if (16 <= Pack && x&16) { \
		IterFunc((mp_+ 0), __VA_ARGS__); \
		IterFunc((mp_+ 1), __VA_ARGS__); \
		IterFunc((mp_+ 2), __VA_ARGS__); \
		IterFunc((mp_+ 3), __VA_ARGS__); \
		IterFunc((mp_+ 4), __VA_ARGS__); \
		IterFunc((mp_+ 5), __VA_ARGS__); \
		IterFunc((mp_+ 6), __VA_ARGS__); \
		IterFunc((mp_+ 7), __VA_ARGS__); \
		IterFunc((mp_+ 8), __VA_ARGS__); \
		IterFunc((mp_+ 9), __VA_ARGS__); \
		IterFunc((mp_+10), __VA_ARGS__); \
		IterFunc((mp_+11), __VA_ARGS__); \
		IterFunc((mp_+12), __VA_ARGS__); \
		IterFunc((mp_+13), __VA_ARGS__); \
		IterFunc((mp_+14), __VA_ARGS__); \
		IterFunc((mp_+15), __VA_ARGS__); \
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
#define print_8d(fmt, a) \
{ \
	double mem[8]; \
	pstoreu_8d(mem, (Packet8d)(a)); \
	for (int i = 0; i < 8; i++) printf(fmt, mem[i]); \
	printf("\n"); \
}
#define print_16f(fmt, a) \
{ \
	float mem[16]; \
	pstoreu_16f(mem, (Packet16f)(a)); \
	for (int i = 0; i < 16; i++) printf(fmt, mem[i]); \
	printf("\n"); \
}

#endif
