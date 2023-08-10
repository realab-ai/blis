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

#include "blis.h"

void bli_cntx_init_skx( cntx_t* cntx )
{
	blksz_t blkszs[ BLIS_NUM_BLKSZS ];

	// Set default kernel blocksizes and functions.
	bli_cntx_init_skx_ref( cntx );

	// -------------------------------------------------------------------------

	// Update the context with optimized native gemm micro-kernels.
	bli_cntx_set_ukrs
	(
	 cntx,
	 
	 // level-3
	 BLIS_GEMM_UKR,     BLIS_FLOAT,     bli_sgemm_cv_skx_int_48x8,
	 BLIS_GEMM_UKR,     BLIS_DOUBLE,    bli_dgemm_cv_skx_int_24x8,

	 // axpyf
	 //BLIS_AXPYF_KER,   BLIS_FLOAT,     bli_saxpyf_zen_int_8,
	 //BLIS_AXPYF_KER,   BLIS_DOUBLE,    bli_daxpyf_zen_int_8,
	 BLIS_AXPYF_KER,    BLIS_FLOAT,     bli_saxpyf_skx_int_128,
	 BLIS_AXPYF_KER,    BLIS_DOUBLE,    bli_daxpyf_skx_int_64,

	 // dotxf
	 //BLIS_DOTXF_KER,   BLIS_FLOAT,     bli_sdotxf_zen_int_8,
	 //BLIS_DOTXF_KER,   BLIS_DOUBLE,    bli_ddotxf_zen_int_8,
	 BLIS_DOTXF_KER,    BLIS_FLOAT,     bli_sdotxf_skx_int_8,
	 BLIS_DOTXF_KER,    BLIS_DOUBLE,    bli_ddotxf_skx_int_8,

	 // amaxv
	 //BLIS_AMAXV_KER,   BLIS_FLOAT,	 bli_samaxv_zen_int,
	 //BLIS_AMAXV_KER,   BLIS_DOUBLE,    bli_damaxv_zen_int,
	 BLIS_AMAXV_KER,    BLIS_FLOAT,	    bli_samaxv_skx_int_16,
	 BLIS_AMAXV_KER,    BLIS_DOUBLE,    bli_damaxv_skx_int_8,

	 // axpyv
	 //BLIS_AXPYV_KER,   BLIS_FLOAT,	 bli_saxpyv_zen_int10,
	 //BLIS_AXPYV_KER,   BLIS_DOUBLE,    bli_daxpyv_zen_int10,
	 BLIS_AXPYV_KER,    BLIS_FLOAT,	    bli_saxpyv_skx_int_128,
	 BLIS_AXPYV_KER,    BLIS_DOUBLE,    bli_daxpyv_skx_int_64,

	 // dotv
	 //BLIS_DOTV_KER,	 BLIS_FLOAT,	 bli_sdotv_zen_int,
	 //BLIS_DOTV_KER,	 BLIS_DOUBLE,    bli_ddotv_zen_int,
	 BLIS_DOTV_KER,     BLIS_FLOAT,     bli_sdotv_skx_int_128,
	 BLIS_DOTV_KER,     BLIS_DOUBLE,    bli_ddotv_skx_int_64,

	 // dotxv
	 //BLIS_DOTXV_KER,    BLIS_FLOAT,	  bli_sdotxv_zen_int,
	 //BLIS_DOTXV_KER,    BLIS_DOUBLE,    bli_ddotxv_zen_int,
	 BLIS_DOTXV_KER,    BLIS_FLOAT,     bli_sdotxv_skx_int_128,
	 BLIS_DOTXV_KER,    BLIS_DOUBLE,    bli_ddotxv_skx_int_64,

	 // scalv
	 //BLIS_SCALV_KER,    BLIS_FLOAT,	  bli_sscalv_zen_int10,
	 //BLIS_SCALV_KER,    BLIS_DOUBLE,    bli_dscalv_zen_int10,
	 BLIS_SCALV_KER,    BLIS_FLOAT,     bli_sscalv_skx_int_128,
	 BLIS_SCALV_KER,    BLIS_DOUBLE,    bli_dscalv_skx_int_64,

	 // setv
	 //BLIS_SETV_KER,     BLIS_FLOAT,     bli_ssetv_zen_int,
	 //BLIS_SETV_KER,     BLIS_DOUBLE,    bli_dsetv_zen_int,
	 BLIS_SETV_KER,     BLIS_FLOAT,     bli_ssetv_skx_int_128,
	 BLIS_SETV_KER,     BLIS_DOUBLE,    bli_dsetv_skx_int_64,

	 // swapv
	 //BLIS_SWAPV_KER,     BLIS_FLOAT,    bli_sswapv_zen_int8,
	 //BLIS_SWAPV_KER,     BLIS_DOUBLE,   bli_dswapv_zen_int8,
	 BLIS_SWAPV_KER,     BLIS_FLOAT,    bli_sswapv_skx_int_128,
	 BLIS_SWAPV_KER,     BLIS_DOUBLE,   bli_dswapv_skx_int_64,

	 // copyv
	 //BLIS_COPYV_KER,     BLIS_FLOAT,    bli_scopyv_zen_int,
	 //BLIS_COPYV_KER,     BLIS_DOUBLE,   bli_dcopyv_zen_int,
	 BLIS_COPYV_KER,     BLIS_FLOAT,    bli_scopyv_skx_int_128,
	 BLIS_COPYV_KER,     BLIS_DOUBLE,   bli_dcopyv_skx_int_64,
	 
	 // gemmsup
	 BLIS_GEMMSUP_RRR_UKR, BLIS_DOUBLE, bli_dgemmsup_cv_skx_int_24x8,
	 BLIS_GEMMSUP_RRC_UKR, BLIS_DOUBLE, bli_dgemmsup_rd_skx_int_2x8,
	 BLIS_GEMMSUP_RCR_UKR, BLIS_DOUBLE, bli_dgemmsup_cv_skx_int_24x8,
	 BLIS_GEMMSUP_RCC_UKR, BLIS_DOUBLE, bli_dgemmsup_cv_skx_int_24x8,
	 BLIS_GEMMSUP_CRR_UKR, BLIS_DOUBLE, bli_dgemmsup_cv_skx_int_24x8,
	 BLIS_GEMMSUP_CRC_UKR, BLIS_DOUBLE, bli_dgemmsup_rd_skx_int_2x8,
	 BLIS_GEMMSUP_CCR_UKR, BLIS_DOUBLE, bli_dgemmsup_cv_skx_int_24x8,
	 BLIS_GEMMSUP_CCC_UKR, BLIS_DOUBLE, bli_dgemmsup_cv_skx_int_24x8,

	 BLIS_GEMMSUP_RRR_UKR, BLIS_FLOAT, bli_sgemmsup_cv_skx_int_48x8,
	 BLIS_GEMMSUP_RRC_UKR, BLIS_FLOAT, bli_sgemmsup_rd_skx_int_2x8,
	 BLIS_GEMMSUP_RCR_UKR, BLIS_FLOAT, bli_sgemmsup_cv_skx_int_48x8,
	 BLIS_GEMMSUP_RCC_UKR, BLIS_FLOAT, bli_sgemmsup_cv_skx_int_48x8,
	 BLIS_GEMMSUP_CRR_UKR, BLIS_FLOAT, bli_sgemmsup_cv_skx_int_48x8,
	 BLIS_GEMMSUP_CRC_UKR, BLIS_FLOAT, bli_sgemmsup_rd_skx_int_2x8,
	 BLIS_GEMMSUP_CCR_UKR, BLIS_FLOAT, bli_sgemmsup_cv_skx_int_48x8,
	 BLIS_GEMMSUP_CCC_UKR, BLIS_FLOAT, bli_sgemmsup_cv_skx_int_48x8,

	 BLIS_VA_END
	);

	// Update the context with storage preferences.
	bli_cntx_set_ukr_prefs
	(
	 cntx,

	 // level-3
	 BLIS_GEMM_UKR_ROW_PREF, BLIS_FLOAT , FALSE,
	 BLIS_GEMM_UKR_ROW_PREF, BLIS_DOUBLE, FALSE,

	 // gemmsup
	 BLIS_GEMMSUP_RRR_UKR_ROW_PREF, BLIS_DOUBLE, FALSE,
	 BLIS_GEMMSUP_RRC_UKR_ROW_PREF, BLIS_DOUBLE, TRUE,
	 BLIS_GEMMSUP_RCR_UKR_ROW_PREF, BLIS_DOUBLE, FALSE,
	 BLIS_GEMMSUP_RCC_UKR_ROW_PREF, BLIS_DOUBLE, FALSE,
	 BLIS_GEMMSUP_CRR_UKR_ROW_PREF, BLIS_DOUBLE, FALSE,
	 BLIS_GEMMSUP_CRC_UKR_ROW_PREF, BLIS_DOUBLE, TRUE,
	 BLIS_GEMMSUP_CCR_UKR_ROW_PREF, BLIS_DOUBLE, FALSE,
	 BLIS_GEMMSUP_CCC_UKR_ROW_PREF, BLIS_DOUBLE, FALSE,

	 BLIS_GEMMSUP_RRR_UKR_ROW_PREF, BLIS_FLOAT, FALSE,
	 BLIS_GEMMSUP_RRC_UKR_ROW_PREF, BLIS_FLOAT, TRUE,
	 BLIS_GEMMSUP_RCR_UKR_ROW_PREF, BLIS_FLOAT, FALSE,
	 BLIS_GEMMSUP_RCC_UKR_ROW_PREF, BLIS_FLOAT, FALSE,
	 BLIS_GEMMSUP_CRR_UKR_ROW_PREF, BLIS_FLOAT, FALSE,
	 BLIS_GEMMSUP_CRC_UKR_ROW_PREF, BLIS_FLOAT, TRUE,
	 BLIS_GEMMSUP_CCR_UKR_ROW_PREF, BLIS_FLOAT, FALSE,
	 BLIS_GEMMSUP_CCC_UKR_ROW_PREF, BLIS_FLOAT, FALSE,

	 BLIS_VA_END
	);

	// Initialize level-3 blocksize objects with architecture-specific values.
	//                                              s      d      c      z
	bli_blksz_init_easy( &blkszs[ BLIS_MR ],       48,    24,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NR ],        8,     8,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],      432,   240,    -1,    -1 );
	bli_blksz_init     ( &blkszs[ BLIS_KC ],      432,   304,    -1,    -1,
	                                              432,   320,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ],     3072,  3072,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_AF ],        8,     8,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_DF ],        8,     8,    -1,    -1 );

	// Initialize sup thresholds with architecture-appropriate values.
	//                                              s      d      c      z
	bli_blksz_init_easy( &blkszs[ BLIS_MT ],      432,   208,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NT ],      432,   208,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_KT ],      432,   208,    -1,    -1 );

	// Initialize level-3 sup blocksize objects with architecture-specific
	// values.
	//                                              s      d      c      z
	bli_blksz_init     ( &blkszs[ BLIS_MR_SUP ],   48,    24,    -1,    -1,
	                                               48,    24,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NR_SUP ],    8,     8,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_MC_SUP ],  432,   240,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC_SUP ],  336,   320,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC_SUP ], 3072,  3072,    -1,    -1 );


	// Update the context with the current architecture's register and cache
	// blocksizes (and multiples) for native execution.
	bli_cntx_set_blkszs
	(
	 cntx,

	 // level-3
	 BLIS_NC, &blkszs[ BLIS_NC ], BLIS_NR,
	 BLIS_KC, &blkszs[ BLIS_KC ], BLIS_KR,
	 BLIS_MC, &blkszs[ BLIS_MC ], BLIS_MR,
	 BLIS_NR, &blkszs[ BLIS_NR ], BLIS_NR,
	 BLIS_MR, &blkszs[ BLIS_MR ], BLIS_MR,

	 // level-1f
	 BLIS_AF, &blkszs[ BLIS_AF ], BLIS_AF,
	 BLIS_DF, &blkszs[ BLIS_DF ], BLIS_DF,

	 // gemmsup thresholds
	 BLIS_MT, &blkszs[ BLIS_MT ], BLIS_MT,
	 BLIS_NT, &blkszs[ BLIS_NT ], BLIS_NT,
	 BLIS_KT, &blkszs[ BLIS_KT ], BLIS_KT,

	 // level-3 sup
	 BLIS_NC_SUP, &blkszs[ BLIS_NC_SUP ], BLIS_NR_SUP,
	 BLIS_KC_SUP, &blkszs[ BLIS_KC_SUP ], BLIS_KR_SUP,
	 BLIS_MC_SUP, &blkszs[ BLIS_MC_SUP ], BLIS_MR_SUP,
	 BLIS_NR_SUP, &blkszs[ BLIS_NR_SUP ], BLIS_NR_SUP,
	 BLIS_MR_SUP, &blkszs[ BLIS_MR_SUP ], BLIS_MR_SUP,

	 BLIS_VA_END
	);
}

