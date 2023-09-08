/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021, The University of Texas at Austin

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

void bli_l3_set_schemas
     (
	         opid_t  family,
             obj_t*  a,
             obj_t*  b,
       const obj_t*  c,
       const cntx_t* cntx,
	   const rntm_t* rntm
     )
{
	// Begin with pack schemas for native execution.
	pack_t schema_a = BLIS_PACKED_ROW_PANELS;
	pack_t schema_b = BLIS_PACKED_COL_PANELS;

	// When executing the 1m method, choose the appropriate pack schemas based
	// on the microkernel preference encoded within the current cntx_t (which
	// was presumably returned by the gks).
	if ( bli_cntx_method( cntx ) == BLIS_1M )
	{
		num_t dt = bli_obj_domain( c ) | bli_obj_comp_prec( c );

		// Note that bli_cntx_l3_vir_ukr_prefers_cols_dt() will use the real
		// projection of dt to query the preference of the corresponding native
		// real-domain microkernel. This is what ultimately determines which
		// variant of 1m is applicable.
		if ( bli_cntx_ukr_prefers_cols_dt( dt, BLIS_GEMM_VIR_UKR, cntx ) )
		{
			schema_a = BLIS_PACKED_ROW_PANELS_1E;
			schema_b = BLIS_PACKED_COL_PANELS_1R;
		}
		else
		{
			schema_a = BLIS_PACKED_ROW_PANELS_1R;
			schema_b = BLIS_PACKED_COL_PANELS_1E;
		}
	}
	else
	{
		if ( bli_info_get_enable_fup() && 
			( family == BLIS_GEMM ) )
		{
			const num_t dt   = bli_obj_dt( c );
			const dim_t m    = bli_obj_length_after_trans( c );
			const dim_t n    = bli_obj_width_after_trans( c );
			const dim_t k    = bli_obj_width_after_trans( a );
			const dim_t mr   = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx );
			const dim_t nr   = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx );
			// const dim_t nt   = bli_rntm_num_threads( rntm );
			// const dim_t jc   = bli_rntm_jc_ways( rntm );
			// const dim_t jr   = bli_rntm_jr_ways( rntm );
			// const dim_t pc   = bli_rntm_pc_ways( rntm );
			const dim_t ic   = bli_rntm_ic_ways( rntm );
			// const dim_t ir   = bli_rntm_ir_ways( rntm );
			const dim_t rs_a = bli_obj_has_notrans( a ) ? bli_obj_row_stride( a ) : bli_obj_col_stride( a );
			const dim_t cs_b = bli_obj_has_notrans( b ) ? bli_obj_col_stride( b ) : bli_obj_row_stride( b );

			// const dim_t md   = m/( pow( (double)nt, (double)BLIS_THREAD_RATIO_M/(BLIS_THREAD_RATIO_M+BLIS_THREAD_RATIO_N) ) );
			const dim_t md   = m/ic;
			if ( bli_cntx_l3_supb_thresh_is_met( dt, md, n, k, cntx ) )
			{
				schema_b = BLIS_NOT_PACKED;
				// even no pack, we still set the panel_dim and panel_stride 
				// for nr loops
				bli_obj_set_panel_dim( nr, b );
				bli_obj_set_panel_stride( nr*cs_b, b );
			}
			if ( bli_cntx_l3_supa_thresh_is_met( dt, md, n, k, cntx ) )
			{
				schema_a = BLIS_NOT_PACKED;
				// even no pack, we still set the panel_dim and panel_stride 
				// for mr loops
				bli_obj_set_panel_dim( mr, a );
				bli_obj_set_panel_stride( mr*rs_a, a );
			}
		}
	}
	// Embed the schemas into the objects for A and B. This is a sort of hack
	// for communicating the desired pack schemas to bli_gemm_cntl_create()
	// (via bli_l3_thread_decorator() and bli_l3_cntl_create_if()). This allows
	// us to subsequently access the schemas from the control tree, which
	// hopefully reduces some confusion, particularly in bli_packm_init().
	bli_obj_set_pack_schema( schema_a, a );
	bli_obj_set_pack_schema( schema_b, b );
}

