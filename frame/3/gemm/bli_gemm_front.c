/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2020, Advanced Micro Devices, Inc.

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

void bli_gemm_front
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
             rntm_t* rntm
     )
{
	bli_init_once();

	obj_t   a_local;
	obj_t   b_local;
	obj_t   c_local;

#if 0
#ifdef BLIS_ENABLE_SMALL_MATRIX
	// Only handle small problems separately for homogeneous datatypes.
	if ( bli_obj_dt( a ) == bli_obj_dt( b ) &&
		 bli_obj_dt( a ) == bli_obj_dt( c ) &&
		 bli_obj_comp_prec( c ) == bli_obj_prec( c ) )
	{
		err_t status = bli_gemm_small( alpha, a, b, beta, c, cntx, cntl );
		if ( status == BLIS_SUCCESS ) return;
	}
#endif
#endif

	// Alias A, B, and C in case we need to apply transformations.
	bli_obj_alias_to( a, &a_local );
	bli_obj_alias_to( b, &b_local );
	bli_obj_alias_to( c, &c_local );

	// Set the obj_t buffer field to the location currently implied by the row
	// and column offsets and then zero the offsets. If any of the original
	// obj_t's were views into larger matrices, this step effectively makes
	// those obj_t's "forget" their lineage.
	bli_obj_reset_origin( &a_local );
	bli_obj_reset_origin( &b_local );
	bli_obj_reset_origin( &c_local );

	// An optimization: If C is stored by rows and the micro-kernel prefers
	// contiguous columns, or if C is stored by columns and the micro-kernel
	// prefers contiguous rows, transpose the entire operation to allow the
	// micro-kernel to access elements of C in its preferred manner.
	const num_t dt                = bli_obj_dt( &c_local  );
	const stor3_t stor_id         = bli_obj_stor3_from_strides( &c_local, &a_local, &b_local  );
	const bool is_rrr_rrc_rcr_crr = ( stor_id == BLIS_RRR ||
			                          stor_id == BLIS_RRC ||
									  stor_id == BLIS_RCR ||
									  stor_id == BLIS_CRR );
	bool is_primary               = true;

	is_primary = ( is_rrr_rrc_rcr_crr == bli_cntx_ukr_prefers_rows_dt( dt, bli_stor3_ukr( stor_id ), cntx ) );

	if ( !is_primary )
	{
		bli_obj_swap( &a_local, &b_local );

		bli_obj_induce_trans( &a_local );
		bli_obj_induce_trans( &b_local );
		bli_obj_induce_trans( &c_local );
	}

#ifdef BLIS_ENABLE_GEMM_MD
	cntx_t cntx_local;

	// If any of the storage datatypes differ, or if the computation precision
	// differs from the storage precision of C, utilize the mixed datatype
	// code path.
	// NOTE: If we ever want to support the caller setting the computation
	// domain explicitly, we will need to check the computation dt against the
	// storage dt of C (instead of the computation precision against the
	// storage precision of C).
	if ( bli_obj_dt( &c_local ) != bli_obj_dt( &a_local ) ||
		 bli_obj_dt( &c_local ) != bli_obj_dt( &b_local ) ||
		 bli_obj_comp_prec( &c_local ) != bli_obj_prec( &c_local ) )
	{
		// Handle mixed datatype cases in bli_gemm_md(), which may modify
		// the objects or the context. (If the context is modified, cntx
		// is adjusted to point to cntx_local.)
		bli_gemm_md( &a_local, &b_local, beta, &c_local, &cntx_local, &cntx );
	}
#endif

	// Next, we handle the possibility of needing to typecast alpha to the
	// computation datatype and/or beta to the storage datatype of C.

	// Attach alpha to B, and in the process typecast alpha to the target
	// datatype of the matrix (which in this case is equal to the computation
	// datatype).
	bli_obj_scalar_attach( BLIS_NO_CONJUGATE, alpha, &b_local );

	// Attach beta to C, and in the process typecast beta to the target
	// datatype of the matrix (which in this case is equal to the storage
	// datatype of C).
	bli_obj_scalar_attach( BLIS_NO_CONJUGATE, beta,  &c_local );

	// Change the alpha and beta pointers to BLIS_ONE since the values have
	// now been typecast and attached to the matrices above.
	alpha = &BLIS_ONE;
	beta  = &BLIS_ONE;

	// Parse and interpret the contents of the rntm_t object to properly
	// set the ways of parallelism for each loop, and then make any
	// additional modifications necessary for the current operation.
	if ( bli_info_get_enable_fup() )
		bli_rntm_set_nt_for_size ( bli_obj_length( &c_local ),
				                   bli_obj_width( &c_local ),
								   bli_obj_width( &a_local ),
								   bli_obj_dt( &c_local ),
								   cntx,
								   rntm );

	bli_rntm_set_ways_for_op( BLIS_GEMM,
			                  BLIS_LEFT, // ignored for gemm/hemm/symm
							  bli_obj_length( &c_local ),
							  bli_obj_width( &c_local ),
							  bli_obj_width( &a_local ),
							  rntm );
	
	// Set the pack schemas within the objects.
	bli_l3_set_schemas( BLIS_GEMM, &a_local, &b_local, &c_local, cntx, rntm );

	      obj_t* cp    = &c_local;
	const obj_t* betap = beta;

	// Invoke the internal back-end via the thread handler.
	bli_l3_thread_decorator
	(
	  bli_l3_int,
	  BLIS_GEMM, // operation family id
	  alpha,
	  &a_local,
	  &b_local,
	  betap,
	  cp,
	  cntx,
	  rntm
	);

	if (bli_info_get_enable_diagnosis())
	{
		bli_rntm_print(rntm);
	}
}

