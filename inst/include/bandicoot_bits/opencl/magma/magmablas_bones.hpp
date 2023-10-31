// Copyright 2023 Ryan Curtin (http://www.ratml.org)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------

// This file contains source code adapted from
// clMAGMA 1.3 (2014-11-14) and/or MAGMA 2.7 (2022-11-09).
// clMAGMA 1.3 and MAGMA 2.7 are distributed under a
// 3-clause BSD license as follows:
//
//  -- Innovative Computing Laboratory
//  -- Electrical Engineering and Computer Science Department
//  -- University of Tennessee
//  -- (C) Copyright 2009-2015
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions
//  are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of the University of Tennessee, Knoxville nor the
//    names of its contributors may be used to endorse or promote products
//    derived from this software without specific prior written permission.
//
//  This software is provided by the copyright holders and contributors
//  ``as is'' and any express or implied warranties, including, but not
//  limited to, the implied warranties of merchantability and fitness for
//  a particular purpose are disclaimed. In no event shall the copyright
//  holders or contributors be liable for any direct, indirect, incidental,
//  special, exemplary, or consequential damages (including, but not
//  limited to, procurement of substitute goods or services; loss of use,
//  data, or profits; or business interruption) however caused and on any
//  theory of liability, whether in contract, strict liability, or tort
//  (including negligence or otherwise) arising in any way out of the use
//  of this software, even if advised of the possibility of such damage.



// Adaptations of magmablas_* functions to use existing bandicoot backend functionality.

#define MAGMABLAS_BLK_X 64
#define MAGMABLAS_BLK_Y 32

#define MAGMABLAS_TRANS_NX 32
#define MAGMABLAS_TRANS_NY 8
#define MAGMABLAS_TRANS_NB 32
#define MAGMABLAS_TRANS_INPLACE_NB 16

#define MAGMABLAS_LASWP_MAX_PIVOTS 32
#define MAGMABLAS_LASWP_NTHREADS 64

#define MAGMABLAS_LASCL_NB 64

#define MAGMABLAS_LASET_BAND_NB 64

#define MAGMABLAS_LANSY_INF_BS 32
#define MAGMABLAS_LANSY_MAX_BS 64

// Utility struct for laswp
typedef struct
  {
  int npivots;
  int ipiv[MAGMABLAS_LASWP_MAX_PIVOTS];
  } magmablas_laswp_params_t;



inline
void
magmablas_slaset(magma_uplo_t uplo, magma_int_t m, magma_int_t n, float offdiag, float diag, magmaFloat_ptr dA, size_t dA_offset, magma_int_t ldda, magma_queue_t queue);



inline
void
magmablas_dlaset(magma_uplo_t uplo, magma_int_t m, magma_int_t n, double offdiag, double diag, magmaDouble_ptr dA, size_t dA_offset, magma_int_t ldda, magma_queue_t queue);



template<typename eT>
inline
void
magmablas_run_laset_kernel(const opencl::magma_real_kernel_id::enum_id num, magma_uplo_t uplo, magma_int_t m, magma_int_t n, eT offdiag, eT diag, cl_mem dA, size_t dA_offset, magma_int_t ldda, magma_queue_t queue);



inline
void
magmablas_slaswp(magma_int_t n, magmaFloat_ptr dAT, size_t dAT_offset, magma_int_t ldda, magma_int_t k1, magma_int_t k2, const magma_int_t* ipiv, magma_int_t inci, magma_queue_t queue);



inline
void
magmablas_dlaswp(magma_int_t n, magmaDouble_ptr dAT, size_t dAT_offset, magma_int_t ldda, magma_int_t k1, magma_int_t k2, const magma_int_t* ipiv, magma_int_t inci, magma_queue_t queue);



template<typename eT>
inline
void
magmablas_laswp(magma_int_t n, cl_mem dAT, size_t dAT_offset, magma_int_t ldda, magma_int_t k1, magma_int_t k2, const magma_int_t* ipiv, magma_int_t inci, magma_queue_t queue);



inline
void
magmablas_slaset_band(magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t k, float offdiag, float diag, magmaFloat_ptr dA, size_t dA_offset, magma_int_t ldda, magma_queue_t queue);



inline
void
magmablas_dlaset_band(magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t k, double offdiag, double diag, magmaDouble_ptr dA, size_t dA_offset, magma_int_t ldda, magma_queue_t queue);



template<typename eT>
inline
void
magmablas_laset_band(magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t k, eT offdiag, eT diag, cl_mem dA, size_t dA_offset, magma_int_t ldda, magma_queue_t queue);



inline
void
magmablas_stranspose(magma_int_t m, magma_int_t n, magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t ldda, magmaFloat_ptr dAT, size_t dAT_offset, magma_int_t lddat, magma_queue_t queue);



inline
void
magmablas_dtranspose(magma_int_t m, magma_int_t n, magmaDouble_const_ptr dA, size_t dA_offset, magma_int_t ldda, magmaDouble_ptr dAT, size_t dAT_offset, magma_int_t lddat, magma_queue_t queue);



template<typename eT>
inline
void
magmablas_transpose(magma_int_t m, magma_int_t n, cl_mem dA, size_t dA_offset, magma_int_t ldda, cl_mem dAT, size_t dAT_offset, magma_int_t lddat, magma_queue_t queue);



inline
void
magmablas_stranspose_inplace(magma_int_t n, magmaFloat_ptr dA, size_t dA_offset, magma_int_t ldda, magma_queue_t queue);



inline
void
magmablas_dtranspose_inplace(magma_int_t n, magmaDouble_ptr dA, size_t dA_offset, magma_int_t ldda, magma_queue_t queue);



template<typename eT>
inline
void
magmablas_transpose_inplace(magma_int_t n, cl_mem dA, size_t dA_offset, magma_int_t ldda, magma_queue_t queue);



inline
void
magmablas_slascl(magma_type_t type, magma_int_t kl, magma_int_t ku, float cfrom, float cto, magma_int_t m, magma_int_t n, magmaFloat_ptr dA, size_t dA_offset, magma_int_t ldda, magma_queue_t queue, magma_int_t *info);



inline
void
magmablas_dlascl(magma_type_t type, magma_int_t kl, magma_int_t ku, double cfrom, double cto, magma_int_t m, magma_int_t n, magmaDouble_ptr dA, size_t dA_offset, magma_int_t ldda, magma_queue_t queue, magma_int_t *info);



template<typename eT>
inline
void
magmablas_lascl(magma_type_t type, magma_int_t kl, magma_int_t ku, eT cfrom, eT cto, magma_int_t m, magma_int_t n, cl_mem dA, size_t dA_offset, magma_int_t ldda, magma_queue_t queue, magma_int_t *info);



inline
float
magmablas_slansy(magma_norm_t norm, magma_uplo_t uplo, magma_int_t n, magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t ldda, magmaFloat_ptr dwork, size_t dwork_offset, magma_int_t lddwork, magma_queue_t queue);



inline
double
magmablas_dlansy(magma_norm_t norm, magma_uplo_t uplo, magma_int_t n, magmaDouble_const_ptr dA, size_t dA_offset, magma_int_t ldda, magmaDouble_ptr dwork, size_t dwork_offset, magma_int_t lddwork, magma_queue_t queue);



template<typename eT>
inline
int
magmablas_lansy(magma_norm_t norm, magma_uplo_t uplo, magma_int_t n, cl_mem dA, size_t dA_offset, magma_int_t ldda, cl_mem dwork, size_t dwork_offset, magma_queue_t queue);



template<typename eT>
inline
void
magmablas_lansy_inf(magma_uplo_t uplo, int n, magmaFloat_const_ptr A, size_t A_offset, int lda, magmaFloat_ptr dwork, size_t dwork_offset, magma_queue_t queue);



template<typename eT>
inline
void
magmablas_lansy_max(magma_uplo_t uplo, int n, magmaFloat_const_ptr A, size_t A_offset, int lda, magmaFloat_ptr dwork, size_t dwork_offset, magma_queue_t queue);
