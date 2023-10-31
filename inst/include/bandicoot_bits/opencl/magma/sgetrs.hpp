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



// Purpose
// -------
// SGETRS solves a system of linear equations
//     A * X = B,
//     A**T * X = B,  or
//     A**H * X = B
// with a general N-by-N matrix A using the LU factorization computed by SGETRF_GPU.



inline
magma_int_t
magma_sgetrs_gpu
  (
  magma_trans_t trans,
  magma_int_t n,
  magma_int_t nrhs,
  magmaFloat_ptr dA,
  magma_int_t ldda,
  magma_int_t *ipiv,
  magmaFloat_ptr dB,
  magma_int_t lddb,
  magma_int_t *info
  )
  {
  // Constants
  const float c_one = MAGMA_S_ONE;

  // Local variables
  float *work = NULL;
  bool notran = (trans == MagmaNoTrans);
  magma_int_t i1, i2;

  *info = 0;
  if ( (! notran) &&
       (trans != MagmaTrans) &&
       (trans != MagmaConjTrans) )
    {
    *info = -1;
    }
  else if (n < 0)
    {
    *info = -2;
    }
  else if (nrhs < 0)
    {
    *info = -3;
    }
  else if (ldda < std::max(1,n))
    {
    *info = -5;
    }
  else if (lddb < std::max(1,n))
    {
    *info = -8;
    }
  if (*info != 0)
    {
    //magma_xerbla( __func__, -(*info) );
    return *info;
    }

  /* Quick return if possible */
  if (n == 0 || nrhs == 0)
    {
    return *info;
    }

  magma_queue_t queue = magma_queue_create();

  i1 = 1;
  i2 = n;
  if (notran)
    {
    /* Solve A * X = B. */
    magmablas_slaswp( nrhs, dB, 0, lddb, i1, i2, ipiv, 1, queue );
    magma_queue_sync( queue );

    if (nrhs == 1)
      {
      magma_strsv( MagmaLower, MagmaNoTrans, MagmaUnit,    n, dA, 0, ldda, dB, 0, 1, queue );
      magma_strsv( MagmaUpper, MagmaNoTrans, MagmaNonUnit, n, dA, 0, ldda, dB, 0, 1, queue );
      }
    else
      {
      magma_strsm( MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,    n, nrhs, c_one, dA, 0, ldda, dB, 0, lddb, queue );
      magma_strsm( MagmaLeft, MagmaUpper, MagmaNoTrans, MagmaNonUnit, n, nrhs, c_one, dA, 0, ldda, dB, 0, lddb, queue );
      }
    }
  else
    {
    /* Solve A**T * X = B  or  A**H * X = B. */
    if (nrhs == 1)
      {
      magma_strsv( MagmaUpper, trans, MagmaNonUnit, n, dA, 0, ldda, dB, 0, 1, queue );
      magma_strsv( MagmaLower, trans, MagmaUnit,    n, dA, 0, ldda, dB, 0, 1, queue );
      }
    else
      {
      magma_strsm( MagmaLeft, MagmaUpper, trans, MagmaNonUnit, n, nrhs, c_one, dA, 0, ldda, dB, 0, lddb, queue );
      magma_strsm( MagmaLeft, MagmaLower, trans, MagmaUnit,    n, nrhs, c_one, dA, 0, ldda, dB, 0, lddb, queue );
      }

    magma_smalloc_cpu( &work, n * nrhs );
    if ( work == NULL )
      {
      *info = MAGMA_ERR_HOST_ALLOC;
      return *info;
      }

    // The MAGMABLAS laswp() implementation does not support applying pivots in reverse order from ipiv, so we use CPU LAPACK instead.
    // TODO: fix MAGMABLAS laswp() implementation!

    magma_int_t inc = -1;
    magma_sgetmatrix( n, nrhs, dB, 0, lddb, work, n, queue );
    lapack::laswp(nrhs, work, n, i1, i2, ipiv, inc);
    //magmablas_slaswp( nrhs, dB, 0, lddb, i1, i2, ipiv, inc, queue );
    magma_ssetmatrix( n, nrhs, work, n, dB, 0, lddb, queue );

    magma_free_cpu(work);
    }

  magma_queue_destroy( queue );

  return *info;
  }
