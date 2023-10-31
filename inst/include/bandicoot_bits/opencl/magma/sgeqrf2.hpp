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



// SGEQRF computes a QR factorization of a real M-by-N matrix A:
// A = Q * R.
//
// This version has LAPACK-complaint arguments.
//
// Other versions (magma_sgeqrf_gpu and magma_sgeqrf3_gpu) store the
// intermediate T matrices.



inline
magma_int_t
magma_sgeqrf2_gpu
  (
  magma_int_t m,
  magma_int_t n,
  magmaFloat_ptr dA,
  size_t dA_offset,
  magma_int_t ldda,
  float* tau,
  magma_int_t* info
  )
  {
  magmaFloat_ptr dwork, dT;
  size_t dT_offset;
  float *work, *hwork;
  magma_int_t cols, i, ib, ldwork, lddwork, lhwork, lwork, minmn, nb, old_i, old_ib, rows;

  // check arguments
  *info = 0;
  if (m < 0)
    *info = -1;
  else if (n < 0)
    *info = -2;
  else if (ldda < std::max(1, m))
      *info = -4;

  if (*info != 0)
    {
    //magma_xerbla( __func__, -(*info) );
    return *info;
    }

  minmn = std::min( m, n );
  if (minmn == 0)
    return *info;

  nb = magma_get_sgeqrf_nb( m, n );

  // dwork is (n-nb)*nb for larfb
  // dT    is nb*nb
  lddwork = n-nb;
  if (MAGMA_SUCCESS != magma_smalloc( &dwork, n*nb ))
    {
    *info = MAGMA_ERR_DEVICE_ALLOC;
    return *info;
    }
  dT = dwork;
  dT_offset = (n-nb)*nb;

  // work  is m*nb for panel
  // hwork is n*nb, and at least 2*nb*nb for T in larft and R in spanel_to_q
  ldwork = m;
  lhwork = std::max( n*nb, 2*nb*nb );
  lwork  = ldwork*nb + lhwork;
  // last block needs rows*cols for matrix and prefers cols*nb for work
  // worst case is n > m*nb, m a small multiple of nb:
  // needs n*nb + n > (m+n)*nb
  // prefers 2*n*nb, about twice above (m+n)*nb.
  i = ((minmn-1)/nb)*nb;
  lwork = std::max( lwork, (m-i)*(n-i) + (n-i)*nb );

  if (MAGMA_SUCCESS != magma_smalloc_pinned( &work, lwork ))
    {
    magma_free( dwork );
    *info = MAGMA_ERR_HOST_ALLOC;
    return *info;
    }
  hwork = work + ldwork*nb;

  magma_queue_t queues[2];
  queues[0] = magma_queue_create();
  queues[1] = magma_queue_create();

  if ( nb > 1 && nb < minmn )
    {
    // need nb*nb for T in larft and R in dpanel_to_q
    assert( lhwork >= 2*nb*nb );

    // Use blocked code initially
    old_i = 0; old_ib = nb;
    for (i = 0; i < minmn-nb; i += nb)
      {
      ib = std::min( minmn-i, nb );
      rows = m - i;

      // get i-th panel from device
      magma_sgetmatrix_async( rows, ib,
                              dA, dA_offset + i + i * ldda, ldda,
                              &work[i], ldwork, queues[1] );
      if (i > 0)
        {
        // Apply H^H to A(i:m,i+2*ib:n) from the left
        cols = n - old_i - 2*old_ib;
        magma_slarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                          m-old_i, cols, old_ib,
                          dA, dA_offset + old_i + old_i * ldda,                ldda, dT,    dT_offset, nb,
                          dA, dA_offset + old_i + (old_i + 2 * old_ib) * ldda, ldda, dwork, 0,         lddwork, queues[0] );

        // Fix the diagonal block
        magma_ssetmatrix_async( old_ib, old_ib,
                                &work[old_i],      ldwork,
                                dA, dA_offset + old_i + old_i * ldda, ldda, queues[0] );
        }

      magma_queue_sync( queues[1] );  // wait to get work(i)
      lapack::geqrf(rows, ib, &work[i], ldwork, &tau[i], hwork, lhwork, info);
      // Form the triangular factor of the block reflector in hwork
      // H = H(i) H(i+1) . . . H(i+ib-1)
      lapack::larft('F', 'C', rows, ib, &work[i], ldwork, &tau[i], hwork, ib);

      // set  the upper triangle of panel (V) to identity
      magma_spanel_to_q( MagmaUpper, ib, &work[i], ldwork, hwork+ib*ib );

      // send i-th V matrix to device
      magma_ssetmatrix( rows, ib,
                        &work[i], ldwork,
                        dA, dA_offset + i + i * ldda, ldda, queues[1] );

      if (i + ib < n)
        {
        // wait for previous trailing matrix update (above) to finish with dT
        magma_queue_sync( queues[0] );

        // send T matrix to device
        magma_ssetmatrix( ib, ib,
                          hwork, ib,
                          dT, dT_offset, nb, queues[1] );

        if (i+nb < minmn-nb)
          {
          // Apply H^H to A(i:m,i+ib:i+2*ib) from the left
          magma_slarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                            rows, ib, ib,
                            dA,    dA_offset + i + i * ldda,        ldda,
                            dT,    dT_offset,                       nb,
                            dA,    dA_offset + i + (i + ib) * ldda, ldda,
                            dwork, 0,                               lddwork, queues[1] );
          // wait for larfb to finish with dwork before larfb in next iteration starts
          magma_queue_sync( queues[1] );
          // restore upper triangle of panel
          magma_sq_to_panel( MagmaUpper, ib, &work[i], ldwork, hwork+ib*ib );
          }
        else
          {
          // Apply H^H to A(i:m,i+ib:n) from the left
          magma_slarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                            rows, n-i-ib, ib,
                            dA,    dA_offset + i + i * ldda,      ldda,
                            dT,    dT_offset,                     nb,
                            dA,    dA_offset + i + (i+ib) * ldda, ldda,
                            dwork, 0,                             lddwork, queues[1] );
          magma_sq_to_panel( MagmaUpper, ib, &work[i], ldwork, hwork+ib*ib );
          // Fix the diagonal block
          magma_ssetmatrix( ib, ib,
                            &work[i], ldwork,
                            dA, dA_offset + i + i * ldda, ldda, queues[1] );
          }
        old_i  = i;
        old_ib = ib;
        }
      }
    }
  else
    {
    i = 0;
    }

  // Use unblocked code to factor the last or only block.
  if (i < minmn)
    {
    rows = m-i;
    cols = n-i;
    magma_sgetmatrix( rows, cols, dA, dA_offset + i + i * ldda, ldda, work, rows, queues[1] );
    // see comments for lwork above
    lhwork = lwork - rows*cols;
    lapack::geqrf(rows, cols, work, rows, &tau[i], &work[rows * cols], lhwork, info);
    magma_ssetmatrix( rows, cols, work, rows, dA, dA_offset + i + i * ldda, ldda, queues[1] );
    }

  magma_queue_destroy( queues[0] );
  magma_queue_destroy( queues[1] );

  magma_free( dwork );
  magma_free_pinned( work );

  return *info;
  } // magma_sgeqrf2_gpu
