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
// clMAGMA 1.3 (2014-11-14) and/or MAGMA 2.2 (2016-11-20).
// clMAGMA 1.3 and MAGMA 2.2 are distributed under a
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
// SLAEX0 computes all eigenvalues and the choosen eigenvectors of a
// symmetric tridiagonal matrix using the divide and conquer method.



inline
magma_int_t
magma_slaex0
  (
  magma_int_t n,
  float *d, float *e,
  float *Q, magma_int_t ldq,
  float *work, magma_int_t *iwork,
  magmaFloat_ptr dwork, size_t dwork_offset,
  magma_range_t range, float vl, float vu,
  magma_int_t il, magma_int_t iu,
  magma_int_t *info
  )
  {
  magma_int_t ione = 1;
  magma_range_t range2;
  magma_int_t i, indxq;
  magma_int_t j, k, matsiz, msd2, smlsiz;
  magma_int_t submat, subpbs;

  // Test the input parameters.
  *info = 0;

  if ( n < 0 )
    {
    *info = -1;
    }
  else if ( ldq < std::max(1, n) )
    {
    *info = -5;
    }

  if ( *info != 0 )
    {
    //magma_xerbla( __func__, -(*info) );
    return *info;
    }

  // Quick return if possible
  if (n == 0)
    {
    return *info;
    }

  magma_queue_t queue = magma_queue_create();

  smlsiz = magma_get_smlsize_divideconquer();

  // Determine the size and placement of the submatrices, and save in
  // the leading elements of IWORK.
  iwork[0] = n;
  subpbs= 1;
  while (iwork[subpbs - 1] > smlsiz)
    {
    for (j = subpbs; j > 0; --j)
      {
      iwork[2*j - 1] = (iwork[j-1]+1)/2;
      iwork[2*j - 2] = iwork[j-1]/2;
      }
    subpbs *= 2;
    }
  for (j=1; j < subpbs; ++j)
    {
    iwork[j] += iwork[j-1];
    }

  // Divide the matrix into SUBPBS submatrices of size at most SMLSIZ+1
  // using rank-1 modifications (cuts).
  for (i=0; i < subpbs-1; ++i)
    {
    submat = iwork[i];
    d[submat-1] -= std::abs(e[submat-1]);
    d[submat] -= std::abs(e[submat-1]);
    }

  indxq = 4*n + 3;

  // Solve each submatrix eigenproblem at the bottom of the divide and
  // conquer tree.
  for (i = 0; i < subpbs; ++i)
    {
    if (i == 0)
      {
      submat = 0;
      matsiz = iwork[0];
      }
    else
      {
      submat = iwork[i-1];
      matsiz = iwork[i] - iwork[i-1];
      }
    lapack::steqr('I', matsiz, &d[submat], &e[submat],
                  Q + (submat) + (submat) * ldq, ldq, work, info); // change to edc?
    if (*info != 0)
      {
      *info = (submat+1)*(n+1) + submat + matsiz;
      return *info;
      }

    k = 1;
    for (j = submat; j < iwork[i]; ++j)
      {
      iwork[indxq+j] = k;
      ++k;
      }
    }

  // Successively merge eigensystems of adjacent submatrices
  // into eigensystem for the corresponding larger matrix.
  while (subpbs > 1)
    {
    for (i=0; i < subpbs-1; i += 2)
      {
      if (i == 0)
        {
        submat = 0;
        matsiz = iwork[1];
        msd2 = iwork[0];
        }
      else
        {
        submat = iwork[i-1];
        matsiz = iwork[i+1] - iwork[i-1];
        msd2 = matsiz / 2;
        }

      // Merge lower order eigensystems (of size MSD2 and MATSIZ - MSD2)
      // into an eigensystem of size MATSIZ.
      // SLAEX1 is used only for the full eigensystem of a tridiagonal
      // matrix.
      if (matsiz == n)
        {
        range2 = range;
        }
      else
        {
        // We need all the eigenvectors if it is not last step
        range2 = MagmaRangeAll;
        }

      magma_slaex1(matsiz, &d[submat], Q + (submat) + (submat) * ldq, ldq,
                   &iwork[indxq+submat], e[submat+msd2-1], msd2,
                   work, &iwork[subpbs], dwork, dwork_offset, queue,
                   range2, vl, vu, il, iu, info);

      if (*info != 0)
        {
        *info = (submat+1)*(n+1) + submat + matsiz;
        return *info;
        }
      iwork[i/2]= iwork[i+1];
      }

    subpbs /= 2;
    }

  // Re-merge the eigenvalues/vectors which were deflated at the final
  // merge step.
  for (i = 0; i < n; ++i)
    {
    j = iwork[indxq+i] - 1;
    work[i] = d[j];
    blas::copy(n, Q + (j) * ldq, ione, &work[n * (i + 1)], ione);
    }

  blas::copy(n, work, ione, d, ione);
  lapack::lacpy('A', n, n, &work[n], n, Q, ldq);

  magma_queue_destroy( queue );

  return *info;
  } /* magma_slaex0 */
