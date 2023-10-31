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
// SSTEDX computes some eigenvalues and, optionally, eigenvectors of a
// symmetric tridiagonal matrix using the divide and conquer method.
//
// This code makes very mild assumptions about floating point
// arithmetic. It will work on machines with a guard digit in
// add/subtract, or on those binary machines without guard digits
// which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or Cray-2.
// It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.  See SLAEX3 for details.



inline
magma_int_t
magma_sstedx
  (
  magma_range_t range, magma_int_t n, float vl, float vu,
  magma_int_t il, magma_int_t iu, float *d, float *e,
  float *Z, magma_int_t ldz,
  float *work, magma_int_t lwork,
  magma_int_t *iwork, magma_int_t liwork,
  magmaFloat_ptr dwork, size_t dwork_offset,
  magma_int_t *info
  )
  {
  float d_zero = 0.;
  float d_one  = 1.;
  magma_int_t izero = 0;
  magma_int_t ione = 1;

  magma_int_t alleig, indeig, valeig, lquery;
  magma_int_t i, j, k, m;
  magma_int_t liwmin, lwmin;
  magma_int_t start, end, smlsiz;
  float eps, orgnrm, p, tiny;

  // Test the input parameters.

  alleig = (range == MagmaRangeAll);
  valeig = (range == MagmaRangeV);
  indeig = (range == MagmaRangeI);
  lquery = (lwork == -1 || liwork == -1);

  *info = 0;

  if (! (alleig || valeig || indeig))
    {
    *info = -1;
    }
  else if (n < 0)
    {
    *info = -2;
    }
  else if (ldz < std::max(1,n))
    {
    *info = -10;
    }
  else
    {
    if (valeig)
      {
      if (n > 0 && vu <= vl)
        {
        *info = -4;
        }
      }
    else if (indeig)
      {
      if (il < 1 || il > std::max(1,n))
        {
        *info = -5;
        }
      else if (iu < std::min(n,il) || iu > n)
        {
        *info = -6;
        }
      }
    }

  if (*info == 0)
    {
    // Compute the workspace requirements

    smlsiz = magma_get_smlsize_divideconquer();
    if ( n <= 1 )
      {
      lwmin = 1;
      liwmin = 1;
      }
    else
      {
      lwmin = 1 + 4*n + n*n;
      liwmin = 3 + 5*n;
      }

    work[0] = magma_smake_lwork( lwmin );
    iwork[0] = liwmin;

    if (lwork < lwmin && ! lquery)
      {
      *info = -12;
      }
    else if (liwork < liwmin && ! lquery)
      {
      *info = -14;
      }
    }

  if (*info != 0)
    {
    //magma_xerbla( __func__, -(*info));
    return *info;
    }
  else if (lquery)
    {
    return *info;
    }

  // Quick return if possible
  if (n == 0)
    {
    return *info;
    }

  if (n == 1)
    {
    *Z = 1.;
    return *info;
    }

  // If N is smaller than the minimum divide size (SMLSIZ+1), then
  // solve the problem with another solver.

  if (n < smlsiz)
    {
    lapack::steqr('I', n, d, e, Z, ldz, work, info);
    }
  else
    {
    lapack::laset('F', n, n, d_zero, d_one, Z, ldz);

    //Scale.
    orgnrm = lapack::lanst('M', n, d, e);

    if (orgnrm == 0)
      {
      work[0]  = magma_smake_lwork( lwmin );
      iwork[0] = liwmin;
      return *info;
      }

    eps = lapack::lamch<float>('E');

    if (alleig)
      {
      start = 0;
      while ( start < n )
        {
        // Let FINISH be the position of the next subdiagonal entry
        // such that E( END ) <= TINY or FINISH = N if no such
        // subdiagonal exists.  The matrix identified by the elements
        // between START and END constitutes an independent
        // sub-problem.

        for (end = start+1; end < n; ++end)
          {
          tiny = eps * std::sqrt( std::abs(d[end-1]*d[end]));
          if (std::abs(e[end-1]) <= tiny)
            {
            break;
            }
          }

        // (Sub) Problem determined.  Compute its size and solve it.

        m = end - start;
        if (m == 1)
          {
          start = end;
          continue;
          }

        if (m > smlsiz)
          {
          // Scale
          orgnrm = lapack::lanst('M', m, &d[start], &e[start]);
          lapack::lascl('G', izero, izero, orgnrm, d_one, m, ione, &d[start], m, info);
          magma_int_t mm = m-1;
          lapack::lascl('G', izero, izero, orgnrm, d_one, mm, ione, &e[start], mm, info);

          magma_slaex0( m, &d[start], &e[start], Z + start + start * ldz, ldz, work, iwork, dwork, dwork_offset, MagmaRangeAll, vl, vu, il, iu, info);

          if ( *info != 0)
            {
            return *info;
            }

          // Scale Back
          lapack::lascl('G', izero, izero, d_one, orgnrm, m, ione, &d[start], m, info);
          }
        else
          {
          lapack::steqr('I', m, &d[start], &e[start], Z + start + start * ldz, ldz, work, info);
          if (*info != 0)
            {
            *info = (start+1) *(n+1) + end;
            }
          }

        start = end;
        }

      // If the problem split any number of times, then the eigenvalues
      // will not be properly ordered.  Here we permute the eigenvalues
      // (and the associated eigenvectors) into ascending order.

      if (m < n)
        {
        // Use Selection Sort to minimize swaps of eigenvectors
        for (i = 1; i < n; ++i)
          {
          k = i-1;
          p = d[i-1];
          for (j = i; j < n; ++j)
            {
            if (d[j] < p)
              {
              k = j;
              p = d[j];
              }
            }

          if (k != i-1)
            {
            d[k] = d[i-1];
            d[i-1] = p;
            blas::swap(n, Z + (i-1) * ldz, ione, Z + k * ldz, ione);
            }
          }
        }
      }
    else
      {
      // Scale
      lapack::lascl('G', izero, izero, orgnrm, d_one, n, ione, d, n, info);
      magma_int_t nm = n-1;
      lapack::lascl('G', izero, izero, orgnrm, d_one, nm, ione, e, nm, info);

      magma_slaex0( n, d, e, Z, ldz, work, iwork, dwork, dwork_offset, range, vl, vu, il, iu, info);

      if ( *info != 0)
        {
        return *info;
        }

      // Scale Back
      lapack::lascl('G', izero, izero, d_one, orgnrm, n, ione, d, n, info);
      }
    }

  work[0]  = magma_smake_lwork( lwmin );
  iwork[0] = liwmin;

  return *info;
  } /* magma_sstedx */
