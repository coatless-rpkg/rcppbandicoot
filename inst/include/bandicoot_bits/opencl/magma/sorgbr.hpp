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



// SORGBR generates one of the real orthogonal matrices Q or P**H
// determined by SGEBRD when reducing a real matrix A to bidiagonal
// form: A = Q * B * P**H.  Q and P**H are defined as products of
// elementary reflectors H(i) or G(i) respectively.



inline
magma_int_t
magma_sorgbr
  (
  magma_vect_t vect, magma_int_t m, magma_int_t n, magma_int_t k,
  float *A, magma_int_t lda,
  float *tau,
  float *work, magma_int_t lwork,
  magma_int_t *info
  )
  {
  // Constants
  const float c_zero = MAGMA_S_ZERO;
  const float c_one  = MAGMA_S_ONE;
  magma_int_t ineg_one = -1;

  // Local variables
  bool lquery, wantq;
  magma_int_t i, iinfo, j, lwkmin, lwkopt, min_mn;

  // Test the input arguments
  *info = 0;
  wantq = (vect == MagmaQ);
  min_mn = std::min( m, n );
  lquery = (lwork == -1);
  if (!wantq && vect != MagmaP)
    *info = -1;
  else if (m < 0)
    *info = -2;
  else if (n < 0 || (wantq && (n > m || n < std::min(m,k))) || ( !wantq && (m > n || m < std::min(n,k))))
    *info = -3;
  else if (k < 0)
    *info = -4;
  else if (lda < std::max(1, m))
    *info = -6;

  // Check workspace size
  if (*info == 0)
    {
    work[0] = c_one;
    if (wantq)
      {
      if (m >= k)
        {
        // magma_sorgqr takes dT instead of work
        // magma_sorgqr2 doesn't take work
        //magma_sorgqr2( m, n, k, A, lda, tau, work, -1, &iinfo );
        lapack::orgqr(m, n, k, A, lda, tau, work, ineg_one, &iinfo);
        }
      else if (m > 1)
        {
        //magma_sorgqr2( m-1, m-1, m-1, A(1,1), lda, tau, work, -1, &iinfo );
        magma_int_t m1 = m-1;
        lapack::orgqr(m1, m1, m1, &A[1 + 1 * lda], lda, tau, work, ineg_one, &iinfo);
        }

      lwkopt = work[0];
      lwkmin = min_mn;
      }
    else
      {
      if (k < n)
        {
        magma_sorglq( m, n, k, A, lda, tau, work, -1, &iinfo );
        }
      else if (n > 1)
        {
        magma_sorglq( n-1, n-1, n-1, &A[1 + 1 * lda], lda, tau, work, -1, &iinfo );
        }

      lwkopt = work[0];
      lwkmin = lwkopt;
      }

    if (lwork < lwkmin && !lquery)
      {
      *info = -9;
      }
    }

  if (*info != 0)
    {
    // magma_xerbla( __func__, -(*info) );
    return *info;
    }
  else if (lquery)
    {
    work[0] = magma_smake_lwork( lwkopt );
    return *info;
    }

  // Quick return if possible
  if (m == 0 || n == 0)
    {
    work[0] = c_one;
    return *info;
    }

  if (wantq)
    {
    // Form Q, determined by a call to DGEBRD to reduce an m-by-k
    // matrix
    if (m >= k)
      {
      // If m >= k, assume m >= n >= k
      magma_sorgqr2( m, n, k, A, lda, tau, /*work, lwork,*/ &iinfo );
      }
    else
      {
      // If m < k, assume m = n

      // Shift the vectors which define the elementary reflectors one
      // column to the right, and set the first row and column of Q
      // to those of the unit matrix
      for (j=m-1; j >= 1; --j)
        {
        A[0 + j * lda] = c_zero;
        for (i=j + 1; i < m; ++i)
          {
          A[i + j * lda] = A[i + (j-1) * lda];
          }
        }
      A[0] = c_one;
      for (i=1; i < m; ++i)
        {
        A[i] = c_zero;
        }
      if (m > 1)
        {
        // Form Q(2:m,2:m)
        magma_sorgqr2( m-1, m-1, m-1, &A[1 + 1 * lda], lda, tau, /*work, lwork,*/ &iinfo );
        }
      }
    }
  else
    {
    // Form P**H, determined by a call to DGEBRD to reduce a k-by-n
    // matrix
    if (k < n)
      {
      // If k < n, assume k <= m <= n
      magma_sorglq( m, n, k, A, lda, tau, work, lwork, &iinfo );
      }
    else
      {
      // If k >= n, assume m = n

      // Shift the vectors which define the elementary reflectors one
      // row downward, and set the first row and column of P**H to
      // those of the unit matrix
      A[0] = c_one;
      for (i=1; i < n; ++i)
        {
        A[i] = c_zero;
        }

      for (j=1; j < n; ++j)
        {
        for (i=j-1; i >= 1; --i)
          {
          A[i + j * lda] = A[i-1 + j * lda];
          }
        A[j * lda] = c_zero;
        }

      if (n > 1)
        {
        // Form P**H(2:n,2:n)
        magma_sorglq( n-1, n-1, n-1, &A[1 + 1 * lda], lda, tau, work, lwork, &iinfo );
        }
      }
    }

  work[0] = magma_smake_lwork( lwkopt );
  return *info;
  }
