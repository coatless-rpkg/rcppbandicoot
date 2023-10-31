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



inline
magma_int_t
magma_get_slaed3_k()
  {
  return 512;
  }



inline
void
magma_svrange(magma_int_t k, float *d, magma_int_t *il, magma_int_t *iu, float vl, float vu)
  {
  magma_int_t i;

  *il = 1;
  *iu = k;
  for (i = 0; i < k; ++i)
    {
    if (d[i] > vu)
      {
      *iu = i;
      break;
      }
    else if (d[i] < vl)
      {
      ++*il;
      }
    }
  return;
  }



inline
void
magma_sirange(magma_int_t k, magma_int_t *indxq, magma_int_t *iil, magma_int_t *iiu, magma_int_t il, magma_int_t iu)
  {
  magma_int_t i;

  *iil = 1;
  *iiu = 0;
  for (i = il; i <= iu; ++i)
    {
    if (indxq[i-1] <= k)
      {
      *iil = indxq[i-1];
      break;
      }
    }

  for (i = iu; i >= il; --i)
    {
    if (indxq[i-1] <= k)
      {
      *iiu = indxq[i-1];
      break;
      }
    }

  return;
  }



// Purpose
// -------
// SLAEX3 finds the roots of the secular equation, as defined by the
// values in D, W, and RHO, between 1 and K.  It makes the
// appropriate calls to SLAED4 and then updates the eigenvectors by
// multiplying the matrix of eigenvectors of the pair of eigensystems
// being combined by the matrix of eigenvectors of the K-by-K system
// which is solved here.
//
// It is used in the last step when only a part of the eigenvectors
// is required. It computes only the required portion of the eigenvectors
// and the rest is not used.
//
// This code makes very mild assumptions about floating point
// arithmetic. It will work on machines with a guard digit in
// add/subtract, or on those binary machines without guard digits
// which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or Cray-2.
// It could conceivably fail on hexadecimal or decimal machines
// without guard digits, but we know of none.



inline
magma_int_t
magma_slaex3
  (
  magma_int_t k, magma_int_t n, magma_int_t n1,
  float *d,
  float *Q, magma_int_t ldq, float rho,
  float *dlamda, float *Q2, magma_int_t *indx,
  magma_int_t *ctot, float *w, float *s, magma_int_t *indxq,
  magmaFloat_ptr dwork, size_t dwork_offset,
  magma_queue_t queue,
  magma_range_t range, float vl, float vu, magma_int_t il, magma_int_t iu,
  magma_int_t *info
  )
  {
  float d_one  = 1.;
  float d_zero = 0.;
  magma_int_t ione = 1;
  magma_int_t ineg_one = -1;

  magma_int_t iil, iiu, rk;

  magma_int_t lddq = n/2 + 1;
  magmaFloat_ptr dQ2 = dwork;
  size_t dQ2_offset = dwork_offset;
  magmaFloat_ptr dS  = dQ2;
  size_t dS_offset = dQ2_offset + n * lddq;
  magmaFloat_ptr dQ  = dS;
  size_t dQ_offset = dS_offset + n * lddq;

  magma_int_t i, iq2, j, n12, n2, n23, tmp, lq2;
  float temp;
  magma_int_t alleig, valeig, indeig;

  alleig = (range == MagmaRangeAll);
  valeig = (range == MagmaRangeV);
  indeig = (range == MagmaRangeI);

  *info = 0;

  if (k < 0)
    {
    *info = -1;
    }
  else if (n < k)
    {
    *info = -2;
    }
  else if (ldq < std::max(1,n))
    {
    *info = -6;
    }
  else if (! (alleig || valeig || indeig))
    {
    *info = -15;
    }
  else
    {
    if (valeig)
      {
      if (n > 0 && vu <= vl)
        {
        *info = -17;
        }
      }
    else if (indeig)
      {
      if (il < 1 || il > std::max(1,n))
        {
        *info = -18;
        }
      else if (iu < std::min(n,il) || iu > n)
        {
        *info = -19;
        }
      }
    }

  if (*info != 0)
    {
    //magma_xerbla(__func__, -(*info));
    return *info;
    }

  // Quick return if possible
  if (k == 0)
    {
    return *info;
    }

  /*
   Modify values DLAMDA(i) to make sure all DLAMDA(i)-DLAMDA(j) can
   be computed with high relative accuracy (barring over/underflow).
   This is a problem on machines without a guard digit in
   add/subtract (Cray XMP, Cray YMP, Cray C 90 and Cray 2).
   The following code replaces DLAMDA(I) by 2*DLAMDA(I)-DLAMDA(I),
   which on any of these machines zeros out the bottommost
   bit of DLAMDA(I) if it is 1; this makes the subsequent
   subtractions DLAMDA(I)-DLAMDA(J) unproblematic when cancellation
   occurs. On binary machines with a guard digit (almost all
   machines) it does not change DLAMDA(I) at all. On hexadecimal
   and decimal machines with a guard digit, it slightly
   changes the bottommost bits of DLAMDA(I). It does not account
   for hexadecimal or decimal machines without guard digits
   (we know of none). We use a subroutine call to compute
   2*DLAMBDA(I) to prevent optimizing compilers from eliminating
   this code.*/

  n2 = n - n1;

  n12 = ctot[0] + ctot[1];
  n23 = ctot[1] + ctot[2];

  iq2 = n1 * n12;
  lq2 = iq2 + n2 * n23;

  magma_ssetvector_async( lq2, Q2, 1, dQ2, dQ2_offset, 1, queue );

#ifdef COOT_USE_OPENMP
  // -------------------------------------------------------------------------
  // openmp implementation
  // -------------------------------------------------------------------------

  #pragma omp parallel private(i, j, tmp, temp)
    {
    magma_int_t tid     = omp_get_thread_num();
    magma_int_t nthread = omp_get_num_threads();

    magma_int_t ibegin = ( tid    * k) / nthread; // start index of local loop
    magma_int_t iend   = ((tid+1) * k) / nthread; // end   index of local loop
    magma_int_t ik     = iend - ibegin;           // number of local indices

    for (i = ibegin; i < iend; ++i)
      {
      dlamda[i] = lapackf77_slamc3(&dlamda[i], &dlamda[i]) - dlamda[i];
      }

    for (j = ibegin; j < iend; ++j)
      {
      magma_int_t tmpp = j+1;
      magma_int_t iinfo = 0;
      lapackf77_slaed4(&k, &tmpp, dlamda, w, Q + j * ldq, &rho, &d[j], &iinfo);
      // If the zero finder fails, the computation is terminated.
      if (iinfo != 0)
        {
        #pragma omp critical (magma_slaex3)
        *info = iinfo;
        break;
        }
      }

    #pragma omp barrier

    if (*info == 0)
      {
      #pragma omp single
        {
        // Prepare the INDXQ sorting permutation.
        magma_int_t nk = n - k;
        lapack::lamrg(k, nk, d, ione, ineg_one, indxq);

        // compute the lower and upper bound of the non-deflated eigenvectors
        if (valeig)
          {
          magma_svrange(k, d, &iil, &iiu, vl, vu);
          }
        else if (indeig)
          {
          magma_sirange(k, indxq, &iil, &iiu, il, iu);
          }
        else
          {
          iil = 1;
          iiu = k;
          }
        rk = iiu - iil + 1;
        }

      if (k == 2)
        {
        #pragma omp single
          {
          for (j = 0; j < k; ++j)
            {
            w[0] = *(Q +     j * ldq);
            w[1] = *(Q + 1 + j * ldq);

            i = indx[0] - 1;
            *(Q + j * ldq) = w[i];
            i = indx[1] - 1;
            *(Q + 1 + j * ldq) = w[i];
            }
          }
        }
      else if (k != 1)
        {
        // Compute updated W.
        blas::copy(ik, &w[ibegin], ione, &s[ibegin], ione);

        // Initialize W(I) = Q(I,I)
        tmp = ldq + 1;
        blas::copy(ik, Q + ibegin + ibegin * ldq, tmp, &w[ibegin], ione);

        for (j = 0; j < k; ++j)
          {
          magma_int_t i_tmp = std::min(j, iend);
          for (i = ibegin; i < i_tmp; ++i)
            {
            w[i] = w[i] * ( *(Q + i + j * ldq) / ( dlamda[i] - dlamda[j] ) );
            }
          i_tmp = std::max(j+1, ibegin);
          for (i = i_tmp; i < iend; ++i)
            {
            w[i] = w[i] * ( *(Q + i + j * ldq) / ( dlamda[i] - dlamda[j] ) );
            }
          }

        for (i = ibegin; i < iend; ++i)
          {
          w[i] = std::copysign( std::sqrt( -w[i] ), s[i]);
          }

        #pragma omp barrier

        // reduce the number of threads used to have enough S workspace
        nthread = min(n1, omp_get_num_threads());

        if (tid < nthread)
          {
          ibegin = ( tid    * rk) / nthread + iil - 1;
          iend   = ((tid+1) * rk) / nthread + iil - 1;
          ik     = iend - ibegin;
          }
        else
          {
          ibegin = -1;
          iend   = -1;
          ik     = -1;
          }

        // Compute eigenvectors of the modified rank-1 modification.
        for (j = ibegin; j < iend; ++j)
          {
          for (i = 0; i < k; ++i)
            {
            s[tid*k + i] = w[i] / *(Q + i + j * ldq);
            }
          temp = magma_cblas_snrm2( k, s + tid*k, 1 );
          for (i = 0; i < k; ++i)
            {
            magma_int_t iii = indx[i] - 1;
            *(Q + i + j * ldq) = s[tid*k + iii] / temp;
            }
          }
        }
      }
    }  // end omp parallel

  if (*info != 0)
    {
    return *info;
    }

#else
  // -------------------------------------------------------------------------
  // Non openmp implementation
  // -------------------------------------------------------------------------

  for (i = 0; i < k; ++i)
    {
    dlamda[i] = lapack::lamc3(&dlamda[i], &dlamda[i]) - dlamda[i];
    }

  for (j = 0; j < k; ++j)
    {
    magma_int_t tmpp = j+1;
    magma_int_t iinfo = 0;
    lapack::laed4(k, tmpp, dlamda, w, Q + j * ldq, rho, &d[j], &iinfo);
    // If the zero finder fails, the computation is terminated.
    if (iinfo != 0)
      {
      *info = iinfo;
      }
    }

  if (*info != 0)
    {
    return *info;
    }

  // Prepare the INDXQ sorting permutation.
  magma_int_t nk = n - k;
  lapack::lamrg(k, nk, d, ione, ineg_one, indxq);

  // compute the lower and upper bound of the non-deflated eigenvectors
  if (valeig)
    {
    magma_svrange(k, d, &iil, &iiu, vl, vu);
    }
  else if (indeig)
    {
    magma_sirange(k, indxq, &iil, &iiu, il, iu);
    }
  else
    {
    iil = 1;
    iiu = k;
    }
  rk = iiu - iil + 1;

  if (k == 2)
    {
    for (j = 0; j < k; ++j)
      {
      w[0] = *(Q +     j * ldq);
      w[1] = *(Q + 1 + j * ldq);

      i = indx[0] - 1;
      *(Q + j * ldq) = w[i];
      i = indx[1] - 1;
      *(Q + 1 + j * ldq) = w[i];
      }
    }
  else if (k != 1)
    {
    // Compute updated W.
    blas::copy(k, w, ione, s, ione);

    // Initialize W(I) = Q(I,I)
    tmp = ldq + 1;
    blas::copy(k, Q, tmp, w, ione);

    for (j = 0; j < k; ++j)
      {
      for (i = 0; i < j; ++i)
        {
        w[i] = w[i] * ( *(Q + i + j * ldq) / ( dlamda[i] - dlamda[j] ) );
        }
      for (i = j+1; i < k; ++i)
        {
        w[i] = w[i] * ( *(Q + i + j * ldq) / ( dlamda[i] - dlamda[j] ) );
        }
      }

    for (i = 0; i < k; ++i)
      {
      w[i] = std::copysign( std::sqrt( -w[i] ), s[i]);
      }

    // Compute eigenvectors of the modified rank-1 modification.
    for (j = iil-1; j < iiu; ++j)
      {
      for (i = 0; i < k; ++i)
        {
        s[i] = w[i] / *(Q + i + j * ldq);
        }
      temp = magma_cblas_snrm2( k, s, 1 );
      for (i = 0; i < k; ++i)
        {
        magma_int_t iii = indx[i] - 1;
        *(Q + i + j * ldq) = s[iii] / temp;
        }
      }
    }

#endif // COOT_USE_OPENMP
  // Compute the updated eigenvectors.

  if (rk != 0)
    {
    if ( n23 != 0 )
      {
      if (rk < magma_get_slaed3_k())
        {
        lapack::lacpy('A', n23, rk, Q + ctot[0] + (iil-1) * ldq, ldq, s, n23);
        blas::gemm('N', 'N', n2, rk, n23, d_one, &Q2[iq2], n2,
                   s, n23, d_zero, Q + n1 + (iil-1) * ldq, ldq);
        }
      else
        {
        magma_ssetmatrix( n23, rk, Q + ctot[0] + (iil-1) * ldq, ldq, dS, dS_offset, n23, queue );
        magma_sgemm( MagmaNoTrans, MagmaNoTrans, n2, rk, n23,
                     d_one,  dQ2, dQ2_offset + iq2, n2,
                             dS,  dS_offset, n23,
                     d_zero, dQ,  dQ_offset, lddq, queue );
        magma_sgetmatrix( n2, rk, dQ, dQ_offset, lddq, Q + n1 + (iil-1) * ldq, ldq, queue );
        }
      }
    else
      {
      lapack::laset('A', n2, rk, d_zero, d_zero, Q + n1 + (iil-1) * ldq, ldq);
      }

    if ( n12 != 0 )
      {
      if (rk < magma_get_slaed3_k())
        {
        lapack::lacpy('A', n12, rk, Q + (iil-1) * ldq, ldq, s, n12);
        blas::gemm('N', 'N', n1, rk, n12, d_one, Q2, n1,
                   s, n12, d_zero, Q + (iil-1) * ldq, ldq);
        }
      else
        {
        magma_ssetmatrix( n12, rk, Q + (iil-1) * ldq, ldq, dS, dS_offset, n12, queue );
        magma_sgemm( MagmaNoTrans, MagmaNoTrans, n1, rk, n12,
                     d_one,  dQ2, dQ2_offset, n1,
                             dS,  dS_offset, n12,
                     d_zero, dQ,  dQ_offset, lddq, queue );
        magma_sgetmatrix( n1, rk, dQ, dQ_offset, lddq, Q + (iil-1) * ldq, ldq, queue );
        }
      }
    else
      {
      lapack::laset('A', n1, rk, d_zero, d_zero, Q + (iil-1) * ldq, ldq);
      }
    }

  return *info;
  } /* magma_slaex3 */
