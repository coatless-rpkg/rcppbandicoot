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
// DLAEX1 computes the updated eigensystem of a diagonal
// matrix after modification by a rank-one symmetric matrix.
//
//   T = Q(in) ( D(in) + RHO * Z*Z' ) Q'(in) = Q(out) * D(out) * Q'(out)
//
// where Z = Q'u, u is a vector of length N with ones in the
// CUTPNT and CUTPNT + 1 th elements and zeros elsewhere.
//
// The eigenvectors of the original matrix are stored in Q, and the
// eigenvalues are in D.  The algorithm consists of three stages:
//
// The first stage consists of deflating the size of the problem
// when there are multiple eigenvalues or if there is a zero in
// the Z vector.  For each such occurence the dimension of the
// secular equation problem is reduced by one.  This stage is
// performed by the routine DLAED2.
//
// The second stage consists of calculating the updated
// eigenvalues. This is done by finding the roots of the secular
// equation via the routine DLAED4 (as called by DLAED3).
// This routine also calculates the eigenvectors of the current
// problem.
//
// The final stage consists of computing the updated eigenvectors
// directly using the updated eigenvalues.  The eigenvectors for
// the current problem are multiplied with the eigenvectors from
// the overall problem.



inline
magma_int_t
magma_dlaex1
  (
  magma_int_t n,
  double *d,
  double *Q, magma_int_t ldq,
  magma_int_t *indxq, double rho, magma_int_t cutpnt,
  double *work, magma_int_t *iwork,
  magmaDouble_ptr dwork, size_t dwork_offset,
  magma_queue_t queue,
  magma_range_t range, double vl, double vu,
  magma_int_t il, magma_int_t iu,
  magma_int_t *info
  )
  {
  magma_int_t coltyp, i, idlmda;
  magma_int_t indx, indxc, indxp;
  magma_int_t iq2, is, iw, iz, k, tmp;
  magma_int_t ione = 1;

  //  Test the input parameters.
  *info = 0;

  if ( n < 0 )
    {
    *info = -1;
    }
  else if ( ldq < std::max(1, n) )
    {
    *info = -4;
    }
  else if ( std::min( 1, n/2 ) > cutpnt || n/2 < cutpnt )
    {
    *info = -7;
    }

  if ( *info != 0 )
    {
    //magma_xerbla( __func__, -(*info) );
    return *info;
    }

  //  Quick return if possible
  if ( n == 0 )
    {
    return *info;
    }

  //  The following values are integer pointers which indicate
  //  the portion of the workspace
  //  used by a particular array in DLAED2 and DLAED3.
  iz = 0;
  idlmda = iz + n;
  iw = idlmda + n;
  iq2 = iw + n;

  indx = 0;
  indxc = indx + n;
  coltyp = indxc + n;
  indxp = coltyp + n;

  //  Form the z-vector which consists of the last row of Q_1 and the
  //  first row of Q_2.
  blas::copy(cutpnt, Q + (cutpnt-1), ldq, &work[iz], ione);
  tmp = n-cutpnt;
  blas::copy(tmp, Q + (cutpnt) + (cutpnt) * ldq, ldq, &work[iz+cutpnt], ione);

  //  Deflate eigenvalues.
  lapack::laed2(&k, n, cutpnt, d, Q, ldq, indxq, &rho, &work[iz],
                &work[idlmda], &work[iw], &work[iq2],
                &iwork[indx], &iwork[indxc], &iwork[indxp],
                &iwork[coltyp], info);

  if ( *info != 0 )
    {
    return *info;
    }

  //  Solve Secular Equation.
  if ( k != 0 )
    {
    is = (iwork[coltyp]+iwork[coltyp+1])*cutpnt + (iwork[coltyp+1]+iwork[coltyp+2])*(n-cutpnt) + iq2;
    magma_dlaex3(k, n, cutpnt, d, Q, ldq, rho,
                 &work[idlmda], &work[iq2], &iwork[indxc],
                 &iwork[coltyp], &work[iw], &work[is],
                 indxq, dwork, dwork_offset, queue, range, vl, vu, il, iu, info );
    if ( *info != 0 )
      {
      return *info;
      }
    }
  else
    {
    for (i = 0; i < n; ++i)
      {
      indxq[i] = i+1;
      }
    }

  return *info;
  } /* magma_dlaex1 */
