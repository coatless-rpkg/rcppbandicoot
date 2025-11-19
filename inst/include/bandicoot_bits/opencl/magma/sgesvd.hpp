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



// DGESVD computes the singular value decomposition (SVD) of a real
// M-by-N matrix A, optionally computing the left and/or right singular
// vectors. The SVD is written
//
//     A = U * SIGMA * transpose(V)
//
// where SIGMA is an M-by-N matrix which is zero except for its
// min(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and
// V is an N-by-N orthogonal matrix.  The diagonal elements of SIGMA
// are the singular values of A; they are real and non-negative, and
// are returned in descending order.  The first min(m,n) columns of
// U and V are the left and right singular vectors of A.
//
// Note that the routine returns VT = V**T, not V.



inline
magma_int_t
magma_get_sgesvd_nb
  (
  magma_int_t m,
  magma_int_t n
  )
  {
  return magma_get_sgebrd_nb(m, n);
  }



inline
magma_int_t
magma_sgesvd
  (
  magma_vec_t jobu, magma_vec_t jobvt, magma_int_t m, magma_int_t n,
  float *A,    magma_int_t lda, float *s,
  float *U,    magma_int_t ldu,
  float *VT,   magma_int_t ldvt,
  float *work, magma_int_t lwork,
  magma_int_t *info
  )
  {
  // Constants
  const float c_zero = MAGMA_S_ZERO;
  const float c_one  = MAGMA_S_ONE;
  const magma_int_t izero    = 0;
  const magma_int_t ione     = 1;
  magma_int_t ineg_one = -1;

  // System generated locals
  magma_int_t lwork2, m_1, n_1;

  // Local variables
  magma_int_t i, ie, ir, iu, ib, ncu;
  float dummy[1], unused[1], eps;
  magma_int_t nru = 0, iscl;
  magma_int_t ierr, itau, ncvt = 0, nrvt;
  magma_int_t chunk, minmn, wrkbl, itaup, itauq, mnthr, iwork;
  magma_int_t ldwrkr, ldwrku, maxwrk, minwrk, gemm_nb;
  float anrm, bignum, smlnum;

  // Parameter adjustments for Fortran indexing
  --work;

  // Function Body
  *info = 0;
  minmn = std::min( m, n );
  ie = 1;

  const bool want_ua  = (jobu == MagmaAllVec);
  const bool want_us  = (jobu == MagmaSomeVec);
  const bool want_uo  = (jobu == MagmaOverwriteVec);
  const bool want_un  = (jobu == MagmaNoVec);
  const bool want_uas = (want_ua || want_us);

  const bool want_va  = (jobvt == MagmaAllVec);
  const bool want_vs  = (jobvt == MagmaSomeVec);
  const bool want_vo  = (jobvt == MagmaOverwriteVec);
  const bool want_vn  = (jobvt == MagmaNoVec);
  const bool want_vas = (want_va || want_vs);

  const bool lquery = (lwork == -1);

  // Test the input arguments
  if (!(want_ua || want_us || want_uo || want_un))
    *info = -1;
  else if (! (want_va || want_vs || want_vo || want_vn) || (want_vo && want_uo) )
    *info = -2;
  else if (m < 0)
    *info = -3;
  else if (n < 0)
    *info = -4;
  else if (lda < std::max(1,m))
    *info = -6;
  else if ((ldu < 1) || (want_uas && (ldu < m)) )
    *info = -9;
  else if ((ldvt < 1) || (want_va && (ldvt < n)) || (want_vs && (ldvt < minmn)) )
    *info = -11;

  gemm_nb = 64;

  // Compute workspace
  // (Note: Comments in the code beginning "Workspace:" describe the
  //  minimal amount of workspace needed at that point in the code,
  //  as well as the preferred amount for good performance.
  //  NB refers to the optimal block size for the immediately
  //  following subroutine, as returned by ILAENV or magma_get_*_nb.)
  minwrk = 1;
  maxwrk = 1;
  wrkbl  = 1;
  mnthr  = magma_int_t( minmn * 1.6 );
  if (*info == 0)
    {
    if (m >= n && minmn > 0)
      {
      // Compute space needed for DBDSQR
      // LAPACK adds n for [e] here; that is separated here for consistency
      magma_int_t lwork_sbdsqr = 4*n;

      // Compute space preferred for each routine
      // For MAGMA, these are all required
      magma_sgebrd(      m,  n, unused,  m, unused, unused, unused, unused, dummy,  ineg_one, &ierr );
      magma_int_t lwork_sgebrd_mn = magma_int_t( dummy[0] );

      magma_sgebrd(      n,  n, unused,  n, unused, unused, unused, unused, dummy,  ineg_one, &ierr );
      magma_int_t lwork_sgebrd_nn = magma_int_t( dummy[0] );

      magma_sgeqrf(      m,  n, unused,  m, unused, dummy,  ineg_one, &ierr );
      magma_int_t lwork_sgeqrf_mn = magma_int_t( dummy[0] );

      magma_sorgbr( MagmaP,   n,  n,  n, unused,  n, unused, dummy,  ineg_one, &ierr );
      magma_int_t lwork_sorgbr_p_nn = magma_int_t( dummy[0] );

      magma_sorgbr( MagmaQ,   m,  n,  n, unused,  m, unused, dummy,  ineg_one, &ierr );
      magma_int_t lwork_sorgbr_q_mn = magma_int_t( dummy[0] );

      magma_sorgbr( MagmaQ,   m,  m,  n, unused,  m, unused, dummy,  ineg_one, &ierr );
      magma_int_t lwork_sorgbr_q_mm = magma_int_t( dummy[0] );

      magma_sorgbr( MagmaQ,   n,  n,  n, unused,  n, unused, dummy,  ineg_one, &ierr );
      magma_int_t lwork_sorgbr_q_nn = magma_int_t( dummy[0] );

      // magma_sorgqr2 does not take workspace; use LAPACK's for compatability
      lapack::orgqr(m, m, n, unused, m, unused, dummy, ineg_one, &ierr);
      magma_int_t lwork_sorgqr_mm = magma_int_t( dummy[0] );

      lapack::orgqr(m, n, n, unused, m, unused, dummy, ineg_one, &ierr);
      magma_int_t lwork_sorgqr_mn = magma_int_t( dummy[0] );

      // missing from LAPACK, since it occurs only in slow paths
      magma_sormbr( MagmaQ, MagmaRight, MagmaNoTrans,  m,  n,  n, unused,  m, unused, unused,  m, dummy,  ineg_one, &ierr );
      magma_int_t lwork_sormbr_qrn_mn = magma_int_t( dummy[0] );

      // wrkbl is everything except R and U matrices.
      // It is used later to compute ldwrkr for R and ldwrku for U.
      // For minwrk, LAPACK used min workspace for each routine (usually m or n);
      // but that size doesn't work for MAGMA, as m*nb or n*nb is usually the min,
      // so here we track MAGMA's min workspace.
      if (m >= mnthr)
        {
        if (want_un)
          {
          // Path 1 (M >> N, JOBU='N')
          wrkbl  = std::max( wrkbl,      n + lwork_sgeqrf_mn   );  // geqrf  = n or n*nb
          wrkbl  = std::max( wrkbl,    3*n + lwork_sgebrd_nn   );  // gebrd  = n or 2*n*nb
          if (want_vo || want_vas)
            {
            wrkbl = std::max( wrkbl, 3*n + lwork_sorgbr_p_nn );   // orgbr  = n or n*nb
            }
          wrkbl  = std::max( wrkbl,      n + lwork_sbdsqr      );  // bdsqr  = 4*n
          maxwrk = wrkbl;                                     // maxwrk = 3*n + 2*n*nb
          minwrk = maxwrk;                                    // minwrk = 3*n + 2*n*nb
          //                                              lapack minwrk = 5*n
          }
        else if (want_uo && want_vn)
          {
          // Path 2 (M >> N, JOBU='O', JOBVT='N')
          // Path 2-fast
          wrkbl  = std::max( wrkbl,   n + lwork_sgeqrf_mn      );  // geqrf  = n or n*nb
          wrkbl  = std::max( wrkbl,   n + lwork_sorgqr_mn      );  // orgqr  = n or n*nb
          wrkbl  = std::max( wrkbl, 3*n + lwork_sgebrd_nn      );  // gebrd  = n or 2*n*nb
          wrkbl  = std::max( wrkbl, 3*n + lwork_sorgbr_q_nn    );  // orgbr  = n or n*nb
          wrkbl  = std::max( wrkbl,   n + lwork_sbdsqr         );  // bdsqr  = 4*n
          // todo: is n*gemm_nb enough? LAPACK has m*n
          maxwrk = n*n + std::max( wrkbl, n + n*gemm_nb        );  // maxwrk = n*n + 3*n + 2*n*nb
          //                                              lapack maxwrk = n*n + max(3*n + 2*n*nb, n + m*n)
          //                                              lapack minwrk = n*n + 5*n

          // Path 2-slow
          minwrk = std::max( minwrk, 3*n + lwork_sgebrd_mn     );  // gebrd  = m or (m+n)*nb (note m)
          minwrk = std::max( minwrk, 3*n + lwork_sorgbr_q_mn   );  // orgbr  = n or n*nb
          minwrk = std::max( minwrk,   n + lwork_sbdsqr        );  // bdsqr  = 4*n
          // if (m+n)*nb (slow) > n*n + n*nb (fast),
          // it's possible that fast subpath requires less workspace
          minwrk = std::min( minwrk, maxwrk );                     // minwrk = 3*n + (m+n)*nb
          //                                              lapack minwrk = max(3*n + m, 5*n)
          }
        else if (want_uo && want_vas)
          {
          // Path 3 (M >> N, JOBU='O', JOBVT='S' or 'A')
          // Path 3-fast
          wrkbl  = std::max( wrkbl,   n + lwork_sgeqrf_mn      );  // geqrf  = n or n*nb
          wrkbl  = std::max( wrkbl,   n + lwork_sorgqr_mn      );  // orgqr  = n or n*nb
          wrkbl  = std::max( wrkbl, 3*n + lwork_sgebrd_nn      );  // gebrd  = n or 2*n*nb
          wrkbl  = std::max( wrkbl, 3*n + lwork_sorgbr_q_nn    );  // orgbr  = n or n*nb
          wrkbl  = std::max( wrkbl, 3*n + lwork_sorgbr_p_nn    );  // orgbr  = n or n*nb
          wrkbl  = std::max( wrkbl,   n + lwork_sbdsqr         );  // bdsqr  = 4*n
          // todo: is n*gemm_nb enough? LAPACK has m*n
          maxwrk = n*n + std::max( wrkbl, n + n*gemm_nb  );        // maxwrk = n*n + 3*n + 2*n*nb
          //                                              lapack maxwrk = n*n + max(3*n + 2*n*nb, n + m*n)
          //                                              lapack minwrk = n*n + 5*n

          // Path 3-slow
          minwrk = std::max( minwrk,   n + lwork_sgeqrf_mn     );  // geqrf  = n or n*nb
          minwrk = std::max( minwrk,   n + lwork_sorgqr_mn     );  // orgqr  = n or n*nb
          minwrk = std::max( minwrk, 3*n + lwork_sgebrd_nn     );  // gebrd  = n or 2*n*nb
          minwrk = std::max( minwrk, 3*n + lwork_sormbr_qrn_mn );  // ormbr  = m or m*nb (note m)
          minwrk = std::max( minwrk, 3*n + lwork_sorgbr_p_nn   );  // orgbr  = n or n*nb
          minwrk = std::max( minwrk,   n + lwork_sbdsqr        );  // bdsqr  = 4*n
          minwrk = std::min( minwrk, maxwrk );                     // minwrk = 3*n + max(m*nb, 2*n*nb)
          //                                              lapack minwrk = max(3*n + m, 5*n)
          }
        else if (want_us && want_vn)
          {
          // Path 4 (M >> N, JOBU='S', JOBVT='N')
          // Path 4-fast
          wrkbl  = std::max( wrkbl,   n + lwork_sgeqrf_mn      );  // geqrf  = n or n*nb
          wrkbl  = std::max( wrkbl,   n + lwork_sorgqr_mn      );  // orgqr  = n or n*nb
          wrkbl  = std::max( wrkbl, 3*n + lwork_sgebrd_nn      );  // gebrd  = n or 2*n*nb
          wrkbl  = std::max( wrkbl, 3*n + lwork_sorgbr_q_nn    );  // orgbr  = n or n*nb
          wrkbl  = std::max( wrkbl,   n + lwork_sbdsqr         );  // bdsqr  = 4*n
          maxwrk = n*n + wrkbl;                               // maxwrk = n*n + 3*n + 2*n*nb
          //                                              lapack minwrk = n*n + 5*n

          // Path 4-slow
          minwrk = std::max( minwrk,   n + lwork_sgeqrf_mn     );  // geqrf  = n or n*nb
          minwrk = std::max( minwrk,   n + lwork_sorgqr_mn     );  // orgqr  = n or n*nb
          minwrk = std::max( minwrk, 3*n + lwork_sgebrd_nn     );  // gebrd  = n or 2*n*nb
          minwrk = std::max( minwrk, 3*n + lwork_sormbr_qrn_mn );  // ormbr  = m or m*nb (note m)
          minwrk = std::max( minwrk,   n + lwork_sbdsqr        );  // bdsqr  = 4*n
          minwrk = std::min( minwrk, maxwrk );                     // minwrk = 3*n + max(m*nb, 2*n*nb)
          //                                              lapack minwrk = max(3*n + m, 5*n)
          }
        else if (want_us && want_vo)
          {
          // Path 5 (M >> N, JOBU='S', JOBVT='O')
          // Path 5-fast
          wrkbl  = std::max( wrkbl,   n + lwork_sgeqrf_mn      );  // geqrf  = n or n*nb
          wrkbl  = std::max( wrkbl,   n + lwork_sorgqr_mn      );  // orgqr  = n or n*nb
          wrkbl  = std::max( wrkbl, 3*n + lwork_sgebrd_nn      );  // gebrd  = n or 2*n*nb
          wrkbl  = std::max( wrkbl, 3*n + lwork_sorgbr_q_nn    );  // orgbr  = n or n*nb
          wrkbl  = std::max( wrkbl, 3*n + lwork_sorgbr_p_nn    );  // orgbr  = n or n*nb
          wrkbl  = std::max( wrkbl,   n + lwork_sbdsqr         );  // bdsqr  = 4*n
          maxwrk = 2*n*n + wrkbl;                             // maxwrk = 2*n*n + 3*n + 2*n*nb
          //                                              lapack minwrk = 2*n*n + 5*n

          // Path 5-slow
          minwrk = std::max( minwrk,   n + lwork_sgeqrf_mn     );  // geqrf  = n or n*nb
          minwrk = std::max( minwrk,   n + lwork_sorgqr_mn     );  // orgqr  = n or n*nb
          minwrk = std::max( minwrk, 3*n + lwork_sgebrd_nn     );  // gebrd  = n or 2*n*nb
          minwrk = std::max( minwrk, 3*n + lwork_sormbr_qrn_mn );  // ormbr  = m or m*nb (note m)
          minwrk = std::max( minwrk, 3*n + lwork_sorgbr_p_nn   );  // orgbr  = n or n*nb
          minwrk = std::max( minwrk,   n + lwork_sbdsqr        );  // bdsqr  = 4*n
          minwrk = std::min( minwrk, maxwrk );                     // minwrk = 3*n + max(m*nb, 2*n*nb)
          //                                              lapack minwrk = max(3*n + m, 5*n)
          }
        else if (want_us && want_vas)
          {
          // Path 6 (M >> N, JOBU='S', JOBVT='S' or 'A')
          // Path 6-fast
          wrkbl  = std::max( wrkbl,   n + lwork_sgeqrf_mn      );  // geqrf  = n or n*nb
          wrkbl  = std::max( wrkbl,   n + lwork_sorgqr_mn      );  // orgqr  = n or n*nb
          wrkbl  = std::max( wrkbl, 3*n + lwork_sgebrd_nn      );  // gebrd  = n or 2*n*nb
          wrkbl  = std::max( wrkbl, 3*n + lwork_sorgbr_q_nn    );  // orgbr  = n or n*nb
          wrkbl  = std::max( wrkbl, 3*n + lwork_sorgbr_p_nn    );  // orgbr  = n or n*nb
          wrkbl  = std::max( wrkbl,   n + lwork_sbdsqr         );  // bdsqr  = 4*n
          maxwrk = n*n + wrkbl;                               // maxwrk = n*n + 3*n + 2*n*nb
          //                                              lapack minwrk = n*n + 5*n

          // Path 6-slow
          minwrk = std::max( minwrk,   n + lwork_sgeqrf_mn     );  // geqrf  = n or n*nb
          minwrk = std::max( minwrk,   n + lwork_sorgqr_mn     );  // orgqr  = n or n*nb
          minwrk = std::max( minwrk, 3*n + lwork_sgebrd_nn     );  // gebrd  = n or 2*n*nb
          minwrk = std::max( minwrk, 3*n + lwork_sormbr_qrn_mn );  // ormbr  = m or m*nb (note m)
          minwrk = std::max( minwrk, 3*n + lwork_sorgbr_p_nn   );  // orgbr  = n or n*nb
          minwrk = std::max( minwrk,   n + lwork_sbdsqr        );  // bdsqr  = 4*n
          minwrk = std::min( minwrk, maxwrk );                     // minwrk = 3*n + max(m*nb, 2*n*nb)
          //                                              lapack minwrk = max(3*n + m, 5*n)
          }
        else if (want_ua && want_vn)
          {
          // Path 7 (M >> N, JOBU='A', JOBVT='N')
          // Path 7-fast
          wrkbl  = std::max( wrkbl,   n + lwork_sgeqrf_mn      );  // geqrf  = n or n*nb
          wrkbl  = std::max( wrkbl,   n + lwork_sorgqr_mm      );  // orgqr  = m or m*nb (note m)
          wrkbl  = std::max( wrkbl, 3*n + lwork_sgebrd_nn      );  // gebrd  = n or 2*n*nb
          wrkbl  = std::max( wrkbl, 3*n + lwork_sorgbr_q_nn    );  // orgbr  = n or n*nb
          wrkbl  = std::max( wrkbl,   n + lwork_sbdsqr         );  // bdsqr  = 4*n
          maxwrk = n*n + wrkbl;                               // maxwrk = n*n + max(n + m*nb, 3*n + 2*n*nb)
          //                                              lapack minwrk = n*n + max(m + n, 5*n)

          // Path 7-slow
          minwrk = std::max( minwrk,   n + lwork_sgeqrf_mn     );  // geqrf  = n or n*nb
          minwrk = std::max( minwrk,   n + lwork_sorgqr_mm     );  // orgqr  = m or m*nb (note m)
          minwrk = std::max( minwrk, 3*n + lwork_sgebrd_nn     );  // gebrd  = n or 2*n*nb
          minwrk = std::max( minwrk, 3*n + lwork_sormbr_qrn_mn );  // ormbr  = m or m*nb (note m)
          minwrk = std::max( minwrk,   n + lwork_sbdsqr        );  // bdsqr  = 4*n
          minwrk = std::min( minwrk, maxwrk );                     // minwrk = 3*n + max(m*nb, 2*n*nb)
          //                                              lapack minwrk = max(3*n + m, 5*n)
          }
        else if (want_ua && want_vo)
          {
          // Path 8 (M >> N, JOBU='A', JOBVT='O')
          // Path 8-fast
          wrkbl  = std::max( wrkbl,   n + lwork_sgeqrf_mn      );  // geqrf  = n or n*nb
          wrkbl  = std::max( wrkbl,   n + lwork_sorgqr_mm      );  // orgqr  = m or m*nb (note m)
          wrkbl  = std::max( wrkbl, 3*n + lwork_sgebrd_nn      );  // gebrd  = n or 2*n*nb
          wrkbl  = std::max( wrkbl, 3*n + lwork_sorgbr_q_nn    );  // orgbr  = n or n*nb
          wrkbl  = std::max( wrkbl, 3*n + lwork_sorgbr_p_nn    );  // orgbr  = n or n*nb
          wrkbl  = std::max( wrkbl,   n + lwork_sbdsqr         );  // bdsqr  = 4*n
          maxwrk = 2*n*n + wrkbl;                             // maxwrk = 2*n*n + max(n + m*nb, 3*n + 2*n*nb)
          //                                              lapack minwrk = 2*n*n + max(m + n, 5*n)

          // Path 8-slow
          minwrk = std::max( minwrk,   n + lwork_sgeqrf_mn     );  // geqrf  = n or n*nb
          minwrk = std::max( minwrk,   n + lwork_sorgqr_mm     );  // orgqr  = m or m*nb (note m)
          minwrk = std::max( minwrk, 3*n + lwork_sgebrd_nn     );  // gebrd  = n or 2*n*nb
          minwrk = std::max( minwrk, 3*n + lwork_sormbr_qrn_mn );  // ormbr  = m or m*nb (note m)
          minwrk = std::max( minwrk, 3*n + lwork_sorgbr_p_nn   );  // orgbr  = n or n*nb
          minwrk = std::max( minwrk,   n + lwork_sbdsqr        );  // bdsqr  = 4*n
          minwrk = std::min( minwrk, maxwrk );                     // minwrk = 3*n + max(m*nb, 2*n*nb)
          //                                              lapack minwrk = max(3*n + m, 5*n)
          }
        else if (want_ua && want_vas)
          {
          // Path 9 (M >> N, JOBU='A', JOBVT='S' or 'A')
          // Path 9-fast
          wrkbl  = std::max( wrkbl,   n + lwork_sgeqrf_mn      );  // geqrf  = n or n*nb
          wrkbl  = std::max( wrkbl,   n + lwork_sorgqr_mm      );  // orgqr  = m or m*nb (note m)
          wrkbl  = std::max( wrkbl, 3*n + lwork_sgebrd_nn      );  // gebrd  = n or 2*n*nb
          wrkbl  = std::max( wrkbl, 3*n + lwork_sorgbr_q_nn    );  // orgbr  = n or n*nb
          wrkbl  = std::max( wrkbl, 3*n + lwork_sorgbr_p_nn    );  // orgbr  = n or n*nb
          wrkbl  = std::max( wrkbl,   n + lwork_sbdsqr         );  // bdsqr  = 4*n
          maxwrk = n*n + wrkbl;                               // maxwrk = n*n + max(n + m*nb, 3*n + 2*n*nb)
          //                                              lapack minwrk = n*n + max(m + n, 5*n)

          // Path 9-slow
          minwrk = std::max( minwrk,   n + lwork_sgeqrf_mn     );  // geqrf  = n or n*nb
          minwrk = std::max( minwrk,   n + lwork_sorgqr_mm     );  // orgqr  = m or m*nb (note m)
          minwrk = std::max( minwrk, 3*n + lwork_sgebrd_nn     );  // gebrd  = n or 2*n*nb
          minwrk = std::max( minwrk, 3*n + lwork_sormbr_qrn_mn );  // ormbr  = m or m*nb (note m)
          minwrk = std::max( minwrk, 3*n + lwork_sorgbr_p_nn   );  // orgbr  = n or n*nb
          minwrk = std::max( minwrk,   n + lwork_sbdsqr        );  // bdsqr  = 4*n
          minwrk = std::min( minwrk, maxwrk );                     // minwrk = 3*n + max(m*nb, 2*n*nb)
          //                                              lapack minwrk = max(3*n + m, 5*n)
          }
        }
      else
        {
        // Path 10 (M >= N, but not much larger)
        wrkbl  = std::max( wrkbl,     3*n + lwork_sgebrd_mn      );  // gebrd  = m or (m+n)*nb (note m)
        if (want_us || want_uo)
          {
          wrkbl  = std::max( wrkbl, 3*n + lwork_sorgbr_q_mn    );  // orgbr  = n or n*nb
          }
        if (want_ua)
          {
          wrkbl  = std::max( wrkbl, 3*n + lwork_sorgbr_q_mm    );  // orgbr  = m or m*nb (note m)
          }
        if (want_vas || want_vo)
          {
          wrkbl  = std::max( wrkbl, 3*n + lwork_sorgbr_p_nn    );  // orgbr  = n or n*nb
          }
        wrkbl  = std::max( wrkbl,       n + lwork_sbdsqr         );  // bdsqr  = 4*n
        maxwrk = wrkbl;                                         // maxwrk = 3*n + (m+n)*nb
        minwrk = maxwrk;                                        // minwrk = 3*n + (m+n)*nb
        //                                                  lapack minwrk = max(3*n + m, 5*n)
        }
      }
    else if (minmn > 0)
      {
      // m < n
      // Compute space needed for DBDSQR
      // needs n for [e] and 4*n for [work]
      magma_int_t lwork_sbdsqr = 4*m;

      // Compute space preferred for each routine
      // For MAGMA, these are all required
      magma_sgebrd(      m,  n, unused,  m, unused, unused, unused, unused, dummy,  ineg_one, &ierr );
      magma_int_t lwork_sgebrd_mn = magma_int_t( dummy[0] );

      magma_sgebrd(      m,  m, unused,  m, unused, unused, unused, unused, dummy,  ineg_one, &ierr );
      magma_int_t lwork_sgebrd_mm = magma_int_t( dummy[0] );

      magma_sgelqf(      m,  n, unused,  m, unused, dummy,  ineg_one, &ierr );
      magma_int_t lwork_sgelqf_mn = magma_int_t( dummy[0] );

      magma_sorgbr( MagmaP,   m,  m,  m, unused,  m, unused, dummy,  ineg_one, &ierr );
      magma_int_t lwork_sorgbr_p_mm = magma_int_t( dummy[0] );

      magma_sorgbr( MagmaP,   m,  n,  m, unused,  m, unused, dummy,  ineg_one, &ierr );
      magma_int_t lwork_sorgbr_p_mn = magma_int_t( dummy[0] );

      magma_sorgbr( MagmaP,   n,  n,  m, unused,  n, unused, dummy,  ineg_one, &ierr );
      magma_int_t lwork_sorgbr_p_nn = magma_int_t( dummy[0] );

      magma_sorgbr( MagmaQ,   m,  m,  m, unused,  m, unused, dummy,  ineg_one, &ierr );
      magma_int_t lwork_sorgbr_q_mm = magma_int_t( dummy[0] );

      magma_sorglq(      m,  n,  m, unused,  m, unused, dummy,  ineg_one, &ierr );
      magma_int_t lwork_sorglq_mn = magma_int_t( dummy[0] );

      magma_sorglq(      n,  n,  m, unused,  n, unused, dummy,  ineg_one, &ierr );
      magma_int_t lwork_sorglq_nn = magma_int_t( dummy[0] );

      // missing from LAPACK, since it occurs only in slow paths
      magma_sormbr( MagmaP, MagmaLeft, MagmaTrans,  m,  n,  m, unused,  m, unused, unused,  m, dummy,  ineg_one, &ierr );
      magma_int_t lwork_sormbr_plt_mn = magma_int_t( dummy[0] );

      if (n >= mnthr)
        {
        if (want_vn)
          {
          // Path 1t (N >> M, JOBVT='N')
          wrkbl  = std::max( wrkbl,      m + lwork_sgelqf_mn   );  // gelqf  = m or m*nb
          wrkbl  = std::max( wrkbl,    3*m + lwork_sgebrd_mm   );  // gebrd  = m or 2*m*nb
          if (want_uo || want_uas)
            {
            wrkbl = std::max( wrkbl, 3*m + lwork_sorgbr_q_mm );  // orgbr  = m or m*nb
            }
          wrkbl  = std::max( wrkbl,      m + lwork_sbdsqr      );  // bdsqr  = 4*m
          maxwrk = wrkbl;                                     // maxwrk = 3*m + 2*m*nb
          minwrk = maxwrk;                                    // minwrk = 3*m + 2*m*nb
          //                                              lapack minwrk = 5*m
          }
        else if (want_vo && want_un)
          {
          // Path 2t (N >> M, JOBU='N', JOBVT='O')
          // Path 2t-fast
          wrkbl  = std::max( wrkbl,   m + lwork_sgelqf_mn      );  // gelqf  = m or m*nb
          wrkbl  = std::max( wrkbl,   m + lwork_sorglq_mn      );  // orglq  = m or m*nb
          wrkbl  = std::max( wrkbl, 3*m + lwork_sgebrd_mm      );  // gebrd  = m or 2*m*nb
          wrkbl  = std::max( wrkbl, 3*m + lwork_sorgbr_p_mm    );  // orgbr  = m or m*nb
          wrkbl  = std::max( wrkbl,   m + lwork_sbdsqr         );  // bdsqr  = 4*m
          // todo: is m*gemm_nb enough? LAPACK has m*n
          maxwrk = m*m + std::max( wrkbl, m + m*gemm_nb        );  // maxwrk = m*m + 3*m + 2*m*nb
          //                                              lapack maxwrk = m*m + max(3*m + 2*m*nb, m + m*n)
          //                                              lapack minwrk = m*m + 5*m

          // Path 2t-slow
          minwrk = std::max( minwrk, 3*m + lwork_sgebrd_mn     );  // gebrd  = n or (m+n)*nb (note n)
          minwrk = std::max( minwrk, 3*m + lwork_sorgbr_p_mn   );  // orgbr  = m or m*nb
          minwrk = std::max( minwrk,   m + lwork_sbdsqr        );  // bdsqr  = 4*m
          minwrk = std::min( minwrk, maxwrk );                     // minwrk = 3*m + (m+n)*nb
          //                                              lapack minwrk = max(3*m + n, 5*m)
          }
        else if (want_vo && want_uas)
          {
          // Path 3t (N >> M, JOBU='S' or 'A', JOBVT='O')
          // Path 3t-fast
          wrkbl  = std::max( wrkbl,   m + lwork_sgelqf_mn      );  // gelqf  = m or m*nb
          wrkbl  = std::max( wrkbl,   m + lwork_sorglq_mn      );  // orglq  = m or m*nb
          wrkbl  = std::max( wrkbl, 3*m + lwork_sgebrd_mm      );  // gebrd  = m or 2*m*nb
          wrkbl  = std::max( wrkbl, 3*m + lwork_sorgbr_p_mm    );  // orgbr  = m or m*nb
          wrkbl  = std::max( wrkbl, 3*m + lwork_sorgbr_q_mm    );  // orgbr  = m or m*nb
          wrkbl  = std::max( wrkbl,   m + lwork_sbdsqr         );  // bdsqr  = 4*m
          // todo: is m*gemm_nb enough? LAPACK has m*n
          maxwrk = m*m + std::max( wrkbl, m + m*gemm_nb        );  // maxwrk = m*m + 3*m + 2*m*nb
          //                                              lapack maxwrk = m*m + max(3*m + 2*m*nb, m + m*n)
          //                                              lapack minwrk = m*m + 5*m

          // Path 3t-slow
          minwrk = std::max( minwrk,   m + lwork_sgelqf_mn     );  // gelqf  = m or m*nb
          minwrk = std::max( minwrk,   m + lwork_sorglq_mn     );  // orglq  = m or m*nb
          minwrk = std::max( minwrk, 3*m + lwork_sgebrd_mm     );  // gebrd  = m or 2*m*nb
          minwrk = std::max( minwrk, 3*m + lwork_sormbr_plt_mn );  // ormbr  = n or n*nb (note n)
          minwrk = std::max( minwrk, 3*m + lwork_sorgbr_q_mm   );  // orgbr  = m or m*nb
          minwrk = std::max( minwrk,   m + lwork_sbdsqr        );  // bdsqr  = 4*m
          minwrk = std::min( minwrk, maxwrk );                     // minwrk = 3*m + max(2*m*nb, n*nb)
          //                                              lapack minwrk = max(3*m + n, 5*m)
          }
        else if (want_vs && want_un)
          {
          // Path 4t (N >> M, JOBU='N', JOBVT='S')
          // Path 4t-fast
          wrkbl  = std::max( wrkbl,   m + lwork_sgelqf_mn      );  // gelqf  = m or m*nb
          wrkbl  = std::max( wrkbl,   m + lwork_sorglq_mn      );  // orglq  = m or m*nb
          wrkbl  = std::max( wrkbl, 3*m + lwork_sgebrd_mm      );  // gebrd  = m or 2*m*nb
          wrkbl  = std::max( wrkbl, 3*m + lwork_sorgbr_p_mm    );  // orgbr  = m or m*nb
          wrkbl  = std::max( wrkbl,   m + lwork_sbdsqr         );  // bdsqr  = 4*m
          maxwrk = m*m + wrkbl;                               // maxwrk = m*m + 3*m + 2*m*nb
          //                                              lapack minwrk = m*m + 5*m

          // Path 4t-slow
          minwrk = std::max( minwrk,   m + lwork_sgelqf_mn     );  // gelqf  = m or m*nb
          minwrk = std::max( minwrk,   m + lwork_sorglq_mn     );  // orglq  = m or m*nb
          minwrk = std::max( minwrk, 3*m + lwork_sgebrd_mm     );  // gebrd  = m or 2*m*nb
          minwrk = std::max( minwrk, 3*m + lwork_sormbr_plt_mn );  // ormbr  = n or n*nb (note n)
          minwrk = std::max( minwrk,   m + lwork_sbdsqr        );  // bdsqr  = 4*m
          minwrk = std::min( minwrk, maxwrk );                     // minwrk = 3*m + max(2*m*nb, n*nb)
          //                                              lapack minwrk = max(3*m + n, 5*m)
          }
        else if (want_vs && want_uo)
          {
          // Path 5t (N >> M, JOBU='O', JOBVT='S')
          // Path 5t-fast
          wrkbl  = std::max( wrkbl,   m + lwork_sgelqf_mn      );  // gelqf  = m or m*nb
          wrkbl  = std::max( wrkbl,   m + lwork_sorglq_mn      );  // orglq  = m or m*nb
          wrkbl  = std::max( wrkbl, 3*m + lwork_sgebrd_mm      );  // gebrd  = m or 2*m*nb
          wrkbl  = std::max( wrkbl, 3*m + lwork_sorgbr_p_mm    );  // orgbr  = m or m*nb
          wrkbl  = std::max( wrkbl, 3*m + lwork_sorgbr_q_mm    );  // orgbr  = m or m*nb
          wrkbl  = std::max( wrkbl,   m + lwork_sbdsqr         );  // bdsqr  = 4*m
          maxwrk = 2*m*m + wrkbl;                             // maxwrk = 2*m*m + 3*m + 2*m*nb
          //                                              lapack minwrk = 2*m*m + 5*m

          // Path 5t-slow
          minwrk = std::max( minwrk,   m + lwork_sgelqf_mn     );  // gelqf  = m or m*nb
          minwrk = std::max( minwrk,   m + lwork_sorglq_mn     );  // orglq  = m or m*nb
          minwrk = std::max( minwrk, 3*m + lwork_sgebrd_mm     );  // gebrd  = m or 2*m*nb
          minwrk = std::max( minwrk, 3*m + lwork_sormbr_plt_mn );  // ormbr  = n or n*nb (note n)
          minwrk = std::max( minwrk, 3*m + lwork_sorgbr_q_mm   );  // orgbr  = m or m*nb
          minwrk = std::max( minwrk,   m + lwork_sbdsqr        );  // bdsqr  = 4*m
          minwrk = std::min( minwrk, maxwrk );                     // minwrk = 3*m + max(2*m*nb, n*nb)
          //                                              lapack minwrk = max(3*m + n, 5*m)
          }
        else if (want_vs && want_uas)
          {
          // Path 6t (N >> M, JOBU='S' or 'A', JOBVT='S')
          // Path 6t-fast
          wrkbl  = std::max( wrkbl,   m + lwork_sgelqf_mn      );  // gelqf  = m or m*nb
          wrkbl  = std::max( wrkbl,   m + lwork_sorglq_mn      );  // orglq  = m or m*nb
          wrkbl  = std::max( wrkbl, 3*m + lwork_sgebrd_mm      );  // gebrd  = m or 2*m*nb
          wrkbl  = std::max( wrkbl, 3*m + lwork_sorgbr_p_mm    );  // orgbr  = m or m*nb
          wrkbl  = std::max( wrkbl, 3*m + lwork_sorgbr_q_mm    );  // orgbr  = m or m*nb
          wrkbl  = std::max( wrkbl,   m + lwork_sbdsqr         );  // bdsqr  = 4*m
          maxwrk = m*m + wrkbl;                               // maxwrk = m*m + 3*m + 2*m*nb
          //                                              lapack minwrk = m*m + 5*m

          // Path 6t-slow
          minwrk = std::max( minwrk,   m + lwork_sgelqf_mn     );  // gelqf  = m or m*nb
          minwrk = std::max( minwrk,   m + lwork_sorglq_mn     );  // orglq  = m or m*nb
          minwrk = std::max( minwrk, 3*m + lwork_sgebrd_mm     );  // gebrd  = m or 2*m*nb
          minwrk = std::max( minwrk, 3*m + lwork_sormbr_plt_mn );  // ormbr  = n or n*nb (note n)
          minwrk = std::max( minwrk, 3*m + lwork_sorgbr_q_mm   );  // orgbr  = m or m*nb
          minwrk = std::max( minwrk,   m + lwork_sbdsqr        );  // bdsqr  = 4*m
          minwrk = std::min( minwrk, maxwrk );                     // minwrk = 3*m + max(2*m*nb, n*nb)
          //                                              lapack minwrk = max(3*m + n, 5*m)
          }
        else if (want_va && want_un)
          {
          // Path 7t (N >> M, JOBU='N', JOBVT='A')
          // Path 7t-fast
          wrkbl  = std::max( wrkbl,   m + lwork_sgelqf_mn      );  // gelqf  = m or m*nb
          wrkbl  = std::max( wrkbl,   m + lwork_sorglq_nn      );  // orglq  = n or n*nb (note n)
          wrkbl  = std::max( wrkbl, 3*m + lwork_sgebrd_mm      );  // gebrd  = m or 2*m*nb
          wrkbl  = std::max( wrkbl, 3*m + lwork_sorgbr_p_mm    );  // orgbr  = m or m*nb
          wrkbl  = std::max( wrkbl,   m + lwork_sbdsqr         );  // bdsqr  = 4*m
          maxwrk = m*m + wrkbl;                               // maxwrk = m*m + max(m + n*nb, 3*m + 2*m*nb)
          //                                              lapack minwrk = m*m + max(m + n, 5*m)

          // Path 7t-slow
          minwrk = std::max( minwrk,   m + lwork_sgelqf_mn     );  // gelqf  = m or m*nb
          minwrk = std::max( minwrk,   m + lwork_sorglq_nn     );  // orglq  = n or n*nb (note n)
          minwrk = std::max( minwrk, 3*m + lwork_sgebrd_mm     );  // gebrd  = m or 2*m*nb
          minwrk = std::max( minwrk, 3*m + lwork_sormbr_plt_mn );  // ormbr  = n or n*nb (note n)
          minwrk = std::max( minwrk,   m + lwork_sbdsqr        );  // bdsqr  = 4*m
          minwrk = std::min( minwrk, maxwrk );                     // minwrk = 3*m + max(2*m*nb, n*nb)
          //                                              lapack minwrk = max(3*m + n, 5*m)
          }
        else if (want_va && want_uo)
          {
          // Path 8t (N >> M, JOBU='O', JOBVT='A')
          // Path 8t-fast
          wrkbl  = std::max( wrkbl,   m + lwork_sgelqf_mn      );  // gelqf  = m or m*nb
          wrkbl  = std::max( wrkbl,   m + lwork_sorglq_nn      );  // orglq  = n or n*nb (note n)
          wrkbl  = std::max( wrkbl, 3*m + lwork_sgebrd_mm      );  // gebrd  = m or 2*m*nb
          wrkbl  = std::max( wrkbl, 3*m + lwork_sorgbr_p_mm    );  // orgbr  = m or m*nb
          wrkbl  = std::max( wrkbl, 3*m + lwork_sorgbr_q_mm    );  // orgbr  = m or m*nb
          wrkbl  = std::max( wrkbl,   m + lwork_sbdsqr         );  // bdsqr  = 4*m
          maxwrk = 2*m*m + wrkbl;                             // maxwrk = 2*m*m + max(m + n*nb, 3*m + 2*m*nb)
          //                                              lapack minwrk = 2*m*m + max(m + n, 5*m)

          // Path 8t-slow
          minwrk = std::max( minwrk,   m + lwork_sgelqf_mn     );  // gelqf  = m or m*nb
          minwrk = std::max( minwrk,   m + lwork_sorglq_nn     );  // orglq  = n or n*nb (note n)
          minwrk = std::max( minwrk, 3*m + lwork_sgebrd_mm     );  // gebrd  = m or 2*m*nb
          minwrk = std::max( minwrk, 3*m + lwork_sormbr_plt_mn );  // ormbr  = n or n*nb (note n)
          minwrk = std::max( minwrk, 3*m + lwork_sorgbr_q_mm   );  // orgbr  = m or m*nb
          minwrk = std::max( minwrk,   m + lwork_sbdsqr        );  // bdsqr  = 4*m
          minwrk = std::min( minwrk, maxwrk );                     // minwrk = 3*m + max(2*m*nb, n*nb)
          //                                              lapack minwrk = max(3*m + n, 5*m)
          }
        else if (want_va && want_uas)
          {
          // Path 9t (N >> M, JOBU='S' or 'A', JOBVT='A')
          // Path 9t-fast
          wrkbl  = std::max( wrkbl,   m + lwork_sgelqf_mn      );  // gelqf  = m or m*nb
          wrkbl  = std::max( wrkbl,   m + lwork_sorglq_nn      );  // orglq  = n or n*nb (note n)
          wrkbl  = std::max( wrkbl, 3*m + lwork_sgebrd_mm      );  // gebrd  = m or 2*m*nb
          wrkbl  = std::max( wrkbl, 3*m + lwork_sorgbr_p_mm    );  // orgbr  = m or m*nb
          wrkbl  = std::max( wrkbl, 3*m + lwork_sorgbr_q_mm    );  // orgbr  = m or m*nb
          wrkbl  = std::max( wrkbl,   m + lwork_sbdsqr         );  // bdsqr  = 4*m
          maxwrk = m*m + wrkbl;                               // maxwrk = m*m + max(m + n*nb, 3*m + 2*m*nb)
          //                                              lapack minwrk = m*m + max(m + n, 5*m)

          // Path 9t-slow
          minwrk = std::max( minwrk,   m + lwork_sgelqf_mn     );  // gelqf  = m or m*nb
          minwrk = std::max( minwrk,   m + lwork_sorglq_nn     );  // orglq  = n or n*nb (note n)
          minwrk = std::max( minwrk, 3*m + lwork_sgebrd_mm     );  // gebrd  = m or 2*m*nb
          minwrk = std::max( minwrk, 3*m + lwork_sormbr_plt_mn );  // ormbr  = n or n*nb (note n)
          minwrk = std::max( minwrk, 3*m + lwork_sorgbr_q_mm   );  // orgbr  = m or m*nb
          minwrk = std::max( minwrk,   m + lwork_sbdsqr        );  // bdsqr  = 4*m
          minwrk = std::min( minwrk, maxwrk );                     // minwrk = 3*m + max(2*m*nb, n*nb)
          //                                              lapack minwrk = max(3*m + n, 5*m)
          }
        }
      else
        {
        // Path 10t (N > M, but not much larger)
        wrkbl  = std::max( wrkbl,     3*m + lwork_sgebrd_mn      );  // gebrd  = n or (m+n)*nb (note n)
        if (want_vs || want_vo)
          {
          wrkbl  = std::max( wrkbl, 3*m + lwork_sorgbr_p_mn    );  // orgbr  = m or m*nb
          }
        if (want_va)
          {
          wrkbl  = std::max( wrkbl, 3*m + lwork_sorgbr_p_nn    );  // orgbr  = n or n*nb (note n)
          }
        if (want_uas || want_uo)
          {
          wrkbl  = std::max( wrkbl, 3*m + lwork_sorgbr_q_mm    );  // orgbr  = m or m*nb
          }
        wrkbl  = std::max( wrkbl, m + lwork_sbdsqr               );  // bdsqr  = 4*m
        maxwrk = wrkbl;                                         // maxwrk = 3*m + (m+n)*nb
        minwrk = maxwrk;                                        // minwrk = 3*m + (m+n)*nb
        //                                                  lapack minwrk = max(3*m + n, 5*m)
        }
      }
      assert( minwrk <= maxwrk );

      work[1] = magma_smake_lwork( maxwrk );

      if (lwork < minwrk && ! lquery)
        {
        *info = -13;
        }
      }

    if (*info != 0)
      {
      //magma_xerbla( __func__, -(*info) );
      return *info;
      }
    else if (lquery)
      {
      return *info;
      }

    // Quick return if possible
    if (m == 0 || n == 0)
      {
      return *info;
      }

    // Get machine constants
    eps = lapack::lamch<float>('P');
    smlnum = std::sqrt(lapack::lamch<float>('S')) / eps;
    bignum = 1. / smlnum;

    // Scale A if max element outside range [SMLNUM,BIGNUM]
    anrm = lapack::lange('M', m, n, A, lda, dummy);
    iscl = 0;
    if (anrm > 0. && anrm < smlnum)
      {
      iscl = 1;
      lapack::lascl('G', izero, izero, anrm, smlnum, m, n, A, lda, &ierr);
      }
    else if (anrm > bignum)
      {
      iscl = 1;
      lapack::lascl('G', izero, izero, anrm, bignum, m, n, A, lda, &ierr);
      }

    m_1 = m - 1;
    n_1 = n - 1;

    if (m >= n)
      {
      // A has at least as many rows as columns.
      // If A has sufficiently more rows than columns, first reduce using
      // the QR decomposition (if sufficient workspace available)
      if (m >= mnthr)
        {
        if (want_un)
          {
          // Path 1 (M >> N, JOBU='N')
          // No left singular vectors to be computed
          itau  = 1;
          iwork = itau + n;

          // Compute A=Q*R
          // Workspace: need   N [tau] + N    [geqrf work]
          // Workspace: prefer N [tau] + N*NB [geqrf work]
          lwork2 = lwork - iwork + 1;
          magma_sgeqrf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );

          // Zero out below R
          lapack::laset('L', n_1, n_1, c_zero, c_zero, &A[1], lda);
          ie    = 1;
          itauq = ie    + n;
          itaup = itauq + n;
          iwork = itaup + n;

          // Bidiagonalize R in A
          // Workspace: need   3*N [e, tauq, taup] + N      [gebrd work]
          // Workspace: prefer 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
          lwork2 = lwork - iwork + 1;
          magma_sgebrd(      n,  n, A,  lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );

          ncvt = 0;
          if (want_vo || want_vas)
            {
            // If right singular vectors desired, generate P**T.
            // Workspace: need     3*N [e, tauq, taup] + N    [orgbr work]
            // Workspace: prefer   3*N [e, tauq, taup] + N*NB [orgbr work]
            lwork2 = lwork - iwork + 1;
            magma_sorgbr( MagmaP,   n,  n,  n, A,  lda, &work[itaup], &work[iwork],  lwork2, &ierr );
            ncvt = n;
            }
          iwork = ie + n;

          // Perform bidiagonal QR iteration, computing right
          // singular vectors of A in A if desired
          // Workspace: need     N [e] + 4*N [bdsqr work]
          lapack::bdsqr('U', n, ncvt, izero, izero, s, &work[ie], A, lda, dummy, ione, dummy, ione, &work[iwork], info);

          // If right singular vectors desired in VT, copy them there
          if (want_vas)
            {
            lapack::lacpy('F', n, n, A, lda, VT, ldvt);
            }
          }
        else if (want_uo && want_vn)
          {
          // Path 2 (M >> N, JOBU='O', JOBVT='N')
          // N left singular vectors to be overwritten on A and
          // no right singular vectors to be computed
          if (lwork >= n*n + wrkbl)
            {
            // Path 2-fast: Sufficient workspace for a fast algorithm
            if (lwork >= std::max(wrkbl, lda*n + n) + lda*n)
              {
              // WORK(IU) is LDA by N
              // WORK(IR) is LDA by N
              ldwrku = lda;
              ldwrkr = lda;
              }
            else if (lwork >= std::max(wrkbl, lda*n + n) + n*n)
              {
              // WORK(IU) is LDA by N
              // WORK(IR) is N by N
              ldwrku = lda;
              ldwrkr = n;
              }
            else
              {
              // WORK(IU) is LDWRKU by N
              // WORK(IR) is N by N
              ldwrku = (lwork - n*n - n) / n;
              ldwrkr = n;
              }
            ir    = 1;
            itau  = ir + ldwrkr*n;
            iwork = itau + n;

            // Compute A=Q*R
            // Workspace: need   N*N [R] + N [tau] + N    [geqrf work]
            // Workspace: prefer N*N [R] + N [tau] + N*NB [geqrf work]
            lwork2 = lwork - iwork + 1;
            magma_sgeqrf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );

            // Copy R to WORK(IR) and zero out below it
            lapack::lacpy('U', n, n, A, lda, &work[ir], ldwrkr);
            lapack::laset('L', n_1, n_1, c_zero, c_zero, &work[ir+1], ldwrkr);

            // Generate Q in A
            // Workspace: need   N*N [R] + N [tau] + N    [orgqr work]
            // Workspace: prefer N*N [R] + N [tau] + N*NB [orgqr work]
            lwork2 = lwork - iwork + 1;
            magma_sorgqr2(     m,  n,  n, A,  lda, &work[itau], &ierr );

            ie    = itau;
            itauq = ie    + n;
            itaup = itauq + n;
            iwork = itaup + n;

            // Bidiagonalize R in WORK(IR)
            // Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N      [gebrd work]
            // Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
            lwork2 = lwork - iwork + 1;
            magma_sgebrd(      n,  n, &work[ir],  ldwrkr, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );

            // Generate left vectors bidiagonalizing R
            // Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N    [orgbr work]
            // Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + N*NB [orgbr work]
            lwork2 = lwork - iwork + 1;
            magma_sorgbr( MagmaQ,   n,  n,  n, &work[ir],  ldwrkr, &work[itauq], &work[iwork],  lwork2, &ierr );
            iwork = ie + n;

            // Perform bidiagonal QR iteration, computing left
            // singular vectors of R in WORK(IR)
            // Workspace: need   N*N [R] + N [e] + 4*N [bdsqr work]
            lapack::bdsqr('U', n, izero, n, izero, s, &work[ie], dummy, ione, &work[ir], ldwrkr, dummy, ione, &work[iwork], info);
            iu = ie + n;

            // Multiply Q in A by left singular vectors of R in
            // WORK(IR), storing result in WORK(IU) and copying to A
            // Workspace: need   N*N [R] + N [e] + N    [U]
            // Workspace: prefer N*N [R] + N [e] + NB*N [U]
            // Workspace: max    N*N [R] + N [e] + M*N  [U]
            // TODO: [e] not need; can change lwork above
            for (i = 1; i <= m; i += ldwrku)
              {
              ib = std::min( m - i + 1, ldwrku );
              blas::gemm('N', 'N', ib, n, n,
                         c_one,  &A[i-1],   lda,
                                 &work[ir], ldwrkr,
                         c_zero, &work[iu], ldwrku);
              lapack::lacpy('F', ib, n, &work[iu], ldwrku, &A[i-1], lda);
              }
            }
          else
            {
            // Path 2-slow: Insufficient workspace for a fast algorithm
            ie    = 1;
            itauq = ie    + n;
            itaup = itauq + n;
            iwork = itaup + n;

            // Bidiagonalize A
            // Workspace: need   3*N [e, tauq, taup] + M        [gebrd work]
            // Workspace: prefer 3*N [e, tauq, taup] + (M+N)*NB [gebrd work]
            lwork2 = lwork - iwork + 1;
            magma_sgebrd(      m,  n, A,  lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );

            // Generate left vectors bidiagonalizing A
            // Workspace: need   3*N [e, tauq, taup] + N    [orgbr work]
            // Workspace: prefer 3*N [e, tauq, taup] + N*NB [orgbr work]
            magma_sorgbr( MagmaQ,   m,  n,  n, A,  lda, &work[itauq], &work[iwork],  lwork2, &ierr );
            iwork = ie + n;

            // Perform bidiagonal QR iteration, computing left
            // singular vectors of A in A
            // Workspace: need     N [e] + 4*N [bdsqr work]
            lapack::bdsqr('U', n, izero, m, izero, s, &work[ie], dummy, ione, A, lda, dummy, ione, &work[iwork], info);
            }
          }
        else if (want_uo && want_vas)
          {
          // Path 3 (M >> N, JOBU='O', JOBVT='S' or 'A')
          // N left singular vectors to be overwritten on A and
          // N right singular vectors to be computed in VT
          if (lwork >= n*n + wrkbl)
            {
            // Path 3-fast: Sufficient workspace for a fast algorithm
            if (lwork >= std::max(wrkbl, lda*n + n) + lda*n)
              {
              // WORK(IU) is LDA by N
              // WORK(IR) is LDA by N
              ldwrku = lda;
              ldwrkr = lda;
              }
            else if (lwork >= std::max(wrkbl, lda*n + n) + n*n)
              {
              // WORK(IU) is LDA by N
              // WORK(IR) is N by N
              ldwrku = lda;
              ldwrkr = n;
              }
            else
              {
              // WORK(IU) is LDWRKU by N
              // WORK(IR) is N by N
              ldwrku = (lwork - n*n - n) / n;
              ldwrkr = n;
              }
            ir    = 1;
            itau  = ir + ldwrkr * n;
            iwork = itau + n;

            // Compute A=Q*R
            // Workspace: need   N*N [R] + N [tau] + N    [geqrf work]
            // Workspace: prefer N*N [R] + N [tau] + N*NB [geqrf work]
            lwork2 = lwork - iwork + 1;
            magma_sgeqrf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );

            // Copy R to VT, zeroing out below it
            lapack::lacpy('U', n, n, A, lda, VT, ldvt);
            if (n > 1)
              {
              lapack::laset('L', n_1, n_1, c_zero, c_zero, &VT[1], ldvt);
              }

            // Generate Q in A
            // Workspace: need   N*N [R] + N [tau] + N    [orgqr work]
            // Workspace: prefer N*N [R] + N [tau] + N*NB [orgqr work]
            lwork2 = lwork - iwork + 1;
            magma_sorgqr2(     m,  n,  n, A,  lda, &work[itau], &ierr );

            ie    = itau;
            itauq = ie    + n;
            itaup = itauq + n;
            iwork = itaup + n;

            // Bidiagonalize R in VT, copying result to WORK(IR)
            // Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N      [gebrd work]
            // Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
            lwork2 = lwork - iwork + 1;
            magma_sgebrd(      n,  n, VT,  ldvt, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
            lapack::lacpy('L', n, n, VT, ldvt, &work[ir], ldwrkr);

            // Generate left vectors bidiagonalizing R in WORK(IR)
            // Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N    [orgbr work]
            // Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + N*NB [orgbr work]
            lwork2 = lwork - iwork + 1;
            magma_sorgbr( MagmaQ,   n,  n,  n, &work[ir],  ldwrkr, &work[itauq], &work[iwork],  lwork2, &ierr );

            // Generate right vectors bidiagonalizing R in VT
            // Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N    [orgbr work]
            // Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + N*NB [orgbr work]
            lwork2 = lwork - iwork + 1;
            magma_sorgbr( MagmaP,   n,  n,  n, VT,  ldvt, &work[itaup], &work[iwork],  lwork2, &ierr );
            iwork = ie + n;

            // Perform bidiagonal QR iteration, computing left
            // singular vectors of R in WORK(IR) and computing right
            // singular vectors of R in VT
            // Workspace: need   N*N [R] + N [e] + 4*N [bdsqr work]
            lapack::bdsqr('U', n, n, n, izero, s, &work[ie], VT, ldvt, &work[ir], ldwrkr, dummy, ione, &work[iwork], info);
            iu = ie + n;

            // Multiply Q in A by left singular vectors of R in
            // WORK(IR), storing result in WORK(IU) and copying to A
            // Workspace: need   N*N [R] + N [e] + N    [U]
            // Workspace: prefer N*N [R] + N [e] + NB*N [U]
            // Workspace: max    N*N [R] + N [e] + M*N  [U]
            // TODO: [e] not need; can change lwork above
            for (i = 1; i <= m; i += ldwrku)
              {
              ib = std::min( m - i + 1, ldwrku );
              blas::gemm('N', 'N', ib, n, n,
                         c_one,  &A[i-1],   lda,
                                 &work[ir], ldwrkr,
                         c_zero, &work[iu], ldwrku);
              lapack::lacpy('F', ib, n, &work[iu], ldwrku, &A[i-1], lda);
              }
            }
          else
            {
            // Path 3-slow: Insufficient workspace for a fast algorithm
            itau  = 1;
            iwork = itau + n;

            // Compute A=Q*R
            // Workspace: need   N [tau] + N    [geqrf work]
            // Workspace: prefer N [tau] + N*NB [geqrf work]
            lwork2 = lwork - iwork + 1;
            magma_sgeqrf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );

            // Copy R to VT, zeroing out below it
            lapack::lacpy('U', n, n, A, lda, VT, ldvt);
            if (n > 1)
              {
              lapack::laset('L', n_1, n_1, c_zero, c_zero, &VT[1], ldvt);
              }

            // Generate Q in A
            // Workspace: need   N [tau] + N    [orgqr work]
            // Workspace: prefer N [tau] + N*NB [orgqr work]
            lwork2 = lwork - iwork + 1;
            magma_sorgqr2(     m,  n,  n, A,  lda, &work[itau], &ierr );

            ie    = itau;
            itauq = ie    + n;
            itaup = itauq + n;
            iwork = itaup + n;

            // Bidiagonalize R in VT
            // Workspace: need   3*N [e, tauq, taup] + N      [gebrd work]
            // Workspace: prefer 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
            lwork2 = lwork - iwork + 1;
            magma_sgebrd(      n,  n, VT,  ldvt, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );

            // Multiply Q in A by left vectors bidiagonalizing R
            // Workspace: need   3*N [e, tauq, taup] + M    [ormbr work]
            // Workspace: prefer 3*N [e, tauq, taup] + M*NB [ormbr work]
            lwork2 = lwork - iwork + 1;
            magma_sormbr( MagmaQ, MagmaRight, MagmaNoTrans,  m,  n,  n, VT,  ldvt, &work[itauq], A,  lda, &work[iwork],  lwork2, &ierr );

            // Generate right vectors bidiagonalizing R in VT
            // Workspace: need   3*N [e, tauq, taup] + N    [orgbr work]
            // Workspace: prefer 3*N [e, tauq, taup] + N*NB [orgbr work]
            lwork2 = lwork - iwork + 1;
            magma_sorgbr( MagmaP,   n,  n,  n, VT,  ldvt, &work[itaup], &work[iwork],  lwork2, &ierr );
            iwork = ie + n;

            // Perform bidiagonal QR iteration, computing left
            // singular vectors of A in A and computing right
            // singular vectors of A in VT
            // Workspace: need     N [e] + 4*N [bdsqr work]
            lapack::bdsqr('U', n, n, m, izero, s, &work[ie], VT, ldvt, A, lda, dummy, ione, &work[iwork], info);
          }
        }
      else if (want_us && want_vn)
        {
        // Path 4 (M >> N, JOBU='S', JOBVT='N')
        // N left singular vectors to be computed in U and
        // no right singular vectors to be computed
        if (lwork >= n*n + wrkbl)
          {
          // Path 4-fast: Sufficient workspace for a fast algorithm
          if (lwork >= wrkbl + lda*n)
            {
            // WORK(IR) is LDA by N
            ldwrkr = lda;
            }
          else
            {
            // WORK(IR) is N by N
            ldwrkr = n;
            }
          ir    = 1;
          itau  = ir + ldwrkr * n;
          iwork = itau + n;

          // Compute A=Q*R
          // Workspace: need   N*N [R] + N [tau] + N    [geqrf work]
          // Workspace: prefer N*N [R] + N [tau] + N*NB [geqrf work]
          lwork2 = lwork - iwork + 1;
          magma_sgeqrf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );

          // Copy R to WORK(IR), zeroing out below it
          lapack::lacpy('U', n, n, A, lda, &work[ir], ldwrkr);
          lapack::laset('L', n_1, n_1, c_zero, c_zero, &work[ir+1], ldwrkr);

          // Generate Q in A
          // Workspace: need   N*N [R] + N [tau] + N    [orgqr work]
          // Workspace: prefer N*N [R] + N [tau] + N*NB [orgqr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgqr2(     m,  n,  n, A,  lda, &work[itau], &ierr );

          ie    = itau;
          itauq = ie    + n;
          itaup = itauq + n;
          iwork = itaup + n;

          // Bidiagonalize R in WORK(IR)
          // Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N      [gebrd work]
          // Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
          lwork2 = lwork - iwork + 1;
          magma_sgebrd(      n,  n, &work[ir],  ldwrkr, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );

          // Generate left vectors bidiagonalizing R in WORK(IR)
          // Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N    [orgbr work]
          // Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + N*NB [orgbr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgbr( MagmaQ,   n,  n,  n, &work[ir],  ldwrkr, &work[itauq], &work[iwork],  lwork2, &ierr );
          iwork = ie + n;

          // Perform bidiagonal QR iteration, computing left
          // singular vectors of R in WORK(IR)
          // Workspace: need   N*N [R] + N [e] + 4*N [bdsqr work]
          lapack::bdsqr('U', n, izero, n, izero, s, &work[ie], dummy, ione, &work[ir], ldwrkr, dummy, ione, &work[iwork], info);

          // Multiply Q in A by left singular vectors of R in
          // WORK(IR), storing result in U
          // Workspace: need   N*N [R]
          blas::gemm('N', 'N', m, n, n,
                     c_one,  A,         lda,
                             &work[ir], ldwrkr,
                     c_zero, U,         ldu);
          }
        else
          {
          // Path 4-slow: Insufficient workspace for a fast algorithm
          itau  = 1;
          iwork = itau + n;

          // Compute A=Q*R, copying result to U
          // Workspace: need   N [tau] + N    [geqrf work]
          // Workspace: prefer N [tau] + N*NB [geqrf work]
          lwork2 = lwork - iwork + 1;
          magma_sgeqrf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
          lapack::lacpy('L', m, n, A, lda, U, ldu);

          // Generate Q in U
          // Workspace: need   N [tau] + N    [orgqr work]
          // Workspace: prefer N [tau] + N*NB [orgqr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgqr2(     m,  n,  n, U,  ldu, &work[itau], &ierr );

          ie    = itau;
          itauq = ie    + n;
          itaup = itauq + n;
          iwork = itaup + n;

          // Zero out below R in A
          lapack::laset('L', n_1, n_1, c_zero, c_zero, &A[1], lda);

          // Bidiagonalize R in A
          // Workspace: need   3*N [e, tauq, taup] + N      [gebrd work]
          // Workspace: prefer 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
          lwork2 = lwork - iwork + 1;
          magma_sgebrd(      n,  n, A,  lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );

          // Multiply Q in U by left vectors bidiagonalizing R
          // Workspace: need   3*N [e, tauq, taup] + M    [ormbr work]
          // Workspace: prefer 3*N [e, tauq, taup] + M*NB [ormbr work]
          lwork2 = lwork - iwork + 1;
          magma_sormbr( MagmaQ, MagmaRight, MagmaNoTrans,  m,  n,  n, A,  lda, &work[itauq], U,  ldu, &work[iwork],  lwork2, &ierr );
          iwork = ie + n;

          // Perform bidiagonal QR iteration, computing left
          // singular vectors of A in U
          // Workspace: need     N [e] + 4*N [bdsqr work]
          lapack::bdsqr('U', n, izero, m, izero, s, &work[ie], dummy, ione, U, ldu, dummy, ione, &work[iwork], info);
          }
        }
      else if (want_us && want_vo)
        {
        // Path 5 (M >> N, JOBU='S', JOBVT='O')
        // N left singular vectors to be computed in U and
        // N right singular vectors to be overwritten on A
        if (lwork >= 2*n*n + wrkbl)
          {
          // Path 5-fast: Sufficient workspace for a fast algorithm
          if (lwork >= wrkbl + 2*lda*n)
            {
            // WORK(IU) is LDA by N
            // WORK(IR) is LDA by N
            ldwrku = lda;
            ldwrkr = lda;
            }
          else if (lwork >= wrkbl + (lda + n)*n)
            {
            // WORK(IU) is LDA by N
            // WORK(IR) is N by N
            ldwrku = lda;
            ldwrkr = n;
            }
          else
            {
            // WORK(IU) is N by N
            // WORK(IR) is N by N
            ldwrku = n;
            ldwrkr = n;
            }
          iu    = 1;
          ir    = iu + ldwrku * n;
          itau  = ir + ldwrkr * n;
          iwork = itau + n;

          // Compute A=Q*R
          // Workspace: need   2*N*N [U,R] + N [tau] + N    [geqrf work]
          // Workspace: prefer 2*N*N [U,R] + N [tau] + N*NB [geqrf work]
          lwork2 = lwork - iwork + 1;
          magma_sgeqrf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );

          // Copy R to WORK(IU), zeroing out below it
          lapack::lacpy('U', n, n, A, lda, &work[iu], ldwrku);
          lapack::laset('L', n_1, n_1, c_zero, c_zero, &work[iu+1], ldwrku);

          // Generate Q in A
          // Workspace: need   2*N*N [U,R] + N [tau] + N    [orgqr work]
          // Workspace: prefer 2*N*N [U,R] + N [tau] + N*NB [orgqr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgqr2(     m,  n,  n, A,  lda, &work[itau], &ierr );

          ie    = itau;
          itauq = ie    + n;
          itaup = itauq + n;
          iwork = itaup + n;

          // Bidiagonalize R in WORK(IU), copying result to
          // WORK(IR)
          // Workspace: need   2*N*N [U,R] + 3*N [e, tauq, taup] + N      [gebrd work]
          // Workspace: prefer 2*N*N [U,R] + 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
          lwork2 = lwork - iwork + 1;
          magma_sgebrd(      n,  n, &work[iu],  ldwrku, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
          lapack::lacpy('U', n, n, &work[iu], ldwrku, &work[ir], ldwrkr);

          // Generate left bidiagonalizing vectors in WORK(IU)
          // Workspace: need   2*N*N [U,R] + 3*N [e, tauq, taup] + N    [orgbr work]
          // Workspace: prefer 2*N*N [U,R] + 3*N [e, tauq, taup] + N*NB [orgbr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgbr( MagmaQ,   n,  n,  n, &work[iu],  ldwrku, &work[itauq], &work[iwork],  lwork2, &ierr );

          // Generate right bidiagonalizing vectors in WORK(IR)
          // Workspace: need   2*N*N [U,R] + 3*N [e, tauq, taup] + N    [orgbr work]
          // Workspace: prefer 2*N*N [U,R] + 3*N [e, tauq, taup] + N*NB [orgbr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgbr( MagmaP,   n,  n,  n, &work[ir],  ldwrkr, &work[itaup], &work[iwork],  lwork2, &ierr );
          iwork = ie + n;

          // Perform bidiagonal QR iteration, computing left
          // singular vectors of R in WORK(IU) and computing
          // right singular vectors of R in WORK(IR)
          // Workspace: need   2*N*N [U,R] + N [e] + 4*N [bdsqr work]
          lapack::bdsqr('U', n, n, n, izero, s, &work[ie], &work[ir], ldwrkr, &work[iu], ldwrku, dummy, ione, &work[iwork], info);

          // Multiply Q in A by left singular vectors of R in
          // WORK(IU), storing result in U
          // Workspace: need   2*N*N [U,R]
          blas::gemm('N', 'N', m, n, n,
                     c_one,  A,         lda,
                             &work[iu], ldwrku,
                     c_zero, U,         ldu);

          // Copy right singular vectors of R to A
          // Workspace: need   2*N*N [U,R]
          lapack::lacpy('F', n, n, &work[ir], ldwrkr, A, lda);
          }
        else
          {
          // Path 5-slow: Insufficient workspace for a fast algorithm
          itau  = 1;
          iwork = itau + n;

          // Compute A=Q*R, copying result to U
          // Workspace: need   N [tau] + N    [geqrf work]
          // Workspace: prefer N [tau] + N*NB [geqrf work]
          lwork2 = lwork - iwork + 1;
          magma_sgeqrf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
          lapack::lacpy('L', m, n, A, lda, U, ldu);

          // Generate Q in U
          // Workspace: need   N [tau] + N    [orgqr work]
          // Workspace: prefer N [tau] + N*NB [orgqr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgqr2(     m,  n,  n, U,  ldu, &work[itau], &ierr );

          ie    = itau;
          itauq = ie    + n;
          itaup = itauq + n;
          iwork = itaup + n;

          // Zero out below R in A
          lapack::laset('L', n_1, n_1, c_zero, c_zero, &A[1], lda);

          // Bidiagonalize R in A
          // Workspace: need   3*N [e, tauq, taup] + N      [gebrd work]
          // Workspace: prefer 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
          lwork2 = lwork - iwork + 1;
          magma_sgebrd(      n,  n, A,  lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );

          // Multiply Q in U by left vectors bidiagonalizing R
          // Workspace: need   3*N [e, tauq, taup] + M    [ormbr work]
          // Workspace: prefer 3*N [e, tauq, taup] + M*NB [ormbr work]
          lwork2 = lwork - iwork + 1;
          magma_sormbr( MagmaQ, MagmaRight, MagmaNoTrans,  m,  n,  n, A,  lda, &work[itauq], U,  ldu, &work[iwork],  lwork2, &ierr );

          // Generate right vectors bidiagonalizing R in A
          // Workspace: need   3*N [e, tauq, taup] + N    [orgbr work]
          // Workspace: prefer 3*N [e, tauq, taup] + N*NB [orgbr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgbr( MagmaP,   n,  n,  n, A,  lda, &work[itaup], &work[iwork],  lwork2, &ierr );
          iwork = ie + n;

          // Perform bidiagonal QR iteration, computing left
          // singular vectors of A in U and computing right
          // singular vectors of A in A
          // Workspace: need     N [e] + 4*N [bdsqr work]
          lapack::bdsqr('U', n, n, m, izero, s, &work[ie], A, lda, U, ldu, dummy, ione, &work[iwork], info);
          }
        }
      else if (want_us && want_vas)
        {
        // Path 6 (M >> N, JOBU='S', JOBVT='S' or 'A')
        // N left singular vectors to be computed in U and
        // N right singular vectors to be computed in VT
        if (lwork >= n*n + wrkbl)
          {
          // Path 6-fast: Sufficient workspace for a fast algorithm
          if (lwork >= wrkbl + lda*n)
            {
            // WORK(IU) is LDA by N
            ldwrku = lda;
            }
          else
            {
            // WORK(IU) is N by N
            ldwrku = n;
            }
          iu    = 1;
          itau  = iu + ldwrku * n;
          iwork = itau + n;

          // Compute A=Q*R
          // Workspace: need   N*N [U] + N [tau] + N    [geqrf work]
          // Workspace: prefer N*N [U] + N [tau] + N*NB [geqrf work]
          lwork2 = lwork - iwork + 1;
          magma_sgeqrf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );

          // Copy R to WORK(IU), zeroing out below it
          lapack::lacpy('U', n, n, A, lda, &work[iu], ldwrku);
          lapack::laset('L', n_1, n_1, c_zero, c_zero, &work[iu+1], ldwrku);

          // Generate Q in A
          // Workspace: need   N*N [U] + N [tau] + N    [orgqr work]
          // Workspace: prefer N*N [U] + N [tau] + N*NB [orgqr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgqr2(     m,  n,  n, A,  lda, &work[itau], &ierr );

          ie    = itau;
          itauq = ie    + n;
          itaup = itauq + n;
          iwork = itaup + n;

          // Bidiagonalize R in WORK(IU), copying result to VT
          // Workspace: need   N*N [U] + 3*N [e, tauq, taup] + N      [gebrd work]
          // Workspace: prefer N*N [U] + 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
          lwork2 = lwork - iwork + 1;
          magma_sgebrd(      n,  n, &work[iu],  ldwrku, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
          lapack::lacpy('U', n, n, &work[iu], ldwrku, VT, ldvt);

          // Generate left bidiagonalizing vectors in WORK(IU)
          // Workspace: need   N*N [U] + 3*N [e, tauq, taup] + N    [orgbr work]
          // Workspace: prefer N*N [U] + 3*N [e, tauq, taup] + N*NB [orgbr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgbr( MagmaQ,   n,  n,  n, &work[iu],  ldwrku, &work[itauq], &work[iwork],  lwork2, &ierr );

          // Generate right bidiagonalizing vectors in VT
          // Workspace: need   N*N [U] + 3*N [e, tauq, taup] + N    [orgbr work]
          // Workspace: prefer N*N [U] + 3*N [e, tauq, taup] + N*NB [orgbr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgbr( MagmaP,   n,  n,  n, VT,  ldvt, &work[itaup], &work[iwork],  lwork2, &ierr );
          iwork = ie + n;

          // Perform bidiagonal QR iteration, computing left
          // singular vectors of R in WORK(IU) and computing
          // right singular vectors of R in VT
          // Workspace: need   N*N [U] + N [e] + 4*N [bdsqr work]
          lapack::bdsqr('U', n, n, n, izero, s, &work[ie], VT, ldvt, &work[iu], ldwrku, dummy, ione, &work[iwork], info);

          // Multiply Q in A by left singular vectors of R in
          // WORK(IU), storing result in U
          // Workspace: need   N*N [U]
          blas::gemm('N', 'N', m, n, n,
                     c_one,  A,         lda,
                             &work[iu], ldwrku,
                     c_zero, U,         ldu);
          }
        else
          {
          // Path 6-slow: Insufficient workspace for a fast algorithm
          itau  = 1;
          iwork = itau + n;

          // Compute A=Q*R, copying result to U
          // Workspace: need   N [tau] + N    [geqrf work]
          // Workspace: prefer N [tau] + N*NB [geqrf work]
          lwork2 = lwork - iwork + 1;
          magma_sgeqrf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
          lapack::lacpy('L', m, n, A, lda, U, ldu);

          // Generate Q in U
          // Workspace: need   N [tau] + N    [orgqr work]
          // Workspace: prefer N [tau] + N*NB [orgqr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgqr2(     m,  n,  n, U,  ldu, &work[itau], &ierr );

          // Copy R to VT, zeroing out below it
          lapack::lacpy('U', n, n, A, lda, VT, ldvt);
          if (n > 1)
            {
            lapack::laset('L', n_1, n_1, c_zero, c_zero, &VT[1], ldvt);
            }
          ie    = itau;
          itauq = ie    + n;
          itaup = itauq + n;
          iwork = itaup + n;

          // Bidiagonalize R in VT
          // Workspace: need   3*N [e, tauq, taup] + N      [gebrd work]
          // Workspace: prefer 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
          lwork2 = lwork - iwork + 1;
          magma_sgebrd(      n,  n, VT,  ldvt, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );

          // Multiply Q in U by left bidiagonalizing vectors in VT
          // Workspace: need   3*N [e, tauq, taup] + M    [ormbr work]
          // Workspace: prefer 3*N [e, tauq, taup] + M*NB [ormbr work]
          lwork2 = lwork - iwork + 1;
          magma_sormbr( MagmaQ, MagmaRight, MagmaNoTrans,  m,  n,  n, VT,  ldvt, &work[itauq], U,  ldu, &work[iwork],  lwork2, &ierr );

          // Generate right bidiagonalizing vectors in VT
          // Workspace: need   3*N [e, tauq, taup] + N    [orgbr work]
          // Workspace: prefer 3*N [e, tauq, taup] + N*NB [orgbr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgbr( MagmaP,   n,  n,  n, VT,  ldvt, &work[itaup], &work[iwork],  lwork2, &ierr );
          iwork = ie + n;

          // Perform bidiagonal QR iteration, computing left
          // singular vectors of A in U and computing right
          // singular vectors of A in VT
          // Workspace: need     N [e] + 4*N [bdsqr work]
          lapack::bdsqr('U', n, n, m, izero, s, &work[ie], VT, ldvt, U, ldu, dummy, ione, &work[iwork], info);
          }
        }
      else if (want_ua && want_vn)
        {
        // Path 7 (M >> N, JOBU='A', JOBVT='N')
        // M left singular vectors to be computed in U and
        // no right singular vectors to be computed
        if (lwork >= n*n + wrkbl)
          {
          // Path 7-fast: Sufficient workspace for a fast algorithm
          if (lwork >= wrkbl + lda*n)
            {
            // WORK(IR) is LDA by N
            ldwrkr = lda;
            }
          else
            {
            // WORK(IR) is N by N
            ldwrkr = n;
            }
          ir    = 1;
          itau  = ir + ldwrkr * n;
          iwork = itau + n;

          // Compute A=Q*R, copying result to U
          // Workspace: need   N*N [R] + N [tau] + N    [geqrf work]
          // Workspace: prefer N*N [R] + N [tau] + N*NB [geqrf work]
          lwork2 = lwork - iwork + 1;
          magma_sgeqrf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
          lapack::lacpy('L', m, n, A, lda, U, ldu);

          // Copy R to WORK(IR), zeroing out below it
          lapack::lacpy('U', n, n, A, lda, &work[ir], ldwrkr);
          lapack::laset('L', n_1, n_1, c_zero, c_zero, &work[ir+1], ldwrkr);

          // Generate Q in U
          // Workspace: need   N*N [R] + N [tau] + M    [orgqr work]
          // Workspace: prefer N*N [R] + N [tau] + M*NB [orgqr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgqr2(     m,  m,  n, U,  ldu, &work[itau], &ierr );

          ie    = itau;
          itauq = ie    + n;
          itaup = itauq + n;
          iwork = itaup + n;

          // Bidiagonalize R in WORK(IR)
          // Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N      [gebrd work]
          // Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
          lwork2 = lwork - iwork + 1;
          magma_sgebrd(      n,  n, &work[ir],  ldwrkr, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );

          // Generate left bidiagonalizing vectors in WORK(IR)
          // Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N    [orgbr work]
          // Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + N*NB [orgbr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgbr( MagmaQ,   n,  n,  n, &work[ir],  ldwrkr, &work[itauq], &work[iwork],  lwork2, &ierr );
          iwork = ie + n;

          // Perform bidiagonal QR iteration, computing left
          // singular vectors of R in WORK(IR)
          // Workspace: need   N*N [R] + N [e] + 4*N [bdsqr work]
          lapack::bdsqr('U', n, izero, n, izero, s, &work[ie], dummy, ione, &work[ir], ldwrkr, dummy, ione, &work[iwork], info);

          // Multiply Q in U by left singular vectors of R in
          // WORK(IR), storing result in A
          // Workspace: need   N*N [R]
          blas::gemm('N', 'N', m, n, n,
                     c_one,  U,         ldu,
                             &work[ir], ldwrkr,
                     c_zero, A,         lda);

          // Copy left singular vectors of A from A to U
          lapack::lacpy('F', m, n, A, lda, U, ldu);
          }
        else
          {
          // Path 7-slow: Insufficient workspace for a fast algorithm
          itau  = 1;
          iwork = itau + n;

          // Compute A=Q*R, copying result to U
          // Workspace: need   N [tau] + N    [geqrf work]
          // Workspace: prefer N [tau] + N*NB [geqrf work]
          lwork2 = lwork - iwork + 1;
          magma_sgeqrf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
          lapack::lacpy('L', m, n, A, lda, U, ldu);

          // Generate Q in U
          // Workspace: need   N [tau] + M    [orgqr work]
          // Workspace: prefer N [tau] + M*NB [orgqr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgqr2(     m,  m,  n, U,  ldu, &work[itau], &ierr );

          ie    = itau;
          itauq = ie    + n;
          itaup = itauq + n;
          iwork = itaup + n;

          // Zero out below R in A
          lapack::laset('L', n_1, n_1, c_zero, c_zero, &A[1], lda);

          // Bidiagonalize R in A
          // Workspace: need   3*N [e, tauq, taup] + N      [gebrd work]
          // Workspace: prefer 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
          lwork2 = lwork - iwork + 1;
          magma_sgebrd(      n,  n, A,  lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );

          // Multiply Q in U by left bidiagonalizing vectors in A
          // Workspace: need   3*N [e, tauq, taup] + M    [ormbr work]
          // Workspace: prefer 3*N [e, tauq, taup] + M*NB [ormbr work]
          lwork2 = lwork - iwork + 1;
          magma_sormbr( MagmaQ, MagmaRight, MagmaNoTrans,  m,  n,  n, A,  lda, &work[itauq], U,  ldu, &work[iwork],  lwork2, &ierr );
          iwork = ie + n;

          // Perform bidiagonal QR iteration, computing left
          // singular vectors of A in U
          // Workspace: need     N [e] + 4*N [bdsqr work]
          lapack::bdsqr('U', n, izero, m, izero, s, &work[ie], dummy, ione, U, ldu, dummy, ione, &work[iwork], info);
          }
        }
      else if (want_ua && want_vo)
        {
        // Path 8 (M >> N, JOBU='A', JOBVT='O')
        // M left singular vectors to be computed in U and
        // N right singular vectors to be overwritten on A
        if (lwork >= 2*n*n + wrkbl)
          {
          // Path 8-fast: Sufficient workspace for a fast algorithm
          if (lwork >= wrkbl + 2*lda*n)
            {
            // WORK(IU) is LDA by N
            // WORK(IR) is LDA by N
            ldwrku = lda;
            ldwrkr = lda;
            }
          else if (lwork >= wrkbl + (lda + n)*n)
            {
            // WORK(IU) is LDA by N
            // WORK(IR) is N by N
            ldwrku = lda;
            ldwrkr = n;
            }
          else
            {
            // WORK(IU) is N by N
            // WORK(IR) is N by N
            ldwrku = n;
            ldwrkr = n;
            }
          iu    = 1;
          ir    = iu + ldwrku * n;
          itau  = ir + ldwrkr * n;
          iwork = itau + n;

          // Compute A=Q*R, copying result to U
          // Workspace: need   2*N*N [U,R] + N [tau] + N    [geqrf work]
          // Workspace: prefer 2*N*N [U,R] + N [tau] + N*NB [geqrf work]
          lwork2 = lwork - iwork + 1;
          magma_sgeqrf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
          lapack::lacpy('L', m, n, A, lda, U, ldu);

          // Generate Q in U
          // Workspace: need   2*N*N [U,R] + N [tau] + M    [orgqr work]
          // Workspace: prefer 2*N*N [U,R] + N [tau] + M*NB [orgqr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgqr2(     m,  m,  n, U,  ldu, &work[itau], &ierr );

          // Copy R to WORK(IU), zeroing out below it
          lapack::lacpy('U', n, n, A, lda, &work[iu], ldwrku);
          lapack::laset('L', n_1, n_1, c_zero, c_zero, &work[iu+1], ldwrku);
          ie    = itau;
          itauq = ie    + n;
          itaup = itauq + n;
          iwork = itaup + n;

          // Bidiagonalize R in WORK(IU), copying result to
          // WORK(IR)
          // Workspace: need   2*N*N [U,R] + 3*N [e, tauq, taup] + N      [gebrd work]
          // Workspace: prefer 2*N*N [U,R] + 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
          lwork2 = lwork - iwork + 1;
          magma_sgebrd(      n,  n, &work[iu],  ldwrku, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
          lapack::lacpy('U', n, n, &work[iu], ldwrku, &work[ir], ldwrkr);

          // Generate left bidiagonalizing vectors in WORK(IU)
          // Workspace: need   2*N*N [U,R] + 3*N [e, tauq, taup] + N    [orgbr work]
          // Workspace: prefer 2*N*N [U,R] + 3*N [e, tauq, taup] + N*NB [orgbr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgbr( MagmaQ,   n,  n,  n, &work[iu],  ldwrku, &work[itauq], &work[iwork],  lwork2, &ierr );

          // Generate right bidiagonalizing vectors in WORK(IR)
          // Workspace: need   2*N*N [U,R] + 3*N [e, tauq, taup] + N    [orgbr work]
          // Workspace: prefer 2*N*N [U,R] + 3*N [e, tauq, taup] + N*NB [orgbr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgbr( MagmaP,   n,  n,  n, &work[ir],  ldwrkr, &work[itaup], &work[iwork],  lwork2, &ierr );
          iwork = ie + n;

          // Perform bidiagonal QR iteration, computing left
          // singular vectors of R in WORK(IU) and computing
          // right singular vectors of R in WORK(IR)
          // Workspace: need   2*N*N [U,R] + N [e] + 4*N [bdsqr work]
          lapack::bdsqr('U', n, n, n, izero, s, &work[ie], &work[ir], ldwrkr, &work[iu], ldwrku, dummy, ione, &work[iwork], info);

          // Multiply Q in U by left singular vectors of R in
          // WORK(IU), storing result in A
          // Workspace: need   N*N [U]
          blas::gemm('N', 'N', m, n, n,
                     c_one,  U,         ldu,
                             &work[iu], ldwrku,
                     c_zero, A,         lda);

          // Copy left singular vectors of A from A to U
          lapack::lacpy('F', m, n, A, lda, U, ldu);

          // Copy right singular vectors of R from WORK(IR) to A
          lapack::lacpy('F', n, n, &work[ir], ldwrkr, A, lda);
          }
        else
          {
          // Path 8-slow: Insufficient workspace for a fast algorithm
          itau  = 1;
          iwork = itau + n;

          // Compute A=Q*R, copying result to U
          // Workspace: need   N [tau] + N    [geqrf work]
          // Workspace: prefer N [tau] + N*NB [geqrf work]
          lwork2 = lwork - iwork + 1;
          magma_sgeqrf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
          lapack::lacpy('L', m, n, A, lda, U, ldu);

          // Generate Q in U
          // Workspace: need   N [tau] + M    [orgqr work]
          // Workspace: prefer N [tau] + M*NB [orgqr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgqr2(     m,  m,  n, U,  ldu, &work[itau], &ierr );

          ie    = itau;
          itauq = ie    + n;
          itaup = itauq + n;
          iwork = itaup + n;

          // Zero out below R in A
          lapack::laset('L', n_1, n_1, c_zero, c_zero, &A[1], lda);

          // Bidiagonalize R in A
          // Workspace: need   3*N [e, tauq, taup] + N      [gebrd work]
          // Workspace: prefer 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
          lwork2 = lwork - iwork + 1;
          magma_sgebrd(      n,  n, A,  lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );

          // Multiply Q in U by left bidiagonalizing vectors in A
          // Workspace: need   3*N [e, tauq, taup] + M    [ormbr work]
          // Workspace: prefer 3*N [e, tauq, taup] + M*NB [ormbr work]
          lwork2 = lwork - iwork + 1;
          magma_sormbr( MagmaQ, MagmaRight, MagmaNoTrans,  m,  n,  n, A,  lda, &work[itauq], U,  ldu, &work[iwork],  lwork2, &ierr );

          // Generate right bidiagonalizing vectors in A
          // Workspace: need   3*N [e, tauq, taup] + N    [orgbr work]
          // Workspace: prefer 3*N [e, tauq, taup] + N*NB [orgbr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgbr( MagmaP,   n,  n,  n, A,  lda, &work[itaup], &work[iwork],  lwork2, &ierr );
          iwork = ie + n;

          // Perform bidiagonal QR iteration, computing left
          // singular vectors of A in U and computing right
          // singular vectors of A in A
          // Workspace: need     N [e] + 4*N [bdsqr work]
          lapack::bdsqr('U', n, n, m, izero, s, &work[ie], A, lda, U, ldu, dummy, ione, &work[iwork], info);
          }
        }
      else if (want_ua && want_vas)
        {
        // Path 9 (M >> N, JOBU='A', JOBVT='S' or 'A')
        // M left singular vectors to be computed in U and
        // N right singular vectors to be computed in VT
        if (lwork >= n*n + wrkbl)
          {
          // Path 9-fast: Sufficient workspace for a fast algorithm
          if (lwork >= wrkbl + lda*n)
            {
            // WORK(IU) is LDA by N
            ldwrku = lda;
            }
          else
            {
            // WORK(IU) is N by N
            ldwrku = n;
            }
          iu    = 1;
          itau  = iu + ldwrku * n;
          iwork = itau + n;

          // Compute A=Q*R, copying result to U
          // Workspace: need   N*N [U] + N [tau] + N    [geqrf work]
          // Workspace: prefer N*N [U] + N [tau] + N*NB [geqrf work]
          lwork2 = lwork - iwork + 1;
          magma_sgeqrf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
          lapack::lacpy('L', m, n, A, lda, U, ldu);

          // Generate Q in U
          // Workspace: need   N*N [U] + N [tau] + M    [orgqr work]
          // Workspace: prefer N*N [U] + N [tau] + M*NB [orgqr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgqr2(     m,  m,  n, U,  ldu, &work[itau], &ierr );

          // Copy R to WORK(IU), zeroing out below it
          lapack::lacpy('U', n, n, A, lda, &work[iu], ldwrku);
          lapack::laset('L', n_1, n_1, c_zero, c_zero, &work[iu+1], ldwrku);
          ie    = itau;
          itauq = ie    + n;
          itaup = itauq + n;
          iwork = itaup + n;

          // Bidiagonalize R in WORK(IU), copying result to VT
          // Workspace: need   N*N [U] + 3*N [e, tauq, taup] + N      [gebrd work]
          // Workspace: prefer N*N [U] + 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
          lwork2 = lwork - iwork + 1;
          magma_sgebrd(      n,  n, &work[iu],  ldwrku, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
          lapack::lacpy('U', n, n, &work[iu], ldwrku, VT, ldvt);

          // Generate left bidiagonalizing vectors in WORK(IU)
          // Workspace: need   N*N [U] + 3*N [e, tauq, taup] + N    [orgbr work]
          // Workspace: prefer N*N [U] + 3*N [e, tauq, taup] + N*NB [orgbr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgbr( MagmaQ,   n,  n,  n, &work[iu],  ldwrku, &work[itauq], &work[iwork],  lwork2, &ierr );

          // Generate right bidiagonalizing vectors in VT
          // Workspace: need   N*N [U] + 3*N [e, tauq, taup] + N    [orgbr work]
          // Workspace: prefer N*N [U] + 3*N [e, tauq, taup] + N*NB [orgbr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgbr( MagmaP,   n,  n,  n, VT,  ldvt, &work[itaup], &work[iwork],  lwork2, &ierr );
          iwork = ie + n;

          // Perform bidiagonal QR iteration, computing left
          // singular vectors of R in WORK(IU) and computing
          // right singular vectors of R in VT
          // Workspace: need   N*N [U] + N [e] + 4*N [bdsqr work]
          lapack::bdsqr('U', n, n, n, izero, s, &work[ie], VT, ldvt, &work[iu], ldwrku, dummy, ione, &work[iwork], info);

          // Multiply Q in U by left singular vectors of R in
          // WORK(IU), storing result in A
          // Workspace: need   N*N [U]
          blas::gemm('N', 'N', m, n, n,
                     c_one,  U,         ldu,
                             &work[iu], ldwrku,
                     c_zero, A,         lda);

          // Copy left singular vectors of A from A to U
          lapack::lacpy('F', m, n, A, lda, U, ldu);
          }
        else
          {
          // Path 9-slow: Insufficient workspace for a fast algorithm
          itau  = 1;
          iwork = itau + n;

          // Compute A=Q*R, copying result to U
          // Workspace: need   N [tau] + N    [geqrf work]
          // Workspace: prefer N [tau] + N*NB [geqrf work]
          lwork2 = lwork - iwork + 1;
          magma_sgeqrf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
          lapack::lacpy('L', m, n, A, lda, U, ldu);

          // Generate Q in U
          // Workspace: need   N [tau] + M    [orgqr work]
          // Workspace: prefer N [tau] + M*NB [orgqr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgqr2(     m,  m,  n, U,  ldu, &work[itau], &ierr );

          // Copy R from A to VT, zeroing out below it
          lapack::lacpy('U', n, n, A, lda, VT, ldvt);
          if (n > 1)
            {
            lapack::laset('L', n_1, n_1, c_zero, c_zero, &VT[1], ldvt);
            }
          ie    = itau;
          itauq = ie    + n;
          itaup = itauq + n;
          iwork = itaup + n;

          // Bidiagonalize R in VT
          // Workspace: need   3*N [e, tauq, taup] + N      [gebrd work]
          // Workspace: prefer 3*N [e, tauq, taup] + 2*N*NB [gebrd work]
          lwork2 = lwork - iwork + 1;
          magma_sgebrd(      n,  n, VT,  ldvt, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );

          // Multiply Q in U by left bidiagonalizing vectors in VT
          // Workspace: need   3*N [e, tauq, taup] + M    [ormbr work]
          // Workspace: prefer 3*N [e, tauq, taup] + M*NB [ormbr work]
          lwork2 = lwork - iwork + 1;
          magma_sormbr( MagmaQ, MagmaRight, MagmaNoTrans,  m,  n,  n, VT,  ldvt, &work[itauq], U,  ldu, &work[iwork],  lwork2, &ierr );

          // Generate right bidiagonalizing vectors in VT
          // Workspace: need   3*N [e, tauq, taup] + N    [orgbr work]
          // Workspace: prefer 3*N [e, tauq, taup] + N*NB [orgbr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgbr( MagmaP,   n,  n,  n, VT,  ldvt, &work[itaup], &work[iwork],  lwork2, &ierr );
          iwork = ie + n;

          // Perform bidiagonal QR iteration, computing left
          // singular vectors of A in U and computing right
          // singular vectors of A in VT
          // Workspace: need     N [e] + 4*N [bdsqr work]
          lapack::bdsqr('U', n, n, m, izero, s, &work[ie], VT, ldvt, U, ldu, dummy, ione, &work[iwork], info);
          }
        }
      }
    else
      {
      // M < MNTHR
      // Path 10 (M >= N, but not much larger)
      // Reduce to bidiagonal form without QR decomposition
      ie    = 1;
      itauq = ie    + n;
      itaup = itauq + n;
      iwork = itaup + n;

      // Bidiagonalize A
      // Workspace: need   3*N [e, tauq, taup] + M        [gebrd work]
      // Workspace: prefer 3*N [e, tauq, taup] + (M+N)*NB [gebrd work]
      lwork2 = lwork - iwork + 1;

      magma_sgebrd(      m,  n, A,  lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );

      if (want_uas)
        {
        // If left singular vectors desired in U, copy result to U
        // and generate left bidiagonalizing vectors in U
        // Workspace: need     3*N [e, tauq, taup] + NCU    [orgbr work]
        // Workspace: prefer   3*N [e, tauq, taup] + NCU*NB [orgbr work]
        lapack::lacpy('L', m, n, A, lda, U, ldu);
        if (want_us)
          {
          ncu = n;
          }
        else
          {
          assert(want_ua);
          ncu = m;
          }
        lwork2 = lwork - iwork + 1;
        magma_sorgbr( MagmaQ,   m,  ncu,  n, U,  ldu, &work[itauq], &work[iwork],  lwork2, &ierr );
        }
      if (want_vas)
        {
        // If right singular vectors desired in VT, copy result to
        // VT and generate right bidiagonalizing vectors in VT
        // Workspace: need     3*N [e, tauq, taup] + N    [orgbr work]
        // Workspace: prefer   3*N [e, tauq, taup] + N*NB [orgbr work]
        lapack::lacpy('U', n, n, A, lda, VT, ldvt);
        lwork2 = lwork - iwork + 1;
        magma_sorgbr( MagmaP,   n,  n,  n, VT,  ldvt, &work[itaup], &work[iwork],  lwork2, &ierr );
        }
      if (want_uo)
        {
        // If left singular vectors desired in A, generate left
        // bidiagonalizing vectors in A
        // Workspace: need     3*N [e, tauq, taup] + N    [orgbr work]
        // Workspace: prefer   3*N [e, tauq, taup] + N*NB [orgbr work]
        lwork2 = lwork - iwork + 1;
        magma_sorgbr( MagmaQ,   m,  n,  n, A,  lda, &work[itauq], &work[iwork],  lwork2, &ierr );
        }
      if (want_vo)
        {
        // If right singular vectors desired in A, generate right
        // bidiagonalizing vectors in A
        // Workspace: need     3*N [e, tauq, taup] + N    [orgbr work]
        // Workspace: prefer   3*N [e, tauq, taup] + N*NB [orgbr work]
        lwork2 = lwork - iwork + 1;
        magma_sorgbr( MagmaP,   n,  n,  n, A,  lda, &work[itaup], &work[iwork],  lwork2, &ierr );
        }

      iwork = ie + n;
      if (want_uas || want_uo)
        {
        nru = m;
        }
      if (want_un)
        {
        nru = 0;
        }
      if (want_vas || want_vo)
        {
        ncvt = n;
        }
      if (want_vn)
        {
        ncvt = 0;
        }
      if (! want_uo && ! want_vo)
        {
        // Perform bidiagonal QR iteration, if desired, computing
        // left singular vectors in U and computing right singular
        // vectors in VT
        // Workspace: need       N [e] + 4*N [bdsqr work]
        lapack::bdsqr('U', n, ncvt, nru, izero, s, &work[ie], VT, ldvt, U, ldu, dummy, ione, &work[iwork], info);
        }
      else if (! want_uo && want_vo)
        {
        // Perform bidiagonal QR iteration, if desired, computing
        // left singular vectors in U and computing right singular
        // vectors in A
        // Workspace: need       N [e] + 4*N [bdsqr work]
        lapack::bdsqr('U', n, ncvt, nru, izero, s, &work[ie], A, lda, U, ldu, dummy, ione, &work[iwork], info);
        }
      else
        {
        // Perform bidiagonal QR iteration, if desired, computing
        // left singular vectors in A and computing right singular
        // vectors in VT
        // Workspace: need       N [e] + 4*N [bdsqr work]
        lapack::bdsqr('U', n, ncvt, nru, izero, s, &work[ie], VT, ldvt, A, lda, dummy, ione, &work[iwork], info);
        }
      }
    }
  else
    {
    // m < n
    // A has more columns than rows.
    // If A has sufficiently more columns than rows, first reduce using
    // the LQ decomposition (if sufficient workspace available)
    if (n >= mnthr)
      {
      if (want_vn)
        {
        // Path 1t (N >> M, JOBVT='N')
        // No right singular vectors to be computed
        itau  = 1;
        iwork = itau + m;

        // Compute A=L*Q
        // Workspace: need   M [tau] + M    [gelqf work]
        // Workspace: prefer M [tau] + M*NB [gelqf work]
        lwork2 = lwork - iwork + 1;
        magma_sgelqf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );

        // Zero out above L
        lapack::laset('U', m_1, m_1, c_zero, c_zero, &A[1 * lda], lda);
        ie    = 1;
        itauq = ie + m;
        itaup = itauq + m;
        iwork = itaup + m;

        // Bidiagonalize L in A
        // Workspace: need   3*M [e, tauq, taup] + M      [gebrd work]
        // Workspace: prefer 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
        lwork2 = lwork - iwork + 1;
        magma_sgebrd(      m,  m, A,  lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );

        if (want_uo || want_uas)
          {
          // If left singular vectors desired, generate Q
          // Workspace: need   3*M [e, tauq, taup] + M    [orgbr work]
          // Workspace: prefer 3*M [e, tauq, taup] + M*NB [orgbr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgbr( MagmaQ,   m,  m,  m, A,  lda, &work[itauq], &work[iwork],  lwork2, &ierr );
          }
        iwork = ie + m;
        nru = 0;
        if (want_uo || want_uas)
          {
          nru = m;
          }

        // Perform bidiagonal QR iteration, computing left singular
        // vectors of A in A if desired
        // Workspace: need     M [e] + 4*M [bdsqr work]
        lapack::bdsqr('U', m, izero, nru, izero, s, &work[ie], dummy, ione, A, lda, dummy, ione, &work[iwork], info);

        // If left singular vectors desired in U, copy them there
        if (want_uas)
          {
          lapack::lacpy('F', m, m, A, lda, U, ldu);
          }
        }
      else if (want_vo && want_un)
        {
        // Path 2t (N >> M, JOBU='N', JOBVT='O')
        // M right singular vectors to be overwritten on A and
        // no left singular vectors to be computed
        if (lwork >= m*m + wrkbl)
          {
          // Path 2t-fast: Sufficient workspace for a fast algorithm
          if (lwork >= std::max(wrkbl, lda*n + m) + lda*m)
            {
            // WORK(IU) is LDA by N
            // WORK(IR) is LDA by M
            ldwrku = lda;
            chunk  = n;
            ldwrkr = lda;
            }
          else if (lwork >= std::max(wrkbl, lda*n + m) + m*m)
            {
            // WORK(IU) is LDA by N
            // WORK(IR) is M by M
            ldwrku = lda;
            chunk  = n;
            ldwrkr = m;
            }
          else
            {
            // WORK(IU) is M by CHUNK
            // WORK(IR) is M by M
            ldwrku = m;
            chunk  = (lwork - m*m - m) / m;
            ldwrkr = m;
            }
          ir    = 1;
          itau  = ir + ldwrkr * m;
          iwork = itau + m;

          // Compute A=L*Q
          // Workspace: need   M*M [R] + M [tau] + M    [gelqf work]
          // Workspace: prefer M*M [R] + M [tau] + M*NB [gelqf work]
          lwork2 = lwork - iwork + 1;
          magma_sgelqf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );

          // Copy L to WORK(IR) and zero out above it
          lapack::lacpy('L', m, m, A, lda, &work[ir], ldwrkr);
          lapack::laset('U', m_1, m_1, c_zero, c_zero, &work[ir+ldwrkr], ldwrkr);

          // Generate Q in A
          // Workspace: need   M*M [R] + M [tau] + M    [orglq work]
          // Workspace: prefer M*M [R] + M [tau] + M*NB [orglq work]
          lwork2 = lwork - iwork + 1;
          magma_sorglq(      m,  n,  m, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
          ie    = itau;
          itauq = ie + m;
          itaup = itauq + m;
          iwork = itaup + m;

          // Bidiagonalize L in WORK(IR)
          // Workspace: need   M*M [R] + 3*M [e, tauq, taup] + M      [gebrd work]
          // Workspace: prefer M*M [R] + 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
          lwork2 = lwork - iwork + 1;
          magma_sgebrd(      m,  m, &work[ir],  ldwrkr, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );

          // Generate right vectors bidiagonalizing L
          // Workspace: need   M*M [R] + 3*M [e, tauq, taup] + M    [orgbr work]
          // Workspace: prefer M*M [R] + 3*M [e, tauq, taup] + M*NB [orgbr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgbr( MagmaP,   m,  m,  m, &work[ir],  ldwrkr, &work[itaup], &work[iwork],  lwork2, &ierr );
          iwork = ie + m;

          // Perform bidiagonal QR iteration, computing right
          // singular vectors of L in WORK(IR)
          // Workspace: need   M*M [R] + M [e] + 4*M [bdsqr work]
          lapack::bdsqr('U', m, m, izero, izero, s, &work[ie], &work[ir], ldwrkr, dummy, ione, dummy, ione, &work[iwork], info);
          iu = ie + m;

          // Multiply right singular vectors of L in WORK(IR) by Q
          // in A, storing result in WORK(IU) and copying to A
          // Workspace: need   M*M [R] + M [e] + M    [U]
          // Workspace: prefer M*M [R] + M [e] + M*NB [U]
          // Workspace: max    M*M [R] + M [e] + M*N  [U]
          // TODO: [e] not need; can change lwork above
          for (i = 1; i <= n; i += chunk)
            {
            ib = std::min( n - i + 1, chunk );
            blas::gemm('N', 'N', m, ib, m,
                       c_one,  &work[ir],       ldwrkr,
                               &A[(i-1) * lda], lda,
                       c_zero, &work[iu],       ldwrku);
            lapack::lacpy('F', m, ib, &work[iu], ldwrku, &A[(i-1) * lda], lda);
            }
          }
        else
          {
          // Path 2t-slow: Insufficient workspace for a fast algorithm
          ie    = 1;
          itauq = ie    + m;
          itaup = itauq + m;
          iwork = itaup + m;

          // Bidiagonalize A
          // Workspace: need   3*M [e, tauq, taup] + M        [gebrd work]
          // Workspace: prefer 3*M [e, tauq, taup] + (M+N)*NB [gebrd work]
          lwork2 = lwork - iwork + 1;
          magma_sgebrd(      m,  n, A,  lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );

          // Generate right vectors bidiagonalizing A
          // Workspace: need   3*M [e, tauq, taup] + M    [orgbr work]
          // Workspace: prefer 3*M [e, tauq, taup] + M*NB [orgbr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgbr( MagmaP,   m,  n,  m, A,  lda, &work[itaup], &work[iwork],  lwork2, &ierr );
          iwork = ie + m;

          // Perform bidiagonal QR iteration, computing right
          // singular vectors of A in A
          // Workspace: need     M [e] + 4*M [bdsqr work]
          lapack::bdsqr('L', m, n, izero, izero, s, &work[ie], A, lda, dummy, ione, dummy, ione, &work[iwork], info);
          }
        }
      else if (want_vo && want_uas)
        {
        // Path 3t (N >> M, JOBU='S' or 'A', JOBVT='O')
        // M right singular vectors to be overwritten on A and
        // M left singular vectors to be computed in U
        if (lwork >= m*m + wrkbl)
          {
          // Path 3t-fast: Sufficient workspace for a fast algorithm
          if (lwork >= std::max(wrkbl, lda*n + m) + lda*m)
            {
            // WORK(IU) is LDA by N
            // WORK(IR) is LDA by M
            ldwrku = lda;
            chunk  = n;
            ldwrkr = lda;
            }
          else if (lwork >= std::max(wrkbl, lda*n + m) + m*m)
            {
            // WORK(IU) is LDA by N
            // WORK(IR) is M by M
            ldwrku = lda;
            chunk  = n;
            ldwrkr = m;
            }
          else
            {
            // WORK(IU) is M by CHUNK
            // WORK(IR) is M by M
            ldwrku = m;
            chunk  = (lwork - m*m - m) / m;
            ldwrkr = m;
            }
          ir    = 1;
          itau  = ir + ldwrkr * m;
          iwork = itau + m;

          // Compute A=L*Q
          // Workspace: need   M*M [R] + M [tau] + M    [gelqf work]
          // Workspace: prefer M*M [R] + M [tau] + M*NB [gelqf work]
          lwork2 = lwork - iwork + 1;
          magma_sgelqf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );

          // Copy L to U, zeroing about above it
          lapack::lacpy('L', m, m, A, lda, U, ldu);
          lapack::laset('U', m_1, m_1, c_zero, c_zero, &U[1 * ldu], ldu);

          // Generate Q in A
          // Workspace: need   M*M [R] + M [tau] + M    [orglq work]
          // Workspace: prefer M*M [R] + M [tau] + M*NB [orglq work]
          lwork2 = lwork - iwork + 1;
          magma_sorglq(      m,  n,  m, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
          ie    = itau;
          itauq = ie + m;
          itaup = itauq + m;
          iwork = itaup + m;

          // Bidiagonalize L in U, copying result to WORK(IR)
          // Workspace: need   M*M [R] + 3*M [e, tauq, taup] + M      [gebrd work]
          // Workspace: prefer M*M [R] + 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
          lwork2 = lwork - iwork + 1;
          magma_sgebrd(      m,  m, U,  ldu, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
          lapack::lacpy('U', m, m, U, ldu, &work[ir], ldwrkr);

          // Generate right vectors bidiagonalizing L in WORK(IR)
          // Workspace: need   M*M [R] + 3*M [e, tauq, taup] + M    [orgbr work]
          // Workspace: prefer M*M [R] + 3*M [e, tauq, taup] + M*NB [orgbr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgbr( MagmaP,   m,  m,  m, &work[ir],  ldwrkr, &work[itaup], &work[iwork],  lwork2, &ierr );

          // Generate left vectors bidiagonalizing L in U
          // Workspace: need   M*M [R] + 3*M [e, tauq, taup] + M    [orgbr work]
          // Workspace: prefer M*M [R] + 3*M [e, tauq, taup] + M*NB [orgbr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgbr( MagmaQ,   m,  m,  m, U,  ldu, &work[itauq], &work[iwork],  lwork2, &ierr );
          iwork = ie + m;

          // Perform bidiagonal QR iteration, computing left
          // singular vectors of L in U, and computing right
          // singular vectors of L in WORK(IR)
          // Workspace: need   M*M [R] + M [e] + 4*M [bdsqr work]
          lapack::bdsqr('U', m, m, m, izero, s, &work[ie], &work[ir], ldwrkr, U, ldu, dummy, ione, &work[iwork], info);
          iu = ie + m;

          // Multiply right singular vectors of L in WORK(IR) by Q
          // in A, storing result in WORK(IU) and copying to A
          // Workspace: need   M*M [R] + M [e] + M    [U]
          // Workspace: prefer M*M [R] + M [e] + M*NB [U]
          // Workspace: max    M*M [R] + M [e] + M*N  [U]
          // TODO: [e] not need; can change lwork above
          for (i = 1; i <= n; i += chunk)
            {
            ib = std::min( n - i + 1, chunk );
            blas::gemm('N', 'N', m, ib, m,
                       c_one,  &work[ir],       ldwrkr,
                               &A[(i-1) * lda], lda,
                       c_zero, &work[iu],       ldwrku);
            lapack::lacpy('F', m, ib, &work[iu], ldwrku, &A[(i-1) * lda], lda);
            }
          }
        else
          {
          // Path 3t-slow: Insufficient workspace for a fast algorithm
          itau  = 1;
          iwork = itau + m;

          // Compute A=L*Q
          // Workspace: need   M [tau] + M    [gelqf work]
          // Workspace: prefer M [tau] + M*NB [gelqf work]
          lwork2 = lwork - iwork + 1;
          magma_sgelqf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );

          // Copy L to U, zeroing out above it
          lapack::lacpy('L', m, m, A, lda, U, ldu);
          lapack::laset('U', m_1, m_1, c_zero, c_zero, &U[1 * ldu], ldu);

          // Generate Q in A
          // Workspace: need   M [tau] + M    [orglq work]
          // Workspace: prefer M [tau] + M*NB [orglq work]
          lwork2 = lwork - iwork + 1;
          magma_sorglq(      m,  n,  m, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
          ie    = itau;
          itauq = ie + m;
          itaup = itauq + m;
          iwork = itaup + m;

          // Bidiagonalize L in U
          // Workspace: need   3*M [e, tauq, taup] + M      [gebrd work]
          // Workspace: prefer 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
          lwork2 = lwork - iwork + 1;
          magma_sgebrd(      m,  m, U,  ldu, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );

          // Multiply right vectors bidiagonalizing L by Q in A
          // Workspace: need   3*M [e, tauq, taup] + N    [ormbr work]
          // Workspace: prefer 3*M [e, tauq, taup] + N*NB [ormbr work]
          lwork2 = lwork - iwork + 1;
          magma_sormbr( MagmaP, MagmaLeft, MagmaTrans,  m,  n,  m, U,  ldu, &work[itaup], A,  lda, &work[iwork],  lwork2, &ierr );

          // Generate left vectors bidiagonalizing L in U
          // Workspace: need   3*M [e, tauq, taup] + M    [orgbr work]
          // Workspace: prefer 3*M [e, tauq, taup] + M*NB [orgbr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgbr( MagmaQ,   m,  m,  m, U,  ldu, &work[itauq], &work[iwork],  lwork2, &ierr );
          iwork = ie + m;

          // Perform bidiagonal QR iteration, computing left
          // singular vectors of A in U and computing right
          // singular vectors of A in A
          // Workspace: need     M [e] + 4*M [bdsqr work]
          lapack::bdsqr('U', m, n, m, izero, s, &work[ie], A, lda, U, ldu, dummy, ione, &work[iwork], info);
          }
        }
      else if (want_vs && want_un)
        {
        // Path 4t (N >> M, JOBU='N', JOBVT='S')
        // M right singular vectors to be computed in VT and
        // no left singular vectors to be computed
        if (lwork >= m*m + wrkbl)
          {
          // Path 4t-fast: Sufficient workspace for a fast algorithm
          if (lwork >= wrkbl + lda*m)
            {
            // WORK(IR) is LDA by M
            ldwrkr = lda;
            }
          else
            {
            // WORK(IR) is M by M
            ldwrkr = m;
            }
          ir    = 1;
          itau  = ir + ldwrkr * m;
          iwork = itau + m;

          // Compute A=L*Q
          // Workspace: need   M*M [R] + M [tau] + M    [gelqf work]
          // Workspace: prefer M*M [R] + M [tau] + M*NB [gelqf work]
          lwork2 = lwork - iwork + 1;
          magma_sgelqf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );

          // Copy L to WORK(IR), zeroing out above it
          lapack::lacpy('L', m, m, A, lda, &work[ir], ldwrkr);
          lapack::laset('U', m_1, m_1, c_zero, c_zero, &work[ir+ldwrkr], ldwrkr);

          // Generate Q in A
          // Workspace: need   M*M [R] + M [tau] + M    [orglq work]
          // Workspace: prefer M*M [R] + M [tau] + M*NB [orglq work]
          lwork2 = lwork - iwork + 1;
          magma_sorglq(      m,  n,  m, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
          ie    = itau;
          itauq = ie + m;
          itaup = itauq + m;
          iwork = itaup + m;

          // Bidiagonalize L in WORK(IR)
          // Workspace: need   M*M [R] + 3*M [e, tauq, taup] + M      [gebrd work]
          // Workspace: prefer M*M [R] + 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
          lwork2 = lwork - iwork + 1;
          magma_sgebrd(      m,  m, &work[ir],  ldwrkr, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );

          // Generate right vectors bidiagonalizing L in
          // WORK(IR)
          // Workspace: need   M*M [R] + 3*M [e, tauq, taup] + M    [orgbr work]
          // Workspace: prefer M*M [R] + 3*M [e, tauq, taup] + M*NB [orgbr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgbr( MagmaP,   m,  m,  m, &work[ir],  ldwrkr, &work[itaup], &work[iwork],  lwork2, &ierr );
          iwork = ie + m;

          // Perform bidiagonal QR iteration, computing right
          // singular vectors of L in WORK(IR)
          // Workspace: need   M*M [R] + M [e] + 4*M [bdsqr work]
          lapack::bdsqr('U', m, m, izero, izero, s, &work[ie], &work[ir], ldwrkr, dummy, ione, dummy, ione, &work[iwork], info);

          // Multiply right singular vectors of L in WORK(IR) by
          // Q in A, storing result in VT
          // Workspace: need   M*M [R]
          blas::gemm('N', 'N', m, n, m,
                     c_one,  &work[ir], ldwrkr,
                             A,         lda,
                     c_zero, VT,        ldvt);
          }
        else
          {
          // Path 4t-slow: Insufficient workspace for a fast algorithm
          itau  = 1;
          iwork = itau + m;

          // Compute A=L*Q
          // Workspace: need   M [tau] + M    [gelqf work]
          // Workspace: prefer M [tau] + M*NB [gelqf work]
          lwork2 = lwork - iwork + 1;
          magma_sgelqf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );

          // Copy result to VT
          lapack::lacpy('U', m, n, A, lda, VT, ldvt);

          // Generate Q in VT
          // Workspace: need   M [tau] + M    [orglq work]
          // Workspace: prefer M [tau] + M*NB [orglq work]
          lwork2 = lwork - iwork + 1;
          magma_sorglq(      m,  n,  m, VT,  ldvt, &work[itau], &work[iwork],  lwork2, &ierr );
          ie    = itau;
          itauq = ie + m;
          itaup = itauq + m;
          iwork = itaup + m;

          // Zero out above L in A
          lapack::laset('U', m_1, m_1, c_zero, c_zero, &A[1 * lda], lda);

          // Bidiagonalize L in A
          // Workspace: need   3*M [e, tauq, taup] + M      [gebrd work]
          // Workspace: prefer 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
          lwork2 = lwork - iwork + 1;
          magma_sgebrd(      m,  m, A,  lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );

          // Multiply right vectors bidiagonalizing L by Q in VT
          // Workspace: need   3*M [e, tauq, taup] + N    [ormbr work]
          // Workspace: prefer 3*M [e, tauq, taup] + N*NB [ormbr work]
          lwork2 = lwork - iwork + 1;
          magma_sormbr( MagmaP, MagmaLeft, MagmaTrans,  m,  n,  m, A,   lda, &work[itaup], VT,  ldvt, &work[iwork],  lwork2, &ierr );
          iwork = ie + m;

          // Perform bidiagonal QR iteration, computing right
          // singular vectors of A in VT
          // Workspace: need     M [e] + 4*M [bdsqr work]
          lapack::bdsqr('U', m, n, izero, izero, s, &work[ie], VT, ldvt, dummy, ione, dummy, ione, &work[iwork], info);
          }
        }
      else if (want_vs && want_uo)
        {
        // Path 5t (N >> M, JOBU='O', JOBVT='S')
        // M right singular vectors to be computed in VT and
        // M left singular vectors to be overwritten on A
        if (lwork >= 2*m*m + wrkbl)
          {
          // Path 5t-fast: Sufficient workspace for a fast algorithm
          if (lwork >= wrkbl + 2*lda*m)
            {
            // WORK(IU) is LDA by M
            // WORK(IR) is LDA by M
            ldwrku = lda;
            ldwrkr = lda;
            }
          else if (lwork >= wrkbl + (lda + m) * m)
            {
            // WORK(IU) is LDA by M
            // WORK(IR) is M by M
            ldwrku = lda;
            ldwrkr = m;
            }
          else
            {
            // WORK(IU) is M by M
            // WORK(IR) is M by M
            ldwrku = m;
            ldwrkr = m;
            }
          iu    = 1;
          ir    = iu + ldwrku * m;
          itau  = ir + ldwrkr * m;
          iwork = itau + m;

          // Compute A=L*Q
          // Workspace: need   2*M*M [U,R] + M [tau] + M    [gelqf work]
          // Workspace: prefer 2*M*M [U,R] + M [tau] + M*NB [gelqf work]
          lwork2 = lwork - iwork + 1;
          magma_sgelqf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );

          // Copy L to WORK(IU), zeroing out below it
          lapack::lacpy('L', m, m, A, lda, &work[iu], ldwrku);
          lapack::laset('U', m_1, m_1, c_zero, c_zero, &work[iu+ldwrku], ldwrku);

          // Generate Q in A
          // Workspace: need   2*M*M [U,R] + M [tau] + M    [orglq work]
          // Workspace: prefer 2*M*M [U,R] + M [tau] + M*NB [orglq work]
          lwork2 = lwork - iwork + 1;
          magma_sorglq(      m,  n,  m, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
          ie    = itau;
          itauq = ie + m;
          itaup = itauq + m;
          iwork = itaup + m;

          // Bidiagonalize L in WORK(IU), copying result to
          // WORK(IR)
          // Workspace: need   2*M*M [U,R] + 3*M [e, tauq, taup] + M      [gebrd work]
          // Workspace: prefer 2*M*M [U,R] + 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
          lwork2 = lwork - iwork + 1;
          magma_sgebrd(      m,  m, &work[iu],  ldwrku, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
          lapack::lacpy('L', m, m, &work[iu], ldwrku, &work[ir], ldwrkr);

          // Generate right bidiagonalizing vectors in WORK(IU)
          // Workspace: need   2*M*M [U,R] + 3*M [e, tauq, taup] + M    [orgbr work]
          // Workspace: prefer 2*M*M [U,R] + 3*M [e, tauq, taup] + M*NB [orgbr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgbr( MagmaP,   m,  m,  m, &work[iu],  ldwrku, &work[itaup], &work[iwork],  lwork2, &ierr );

          // Generate left bidiagonalizing vectors in WORK(IR)
          // Workspace: need   2*M*M [U,R] + 3*M [e, tauq, taup] + M    [orgbr work]
          // Workspace: prefer 2*M*M [U,R] + 3*M [e, tauq, taup] + M*NB [orgbr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgbr( MagmaQ,   m,  m,  m, &work[ir],  ldwrkr, &work[itauq], &work[iwork],  lwork2, &ierr );
          iwork = ie + m;

          // Perform bidiagonal QR iteration, computing left
          // singular vectors of L in WORK(IR) and computing
          // right singular vectors of L in WORK(IU)
          // Workspace: need   2*M*M [U,R] + M [e] + 4*M [bdsqr work]
          lapack::bdsqr('U', m, m, m, izero, s, &work[ie], &work[iu], ldwrku, &work[ir], ldwrkr, dummy, ione, &work[iwork], info);

          // Multiply right singular vectors of L in WORK(IU) by
          // Q in A, storing result in VT
          // Workspace: need   2*M*M [U,R]
          blas::gemm('N', 'N', m, n, m,
                     c_one,  &work[iu], ldwrku,
                             A,         lda,
                     c_zero, VT,        ldvt);

          // Copy left singular vectors of L to A
          // Workspace: need   2*M*M [U,R]
          lapack::lacpy('F', m, m, &work[ir], ldwrkr, A, lda);
          }
        else
          {
          // Path 5t-slow: Insufficient workspace for a fast algorithm
          itau  = 1;
          iwork = itau + m;

          // Compute A=L*Q, copying result to VT
          // Workspace: need   M [tau] + M    [gelqf work]
          // Workspace: prefer M [tau] + M*NB [gelqf work]
          lwork2 = lwork - iwork + 1;
          magma_sgelqf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
          lapack::lacpy('U', m, n, A, lda, VT, ldvt);

          // Generate Q in VT
          // Workspace: need   M [tau] + M    [orglq work]
          // Workspace: prefer M [tau] + M*NB [orglq work]
          lwork2 = lwork - iwork + 1;
          magma_sorglq(      m,  n,  m, VT,  ldvt, &work[itau], &work[iwork],  lwork2, &ierr );
          ie    = itau;
          itauq = ie + m;
          itaup = itauq + m;
          iwork = itaup + m;

          // Zero out above L in A
          lapack::laset('U', m_1, m_1, c_zero, c_zero, &A[1 * lda], lda);

          // Bidiagonalize L in A
          // Workspace: need   3*M [e, tauq, taup] + M      [gebrd work]
          // Workspace: prefer 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
          lwork2 = lwork - iwork + 1;
          magma_sgebrd(      m,  m, A,  lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );

          // Multiply right vectors bidiagonalizing L by Q in VT
          // Workspace: need   3*M [e, tauq, taup] + N    [ormbr work]
          // Workspace: prefer 3*M [e, tauq, taup] + N*NB [ormbr work]
          lwork2 = lwork - iwork + 1;
          magma_sormbr( MagmaP, MagmaLeft, MagmaTrans,  m,  n,  m, A,   lda, &work[itaup], VT,  ldvt, &work[iwork],  lwork2, &ierr );

          // Generate left bidiagonalizing vectors of L in A
          // Workspace: need   3*M [e, tauq, taup] + M    [orgbr work]
          // Workspace: prefer 3*M [e, tauq, taup] + M*NB [orgbr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgbr( MagmaQ,   m,  m,  m, A,  lda, &work[itauq], &work[iwork],  lwork2, &ierr );
          iwork = ie + m;

          // Perform bidiagonal QR iteration, computing left
          // singular vectors of A in A and computing right
          // singular vectors of A in VT
          // Workspace: need     M [e] + 4*M [bdsqr work]
          lapack::bdsqr('U', m, n, m, izero, s, &work[ie], VT, ldvt, A, lda, dummy, ione, &work[iwork], info);
          }
        }
      else if (want_vs && want_uas)
        {
        // Path 6t (N >> M, JOBU='S' or 'A', JOBVT='S')
        // M right singular vectors to be computed in VT and
        // M left singular vectors to be computed in U
        if (lwork >= m*m + wrkbl)
          {
          // Path 6t-fast: Sufficient workspace for a fast algorithm
          if (lwork >= wrkbl + lda*m)
            {
            // WORK(IU) is LDA by N
            ldwrku = lda;
            }
          else
            {
            // WORK(IU) is LDA by M
            ldwrku = m;
            }
          iu    = 1;
          itau  = iu + ldwrku * m;
          iwork = itau + m;

          // Compute A=L*Q
          // Workspace: need   M*M [U] + M [tau] + M    [gelqf work]
          // Workspace: prefer M*M [U] + M [tau] + M*NB [gelqf work]
          lwork2 = lwork - iwork + 1;
          magma_sgelqf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );

          // Copy L to WORK(IU), zeroing out above it
          lapack::lacpy('L', m, m, A, lda, &work[iu], ldwrku);
          lapack::laset('U', m_1, m_1, c_zero, c_zero, &work[iu+ldwrku], ldwrku);

          // Generate Q in A
          // Workspace: need   M*M [U] + M [tau] + M    [orglq work]
          // Workspace: prefer M*M [U] + M [tau] + M*NB [orglq work]
          lwork2 = lwork - iwork + 1;
          magma_sorglq(      m,  n,  m, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
          ie    = itau;
          itauq = ie + m;
          itaup = itauq + m;
          iwork = itaup + m;

          // Bidiagonalize L in WORK(IU), copying result to U
          // Workspace: need   M*M [U] + 3*M [e, tauq, taup] + M      [gebrd work]
          // Workspace: prefer M*M [U] + 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
          lwork2 = lwork - iwork + 1;
          magma_sgebrd(      m,  m, &work[iu],  ldwrku, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
          lapack::lacpy('L', m, m, &work[iu], ldwrku, U, ldu);

          // Generate right bidiagonalizing vectors in WORK(IU)
          // Workspace: need   M*M [U] + 3*M [e, tauq, taup] + M    [orgbr work]
          // Workspace: prefer M*M [U] + 3*M [e, tauq, taup] + M*NB [orgbr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgbr( MagmaP,   m,  m,  m, &work[iu],  ldwrku, &work[itaup], &work[iwork],  lwork2, &ierr );

          // Generate left bidiagonalizing vectors in U
          // Workspace: need   M*M [U] + 3*M [e, tauq, taup] + M    [orgbr work]
          // Workspace: prefer M*M [U] + 3*M [e, tauq, taup] + M*NB [orgbr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgbr( MagmaQ,   m,  m,  m, U,  ldu, &work[itauq], &work[iwork],  lwork2, &ierr );
          iwork = ie + m;

          // Perform bidiagonal QR iteration, computing left
          // singular vectors of L in U and computing right
          // singular vectors of L in WORK(IU)
          // Workspace: need   M*M [U] + M [e] + 4*M [bdsqr work]
          lapack::bdsqr('U', m, m, m, izero, s, &work[ie], &work[iu], ldwrku, U, ldu, dummy, ione, &work[iwork], info);

          // Multiply right singular vectors of L in WORK(IU) by
          // Q in A, storing result in VT
          // Workspace: need   M*M [U]
          blas::gemm('N', 'N', m, n, m,
                     c_one,  &work[iu], ldwrku,
                             A,         lda,
                     c_zero, VT,        ldvt);
          }
        else
          {
          // Path 6t-slow: Insufficient workspace for a fast algorithm
          itau  = 1;
          iwork = itau + m;

          // Compute A=L*Q, copying result to VT
          // Workspace: need   M [tau] + M    [gelqf work]
          // Workspace: prefer M [tau] + M*NB [gelqf work]
          lwork2 = lwork - iwork + 1;
          magma_sgelqf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
          lapack::lacpy('U', m, n, A, lda, VT, ldvt);

          // Generate Q in VT
          // Workspace: need   M [tau] + M    [orglq work]
          // Workspace: prefer M [tau] + M*NB [orglq work]
          lwork2 = lwork - iwork + 1;
          magma_sorglq(      m,  n,  m, VT,  ldvt, &work[itau], &work[iwork],  lwork2, &ierr );

          // Copy L to U, zeroing out above it
          lapack::lacpy('L', m, m, A, lda, U, ldu);
          lapack::laset('U', m_1, m_1, c_zero, c_zero, &U[1 * ldu], ldu);
          ie    = itau;
          itauq = ie + m;
          itaup = itauq + m;
          iwork = itaup + m;

          // Bidiagonalize L in U
          // Workspace: need   3*M [e, tauq, taup] + M      [gebrd work]
          // Workspace: prefer 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
          lwork2 = lwork - iwork + 1;
          magma_sgebrd(      m,  m, U,  ldu, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );

          // Multiply right bidiagonalizing vectors in U by Q in VT
          // Workspace: need   3*M [e, tauq, taup] + N    [ormbr work]
          // Workspace: prefer 3*M [e, tauq, taup] + N*NB [ormbr work]
          lwork2 = lwork - iwork + 1;
          magma_sormbr( MagmaP, MagmaLeft, MagmaTrans,  m,  n,  m, U,  ldu, &work[itaup], VT,  ldvt, &work[iwork],  lwork2, &ierr );

          // Generate left bidiagonalizing vectors in U
          // Workspace: need   3*M [e, tauq, taup] + M    [orgbr work]
          // Workspace: prefer 3*M [e, tauq, taup] + M*NB [orgbr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgbr( MagmaQ,   m,  m,  m, U,  ldu, &work[itauq], &work[iwork],  lwork2, &ierr );
          iwork = ie + m;

          // Perform bidiagonal QR iteration, computing left
          // singular vectors of A in U and computing right
          // singular vectors of A in VT
          // Workspace: need     M [e] + 4*M [bdsqr work]
          lapack::bdsqr('U', m, n, m, izero, s, &work[ie], VT, ldvt, U, ldu, dummy, ione, &work[iwork], info);
          }
        }
      else if (want_va && want_un)
        {
        // Path 7t (N >> M, JOBU='N', JOBVT='A')
        // N right singular vectors to be computed in VT and
        // no left singular vectors to be computed
        if (lwork >= m*m + wrkbl)
          {
          // Path 7t-fast: Sufficient workspace for a fast algorithm
          if (lwork >= wrkbl + lda*m)
            {
            // WORK(IR) is LDA by M
            ldwrkr = lda;
            }
          else
            {
            // WORK(IR) is M by M
            ldwrkr = m;
            }
          ir    = 1;
          itau  = ir + ldwrkr * m;
          iwork = itau + m;

          // Compute A=L*Q, copying result to VT
          // Workspace: need   M*M [R] + M [tau] + M    [gelqf work]
          // Workspace: prefer M*M [R] + M [tau] + M*NB [gelqf work]
          lwork2 = lwork - iwork + 1;
          magma_sgelqf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
          lapack::lacpy('U', m, n, A, lda, VT, ldvt);

          // Copy L to WORK(IR), zeroing out above it
          lapack::lacpy('L', m, m, A, lda, &work[ir], ldwrkr);
          lapack::laset('U', m_1, m_1, c_zero, c_zero, &work[ir+ldwrkr], ldwrkr);

          // Generate Q in VT
          // Workspace: need   M*M [R] + M [tau] + N    [orglq work]
          // Workspace: prefer M*M [R] + M [tau] + N*NB [orglq work]
          lwork2 = lwork - iwork + 1;
          magma_sorglq(      n,  n,  m, VT,  ldvt, &work[itau], &work[iwork],  lwork2, &ierr );
          ie    = itau;
          itauq = ie + m;
          itaup = itauq + m;
          iwork = itaup + m;

          // Bidiagonalize L in WORK(IR)
          // Workspace: need   M*M [R] + 3*M [e, tauq, taup] + M      [gebrd work]
          // Workspace: prefer M*M [R] + 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
          lwork2 = lwork - iwork + 1;
          magma_sgebrd(      m,  m, &work[ir],  ldwrkr, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );

          // Generate right bidiagonalizing vectors in WORK(IR)
          // Workspace: need   M*M [R] + 3*M [e, tauq, taup] + M    [orgbr work]
          // Workspace: prefer M*M [R] + 3*M [e, tauq, taup] + M*NB [orgbr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgbr( MagmaP,   m,  m,  m, &work[ir],  ldwrkr, &work[itaup], &work[iwork],  lwork2, &ierr );
          iwork = ie + m;

          // Perform bidiagonal QR iteration, computing right
          // singular vectors of L in WORK(IR)
          // Workspace: need   M*M [R] + M [e] + 4*M [bdsqr work]
          lapack::bdsqr('U', m, m, izero, izero, s, &work[ie], &work[ir], ldwrkr, dummy, ione, dummy, ione, &work[iwork], info);

          // Multiply right singular vectors of L in WORK(IR) by
          // Q in VT, storing result in A
          // Workspace: need   M*M [R]
          blas::gemm('N', 'N', m, n, m,
                     c_one,  &work[ir], ldwrkr,
                             VT,        ldvt,
                     c_zero, A,         lda);

          // Copy right singular vectors of A from A to VT
          lapack::lacpy('F', m, n, A, lda, VT, ldvt);
          }
        else
          {
          // Path 7t-slow: Insufficient workspace for a fast algorithm
          itau  = 1;
          iwork = itau + m;

          // Compute A=L*Q, copying result to VT
          // Workspace: need   M [tau] + M    [gelqf work]
          // Workspace: prefer M [tau] + M*NB [gelqf work]
          lwork2 = lwork - iwork + 1;
          magma_sgelqf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
          lapack::lacpy('U', m, n, A, lda, VT, ldvt);

          // Generate Q in VT
          // Workspace: need   M [tau] + N    [orglq work]
          // Workspace: prefer M [tau] + N*NB [orglq work]
          lwork2 = lwork - iwork + 1;
          magma_sorglq(      n,  n,  m, VT,  ldvt, &work[itau], &work[iwork],  lwork2, &ierr );
          ie    = itau;
          itauq = ie + m;
          itaup = itauq + m;
          iwork = itaup + m;

          // Zero out above L in A
          lapack::laset('U', m_1, m_1, c_zero, c_zero, &A[1 * lda], lda);

          // Bidiagonalize L in A
          // Workspace: need   3*M [e, tauq, taup] + M      [gebrd work]
          // Workspace: prefer 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
          lwork2 = lwork - iwork + 1;
          magma_sgebrd(      m,  m, A,  lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );

          // Multiply right bidiagonalizing vectors in A by Q in VT
          // Workspace: need   3*M [e, tauq, taup] + N    [ormbr work]
          // Workspace: prefer 3*M [e, tauq, taup] + N*NB [ormbr work]
          lwork2 = lwork - iwork + 1;
          magma_sormbr( MagmaP, MagmaLeft, MagmaTrans,  m,  n,  m, A,   lda, &work[itaup], VT,  ldvt, &work[iwork],  lwork2, &ierr );
          iwork = ie + m;

          // Perform bidiagonal QR iteration, computing right
          // singular vectors of A in VT
          // Workspace: need     M [e] + 4*M [bdsqr work]
          lapack::bdsqr('U', m, n, izero, izero, s, &work[ie], VT, ldvt, dummy, ione, dummy, ione, &work[iwork], info);
          }
        }
      else if (want_va && want_uo)
        {
        // Path 8t (N >> M, JOBU='O', JOBVT='A')
        // N right singular vectors to be computed in VT and
        // M left singular vectors to be overwritten on A
        if (lwork >= 2*m*m + wrkbl)
          {
          // Path 8t-fast: Sufficient workspace for a fast algorithm
          if (lwork >= wrkbl + 2*lda*m)
            {
            // WORK(IU) is LDA by M
            // WORK(IR) is LDA by M
            ldwrku = lda;
            ldwrkr = lda;
            }
          else if (lwork >= wrkbl + (lda + m) * m)
            {
            // WORK(IU) is LDA by M
            // WORK(IR) is M by M
            ldwrku = lda;
            ldwrkr = m;
            }
          else
            {
            // WORK(IU) is M by M
            // WORK(IR) is M by M
            ldwrku = m;
            ldwrkr = m;
            }
          iu    = 1;
          ir    = iu + ldwrku * m;
          itau  = ir + ldwrkr * m;
          iwork = itau + m;

          // Compute A=L*Q, copying result to VT
          // Workspace: need   2*M*M [U,R] + M [tau] + M    [gelqf work]
          // Workspace: prefer 2*M*M [U,R] + M [tau] + M*NB [gelqf work]
          lwork2 = lwork - iwork + 1;
          magma_sgelqf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
          lapack::lacpy('U', m, n, A, lda, VT, ldvt);

          // Generate Q in VT
          // Workspace: need   2*M*M [U,R] + M [tau] + N    [orglq work]
          // Workspace: prefer 2*M*M [U,R] + M [tau] + N*NB [orglq work]
          lwork2 = lwork - iwork + 1;
          magma_sorglq(      n,  n,  m, VT,  ldvt, &work[itau], &work[iwork],  lwork2, &ierr );

          // Copy L to WORK(IU), zeroing out above it
          lapack::lacpy('L', m, m, A, lda, &work[iu], ldwrku);
          lapack::laset('U', m_1, m_1, c_zero, c_zero, &work[iu+ldwrku], ldwrku);
          ie    = itau;
          itauq = ie + m;
          itaup = itauq + m;
          iwork = itaup + m;

          // Bidiagonalize L in WORK(IU), copying result to WORK(IR)
          // Workspace: need   2*M*M [U,R] + 3*M [e, tauq, taup] + M      [gebrd work]
          // Workspace: prefer 2*M*M [U,R] + 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
          lwork2 = lwork - iwork + 1;
          magma_sgebrd(      m,  m, &work[iu],  ldwrku, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
          lapack::lacpy('L', m, m, &work[iu], ldwrku, &work[ir], ldwrkr);

          // Generate right bidiagonalizing vectors in WORK(IU)
          // Workspace: need   2*M*M [U,R] + 3*M [e, tauq, taup] + M    [orgbr work]
          // Workspace: prefer 2*M*M [U,R] + 3*M [e, tauq, taup] + M*NB [orgbr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgbr( MagmaP,   m,  m,  m, &work[iu],  ldwrku, &work[itaup], &work[iwork],  lwork2, &ierr );

          // Generate left bidiagonalizing vectors in WORK(IR)
          // Workspace: need   2*M*M [U,R] + 3*M [e, tauq, taup] + M    [orgbr work]
          // Workspace: prefer 2*M*M [U,R] + 3*M [e, tauq, taup] + M*NB [orgbr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgbr( MagmaQ,   m,  m,  m, &work[ir],  ldwrkr, &work[itauq], &work[iwork],  lwork2, &ierr );
          iwork = ie + m;

          // Perform bidiagonal QR iteration, computing left
          // singular vectors of L in WORK(IR) and computing
          // right singular vectors of L in WORK(IU)
          // Workspace: need   2*M*M [U,R] + M [e] + 4*M [bdsqr work]
          lapack::bdsqr('U', m, m, m, izero, s, &work[ie], &work[iu], ldwrku, &work[ir], ldwrkr, dummy, ione, &work[iwork], info);

          // Multiply right singular vectors of L in WORK(IU) by
          // Q in VT, storing result in A
          // Workspace: need   M*M [U]
          blas::gemm('N', 'N', m, n, m,
                     c_one,  &work[iu], ldwrku,
                             VT,        ldvt,
                     c_zero, A,         lda);

          // Copy right singular vectors of A from A to VT
          lapack::lacpy('F', m, n, A, lda, VT, ldvt);

          // Copy left singular vectors of A from WORK(IR) to A
          lapack::lacpy('F', m, m, &work[ir], ldwrkr, A, lda);
          }
        else
          {
          // Path 8t-slow: Insufficient workspace for a fast algorithm
          itau  = 1;
          iwork = itau + m;

          // Compute A=L*Q, copying result to VT
          // Workspace: need   M [tau] + M    [gelqf work]
          // Workspace: prefer M [tau] + M*NB [gelqf work]
          lwork2 = lwork - iwork + 1;
          magma_sgelqf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
          lapack::lacpy('U', m, n, A, lda, VT, ldvt);

          // Generate Q in VT
          // Workspace: need   M [tau] + N    [orglq work]
          // Workspace: prefer M [tau] + N*NB [orglq work]
          lwork2 = lwork - iwork + 1;
          magma_sorglq(      n,  n,  m, VT,  ldvt, &work[itau], &work[iwork],  lwork2, &ierr );
          ie    = itau;
          itauq = ie + m;
          itaup = itauq + m;
          iwork = itaup + m;

          // Zero out above L in A
          lapack::laset('U', m_1, m_1, c_zero, c_zero, &A[1 * lda], lda);

          // Bidiagonalize L in A
          // Workspace: need   3*M [e, tauq, taup] + M      [gebrd work]
          // Workspace: prefer 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
          lwork2 = lwork - iwork + 1;
          magma_sgebrd(      m,  m, A,  lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );

          // Multiply right bidiagonalizing vectors in A by Q in VT
          // Workspace: need   3*M [e, tauq, taup] + N    [ormbr work]
          // Workspace: prefer 3*M [e, tauq, taup] + N*NB [ormbr work]
          lwork2 = lwork - iwork + 1;
          magma_sormbr( MagmaP, MagmaLeft, MagmaTrans,  m,  n,  m, A,   lda, &work[itaup], VT,  ldvt, &work[iwork],  lwork2, &ierr );

          // Generate left bidiagonalizing vectors in A
          // Workspace: need   3*M [e, tauq, taup] + M    [orgbr work]
          // Workspace: prefer 3*M [e, tauq, taup] + M*NB [orgbr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgbr( MagmaQ,   m,  m,  m, A,  lda, &work[itauq], &work[iwork],  lwork2, &ierr );
          iwork = ie + m;

          // Perform bidiagonal QR iteration, computing left
          // singular vectors of A in A and computing right
          // singular vectors of A in VT
          // Workspace: need     M [e] + 4*M [bdsqr work]
          lapack::bdsqr('U', m, n, m, izero, s, &work[ie], VT, ldvt, A, lda, dummy, ione, &work[iwork], info);
          }
        }
      else if (want_va && want_uas)
        {
        // Path 9t (N >> M, JOBU='S' or 'A', JOBVT='A')
        // N right singular vectors to be computed in VT and
        // M left singular vectors to be computed in U
        if (lwork >= m*m + wrkbl)
          {
          // Path 9t-fast: Sufficient workspace for a fast algorithm
          if (lwork >= wrkbl + lda*m)
            {
            // WORK(IU) is LDA by M
            ldwrku = lda;
            }
          else
            {
            // WORK(IU) is M by M
            ldwrku = m;
            }
          iu    = 1;
          itau  = iu + ldwrku * m;
          iwork = itau + m;

          // Compute A=L*Q, copying result to VT
          // Workspace: need   M*M [U] + M [tau] + M    [gelqf work]
          // Workspace: prefer M*M [U] + M [tau] + M*NB [gelqf work]
          lwork2 = lwork - iwork + 1;
          magma_sgelqf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
          lapack::lacpy('U', m, n, A, lda, VT, ldvt);

          // Generate Q in VT
          // Workspace: need   M*M [U] + M [tau] + N    [orglq work]
          // Workspace: prefer M*M [U] + M [tau] + N*NB [orglq work]
          lwork2 = lwork - iwork + 1;
          magma_sorglq(      n,  n,  m, VT,  ldvt, &work[itau], &work[iwork],  lwork2, &ierr );

          // Copy L to WORK(IU), zeroing out above it
          lapack::lacpy('L', m, m, A, lda, &work[iu], ldwrku);
          lapack::laset('U', m_1, m_1, c_zero, c_zero, &work[iu+ldwrku], ldwrku);
          ie    = itau;
          itauq = ie + m;
          itaup = itauq + m;
          iwork = itaup + m;

          // Bidiagonalize L in WORK(IU), copying result to U
          // Workspace: need   M*M [U] + 3*M [e, tauq, taup] + M      [gebrd work]
          // Workspace: prefer M*M [U] + 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
          lwork2 = lwork - iwork + 1;
          magma_sgebrd(      m,  m, &work[iu],  ldwrku, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );
          lapack::lacpy('L', m, m, &work[iu], ldwrku, U, ldu);

          // Generate right bidiagonalizing vectors in WORK(IU)
          // Workspace: need   M*M [U] + 3*M [e, tauq, taup] + M    [orgbr work]
          // Workspace: prefer M*M [U] + 3*M [e, tauq, taup] + M*NB [orgbr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgbr( MagmaP,   m,  m,  m, &work[iu],  ldwrku, &work[itaup], &work[iwork],  lwork2, &ierr );

          // Generate left bidiagonalizing vectors in U
          // Workspace: need   M*M [U] + 3*M [e, tauq, taup] + M    [orgbr work]
          // Workspace: prefer M*M [U] + 3*M [e, tauq, taup] + M*NB [orgbr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgbr( MagmaQ,   m,  m,  m, U,  ldu, &work[itauq], &work[iwork],  lwork2, &ierr );
          iwork = ie + m;

          // Perform bidiagonal QR iteration, computing left
          // singular vectors of L in U and computing right
          // singular vectors of L in WORK(IU)
          // Workspace: need   M*M [U] + M [e] + 4*M [bdsqr work]
          lapack::bdsqr('U', m, m, m, izero, s, &work[ie], &work[iu], ldwrku, U, ldu, dummy, ione, &work[iwork], info);

          // Multiply right singular vectors of L in WORK(IU) by
          // Q in VT, storing result in A
          // Workspace: need   M*M [U]
          blas::gemm('N', 'N', m, n, m,
                     c_one,  &work[iu], ldwrku,
                             VT,        ldvt,
                     c_zero, A,         lda);

          // Copy right singular vectors of A from A to VT
          lapack::lacpy('F', m, n, A, lda, VT, ldvt);
          }
        else
          {
          // Path 9t-slow: Insufficient workspace for a fast algorithm
          itau  = 1;
          iwork = itau + m;

          // Compute A=L*Q, copying result to VT
          // Workspace: need   M [tau] + M    [gelqf work]
          // Workspace: prefer M [tau] + M*NB [gelqf work]
          lwork2 = lwork - iwork + 1;
          magma_sgelqf(      m,  n, A,  lda, &work[itau], &work[iwork],  lwork2, &ierr );
          lapack::lacpy('U', m, n, A, lda, VT, ldvt);

          // Generate Q in VT
          // Workspace: need   M [tau] + N    [orglq work]
          // Workspace: prefer M [tau] + N*NB [orglq work]
          lwork2 = lwork - iwork + 1;
          magma_sorglq(      n,  n,  m, VT,  ldvt, &work[itau], &work[iwork],  lwork2, &ierr );

          // Copy L to U, zeroing out above it
          lapack::lacpy('L', m, m, A, lda, U, ldu);
          lapack::laset('U', m_1, m_1, c_zero, c_zero, &U[1 * ldu], ldu);
          ie    = itau;
          itauq = ie + m;
          itaup = itauq + m;
          iwork = itaup + m;

          // Bidiagonalize L in U
          // Workspace: need   3*M [e, tauq, taup] + M      [gebrd work]
          // Workspace: prefer 3*M [e, tauq, taup] + 2*M*NB [gebrd work]
          lwork2 = lwork - iwork + 1;
          magma_sgebrd(      m,  m, U,  ldu, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );

          // Multiply right bidiagonalizing vectors in U by Q in VT
          // Workspace: need   3*M [e, tauq, taup] + N    [ormbr work]
          // Workspace: prefer 3*M [e, tauq, taup] + N*NB [ormbr work]
          lwork2 = lwork - iwork + 1;
          magma_sormbr( MagmaP, MagmaLeft, MagmaTrans,  m,  n,  m, U,  ldu, &work[itaup], VT,  ldvt, &work[iwork],  lwork2, &ierr );

          // Generate left bidiagonalizing vectors in U
          // Workspace: need   3*M [e, tauq, taup] + M    [orgbr work]
          // Workspace: prefer 3*M [e, tauq, taup] + M*NB [orgbr work]
          lwork2 = lwork - iwork + 1;
          magma_sorgbr( MagmaQ,   m,  m,  m, U,  ldu, &work[itauq], &work[iwork],  lwork2, &ierr );
          iwork = ie + m;

          // Perform bidiagonal QR iteration, computing left
          // singular vectors of A in U and computing right
          // singular vectors of A in VT
          // Workspace: need     M [e] + 4*M [bdsqr work]
          lapack::bdsqr('U', m, n, m, izero, s, &work[ie], VT, ldvt, U, ldu, dummy, ione, &work[iwork], info);
          }
        }
      }
    else
      {
      // N < MNTHR
      // Path 10t (N > M, but not much larger)
      // Reduce to bidiagonal form without LQ decomposition
      ie    = 1;
      itauq = ie + m;
      itaup = itauq + m;
      iwork = itaup + m;

      // Bidiagonalize A
      // Workspace: need   3*M [e, tauq, taup] + N        [gebrd work]
      // Workspace: prefer 3*M [e, tauq, taup] + (M+N)*NB [gebrd work]
      lwork2 = lwork - iwork + 1;
      magma_sgebrd(      m,  n, A,  lda, s, &work[ie], &work[itauq], &work[itaup], &work[iwork],  lwork2, &ierr );

      if (want_uas)
        {
        // If left singular vectors desired in U, copy result to U
        // and generate left bidiagonalizing vectors in U
        // Workspace: need     3*M [e, tauq, taup] + M    [orgbr work]
        // Workspace: prefer   3*M [e, tauq, taup] + M*NB [orgbr work]
        lapack::lacpy('L', m, m, A, lda, U, ldu);
        lwork2 = lwork - iwork + 1;
        magma_sorgbr( MagmaQ,   m,  m,  n, U,  ldu, &work[itauq], &work[iwork],  lwork2, &ierr );
        }
      if (want_vas)
        {
        // If right singular vectors desired in VT, copy result to
        // VT and generate right bidiagonalizing vectors in VT
        // Workspace: need     3*M [e, tauq, taup] + NRVT     [orgbr work]
        // Workspace: prefer   3*M [e, tauq, taup] + NRVT*NB  [orgbr work]
        lapack::lacpy('U', m, n, A, lda, VT, ldvt);
        if (want_va)
          {
          nrvt = n;
          }
        else
          {
          assert(want_vs);
          nrvt = m;
          }
        lwork2 = lwork - iwork + 1;
        magma_sorgbr( MagmaP,  nrvt,  n,  m, VT,  ldvt, &work[itaup], &work[iwork],  lwork2, &ierr );
        }
      if (want_uo)
        {
        // If left singular vectors desired in A, generate left
        // bidiagonalizing vectors in A
        // Workspace: need     3*M [e, tauq, taup] + M    [orgbr work]
        // Workspace: prefer   3*M [e, tauq, taup] + M*NB [orgbr work]
        lwork2 = lwork - iwork + 1;
        magma_sorgbr( MagmaQ,   m,  m,  n, A,  lda, &work[itauq], &work[iwork],  lwork2, &ierr );
        }
      if (want_vo)
        {
        // If right singular vectors desired in A, generate right
        // bidiagonalizing vectors in A
        // Workspace: need     3*M [e, tauq, taup] + M    [orgbr work]
        // Workspace: prefer   3*M [e, tauq, taup] + M*NB [orgbr work]
        lwork2 = lwork - iwork + 1;
        magma_sorgbr( MagmaP,   m,  n,  m, A,  lda, &work[itaup], &work[iwork],  lwork2, &ierr );
        }

      iwork = ie + m;
      if (want_uas || want_uo)
        {
        nru = m;
        }
      if (want_un)
        {
        nru = 0;
        }
      if (want_vas || want_vo)
        {
        ncvt = n;
        }
      if (want_vn)
        {
        ncvt = 0;
        }
      if (!want_uo && ! want_vo)
        {
        // Perform bidiagonal QR iteration, if desired, computing
        // left singular vectors in U and computing right singular
        // vectors in VT
        // Workspace: need       M [e] + 4*M [bdsqr work]
        lapack::bdsqr('L', m, ncvt, nru, izero, s, &work[ie], VT, ldvt, U, ldu, dummy, ione, &work[iwork], info);
        }
      else if (!want_uo && want_vo)
        {
        // Perform bidiagonal QR iteration, if desired, computing
        // left singular vectors in U and computing right singular
        // vectors in A
        // Workspace: need       M [e] + 4*M [bdsqr work]
        lapack::bdsqr('L', m, ncvt, nru, izero, s, &work[ie], A, lda, U, ldu, dummy, ione, &work[iwork], info);
        }
      else
        {
        // Perform bidiagonal QR iteration, if desired, computing
        // left singular vectors in A and computing right singular
        // vectors in VT
        // Workspace: need       M [e] + 4*M [bdsqr work]
        lapack::bdsqr('L', m, ncvt, nru, izero, s, &work[ie], VT, ldvt, A, lda, dummy, ione, &work[iwork], info);
        }
      }
    }

  // If DBDSQR failed to converge, copy unconverged superdiagonals
  // to WORK( 2:MINMN )
  if (*info != 0)
    {
    if (ie > 2)
      {
      m_1 = minmn - 1;
      for (i = 1; i <= m_1; ++i)
        {
        work[i + 1] = work[i + ie - 1];
        }
      }
    if (ie < 2)
      {
      for (i = minmn - 1; i >= 1; --i)
        {
        work[i + 1] = work[i + ie - 1];
        }
      }
    }

  // Undo scaling if necessary
  if (iscl == 1)
    {
    if (anrm > bignum)
      {
      lapack::lascl('G', izero, izero, bignum, anrm, minmn, ione, s, minmn, &ierr);
      }
    if (*info != 0 && anrm > bignum)
      {
      m_1 = minmn - 1;
      lapack::lascl('G', izero, izero, bignum, anrm, m_1, ione, &work[2], minmn, &ierr);
      }
    if (anrm < smlnum)
      {
      lapack::lascl('G', izero, izero, smlnum, anrm, minmn, ione, s, minmn, &ierr);
      }
    if (*info != 0 && anrm < smlnum)
      {
      m_1 = minmn - 1;
      lapack::lascl('G', izero, izero, smlnum, anrm, m_1, ione, &work[2], minmn, &ierr);
      }
    }

  // Return optimal workspace in WORK[1] (Fortran index)
  work[1] = magma_smake_lwork( maxwrk );

  return *info;
  } // magma_sgesvd
