// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2023 Ryan Curtin (https://www.ratml.org)
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



// This file contains wrappers for LAPACK functions.
// These should be used preferentially to calling coot_fortran(coot_XXXXX),
// because the compiler may need extra hidden arguments for FORTRAN calls.
// (See the definition of COOT_USE_FORTRAN_HIDDEN_ARGS for more details.)

namespace lapack
  {



  // LU factorisation
  template<typename eT>
  inline
  void
  getrf(const blas_int m, const blas_int n, eT* A, const blas_int lda, blas_int* ipiv, blas_int* info)
    {
    coot_type_check((is_supported_blas_type<eT>::value == false));

    // No hidden arguments.
    if     (    is_float<eT>::value) { coot_fortran(coot_sgetrf)(&m, &n,    (float*) A, &lda, ipiv, info); }
    else if(   is_double<eT>::value) { coot_fortran(coot_dgetrf)(&m, &n,   (double*) A, &lda, ipiv, info); }
    else if( is_cx_float<eT>::value) { coot_fortran(coot_cgetrf)(&m, &n, (blas_cxf*) A, &lda, ipiv, info); }
    else if(is_cx_double<eT>::value) { coot_fortran(coot_zgetrf)(&m, &n, (blas_cxd*) A, &lda, ipiv, info); }
    }



  // matrix inversion (triangular matrices)
  template<typename eT>
  inline
  void
  trtri(const char uplo, const char diag, const blas_int n, eT* A, const blas_int lda, blas_int* info)
    {
    coot_type_check((is_supported_blas_type<eT>::value == false));

    #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
      {
      if     (    is_float<eT>::value) { coot_fortran(coot_strtri)(&uplo, &diag, &n,    (float*) A, &lda, info, 1, 1); }
      else if(   is_double<eT>::value) { coot_fortran(coot_dtrtri)(&uplo, &diag, &n,   (double*) A, &lda, info, 1, 1); }
      else if( is_cx_float<eT>::value) { coot_fortran(coot_ctrtri)(&uplo, &diag, &n, (blas_cxf*) A, &lda, info, 1, 1); }
      else if(is_cx_double<eT>::value) { coot_fortran(coot_ztrtri)(&uplo, &diag, &n, (blas_cxd*) A, &lda, info, 1, 1); }
      }
    #else
      {
      if     (    is_float<eT>::value) { coot_fortran(coot_strtri)(&uplo, &diag, &n,    (float*) A, &lda, info); }
      else if(   is_double<eT>::value) { coot_fortran(coot_dtrtri)(&uplo, &diag, &n,   (double*) A, &lda, info); }
      else if( is_cx_float<eT>::value) { coot_fortran(coot_ctrtri)(&uplo, &diag, &n, (blas_cxf*) A, &lda, info); }
      else if(is_cx_double<eT>::value) { coot_fortran(coot_ztrtri)(&uplo, &diag, &n, (blas_cxd*) A, &lda, info); }
      }
    #endif
    }



  // eigendecomposition of symmetric real matrices by divide and conquer
  template<typename eT>
  inline
  void
  syevd(const char jobz, const char uplo, const blas_int n, eT* a, const blas_int lda, eT* w, eT* work, const blas_int lwork, blas_int* iwork, const blas_int liwork, blas_int* info)
    {
    coot_type_check((is_supported_real_blas_type<eT>::value == false));

    #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
      {
      if     ( is_float<eT>::value) { coot_fortran(coot_ssyevd)(&jobz, &uplo, &n,  (float*) a, &lda,  (float*) w,  (float*) work, &lwork, iwork, &liwork, info, 1, 1); }
      else if(is_double<eT>::value) { coot_fortran(coot_dsyevd)(&jobz, &uplo, &n, (double*) a, &lda, (double*) w, (double*) work, &lwork, iwork, &liwork, info, 1, 1); }
      }
    #else
      {
      if     ( is_float<eT>::value) { coot_fortran(coot_ssyevd)(&jobz, &uplo, &n,  (float*) a, &lda,  (float*) w,  (float*) work, &lwork, iwork, &liwork, info); }
      else if(is_double<eT>::value) { coot_fortran(coot_dsyevd)(&jobz, &uplo, &n, (double*) a, &lda, (double*) w, (double*) work, &lwork, iwork, &liwork, info); }
      }
    #endif
    }



  // Cholesky decomposition
  template<typename eT>
  inline
  void
  potrf(const char uplo, const blas_int n, eT* a, const blas_int lda, blas_int* info)
    {
    coot_type_check((is_supported_blas_type<eT>::value == false));

    #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
      {
      if     (    is_float<eT>::value) { coot_fortran(coot_spotrf)(&uplo, &n,    (float*) a, &lda, info, 1); }
      else if(   is_double<eT>::value) { coot_fortran(coot_dpotrf)(&uplo, &n,   (double*) a, &lda, info, 1); }
      else if( is_cx_float<eT>::value) { coot_fortran(coot_cpotrf)(&uplo, &n, (blas_cxf*) a, &lda, info, 1); }
      else if(is_cx_double<eT>::value) { coot_fortran(coot_zpotrf)(&uplo, &n, (blas_cxd*) a, &lda, info, 1); }
      }
    #else
      {
      if     (    is_float<eT>::value) { coot_fortran(coot_spotrf)(&uplo, &n,    (float*) a, &lda, info); }
      else if(   is_double<eT>::value) { coot_fortran(coot_dpotrf)(&uplo, &n,   (double*) a, &lda, info); }
      else if( is_cx_float<eT>::value) { coot_fortran(coot_cpotrf)(&uplo, &n, (blas_cxf*) a, &lda, info); }
      else if(is_cx_double<eT>::value) { coot_fortran(coot_zpotrf)(&uplo, &n, (blas_cxd*) a, &lda, info); }
      }
    #endif
    }



  // QR decomposition
  template<typename eT>
  inline
  void
  geqrf(const blas_int m, const blas_int n, eT* a, const blas_int lda, eT* tau, eT* work, const blas_int lwork, blas_int* info)
    {
    coot_type_check((is_supported_blas_type<eT>::value == false));

    // No hidden arguments.
    if     (    is_float<eT>::value) { coot_fortran(coot_sgeqrf)(&m, &n,    (float*) a, &lda,    (float*) tau,    (float*) work, &lwork, info); }
    else if(   is_double<eT>::value) { coot_fortran(coot_dgeqrf)(&m, &n,   (double*) a, &lda,   (double*) tau,   (double*) work, &lwork, info); }
    else if( is_cx_float<eT>::value) { coot_fortran(coot_cgeqrf)(&m, &n, (blas_cxf*) a, &lda, (blas_cxf*) tau, (blas_cxf*) work, &lwork, info); }
    else if(is_cx_double<eT>::value) { coot_fortran(coot_zgeqrf)(&m, &n, (blas_cxd*) a, &lda, (blas_cxd*) tau, (blas_cxd*) work, &lwork, info); }
    }



  // Q matrix calculation from QR decomposition (real matrices)
  template<typename eT>
  inline
  void
  orgqr(const blas_int m, const blas_int n, const blas_int k, eT* a, const blas_int lda, eT* tau, eT* work, const blas_int lwork, blas_int* info)
    {
    coot_type_check((is_supported_real_blas_type<eT>::value == false));

    // No hidden arguments.
    if     ( is_float<eT>::value) { coot_fortran(coot_sorgqr)(&m, &n, &k,  (float*) a, &lda,  (float*) tau,  (float*) work, &lwork, info); }
    else if(is_double<eT>::value) { coot_fortran(coot_dorgqr)(&m, &n, &k, (double*) a, &lda, (double*) tau, (double*) work, &lwork, info); }
    }



  // 1-norm
  template<typename eT>
  inline
  typename get_pod_type<eT>::result
  lange(const char norm, const blas_int m, const blas_int n, eT* a, const blas_int lda, typename get_pod_type<eT>::result* work)
    {
    coot_type_check((is_supported_blas_type<eT>::value == false));

    #if defined(COOT_FORTRAN_FLOAT_WORKAROUND)
    typedef typename get_pod_type<eT>::result out_eT;
    if (is_float<eT>::value || is_cx_float<eT>::value)
      {
      // We can't call slange or clange, because we are on a system where they
      // may return doubles instead of floats---and we have defined the
      // prototype as returning a float.  Therefore, we have to use a by-hand
      // implementation...
      out_eT retval = out_eT(0);
      if (norm == 'm' || norm == 'M')
        {
        // max-norm: maximum absolute valued element
        for (uword c = 0; c < n; ++c)
          {
          for (uword r = 0; r < m; ++r)
            {
            retval = std::max(retval, std::abs(a[c * lda + r]));
            }
          }
        }
      else if (norm == '1' || norm == 'O' || norm == 'o')
        {
        // 1-norm: maximum column sum
        for (uword c = 0; c < n; ++c)
          {
          out_eT colsum = out_eT(0);
          for (uword r = 0; r < m; ++r)
            {
            colsum += std::abs(a[c * lda + r]);
            }

          retval = std::max(retval, colsum);
          }
        }
      else if (norm == 'I' || norm == 'i')
        {
        // Inf-norm: maximum row sum
        // We'll need to use the work array for this...

        for (uword r = 0; r < m; ++r) { work[r] = out_eT(0); }

        for (uword c = 0; c < n; ++c)
          {
          for (uword r = 0; r < m; ++r)
            {
            work[r] += std::abs(a[c * lda + r]);
            }
          }

        // Now find the maximum row sum.
        for (uword r = 0; r < m; ++r)
          {
          retval = std::max(retval, work[r]);
          }
        }
      else if (norm == 'F' || norm == 'f' || norm == 'E' || norm == 'e')
        {
        // Frobenius norm: square root of sum of squares.
        // (our implementation doesn't have the scaling checks that LAPACK typically does)
        for (uword c = 0; c < n; ++c)
          {
          for (uword r = 0; r < m; ++r)
            {
            const out_eT tmp = std::abs(a[c * lda + r]);
            retval += (tmp * tmp);
            }
          }

        retval = std::sqrt(retval);
        }

      return retval;
      }
    #endif

    #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
      {
      if     (    is_float<eT>::value) { return coot_fortran(coot_slange)(&norm, &m, &n,    (float*) a, &lda,  (float*) work, 1); }
      else if(   is_double<eT>::value) { return coot_fortran(coot_dlange)(&norm, &m, &n,   (double*) a, &lda, (double*) work, 1); }
      else if( is_cx_float<eT>::value) { return coot_fortran(coot_clange)(&norm, &m, &n, (blas_cxf*) a, &lda,  (float*) work, 1); }
      else if(is_cx_double<eT>::value) { return coot_fortran(coot_zlange)(&norm, &m, &n, (blas_cxd*) a, &lda, (double*) work, 1); }
      }
    #else
      {
      if     (    is_float<eT>::value) { return coot_fortran(coot_slange)(&norm, &m, &n,    (float*) a, &lda,  (float*) work); }
      else if(   is_double<eT>::value) { return coot_fortran(coot_dlange)(&norm, &m, &n,   (double*) a, &lda, (double*) work); }
      else if( is_cx_float<eT>::value) { return coot_fortran(coot_clange)(&norm, &m, &n, (blas_cxf*) a, &lda,  (float*) work); }
      else if(is_cx_double<eT>::value) { return coot_fortran(coot_zlange)(&norm, &m, &n, (blas_cxd*) a, &lda, (double*) work); }
      }
    #endif
    }



  // triangular factor of block reflector
  template<typename eT>
  inline
  void
  larft(const char direct, const char storev, const blas_int n, const blas_int k, eT* v, const blas_int ldv, eT* tau, eT* t, const blas_int ldt)
    {
    coot_type_check((is_supported_blas_type<eT>::value == false));

    #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
      {
      if     (    is_float<eT>::value) { coot_fortran(coot_slarft)(&direct, &storev, &n, &k,    (float*) v, &ldv,    (float*) tau,    (float*) t, &ldt, 1, 1); }
      else if(   is_double<eT>::value) { coot_fortran(coot_dlarft)(&direct, &storev, &n, &k,   (double*) v, &ldv,   (double*) tau,   (double*) t, &ldt, 1, 1); }
      else if( is_cx_float<eT>::value) { coot_fortran(coot_clarft)(&direct, &storev, &n, &k, (blas_cxf*) v, &ldv, (blas_cxf*) tau, (blas_cxf*) t, &ldt, 1, 1); }
      else if(is_cx_double<eT>::value) { coot_fortran(coot_zlarft)(&direct, &storev, &n, &k, (blas_cxd*) v, &ldv, (blas_cxd*) tau, (blas_cxd*) t, &ldt, 1, 1); }
      }
    #else
      {
      if     (    is_float<eT>::value) { coot_fortran(coot_slarft)(&direct, &storev, &n, &k,    (float*) v, &ldv,    (float*) tau,    (float*) t, &ldt); }
      else if(   is_double<eT>::value) { coot_fortran(coot_dlarft)(&direct, &storev, &n, &k,   (double*) v, &ldv,   (double*) tau,   (double*) t, &ldt); }
      else if( is_cx_float<eT>::value) { coot_fortran(coot_clarft)(&direct, &storev, &n, &k, (blas_cxf*) v, &ldv, (blas_cxf*) tau, (blas_cxf*) t, &ldt); }
      else if(is_cx_double<eT>::value) { coot_fortran(coot_zlarft)(&direct, &storev, &n, &k, (blas_cxd*) v, &ldv, (blas_cxd*) tau, (blas_cxd*) t, &ldt); }
      }
    #endif
    }



  // generate an elementary reflector
  template<typename eT>
  inline
  void
  larfg(const blas_int n, eT* alpha, eT* x, const blas_int incx, eT* tau)
    {
    coot_type_check((is_supported_blas_type<eT>::value == false));

    // No hidden arguments.
    if     (    is_float<eT>::value) { coot_fortran(coot_slarfg)(&n,    (float*) alpha,    (float*) x, &incx,    (float*) tau); }
    else if(   is_double<eT>::value) { coot_fortran(coot_dlarfg)(&n,   (double*) alpha,   (double*) x, &incx,   (double*) tau); }
    else if( is_cx_float<eT>::value) { coot_fortran(coot_clarfg)(&n, (blas_cxf*) alpha, (blas_cxf*) x, &incx, (blas_cxf*) tau); }
    else if(is_cx_double<eT>::value) { coot_fortran(coot_zlarfg)(&n, (blas_cxd*) alpha, (blas_cxd*) x, &incx, (blas_cxd*) tau); }
    }



  // reduce a general matrix to bidiagonal form
  template<typename eT>
  inline
  void
  gebrd(const blas_int m, const blas_int n, eT* a, const blas_int lda, eT* d, eT* e, eT* tauq, eT* taup, eT* work, const blas_int lwork, blas_int* info)
    {
    coot_type_check((is_supported_blas_type<eT>::value == false));

    // No hidden arguments.
    if     (    is_float<eT>::value) { coot_fortran(coot_sgebrd)(&m, &n,    (float*) a, &lda,    (float*) d,    (float*) e,    (float*) tauq,    (float*) taup,    (float*) work, &lwork, info); }
    else if(   is_double<eT>::value) { coot_fortran(coot_dgebrd)(&m, &n,   (double*) a, &lda,   (double*) d,   (double*) e,   (double*) tauq,   (double*) taup,   (double*) work, &lwork, info); }
    else if( is_cx_float<eT>::value) { coot_fortran(coot_cgebrd)(&m, &n, (blas_cxf*) a, &lda, (blas_cxf*) d, (blas_cxf*) e, (blas_cxf*) tauq, (blas_cxf*) taup, (blas_cxf*) work, &lwork, info); }
    else if(is_cx_double<eT>::value) { coot_fortran(coot_zgebrd)(&m, &n, (blas_cxd*) a, &lda, (blas_cxd*) d, (blas_cxd*) e, (blas_cxd*) tauq, (blas_cxd*) taup, (blas_cxd*) work, &lwork, info); }
    }



  // overwrite matrix with geqrf-generated orthogonal transformation
  template<typename eT>
  inline
  void
  ormqr(const char side, const char trans, const blas_int m, const blas_int n, const blas_int k, const eT* A, const blas_int lda, const eT* tau, eT* C, const blas_int ldc, eT* work, const blas_int lwork, blas_int* info)
    {
    coot_type_check((is_supported_real_blas_type<eT>::value == false));

    #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
      {
      if     ( is_float<eT>::value) { coot_fortran(coot_sormqr)(&side, &trans, &m, &n, &k,  (const float*) A, &lda,  (const float*) tau,  (float*) C, &ldc,  (float*) work, &lwork, info, 1, 1); }
      else if(is_double<eT>::value) { coot_fortran(coot_dormqr)(&side, &trans, &m, &n, &k, (const double*) A, &lda, (const double*) tau, (double*) C, &ldc, (double*) work, &lwork, info, 1, 1); }
      }
    #else
      {
      if     ( is_float<eT>::value) { coot_fortran(coot_sormqr)(&side, &trans, &m, &n, &k,  (const float*) A, &lda,  (const float*) tau,  (float*) C, &ldc,  (float*) work, &lwork, info); }
      else if(is_double<eT>::value) { coot_fortran(coot_dormqr)(&side, &trans, &m, &n, &k, (const double*) A, &lda, (const double*) tau, (double*) C, &ldc, (double*) work, &lwork, info); }
      }
    #endif
    }



  // overwrite matrix with gelqf-generated orthogonal matrix
  template<typename eT>
  inline
  void
  ormlq(const char side, const char trans, const blas_int m, const blas_int n, const blas_int k, const eT* A, const blas_int lda, const eT* tau, eT* C, const blas_int ldc, eT* work, const blas_int lwork, blas_int* info)
    {
    coot_type_check((is_supported_real_blas_type<eT>::value == false));

    #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
      {
      if     ( is_float<eT>::value) { coot_fortran(coot_sormlq)(&side, &trans, &m, &n, &k,  (const float*) A, &lda,  (const float*) tau,  (float*) C, &ldc,  (float*) work, &lwork, info, 1, 1); }
      else if(is_double<eT>::value) { coot_fortran(coot_dormlq)(&side, &trans, &m, &n, &k, (const double*) A, &lda, (const double*) tau, (double*) C, &ldc, (double*) work, &lwork, info, 1, 1); }
      }
    #else
      {
      if     ( is_float<eT>::value) { coot_fortran(coot_sormlq)(&side, &trans, &m, &n, &k,  (const float*) A, &lda,  (const float*) tau,  (float*) C, &ldc,  (float*) work, &lwork, info); }
      else if(is_double<eT>::value) { coot_fortran(coot_dormlq)(&side, &trans, &m, &n, &k, (const double*) A, &lda, (const double*) tau, (double*) C, &ldc, (double*) work, &lwork, info); }
      }
    #endif
    }



  // copy all or part of one 2d array to another
  template<typename eT>
  inline
  void
  lacpy(const char uplo, const blas_int m, const blas_int n, const eT* A, const blas_int lda, eT* B, const blas_int ldb)
    {
    coot_type_check((is_supported_blas_type<eT>::value == false));

    #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
      {
      if     (    is_float<eT>::value) { coot_fortran(coot_slacpy)(&uplo, &m, &n,    (const float*) A, &lda,    (float*) B, &ldb, 1); }
      else if(   is_double<eT>::value) { coot_fortran(coot_dlacpy)(&uplo, &m, &n,   (const double*) A, &lda,   (double*) B, &ldb, 1); }
      else if( is_cx_float<eT>::value) { coot_fortran(coot_clacpy)(&uplo, &m, &n, (const blas_cxf*) A, &lda, (blas_cxf*) B, &ldb, 1); }
      else if(is_cx_double<eT>::value) { coot_fortran(coot_zlacpy)(&uplo, &m, &n, (const blas_cxd*) A, &lda, (blas_cxd*) B, &ldb, 1); }
      }
    #else
      {
      if     (    is_float<eT>::value) { coot_fortran(coot_slacpy)(&uplo, &m, &n,    (const float*) A, &lda,    (float*) B, &ldb); }
      else if(   is_double<eT>::value) { coot_fortran(coot_dlacpy)(&uplo, &m, &n,   (const double*) A, &lda,   (double*) B, &ldb); }
      else if( is_cx_float<eT>::value) { coot_fortran(coot_clacpy)(&uplo, &m, &n, (const blas_cxf*) A, &lda, (blas_cxf*) B, &ldb); }
      else if(is_cx_double<eT>::value) { coot_fortran(coot_zlacpy)(&uplo, &m, &n, (const blas_cxd*) A, &lda, (blas_cxd*) B, &ldb); }
      }
    #endif
    }



  // initialize a matrix with different elements on and off the diagonal
  template<typename eT>
  inline
  void
  laset(const char uplo, const blas_int m, const blas_int n, const eT alpha, const eT beta, eT* A, const blas_int lda)
    {
    coot_type_check((is_supported_blas_type<eT>::value == false));

    #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
      {
      if     (    is_float<eT>::value) { coot_fortran(coot_slaset)(&uplo, &m, &n,    (const float*) &alpha,    (const float*) &beta,    (float*) A, &lda, 1); }
      else if(   is_double<eT>::value) { coot_fortran(coot_dlaset)(&uplo, &m, &n,   (const double*) &alpha,   (const double*) &beta,   (double*) A, &lda, 1); }
      else if( is_cx_float<eT>::value) { coot_fortran(coot_claset)(&uplo, &m, &n, (const blas_cxf*) &alpha, (const blas_cxf*) &beta, (blas_cxf*) A, &lda, 1); }
      else if(is_cx_double<eT>::value) { coot_fortran(coot_zlaset)(&uplo, &m, &n, (const blas_cxd*) &alpha, (const blas_cxd*) &beta, (blas_cxd*) A, &lda, 1); }
      }
    #else
      {
      if     (    is_float<eT>::value) { coot_fortran(coot_slaset)(&uplo, &m, &n,    (const float*) &alpha,    (const float*) &beta,    (float*) A, &lda); }
      else if(   is_double<eT>::value) { coot_fortran(coot_dlaset)(&uplo, &m, &n,   (const double*) &alpha,   (const double*) &beta,   (double*) A, &lda); }
      else if( is_cx_float<eT>::value) { coot_fortran(coot_claset)(&uplo, &m, &n, (const blas_cxf*) &alpha, (const blas_cxf*) &beta, (blas_cxf*) A, &lda); }
      else if(is_cx_double<eT>::value) { coot_fortran(coot_zlaset)(&uplo, &m, &n, (const blas_cxd*) &alpha, (const blas_cxd*) &beta, (blas_cxd*) A, &lda); }
      }
    #endif
    }



  // apply block reflector to general rectangular matrix
  template<typename eT>
  inline
  void
  larfb(const char side, const char trans, const char direct, const char storev, const blas_int M, const blas_int N, const blas_int K, const eT* V, const blas_int ldv, const eT* T, const blas_int ldt, eT* C, const blas_int ldc, eT* work, const blas_int ldwork)
    {
    coot_type_check((is_supported_blas_type<eT>::value == false));

    #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
      {
      if     (    is_float<eT>::value) { coot_fortran(coot_slarfb)(&side, &trans, &direct, &storev, &M, &N, &K,    (const float*) V, &ldv,    (const float*) T, &ldt,    (float*) C, &ldc,    (float*) work, &ldwork, 1, 1, 1, 1); }
      else if(   is_double<eT>::value) { coot_fortran(coot_dlarfb)(&side, &trans, &direct, &storev, &M, &N, &K,   (const double*) V, &ldv,   (const double*) T, &ldt,   (double*) C, &ldc,   (double*) work, &ldwork, 1, 1, 1, 1); }
      else if( is_cx_float<eT>::value) { coot_fortran(coot_clarfb)(&side, &trans, &direct, &storev, &M, &N, &K, (const blas_cxf*) V, &ldv, (const blas_cxf*) T, &ldt, (blas_cxf*) C, &ldc, (blas_cxf*) work, &ldwork, 1, 1, 1, 1); }
      else if(is_cx_double<eT>::value) { coot_fortran(coot_zlarfb)(&side, &trans, &direct, &storev, &M, &N, &K, (const blas_cxd*) V, &ldv, (const blas_cxd*) T, &ldt, (blas_cxd*) C, &ldc, (blas_cxd*) work, &ldwork, 1, 1, 1, 1); }
      }
    #else
      {
      if     (    is_float<eT>::value) { coot_fortran(coot_slarfb)(&side, &trans, &direct, &storev, &M, &N, &K,    (const float*) V, &ldv,    (const float*) T, &ldt,    (float*) C, &ldc,    (float*) work, &ldwork); }
      else if(   is_double<eT>::value) { coot_fortran(coot_dlarfb)(&side, &trans, &direct, &storev, &M, &N, &K,   (const double*) V, &ldv,   (const double*) T, &ldt,   (double*) C, &ldc,   (double*) work, &ldwork); }
      else if( is_cx_float<eT>::value) { coot_fortran(coot_clarfb)(&side, &trans, &direct, &storev, &M, &N, &K, (const blas_cxf*) V, &ldv, (const blas_cxf*) T, &ldt, (blas_cxf*) C, &ldc, (blas_cxf*) work, &ldwork); }
      else if(is_cx_double<eT>::value) { coot_fortran(coot_zlarfb)(&side, &trans, &direct, &storev, &M, &N, &K, (const blas_cxd*) V, &ldv, (const blas_cxd*) T, &ldt, (blas_cxd*) C, &ldc, (blas_cxd*) work, &ldwork); }
      }
    #endif
    }



  // get machine parameters
  template<typename eT>
  inline
  eT
  lamch(const char cmach)
    {
    coot_type_check((is_supported_real_blas_type<eT>::value == false));

    #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
      {
      if     ( is_float<eT>::value) { return coot_fortran(coot_slamch)(&cmach, 1); }
      else if(is_double<eT>::value) { return coot_fortran(coot_dlamch)(&cmach, 1); }
      }
    #else
      {
      if     ( is_float<eT>::value) { return coot_fortran(coot_slamch)(&cmach); }
      else if(is_double<eT>::value) { return coot_fortran(coot_dlamch)(&cmach); }
      }
    #endif
    }



  // scale matrix by a scalar
  template<typename eT>
  inline
  void
  lascl(const char type, const blas_int kl, const blas_int ku, const eT cfrom, const eT cto, const blas_int m, const blas_int n, eT* a, const blas_int lda, blas_int* info)
    {
    coot_type_check((is_supported_blas_type<eT>::value == false));

    #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
      {
      if     (    is_float<eT>::value) { coot_fortran(coot_slascl)(&type, &kl, &ku,    (const float*) &cfrom,    (const float*) &cto, &m, &n,    (float*) a, &lda, info, 1); }
      else if(   is_double<eT>::value) { coot_fortran(coot_dlascl)(&type, &kl, &ku,   (const double*) &cfrom,   (const double*) &cto, &m, &n,   (double*) a, &lda, info, 1); }
      else if( is_cx_float<eT>::value) { coot_fortran(coot_clascl)(&type, &kl, &ku, (const blas_cxf*) &cfrom, (const blas_cxf*) &cto, &m, &n, (blas_cxf*) a, &lda, info, 1); }
      else if(is_cx_double<eT>::value) { coot_fortran(coot_zlascl)(&type, &kl, &ku, (const blas_cxd*) &cfrom, (const blas_cxd*) &cto, &m, &n, (blas_cxd*) a, &lda, info, 1); }
      }
    #else
      {
      if     (    is_float<eT>::value) { coot_fortran(coot_slascl)(&type, &kl, &ku,    (const float*) &cfrom,    (const float*) &cto, &m, &n,    (float*) a, &lda, info); }
      else if(   is_double<eT>::value) { coot_fortran(coot_dlascl)(&type, &kl, &ku,   (const double*) &cfrom,   (const double*) &cto, &m, &n,   (double*) a, &lda, info); }
      else if( is_cx_float<eT>::value) { coot_fortran(coot_clascl)(&type, &kl, &ku, (const blas_cxf*) &cfrom, (const blas_cxf*) &cto, &m, &n, (blas_cxf*) a, &lda, info); }
      else if(is_cx_double<eT>::value) { coot_fortran(coot_zlascl)(&type, &kl, &ku, (const blas_cxd*) &cfrom, (const blas_cxd*) &cto, &m, &n, (blas_cxd*) a, &lda, info); }
      }
    #endif
    }



  // compute singular values of bidiagonal matrix
  template<typename eT>
  inline
  void
  bdsqr(const char uplo, const blas_int n, const blas_int ncvt, const blas_int nru, const blas_int ncc, typename get_pod_type<eT>::result* d, typename get_pod_type<eT>::result* e, eT* vt, const blas_int ldvt, eT* u, const blas_int ldu, eT* c, const blas_int ldc, typename get_pod_type<eT>::result* work, blas_int* info)
    {
    coot_type_check((is_supported_blas_type<eT>::value == false));

    #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
      {
      if     (    is_float<eT>::value) { coot_fortran(coot_sbdsqr)(&uplo, &n, &ncvt, &nru, &ncc,  (float*) d,  (float*) e,    (float*) vt, &ldvt,    (float*) u, &ldu,    (float*) c, &ldc,  (float*) work, info, 1); }
      else if(   is_double<eT>::value) { coot_fortran(coot_dbdsqr)(&uplo, &n, &ncvt, &nru, &ncc, (double*) d, (double*) e,   (double*) vt, &ldvt,   (double*) u, &ldu,   (double*) c, &ldc, (double*) work, info, 1); }
      else if( is_cx_float<eT>::value) { coot_fortran(coot_cbdsqr)(&uplo, &n, &ncvt, &nru, &ncc,  (float*) d,  (float*) e, (blas_cxf*) vt, &ldvt, (blas_cxf*) u, &ldu, (blas_cxf*) c, &ldc,  (float*) work, info, 1); }
      else if(is_cx_double<eT>::value) { coot_fortran(coot_zbdsqr)(&uplo, &n, &ncvt, &nru, &ncc, (double*) d, (double*) e, (blas_cxd*) vt, &ldvt, (blas_cxd*) u, &ldu, (blas_cxd*) c, &ldc, (double*) work, info, 1); }
      }
    #else
      {
      if     (    is_float<eT>::value) { coot_fortran(coot_sbdsqr)(&uplo, &n, &ncvt, &nru, &ncc,  (float*) d,  (float*) e,    (float*) vt, &ldvt,    (float*) u, &ldu,    (float*) c, &ldc,  (float*) work, info); }
      else if(   is_double<eT>::value) { coot_fortran(coot_dbdsqr)(&uplo, &n, &ncvt, &nru, &ncc, (double*) d, (double*) e,   (double*) vt, &ldvt,   (double*) u, &ldu,   (double*) c, &ldc, (double*) work, info); }
      else if( is_cx_float<eT>::value) { coot_fortran(coot_cbdsqr)(&uplo, &n, &ncvt, &nru, &ncc,  (float*) d,  (float*) e, (blas_cxf*) vt, &ldvt, (blas_cxf*) u, &ldu, (blas_cxf*) c, &ldc,  (float*) work, info); }
      else if(is_cx_double<eT>::value) { coot_fortran(coot_zbdsqr)(&uplo, &n, &ncvt, &nru, &ncc, (double*) d, (double*) e, (blas_cxd*) vt, &ldvt, (blas_cxd*) u, &ldu, (blas_cxd*) c, &ldc, (double*) work, info); }
      }
    #endif
    }



  // merges two sets of eigenvalues together into a single sorted set
  template<typename eT>
  inline
  void
  laed2(blas_int* k, const blas_int n, const blas_int n1, eT* D, eT* Q, const blas_int ldq, blas_int* indxq, eT* rho, const eT* Z, eT* dlambda, eT* W, eT* Q2, blas_int* indx, blas_int* indxc, blas_int* indxp, blas_int* coltyp, blas_int* info)
    {
    coot_type_check((is_supported_real_blas_type<eT>::value == false));

    // No hidden arguments.
    if     ( is_float<eT>::value) { coot_fortran(coot_slaed2)(k, &n, &n1,  (float*) D,  (float*) Q, &ldq, indxq,  (float*) rho,  (const float*) Z,  (float*) dlambda,  (float*) W,  (float*) Q2, indx, indxc, indxp, coltyp, info); }
    else if(is_double<eT>::value) { coot_fortran(coot_dlaed2)(k, &n, &n1, (double*) D, (double*) Q, &ldq, indxq, (double*) rho, (const double*) Z, (double*) dlambda, (double*) W, (double*) Q2, indx, indxc, indxp, coltyp, info); }
    }



  // compute all eigenvalues (and optionally eigenvectors) of symmetric tridiagonal matrix
  template<typename eT>
  inline
  void
  steqr(const char compz, const blas_int n, eT* D, eT* E, eT* Z, const blas_int ldz, eT* work, blas_int* info)
    {
    coot_type_check((is_supported_blas_type<eT>::value == false));

    #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
      {
      if     (    is_float<eT>::value) { coot_fortran(coot_ssteqr)(&compz, &n,    (float*) D,    (float*) E,    (float*) Z, &ldz,    (float*) work, info, 1); }
      else if(   is_double<eT>::value) { coot_fortran(coot_dsteqr)(&compz, &n,   (double*) D,   (double*) E,   (double*) Z, &ldz,   (double*) work, info, 1); }
      else if( is_cx_float<eT>::value) { coot_fortran(coot_csteqr)(&compz, &n, (blas_cxd*) D, (blas_cxf*) E, (blas_cxf*) Z, &ldz, (blas_cxf*) work, info, 1); }
      else if(is_cx_double<eT>::value) { coot_fortran(coot_zsteqr)(&compz, &n, (blas_cxd*) D, (blas_cxd*) E, (blas_cxd*) Z, &ldz, (blas_cxd*) work, info, 1); }
      }
    #else
      {
      if     (    is_float<eT>::value) { coot_fortran(coot_ssteqr)(&compz, &n,    (float*) D,    (float*) E,    (float*) Z, &ldz,    (float*) work, info); }
      else if(   is_double<eT>::value) { coot_fortran(coot_dsteqr)(&compz, &n,   (double*) D,   (double*) E,   (double*) Z, &ldz,   (double*) work, info); }
      else if( is_cx_float<eT>::value) { coot_fortran(coot_csteqr)(&compz, &n, (blas_cxd*) D, (blas_cxf*) E, (blas_cxf*) Z, &ldz, (blas_cxf*) work, info); }
      else if(is_cx_double<eT>::value) { coot_fortran(coot_zsteqr)(&compz, &n, (blas_cxd*) D, (blas_cxd*) E, (blas_cxd*) Z, &ldz, (blas_cxd*) work, info); }
      }
    #endif
    }



  // compute 1-norm/Frobenius norm/inf norm of real symmetric tridiagonal matrix
  template<typename eT>
  inline
  eT
  lanst(const char norm, const blas_int n, const eT* D, const eT* E)
    {
    coot_type_check((is_supported_real_blas_type<eT>::value == false));

    #if defined(COOT_FORTRAN_FLOAT_WORKAROUND)
    if (is_float<eT>::value)
      {
      // We can't call slanst, because we are on a system where it may return
      // double instead of float---and we have defined the prototype as
      // returning a float.  Therefore, we have to use a by-hand
      // implementation...
      eT retval = eT(0);
      if (norm == 'M' || norm == 'm')
        {
        // Max-abs value of matrix.
        retval = std::abs(D[n - 1]);
        for (uword i = 0; i < n - 1; ++i)
          {
          retval = std::max(retval, std::abs(D[i]));
          retval = std::max(retval, std::abs(E[i]));
          }
        }
      else if (norm == '1' || norm == 'O' || norm == 'o' || norm == 'I' || norm == 'i')
        {
        // 1-norm: maximum column sum, or Inf-norm: maximum row sum.
        if (n == 1)
          {
          retval = std::abs(D[0]);
          }
        else
          {
          retval = std::max(std::abs(D[0]) + std::abs(E[0]),
                            std::abs(E[n - 2]) + std::abs(D[n - 1]));
          for (uword i = 1; i < n - 1; ++i)
            {
            retval = std::max(retval, std::abs(D[i]) + std::abs(E[i]) + std::abs(E[i - 1]));
            }
          }
        }
      else if (norm == 'F' || norm == 'f' || norm == 'E' || norm == 'e')
        {
        retval = (D[0] * D[0]);
        for (uword i = 0; i < n - 1; ++i)
          {
          retval += (D[i + 1] * D[i + 1]);
          retval += (E[i] * E[i]);
          }
        retval = std::sqrt(retval);
        }

      return retval;
      }
    #endif

    #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
      {
      if     ( is_float<eT>::value) { return coot_fortran(coot_slanst)(&norm, &n,  (float*) D,  (float*) E, 1); }
      else if(is_double<eT>::value) { return coot_fortran(coot_dlanst)(&norm, &n, (double*) D, (double*) E, 1); }
      }
    #else
      {
      if     ( is_float<eT>::value) { return coot_fortran(coot_slanst)(&norm, &n,  (float*) D,  (float*) E); }
      else if(is_double<eT>::value) { return coot_fortran(coot_dlanst)(&norm, &n, (double*) D, (double*) E); }
      }
    #endif
    }



  // reduce real symmetric matrix to tridiagonal form
  template<typename eT>
  inline
  void
  sytrd(const char uplo, const blas_int n, eT* A, const blas_int lda, eT* D, eT* E, eT* tau, eT* work, const blas_int lwork, blas_int* info)
    {
    coot_type_check((is_supported_real_blas_type<eT>::value == false));

    #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
      {
      if     ( is_float<eT>::value) { coot_fortran(coot_ssytrd)(&uplo, &n,  (float*) A, &lda,  (float*) D,  (float*) E,  (float*) tau,  (float*) work, &lwork, info, 1); }
      else if(is_double<eT>::value) { coot_fortran(coot_dsytrd)(&uplo, &n, (double*) A, &lda, (double*) D, (double*) E, (double*) tau, (double*) work, &lwork, info, 1); }
      }
    #else
      {
      if     ( is_float<eT>::value) { coot_fortran(coot_ssytrd)(&uplo, &n,  (float*) A, &lda,  (float*) D,  (float*) E,  (float*) tau,  (float*) work, &lwork, info); }
      else if(is_double<eT>::value) { coot_fortran(coot_dsytrd)(&uplo, &n, (double*) A, &lda, (double*) D, (double*) E, (double*) tau, (double*) work, &lwork, info); }
      }
    #endif
    }



  // force A and B to be stored prior to doing the addition of A and B
  template<typename eT>
  inline
  eT
  lamc3(const eT* A, const eT* B)
    {
    coot_type_check((is_supported_real_blas_type<eT>::value == false));

    #if defined(COOT_FORTRAN_FLOAT_WORKAROUND)
    if (is_float<eT>::value || is_cx_float<eT>::value)
      {
      // We can't call slamc3, because we are on a system where it may return
      // double instead of float---and we have defined the prototype as
      // returning a float.  Therefore, we have to use a by-hand
      // implementation...
      return (*A) + (*B);
      }
    #endif

    // No hidden arguments.
    if     ( is_float<eT>::value) { return coot_fortran(coot_slamc3)( (float*) A,  (float*) B); }
    else if(is_double<eT>::value) { return coot_fortran(coot_dlamc3)((double*) A, (double*) B); }
    }



  // compute the i'th updated eigenvalue of a symmetric rank-one modification to the diagonal matrix in d
  template<typename eT>
  inline
  void
  laed4(const blas_int n, const blas_int i, const eT* D, const eT* Z, eT* delta, const eT rho, eT* dlam, blas_int* info)
    {
    coot_type_check((is_supported_real_blas_type<eT>::value == false));

    // No hidden arguments.
    if     ( is_float<eT>::value) { coot_fortran(coot_slaed4)(&n, &i,  (const float*) D,  (const float*) Z,  (float*) delta,  (const float*) &rho,  (float*) dlam, info); }
    else if(is_double<eT>::value) { coot_fortran(coot_dlaed4)(&n, &i, (const double*) D, (const double*) Z, (double*) delta, (const double*) &rho, (double*) dlam, info); }
    }



  // create a permutation list to merge the elements of A into a single set
  template<typename eT>
  inline
  void
  lamrg(const blas_int n1, const blas_int n2, const eT* A, const blas_int dtrd1,const blas_int dtrd2, blas_int* index)
    {
    coot_type_check((is_supported_real_blas_type<eT>::value == false));

    // No hidden arguments.
    if     ( is_float<eT>::value) { coot_fortran(coot_slamrg)(&n1, &n2,  (const float*) A, &dtrd1, &dtrd2, index); }
    else if(is_double<eT>::value) { coot_fortran(coot_dlamrg)(&n1, &n2, (const double*) A, &dtrd1, &dtrd2, index); }
    }



  // compute all eigenvalues of symmetric tridiagonal matrix
  template<typename eT>
  inline
  void
  sterf(const blas_int n, eT* D, eT* E, blas_int* info)
    {
    coot_type_check((is_supported_real_blas_type<eT>::value == false));

    // No hidden arguments.
    if     ( is_float<eT>::value) { coot_fortran(coot_ssterf)(&n,  (float*) D,  (float*) E, info); }
    else if(is_double<eT>::value) { coot_fortran(coot_dsterf)(&n, (double*) D, (double*) E, info); }
    }



  // perform a series of row interchanges
  template<typename eT>
  inline
  void
  laswp(const blas_int n, eT* A, const blas_int lda, const blas_int k1, const blas_int k2, const blas_int* ipiv, const blas_int incx)
    {
    coot_type_check((is_supported_real_blas_type<eT>::value == false));

    // No hidden arguments.
    if     ( is_float<eT>::value) { coot_fortran(coot_slaswp)(&n,  (float*) A, &lda, &k1, &k2, ipiv, &incx); }
    else if(is_double<eT>::value) { coot_fortran(coot_dlaswp)(&n, (double*) A, &lda, &k1, &k2, ipiv, &incx); }
    }



  } // namespace lapack
