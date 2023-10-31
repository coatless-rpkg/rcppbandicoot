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



// This file contains wrappers for BLAS functions.
// These should be used preferentially to calling coot_fortran(coot_XXXXX),
// because the compiler may need extra hidden arguments for FORTRAN calls.
// (See the definition of COOT_USE_FORTRAN_HIDDEN_ARGS for more details.)

namespace blas
  {



  // matrix-matrix multiplication
  template<typename eT>
  inline
  void
  gemm(const char transA, const char transB, const blas_int m, const blas_int n, const blas_int k, const eT alpha, const eT* A, const blas_int lda, const eT* B, const blas_int ldb, const eT beta, eT* C, blas_int ldc)
    {
    coot_type_check((is_supported_blas_type<eT>::value == false));

    #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
      {
      if     (    is_float<eT>::value) { coot_fortran(coot_sgemm)(&transA, &transB, &m, &n, &k,    (const float*) &alpha,    (const float*) A, &lda,    (const float*) B, &ldb,    (const float*) &beta,    (float*) C, &ldc, 1, 1); }
      else if(   is_double<eT>::value) { coot_fortran(coot_dgemm)(&transA, &transB, &m, &n, &k,   (const double*) &alpha,   (const double*) A, &lda,   (const double*) B, &ldb,   (const double*) &beta,   (double*) C, &ldc, 1, 1); }
      else if( is_cx_float<eT>::value) { coot_fortran(coot_cgemm)(&transA, &transB, &m, &n, &k, (const blas_cxf*) &alpha, (const blas_cxf*) A, &lda, (const blas_cxf*) B, &ldb, (const blas_cxf*) &beta, (blas_cxf*) C, &ldc, 1, 1); }
      else if(is_cx_double<eT>::value) { coot_fortran(coot_zgemm)(&transA, &transB, &m, &n, &k, (const blas_cxd*) &alpha, (const blas_cxd*) A, &lda, (const blas_cxd*) B, &ldb, (const blas_cxd*) &beta, (blas_cxd*) C, &ldc, 1, 1); }
      }
    #else
      {
      if     (    is_float<eT>::value) { coot_fortran(coot_sgemm)(&transA, &transB, &m, &n, &k,    (const float*) &alpha,    (const float*) A, &lda,    (const float*) B, &ldb,    (const float*) &beta,    (float*) C, &ldc); }
      else if(   is_double<eT>::value) { coot_fortran(coot_dgemm)(&transA, &transB, &m, &n, &k,   (const double*) &alpha,   (const double*) A, &lda,   (const double*) B, &ldb,   (const double*) &beta,   (double*) C, &ldc); }
      else if( is_cx_float<eT>::value) { coot_fortran(coot_cgemm)(&transA, &transB, &m, &n, &k, (const blas_cxf*) &alpha, (const blas_cxf*) A, &lda, (const blas_cxf*) B, &ldb, (const blas_cxf*) &beta, (blas_cxf*) C, &ldc); }
      else if(is_cx_double<eT>::value) { coot_fortran(coot_zgemm)(&transA, &transB, &m, &n, &k, (const blas_cxd*) &alpha, (const blas_cxd*) A, &lda, (const blas_cxd*) B, &ldb, (const blas_cxd*) &beta, (blas_cxd*) C, &ldc); }
      }
    #endif
    }



  // matrix-vector multiplication
  template<typename eT>
  inline
  void
  gemv(const char transA, const blas_int m, const blas_int n, const eT alpha, eT* A, const blas_int lda, const eT* x, const blas_int incx, const eT beta, eT* y, const blas_int incy)
    {
    coot_type_check((is_supported_blas_type<eT>::value == false));

    #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
      {
      if     (    is_float<eT>::value) { coot_fortran(coot_sgemv)(&transA, &m, &n,    (const float*) &alpha,    (const float*) A, &lda,    (const float*) x, &incx,    (const float*) &beta,    (float*) y, &incy, 1); }
      else if(   is_double<eT>::value) { coot_fortran(coot_dgemv)(&transA, &m, &n,   (const double*) &alpha,   (const double*) A, &lda,   (const double*) x, &incx,   (const double*) &beta,   (double*) y, &incy, 1); }
      else if( is_cx_float<eT>::value) { coot_fortran(coot_cgemv)(&transA, &m, &n, (const blas_cxf*) &alpha, (const blas_cxf*) A, &lda, (const blas_cxf*) x, &incx, (const blas_cxf*) &beta, (blas_cxf*) y, &incy, 1); }
      else if(is_cx_double<eT>::value) { coot_fortran(coot_zgemv)(&transA, &m, &n, (const blas_cxd*) &alpha, (const blas_cxd*) A, &lda, (const blas_cxd*) x, &incx, (const blas_cxd*) &beta, (blas_cxd*) y, &incy, 1); }
      }
    #else
      {
      if     (    is_float<eT>::value) { coot_fortran(coot_sgemv)(&transA, &m, &n,    (const float*) &alpha,    (const float*) A, &lda,    (const float*) x, &incx,    (const float*) &beta,    (float*) y, &incy); }
      else if(   is_double<eT>::value) { coot_fortran(coot_dgemv)(&transA, &m, &n,   (const double*) &alpha,   (const double*) A, &lda,   (const double*) x, &incx,   (const double*) &beta,   (double*) y, &incy); }
      else if( is_cx_float<eT>::value) { coot_fortran(coot_cgemv)(&transA, &m, &n, (const blas_cxf*) &alpha, (const blas_cxf*) A, &lda, (const blas_cxf*) x, &incx, (const blas_cxf*) &beta, (blas_cxf*) y, &incy); }
      else if(is_cx_double<eT>::value) { coot_fortran(coot_zgemv)(&transA, &m, &n, (const blas_cxd*) &alpha, (const blas_cxd*) A, &lda, (const blas_cxd*) x, &incx, (const blas_cxd*) &beta, (blas_cxd*) y, &incy); }
      }
    #endif
    }



  // scalar multiply + add
  template<typename eT>
  inline
  void
  axpy(const blas_int n, const eT da, const eT* dx, blas_int incx, eT* dy, blas_int incy)
    {
    coot_type_check((is_supported_blas_type<eT>::value == false));

    // No hidden arguments so COOT_USE_FORTRAN_HIDDEN_ARGS is not needed.
    if     (    is_float<eT>::value) { coot_fortran(coot_saxpy)(&n,    (const float*) &da,    (const float*) dx, &incx,    (float*) dy, &incy); }
    else if(   is_double<eT>::value) { coot_fortran(coot_daxpy)(&n,   (const double*) &da,   (const double*) dx, &incx,   (double*) dy, &incy); }
    else if( is_cx_float<eT>::value) { coot_fortran(coot_caxpy)(&n, (const blas_cxf*) &da, (const blas_cxf*) dx, &incx, (blas_cxf*) dy, &incy); }
    else if(is_cx_double<eT>::value) { coot_fortran(coot_zaxpy)(&n, (const blas_cxd*) &da, (const blas_cxd*) dx, &incx, (blas_cxd*) dy, &incy); }
    }



  // scale vector by constant
  template<typename eT>
  inline
  void
  scal(const blas_int n, const eT da, eT* dx, blas_int incx)
    {
    coot_type_check((is_supported_blas_type<eT>::value == false));

    // No hidden arguments so COOT_USE_FORTRAN_HIDDEN_ARGS is not needed.
    if     (    is_float<eT>::value) { coot_fortran(coot_sscal)(&n,    (const float*) &da,    (float*) dx, &incx); }
    else if(   is_double<eT>::value) { coot_fortran(coot_dscal)(&n,   (const double*) &da,   (double*) dx, &incx); }
    else if( is_cx_float<eT>::value) { coot_fortran(coot_cscal)(&n, (const blas_cxf*) &da, (blas_cxf*) dx, &incx); }
    else if(is_cx_double<eT>::value) { coot_fortran(coot_zscal)(&n, (const blas_cxd*) &da, (blas_cxd*) dx, &incx); }
    }



  // symmetric rank-k a*A*A' + b*C
  template<typename eT>
  inline
  void
  syrk(const char uplo, const char transA, const blas_int n, const blas_int k, const eT alpha, const eT* A, const blas_int lda, const eT beta, eT* C, const blas_int ldc)
    {
    coot_type_check((is_supported_real_blas_type<eT>::value == false));

    #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
      {
      if     (    is_float<eT>::value) { coot_fortran(coot_ssyrk)(&uplo, &transA, &n, &k,  (const float*) &alpha,  (const float*) A, &lda,  (const float*) &beta,  (float*) C, &ldc, 1, 1); }
      else if(   is_double<eT>::value) { coot_fortran(coot_dsyrk)(&uplo, &transA, &n, &k, (const double*) &alpha, (const double*) A, &lda, (const double*) &beta, (double*) C, &ldc, 1, 1); }
      }
    #else
      {
      if     (    is_float<eT>::value) { coot_fortran(coot_ssyrk)(&uplo, &transA, &n, &k,  (const float*) &alpha,  (const float*) A, &lda,  (const float*) &beta,  (float*) C, &ldc); }
      else if(   is_double<eT>::value) { coot_fortran(coot_dsyrk)(&uplo, &transA, &n, &k, (const double*) &alpha, (const double*) A, &lda, (const double*) &beta, (double*) C, &ldc); }
      }
    #endif
    }



  // copy a vector X to Y
  template<typename eT>
  inline
  void
  copy(const blas_int n, const eT* X, const blas_int incx, eT* Y, const blas_int incy)
    {
    coot_type_check((is_supported_blas_type<eT>::value == false));

    // No hidden arguments so COOT_USE_FORTRAN_HIDDEN_ARGS is not needed.
    if     (    is_float<eT>::value) { coot_fortran(coot_scopy)(&n,    (const float*) X, &incx,    (float*) Y, &incy); }
    else if(   is_double<eT>::value) { coot_fortran(coot_dcopy)(&n,   (const double*) X, &incx,   (double*) Y, &incy); }
    else if( is_cx_float<eT>::value) { coot_fortran(coot_ccopy)(&n, (const blas_cxf*) X, &incx, (blas_cxf*) Y, &incy); }
    else if(is_cx_double<eT>::value) { coot_fortran(coot_zcopy)(&n, (const blas_cxd*) X, &incx, (blas_cxd*) Y, &incy); }
    }



  // interchange two vectors
  template<typename eT>
  inline
  void
  swap(const blas_int n, eT* dx, const blas_int incx, eT* dy, const blas_int incy)
    {
    coot_type_check((is_supported_blas_type<eT>::value == false));

    // No hidden arguments so COOT_USE_FORTRAN_HIDDEN_ARGS is not needed.
    if     (    is_float<eT>::value) { coot_fortran(coot_sswap)(&n,    (float*) dx, &incx,    (float*) dy, &incy); }
    else if(   is_double<eT>::value) { coot_fortran(coot_dswap)(&n,   (double*) dx, &incx,   (double*) dy, &incy); }
    else if( is_cx_float<eT>::value) { coot_fortran(coot_cswap)(&n, (blas_cxf*) dx, &incx, (blas_cxf*) dy, &incy); }
    else if(is_cx_double<eT>::value) { coot_fortran(coot_zswap)(&n, (blas_cxd*) dx, &incx, (blas_cxd*) dy, &incy); }
    }



  } // namespace blas
