// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2017-2023 Ryan Curtin (https://www.ratml.org)
// Copyright 2008-2017 Conrad Sanderson (https://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
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



#if defined(COOT_BLAS_NOEXCEPT)
  #undef  COOT_NOEXCEPT
  #define COOT_NOEXCEPT noexcept
#else
  #undef  COOT_NOEXCEPT
  #define COOT_NOEXCEPT
#endif


#if !defined(COOT_BLAS_CAPITALS)

  #define coot_sgemm sgemm
  #define coot_dgemm dgemm
  #define coot_cgemm cgemm
  #define coot_zgemm zgemm

  #define coot_sgemv sgemv
  #define coot_dgemv dgemv
  #define coot_cgemv cgemv
  #define coot_zgemv zgemv

  #define coot_saxpy saxpy
  #define coot_daxpy daxpy
  #define coot_caxpy caxpy
  #define coot_zaxpy zaxpy

  #define coot_sscal sscal
  #define coot_dscal dscal
  #define coot_cscal cscal
  #define coot_zscal zscal

  #define coot_ssyrk ssyrk
  #define coot_dsyrk dsyrk

  #define coot_scopy scopy
  #define coot_dcopy dcopy
  #define coot_ccopy ccopy
  #define coot_zcopy zcopy

  #define coot_sswap sswap
  #define coot_dswap dswap
  #define coot_cswap cswap
  #define coot_zswap zswap

#else

  #define coot_sgemm SGEMM
  #define coot_dgemm DGEMM
  #define coot_cgemm CGEMM
  #define coot_zgemm ZGEMM

  #define coot_sgemv SGEMV
  #define coot_dgemv DGEMV
  #define coot_cgemv CGEMV
  #define coot_zgemv ZGEMV

  #define coot_saxpy SAXPY
  #define coot_daxpy DAXPY
  #define coot_caxpy CAXPY
  #define coot_zaxpy ZAXPY

  #define coot_sscal SSCAL
  #define coot_dscal DSCAL
  #define coot_cscal CSCAL
  #define coot_zscal ZSCAL

  #define coot_ssyrk SSYRK
  #define coot_dsyrk DSYRK

  #define coot_scopy SCOPY
  #define coot_dcopy DCOPY
  #define coot_ccopy CCOPY
  #define coot_zcopy ZCOPY

  #define coot_sswap SSWAP
  #define coot_dswap DSWAP
  #define coot_cswap CSWAP
  #define coot_zswap ZSWAP

#endif



extern "C"
  {
  // matrix-matrix multiplication
  #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
  void coot_fortran(coot_sgemm)(const char* transA, const char* transB, const blas_int* m, const blas_int* n, const blas_int* k, const float*     alpha, const float*   A, const blas_int* ldA, const float*    B, const blas_int* ldB, const float*    beta, float*    C, const blas_int* ldC, blas_len transA_len, blas_len transB_len) COOT_NOEXCEPT;
  void coot_fortran(coot_dgemm)(const char* transA, const char* transB, const blas_int* m, const blas_int* n, const blas_int* k, const double*   alpha, const double*   A, const blas_int* ldA, const double*   B, const blas_int* ldB, const double*   beta, double*   C, const blas_int* ldC, blas_len transA_len, blas_len transB_len) COOT_NOEXCEPT;
  void coot_fortran(coot_cgemm)(const char* transA, const char* transB, const blas_int* m, const blas_int* n, const blas_int* k, const blas_cxf* alpha, const blas_cxf* A, const blas_int* ldA, const blas_cxf* B, const blas_int* ldB, const blas_cxf* beta, blas_cxf* C, const blas_int* ldC, blas_len transA_len, blas_len transB_len) COOT_NOEXCEPT;
  void coot_fortran(coot_zgemm)(const char* transA, const char* transB, const blas_int* m, const blas_int* n, const blas_int* k, const blas_cxd* alpha, const blas_cxd* A, const blas_int* ldA, const blas_cxd* B, const blas_int* ldB, const blas_cxd* beta, blas_cxd* C, const blas_int* ldC, blas_len transA_len, blas_len transB_len) COOT_NOEXCEPT;
  #else
  void coot_fortran(coot_sgemm)(const char* transA, const char* transB, const blas_int* m, const blas_int* n, const blas_int* k, const float*    alpha, const float*    A, const blas_int* ldA, const float*    B, const blas_int* ldB, const float*    beta, float*    C, const blas_int* ldC) COOT_NOEXCEPT;
  void coot_fortran(coot_dgemm)(const char* transA, const char* transB, const blas_int* m, const blas_int* n, const blas_int* k, const double*   alpha, const double*   A, const blas_int* ldA, const double*   B, const blas_int* ldB, const double*   beta, double*   C, const blas_int* ldC) COOT_NOEXCEPT;
  void coot_fortran(coot_cgemm)(const char* transA, const char* transB, const blas_int* m, const blas_int* n, const blas_int* k, const blas_cxf* alpha, const blas_cxf* A, const blas_int* ldA, const blas_cxf* B, const blas_int* ldB, const blas_cxf* beta, blas_cxf* C, const blas_int* ldC) COOT_NOEXCEPT;
  void coot_fortran(coot_zgemm)(const char* transA, const char* transB, const blas_int* m, const blas_int* n, const blas_int* k, const blas_cxd* alpha, const blas_cxd* A, const blas_int* ldA, const blas_cxd* B, const blas_int* ldB, const blas_cxd* beta, blas_cxd* C, const blas_int* ldC) COOT_NOEXCEPT;
  #endif

  // matrix-vector multiplication
  #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
  void coot_fortran(coot_sgemv)(const char* transA, const blas_int* m, const blas_int* n, const float*    alpha, const float*    A, const blas_int* ldA, const float*    x, const blas_int* incx, const float*    beta, float*    y, const blas_int* incy, blas_len transA_len) COOT_NOEXCEPT;
  void coot_fortran(coot_dgemv)(const char* transA, const blas_int* m, const blas_int* n, const double*   alpha, const double*   A, const blas_int* ldA, const double*   x, const blas_int* incx, const double*   beta, double*   y, const blas_int* incy, blas_len transA_len) COOT_NOEXCEPT;
  void coot_fortran(coot_cgemv)(const char* transA, const blas_int* m, const blas_int* n, const blas_cxf* alpha, const blas_cxf* A, const blas_int* ldA, const blas_cxf* x, const blas_int* incx, const blas_cxf* beta, blas_cxf* y, const blas_int* incy, blas_len transA_len) COOT_NOEXCEPT;
  void coot_fortran(coot_zgemv)(const char* transA, const blas_int* m, const blas_int* n, const blas_cxd* alpha, const blas_cxd* A, const blas_int* ldA, const blas_cxd* x, const blas_int* incx, const blas_cxd* beta, blas_cxd* y, const blas_int* incy, blas_len transA_len) COOT_NOEXCEPT;
  #else
  void coot_fortran(coot_sgemv)(const char* transA, const blas_int* m, const blas_int* n, const float*    alpha, const float*    A, const blas_int* ldA, const float*    x, const blas_int* incx, const float*    beta, float*    y, const blas_int* incy) COOT_NOEXCEPT;
  void coot_fortran(coot_dgemv)(const char* transA, const blas_int* m, const blas_int* n, const double*   alpha, const double*   A, const blas_int* ldA, const double*   x, const blas_int* incx, const double*   beta, double*   y, const blas_int* incy) COOT_NOEXCEPT;
  void coot_fortran(coot_cgemv)(const char* transA, const blas_int* m, const blas_int* n, const blas_cxf* alpha, const blas_cxf* A, const blas_int* ldA, const blas_cxf* x, const blas_int* incx, const blas_cxf* beta, blas_cxf* y, const blas_int* incy) COOT_NOEXCEPT;
  void coot_fortran(coot_zgemv)(const char* transA, const blas_int* m, const blas_int* n, const blas_cxd* alpha, const blas_cxd* A, const blas_int* ldA, const blas_cxd* x, const blas_int* incx, const blas_cxd* beta, blas_cxd* y, const blas_int* incy) COOT_NOEXCEPT;
  #endif

  // scalar multiply + add
  void coot_fortran(coot_saxpy)(const blas_int* m, const float*    da, const float*    dx, const blas_int* incx, float*    dy, const blas_int* incy) COOT_NOEXCEPT;
  void coot_fortran(coot_daxpy)(const blas_int* m, const double*   da, const double*   dx, const blas_int* incx, double*   dy, const blas_int* incy) COOT_NOEXCEPT;
  void coot_fortran(coot_caxpy)(const blas_int* m, const blas_cxf* da, const blas_cxf* dx, const blas_int* incx, blas_cxf* dy, const blas_int* incy) COOT_NOEXCEPT;
  void coot_fortran(coot_zaxpy)(const blas_int* m, const blas_cxd* da, const blas_cxd* dx, const blas_int* incx, blas_cxd* dy, const blas_int* incy) COOT_NOEXCEPT;

  // scale vector by constant
  void coot_fortran(coot_sscal)(const blas_int* n, const float*    da, float*    dx, const blas_int* incx) COOT_NOEXCEPT;
  void coot_fortran(coot_dscal)(const blas_int* n, const double*   da, double*   dx, const blas_int* incx) COOT_NOEXCEPT;
  void coot_fortran(coot_cscal)(const blas_int* n, const blas_cxf* da, blas_cxf* dx, const blas_int* incx) COOT_NOEXCEPT;
  void coot_fortran(coot_zscal)(const blas_int* n, const blas_cxd* da, blas_cxd* dx, const blas_int* incx) COOT_NOEXCEPT;

  // symmetric rank-k a*A*A' + b*C
  #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
  void coot_fortran(coot_ssyrk)(const char* uplo, const char* transA, const blas_int* n, const blas_int* k, const  float* alpha, const  float* A, const blas_int* ldA, const  float* beta,  float* C, const blas_int* ldC, blas_len uplo_len, blas_len transA_len) COOT_NOEXCEPT;
  void coot_fortran(coot_dsyrk)(const char* uplo, const char* transA, const blas_int* n, const blas_int* k, const double* alpha, const double* A, const blas_int* ldA, const double* beta, double* C, const blas_int* ldC, blas_len uplo_len, blas_len transA_len) COOT_NOEXCEPT;
  #else
  void coot_fortran(coot_ssyrk)(const char* uplo, const char* transA, const blas_int* n, const blas_int* k, const  float* alpha, const  float* A, const blas_int* ldA, const  float* beta,  float* C, const blas_int* ldC) COOT_NOEXCEPT;
  void coot_fortran(coot_dsyrk)(const char* uplo, const char* transA, const blas_int* n, const blas_int* k, const double* alpha, const double* A, const blas_int* ldA, const double* beta, double* C, const blas_int* ldC) COOT_NOEXCEPT;
  #endif

  // copy a vector X to Y
  void coot_fortran(coot_scopy)(const blas_int* n, const float*    X, const blas_int* incx, float*    Y, const blas_int* incy) COOT_NOEXCEPT;
  void coot_fortran(coot_dcopy)(const blas_int* n, const double*   X, const blas_int* incx, double*   Y, const blas_int* incy) COOT_NOEXCEPT;
  void coot_fortran(coot_ccopy)(const blas_int* n, const blas_cxf* X, const blas_int* incx, blas_cxf* Y, const blas_int* incy) COOT_NOEXCEPT;
  void coot_fortran(coot_zcopy)(const blas_int* n, const blas_cxd* X, const blas_int* incx, blas_cxd* Y, const blas_int* incy) COOT_NOEXCEPT;

  // interchange two vectors
  void coot_fortran(coot_sswap)(const blas_int* n, float*    dx, const blas_int* incx, float*    dy, const blas_int* incy) COOT_NOEXCEPT;
  void coot_fortran(coot_dswap)(const blas_int* n, double*   dx, const blas_int* incx, double*   dy, const blas_int* incy) COOT_NOEXCEPT;
  void coot_fortran(coot_cswap)(const blas_int* n, blas_cxf* dx, const blas_int* incx, blas_cxf* dy, const blas_int* incy) COOT_NOEXCEPT;
  void coot_fortran(coot_zswap)(const blas_int* n, blas_cxd* dx, const blas_int* incx, blas_cxd* dy, const blas_int* incy) COOT_NOEXCEPT;
  }

#undef COOT_NOEXCEPT
