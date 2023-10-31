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



#if defined(COOT_LAPACK_NOEXCEPT)
  #undef  COOT_NOEXCEPT
  #define COOT_NOEXCEPT noexcept
#else
  #undef  COOT_NOEXCEPT
  #define COOT_NOEXCEPT
#endif


#if !defined(COOT_BLAS_CAPITALS)

  #define coot_sgetrf sgetrf
  #define coot_dgetrf dgetrf
  #define coot_cgetrf cgetrf
  #define coot_zgetrf zgetrf

  #define coot_strtri strtri
  #define coot_dtrtri dtrtri
  #define coot_ctrtri ctrtri
  #define coot_ztrtri ztrtri

  #define coot_ssyevd ssyevd
  #define coot_dsyevd dsyevd

  #define coot_spotrf spotrf
  #define coot_dpotrf dpotrf
  #define coot_cpotrf cpotrf
  #define coot_zpotrf zpotrf

  #define coot_sgeqrf sgeqrf
  #define coot_dgeqrf dgeqrf
  #define coot_cgeqrf cgeqrf
  #define coot_zgeqrf zgeqrf

  #define coot_sorgqr sorgqr
  #define coot_dorgqr dorgqr

  #define coot_slange slange
  #define coot_dlange dlange
  #define coot_clange clange
  #define coot_zlange zlange

  #define coot_slarft slarft
  #define coot_dlarft dlarft
  #define coot_clarft clarft
  #define coot_zlarft zlarft

  #define coot_slarfg slarfg
  #define coot_dlarfg dlarfg
  #define coot_clarfg clarfg
  #define coot_zlarfg zlarfg

  #define coot_sgebrd sgebrd
  #define coot_dgebrd dgebrd
  #define coot_cgebrd cgebrd
  #define coot_zgebrd zgebrd

  #define coot_sormqr sormqr
  #define coot_dormqr dormqr

  #define coot_sormlq sormlq
  #define coot_dormlq dormlq

  #define coot_slacpy slacpy
  #define coot_dlacpy dlacpy
  #define coot_clacpy clacpy
  #define coot_zlacpy zlacpy

  #define coot_slaset slaset
  #define coot_dlaset dlaset
  #define coot_claset claset
  #define coot_zlaset zlaset

  #define coot_slarfb slarfb
  #define coot_dlarfb dlarfb
  #define coot_clarfb clarfb
  #define coot_zlarfb zlarfb

  #define coot_slamch slamch
  #define coot_dlamch dlamch

  #define coot_slascl slascl
  #define coot_dlascl dlascl
  #define coot_clascl clascl
  #define coot_zlascl zlascl

  #define coot_sbdsqr sbdsqr
  #define coot_dbdsqr dbdsqr
  #define coot_cbdsqr cbdsqr
  #define coot_zbdsqr zbdsqr

  #define coot_slaed2 slaed2
  #define coot_dlaed2 dlaed2

  #define coot_ssteqr ssteqr
  #define coot_dsteqr dsteqr
  #define coot_csteqr csteqr
  #define coot_zsteqr zsteqr

  #define coot_slanst slanst
  #define coot_dlanst dlanst

  #define coot_ssytrd ssytrd
  #define coot_dsytrd dsytrd

  #define coot_slamc3 slamc3
  #define coot_dlamc3 dlamc3

  #define coot_slaed4 slaed4
  #define coot_dlaed4 dlaed4

  #define coot_slamrg slamrg
  #define coot_dlamrg dlamrg

  #define coot_ssterf ssterf
  #define coot_dsterf dsterf

  #define coot_slaswp slaswp
  #define coot_dlaswp dlaswp

#else

  #define coot_sgetrf SGETRF
  #define coot_dgetrf DGETRF
  #define coot_cgetrf CGETRF
  #define coot_zgetrf ZGETRF

  #define coot_strtri STRTRI
  #define coot_dtrtri DTRTRI
  #define coot_ctrtri CTRTRI
  #define coot_ztrtri ZTRTRI

  #define coot_ssyevd SSYEVD
  #define coot_dsyevd DSYEVD

  #define coot_spotrf SPOTRF
  #define coot_dpotrf DPOTRF
  #define coot_cpotrf CPOTRF
  #define coot_zpotrf ZPOTRF

  #define coot_sgeqrf SGEQRF
  #define coot_dgeqrf DGEQRF
  #define coot_cgeqrf CGEQRF
  #define coot_zgeqrf ZGEQRF

  #define coot_sorgqr SORGQR
  #define coot_dorgqr DORGQR

  #define coot_slange SLANGE
  #define coot_dlange DLANGE
  #define coot_clange CLANGE
  #define coot_zlange ZLANGE

  #define coot_slarft SLARFT
  #define coot_dlarft DLARFT
  #define coot_clarft CLARFT
  #define coot_zlarft ZLARFT

  #define coot_slarfg SLARFG
  #define coot_dlarfg DLARFG
  #define coot_clarfg CLARFG
  #define coot_zlarfg ZLARFG

  #define coot_sgebrd SGEBRD
  #define coot_dgebrd DGEBRD
  #define coot_cgebrd CGEBRD
  #define coot_zgebrd ZGEBRD

  #define coot_sormqr SORMQR
  #define coot_dormqr DORMQR

  #define coot_sormlq SORMLQ
  #define coot_dormlq DORMLQ

  #define coot_slacpy SLACPY
  #define coot_dlacpy DLACPY
  #define coot_clacpy CLACPY
  #define coot_zlacpy ZLACPY

  #define coot_slaset SLASET
  #define coot_dlaset DLASET
  #define coot_claset CLASET
  #define coot_zlaset ZLASET

  #define coot_slarfb SLARFB
  #define coot_dlarfb DLARFB
  #define coot_clarfb CLARFB
  #define coot_zlarfb ZLARFB

  #define coot_slamch SLAMCH
  #define coot_dlamch DLAMCH

  #define coot_slascl SLASCL
  #define coot_dlascl DLASCL
  #define coot_clascl CLASCL
  #define coot_zlascl ZLASCL

  #define coot_sbdsqr SBDSQR
  #define coot_dbdsqr DBDSQR
  #define coot_cbdsqr CBDSQR
  #define coot_zbdsqr ZBDSQR

  #define coot_slaed2 SLAED2
  #define coot_dlaed2 DLAED2

  #define coot_ssteqr SSTEQR
  #define coot_dsteqr DSTEQR
  #define coot_csteqr CSTEQR
  #define coot_zsteqr ZSTEQR

  #define coot_slanst SLANST
  #define coot_dlanst DLANST

  #define coot_ssytrd SSYTRD
  #define coot_dsytrd DSYTRD

  #define coot_slamc3 SLAMC3
  #define coot_dlamc3 DLAMC3

  #define coot_slaed4 SLAED4
  #define coot_dlaed4 DLAED4

  #define coot_slamrg SLAMRG
  #define coot_dlamrg DLAMRG

  #define coot_ssterf SSTERF
  #define coot_dsterf DSTERF

  #define coot_slaswp SLASWP
  #define coot_dlaswp DLASWP

#endif



extern "C"
  {
  // LU factorisation
  void coot_fortran(coot_sgetrf)(const blas_int* m, const blas_int* n,    float* a, const blas_int* lda, blas_int* ipiv, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_dgetrf)(const blas_int* m, const blas_int* n,   double* a, const blas_int* lda, blas_int* ipiv, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_cgetrf)(const blas_int* m, const blas_int* n, blas_cxf* a, const blas_int* lda, blas_int* ipiv, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_zgetrf)(const blas_int* m, const blas_int* n, blas_cxd* a, const blas_int* lda, blas_int* ipiv, blas_int* info) COOT_NOEXCEPT;

  // matrix inversion (triangular matrices)
  #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
  void coot_fortran(coot_strtri)(const char* uplo, const char* diag, const blas_int* n,    float* a, const blas_int* lda, blas_int* info, blas_len uplo_len, blas_len diag_len) COOT_NOEXCEPT;
  void coot_fortran(coot_dtrtri)(const char* uplo, const char* diag, const blas_int* n,   double* a, const blas_int* lda, blas_int* info, blas_len uplo_len, blas_len diag_len) COOT_NOEXCEPT;
  void coot_fortran(coot_ctrtri)(const char* uplo, const char* diag, const blas_int* n, blas_cxf* a, const blas_int* lda, blas_int* info, blas_len uplo_len, blas_len diag_len) COOT_NOEXCEPT;
  void coot_fortran(coot_ztrtri)(const char* uplo, const char* diag, const blas_int* n, blas_cxd* a, const blas_int* lda, blas_int* info, blas_len uplo_len, blas_len diag_len) COOT_NOEXCEPT;
  #else
  void coot_fortran(coot_strtri)(const char* uplo, const char* diag, const blas_int* n,    float* a, const blas_int* lda, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_dtrtri)(const char* uplo, const char* diag, const blas_int* n,   double* a, const blas_int* lda, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_ctrtri)(const char* uplo, const char* diag, const blas_int* n, blas_cxf* a, const blas_int* lda, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_ztrtri)(const char* uplo, const char* diag, const blas_int* n, blas_cxd* a, const blas_int* lda, blas_int* info) COOT_NOEXCEPT;
  #endif

  // eigen decomposition of symmetric real matrices by divide and conquer
  #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
  void coot_fortran(coot_ssyevd)(const char* jobz, const char* uplo, const blas_int* n,  float* a, const blas_int* lda,  float* w,  float* work, const blas_int* lwork, blas_int* iwork, const blas_int* liwork, blas_int* info, blas_len jobz_len, blas_len uplo_len) COOT_NOEXCEPT;
  void coot_fortran(coot_dsyevd)(const char* jobz, const char* uplo, const blas_int* n, double* a, const blas_int* lda, double* w, double* work, const blas_int* lwork, blas_int* iwork, const blas_int* liwork, blas_int* info, blas_len jobz_len, blas_len uplo_len) COOT_NOEXCEPT;
  #else
  void coot_fortran(coot_ssyevd)(const char* jobz, const char* uplo, const blas_int* n,  float* a, const blas_int* lda,  float* w,  float* work, const blas_int* lwork, blas_int* iwork, const blas_int* liwork, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_dsyevd)(const char* jobz, const char* uplo, const blas_int* n, double* a, const blas_int* lda, double* w, double* work, const blas_int* lwork, blas_int* iwork, const blas_int* liwork, blas_int* info) COOT_NOEXCEPT;
  #endif

  // Cholesky decomposition
  #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
  void coot_fortran(coot_spotrf)(const char* uplo, const blas_int* n,    float* a, const blas_int* lda, blas_int* info, blas_len uplo_len) COOT_NOEXCEPT;
  void coot_fortran(coot_dpotrf)(const char* uplo, const blas_int* n,   double* a, const blas_int* lda, blas_int* info, blas_len uplo_len) COOT_NOEXCEPT;
  void coot_fortran(coot_cpotrf)(const char* uplo, const blas_int* n, blas_cxf* a, const blas_int* lda, blas_int* info, blas_len uplo_len) COOT_NOEXCEPT;
  void coot_fortran(coot_zpotrf)(const char* uplo, const blas_int* n, blas_cxd* a, const blas_int* lda, blas_int* info, blas_len uplo_len) COOT_NOEXCEPT;
  #else
  void coot_fortran(coot_spotrf)(const char* uplo, const blas_int* n,    float* a, const blas_int* lda, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_dpotrf)(const char* uplo, const blas_int* n,   double* a, const blas_int* lda, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_cpotrf)(const char* uplo, const blas_int* n, blas_cxf* a, const blas_int* lda, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_zpotrf)(const char* uplo, const blas_int* n, blas_cxd* a, const blas_int* lda, blas_int* info) COOT_NOEXCEPT;
  #endif

  // QR decomposition
  void coot_fortran(coot_sgeqrf)(const blas_int* m, const blas_int* n,    float* a, const blas_int* lda,    float* tau,    float* work, const blas_int* lwork, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_dgeqrf)(const blas_int* m, const blas_int* n,   double* a, const blas_int* lda,   double* tau,   double* work, const blas_int* lwork, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_cgeqrf)(const blas_int* m, const blas_int* n, blas_cxf* a, const blas_int* lda, blas_cxf* tau, blas_cxf* work, const blas_int* lwork, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_zgeqrf)(const blas_int* m, const blas_int* n, blas_cxd* a, const blas_int* lda, blas_cxd* tau, blas_cxd* work, const blas_int* lwork, blas_int* info) COOT_NOEXCEPT;

  // Q matrix calculation from QR decomposition (real matrices)
  void coot_fortran(coot_sorgqr)(const blas_int* m, const blas_int* n, const blas_int* k,  float* a, const blas_int* lda,  float* tau,  float* work, const blas_int* lwork, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_dorgqr)(const blas_int* m, const blas_int* n, const blas_int* k, double* a, const blas_int* lda, double* tau, double* work, const blas_int* lwork, blas_int* info) COOT_NOEXCEPT;

  // 1-norm
  #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
  float  coot_fortran(coot_slange)(const char* norm, const blas_int* m, const blas_int* n,    float* a, const blas_int* lda,  float* work, blas_len norm_len) COOT_NOEXCEPT;
  double coot_fortran(coot_dlange)(const char* norm, const blas_int* m, const blas_int* n,   double* a, const blas_int* lda, double* work, blas_len norm_len) COOT_NOEXCEPT;
  float  coot_fortran(coot_clange)(const char* norm, const blas_int* m, const blas_int* n, blas_cxf* a, const blas_int* lda,  float* work, blas_len norm_len) COOT_NOEXCEPT;
  double coot_fortran(coot_zlange)(const char* norm, const blas_int* m, const blas_int* n, blas_cxd* a, const blas_int* lda, double* work, blas_len norm_len) COOT_NOEXCEPT;
  #else
  float  coot_fortran(coot_slange)(const char* norm, const blas_int* m, const blas_int* n,    float* a, const blas_int* lda,  float* work) COOT_NOEXCEPT;
  double coot_fortran(coot_dlange)(const char* norm, const blas_int* m, const blas_int* n,   double* a, const blas_int* lda, double* work) COOT_NOEXCEPT;
  float  coot_fortran(coot_clange)(const char* norm, const blas_int* m, const blas_int* n, blas_cxf* a, const blas_int* lda,  float* work) COOT_NOEXCEPT;
  double coot_fortran(coot_zlange)(const char* norm, const blas_int* m, const blas_int* n, blas_cxd* a, const blas_int* lda, double* work) COOT_NOEXCEPT;
  #endif

  // triangular factor of block reflector
  #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
  void coot_fortran(coot_slarft)(const char* direct, const char* storev, const blas_int* n, const blas_int* k, float*    v, const blas_int* ldv, const float*    tau, float*    t, const blas_int* ldt, blas_len direct_len, blas_len storev_len) COOT_NOEXCEPT;
  void coot_fortran(coot_dlarft)(const char* direct, const char* storev, const blas_int* n, const blas_int* k, double*   v, const blas_int* ldv, const double*   tau, double*   t, const blas_int* ldt, blas_len direct_len, blas_len storev_len) COOT_NOEXCEPT;
  void coot_fortran(coot_clarft)(const char* direct, const char* storev, const blas_int* n, const blas_int* k, blas_cxf* v, const blas_int* ldv, const blas_cxf* tau, blas_cxf* t, const blas_int* ldt, blas_len direct_len, blas_len storev_len) COOT_NOEXCEPT;
  void coot_fortran(coot_zlarft)(const char* direct, const char* storev, const blas_int* n, const blas_int* k, blas_cxd* v, const blas_int* ldv, const blas_cxd* tau, blas_cxd* t, const blas_int* ldt, blas_len direct_len, blas_len storev_len) COOT_NOEXCEPT;
  #else
  void coot_fortran(coot_slarft)(const char* direct, const char* storev, const blas_int* n, const blas_int* k, float*    v, const blas_int* ldv, const float*    tau, float*    t, const blas_int* ldt) COOT_NOEXCEPT;
  void coot_fortran(coot_dlarft)(const char* direct, const char* storev, const blas_int* n, const blas_int* k, double*   v, const blas_int* ldv, const double*   tau, double*   t, const blas_int* ldt) COOT_NOEXCEPT;
  void coot_fortran(coot_clarft)(const char* direct, const char* storev, const blas_int* n, const blas_int* k, blas_cxf* v, const blas_int* ldv, const blas_cxf* tau, blas_cxf* t, const blas_int* ldt) COOT_NOEXCEPT;
  void coot_fortran(coot_zlarft)(const char* direct, const char* storev, const blas_int* n, const blas_int* k, blas_cxd* v, const blas_int* ldv, const blas_cxd* tau, blas_cxd* t, const blas_int* ldt) COOT_NOEXCEPT;
  #endif

  // generate an elementary reflector
  void coot_fortran(coot_slarfg)(const blas_int* n, float*    alpha, float*    x, const blas_int* incx, float*    tau) COOT_NOEXCEPT;
  void coot_fortran(coot_dlarfg)(const blas_int* n, double*   alpha, double*   x, const blas_int* incx, double*   tau) COOT_NOEXCEPT;
  void coot_fortran(coot_clarfg)(const blas_int* n, blas_cxf* alpha, blas_cxf* x, const blas_int* incx, blas_cxf* tau) COOT_NOEXCEPT;
  void coot_fortran(coot_zlarfg)(const blas_int* n, blas_cxd* alpha, blas_cxd* x, const blas_int* incx, blas_cxd* tau) COOT_NOEXCEPT;

  // reduce a general matrix to bidiagonal form
  void coot_fortran(coot_sgebrd)(const blas_int* m, const blas_int* n, float*    a, const blas_int* lda, float*    d, float*    e, float*    tauq, float*    taup, float*    work, const blas_int* lwork, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_dgebrd)(const blas_int* m, const blas_int* n, double*   a, const blas_int* lda, double*   d, double*   e, double*   tauq, double*   taup, double*   work, const blas_int* lwork, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_cgebrd)(const blas_int* m, const blas_int* n, blas_cxf* a, const blas_int* lda, blas_cxf* d, blas_cxf* e, blas_cxf* tauq, blas_cxf* taup, blas_cxf* work, const blas_int* lwork, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_zgebrd)(const blas_int* m, const blas_int* n, blas_cxd* a, const blas_int* lda, blas_cxd* d, blas_cxd* e, blas_cxd* tauq, blas_cxd* taup, blas_cxd* work, const blas_int* lwork, blas_int* info) COOT_NOEXCEPT;

  // overwrite matrix with geqrf-generated orthogonal transformation
  #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
  void coot_fortran(coot_sormqr)(const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const float*  A, const blas_int* lda, const float*  tau, float*  C, const blas_int* ldc, float*  work, const blas_int* lwork, blas_int* info, blas_len side_len, blas_len trans_len) COOT_NOEXCEPT;
  void coot_fortran(coot_dormqr)(const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const double* A, const blas_int* lda, const double* tau, double* C, const blas_int* ldc, double* work, const blas_int* lwork, blas_int* info, blas_len side_len, blas_len trans_len) COOT_NOEXCEPT;
  #else
  void coot_fortran(coot_sormqr)(const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const float*  A, const blas_int* lda, const float*  tau, float*  C, const blas_int* ldc, float*  work, const blas_int* lwork, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_dormqr)(const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const double* A, const blas_int* lda, const double* tau, double* C, const blas_int* ldc, double* work, const blas_int* lwork, blas_int* info) COOT_NOEXCEPT;
  #endif

  // overwrite matrix with gelqf-generated orthogonal matrix
  #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
  void coot_fortran(coot_sormlq)(const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const float*  A, const blas_int* lda, const float*  tau, float*  C, const blas_int* ldc, float*  work, const blas_int* lwork, blas_int* info, blas_len side_len, blas_len trans_len) COOT_NOEXCEPT;
  void coot_fortran(coot_dormlq)(const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const double* A, const blas_int* lda, const double* tau, double* C, const blas_int* ldc, double* work, const blas_int* lwork, blas_int* info, blas_len side_len, blas_len trans_len) COOT_NOEXCEPT;
  #else
  void coot_fortran(coot_sormlq)(const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const float*  A, const blas_int* lda, const float*  tau, float*  C, const blas_int* ldc, float*  work, const blas_int* lwork, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_dormlq)(const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const double* A, const blas_int* lda, const double* tau, double* C, const blas_int* ldc, double* work, const blas_int* lwork, blas_int* info) COOT_NOEXCEPT;
  #endif

  // copy all or part of one 2d array to another
  #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
  void coot_fortran(coot_slacpy)(const char* uplo, const blas_int* m, const blas_int* n, const float*    A, const blas_int* lda, float*    B, const blas_int* ldb, blas_len uplo_len) COOT_NOEXCEPT;
  void coot_fortran(coot_dlacpy)(const char* uplo, const blas_int* m, const blas_int* n, const double*   A, const blas_int* lda, double*   B, const blas_int* ldb, blas_len uplo_len) COOT_NOEXCEPT;
  void coot_fortran(coot_clacpy)(const char* uplo, const blas_int* m, const blas_int* n, const blas_cxf* A, const blas_int* lda, blas_cxf* B, const blas_int* ldb, blas_len uplo_len) COOT_NOEXCEPT;
  void coot_fortran(coot_zlacpy)(const char* uplo, const blas_int* m, const blas_int* n, const blas_cxd* A, const blas_int* lda, blas_cxd* B, const blas_int* ldb, blas_len uplo_len) COOT_NOEXCEPT;
  #else
  void coot_fortran(coot_slacpy)(const char* uplo, const blas_int* m, const blas_int* n, const float*    A, const blas_int* lda, float*    B, const blas_int* ldb) COOT_NOEXCEPT;
  void coot_fortran(coot_dlacpy)(const char* uplo, const blas_int* m, const blas_int* n, const double*   A, const blas_int* lda, double*   B, const blas_int* ldb) COOT_NOEXCEPT;
  void coot_fortran(coot_clacpy)(const char* uplo, const blas_int* m, const blas_int* n, const blas_cxf* A, const blas_int* lda, blas_cxf* B, const blas_int* ldb) COOT_NOEXCEPT;
  void coot_fortran(coot_zlacpy)(const char* uplo, const blas_int* m, const blas_int* n, const blas_cxd* A, const blas_int* lda, blas_cxd* B, const blas_int* ldb) COOT_NOEXCEPT;
  #endif

  // initialize a matrix with different elements on and off the diagonal
  #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
  void coot_fortran(coot_slaset)(const char* uplo, const blas_int* m, const blas_int* n, const float*    alpha, const float*    beta, float*    A, const blas_int* lda, blas_len uplo_len) COOT_NOEXCEPT;
  void coot_fortran(coot_dlaset)(const char* uplo, const blas_int* m, const blas_int* n, const double*   alpha, const double*   beta, double*   A, const blas_int* lda, blas_len uplo_len) COOT_NOEXCEPT;
  void coot_fortran(coot_claset)(const char* uplo, const blas_int* m, const blas_int* n, const blas_cxf* alpha, const blas_cxf* beta, blas_cxf* A, const blas_int* lda, blas_len uplo_len) COOT_NOEXCEPT;
  void coot_fortran(coot_zlaset)(const char* uplo, const blas_int* m, const blas_int* n, const blas_cxd* alpha, const blas_cxd* beta, blas_cxd* A, const blas_int* lda, blas_len uplo_len) COOT_NOEXCEPT;
  #else
  void coot_fortran(coot_slaset)(const char* uplo, const blas_int* m, const blas_int* n, const float*    alpha, const float*    beta, float*    A, const blas_int* lda) COOT_NOEXCEPT;
  void coot_fortran(coot_dlaset)(const char* uplo, const blas_int* m, const blas_int* n, const double*   alpha, const double*   beta, double*   A, const blas_int* lda) COOT_NOEXCEPT;
  void coot_fortran(coot_claset)(const char* uplo, const blas_int* m, const blas_int* n, const blas_cxf* alpha, const blas_cxf* beta, blas_cxf* A, const blas_int* lda) COOT_NOEXCEPT;
  void coot_fortran(coot_zlaset)(const char* uplo, const blas_int* m, const blas_int* n, const blas_cxd* alpha, const blas_cxd* beta, blas_cxd* A, const blas_int* lda) COOT_NOEXCEPT;
  #endif

  // apply block reflector to general rectangular matrix
  #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
  void coot_fortran(coot_slarfb)(const char* side, const char* trans, const char* direct, const char* storev, const blas_int* M, const blas_int* N, const blas_int* K, const float*    V, const blas_int* ldv, const float*    T, const blas_int* ldt, float*    C, const blas_int* ldc, float*    work, const blas_int* ldwork, blas_len side_len, blas_len trans_len, blas_len direct_len, blas_len storev_len) COOT_NOEXCEPT;
  void coot_fortran(coot_dlarfb)(const char* side, const char* trans, const char* direct, const char* storev, const blas_int* M, const blas_int* N, const blas_int* K, const double*   V, const blas_int* ldv, const double*   T, const blas_int* ldt, double*   C, const blas_int* ldc, double*   work, const blas_int* ldwork, blas_len side_len, blas_len trans_len, blas_len direct_len, blas_len storev_len) COOT_NOEXCEPT;
  void coot_fortran(coot_clarfb)(const char* side, const char* trans, const char* direct, const char* storev, const blas_int* M, const blas_int* N, const blas_int* K, const blas_cxf* V, const blas_int* ldv, const blas_cxf* T, const blas_int* ldt, blas_cxf* C, const blas_int* ldc, blas_cxf* work, const blas_int* ldwork, blas_len side_len, blas_len trans_len, blas_len direct_len, blas_len storev_len) COOT_NOEXCEPT;
  void coot_fortran(coot_zlarfb)(const char* side, const char* trans, const char* direct, const char* storev, const blas_int* M, const blas_int* N, const blas_int* K, const blas_cxd* V, const blas_int* ldv, const blas_cxd* T, const blas_int* ldt, blas_cxd* C, const blas_int* ldc, blas_cxd* work, const blas_int* ldwork, blas_len side_len, blas_len trans_len, blas_len direct_len, blas_len storev_len) COOT_NOEXCEPT;
  #else
  void coot_fortran(coot_slarfb)(const char* side, const char* trans, const char* direct, const char* storev, const blas_int* M, const blas_int* N, const blas_int* K, const float*    V, const blas_int* ldv, const float*    T, const blas_int* ldt, float*    C, const blas_int* ldc, float*    work, const blas_int* ldwork) COOT_NOEXCEPT;
  void coot_fortran(coot_dlarfb)(const char* side, const char* trans, const char* direct, const char* storev, const blas_int* M, const blas_int* N, const blas_int* K, const double*   V, const blas_int* ldv, const double*   T, const blas_int* ldt, double*   C, const blas_int* ldc, double*   work, const blas_int* ldwork) COOT_NOEXCEPT;
  void coot_fortran(coot_clarfb)(const char* side, const char* trans, const char* direct, const char* storev, const blas_int* M, const blas_int* N, const blas_int* K, const blas_cxf* V, const blas_int* ldv, const blas_cxf* T, const blas_int* ldt, blas_cxf* C, const blas_int* ldc, blas_cxf* work, const blas_int* ldwork) COOT_NOEXCEPT;
  void coot_fortran(coot_zlarfb)(const char* side, const char* trans, const char* direct, const char* storev, const blas_int* M, const blas_int* N, const blas_int* K, const blas_cxd* V, const blas_int* ldv, const blas_cxd* T, const blas_int* ldt, blas_cxd* C, const blas_int* ldc, blas_cxd* work, const blas_int* ldwork) COOT_NOEXCEPT;
  #endif

  // get machine parameters
  #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
  float  coot_fortran(coot_slamch)(const char* cmach, blas_len cmach_len) COOT_NOEXCEPT;
  double coot_fortran(coot_dlamch)(const char* cmach, blas_len cmach_len) COOT_NOEXCEPT;
  #else
  float  coot_fortran(coot_slamch)(const char* cmach) COOT_NOEXCEPT;
  double coot_fortran(coot_dlamch)(const char* cmach) COOT_NOEXCEPT;
  #endif

  // scale matrix by a scalar
  #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
  void coot_fortran(coot_slascl)(const char* type, const blas_int* kl, const blas_int* ku, const float*    cfrom, const float*    cto, const blas_int* m, const blas_int* n, float*    a, const blas_int* lda, blas_int* info, blas_len type_len) COOT_NOEXCEPT;
  void coot_fortran(coot_dlascl)(const char* type, const blas_int* kl, const blas_int* ku, const double*   cfrom, const double*   cto, const blas_int* m, const blas_int* n, double*   a, const blas_int* lda, blas_int* info, blas_len type_len) COOT_NOEXCEPT;
  void coot_fortran(coot_clascl)(const char* type, const blas_int* kl, const blas_int* ku, const blas_cxf* cfrom, const blas_cxf* cto, const blas_int* m, const blas_int* n, blas_cxf* a, const blas_int* lda, blas_int* info, blas_len type_len) COOT_NOEXCEPT;
  void coot_fortran(coot_zlascl)(const char* type, const blas_int* kl, const blas_int* ku, const blas_cxd* cfrom, const blas_cxd* cto, const blas_int* m, const blas_int* n, blas_cxd* a, const blas_int* lda, blas_int* info, blas_len type_len) COOT_NOEXCEPT;
  #else
  void coot_fortran(coot_slascl)(const char* type, const blas_int* kl, const blas_int* ku, const float*    cfrom, const float*    cto, const blas_int* m, const blas_int* n, float*    a, const blas_int* lda, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_dlascl)(const char* type, const blas_int* kl, const blas_int* ku, const double*   cfrom, const double*   cto, const blas_int* m, const blas_int* n, double*   a, const blas_int* lda, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_clascl)(const char* type, const blas_int* kl, const blas_int* ku, const blas_cxf* cfrom, const blas_cxf* cto, const blas_int* m, const blas_int* n, blas_cxf* a, const blas_int* lda, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_zlascl)(const char* type, const blas_int* kl, const blas_int* ku, const blas_cxd* cfrom, const blas_cxd* cto, const blas_int* m, const blas_int* n, blas_cxd* a, const blas_int* lda, blas_int* info) COOT_NOEXCEPT;
  #endif

  // compute singular values of bidiagonal matrix
  #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
  void coot_fortran(coot_sbdsqr)(const char* uplo, const blas_int* n, const blas_int* ncvt, const blas_int* nru, const blas_int* ncc, float*  d, float*  e, float*    vt, const blas_int* ldvt, float*    u, const blas_int* ldu, float*    c, const blas_int* ldc, float*  work, blas_int* info, blas_len uplo_len) COOT_NOEXCEPT;
  void coot_fortran(coot_dbdsqr)(const char* uplo, const blas_int* n, const blas_int* ncvt, const blas_int* nru, const blas_int* ncc, double* d, double* e, double*   vt, const blas_int* ldvt, double*   u, const blas_int* ldu, double*   c, const blas_int* ldc, double* work, blas_int* info, blas_len uplo_len) COOT_NOEXCEPT;
  void coot_fortran(coot_cbdsqr)(const char* uplo, const blas_int* n, const blas_int* ncvt, const blas_int* nru, const blas_int* ncc, float*  d, float*  e, blas_cxf* vt, const blas_int* ldvt, blas_cxf* u, const blas_int* ldu, blas_cxf* c, const blas_int* ldc, float*  work, blas_int* info, blas_len uplo_len) COOT_NOEXCEPT;
  void coot_fortran(coot_zbdsqr)(const char* uplo, const blas_int* n, const blas_int* ncvt, const blas_int* nru, const blas_int* ncc, double* d, double* e, blas_cxd* vt, const blas_int* ldvt, blas_cxd* u, const blas_int* ldu, blas_cxd* c, const blas_int* ldc, double* work, blas_int* info, blas_len uplo_len) COOT_NOEXCEPT;
  #else
  void coot_fortran(coot_sbdsqr)(const char* uplo, const blas_int* n, const blas_int* ncvt, const blas_int* nru, const blas_int* ncc, float*  d, float*  e, float*    vt, const blas_int* ldvt, float*    u, const blas_int* ldu, float*    c, const blas_int* ldc, float*  work, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_dbdsqr)(const char* uplo, const blas_int* n, const blas_int* ncvt, const blas_int* nru, const blas_int* ncc, double* d, double* e, double*   vt, const blas_int* ldvt, double*   u, const blas_int* ldu, double*   c, const blas_int* ldc, double* work, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_cbdsqr)(const char* uplo, const blas_int* n, const blas_int* ncvt, const blas_int* nru, const blas_int* ncc, float*  d, float*  e, blas_cxf* vt, const blas_int* ldvt, blas_cxf* u, const blas_int* ldu, blas_cxf* c, const blas_int* ldc, float*  work, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_zbdsqr)(const char* uplo, const blas_int* n, const blas_int* ncvt, const blas_int* nru, const blas_int* ncc, double* d, double* e, blas_cxd* vt, const blas_int* ldvt, blas_cxd* u, const blas_int* ldu, blas_cxd* c, const blas_int* ldc, double* work, blas_int* info) COOT_NOEXCEPT;
  #endif

  // merges two sets of eigenvalues together into a single sorted set
  void coot_fortran(coot_slaed2)(blas_int* k, const blas_int* n, const blas_int* n1, float*  D, float*  Q, const blas_int* ldq, blas_int* indxq, float*  rho, const float*  Z, float*  dlamda, float*  W, float*  Q2, blas_int* indx, blas_int* indxc, blas_int* indxp, blas_int* coltyp, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_dlaed2)(blas_int* k, const blas_int* n, const blas_int* n1, double* D, double* Q, const blas_int* ldq, blas_int* indxq, double* rho, const double* Z, double* dlamda, double* W, double* Q2, blas_int* indx, blas_int* indxc, blas_int* indxp, blas_int* coltyp, blas_int* info) COOT_NOEXCEPT;

  // compute all eigenvalues (and optionally eigenvectors) of symmetric tridiagonal matrix
  #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
  void coot_fortran(coot_ssteqr)(const char* compz, const blas_int* n, float*    D, float*    E, float*    Z, const blas_int* ldz, float*    work, blas_int* info, blas_len compz_len) COOT_NOEXCEPT;
  void coot_fortran(coot_dsteqr)(const char* compz, const blas_int* n, double*   D, double*   E, double*   Z, const blas_int* ldz, double*   work, blas_int* info, blas_len compz_len) COOT_NOEXCEPT;
  void coot_fortran(coot_csteqr)(const char* compz, const blas_int* n, blas_cxf* D, blas_cxf* E, blas_cxf* Z, const blas_int* ldz, blas_cxf* work, blas_int* info, blas_len compz_len) COOT_NOEXCEPT;
  void coot_fortran(coot_zsteqr)(const char* compz, const blas_int* n, blas_cxd* D, blas_cxd* E, blas_cxd* Z, const blas_int* ldz, blas_cxd* work, blas_int* info, blas_len compz_len) COOT_NOEXCEPT;
  #else
  void coot_fortran(coot_ssteqr)(const char* compz, const blas_int* n, float*    D, float*    E, float*    Z, const blas_int* ldz, float*    work, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_dsteqr)(const char* compz, const blas_int* n, double*   D, double*   E, double*   Z, const blas_int* ldz, double*   work, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_csteqr)(const char* compz, const blas_int* n, blas_cxf* D, blas_cxf* E, blas_cxf* Z, const blas_int* ldz, blas_cxf* work, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_zsteqr)(const char* compz, const blas_int* n, blas_cxd* D, blas_cxd* E, blas_cxd* Z, const blas_int* ldz, blas_cxd* work, blas_int* info) COOT_NOEXCEPT;
  #endif

  // compute 1-norm/Frobenius norm/inf norm of real symmetric tridiagonal matrix
  #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
  float  coot_fortran(coot_slanst)(const char* norm, const blas_int* n, const float*  D, const float*  E, blas_len norm_len) COOT_NOEXCEPT;
  double coot_fortran(coot_dlanst)(const char* norm, const blas_int* n, const double* D, const double* E, blas_len norm_len) COOT_NOEXCEPT;
  #else
  float  coot_fortran(coot_slanst)(const char* norm, const blas_int* n, const float*  D, const float*  E) COOT_NOEXCEPT;
  double coot_fortran(coot_dlanst)(const char* norm, const blas_int* n, const double* D, const double* E) COOT_NOEXCEPT;
  #endif

  // reduce real symmetric matrix to tridiagonal form
  #if defined(COOT_USE_FORTRAN_HIDDEN_ARGS)
  void coot_fortran(coot_ssytrd)(const char* uplo, const blas_int* n, float*  A, const blas_int* lda, float*  D, float*  E, float*  tau, float*  work, const blas_int* lwork, blas_int* info, blas_len uplo_len) COOT_NOEXCEPT;
  void coot_fortran(coot_dsytrd)(const char* uplo, const blas_int* n, double* A, const blas_int* lda, double* D, double* E, double* tau, double* work, const blas_int* lwork, blas_int* info, blas_len uplo_len) COOT_NOEXCEPT;
  #else
  void coot_fortran(coot_ssytrd)(const char* uplo, const blas_int* n, float*  A, const blas_int* lda, float*  D, float*  E, float*  tau, float*  work, const blas_int* lwork, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_dsytrd)(const char* uplo, const blas_int* n, double* A, const blas_int* lda, double* D, double* E, double* tau, double* work, const blas_int* lwork, blas_int* info) COOT_NOEXCEPT;
  #endif

  // force A and B to be stored prior to doing the addition of A and B
  float  coot_fortran(coot_slamc3)(const float*  A, const float*  B) COOT_NOEXCEPT;
  double coot_fortran(coot_dlamc3)(const double* A, const double* B) COOT_NOEXCEPT;

  // compute the i'th updated eigenvalue of a symmetric rank-one modification to the diagonal matrix in d
  void coot_fortran(coot_slaed4)(const blas_int* n, const blas_int* i, const float*  D, const float*  Z, float*  delta, const float*  rho, float*  dlam, const blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_dlaed4)(const blas_int* n, const blas_int* i, const double* D, const double* Z, double* delta, const double* rho, double* dlam, const blas_int* info) COOT_NOEXCEPT;

  // create a permutation list to merge the element of A into a single set
  void coot_fortran(coot_slamrg)(const blas_int* n1, const blas_int* n2, const float*  A, const blas_int* dtrd1, const blas_int* dtrd2, blas_int* index) COOT_NOEXCEPT;
  void coot_fortran(coot_dlamrg)(const blas_int* n1, const blas_int* n2, const double* A, const blas_int* dtrd1, const blas_int* dtrd2, blas_int* index) COOT_NOEXCEPT;

  // compute all eigenvalues of symmetric tridiagonal matrix
  void coot_fortran(coot_ssterf)(const blas_int* n, float*  D, float*  E, blas_int* info) COOT_NOEXCEPT;
  void coot_fortran(coot_dsterf)(const blas_int* n, double* D, double* E, blas_int* info) COOT_NOEXCEPT;

  // perform a series of row interchanges
  void coot_fortran(coot_slaswp)(const blas_int* n, float*  A, const blas_int* lda, const blas_int* k1, const blas_int* k2, const blas_int* ipiv, const blas_int* incx) COOT_NOEXCEPT;
  void coot_fortran(coot_dlaswp)(const blas_int* n, double* A, const blas_int* lda, const blas_int* k1, const blas_int* k2, const blas_int* ipiv, const blas_int* incx) COOT_NOEXCEPT;
  }

#undef COOT_NOEXCEPT
