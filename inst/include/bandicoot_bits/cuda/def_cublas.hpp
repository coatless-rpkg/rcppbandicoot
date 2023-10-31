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



extern "C"
  {



  //
  // setup/teardown functions
  //



  extern cublasStatus_t coot_wrapper(cublasCreate)(cublasHandle_t* handle);
  extern cublasStatus_t coot_wrapper(cublasDestroy)(cublasHandle_t handle);



  //
  // matrix-vector multiplications
  //



  extern cublasStatus_t coot_wrapper(cublasSgemv)(cublasHandle_t handle,
                                                  cublasOperation_t trans,
                                                  int m,
                                                  int n,
                                                  const float* alpha,
                                                  const float* A,
                                                  int lda,
                                                  const float* x,
                                                  int incx,
                                                  const float* beta,
                                                  float* y,
                                                  int incy);



  extern cublasStatus_t coot_wrapper(cublasDgemv)(cublasHandle_t handle,
                                                  cublasOperation_t trans,
                                                  int m,
                                                  int n,
                                                  const double* alpha,
                                                  const double* A,
                                                  int lda,
                                                  const double* x,
                                                  int incx,
                                                  const double* beta,
                                                  double* y,
                                                  int incy);



  //
  // matrix-matrix multiplications
  //



  extern cublasStatus_t coot_wrapper(cublasSgemm)(cublasHandle_t handle,
                                                  cublasOperation_t transa,
                                                  cublasOperation_t transb,
                                                  int m,
                                                  int n,
                                                  int k,
                                                  const float* alpha,
                                                  const float* A,
                                                  int lda,
                                                  const float* B,
                                                  int ldb,
                                                  const float* beta,
                                                  float* C,
                                                  int ldc);



  extern cublasStatus_t coot_wrapper(cublasDgemm)(cublasHandle_t handle,
                                                  cublasOperation_t transa,
                                                  cublasOperation_t transb,
                                                  int m,
                                                  int n,
                                                  int k,
                                                  const double* alpha,
                                                  const double* A,
                                                  int lda,
                                                  const double* B,
                                                  int ldb,
                                                  const double* beta,
                                                  double* C,
                                                  int ldc);



  //
  // matrix addition and transposition
  //



  extern cublasStatus_t coot_wrapper(cublasSgeam)(cublasHandle_t handle,
                                                  cublasOperation_t transa,
                                                  cublasOperation_t transb,
                                                  int m,
                                                  int n,
                                                  const float* alpha,
                                                  const float* A,
                                                  int lda,
                                                  const float* beta,
                                                  const float* B,
                                                  int ldb,
                                                  float* C,
                                                  int ldc);



  extern cublasStatus_t coot_wrapper(cublasDgeam)(cublasHandle_t handle,
                                                  cublasOperation_t transa,
                                                  cublasOperation_t transb,
                                                  int m,
                                                  int n,
                                                  const double* alpha,
                                                  const double* A,
                                                  int lda,
                                                  const double* beta,
                                                  const double* B,
                                                  int ldb,
                                                  double* C,
                                                  int ldc);



  //
  // compute Euclidean norm
  //



  extern cublasStatus_t coot_wrapper(cublasSnrm2)(cublasHandle_t handle,
                                                  int n,
                                                  const float* x,
                                                  int incx,
                                                  float* result);



  extern cublasStatus_t coot_wrapper(cublasDnrm2)(cublasHandle_t handle,
                                                  int n,
                                                  const double* x,
                                                  int incx,
                                                  double* result);



  //
  // dot product
  //



  extern cublasStatus_t coot_wrapper(cublasSdot)(cublasHandle_t handle,
                                                 int n,
                                                 const float* x,
                                                 int incx,
                                                 const float* y,
                                                 int incy,
                                                 float* result);



  extern cublasStatus_t coot_wrapper(cublasDdot)(cublasHandle_t handle,
                                                 int n,
                                                 const double* x,
                                                 int incx,
                                                 const double* y,
                                                 int incy,
                                                 double* result);



  }
