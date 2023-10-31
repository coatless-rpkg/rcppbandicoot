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



  extern cusolverStatus_t coot_wrapper(cusolverDnCreate)(cusolverDnHandle_t* handle);
  extern cusolverStatus_t coot_wrapper(cusolverDnDestroy)(cusolverDnHandle_t handle);



  //
  // eigendecomposition
  //



  extern cusolverStatus_t coot_wrapper(cusolverDnXsyevd_bufferSize)(cusolverDnHandle_t handle,
                                                                    cusolverDnParams_t params,
                                                                    cusolverEigMode_t jobz,
                                                                    cublasFillMode_t uplo,
                                                                    int64_t n,
                                                                    cudaDataType dataTypeA,
                                                                    const void* A,
                                                                    int64_t lda,
                                                                    cudaDataType dataTypeW,
                                                                    const void* W,
                                                                    cudaDataType computeType,
                                                                    size_t* workspaceInBytesOnDevice,
                                                                    size_t* workspaceInBytesOnHost);



  extern cusolverStatus_t coot_wrapper(cusolverDnXsyevd)(cusolverDnHandle_t handle,
                                                         cusolverDnParams_t params,
                                                         cusolverEigMode_t jobz,
                                                         cublasFillMode_t uplo,
                                                         int64_t n,
                                                         cudaDataType dataTypeA,
                                                         void* A,
                                                         int64_t lda,
                                                         cudaDataType dataTypeW,
                                                         void* W,
                                                         cudaDataType computeType,
                                                         void* bufferOnDevice,
                                                         size_t workspaceInBytesOnDevice,
                                                         void* bufferOnHost,
                                                         size_t workspaceInBytesOnHost,
                                                         int* info);



  //
  // cholesky decomposition
  //



  extern cusolverStatus_t coot_wrapper(cusolverDnXpotrf_bufferSize)(cusolverDnHandle_t handle,
                                                                    cusolverDnParams_t params,
                                                                    cublasFillMode_t uplo,
                                                                    int64_t n,
                                                                    cudaDataType dataTypeA,
                                                                    const void* A,
                                                                    int64_t lda,
                                                                    cudaDataType computeType,
                                                                    size_t* workspaceInBytesOnDevice,
                                                                    size_t* workspaceInBytesOnHost);



  extern cusolverStatus_t coot_wrapper(cusolverDnXpotrf)(cusolverDnHandle_t handle,
                                                         cusolverDnParams_t params,
                                                         cublasFillMode_t uplo,
                                                         int64_t n,
                                                         cudaDataType dataTypeA,
                                                         void* A,
                                                         int64_t lda,
                                                         cudaDataType computeType,
                                                         void* bufferOnDevice,
                                                         size_t workspaceInBytesOnDevice,
                                                         void* bufferOnHost,
                                                         size_t workspaceInBytesOnHost,
                                                         int* info);



  //
  // lu decomposition
  //



  extern cusolverStatus_t coot_wrapper(cusolverDnXgetrf_bufferSize)(cusolverDnHandle_t handle,
                                                                    cusolverDnParams_t params,
                                                                    int64_t m,
                                                                    int64_t n,
                                                                    cudaDataType dataTypeA,
                                                                    const void* A,
                                                                    int64_t lda,
                                                                    cudaDataType computeType,
                                                                    size_t* workspaceInBytesOnDevice,
                                                                    size_t* workspaceInBytesOnHost);



  extern cusolverStatus_t coot_wrapper(cusolverDnXgetrf)(cusolverDnHandle_t handle,
                                                         cusolverDnParams_t params,
                                                         int64_t m,
                                                         int64_t n,
                                                         cudaDataType dataTypeA,
                                                         void* A,
                                                         int64_t lda,
                                                         int64_t* ipiv,
                                                         cudaDataType computeType,
                                                         void* bufferOnDevice,
                                                         size_t workspaceInBytesOnDevice,
                                                         void* bufferOnHost,
                                                         size_t workspaceInBytesOnHost,
                                                         int* info);



  //
  // singular value decomposition
  //



  extern cusolverStatus_t coot_wrapper(cusolverDnXgesvd_bufferSize)(cusolverDnHandle_t handle,
                                                                    cusolverDnParams_t params,
                                                                    signed char jobu,
                                                                    signed char jobvt,
                                                                    int64_t m,
                                                                    int64_t n,
                                                                    cudaDataType dataTypeA,
                                                                    const void* A,
                                                                    int64_t lda,
                                                                    cudaDataType dataTypeS,
                                                                    const void* S,
                                                                    cudaDataType dataTypeU,
                                                                    const void* U,
                                                                    int64_t ldu,
                                                                    cudaDataType dataTypeVT,
                                                                    const void* VT,
                                                                    int64_t ldvt,
                                                                    cudaDataType computeType,
                                                                    size_t* workspaceInBytesOnDevice,
                                                                    size_t* workspaceInBytesOnHost);



  extern cusolverStatus_t coot_wrapper(cusolverDnXgesvd)(cusolverDnHandle_t handle,
                                                         cusolverDnParams_t params,
                                                         signed char jobu,
                                                         signed char jobvt,
                                                         int64_t m,
                                                         int64_t n,
                                                         cudaDataType dataTypeA,
                                                         void* A,
                                                         int64_t lda,
                                                         cudaDataType dataTypeS,
                                                         void* S,
                                                         cudaDataType dataTypeU,
                                                         void* U,
                                                         int64_t ldu,
                                                         cudaDataType dataTypeVT,
                                                         void* VT,
                                                         int64_t ldvt,
                                                         cudaDataType computeType,
                                                         void* bufferOnDevice,
                                                         size_t workspaceInBytesOnDevice,
                                                         void* bufferOnHost,
                                                         size_t workspaceInBytesOnHost,
                                                         int* info);



  //
  // LU-decomposition-based solver
  //



  extern cusolverStatus_t coot_wrapper(cusolverDnXgetrs)(cusolverDnHandle_t handle,
                                                         cusolverDnParams_t params,
                                                         cublasOperation_t trans,
                                                         int64_t n,
                                                         int64_t nrhs,
                                                         cudaDataType dataTypeA,
                                                         const void* A,
                                                         int64_t lda,
                                                         const int64_t* ipiv,
                                                         cudaDataType dataTypeB,
                                                         void* B,
                                                         int64_t ldb,
                                                         int* info);



  }
