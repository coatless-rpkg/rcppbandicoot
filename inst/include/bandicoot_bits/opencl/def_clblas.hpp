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



#if defined(COOT_USE_CLBLAS)
extern "C"
  {
  //
  // setup and teardown
  //



  extern clblasStatus coot_wrapper(clblasSetup)();
  extern void         coot_wrapper(clblasTeardown)();



  //
  // matrix-vector multiplication
  //



  extern clblasStatus coot_wrapper(clblasSgemv)(clblasOrder order,
                                                clblasTranspose transA,
                                                size_t M,
                                                size_t N,
                                                cl_float alpha,
                                                const cl_mem A,
                                                size_t offA,
                                                size_t lda,
                                                const cl_mem x,
                                                size_t offx,
                                                int incx,
                                                cl_float beta,
                                                cl_mem y,
                                                size_t offy,
                                                int incy,
                                                cl_uint numCommandQueues,
                                                cl_command_queue* commandQueues,
                                                cl_uint numEventsInWaitList,
                                                const cl_event* eventWaitList,
                                                cl_event* events);



  extern clblasStatus coot_wrapper(clblasDgemv)(clblasOrder order,
                                                clblasTranspose transA,
                                                size_t M,
                                                size_t N,
                                                cl_double alpha,
                                                const cl_mem A,
                                                size_t offA,
                                                size_t lda,
                                                const cl_mem x,
                                                size_t offx,
                                                int incx,
                                                cl_double beta,
                                                cl_mem y,
                                                size_t offy,
                                                int incy,
                                                cl_uint numCommandQueues,
                                                cl_command_queue* commandQueues,
                                                cl_uint numEventsInWaitList,
                                                const cl_event* eventWaitList,
                                                cl_event* events);



  //
  // matrix-matrix multiplication
  //



  extern clblasStatus coot_wrapper(clblasSgemm)(clblasOrder order,
                                                clblasTranspose transA,
                                                clblasTranspose transB,
                                                size_t M,
                                                size_t N,
                                                size_t K,
                                                cl_float alpha,
                                                const cl_mem A,
                                                size_t offA,
                                                size_t lda,
                                                const cl_mem B,
                                                size_t offB,
                                                size_t ldb,
                                                cl_float beta,
                                                cl_mem C,
                                                size_t offC,
                                                size_t ldc,
                                                cl_uint numCommandQueues,
                                                cl_command_queue* commandQueues,
                                                cl_uint numEventsInWaitList,
                                                const cl_event* eventWaitList,
                                                cl_event* events);



  extern clblasStatus coot_wrapper(clblasDgemm)(clblasOrder order,
                                                clblasTranspose transA,
                                                clblasTranspose transB,
                                                size_t M,
                                                size_t N,
                                                size_t K,
                                                cl_double alpha,
                                                const cl_mem A,
                                                size_t offA,
                                                size_t lda,
                                                const cl_mem B,
                                                size_t offB,
                                                size_t ldb,
                                                cl_double beta,
                                                cl_mem C,
                                                size_t offC,
                                                size_t ldc,
                                                cl_uint numCommandQueues,
                                                cl_command_queue* commandQueues,
                                                cl_uint numEventsInWaitList,
                                                const cl_event* eventWaitList,
                                                cl_event* events);

  //
  // symmetric rank-k update
  //



  extern clblasStatus coot_wrapper(clblasSsyrk)(clblasOrder order,
                                                clblasUplo uplo,
                                                clblasTranspose transA,
                                                size_t N,
                                                size_t K,
                                                cl_float alpha,
                                                const cl_mem A,
                                                size_t offA,
                                                size_t lda,
                                                cl_float beta,
                                                cl_mem C,
                                                size_t offC,
                                                size_t ldc,
                                                cl_uint numCommandQueues,
                                                cl_command_queue* commandQueues,
                                                cl_uint numEventsInWaitList,
                                                const cl_event* eventWaitList,
                                                cl_event* events);



  extern clblasStatus coot_wrapper(clblasDsyrk)(clblasOrder order,
                                                clblasUplo uplo,
                                                clblasTranspose transA,
                                                size_t N,
                                                size_t K,
                                                cl_double alpha,
                                                const cl_mem A,
                                                size_t offA,
                                                size_t lda,
                                                cl_double beta,
                                                cl_mem C,
                                                size_t offC,
                                                size_t ldc,
                                                cl_uint numCommandQueues,
                                                cl_command_queue* commandQueues,
                                                cl_uint numEventsInWaitList,
                                                const cl_event* eventWaitList,
                                                cl_event* events);



  //
  // solve triangular systems of equations
  //



  extern clblasStatus coot_wrapper(clblasStrsm)(clblasOrder order,
                                                clblasSide side,
                                                clblasUplo uplo,
                                                clblasTranspose transA,
                                                clblasDiag diag,
                                                size_t M,
                                                size_t N,
                                                cl_float alpha,
                                                const cl_mem A,
                                                size_t offA,
                                                size_t lda,
                                                cl_mem B,
                                                size_t offB,
                                                size_t ldb,
                                                cl_uint numCommandQueues,
                                                cl_command_queue* commandQueues,
                                                cl_uint numEventsInWaitList,
                                                const cl_event* eventWaitList,
                                                cl_event* events);



  extern clblasStatus coot_wrapper(clblasDtrsm)(clblasOrder order,
                                                clblasSide side,
                                                clblasUplo uplo,
                                                clblasTranspose transA,
                                                clblasDiag diag,
                                                size_t M,
                                                size_t N,
                                                cl_double alpha,
                                                const cl_mem A,
                                                size_t offA,
                                                size_t lda,
                                                cl_mem B,
                                                size_t offB,
                                                size_t ldb,
                                                cl_uint numCommandQueues,
                                                cl_command_queue* commandQueues,
                                                cl_uint numEventsInWaitList,
                                                const cl_event* eventWaitList,
                                                cl_event* events);



  //
  // triangular matrix-matrix multiplication
  //



  extern clblasStatus coot_wrapper(clblasStrmm)(clblasOrder order,
                                                clblasSide side,
                                                clblasUplo uplo,
                                                clblasTranspose transA,
                                                clblasDiag diag,
                                                size_t M,
                                                size_t N,
                                                cl_float alpha,
                                                const cl_mem A,
                                                size_t offA,
                                                size_t lda,
                                                cl_mem B,
                                                size_t offB,
                                                size_t ldb,
                                                cl_uint numCommandQueues,
                                                cl_command_queue* commandQueues,
                                                cl_uint numEventsInWaitList,
                                                const cl_event* eventWaitList,
                                                cl_event* events);



  extern clblasStatus coot_wrapper(clblasDtrmm)(clblasOrder order,
                                                clblasSide side,
                                                clblasUplo uplo,
                                                clblasTranspose transA,
                                                clblasDiag diag,
                                                size_t M,
                                                size_t N,
                                                cl_double alpha,
                                                const cl_mem A,
                                                size_t offA,
                                                size_t lda,
                                                cl_mem B,
                                                size_t offB,
                                                size_t ldb,
                                                cl_uint numCommandQueues,
                                                cl_command_queue* commandQueues,
                                                cl_uint numEventsInWaitList,
                                                const cl_event* eventWaitList,
                                                cl_event* events);



  //
  // triangular matrix vector solve
  //



  extern clblasStatus coot_wrapper(clblasDtrsv)(clblasOrder order,
                                                clblasUplo uplo,
                                                clblasTranspose trans,
                                                clblasDiag diag,
                                                size_t N,
                                                const cl_mem A,
                                                size_t offa,
                                                size_t lda,
                                                cl_mem x,
                                                size_t offx,
                                                int incx,
                                                cl_uint numCommandQueues,
                                                cl_command_queue* commandQueues,
                                                cl_uint numEventsInWaitList,
                                                const cl_event* eventWaitList,
                                                cl_event* events);



  extern clblasStatus coot_wrapper(clblasStrsv)(clblasOrder order,
                                                clblasUplo uplo,
                                                clblasTranspose trans,
                                                clblasDiag diag,
                                                size_t N,
                                                const cl_mem A,
                                                size_t offa,
                                                size_t lda,
                                                cl_mem x,
                                                size_t offx,
                                                int incx,
                                                cl_uint numCommandQueues,
                                                cl_command_queue* commandQueues,
                                                cl_uint numEventsInWaitList,
                                                const cl_event* eventWaitList,
                                                cl_event* events);



  //
  // symmetric matrix-vector multiplication
  //



  extern clblasStatus coot_wrapper(clblasSsymv)(clblasOrder order,
                                                clblasUplo uplo,
                                                size_t N,
                                                cl_float alpha,
                                                const cl_mem A,
                                                size_t offA,
                                                size_t lda,
                                                const cl_mem x,
                                                size_t offx,
                                                int incx,
                                                cl_float beta,
                                                cl_mem y,
                                                size_t offy,
                                                int incy,
                                                cl_uint numCommandQueues,
                                                cl_command_queue* commandQueues,
                                                cl_uint numEventsInWaitList,
                                                const cl_event* eventWaitList,
                                                cl_event* events);



  extern clblasStatus coot_wrapper(clblasDsymv)(clblasOrder order,
                                                clblasUplo uplo,
                                                size_t N,
                                                cl_double alpha,
                                                const cl_mem A,
                                                size_t offA,
                                                size_t lda,
                                                const cl_mem x,
                                                size_t offx,
                                                int incx,
                                                cl_double beta,
                                                cl_mem y,
                                                size_t offy,
                                                int incy,
                                                cl_uint numCommandQueues,
                                                cl_command_queue* commandQueues,
                                                cl_uint numEventsInWaitList,
                                                const cl_event* eventWaitList,
                                                cl_event* events);



  //
  // symmetric rank-2k update to a matrix
  //



  extern clblasStatus coot_wrapper(clblasSsyr2k)(clblasOrder order,
                                                 clblasUplo uplo,
                                                 clblasTranspose transAB,
                                                 size_t N,
                                                 size_t K,
                                                 cl_float alpha,
                                                 const cl_mem A,
                                                 size_t offA,
                                                 size_t lda,
                                                 const cl_mem B,
                                                 size_t offB,
                                                 size_t ldb,
                                                 cl_float beta,
                                                 cl_mem C,
                                                 size_t offC,
                                                 size_t ldc,
                                                 cl_uint numCommandQueues,
                                                 cl_command_queue* commandQueues,
                                                 cl_uint numEventsInWaitList,
                                                 const cl_event* eventWaitList,
                                                 cl_event* events);



  extern clblasStatus coot_wrapper(clblasDsyr2k)(clblasOrder order,
                                                 clblasUplo uplo,
                                                 clblasTranspose transAB,
                                                 size_t N,
                                                 size_t K,
                                                 cl_double alpha,
                                                 const cl_mem A,
                                                 size_t offA,
                                                 size_t lda,
                                                 const cl_mem B,
                                                 size_t offB,
                                                 size_t ldb,
                                                 cl_double beta,
                                                 cl_mem C,
                                                 size_t offC,
                                                 size_t ldc,
                                                 cl_uint numCommandQueues,
                                                 cl_command_queue* commandQueues,
                                                 cl_uint numEventsInWaitList,
                                                 const cl_event* eventWaitList,
                                                 cl_event* events);


  //
  // index of max absolute value
  //



  extern clblasStatus coot_wrapper(clblasiSamax)(size_t N,
                                                 cl_mem iMax,
                                                 size_t offiMax,
                                                 const cl_mem X,
                                                 size_t offx,
                                                 int incx,
                                                 cl_mem scratchBuff,
                                                 cl_uint numCommandQueues,
                                                 cl_command_queue* commandQueues,
                                                 cl_uint numEventsInWaitList,
                                                 const cl_event* eventWaitList,
                                                 cl_event* events);



  extern clblasStatus coot_wrapper(clblasiDamax)(size_t N,
                                                 cl_mem iMax,
                                                 size_t offiMax,
                                                 const cl_mem X,
                                                 size_t offx,
                                                 int incx,
                                                 cl_mem scratchBuff,
                                                 cl_uint numCommandQueues,
                                                 cl_command_queue* commandQueues,
                                                 cl_uint numEventsInWaitList,
                                                 const cl_event* eventWaitList,
                                                 cl_event* events);



  }
#endif
