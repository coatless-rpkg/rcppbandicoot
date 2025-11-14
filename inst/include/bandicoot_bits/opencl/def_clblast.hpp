// Copyright 2024 Ryan Curtin (http://www.ratml.org)
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



#if defined(COOT_USE_CLBLAST)
extern "C"
  {

  //
  // matrix-vector multiplication
  //



  extern CLBlastStatusCode coot_wrapper(CLBlastHgemv)(const CLBlastLayout layout,
                                                      const CLBlastTranspose a_transpose,
                                                      const size_t m,
                                                      const size_t n,
                                                      const cl_half alpha,
                                                      const cl_mem a_buffer,
                                                      const size_t a_offset,
                                                      const size_t a_ld,
                                                      const cl_mem x_buffer,
                                                      const size_t x_offset,
                                                      const size_t x_inc,
                                                      const cl_half beta,
                                                      cl_mem y_buffer,
                                                      const size_t y_offset,
                                                      const size_t y_inc,
                                                      cl_command_queue* queue,
                                                      cl_event* event);



  extern CLBlastStatusCode coot_wrapper(CLBlastSgemv)(const CLBlastLayout layout,
                                                      const CLBlastTranspose a_transpose,
                                                      const size_t m,
                                                      const size_t n,
                                                      const float alpha,
                                                      const cl_mem a_buffer,
                                                      const size_t a_offset,
                                                      const size_t a_ld,
                                                      const cl_mem x_buffer,
                                                      const size_t x_offset,
                                                      const size_t x_inc,
                                                      const float beta,
                                                      cl_mem y_buffer,
                                                      const size_t y_offset,
                                                      const size_t y_inc,
                                                      cl_command_queue* queue,
                                                      cl_event* event);



  extern CLBlastStatusCode coot_wrapper(CLBlastDgemv)(const CLBlastLayout layout,
                                                      const CLBlastTranspose a_transpose,
                                                      const size_t m,
                                                      const size_t n,
                                                      const double alpha,
                                                      const cl_mem a_buffer,
                                                      const size_t a_offset,
                                                      const size_t a_ld,
                                                      const cl_mem x_buffer,
                                                      const size_t x_offset,
                                                      const size_t x_inc,
                                                      const double beta,
                                                      cl_mem y_buffer,
                                                      const size_t y_offset,
                                                      const size_t y_inc,
                                                      cl_command_queue* queue,
                                                      cl_event* event);



  //
  // matrix-matrix multiplication
  //



  extern CLBlastStatusCode coot_wrapper(CLBlastHgemm)(const CLBlastLayout layout,
                                                      const CLBlastTranspose a_transpose,
                                                      const CLBlastTranspose b_transpose,
                                                      const size_t m,
                                                      const size_t n,
                                                      const size_t k,
                                                      const cl_half alpha,
                                                      const cl_mem a_buffer,
                                                      const size_t a_offset,
                                                      const size_t a_ld,
                                                      const cl_mem b_buffer,
                                                      const size_t b_offset,
                                                      const size_t b_ld,
                                                      const cl_half beta,
                                                      cl_mem c_buffer,
                                                      const size_t c_offset,
                                                      const size_t c_ld,
                                                      cl_command_queue* queue,
                                                      cl_event* event);



  extern CLBlastStatusCode coot_wrapper(CLBlastSgemm)(const CLBlastLayout layout,
                                                      const CLBlastTranspose a_transpose,
                                                      const CLBlastTranspose b_transpose,
                                                      const size_t m,
                                                      const size_t n,
                                                      const size_t k,
                                                      const float alpha,
                                                      const cl_mem a_buffer,
                                                      const size_t a_offset,
                                                      const size_t a_ld,
                                                      const cl_mem b_buffer,
                                                      const size_t b_offset,
                                                      const size_t b_ld,
                                                      const float beta,
                                                      cl_mem c_buffer,
                                                      const size_t c_offset,
                                                      const size_t c_ld,
                                                      cl_command_queue* queue,
                                                      cl_event* event);



  extern CLBlastStatusCode coot_wrapper(CLBlastDgemm)(const CLBlastLayout layout,
                                                      const CLBlastTranspose a_transpose,
                                                      const CLBlastTranspose b_transpose,
                                                      const size_t m,
                                                      const size_t n,
                                                      const size_t k,
                                                      const double alpha,
                                                      const cl_mem a_buffer,
                                                      const size_t a_offset,
                                                      const size_t a_ld,
                                                      const cl_mem b_buffer,
                                                      const size_t b_offset,
                                                      const size_t b_ld,
                                                      const double beta,
                                                      cl_mem c_buffer,
                                                      const size_t c_offset,
                                                      const size_t c_ld,
                                                      cl_command_queue* queue,
                                                      cl_event* event);



  //
  // rank-k update of symmetric matrix
  //



  extern CLBlastStatusCode coot_wrapper(CLBlastHsyrk)(const CLBlastLayout layout,
                                                      const CLBlastTriangle triangle,
                                                      const CLBlastTranspose a_transpose,
                                                      const size_t n,
                                                      const size_t k,
                                                      const cl_half alpha,
                                                      const cl_mem a_buffer,
                                                      const size_t a_offset,
                                                      const size_t a_ld,
                                                      const cl_half beta,
                                                      cl_mem c_buffer,
                                                      const size_t c_offset,
                                                      const size_t c_ld,
                                                      cl_command_queue* queue,
                                                      cl_event* event);



  extern CLBlastStatusCode coot_wrapper(CLBlastSsyrk)(const CLBlastLayout layout,
                                                      const CLBlastTriangle triangle,
                                                      const CLBlastTranspose a_transpose,
                                                      const size_t n,
                                                      const size_t k,
                                                      const float alpha,
                                                      const cl_mem a_buffer,
                                                      const size_t a_offset,
                                                      const size_t a_ld,
                                                      const float beta,
                                                      cl_mem c_buffer,
                                                      const size_t c_offset,
                                                      const size_t c_ld,
                                                      cl_command_queue* queue,
                                                      cl_event* event);



  extern CLBlastStatusCode coot_wrapper(CLBlastDsyrk)(const CLBlastLayout layout,
                                                      const CLBlastTriangle triangle,
                                                      const CLBlastTranspose a_transpose,
                                                      const size_t n,
                                                      const size_t k,
                                                      const double alpha,
                                                      const cl_mem a_buffer,
                                                      const size_t a_offset,
                                                      const size_t a_ld,
                                                      const double beta,
                                                      cl_mem c_buffer,
                                                      const size_t c_offset,
                                                      const size_t c_ld,
                                                      cl_command_queue* queue,
                                                      cl_event* event);



  //
  // solve a triangular system of equations
  //



  extern CLBlastStatusCode coot_wrapper(CLBlastStrsm)(const CLBlastLayout layout,
                                                      const CLBlastSide side,
                                                      const CLBlastTriangle triangle,
                                                      const CLBlastTranspose a_transpose,
                                                      const CLBlastDiagonal diagonal,
                                                      const size_t m,
                                                      const size_t n,
                                                      const float alpha,
                                                      const cl_mem a_buffer,
                                                      const size_t a_offset,
                                                      const size_t a_ld,
                                                      cl_mem b_buffer,
                                                      const size_t b_offset,
                                                      const size_t b_ld,
                                                      cl_command_queue* queue,
                                                      cl_event* event);



  extern CLBlastStatusCode coot_wrapper(CLBlastDtrsm)(const CLBlastLayout layout,
                                                      const CLBlastSide side,
                                                      const CLBlastTriangle triangle,
                                                      const CLBlastTranspose a_transpose,
                                                      const CLBlastDiagonal diagonal,
                                                      const size_t m,
                                                      const size_t n,
                                                      const double alpha,
                                                      const cl_mem a_buffer,
                                                      const size_t a_offset,
                                                      const size_t a_ld,
                                                      cl_mem b_buffer,
                                                      const size_t b_offset,
                                                      const size_t b_ld,
                                                      cl_command_queue* queue,
                                                      cl_event* event);



  //
  // triangular matrix-matrix multiplication
  //



  extern CLBlastStatusCode coot_wrapper(CLBlastHtrmm)(const CLBlastLayout layout,
                                                      const CLBlastSide side,
                                                      const CLBlastTriangle triangle,
                                                      const CLBlastTranspose a_transpose,
                                                      const CLBlastDiagonal diagonal,
                                                      const size_t m,
                                                      const size_t n,
                                                      const cl_half alpha,
                                                      const cl_mem a_buffer,
                                                      const size_t a_offset,
                                                      const size_t a_ld,
                                                      cl_mem b_buffer,
                                                      const size_t b_offset,
                                                      const size_t b_ld,
                                                      cl_command_queue* queue,
                                                      cl_event* event);



  extern CLBlastStatusCode coot_wrapper(CLBlastStrmm)(const CLBlastLayout layout,
                                                      const CLBlastSide side,
                                                      const CLBlastTriangle triangle,
                                                      const CLBlastTranspose a_transpose,
                                                      const CLBlastDiagonal diagonal,
                                                      const size_t m,
                                                      const size_t n,
                                                      const float alpha,
                                                      const cl_mem a_buffer,
                                                      const size_t a_offset,
                                                      const size_t a_ld,
                                                      cl_mem b_buffer,
                                                      const size_t b_offset,
                                                      const size_t b_ld,
                                                      cl_command_queue* queue,
                                                      cl_event* event);



  extern CLBlastStatusCode coot_wrapper(CLBlastDtrmm)(const CLBlastLayout layout,
                                                      const CLBlastSide side,
                                                      const CLBlastTriangle triangle,
                                                      const CLBlastTranspose a_transpose,
                                                      const CLBlastDiagonal diagonal,
                                                      const size_t m,
                                                      const size_t n,
                                                      const double alpha,
                                                      const cl_mem a_buffer,
                                                      const size_t a_offset,
                                                      const size_t a_ld,
                                                      cl_mem b_buffer,
                                                      const size_t b_offset,
                                                      const size_t b_ld,
                                                      cl_command_queue* queue,
                                                      cl_event* event);



  //
  // solve a triangular system of equations
  //



  extern CLBlastStatusCode coot_wrapper(CLBlastStrsv)(const CLBlastLayout layout,
                                                      const CLBlastTriangle triangle,
                                                      const CLBlastTranspose a_transpose,
                                                      const CLBlastDiagonal diagonal,
                                                      const size_t n,
                                                      const cl_mem a_buffer,
                                                      const size_t a_offset,
                                                      const size_t a_ld,
                                                      cl_mem x_buffer,
                                                      const size_t x_offset,
                                                      const size_t x_inc,
                                                      cl_command_queue* queue,
                                                      cl_event* event);



  extern CLBlastStatusCode coot_wrapper(CLBlastDtrsv)(const CLBlastLayout layout,
                                                      const CLBlastTriangle triangle,
                                                      const CLBlastTranspose a_transpose,
                                                      const CLBlastDiagonal diagonal,
                                                      const size_t n,
                                                      const cl_mem a_buffer,
                                                      const size_t a_offset,
                                                      const size_t a_ld,
                                                      cl_mem x_buffer,
                                                      const size_t x_offset,
                                                      const size_t x_inc,
                                                      cl_command_queue* queue,
                                                      cl_event* event);



  //
  // symmetric matrix-vector multiplication
  //



  extern CLBlastStatusCode coot_wrapper(CLBlastHsymv)(const CLBlastLayout layout,
                                                      const CLBlastTriangle triangle,
                                                      const size_t n,
                                                      const cl_half alpha,
                                                      const cl_mem a_buffer,
                                                      const size_t a_offset,
                                                      const size_t a_ld,
                                                      const cl_mem x_buffer,
                                                      const size_t x_offset,
                                                      const size_t x_inc,
                                                      const cl_half beta,
                                                      cl_mem y_buffer,
                                                      const size_t y_offset,
                                                      const size_t y_inc,
                                                      cl_command_queue* queue,
                                                      cl_event* event);



  extern CLBlastStatusCode coot_wrapper(CLBlastSsymv)(const CLBlastLayout layout,
                                                      const CLBlastTriangle triangle,
                                                      const size_t n,
                                                      const float alpha,
                                                      const cl_mem a_buffer,
                                                      const size_t a_offset,
                                                      const size_t a_ld,
                                                      const cl_mem x_buffer,
                                                      const size_t x_offset,
                                                      const size_t x_inc,
                                                      const float beta,
                                                      cl_mem y_buffer,
                                                      const size_t y_offset,
                                                      const size_t y_inc,
                                                      cl_command_queue* queue,
                                                      cl_event* event);



  extern CLBlastStatusCode coot_wrapper(CLBlastDsymv)(const CLBlastLayout layout,
                                                      const CLBlastTriangle triangle,
                                                      const size_t n,
                                                      const double alpha,
                                                      const cl_mem a_buffer,
                                                      const size_t a_offset,
                                                      const size_t a_ld,
                                                      const cl_mem x_buffer,
                                                      const size_t x_offset,
                                                      const size_t x_inc,
                                                      const double beta,
                                                      cl_mem y_buffer,
                                                      const size_t y_offset,
                                                      const size_t y_inc,
                                                      cl_command_queue* queue,
                                                      cl_event* event);



  //
  // rank 2K-update of symmetric matrix
  //



  extern CLBlastStatusCode coot_wrapper(CLBlastHsyr2k)(const CLBlastLayout layout,
                                                       const CLBlastTriangle triangle,
                                                       const CLBlastTranspose ab_transpose,
                                                       const size_t n,
                                                       const size_t k,
                                                       const cl_half alpha,
                                                       const cl_mem a_buffer,
                                                       const size_t a_offset,
                                                       const size_t a_ld,
                                                       const cl_mem b_buffer,
                                                       const size_t b_offset,
                                                       const size_t b_ld,
                                                       const cl_half beta,
                                                       cl_mem c_buffer,
                                                       const size_t c_offset,
                                                       const size_t c_ld,
                                                       cl_command_queue* queue,
                                                       cl_event* event);



  extern CLBlastStatusCode coot_wrapper(CLBlastSsyr2k)(const CLBlastLayout layout,
                                                       const CLBlastTriangle triangle,
                                                       const CLBlastTranspose ab_transpose,
                                                       const size_t n,
                                                       const size_t k,
                                                       const float alpha,
                                                       const cl_mem a_buffer,
                                                       const size_t a_offset,
                                                       const size_t a_ld,
                                                       const cl_mem b_buffer,
                                                       const size_t b_offset,
                                                       const size_t b_ld,
                                                       const float beta,
                                                       cl_mem c_buffer,
                                                       const size_t c_offset,
                                                       const size_t c_ld,
                                                       cl_command_queue* queue,
                                                       cl_event* event);



  extern CLBlastStatusCode coot_wrapper(CLBlastDsyr2k)(const CLBlastLayout layout,
                                                       const CLBlastTriangle triangle,
                                                       const CLBlastTranspose ab_transpose,
                                                       const size_t n,
                                                       const size_t k,
                                                       const double alpha,
                                                       const cl_mem a_buffer,
                                                       const size_t a_offset,
                                                       const size_t a_ld,
                                                       const cl_mem b_buffer,
                                                       const size_t b_offset,
                                                       const size_t b_ld,
                                                       const double beta,
                                                       cl_mem c_buffer,
                                                       const size_t c_offset,
                                                       const size_t c_ld,
                                                       cl_command_queue* queue,
                                                       cl_event* event);



  //
  // index of maximum absolute value in vector
  //



  extern CLBlastStatusCode coot_wrapper(CLBlastiHamax)(const size_t n,
                                                       cl_mem imax_buffer,
                                                       const size_t imax_offset,
                                                       const cl_mem x_buffer,
                                                       const size_t x_offset,
                                                       const size_t x_inc,
                                                       cl_command_queue* queue,
                                                       cl_event* event);



  extern CLBlastStatusCode coot_wrapper(CLBlastiSamax)(const size_t n,
                                                       cl_mem imax_buffer,
                                                       const size_t imax_offset,
                                                       const cl_mem x_buffer,
                                                       const size_t x_offset,
                                                       const size_t x_inc,
                                                       cl_command_queue* queue,
                                                       cl_event* event);



  extern CLBlastStatusCode coot_wrapper(CLBlastiDamax)(const size_t n,
                                                       cl_mem imax_buffer,
                                                       const size_t imax_offset,
                                                       const cl_mem x_buffer,
                                                       const size_t x_offset,
                                                       const size_t x_inc,
                                                       cl_command_queue* queue,
                                                       cl_event* event);
  }
#endif
