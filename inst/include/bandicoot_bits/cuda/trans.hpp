// Copyright 2022 Ryan Curtin (http://www.ratml.org/)
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


inline
void
htrans(dev_mem_t<float> out, const dev_mem_t<float> in, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "cuda::htrans(): cuda runtime not valid");

  cublasStatus_t result;
  float alpha = 1.0;
  float beta = 0.0;

  result = coot_wrapper(cublasSgeam)(get_rt().cuda_rt.cublas_handle,
                                     CUBLAS_OP_C,
                                     CUBLAS_OP_N,
                                     n_cols,
                                     n_rows,
                                     &alpha,
                                     in.cuda_mem_ptr,
                                     n_rows,
                                     &beta,
                                     /* should be ignored */ in.cuda_mem_ptr,
                                     /* should be ignored */ n_cols,
                                     out.cuda_mem_ptr,
                                     n_cols);


  coot_check_cublas_error( result, "coot::cuda::htrans(): call to cublasSgeam() failed" );
  }



inline
void
htrans(dev_mem_t<double> out, const dev_mem_t<double> in, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "cuda::htrans(): cuda runtime not valid");

  cublasStatus_t result;
  double alpha = 1.0;
  double beta = 0.0;

  result = coot_wrapper(cublasDgeam)(get_rt().cuda_rt.cublas_handle,
                                     CUBLAS_OP_C,
                                     CUBLAS_OP_N,
                                     n_cols,
                                     n_rows,
                                     &alpha,
                                     in.cuda_mem_ptr,
                                     n_rows,
                                     &beta,
                                     /* should be ignored */ in.cuda_mem_ptr,
                                     /* should be ignored */ n_cols,
                                     out.cuda_mem_ptr,
                                     n_cols);

  coot_check_cublas_error( result, "coot::cuda::htrans(): call to cublasDgeam() failed" );
  }



template<typename eT1, typename eT2>
inline
void
htrans(dev_mem_t<eT2> out, const dev_mem_t<eT1> in, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "cuda::htrans(): cuda runtime not valid");

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT2, eT1>(twoway_kernel_id::htrans);

  const void* args[] = {
      &(out.cuda_mem_ptr),
      &(in.cuda_mem_ptr),
      (uword*) &n_rows,
      (uword*) &n_cols };

  const kernel_dims dims = two_dimensional_grid_dims(n_rows, n_cols);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, "coot::cuda::htrans(): cuLaunchKernel() failed");
  }



inline
void
strans(dev_mem_t<float> out, const dev_mem_t<float> in, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "cuda::strans(): cuda runtime not valid");

  cublasStatus_t result;
  float alpha = 1.0;
  float beta = 0.0;

  result = coot_wrapper(cublasSgeam)(get_rt().cuda_rt.cublas_handle,
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     n_cols,
                                     n_rows,
                                     &alpha,
                                     in.cuda_mem_ptr,
                                     n_rows,
                                     &beta,
                                     /* should be ignored */ in.cuda_mem_ptr,
                                     /* should be ignored */ n_cols,
                                     out.cuda_mem_ptr,
                                     n_cols);

  coot_check_cublas_error( result, "coot::cuda::strans(): call to cublasSgeam() failed" );
  }



inline
void
strans(dev_mem_t<double> out, const dev_mem_t<double> in, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "cuda::strans(): cuda runtime not valid");

  cublasStatus_t result;
  double alpha = 1.0;
  double beta = 0.0;

  result = coot_wrapper(cublasDgeam)(get_rt().cuda_rt.cublas_handle,
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     n_cols,
                                     n_rows,
                                     &alpha,
                                     in.cuda_mem_ptr,
                                     n_rows,
                                     &beta,
                                     /* should be ignored */ in.cuda_mem_ptr,
                                     /* should be ignored */ n_cols,
                                     out.cuda_mem_ptr,
                                     n_cols);

  coot_check_cublas_error( result, "coot::cuda::strans(): call to cublasDgeam() failed" );
  }



template<typename eT1, typename eT2>
inline
void
strans(dev_mem_t<eT2> out, const dev_mem_t<eT1> in, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "cuda::strans(): cuda runtime not valid");

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT2, eT1>(twoway_kernel_id::strans);

  const void* args[] = {
      &(out.cuda_mem_ptr),
      &(in.cuda_mem_ptr),
      (uword*) &n_rows,
      (uword*) &n_cols };

  const kernel_dims dims = two_dimensional_grid_dims(n_rows, n_cols);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, "coot::cuda::strans(): cuLaunchKernel() failed");

  }
