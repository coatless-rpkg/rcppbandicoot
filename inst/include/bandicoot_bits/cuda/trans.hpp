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



//
// conjugate does not matter for float and double
//

template<bool conj>
inline
void
trans(dev_mem_t<float> out, const dev_mem_t<float> in, const uword n_rows, const uword n_cols)
  {
  coot_debug_sigprint();

  if (n_rows == 0 || n_cols == 0)
    {
    return;
    }

  coot_check_runtime_error( (get_rt().cuda_rt.is_valid() == false), "cuda::trans(): CUDA runtime not valid");

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


  coot_check_cublas_error( result, "coot::cuda::trans(): call to cublasSgeam() failed" );
  }



template<bool conj>
inline
void
trans(dev_mem_t<double> out, const dev_mem_t<double> in, const uword n_rows, const uword n_cols)
  {
  coot_debug_sigprint();

  if (n_rows == 0 || n_cols == 0)
    {
    return;
    }

  coot_check_runtime_error( (get_rt().cuda_rt.is_valid() == false), "cuda::trans(): CUDA runtime not valid");

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

  coot_check_cublas_error( result, "coot::cuda::trans(): call to cublasDgeam() failed" );
  }



template<bool conj>
inline
void
trans(dev_mem_t<cx_float> out, const dev_mem_t<cx_float> in, const uword n_rows, const uword n_cols)
  {
  coot_debug_sigprint();

  if (n_rows == 0 || n_cols == 0)
    {
    return;
    }

  coot_check_runtime_error( (get_rt().cuda_rt.is_valid() == false), "cuda::trans(): CUDA runtime not valid");

  cublasStatus_t result;
  cx_float alpha(1.0, 0.0);
  cx_float beta(0.0, 0.0);

  result = coot_wrapper(cublasCgeam)(get_rt().cuda_rt.cublas_handle,
                                     conj ? CUBLAS_OP_C : CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     n_cols,
                                     n_rows,
                                     (cuComplex*) &alpha,
                                     (cuComplex*) in.cuda_mem_ptr,
                                     n_rows,
                                     (cuComplex*) &beta,
                                     /* should be ignored */ (cuComplex*) in.cuda_mem_ptr,
                                     /* should be ignored */ n_cols,
                                     (cuComplex*) out.cuda_mem_ptr,
                                     n_cols);

  coot_check_cublas_error( result, "coot::cuda::trans(): call to cublasCgeam() failed" );
  }



template<bool conj>
inline
void
trans(dev_mem_t<cx_double> out, const dev_mem_t<cx_double> in, const uword n_rows, const uword n_cols)
  {
  coot_debug_sigprint();

  if (n_rows == 0 || n_cols == 0)
    {
    return;
    }

  coot_check_runtime_error( (get_rt().cuda_rt.is_valid() == false), "cuda::trans(): CUDA runtime not valid");

  cublasStatus_t result;
  cx_double alpha(1.0, 0.0);
  cx_double beta(0.0, 0.0);

  result = coot_wrapper(cublasZgeam)(get_rt().cuda_rt.cublas_handle,
                                     conj ? CUBLAS_OP_C : CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     n_cols,
                                     n_rows,
                                     (cuDoubleComplex*) &alpha,
                                     (cuDoubleComplex*) in.cuda_mem_ptr,
                                     n_rows,
                                     (cuDoubleComplex*) &beta,
                                     /* should be ignored */ (cuDoubleComplex*) in.cuda_mem_ptr,
                                     /* should be ignored */ n_cols,
                                     (cuDoubleComplex*) out.cuda_mem_ptr,
                                     n_cols);

  coot_check_cublas_error( result, "coot::cuda::trans(): call to cublasZgeam() failed" );
  }
