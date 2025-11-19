// Copyright 2019 Ryan Curtin (http://www.ratml.org/)
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



template<bool do_trans_A = false>
struct gemv
  {

  template<typename eT>
  static
  inline
  void
  apply(dev_mem_t<eT> y_mem,
        const dev_mem_t<eT> A_mem,
        const uword A_n_rows,
        const uword A_n_cols,
        const dev_mem_t<eT> x_mem,
        const eT alpha,
        const eT beta,
        // subview arguments
        const uword y_offset,
        const uword y_mem_incr,
        const uword A_row_offset,
        const uword A_col_offset,
        const uword A_M_n_rows,
        const uword x_offset,
        const uword x_mem_incr)
    {
    coot_extra_debug_sigprint();
    coot_ignore(y_mem);
    coot_ignore(A_mem);
    coot_ignore(A_n_rows);
    coot_ignore(A_n_cols);
    coot_ignore(x_mem);
    coot_ignore(alpha);
    coot_ignore(beta);
    coot_ignore(y_offset);
    coot_ignore(y_mem_incr);
    coot_ignore(A_row_offset);
    coot_ignore(A_col_offset);
    coot_ignore(A_M_n_rows);
    coot_ignore(x_offset);
    coot_ignore(x_mem_incr);

    coot_stop_runtime_error("coot::cuda::gemv(): unsupported type");
    }



  static
  inline
  void
  apply(dev_mem_t<float> y_mem,
        const dev_mem_t<float> A_mem,
        const uword A_n_rows,
        const uword A_n_cols,
        const dev_mem_t<float> x_mem,
        const float alpha,
        const float beta,
        // subview arguments
        const uword y_offset,
        const uword y_mem_incr,
        const uword A_row_offset,
        const uword A_col_offset,
        const uword A_M_n_rows,
        const uword x_offset,
        const uword x_mem_incr)
    {
    coot_extra_debug_sigprint();

    // coot_debug_assert_blas_size(A);  // TODO: adapt this assert for size_t

    cublasOperation_t trans_a = (do_trans_A) ? CUBLAS_OP_T : CUBLAS_OP_N;

    // cuBLAS takes parameters as `int`s

    const int M = int(A_n_rows);
    const int N = int(A_n_cols);

    const int cuda_lda = int(A_M_n_rows);
    const int cuda_incx = int(x_mem_incr);
    const int cuda_incy = int(y_mem_incr);

    cublasStatus_t result;

    result = coot_wrapper(cublasSgemv)(get_rt().cuda_rt.cublas_handle,
                                       trans_a,
                                       M,
                                       N,
                                       (float*) &alpha,
                                       A_mem.cuda_mem_ptr + A_row_offset + A_col_offset * A_M_n_rows,
                                       cuda_lda,
                                       x_mem.cuda_mem_ptr + x_offset,
                                       cuda_incx,
                                       (float*) &beta,
                                       y_mem.cuda_mem_ptr + y_offset,
                                       cuda_incy);

    coot_check_cublas_error( result, "coot::cuda::gemv(): call to cublasSgemv() failed" );
    }



  static
  inline
  void
  apply(dev_mem_t<double> y_mem,
        const dev_mem_t<double> A_mem,
        const uword A_n_rows,
        const uword A_n_cols,
        const dev_mem_t<double> x_mem,
        const double alpha,
        const double beta,
        // subview arguments
        const uword y_offset,
        const uword y_mem_incr,
        const uword A_row_offset,
        const uword A_col_offset,
        const uword A_M_n_rows,
        const uword x_offset,
        const uword x_mem_incr)
    {
    coot_extra_debug_sigprint();

    // coot_debug_assert_blas_size(A); // TODO: adapt this assert for size_t

    cublasOperation_t trans_a = (do_trans_A) ? CUBLAS_OP_T : CUBLAS_OP_N;

    // cuBLAS takes parameters as `int`s

    const int M = int(A_n_rows);
    const int N = int(A_n_cols);

    const int cuda_lda = int(A_M_n_rows);
    const int cuda_incx = int(x_mem_incr);
    const int cuda_incy = int(y_mem_incr);

    cublasStatus_t result;

    result = coot_wrapper(cublasDgemv)(get_rt().cuda_rt.cublas_handle,
                                       trans_a,
                                       M,
                                       N,
                                       (double*) &alpha,
                                       A_mem.cuda_mem_ptr + A_row_offset + A_col_offset * A_M_n_rows,
                                       cuda_lda,
                                       x_mem.cuda_mem_ptr + x_offset,
                                       cuda_incx,
                                       (double*) &beta,
                                       y_mem.cuda_mem_ptr + y_offset,
                                       cuda_incy);

    coot_check_cublas_error( result, "coot::cuda::gemv(): call to cublasDgemv() failed" );
    }



  static
  inline
  void
  apply(dev_mem_t<fp16> y_mem,
        const dev_mem_t<fp16> A_mem,
        const uword A_n_rows,
        const uword A_n_cols,
        const dev_mem_t<fp16> x_mem,
        const fp16 alpha,
        const fp16 beta,
        // subview arguments
        const uword y_offset,
        const uword y_mem_incr,
        const uword A_row_offset,
        const uword A_col_offset,
        const uword A_M_n_rows,
        const uword x_offset,
        const uword x_mem_incr)
    {
    coot_extra_debug_sigprint();

    coot_check_runtime_error( !get_rt().cuda_rt.is_supported_type<__half>(), "coot::cuda::gemv(): half-precision not supported by device" );

    // coot_debug_assert_blas_size(A);  // TODO: adapt this assert for size_t

    // There is no cublasHgemv, so we have to use cublasHgemm instead.
    // To do this, instead of computing y = Ax or y = A'x, we actually compute
    // y' = x'A' or y' = x'A, because y_mem_incr and x_mem_incr cannot be used as ldb/ldc otherwise.

    cublasOperation_t trans_a = (do_trans_A) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t trans_x = CUBLAS_OP_N;

    // cuBLAS takes parameters as `int`s

    const int M = 1;
    const int N = (do_trans_A) ? int(A_n_cols) : int(A_n_rows);
    const int K = (do_trans_A) ? int(A_n_rows) : int(A_n_cols);

    const int cuda_lda = int(x_mem_incr);
    const int cuda_ldb = int(A_M_n_rows);
    const int cuda_ldc = int(y_mem_incr);

    cublasStatus_t result;

    result = coot_wrapper(cublasHgemm)(get_rt().cuda_rt.cublas_handle,
                                       trans_x,
                                       trans_a,
                                       M,
                                       N,
                                       K,
                                       (const __half*) &alpha,
                                       x_mem.cuda_mem_ptr + x_offset,
                                       cuda_lda,
                                       A_mem.cuda_mem_ptr + A_row_offset + A_col_offset * A_M_n_rows,
                                       cuda_ldb,
                                       (const __half*) &beta,
                                       y_mem.cuda_mem_ptr + y_offset,
                                       cuda_ldc);

    coot_check_cublas_error( result, "coot::cuda::gemv(): call to cublasHgemm() failed" );
    }
  };
