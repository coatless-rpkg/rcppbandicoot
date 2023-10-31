// Copyright 2019 Ryan Curtin (http://www.ratml.org/)
//~
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//~
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------



template<bool do_trans_A = false, bool do_trans_B = false>
struct gemm
  {

  template<typename eT>
  static
  inline
  void
  apply(dev_mem_t<eT> C_mem,
        const uword C_n_rows,
        const uword C_n_cols,
        const dev_mem_t<eT> A_mem,
        const uword A_n_rows,
        const uword A_n_cols,
        const dev_mem_t<eT> B_mem,
        const eT alpha,
        const eT beta,
        // subview arguments
        const uword C_row_offset,
        const uword C_col_offset,
        const uword C_M_n_rows,
        const uword A_row_offset,
        const uword A_col_offset,
        const uword A_M_n_rows,
        const uword B_row_offset,
        const uword B_col_offset,
        const uword B_M_n_rows)
    {
    coot_extra_debug_sigprint();

    coot_stop_runtime_error("coot::opencl::gemm(): unsupported type");
    }



  static
  inline
  void
  apply(dev_mem_t<float> C_mem,
        const uword C_n_rows,
        const uword C_n_cols,
        const dev_mem_t<float> A_mem,
        const uword A_n_rows,
        const uword A_n_cols,
        const dev_mem_t<float> B_mem,
        const float alpha,
        const float beta,
        // subview arguments
        const uword C_row_offset,
        const uword C_col_offset,
        const uword C_M_n_rows,
        const uword A_row_offset,
        const uword A_col_offset,
        const uword A_M_n_rows,
        const uword B_row_offset,
        const uword B_col_offset,
        const uword B_M_n_rows)
    {
    coot_extra_debug_sigprint();

    // coot_debug_assert_blas_size(A,B);  // TODO: adapt this assert for size_t

    const clblasTranspose transA = (do_trans_A) ? clblasTrans : clblasNoTrans;
    const clblasTranspose transB = (do_trans_B) ? clblasTrans : clblasNoTrans;

    const size_t M = size_t(C_n_rows);
    const size_t N = size_t(C_n_cols);
    const size_t K = (do_trans_A) ? size_t(A_n_rows) : size_t(A_n_cols);

    const size_t lda = size_t(A_M_n_rows);
    const size_t ldb = size_t(B_M_n_rows);
    const size_t ldc = size_t(C_M_n_rows);

    const size_t A_mem_offset = A_row_offset + A_col_offset * A_M_n_rows;
    const size_t B_mem_offset = B_row_offset + B_col_offset * B_M_n_rows;
    const size_t C_mem_offset = C_row_offset + C_col_offset * C_M_n_rows;

    cl_command_queue queue = get_rt().cl_rt.get_cq();

    cl_int status = 0;

    status |= coot_wrapper(clblasSgemm)(clblasColumnMajor,
                                        transA,
                                        transB,
                                        M,
                                        N,
                                        K,
                                        alpha,
                                        A_mem.cl_mem_ptr,
                                        A_mem_offset,
                                        lda,
                                        B_mem.cl_mem_ptr,
                                        B_mem_offset,
                                        ldb,
                                        beta,
                                        C_mem.cl_mem_ptr,
                                        C_mem_offset,
                                        ldc,
                                        1,
                                        &queue,
                                        0,
                                        NULL,
                                        NULL);

    coot_check_cl_error(status, "coot::opencl::gemm(): eT = float");
    }



  static
  inline
  void
  apply(dev_mem_t<double> C_mem,
        const uword C_n_rows,
        const uword C_n_cols,
        const dev_mem_t<double> A_mem,
        const uword A_n_rows,
        const uword A_n_cols,
        const dev_mem_t<double> B_mem,
        const double alpha,
        const double beta,
        // subview arguments
        const uword C_row_offset,
        const uword C_col_offset,
        const uword C_M_n_rows,
        const uword A_row_offset,
        const uword A_col_offset,
        const uword A_M_n_rows,
        const uword B_row_offset,
        const uword B_col_offset,
        const uword B_M_n_rows)
    {
    coot_extra_debug_sigprint();

    // coot_debug_assert_blas_size(A,B);  // TODO: adapt this assert for size_t

    const clblasTranspose transA = (do_trans_A) ? clblasTrans : clblasNoTrans;
    const clblasTranspose transB = (do_trans_B) ? clblasTrans : clblasNoTrans;

    const size_t M = size_t(C_n_rows);
    const size_t N = size_t(C_n_cols);
    const size_t K = (do_trans_A) ? size_t(A_n_rows) : size_t(A_n_cols);

    const size_t lda = size_t(A_M_n_rows);
    const size_t ldb = size_t(B_M_n_rows);
    const size_t ldc = size_t(C_M_n_rows);

    const size_t A_mem_offset = A_row_offset + A_col_offset * A_M_n_rows;
    const size_t B_mem_offset = B_row_offset + B_col_offset * B_M_n_rows;
    const size_t C_mem_offset = C_row_offset + C_col_offset * C_M_n_rows;

    cl_command_queue queue = get_rt().cl_rt.get_cq();

    cl_int status = 0;

    status |= coot_wrapper(clblasDgemm)(clblasColumnMajor,
                                        transA,
                                        transB,
                                        M,
                                        N,
                                        K,
                                        alpha,
                                        A_mem.cl_mem_ptr,
                                        A_mem_offset,
                                        lda,
                                        B_mem.cl_mem_ptr,
                                        B_mem_offset,
                                        ldb,
                                        beta,
                                        C_mem.cl_mem_ptr,
                                        C_mem_offset,
                                        ldc,
                                        1,
                                        &queue,
                                        0,
                                        NULL,
                                        NULL);

    coot_check_cl_error(status, "coot::opencl::gemm(): eT = double");
    }
  };
