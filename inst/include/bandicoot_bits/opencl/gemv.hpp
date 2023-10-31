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

    coot_stop_runtime_error("coot::opencl::gemv(): unsupported type");
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

    const clblasTranspose transA = (do_trans_A) ? clblasTrans : clblasNoTrans;

    // clBLAS takes parameters as `size_t`s

    const size_t M = size_t(A_n_rows);
    const size_t N = size_t(A_n_cols);

    const size_t cl_lda = size_t(A_M_n_rows);
    const size_t cl_incx = size_t(x_mem_incr);
    const size_t cl_incy = size_t(y_mem_incr);

    const size_t cl_y_offset = size_t(y_offset);
    const size_t cl_A_offset = size_t(A_row_offset + A_col_offset * A_M_n_rows);
    const size_t cl_x_offset = size_t(x_offset);

    cl_command_queue queue = get_rt().cl_rt.get_cq();

    cl_int status = 0;

    status |= coot_wrapper(clblasSgemv)(clblasColumnMajor,
                                        transA,
                                        M,
                                        N,
                                        alpha,
                                        A_mem.cl_mem_ptr,
                                        cl_A_offset,
                                        cl_lda,
                                        x_mem.cl_mem_ptr,
                                        cl_x_offset,
                                        cl_incx,
                                        beta,
                                        y_mem.cl_mem_ptr,
                                        cl_y_offset,
                                        cl_incy,
                                        1,
                                        &queue,
                                        0,
                                        NULL,
                                        NULL);

    coot_check_cl_error(status, "coot::opencl::gemv(): eT = float");
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

    const clblasTranspose transA = (do_trans_A) ? clblasTrans : clblasNoTrans;

    // clBLAS takes parameters as `size_t`s

    const size_t M = size_t(A_n_rows);
    const size_t N = size_t(A_n_cols);

    const size_t cl_lda = size_t(A_M_n_rows);
    const size_t cl_incx = size_t(x_mem_incr);
    const size_t cl_incy = size_t(y_mem_incr);

    const size_t cl_y_offset = size_t(y_offset);
    const size_t cl_A_offset = size_t(A_row_offset + A_col_offset * A_M_n_rows);
    const size_t cl_x_offset = size_t(x_offset);

    cl_command_queue queue = get_rt().cl_rt.get_cq();

    cl_int status = 0;

    status |= coot_wrapper(clblasDgemv)(clblasColumnMajor,
                                        transA,
                                        M,
                                        N,
                                        alpha,
                                        A_mem.cl_mem_ptr,
                                        cl_A_offset,
                                        cl_lda,
                                        x_mem.cl_mem_ptr,
                                        cl_x_offset,
                                        cl_incx,
                                        beta,
                                        y_mem.cl_mem_ptr,
                                        cl_y_offset,
                                        cl_incy,
                                        1,
                                        &queue,
                                        0,
                                        NULL,
                                        NULL);

    coot_check_cl_error(status, "coot::opencl::gemv(): eT = double");
    }
  };
