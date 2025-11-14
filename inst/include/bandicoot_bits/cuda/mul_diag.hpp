// Copyright 2023 Ryan Curtin (http://www.ratml.org/)
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



// Assumption: the operation is not diagmat(A) * diagmat(B)!
// Assumption: if the non-diagonal matrix is transposed, it is not also the output!
template<typename eT>
inline
void
mul_diag
  (
  dev_mem_t<eT> C_mem,
  const uword C_n_rows,
  const uword C_n_cols,
  const eT alpha,
  const dev_mem_t<eT> A_mem,
  const bool A_is_diag,
  const bool A_trans,
  const uword A_row_offset,
  const uword A_col_offset,
  const uword A_M_n_rows,
  const dev_mem_t<eT> B_mem,
  const bool B_is_diag,
  const bool B_trans,
  const uword B_row_offset,
  const uword B_col_offset,
  const uword B_M_n_rows
  )
  {
  coot_extra_debug_sigprint();

  typedef typename cuda_type<eT>::type ceT;

  ceT* diag_arg_ptr;
  ceT* mat_arg_ptr;
  uword diag_arg_incr;
  uword mat_arg_M_n_rows;

  CUfunction kernel;
  kernel_dims dims;

  if (A_is_diag && !B_is_diag && !B_trans)
    {
    // diagmat(A) * B
    kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::mul_rowwise);

    diag_arg_ptr = A_mem.cuda_mem_ptr + A_row_offset + A_col_offset * A_M_n_rows;
    mat_arg_ptr  = B_mem.cuda_mem_ptr + B_row_offset + B_col_offset * B_M_n_rows;

    diag_arg_incr = A_M_n_rows;
    mat_arg_M_n_rows = B_M_n_rows;

    dims = one_dimensional_grid_dims(C_n_cols);
    }
  else if (!A_is_diag && !A_trans && B_is_diag)
    {
    // A * diagmat(B)
    kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::mul_colwise);

    diag_arg_ptr = B_mem.cuda_mem_ptr + B_row_offset + B_col_offset * B_M_n_rows;
    mat_arg_ptr  = A_mem.cuda_mem_ptr + A_row_offset + A_col_offset * A_M_n_rows;

    diag_arg_incr = B_M_n_rows;
    mat_arg_M_n_rows = A_M_n_rows;

    dims = one_dimensional_grid_dims(C_n_rows);
    }
  else if (A_is_diag && !B_is_diag && B_trans)
    {
    // diagmat(A) * B'
    kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::mul_rowwise_trans);

    coot_debug_check( mem_overlaps(C_mem, 0, C_n_rows * C_n_cols, B_mem, 0, C_n_rows * C_n_cols), "coot::cuda::mul_diag(): incorrect call, alias and transpose not allowed" );

    diag_arg_ptr = A_mem.cuda_mem_ptr + A_row_offset + A_col_offset * A_M_n_rows;
    mat_arg_ptr  = B_mem.cuda_mem_ptr + B_row_offset + B_col_offset * B_M_n_rows;

    diag_arg_incr = A_M_n_rows;
    mat_arg_M_n_rows = B_M_n_rows;

    dims = one_dimensional_grid_dims(C_n_rows);
    }
  else if (!A_is_diag && A_trans && B_is_diag)
    {
    // A' * diagmat(B)
    kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::mul_colwise_trans);

    coot_debug_check( mem_overlaps(C_mem, 0, C_n_rows * C_n_cols, A_mem, 0, C_n_rows * C_n_cols), "coot::cuda::mul_diag(): incorrect call, alias and transpose not allowed" );

    diag_arg_ptr = B_mem.cuda_mem_ptr + B_row_offset + B_col_offset * B_M_n_rows;
    mat_arg_ptr  = A_mem.cuda_mem_ptr + A_row_offset + A_col_offset * A_M_n_rows;

    diag_arg_incr = B_M_n_rows;
    mat_arg_M_n_rows = A_M_n_rows;

    dims = one_dimensional_grid_dims(C_n_cols);
    }
  else
    {
    coot_stop_runtime_error("coot::cuda::mul_diag(): incorrect call, at least one matrix must be non-diagonal");
    }

  ceT cuda_alpha = to_cuda_type(alpha);

  const void* args[] = {
      &(C_mem.cuda_mem_ptr),
      &diag_arg_ptr,
      (uword*) &diag_arg_incr,
      &mat_arg_ptr,
      (ceT*) &cuda_alpha,
      (uword*) &C_n_rows,
      (uword*) &C_n_cols,
      (uword*) &mat_arg_M_n_rows };

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL, // shared mem and stream
      (void**) args, // arguments
      0);

  coot_check_cuda_error( result, "coot::cuda::mul_diag(): cuLaunchKernel() failed" );
  }
