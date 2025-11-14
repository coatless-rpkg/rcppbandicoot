// Copyright 2022-2025 Ryan Curtin (http://www.ratml.org)
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



/**
 * Perform a broadcasting operation from the matrix `src` into `dest`, making `copies_per_col` copies of each column
 * and `copies_per_row` copies of each row.
 */
template<typename eT1, typename eT2>
inline
void
broadcast_op(const twoway_kernel_id::enum_id op,
             dev_mem_t<eT2> dest,
             const dev_mem_t<eT2> dest_in,
             const dev_mem_t<eT1> src,
             const uword src_n_rows,
             const uword src_n_cols,
             const uword copies_per_row,
             const uword copies_per_col,
             const uword dest_row_offset,
             const uword dest_col_offset,
             const uword dest_M_n_rows,
             const uword dest_in_row_offset,
             const uword dest_in_col_offset,
             const uword dest_in_M_n_rows,
             const uword src_row_offset,
             const uword src_col_offset,
             const uword src_M_n_rows)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "coot::cuda::broadcast_op(): CUDA runtime not valid");

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT2, eT1>(op);

  typedef typename cuda_type<eT1>::type ceT1;
  typedef typename cuda_type<eT2>::type ceT2;

        ceT2* dest_ptr    =    dest.cuda_mem_ptr +    dest_row_offset +    dest_col_offset * dest_M_n_rows;
  const ceT2* dest_in_ptr = dest_in.cuda_mem_ptr + dest_in_row_offset + dest_in_col_offset * dest_in_M_n_rows;
  const ceT1*  src_ptr    =     src.cuda_mem_ptr +     src_row_offset +     src_col_offset * src_M_n_rows;

  const uword new_n_rows = src_n_rows * copies_per_row;
  const uword new_n_cols = src_n_cols * copies_per_col;

  const void* args[] = {
      &dest_ptr,
      &dest_in_ptr,
      &src_ptr,
      (uword*) &src_n_rows,
      (uword*) &src_n_cols,
      (uword*) &copies_per_row,
      (uword*) &copies_per_col,
      (uword*) &dest_M_n_rows,
      (uword*) &dest_in_M_n_rows,
      (uword*) &src_M_n_rows };

  const kernel_dims dims = two_dimensional_grid_dims(new_n_rows, new_n_cols);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, "coot::cuda::broadcast_op(): cuLaunchKernel() failed");
  }



/**
 * Perform a broadcasting operation on a subset of the rows or columns of the matrix `src` into `dest`,
 * making `copies_per_col` copies of each column and `copies_per_row` copies of each row.
 */
template<typename eT1, typename eT2>
inline
void
broadcast_subset_op(const twoway_kernel_id::enum_id op,
                    dev_mem_t<eT2> dest,
                    const dev_mem_t<eT2> dest_in,
                    const dev_mem_t<eT1> src,
                    const dev_mem_t<uword> indices,
                    const uword mode,
                    const uword src_n_rows,
                    const uword src_n_cols,
                    const uword copies_per_row,
                    const uword copies_per_col,
                    const uword dest_row_offset,
                    const uword dest_col_offset,
                    const uword dest_M_n_rows,
                    const uword dest_in_row_offset,
                    const uword dest_in_col_offset,
                    const uword dest_in_M_n_rows,
                    const uword src_row_offset,
                    const uword src_col_offset,
                    const uword src_M_n_rows,
                    const uword indices_offset,
                    const uword indices_incr)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "coot::cuda::broadcast_subset_op(): CUDA runtime not valid");

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT2, eT1>(op);

  typedef typename cuda_type<eT1>::type ceT1;
  typedef typename cuda_type<eT2>::type ceT2;

        ceT2* dest_ptr      =    dest.cuda_mem_ptr +    dest_row_offset +    dest_col_offset * dest_M_n_rows;
  const ceT2* dest_in_ptr   = dest_in.cuda_mem_ptr + dest_in_row_offset + dest_in_col_offset * dest_in_M_n_rows;
  const ceT1*  src_ptr      =     src.cuda_mem_ptr +     src_row_offset +     src_col_offset * src_M_n_rows;
  const uword* indices_ptr = indices.cuda_mem_ptr + indices_offset;

  const uword new_n_rows = src_n_rows * copies_per_row;
  const uword new_n_cols = src_n_cols * copies_per_col;

  const void* args[] = {
      &dest_ptr,
      &dest_in_ptr,
      &src_ptr,
      &indices_ptr,
      (uword*) &mode,
      (uword*) &src_n_rows,
      (uword*) &src_n_cols,
      (uword*) &copies_per_row,
      (uword*) &copies_per_col,
      (uword*) &dest_M_n_rows,
      (uword*) &dest_in_M_n_rows,
      (uword*) &src_M_n_rows,
      (uword*) &indices_incr };

  const kernel_dims dims = two_dimensional_grid_dims(new_n_rows, new_n_cols);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, "coot::cuda::broadcast_subset_op(): cuLaunchKernel() failed");
  }
