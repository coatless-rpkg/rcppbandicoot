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



/**
 * Compute the row-wise or column-wise variance of the input matrix, storing the result in the output matrix.
 */
template<typename eT>
inline
void
var(dev_mem_t<eT> dest,
    const dev_mem_t<eT> src,
    const dev_mem_t<eT> src_means,
    const uword n_rows,
    const uword n_cols,
    const uword dim,
    const uword norm_type,
    // subview arguments
    const uword dest_offset,
    const uword dest_mem_incr,
    const uword src_row_offset,
    const uword src_col_offset,
    const uword src_M_n_rows,
    const uword src_means_offset,
    const uword src_means_mem_incr)
  {
  coot_extra_debug_sigprint();

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>((dim == 0) ? oneway_kernel_id::var_colwise : oneway_kernel_id::var_rowwise);
  const uword norm_correction = (norm_type == 0) ? 1 : 0;

  const uword src_offset = src_row_offset + src_col_offset * src_M_n_rows;

  const eT* dest_ptr      =      dest.cuda_mem_ptr + dest_offset;
  const eT* src_ptr       =       src.cuda_mem_ptr + src_offset;
  const eT* src_means_ptr = src_means.cuda_mem_ptr + src_means_offset;

  const void* args[] = {
      &dest_ptr,
      &src_ptr,
      &src_means_ptr,
      (uword*) &n_rows,
      (uword*) &n_cols,
      (uword*) &norm_correction,
      (uword*) &dest_mem_incr,
      (uword*) &src_M_n_rows,
      (uword*) &src_means_mem_incr };

  const kernel_dims dims = one_dimensional_grid_dims((dim == 0) ? n_cols : n_rows);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, "coot::cuda::var(): cuLaunchKernel() failed");
  }



template<typename eT>
inline
eT
var_vec(const dev_mem_t<eT> mem, const eT mean, const uword n_elem, const uword norm_type)
  {
  coot_extra_debug_sigprint();

  CUfunction k = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::var);
  CUfunction k_small = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::var_small);

  CUfunction accu_k = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::accu);
  CUfunction accu_k_small = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::accu_small);

  const eT result = generic_reduce<eT, eT>(mem,
                                           n_elem,
                                           "var_vec",
                                           k,
                                           k_small,
                                           std::make_tuple(mean),
                                           accu_k,
                                           accu_k_small,
                                           std::make_tuple(/* no extra args for second and later passes */));
  const uword norm_correction = (norm_type == 0) ? 1 : 0;
  return result / ((eT) (n_elem - norm_correction));
  }



template<typename eT>
inline
eT
var_vec_subview(const dev_mem_t<eT> mem, const eT mean, const uword M_n_rows, const uword M_n_cols, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const uword norm_type)
  {
  coot_extra_debug_sigprint();
  coot_ignore(M_n_cols);

  CUfunction k = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::submat_var);
  CUfunction k_small = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::submat_var_small);

  const uword submat_n_elem = n_rows * n_cols;

  const eT result = generic_reduce<eT, eT>(mem,
                                           submat_n_elem,
                                           "var_vec_subview",
                                           k,
                                           k_small,
                                           std::make_tuple(mean, M_n_rows, aux_row1, aux_col1, n_rows, n_cols));
  const uword norm_correction = (norm_type == 0) ? 1 : 0;
  return result / ((eT) (submat_n_elem - norm_correction));
  }
