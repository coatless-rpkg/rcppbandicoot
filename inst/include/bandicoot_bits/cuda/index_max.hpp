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



template<typename eT>
inline
void
index_max(dev_mem_t<uword> dest,
          const dev_mem_t<eT> src,
          const uword n_rows,
          const uword n_cols,
          const uword dim,
          // subview arguments
          const uword dest_offset,
          const uword dest_mem_incr,
          const uword src_row_offset,
          const uword src_col_offset,
          const uword src_M_n_rows)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "coot::cuda::index_max(): cuda runtime not valid" );

  CUfunction kernel;
  if (dim == 0)
    {
    kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::index_max_colwise);
    }
  else
    {
    kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::index_max_rowwise);
    }

  const uword src_offset = src_row_offset + src_col_offset * src_M_n_rows;

  typedef typename cuda_type<eT>::type ceT;

  const uword* dest_ptr = dest.cuda_mem_ptr + dest_offset;
  const ceT*   src_ptr =  src.cuda_mem_ptr + src_offset;

  const void* args[] = {
      &dest_ptr,
      &src_ptr,
      (uword*) &n_rows,
      (uword*) &n_cols,
      (uword*) &dest_mem_incr,
      (uword*) &src_M_n_rows };

  const kernel_dims dims = one_dimensional_grid_dims((dim == 0) ? n_cols : n_rows);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, "coot::cuda::index_max(): cuLaunchKernel() failed");
  }



/**
 * Compute the maximum index of each column in a cube.
 */
template<typename eT>
inline
void
index_max_cube_col(dev_mem_t<uword> dest, const dev_mem_t<eT> src, const uword n_rows, const uword n_cols, const uword n_slices)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "coot::cuda::index_max_cube_col(): CUDA runtime not valid" );

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::index_max_cube_col);

  const void* args[] = {
      &(dest.cuda_mem_ptr),
      &(src.cuda_mem_ptr),
      (uword*) &n_rows,
      (uword*) &n_cols,
      (uword*) &n_slices };

  const kernel_dims dims = two_dimensional_grid_dims(n_rows, n_slices);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, "coot::cuda::index_max_cube_col(): cuLaunchKernel() failed");
  }



/**
 * Compute the maximum index of all elements in `mem`.
 */
template<typename eT>
inline
uword
index_max_vec(dev_mem_t<eT> mem, const uword n_elem, eT* max_val)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "coot::cuda::index_max_vec(): cuda runtime not valid" );

  CUfunction k = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::index_max);
  CUfunction k_small = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::index_max_small);

  return generic_reduce_uword_aux<eT, eT>(mem, n_elem, "max_vec", max_val, k, k_small, std::make_tuple(/* no extra args */));
  }
