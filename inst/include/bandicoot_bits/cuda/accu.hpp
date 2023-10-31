// Copyright 2019 Ryan Curtin (http://www.ratml.org)
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
 * Accumulate all elements in `mem`.
 */
template<typename eT>
inline
eT
accu(dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "coot::cuda::accu(): cuda runtime not valid" );

  CUfunction k = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::accu);
  CUfunction k_small = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::accu_small);

  return generic_reduce<eT, eT>(mem, n_elem, "accu", k, k_small, std::make_tuple(/* no extra args */));
  }



template<typename eT>
inline
eT
accu_subview(dev_mem_t<eT> mem, const uword m_n_rows, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "coot::cuda::accu(): cuda runtime not valid" );

  // TODO: implement specialised handling for two cases: (i) n_cols = 1, (ii) n_rows = 1

  Mat<eT> tmp(1, n_cols);

  const uword src_offset = aux_row1 + aux_col1 * m_n_rows;

  CUfunction k1 = get_rt().cuda_rt.get_kernel<eT, eT>(twoway_kernel_id::sum_colwise_conv_pre);

  dev_mem_t<eT> tmp_mem = tmp.get_dev_mem(false);

  const eT* src_ptr = mem.cuda_mem_ptr + src_offset;
  const uword dest_mem_incr = 1;

  const void* args[] = {
      &(tmp_mem.cuda_mem_ptr),
      &src_ptr,
      (uword*) &n_rows,
      (uword*) &n_cols,
      (uword*) &dest_mem_incr,
      (uword*) &m_n_rows };

  const kernel_dims dims = one_dimensional_grid_dims(n_cols);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      k1,
      dims.d[0], dims.d[1], dims.d[2], // grid dims
      dims.d[3], dims.d[4], dims.d[5], // block dims
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, "coot::cuda::accu_subview(): cuLaunchKernel() failed");

  // combine the column sums

  CUfunction k2 = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::accu_simple);

  const void* args2[] = {
      &(tmp_mem.cuda_mem_ptr),
      &(tmp_mem.cuda_mem_ptr),
      (uword*) &n_cols };

  result = coot_wrapper(cuLaunchKernel)(
      k2,
      1, 1, 1, // grid dims
      1, 1, 1, // block dims
      0, NULL,
      (void**) args2,
      0);

  coot_check_cuda_error(result, "coot::cuda::accu_subview(): cuLaunchKernel() failed");

  return tmp(0);
  }
