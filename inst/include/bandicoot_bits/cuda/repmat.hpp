// Copyright 2022 Ryan Curtin (http://www.ratml.org)
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
 * Copy the given matrix `src` into `dest`, making `copies_per_col` copies of each column, and `copies_per_row` copies of each row.
 */
template<typename eT1, typename eT2>
inline
void
repmat(const dev_mem_t<eT1> src, dev_mem_t<eT2> dest, const uword n_rows, const uword n_cols, const uword copies_per_row, const uword copies_per_col)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "coot::cuda::repmat(): cuda runtime not valid");

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT2, eT1>(twoway_kernel_id::repmat);

  const uword new_n_rows = n_rows * copies_per_row;

  const void* args[] = {
      &(src.cuda_mem_ptr),
      &(dest.cuda_mem_ptr),
      (uword*) &n_rows,
      (uword*) &n_cols,
      (uword*) &copies_per_row,
      (uword*) &copies_per_col,
      (uword*) &new_n_rows };

  const kernel_dims dims = two_dimensional_grid_dims(n_rows, n_cols);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, "coot::cuda::repmat(): cuLaunchKernel() failed");
  }
