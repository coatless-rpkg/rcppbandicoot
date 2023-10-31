// Copyright 2021 Marcus Edel (http://kurg.org)
// Copyright 2023 Ryan Curtin (http://ratml.org)
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
 * Clamp `dest` to have the elements of `src` limited to the range `[min_val, max_val]`.
 */
template<typename eT1, typename eT2>
inline
void
clamp(dev_mem_t<eT2> dest,
      const dev_mem_t<eT1> src,
      const eT1 min_val,
      const eT1 max_val,
      const uword n_rows,
      const uword n_cols,
      const uword dest_row_offset,
      const uword dest_col_offset,
      const uword dest_M_n_rows,
      const uword src_row_offset,
      const uword src_col_offset,
      const uword src_M_n_rows)
  {
  coot_extra_debug_sigprint();

  if (n_rows == 0 || n_cols == 0)
    {
    return;
    }

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "cuda::clamp(): cuda runtime not valid");

  const kernel_dims dims = two_dimensional_grid_dims(n_rows, n_cols);

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT2, eT1>(twoway_kernel_id::clamp);

  const uword dest_offset = dest_row_offset + dest_col_offset * dest_M_n_rows;
  const uword  src_offset =  src_row_offset +  src_col_offset * src_M_n_rows;

  const eT2* dest_ptr = dest.cuda_mem_ptr + dest_offset;
  const eT1*  src_ptr =  src.cuda_mem_ptr + src_offset;

  const void* args[] = {
      &dest_ptr,
      &src_ptr,
      (eT1*) &min_val,
      (eT1*) &max_val,
      (uword*) &n_rows,
      (uword*) &n_cols,
      (uword*) &dest_M_n_rows,
      (uword*) &src_M_n_rows };

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2], // grid dims
      dims.d[3], dims.d[4], dims.d[5], // block dims
      0,
      NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, "cuda::clamp(): cuLaunchKernel() failed");
  }
