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
 * Replace `val_find` with `val_replace`.
 */
template<typename eT1, typename eT2>
inline
void
replace(dev_mem_t<eT2> dest,
        const dev_mem_t<eT1> src,
        const eT1 val_find,
        const eT1 val_replace,
        const uword n_rows,
        const uword n_cols,
        const uword n_slices,
        const uword dest_row_offset,
        const uword dest_col_offset,
        const uword dest_slice_offset,
        const uword dest_M_n_rows,
        const uword dest_M_n_cols,
        const uword src_row_offset,
        const uword src_col_offset,
        const uword src_slice_offset,
        const uword src_M_n_rows,
        const uword src_M_n_cols)
  {
  coot_extra_debug_sigprint();

  // Get kernel.
  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT2, eT1>(twoway_kernel_id::replace);

  const uword dest_offset = dest_row_offset + dest_col_offset * dest_M_n_rows + dest_slice_offset * dest_M_n_rows * dest_M_n_cols;
  const uword src_offset  =  src_row_offset +  src_col_offset * src_M_n_rows  +  src_slice_offset * src_M_n_rows * src_M_n_cols;

  typedef typename cuda_type<eT1>::type ceT1;
  typedef typename cuda_type<eT2>::type ceT2;

  const ceT1* src_ptr  =  src.cuda_mem_ptr + src_offset;
  const ceT2* dest_ptr = dest.cuda_mem_ptr + dest_offset;

  ceT1 cuda_val_find = to_cuda_type(val_find);
  ceT1 cuda_val_replace = to_cuda_type(val_replace);

  const void* args[] = {
      &dest_ptr,
      &src_ptr,
      &cuda_val_find,
      &cuda_val_replace,
      (uword*) &n_rows,
      (uword*) &n_cols,
      (uword*) &n_slices,
      (uword*) &dest_M_n_rows,
      (uword*) &dest_M_n_cols,
      (uword*) &src_M_n_rows,
      (uword*) &src_M_n_cols };

  const kernel_dims dims = three_dimensional_grid_dims(n_rows, n_cols, n_slices);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL, // shared mem and stream
      (void**) args, // arguments
      0);

  coot_check_cuda_error(result, "coot::cuda::replace(): cuLaunchKernel() failed");
  }
