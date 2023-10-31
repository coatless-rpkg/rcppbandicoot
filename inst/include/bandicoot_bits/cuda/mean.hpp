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
 * Compute the row-wise or column-wise mean of the input matrix, storing the result in the output matrix.
 */
template<typename eT2, typename eT1>
inline
void
mean(dev_mem_t<eT2> dest,
     const dev_mem_t<eT1> src,
     const uword n_rows,
     const uword n_cols,
     const uword dim,
     const bool post_conv_apply,
     // subview arguments
     const uword dest_offset,
     const uword dest_mem_incr,
     const uword src_row_offset,
     const uword src_col_offset,
     const uword src_M_n_rows)
  {
  coot_extra_debug_sigprint();

  // Which kernel we need to use depends on whether we want to apply conversion before or after we compute the mean.
  CUfunction kernel;
  if (dim == 0)
    {
    kernel = get_rt().cuda_rt.get_kernel<eT2, eT1>(post_conv_apply ? twoway_kernel_id::mean_colwise_conv_post : twoway_kernel_id::mean_colwise_conv_pre);
    }
  else
    {
    kernel = get_rt().cuda_rt.get_kernel<eT2, eT1>(post_conv_apply ? twoway_kernel_id::mean_rowwise_conv_post : twoway_kernel_id::mean_rowwise_conv_pre);
    }

  const uword  src_offset = src_row_offset + src_col_offset * src_M_n_rows;

  const eT2* dest_ptr = dest.cuda_mem_ptr + dest_offset;
  const eT1*  src_ptr =  src.cuda_mem_ptr + src_offset;

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

  coot_check_cuda_error(result, "coot::cuda::mean(): cuLaunchKernel() failed");
  }
