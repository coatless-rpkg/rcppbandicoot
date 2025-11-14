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
 * Run a CUDA elementwise kernel that uses a scalar.
 */
template<typename eT>
inline
void
fill(dev_mem_t<eT> dest,
     const eT val,
     const uword n_rows,
     const uword n_cols,
     const uword row_offset,
     const uword col_offset,
     const uword M_n_rows)
  {
  coot_extra_debug_sigprint();

  if (n_rows == 0 || n_cols == 0)
    return;

  // Get kernel.
  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::fill);

  typedef typename cuda_type<eT>::type ceT;

  const ceT* dest_ptr = dest.cuda_mem_ptr + row_offset + (col_offset * M_n_rows);
  const ceT conv_val = to_cuda_type(val);
  const void* args[] = {
      &dest_ptr,
      &conv_val,
      (uword*) &n_rows,
      (uword*) &n_cols,
      (uword*) &M_n_rows };

  const kernel_dims dims = two_dimensional_grid_dims(n_rows, n_cols);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL, // shared mem and stream
      (void**) args, // arguments
      0);

  coot_check_cuda_error( result, "coot::cuda::fill(): cuLaunchKernel() failed" );
  }
