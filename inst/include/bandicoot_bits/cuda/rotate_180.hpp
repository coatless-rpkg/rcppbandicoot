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
 * Rotate the given matrix 180 degrees (i.e. flip horizontally and vertically).
 */
template<typename eT>
inline
void
rotate_180(dev_mem_t<eT> out, const dev_mem_t<eT> in, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "coot::cuda::rotate_180(): CUDA runtime not valid" );

  CUfunction k = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::rotate_180);

  kernel_dims dims = two_dimensional_grid_dims(n_rows, n_cols);

  const void* args[] = {
      &(out.cuda_mem_ptr),
      &(in.cuda_mem_ptr),
      (uword*) &n_rows,
      (uword*) &n_cols };

  CUresult result = coot_wrapper(cuLaunchKernel)(
      k,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL, (void**) args, 0);

  coot_check_cuda_error(result, "coot::cuda::rotate_180(): cuLaunchKernel() failed");
  }
