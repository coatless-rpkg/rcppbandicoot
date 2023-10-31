// Copyright 2023 Ryan Curtin (http://www.ratml.org/)
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
reorder_cols(dev_mem_t<eT> out, const dev_mem_t<eT> mem, const uword n_rows, const dev_mem_t<uword> order, const uword out_n_cols)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "coot::cuda::reorder_cols(): CUDA runtime not valid");

  // If the input is empty, don't do anything.
  if (out_n_cols == 0 || n_rows == 0)
    {
    return;
    }

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::reorder_cols);

  const void* args[] = {
      &(out.cuda_mem_ptr),
      &(mem.cuda_mem_ptr),
      (uword*) &n_rows,
      &(order.cuda_mem_ptr),
      (uword*) &out_n_cols };

  const kernel_dims dims = one_dimensional_grid_dims(out_n_cols);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, "coot::cuda::reorder_cols(): cuLaunchKernel() failed");
  }
