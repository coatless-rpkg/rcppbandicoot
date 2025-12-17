// Copyright 2025 Ryan Curtin (https://www.ratml.org/)
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



template<typename eT1, typename eT2>
inline
void
extract_subview_elem1(dev_mem_t<eT2> out_mem,
                      const dev_mem_t<eT1> in_mem,
                      const dev_mem_t<uword> in_locs,
                      const uword n_elem)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "cuda::extract_subview_elem1(): CUDA runtime not valid");

  const kernel_dims dims = one_dimensional_grid_dims(n_elem);

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT2, eT1>(twoway_kernel_id::extract_sve1);

  const void* args[] = {
      &(out_mem.cuda_mem_ptr),
      &(in_mem.cuda_mem_ptr),
      &(in_locs.cuda_mem_ptr),
      (uword*) &n_elem };

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2], // grid dims
      dims.d[3], dims.d[4], dims.d[5], // block dims
      0,
      NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, "cuda::extract_subview_elem1(): cuLaunchKernel() failed");
  }
