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
 * Compute the 3-dimensional cross-product of A and B into out.
 */
template<typename eT1, typename eT2>
inline
void
cross(dev_mem_t<eT2> out, const dev_mem_t<eT1> A, const dev_mem_t<eT1> B)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "coot::cuda::cross(): CUDA runtime not valid");

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT2, eT1>(twoway_kernel_id::cross);

  // We assume that the inputs have three elements each.
  // (So, this isn't really a great GPU kernel!)
  const void* args[] = {
      &(out.cuda_mem_ptr),
      &(A.cuda_mem_ptr),
      &(B.cuda_mem_ptr) };

  const kernel_dims dims = one_dimensional_grid_dims(3);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, "coot::cuda::cross(): cuLaunchKernel() failed");
  }
