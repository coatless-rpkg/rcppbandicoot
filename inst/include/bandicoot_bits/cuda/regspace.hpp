// Copyright 2021 Marcus Edel (http://kurg.org)
// Copyright 2025 Ryan Curtin (http://ratml.org)
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
 * Generate a vector with elements between start and end, spaced by `delta`.
 */
template<typename eT>
inline
void
regspace(dev_mem_t<eT> mem, const uword mem_incr, const eT start, const eT delta, const eT end, const uword num, const bool desc)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "cuda::regspace(): CUDA runtime not valid");

  const kernel_dims dims = one_dimensional_grid_dims(num);

  CUfunction kernel = desc ? get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::regspace_desc)
                           : get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::linspace);

  dev_mem_t<eT>* in_mem = &mem;

  typedef typename cuda_type<eT>::type ceT;

  const ceT cuda_start = to_cuda_type(start);
  const ceT cuda_delta = to_cuda_type(delta);
  const ceT cuda_end = to_cuda_type(end);

  void* args[] = {
      &(in_mem->cuda_mem_ptr),
      (uword*) &mem_incr,
      (ceT*) &cuda_start,
      (ceT*) &cuda_end,
      (ceT*) &cuda_delta,
      (uword*) &num };

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2], // grid dims
      dims.d[3], dims.d[4], dims.d[5], // block dims
      0,
      NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, "cuda::regspace(): cuLaunchKernel() failed");
  }
