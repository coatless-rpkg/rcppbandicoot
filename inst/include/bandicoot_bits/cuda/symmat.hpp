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
 * If `lower` is 0, copy the upper triangular part of `in` to the lower triangle and store the resulting symmetric matrix in `out`.
 * If `lower` is 1, copy the lower triangular part of `in` to the upper triangle and store the resulting symmetric matrix in `out`.
 *
 * The input matrix is assumed to be square, with `size` rows and `size` columns.
 */
template<typename eT1, typename eT2>
inline
void
symmat(dev_mem_t<eT2> out, const dev_mem_t<eT1> in, const uword size, const uword lower)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "coot::cuda::symmat(): CUDA runtime not valid");

  kernel_dims dims = two_dimensional_grid_dims(size, size);

  // If out == in, then we can avoid the copy of the input triangle.
  CUfunction k;
  const void** args;
  const void* args_inplace[]  = { &(out.cuda_mem_ptr), (uword*) &size };
  const void* args_separate[] = { &(out.cuda_mem_ptr), &(in.cuda_mem_ptr), (uword*) &size };

  if ((void*) out.cuda_mem_ptr == (void*) in.cuda_mem_ptr)
    {
    k = (lower == 1) ? get_rt().cuda_rt.get_kernel<eT2>(oneway_kernel_id::symmatl_inplace) : get_rt().cuda_rt.get_kernel<eT2>(oneway_kernel_id::symmatu_inplace);
    args = args_inplace;
    }
  else
    {
    k = (lower == 1) ? get_rt().cuda_rt.get_kernel<eT2, eT1>(twoway_kernel_id::symmatl) : get_rt().cuda_rt.get_kernel<eT2, eT1>(twoway_kernel_id::symmatu);
    args = args_separate;
    }

  CUresult result = coot_wrapper(cuLaunchKernel)(
      k, dims.d[0], dims.d[1], dims.d[2], dims.d[3], dims.d[4], dims.d[5],
      0, NULL, (void**) args, 0);

  coot_check_cuda_error(result, "coot::cuda::symmat(): cuLaunchKernel() failed");
  }
