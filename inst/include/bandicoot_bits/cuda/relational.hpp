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



template<typename eT1, typename eT2>
inline
void
relational_scalar_op(dev_mem_t<uword> out_mem, const dev_mem_t<eT1> in_mem, const uword n_elem, const eT2 val, const twoway_kernel_id::enum_id num, const std::string& name)
  {
  coot_extra_debug_sigprint();

  // Shortcut: if there is nothing to do, return.
  if (n_elem == 0)
    return;

  // Get kernel.
  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT2, eT1>(num);

  const void* args[] = {
      &(out_mem.cuda_mem_ptr),
      &(in_mem.cuda_mem_ptr),
      (uword*) &n_elem,
      &val };

  const kernel_dims dims = one_dimensional_grid_dims(n_elem);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, "coot::cuda::relational_scalar_op() (" + name + "): cuLaunchKernel() failed");
  }



template<typename eT1>
inline
void
relational_unary_array_op(dev_mem_t<uword> out_mem, const dev_mem_t<eT1> in_mem, const uword n_elem, const oneway_real_kernel_id::enum_id num, const std::string& name)
  {
  coot_extra_debug_sigprint();

  // Shortcut: if there is nothing to do, return.
  if (n_elem == 0)
    return;

  // Get kernel.
  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT1>(num);

  const void* args[] = {
      &(out_mem.cuda_mem_ptr),
      &(in_mem.cuda_mem_ptr),
      (uword*) &n_elem };

  const kernel_dims dims = one_dimensional_grid_dims(n_elem);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, "coot::cuda::relational_unary_array_op() (" + name + "): cuLaunchKernel() failed");
  }



template<typename eT1, typename eT2>
inline
void
relational_array_op(dev_mem_t<uword> out_mem, const dev_mem_t<eT1> X_mem, const dev_mem_t<eT2> Y_mem, const uword n_elem, const twoway_kernel_id::enum_id num, const std::string& name)
  {
  coot_extra_debug_sigprint();

  // Shortcut: if there is nothing to do, return.
  if (n_elem == 0)
    return;

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT2, eT1>(num);

  const void* args[] = {
      &(out_mem.cuda_mem_ptr),
      &(X_mem.cuda_mem_ptr),
      &(Y_mem.cuda_mem_ptr),
      (uword*) &n_elem };

  const kernel_dims dims = one_dimensional_grid_dims(n_elem);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, "coot::cuda::relational_array_op() (" + name + "): cuLaunchKernel() failed");
  }
