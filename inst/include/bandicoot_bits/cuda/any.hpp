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



// Determine whether any elements in the memory satisfy the conditions imposed by the kernel `num` (and its small version `num_small`).
template<typename eT1, typename eT2>
inline
bool
any_vec(const dev_mem_t<eT1> mem, const uword n_elem, const eT2 val, const twoway_kernel_id::enum_id num, const twoway_kernel_id::enum_id num_small)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "coot::cuda::any_vec(): CUDA runtime not valid" );

  CUfunction k = get_rt().cuda_rt.get_kernel<eT2, eT1>(num);
  CUfunction k_small = get_rt().cuda_rt.get_kernel<eT2, eT1>(num_small);
  // Second (and later) passes use the "or" reduction.
  CUfunction second_k = get_rt().cuda_rt.get_kernel<u32>(oneway_integral_kernel_id::or_reduce);
  CUfunction second_k_small = get_rt().cuda_rt.get_kernel<u32>(oneway_integral_kernel_id::or_reduce_small);

  u32 result = generic_reduce<eT1, u32>(mem,
                                        n_elem,
                                        "any_vec",
                                        k,
                                        k_small,
                                        std::make_tuple(to_cuda_type(val)),
                                        second_k,
                                        second_k_small,
                                        std::make_tuple(/* no extra args for second pass */));
  return (result == 0) ? false : true;
  }



template<typename eT>
inline
bool
any_vec(const dev_mem_t<eT> mem, const uword n_elem, const eT val, const oneway_real_kernel_id::enum_id num, const oneway_real_kernel_id::enum_id num_small)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "coot::cuda::any_vec(): CUDA runtime not valid" );

  CUfunction k = get_rt().cuda_rt.get_kernel<eT>(num);
  CUfunction k_small = get_rt().cuda_rt.get_kernel<eT>(num_small);
  // Second (and later) passes use the "or" reduction.
  CUfunction second_k = get_rt().cuda_rt.get_kernel<u32>(oneway_integral_kernel_id::or_reduce);
  CUfunction second_k_small = get_rt().cuda_rt.get_kernel<u32>(oneway_integral_kernel_id::or_reduce_small);

  u32 result = generic_reduce<eT, u32>(mem,
                                       n_elem,
                                       "any_vec",
                                       k,
                                       k_small,
                                       std::make_tuple(to_cuda_type(val)),
                                       second_k,
                                       second_k_small,
                                       std::make_tuple(/* no extra args for second pass */));
  return (result == 0) ? false : true;
  }



template<typename eT1, typename eT2>
inline
void
any(dev_mem_t<uword> out_mem, const dev_mem_t<eT1> in_mem, const uword n_rows, const uword n_cols, const eT2 val, const twoway_kernel_id::enum_id num, const bool colwise)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "coot::cuda::any(): CUDA runtime not valid" );

  CUfunction k = get_rt().cuda_rt.get_kernel<eT2, eT1>(num);

  const void* args[] = {
      &(out_mem.cuda_mem_ptr),
      &(in_mem.cuda_mem_ptr),
      (eT2*) &val,
      (uword*) &n_rows,
      (uword*) &n_cols };

  const kernel_dims dims = one_dimensional_grid_dims(colwise ? n_cols : n_rows);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      k,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, "coot::cuda::any(): cuLaunchKernel() failed");
  }
