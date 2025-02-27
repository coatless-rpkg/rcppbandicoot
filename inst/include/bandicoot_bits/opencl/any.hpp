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

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot::opencl::any_vec(): OpenCL runtime not valid" );

  cl_kernel k = get_rt().cl_rt.get_kernel<eT2, eT1>(num);
  cl_kernel k_small = get_rt().cl_rt.get_kernel<eT2, eT1>(num_small);
  // Second (and later) passes use the "and" reduction.
  cl_kernel second_k = get_rt().cl_rt.get_kernel<u32>(oneway_integral_kernel_id::or_reduce);
  cl_kernel second_k_small = get_rt().cl_rt.get_kernel<u32>(oneway_integral_kernel_id::or_reduce_small);

  u32 result = generic_reduce<eT1, u32>(mem,
                                        n_elem,
                                        "any",
                                        k,
                                        k_small,
                                        std::make_tuple(val),
                                        second_k,
                                        second_k_small,
                                        std::make_tuple(/* no extra args for second pass */));

  return (result == 0) ? false : true;
  }



// Determine whether any elements in the memory satisfy the conditions imposed by the kernel `num` (and its small version `num_small`).
template<typename eT>
inline
bool
any_vec(const dev_mem_t<eT> mem, const uword n_elem, const eT val, const oneway_real_kernel_id::enum_id num, const oneway_real_kernel_id::enum_id num_small)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot::opencl::any_vec(): OpenCL runtime not valid" );

  cl_kernel k = get_rt().cl_rt.get_kernel<eT>(num);
  cl_kernel k_small = get_rt().cl_rt.get_kernel<eT>(num_small);
  // Second (and later) passes use the "and" reduction.
  cl_kernel second_k = get_rt().cl_rt.get_kernel<u32>(oneway_integral_kernel_id::or_reduce);
  cl_kernel second_k_small = get_rt().cl_rt.get_kernel<u32>(oneway_integral_kernel_id::or_reduce_small);

  u32 result = generic_reduce<eT, u32>(mem,
                                       n_elem,
                                       "any",
                                       k,
                                       k_small,
                                       std::make_tuple(val),
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

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot::opencl::any(): OpenCL runtime not valid" );

  runtime_t::cq_guard guard;

  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT2, eT1>(num);

  cl_int status = 0;

  runtime_t::adapt_uword out_offset(out_mem.cl_mem_ptr.offset);
  runtime_t::adapt_uword A_offset(in_mem.cl_mem_ptr.offset);
  runtime_t::adapt_uword A_n_rows(n_rows);
  runtime_t::adapt_uword A_n_cols(n_cols);

  status |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem),  &(out_mem.cl_mem_ptr.ptr));
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, out_offset.size, out_offset.addr          );
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, sizeof(cl_mem),  &(in_mem.cl_mem_ptr.ptr) );
  status |= coot_wrapper(clSetKernelArg)(kernel, 3, A_offset.size,   A_offset.addr            );
  status |= coot_wrapper(clSetKernelArg)(kernel, 4, sizeof(eT2),    &val                      );
  status |= coot_wrapper(clSetKernelArg)(kernel, 5, A_n_rows.size,  A_n_rows.addr             );
  status |= coot_wrapper(clSetKernelArg)(kernel, 6, A_n_cols.size,  A_n_cols.addr             );
  coot_check_cl_error(status, "coot::opencl::any(): failed to set kernel arguments");

  const size_t k1_work_dim       = 1;
  const size_t k1_work_offset[1] = { 0                            };
  const size_t k1_work_size[1]   = { (colwise ? n_cols : n_rows ) };

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, k1_work_dim, k1_work_offset, k1_work_size, NULL, 0, NULL, NULL);
  coot_check_cl_error(status, "coot::opencl::any(): failed to run kernel");
  }
