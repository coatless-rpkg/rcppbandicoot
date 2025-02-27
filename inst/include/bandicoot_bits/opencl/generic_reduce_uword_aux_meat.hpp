// Copyright 2021-2023 Ryan Curtin (http://www.ratml.org)
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



template<typename eT, typename aux_eT, typename... A1, typename... A2>
inline
uword
generic_reduce_uword_aux(const dev_mem_t<eT> mem,
                         const uword n_elem,
                         const char* kernel_name,
                         eT* reduce_result,
                         cl_kernel& first_kernel,
                         cl_kernel& first_kernel_small,
                         const std::tuple<A1...>& first_kernel_extra_args,
                         cl_kernel& second_kernel,
                         cl_kernel& second_kernel_small,
                         const std::tuple<A2...>& second_kernel_extra_args)
  {
  // Compute workgroup sizes.  We use CL_KERNEL_WORK_GROUP_SIZE as an upper bound, which
  // depends on the compiled kernel.  I assume that the results for k will be identical to k_small.
  size_t kernel_wg_size;
  cl_int status = coot_wrapper(clGetKernelWorkGroupInfo)(first_kernel, get_rt().cl_rt.get_device(), CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &kernel_wg_size, NULL);
  // TODO: should we multiply by CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE?
  coot_check_cl_error(status, std::string("coot::opencl::") + std::string(kernel_name) + std::string("()"));

  uword total_num_threads, local_group_size;
  reduce_kernel_group_info(first_kernel, n_elem, kernel_name, total_num_threads, local_group_size);

  // Create auxiliary memory.
  const uword first_aux_size = std::ceil((total_num_threads + (local_group_size - 1)) / local_group_size);
  const uword second_aux_size = (first_aux_size == 1) ? 0 : std::ceil((first_aux_size + (local_group_size - 1)) / local_group_size);
  Col<aux_eT> first_aux(first_aux_size);
  Col<uword> first_uword_aux(first_aux_size);
  Col<aux_eT> second_aux(second_aux_size);
  Col<uword> second_uword_aux(second_aux_size);

  dev_mem_t<aux_eT> first_aux_mem_ptr = first_aux.get_dev_mem(false);
  dev_mem_t<uword> first_uword_aux_mem_ptr = first_uword_aux.get_dev_mem(false);
  // Just use the first pointer if there is no need for secondary auxiliary
  // space.
  dev_mem_t<aux_eT> second_aux_mem_ptr = (second_aux_size == 0) ? first_aux_mem_ptr : second_aux.get_dev_mem(false);
  dev_mem_t<uword> second_uword_aux_mem_ptr = (second_aux_size == 0) ? first_uword_aux_mem_ptr : second_uword_aux.get_dev_mem(false);

  const bool first_buffer = generic_reduce_uword_aux_inner(mem,
                                                           second_uword_aux_mem_ptr /* dummy pointer */,
                                                           0 /* don't use uword memory for the first pass */,
                                                           n_elem,
                                                           first_aux_mem_ptr,
                                                           first_uword_aux_mem_ptr,
                                                           kernel_name,
                                                           total_num_threads,
                                                           local_group_size,
                                                           first_kernel,
                                                           first_kernel_small,
                                                           first_kernel_extra_args,
                                                           second_kernel,
                                                           second_kernel_small,
                                                           second_kernel_extra_args,
                                                           second_aux_mem_ptr,
                                                           second_uword_aux_mem_ptr);

  if (reduce_result != nullptr)
    {
    (*reduce_result) = (first_buffer ? eT(first_aux[0]) : eT(second_aux[0]));
    }

  return (first_buffer) ? uword(first_uword_aux[0]) : uword(second_uword_aux[0]);
  }



template<typename eT, typename aux_eT, typename... Args>
inline
uword
generic_reduce_uword_aux(const dev_mem_t<eT> mem,
                         const uword n_elem,
                         const char* kernel_name,
                         eT* reduce_result,
                         cl_kernel& kernel,
                         cl_kernel& kernel_small,
                         const std::tuple<Args...>& kernel_extra_args)
  {
  return generic_reduce_uword_aux<eT, aux_eT>(mem,
                                              n_elem,
                                              kernel_name,
                                              reduce_result,
                                              kernel,
                                              kernel_small,
                                              kernel_extra_args,
                                              kernel,
                                              kernel_small,
                                              kernel_extra_args);
  }



template<typename eT, typename aux_eT, typename... A1, typename... A2>
inline
bool
generic_reduce_uword_aux_inner(const dev_mem_t<eT> mem,
                               const dev_mem_t<uword> uword_mem,
                               const uword use_uword_mem,
                               const uword n_elem,
                               dev_mem_t<aux_eT> aux_mem,
                               dev_mem_t<uword> aux_uword_mem,
                               const char* kernel_name,
                               const size_t total_num_threads,
                               const size_t local_group_size,
                               cl_kernel& first_kernel,
                               cl_kernel& first_kernel_small,
                               const std::tuple<A1...>& first_kernel_extra_args,
                               cl_kernel& second_kernel,
                               cl_kernel& second_kernel_small,
                               const std::tuple<A2...>& second_kernel_extra_args,
                               dev_mem_t<aux_eT> second_aux_mem,
                               dev_mem_t<uword> second_uword_aux_mem)
  {
  if (total_num_threads <= local_group_size)
    {
    // Only one reduce is necessary.
    generic_reduce_uword_aux_inner_small(mem,
                                         uword_mem,
                                         use_uword_mem,
                                         n_elem,
                                         aux_mem,
                                         aux_uword_mem,
                                         kernel_name,
                                         total_num_threads,
                                         local_group_size,
                                         first_kernel,
                                         first_kernel_small,
                                         first_kernel_extra_args);
    return true;
    }
  else
    {
    // Recompute size of auxiliary memory so that we know the size we get after
    // this pass.
    const uword aux_size = std::ceil((total_num_threads + (local_group_size - 1)) / local_group_size);

    runtime_t::cq_guard guard;

    runtime_t::adapt_uword dev_n_elem(n_elem);
    runtime_t::adapt_uword dev_use_uword_mem(use_uword_mem);
    runtime_t::adapt_uword dev_mem_offset(mem.cl_mem_ptr.offset);
    runtime_t::adapt_uword dev_uword_mem_offset(uword_mem.cl_mem_ptr.offset);
    runtime_t::adapt_uword dev_aux_mem_offset(aux_mem.cl_mem_ptr.offset);
    runtime_t::adapt_uword dev_aux_uword_mem_offset(aux_uword_mem.cl_mem_ptr.offset);

    // We need to round total_num_threads up to the next power of 2.  (The kernel assumes this.)
    const uword pow2_total_num_threads = (total_num_threads % local_group_size == 0) ? total_num_threads : ((total_num_threads / local_group_size) + 1) * local_group_size;

    cl_int status;
    status  = coot_wrapper(clSetKernelArg)(first_kernel, 0,  sizeof(cl_mem),                     &mem.cl_mem_ptr.ptr);
    status |= coot_wrapper(clSetKernelArg)(first_kernel, 1,  dev_mem_offset.size,                dev_mem_offset.addr);
    status |= coot_wrapper(clSetKernelArg)(first_kernel, 2,  sizeof(cl_mem),                     &uword_mem.cl_mem_ptr.ptr);
    status |= coot_wrapper(clSetKernelArg)(first_kernel, 3,  dev_uword_mem_offset.size,          dev_uword_mem_offset.addr);
    status |= coot_wrapper(clSetKernelArg)(first_kernel, 4,  dev_use_uword_mem.size,             dev_use_uword_mem.addr);
    status |= coot_wrapper(clSetKernelArg)(first_kernel, 5,  dev_n_elem.size,                    dev_n_elem.addr);
    status |= coot_wrapper(clSetKernelArg)(first_kernel, 6,  sizeof(cl_mem),                     &aux_mem.cl_mem_ptr.ptr);
    status |= coot_wrapper(clSetKernelArg)(first_kernel, 7,  dev_aux_mem_offset.size,            dev_aux_mem_offset.addr);
    status |= coot_wrapper(clSetKernelArg)(first_kernel, 8,  sizeof(cl_mem),                     &aux_uword_mem.cl_mem_ptr.ptr);
    status |= coot_wrapper(clSetKernelArg)(first_kernel, 9,  dev_aux_uword_mem_offset.size,      dev_aux_uword_mem_offset.addr);
    status |= coot_wrapper(clSetKernelArg)(first_kernel, 10, sizeof(aux_eT) * local_group_size,  NULL);
    status |= coot_wrapper(clSetKernelArg)(first_kernel, 11, dev_n_elem.size * local_group_size, NULL);

    // If we have any uwords in extra_args, we need to allocate adapt_uwords for them, which will be filled in set_extra_args().
    constexpr const uword num_uwords = count_uwords<void, A1...>();
    runtime_t::adapt_uword adapt_uwords[num_uwords == 0 ? 1 : num_uwords];
    uword adapt_uword_index = 0;
    status |= set_extra_args<sizeof...(A1), 12, A1...>::apply(first_kernel, adapt_uwords, adapt_uword_index, first_kernel_extra_args);
    coot_check_cl_error(status, std::string("coot::opencl::") + std::string(kernel_name) + std::string("()"));

    status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), first_kernel, 1, NULL, &pow2_total_num_threads, &local_group_size, 0, NULL, NULL);
    coot_check_cl_error(status, std::string("coot::opencl::") + std::string(kernel_name) + std::string("()"));

    size_t new_total_num_threads, new_local_group_size;
    reduce_kernel_group_info(second_kernel, aux_size, kernel_name, new_total_num_threads, new_local_group_size);

    return !generic_reduce_uword_aux_inner(aux_mem,
                                           aux_uword_mem,
                                           1 /* use the uword mem instead of the direct index */,
                                           aux_size,
                                           second_aux_mem,
                                           second_uword_aux_mem,
                                           kernel_name,
                                           new_total_num_threads,
                                           new_local_group_size,
                                           second_kernel,
                                           second_kernel_small,
                                           second_kernel_extra_args,
                                           second_kernel,
                                           second_kernel_small,
                                           second_kernel_extra_args,
                                           aux_mem,
                                           aux_uword_mem);
    }
  }



template<typename eT, typename aux_eT, typename... Args>
inline
void
generic_reduce_uword_aux_inner_small(const dev_mem_t<eT> mem,
                                     const dev_mem_t<uword> uword_mem,
                                     const uword use_uword_mem,
                                     const uword n_elem,
                                     dev_mem_t<aux_eT> aux_mem,
                                     dev_mem_t<uword> aux_uword_mem,
                                     const char* kernel_name,
                                     const size_t total_num_threads,
                                     const size_t local_group_size,
                                     cl_kernel& kernel,
                                     cl_kernel& kernel_small,
                                     const std::tuple<Args...>& first_kernel_extra_args)
  {
  const uword subgroup_size = get_rt().cl_rt.get_subgroup_size();

  runtime_t::cq_guard guard;

  runtime_t::adapt_uword dev_n_elem(n_elem);
  runtime_t::adapt_uword dev_use_uword_mem(use_uword_mem);
  runtime_t::adapt_uword dev_mem_offset(mem.cl_mem_ptr.offset);
  runtime_t::adapt_uword dev_uword_mem_offset(uword_mem.cl_mem_ptr.offset);
  runtime_t::adapt_uword dev_aux_mem_offset(aux_mem.cl_mem_ptr.offset);
  runtime_t::adapt_uword dev_aux_uword_mem_offset(aux_uword_mem.cl_mem_ptr.offset);

  // We need to round total_num_threads up to the next power of 2.  (The kernel assumes this.)
  const uword pow2_total_num_threads = (total_num_threads % local_group_size == 0) ? total_num_threads : ((total_num_threads / local_group_size) + 1) * local_group_size;

  // If the number of threads is less than the subgroup size (if subgroups are
  // available), then we can use a more optimized kernel with subgroup barriers
  // only.
  cl_kernel* k_use = (local_group_size <= subgroup_size) ? &kernel_small : &kernel;

  cl_int status;
  status  = coot_wrapper(clSetKernelArg)(*k_use, 0,  sizeof(cl_mem),                     &mem.cl_mem_ptr.ptr);
  status |= coot_wrapper(clSetKernelArg)(*k_use, 1,  dev_mem_offset.size,                dev_mem_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(*k_use, 2,  sizeof(cl_mem),                     &uword_mem.cl_mem_ptr.ptr);
  status |= coot_wrapper(clSetKernelArg)(*k_use, 3,  dev_uword_mem_offset.size,          dev_uword_mem_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(*k_use, 4,  dev_use_uword_mem.size,             dev_use_uword_mem.addr);
  status |= coot_wrapper(clSetKernelArg)(*k_use, 5,  dev_n_elem.size,                    dev_n_elem.addr);
  status |= coot_wrapper(clSetKernelArg)(*k_use, 6,  sizeof(cl_mem),                     &aux_mem.cl_mem_ptr.ptr);
  status |= coot_wrapper(clSetKernelArg)(*k_use, 7,  dev_aux_mem_offset.size,            dev_aux_mem_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(*k_use, 8,  sizeof(cl_mem),                     &aux_uword_mem.cl_mem_ptr.ptr);
  status |= coot_wrapper(clSetKernelArg)(*k_use, 9,  dev_aux_uword_mem_offset.size,      dev_aux_uword_mem_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(*k_use, 10, sizeof(aux_eT) * local_group_size,  NULL);
  status |= coot_wrapper(clSetKernelArg)(*k_use, 11, dev_n_elem.size * local_group_size, NULL);

  // If we have any uwords in extra_args, we need to allocate adapt_uwords for them, which will be filled in set_extra_args().
  constexpr const uword num_uwords = count_uwords<void, Args...>();
  runtime_t::adapt_uword adapt_uwords[num_uwords == 0 ? 1 : num_uwords];
  uword adapt_uword_index = 0;
  status |= set_extra_args<sizeof...(Args), 12, Args...>::apply(*k_use, adapt_uwords, adapt_uword_index, first_kernel_extra_args);
  coot_check_cl_error(status, std::string("coot::opencl::") + std::string(kernel_name) + std::string("()"));

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), *k_use, 1, NULL, &pow2_total_num_threads, &local_group_size, 0, NULL, NULL);
  coot_check_cl_error(status, std::string("coot::opencl::") + std::string(kernel_name) + std::string("()"));
  }
