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



template<typename T> inline constexpr uword is_uword()        { return 0; }
template<>           inline constexpr uword is_uword<uword>() { return 1; }



template<typename... Args>
inline
constexpr
uword count_uwords();



template<typename T, typename... Args>
inline
constexpr
uword count_uword_helper()
  {
  return is_uword<T>() + count_uwords<Args...>();
  }



template<typename... Args>
inline
constexpr
uword count_uwords()
  {
  return count_uword_helper<Args...>();
  }



template<>
inline
constexpr
uword count_uwords<>()
  {
  return 0;
  }



template<typename T>
inline
cl_int
set_extra_arg(cl_kernel& kernel,
              const uword index,
              runtime_t::adapt_uword* adapt_uwords,
              uword& adapt_uword_index,
              const T& arg)
  {
  coot_ignore(adapt_uwords);
  coot_ignore(adapt_uword_index);
  // The addition of 4 is to account for the first four arguments that every
  // generic reduce kernel must have.
  return coot_wrapper(clSetKernelArg)(kernel, 4 + index, sizeof(T), &arg);
  }



template<>
inline
cl_int
set_extra_arg(cl_kernel& kernel,
              const uword index,
              runtime_t::adapt_uword* adapt_uwords,
              uword& adapt_uword_index,
              const uword& arg)
  {
  adapt_uwords[adapt_uword_index] = runtime_t::adapt_uword(arg);
  // The addition of 4 is to account for the first four arguments that every
  // generic reduce kernel must have.
  cl_int status = coot_wrapper(clSetKernelArg)(kernel, 4 + index, adapt_uwords[adapt_uword_index].size, adapt_uwords[adapt_uword_index].addr);
  adapt_uword_index++;
  return status;
  }



template<size_t i, typename... Args>
struct
set_extra_args
  {
  inline static cl_int apply(cl_kernel& kernel,
                             runtime_t::adapt_uword* adapt_uwords,
                             uword& adapt_uword_index,
                             const std::tuple<Args...> args)
    {
    cl_int status = set_extra_arg(kernel, i - 1, adapt_uwords, adapt_uword_index, std::get<i - 1>(args));
    return status | set_extra_args<i - 1, Args...>::apply(kernel, i - 1, adapt_uwords, adapt_uword_index, args);
    }
  };



template<typename... Args>
struct
set_extra_args<1, Args...>
  {
  inline static cl_int apply(cl_kernel& kernel,
                             runtime_t::adapt_uword* adapt_uwords,
                             uword& adapt_uword_index,
                             const std::tuple<Args...> args)
    {
    return set_extra_arg(kernel, 0, adapt_uwords, adapt_uword_index, std::get<0>(args));
    }
  };



template<typename... Args>
struct
set_extra_args<0, Args...>
  {
  inline static cl_int apply(cl_kernel& kernel,
                             runtime_t::adapt_uword* adapt_uwords,
                             uword& adapt_uword_index,
                             const std::tuple<Args...> args)
    {
    coot_ignore(kernel);
    coot_ignore(adapt_uwords);
    coot_ignore(adapt_uword_index);
    coot_ignore(args);
    return CL_SUCCESS;
    }
  };



template<typename eT, typename aux_eT, typename... A1, typename... A2>
inline
aux_eT
generic_reduce(const dev_mem_t<eT> mem,
               const uword n_elem,
               const char* kernel_name,
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

  const uword total_num_threads = std::ceil(n_elem / std::max(1.0, (2 * std::ceil(std::log2(n_elem)))));
  const uword local_group_size = std::min(kernel_wg_size, total_num_threads);

  // Create auxiliary memory.
  const uword first_aux_size = std::ceil((total_num_threads + (local_group_size - 1)) / local_group_size);
  const uword second_aux_size = (first_aux_size == 1) ? 0 : std::ceil((first_aux_size + (local_group_size - 1)) / local_group_size);
  Col<aux_eT> first_aux(first_aux_size);
  Col<aux_eT> second_aux(second_aux_size);

  dev_mem_t<aux_eT> first_aux_mem_ptr = first_aux.get_dev_mem(false);
  // Just use the first pointer if there is no need for secondary auxiliary
  // space.
  dev_mem_t<aux_eT> second_aux_mem_ptr = (second_aux_size == 0) ? first_aux_mem_ptr : second_aux.get_dev_mem(false);;

  const bool first_buffer = generic_reduce_inner(mem,
                                                 n_elem,
                                                 first_aux_mem_ptr,
                                                 kernel_name,
                                                 kernel_wg_size,
                                                 first_kernel,
                                                 first_kernel_small,
                                                 first_kernel_extra_args,
                                                 second_kernel,
                                                 second_kernel_small,
                                                 second_kernel_extra_args,
                                                 second_aux_mem_ptr);
  return (first_buffer) ? aux_eT(first_aux[0]) : aux_eT(second_aux[0]);
  }



template<typename eT, typename aux_eT, typename... Args>
inline
aux_eT
generic_reduce(const dev_mem_t<eT> mem,
               const uword n_elem,
               const char* kernel_name,
               cl_kernel& kernel,
               cl_kernel& kernel_small,
               const std::tuple<Args...>& kernel_extra_args)
  {
  return generic_reduce<eT, aux_eT>(mem,
                                    n_elem,
                                    kernel_name,
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
generic_reduce_inner(const dev_mem_t<eT> mem,
                     const uword n_elem,
                     dev_mem_t<aux_eT> aux_mem,
                     const char* kernel_name,
                     const size_t kernel_wg_size,
                     cl_kernel& first_kernel,
                     cl_kernel& first_kernel_small,
                     const std::tuple<A1...>& first_kernel_extra_args,
                     cl_kernel& second_kernel,
                     cl_kernel& second_kernel_small,
                     const std::tuple<A2...>& second_kernel_extra_args,
                     dev_mem_t<aux_eT> second_aux_mem)
  {
  const uword total_num_threads = std::ceil(n_elem / std::max(1.0, (2 * std::ceil(std::log2(n_elem)))));

  if (total_num_threads <= kernel_wg_size)
    {
    // Only one reduce is necessary.
    generic_reduce_inner_small(mem,
                               n_elem,
                               aux_mem,
                               kernel_name,
                               kernel_wg_size,
                               first_kernel,
                               first_kernel_small,
                               first_kernel_extra_args);
    return true;
    }
  else
    {
    const size_t k1_work_dim       = 1;
    const size_t k1_work_offset    = 0;
    const uword local_group_size = std::min(kernel_wg_size, total_num_threads);

    // Recompute size of auxiliary memory so that we know the size we get after
    // this pass.
    const uword aux_size = std::ceil((total_num_threads + (local_group_size - 1)) / local_group_size);

    runtime_t::cq_guard guard;

    runtime_t::adapt_uword dev_n_elem(n_elem);

    // We need to round total_num_threads up to the next power of 2.  (The kernel assumes this.)
    const uword pow2_group_size = (uword) std::pow(2.0f, std::ceil(std::log2((float) local_group_size)));
    const uword pow2_total_num_threads = (total_num_threads % pow2_group_size == 0) ? total_num_threads : ((total_num_threads / pow2_group_size) + 1) * pow2_group_size;

    cl_int status;
    status  = coot_wrapper(clSetKernelArg)(first_kernel, 0, sizeof(cl_mem),                   &mem.cl_mem_ptr);
    status |= coot_wrapper(clSetKernelArg)(first_kernel, 1, dev_n_elem.size,                  dev_n_elem.addr);
    status |= coot_wrapper(clSetKernelArg)(first_kernel, 2, sizeof(cl_mem),                   &aux_mem.cl_mem_ptr);
    status |= coot_wrapper(clSetKernelArg)(first_kernel, 3, sizeof(aux_eT) * pow2_group_size, NULL);

    // If we have any uwords in extra_args, we need to allocate adapt_uwords for them, which will be filled in set_extra_args().
    constexpr const uword num_uwords = count_uwords<void, A1...>();
    runtime_t::adapt_uword adapt_uwords[num_uwords == 0 ? 1 : num_uwords];
    uword adapt_uword_index = 0;
    status |= set_extra_args<sizeof...(A1), A1...>::apply(first_kernel, adapt_uwords, adapt_uword_index, first_kernel_extra_args);

    coot_check_cl_error(status, std::string("coot::opencl::") + std::string(kernel_name) + std::string("()"));

    status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), first_kernel, k1_work_dim, &k1_work_offset, &pow2_total_num_threads, &pow2_group_size, 0, NULL, NULL);

    coot_check_cl_error(status, std::string("coot::opencl::") + std::string(kernel_name) + std::string("()"));

    return !generic_reduce_inner(aux_mem,
                                 aux_size,
                                 second_aux_mem,
                                 kernel_name,
                                 kernel_wg_size,
                                 second_kernel,
                                 second_kernel_small,
                                 second_kernel_extra_args,
                                 second_kernel,
                                 second_kernel_small,
                                 second_kernel_extra_args,
                                 aux_mem);
    }
  }



template<typename eT, typename aux_eT, typename... Args>
inline
void
generic_reduce_inner_small(const dev_mem_t<eT> mem,
                           const uword n_elem,
                           dev_mem_t<aux_eT> aux_mem,
                           const char* kernel_name,
                           const size_t kernel_wg_size,
                           cl_kernel& kernel,
                           cl_kernel& kernel_small,
                           const std::tuple<Args...>& first_kernel_extra_args)
  {
  const size_t k1_work_dim       = 1;
  const size_t k1_work_offset    = 0;
  const uword total_num_threads = std::ceil(n_elem / std::max(1.0, (2 * std::ceil(std::log2(n_elem)))));
  const uword subgroup_size = get_rt().cl_rt.get_subgroup_size();
  const uword local_group_size = std::min(kernel_wg_size, total_num_threads);

  runtime_t::cq_guard guard;

  runtime_t::adapt_uword dev_n_elem(n_elem);

  // We need to round total_num_threads up to the next power of 2.  (The kernel assumes this.)
  const uword pow2_group_size = (uword) std::pow(2.0f, std::ceil(std::log2((float) local_group_size)));
  const uword pow2_total_num_threads = (total_num_threads % pow2_group_size == 0) ? total_num_threads : ((total_num_threads / pow2_group_size) + 1) * pow2_group_size;

  // If the number of threads is less than the subgroup size (if subgroups are
  // available), then we can use a more optimized kernel with subgroup barriers
  // only.
  cl_kernel* k_use = (pow2_group_size <= subgroup_size) ? &kernel_small : &kernel;

  cl_int status;
  status  = coot_wrapper(clSetKernelArg)(*k_use, 0, sizeof(cl_mem),                   &mem.cl_mem_ptr);
  status |= coot_wrapper(clSetKernelArg)(*k_use, 1, dev_n_elem.size,                  dev_n_elem.addr);
  status |= coot_wrapper(clSetKernelArg)(*k_use, 2, sizeof(cl_mem),                   &aux_mem.cl_mem_ptr);
  status |= coot_wrapper(clSetKernelArg)(*k_use, 3, sizeof(aux_eT) * pow2_group_size, NULL);

  // If we have any uwords in extra_args, we need to allocate adapt_uwords for them, which will be filled in set_extra_args().
  constexpr const uword num_uwords = count_uwords<void, Args...>();
  runtime_t::adapt_uword adapt_uwords[num_uwords == 0 ? 1 : num_uwords];
  uword adapt_uword_index = 0;
  status |= set_extra_args<sizeof...(Args), Args...>::apply(*k_use, adapt_uwords, adapt_uword_index, first_kernel_extra_args);

  coot_check_cl_error(status, std::string("coot::opencl::") + std::string(kernel_name) + std::string("()"));

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), *k_use, k1_work_dim, &k1_work_offset, &pow2_total_num_threads, &pow2_group_size, 0, NULL, NULL);

  coot_check_cl_error(status, std::string("coot::opencl::") + std::string(kernel_name) + std::string("()"));
  }
