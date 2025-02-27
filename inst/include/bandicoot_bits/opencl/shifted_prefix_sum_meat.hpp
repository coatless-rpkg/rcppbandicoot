// Copyright 2024 Ryan Curtin (http://www.ratml.org)
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



// NOTE: this is not exactly prefix-sum as typically taught, but it's what's needed for radix search.  It returns a result that is shifted by one element.
// An input of [1, 3, 2, 4] returns an output of [0, 1, 4, 6]---*not* the "typical" output of [1, 4, 6, 10].



template<typename eT>
inline
void
shifted_prefix_sum_small(dev_mem_t<eT> mem, const uword n_elem, const size_t total_num_threads, const size_t local_group_size)
  {
  coot_extra_debug_sigprint();

  runtime_t::cq_guard guard;

  runtime_t::adapt_uword dev_n_elem(n_elem);
  runtime_t::adapt_uword dev_offset(mem.cl_mem_ptr.offset);

  // When the array is small enough, it fits in one workgroup and we can perform the entire operation in one kernel.
  // This kernel does both the upsweep and the downsweep in the same kernel, unlike the large variant.
  cl_kernel k = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::shifted_prefix_sum_small);

  cl_int status;
  status  = coot_wrapper(clSetKernelArg)(k, 0, sizeof(cl_mem),                    &mem.cl_mem_ptr.ptr);
  status |= coot_wrapper(clSetKernelArg)(k, 1, dev_offset.size,                   dev_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(k, 2, dev_n_elem.size,                   dev_n_elem.addr);
  status |= coot_wrapper(clSetKernelArg)(k, 3, sizeof(eT) * 2 * local_group_size, NULL);
  coot_check_cl_error(status, "coot::opencl::shifted_prefix_sum_small()");

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), k, 1, NULL, &total_num_threads, &local_group_size, 0, NULL, NULL);
  coot_check_cl_error(status, "coot::opencl::shifted_prefix_sum_small()");
  }



template<typename eT>
inline
void
shifted_prefix_sum_large(dev_mem_t<eT> mem, const uword n_elem, const size_t total_num_threads, const size_t local_group_size)
  {
  coot_extra_debug_sigprint();

  // For arrays larger than we can handle in a single work-group, we operate recursively:
  // we perform up-sweeps until it fits in a single workgroup;
  // then we prefix-sum the single workgroup with shifted_prefix_sum_small();
  // then we perform down-sweeps until the entire array is prefix-summed.

  const uword out_n_elem = ((n_elem / 2) + local_group_size - 1) / local_group_size;

  // After the up-sweep, this will hold the total number of elements in this local group.
  Col<eT> tmp(out_n_elem);
  dev_mem_t<eT> tmp_mem = tmp.get_dev_mem(false);

  runtime_t::cq_guard guard;

  runtime_t::adapt_uword dev_n_elem(n_elem);
  runtime_t::adapt_uword dev_offset(mem.cl_mem_ptr.offset);

  cl_kernel k_subgroups = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::shifted_prefix_sum_subgroups);

  cl_int status;
  status  = coot_wrapper(clSetKernelArg)(k_subgroups, 0, sizeof(cl_mem),                     &mem.cl_mem_ptr.ptr);
  status |= coot_wrapper(clSetKernelArg)(k_subgroups, 1, dev_offset.size,                    dev_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(k_subgroups, 2, sizeof(cl_mem),                     &tmp_mem.cl_mem_ptr.ptr);
  status |= coot_wrapper(clSetKernelArg)(k_subgroups, 3, dev_n_elem.size,                    dev_n_elem.addr);
  status |= coot_wrapper(clSetKernelArg)(k_subgroups, 4, sizeof(eT) * 2 * local_group_size, NULL);
  coot_check_cl_error(status, "coot::opencl::shifted_prefix_sum_large()");

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), k_subgroups, 1, NULL, &total_num_threads, &local_group_size, 0, NULL, NULL);
  coot_check_cl_error(status, "coot::opencl::shifted_prefix_sum_large()");

  // Now prefix-sum the result memory recursively before we take a second pass to add offsets to every workgroup's memory.
  // After this, tmp_mem will properly hold shifted-prefix-summed memory.
  shifted_prefix_sum(tmp_mem, out_n_elem);

  // Finally, perform the down-sweep.
  cl_kernel k_add_offset = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::shifted_prefix_sum_add_offset);

  status  = coot_wrapper(clSetKernelArg)(k_add_offset, 0, sizeof(cl_mem),  &mem.cl_mem_ptr.ptr);
  status |= coot_wrapper(clSetKernelArg)(k_add_offset, 1, dev_offset.size, dev_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(k_add_offset, 2, sizeof(cl_mem),  &tmp_mem.cl_mem_ptr.ptr);
  status |= coot_wrapper(clSetKernelArg)(k_add_offset, 3, dev_n_elem.size, dev_n_elem.addr);
  status |= coot_wrapper(clSetKernelArg)(k_add_offset, 4, sizeof(eT) * 2 * local_group_size, NULL);
  coot_check_cl_error(status, "coot::opencl::shifted_prefix_sum_large()");

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), k_add_offset, 1, NULL, &total_num_threads, &local_group_size, 0, NULL, NULL);
  coot_check_cl_error(status, "coot::opencl::shifted_prefix_sum_large()");
  }



template<typename eT>
inline
void
shifted_prefix_sum(dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (n_elem == 0)
    {
    return;
    }
  else if (n_elem == 1)
    {
    // Set the memory to 0.
    set_val(mem, 0, eT(0));
    return;
    }

  // Compute the number of threads we need to handle an array of this size.
  // Each thread will handle 2 memory elements.
  size_t kernel_wg_size;
  cl_kernel k = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::shifted_prefix_sum_small);
  cl_int status = coot_wrapper(clGetKernelWorkGroupInfo)(k, get_rt().cl_rt.get_device(), CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &kernel_wg_size, NULL);
  coot_check_cl_error(status, "coot::opencl::shifted_prefix_sum(): clGetKernelWorkGroupInfo() failed");

  const size_t max_wg_dim_size = get_rt().cl_rt.get_max_wg_dim(0);

  const size_t pow2_num_threads = next_pow2(n_elem / 2);
  const size_t local_group_size = std::min(std::min(max_wg_dim_size, kernel_wg_size), pow2_num_threads);

  if (pow2_num_threads == local_group_size)
    {
    shifted_prefix_sum_small(mem, n_elem, pow2_num_threads, local_group_size);
    }
  else
    {
    shifted_prefix_sum_large(mem, n_elem, pow2_num_threads, local_group_size);
    }
  }
