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



template<typename eT>
inline
void
sort_index_vec(dev_mem_t<uword> out, dev_mem_t<eT> A, const uword n_elem, const uword sort_type, const uword stable_sort)
  {
  coot_extra_debug_sigprint();

  runtime_t::cq_guard guard;

  cl_kernel k;
  if (stable_sort == 0 && sort_type == 0)
    {
    k = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::radix_sort_index_ascending);
    }
  else if (stable_sort == 0 && sort_type == 1)
    {
    k = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::radix_sort_index_descending);
    }
  else if (stable_sort == 1 && sort_type == 0)
    {
    k = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::stable_radix_sort_index_ascending);
    }
  else
    {
    k = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::stable_radix_sort_index_descending);
    }

  size_t kernel_wg_size;
  cl_int status = coot_wrapper(clGetKernelWorkGroupInfo)(k, get_rt().cl_rt.get_device(), CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &kernel_wg_size, NULL);
  coot_check_cl_error(status, "coot::opencl::sort_index_vec(): clGetKernelWorkGroupInfo() failed");

  const size_t total_num_threads = std::ceil(n_elem / std::max(1.0, (2 * std::ceil(std::log2(n_elem)))));
  const size_t pow2_num_threads = std::min(kernel_wg_size, (size_t) std::pow(2.0f, std::ceil(std::log2((float) total_num_threads))));

  // First, allocate temporary matrices we will use during computation.
  dev_mem_t<eT> tmp_mem;
  tmp_mem.cl_mem_ptr = get_rt().cl_rt.acquire_memory<eT>(n_elem);
  dev_mem_t<uword> tmp_mem_index;
  tmp_mem_index.cl_mem_ptr = get_rt().cl_rt.acquire_memory<uword>(n_elem);

  status = 0;

  const size_t aux_mem_size = (stable_sort == 0) ? (2 * sizeof(eT) * pow2_num_threads) : (4 * sizeof(eT) * pow2_num_threads);

  runtime_t::adapt_uword cl_n_elem(n_elem);

  status |= coot_wrapper(clSetKernelArg)(k, 0, sizeof(cl_mem), &(A.cl_mem_ptr));
  status |= coot_wrapper(clSetKernelArg)(k, 1, sizeof(cl_mem), &(out.cl_mem_ptr));
  status |= coot_wrapper(clSetKernelArg)(k, 2, sizeof(cl_mem), &(tmp_mem.cl_mem_ptr));
  status |= coot_wrapper(clSetKernelArg)(k, 3, sizeof(cl_mem), &(tmp_mem_index.cl_mem_ptr));
  status |= coot_wrapper(clSetKernelArg)(k, 4, cl_n_elem.size, cl_n_elem.addr);
  status |= coot_wrapper(clSetKernelArg)(k, 5, aux_mem_size,   NULL);

  coot_check_cl_error(status, "coot::opencl::sort_index_vec(): failed to set kernel arguments");

  const size_t k1_work_dim       = 1;
  const size_t k1_work_offset[1] = { 0 };
  const size_t k1_work_size[1]   = { pow2_num_threads };

  status = coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), k, k1_work_dim, k1_work_offset, k1_work_size, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "coot::opencl::sort_index_vec(): failed to run kernel");

  get_rt().cl_rt.synchronise();
  get_rt().cl_rt.release_memory(tmp_mem.cl_mem_ptr);
  get_rt().cl_rt.release_memory(tmp_mem_index.cl_mem_ptr);
  }
