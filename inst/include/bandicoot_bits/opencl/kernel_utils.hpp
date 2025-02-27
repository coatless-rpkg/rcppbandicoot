// Copyright 2023 Ryan Curtin (http://www.ratml.org/)
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



// Compute the number of threads to use for a one-dimensional reduce.
// Note that this should be used as *both* the global size and the local size.
// The name should be the function name, without any namespace specified.



inline
size_t
reduce_kernel_total_num_threads(const size_t n_elem)
  {
  return std::ceil(n_elem / std::max(1.0, (2 * std::ceil(std::log2(n_elem)))));
  }



inline
void
reduce_kernel_group_info(const cl_kernel& kernel, const uword n_elem, const char* name, size_t& total_num_threads, size_t& local_group_size)
  {
  // Compute the preferred workgroup size for the kernel.  This will be a maximum on the number of threads.
  size_t kernel_wg_size;
  cl_int status = coot_wrapper(clGetKernelWorkGroupInfo)(kernel, get_rt().cl_rt.get_device(), CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &kernel_wg_size, NULL);
  coot_check_cl_error(status, std::string("coot::opencl::") + std::string(name) + std::string("(): clGetKernelWorkGroupInfo() failed"));

  // We must ensure that the number of threads is fewer than the maximum number of threads in one dimension.
  const size_t max_wg_dim_size = get_rt().cl_rt.get_max_wg_dim(0);

  // TODO: optimize this computation
  total_num_threads = reduce_kernel_total_num_threads(n_elem);
  const size_t pow2_num_threads = next_pow2(total_num_threads);
  local_group_size = std::min(std::min(max_wg_dim_size, kernel_wg_size), pow2_num_threads);
  }



inline
size_t
reduce_kernel_group_size(const cl_kernel& kernel, const uword n_elem, const char* name)
  {
  size_t total_num_threads, local_group_size;
  reduce_kernel_group_info(kernel, n_elem, name, total_num_threads, local_group_size);
  return local_group_size;
  }
