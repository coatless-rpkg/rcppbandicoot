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
 * Sort the data in the block of memory.
 */
template<typename eT>
inline
void
sort_index_vec(dev_mem_t<uword> out, dev_mem_t<eT> A, const uword n_elem, const uword sort_type, const uword stable_sort)
  {
  coot_extra_debug_sigprint();

  // If the vector is empty, don't do anything.
  if (n_elem == 0)
    {
    return;
    }

  // The kernel requires that all threads are in one block.
  const size_t mtpb = (size_t) get_rt().cuda_rt.dev_prop.maxThreadsPerBlock;
  const size_t num_threads = std::min(mtpb, size_t(std::ceil(n_elem / std::max(1.0, (2 * std::ceil(std::log2(std::max((double) n_elem, 2.0))))))));
  // The number of threads needs to be a power of two.
  const size_t pow2_num_threads = std::min(mtpb, next_pow2(num_threads));

  // First, allocate temporary memory we will use during computation.
  dev_mem_t<eT> tmp_mem;
  tmp_mem.cuda_mem_ptr = get_rt().cuda_rt.acquire_memory<eT>(n_elem);
  dev_mem_t<uword> tmp_mem_index;
  tmp_mem_index.cuda_mem_ptr = get_rt().cuda_rt.acquire_memory<uword>(n_elem);

  CUfunction kernel;
  if (stable_sort == 0 && sort_type == 0)
    {
    kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::radix_sort_index_asc);
    }
  else if (stable_sort == 0 && sort_type == 1)
    {
    kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::radix_sort_index_desc);
    }
  else if (stable_sort == 1 && sort_type == 0)
    {
    kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::stable_radix_sort_index_asc);
    }
  else
    {
    kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::stable_radix_sort_index_desc);
    }

  const size_t aux_mem_size = (stable_sort == 0) ? 2 * pow2_num_threads * sizeof(uword) : 4 * pow2_num_threads * sizeof(uword);

  const void* args[] = {
      &(A.cuda_mem_ptr),
      &(out.cuda_mem_ptr),
      &(tmp_mem.cuda_mem_ptr),
      &(tmp_mem_index.cuda_mem_ptr),
      (uword*) &n_elem };

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      1, 1, 1, pow2_num_threads, 1, 1,
      aux_mem_size,
      NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, "coot::cuda::sort_index_vec(): cuLaunchKernel() failed");

  get_rt().cuda_rt.synchronise();
  get_rt().cuda_rt.release_memory(tmp_mem.cuda_mem_ptr);
  get_rt().cuda_rt.release_memory(tmp_mem_index.cuda_mem_ptr);
  }
