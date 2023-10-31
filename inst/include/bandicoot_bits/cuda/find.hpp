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
 * Find indices of nonzero values in the given vector.
 */
template<typename eT>
inline
void
find(dev_mem_t<uword>& out, uword& out_len, const dev_mem_t<eT> A, const uword n_elem, const uword k, const uword find_type)
  {
  coot_extra_debug_sigprint();

  // If the vector is empty, don't do anything.
  if (n_elem == 0)
    {
    out.cuda_mem_ptr = NULL;
    out_len = 0;
    return;
    }

  // The kernel requires that all threads are in one block.
  const size_t mtpb = (size_t) get_rt().cuda_rt.dev_prop.maxThreadsPerBlock;
  const size_t num_threads = std::min(mtpb, size_t(std::ceil(n_elem / std::max(1.0, (2 * std::ceil(std::log2(n_elem)))))));
  // The number of threads needs to be a power of two.
  const size_t pow2_num_threads = std::min(mtpb, (size_t) std::pow(2.0f, std::ceil(std::log2((float) num_threads))));

  // First, allocate temporary memory for the prefix sum.
  dev_mem_t<uword> counts_mem;
  counts_mem.cuda_mem_ptr = get_rt().cuda_rt.acquire_memory<uword>(pow2_num_threads + 1);

  CUfunction nnz_k = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::count_nonzeros);

  const void* nnz_args[] = {
      &(A.cuda_mem_ptr),
      &(counts_mem.cuda_mem_ptr),
      (uword*) &n_elem };

  CUresult result = coot_wrapper(cuLaunchKernel)(
      nnz_k,
      1, 1, 1, pow2_num_threads, 1, 1,
      sizeof(uword) * pow2_num_threads,
      NULL,
      (void**) nnz_args,
      0);

  coot_check_cuda_error(result, "coot::cuda::find(): cuLaunchKernel() failed for count_nonzeros kernel");

  get_rt().cuda_rt.synchronise();

  const uword total_nonzeros = get_val(counts_mem, pow2_num_threads);
  out_len = (k == 0) ? total_nonzeros : (std::min)(k, total_nonzeros);
  out.cuda_mem_ptr = get_rt().cuda_rt.acquire_memory<uword>(out_len);

  if (out_len == 0)
    {
    // There are no nonzero values---we're done.
    return;
    }

  if (k == 0 || total_nonzeros < k)
    {
    // Get all nonzero elements.
    CUfunction find_k = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::find);

    const void* find_args[] = {
        &(A.cuda_mem_ptr),
        &(counts_mem.cuda_mem_ptr),
        &(out.cuda_mem_ptr),
        (uword*) &n_elem };

    result = coot_wrapper(cuLaunchKernel)(
        find_k,
        1, 1, 1, pow2_num_threads, 1, 1,
        0, NULL, (void**) find_args, 0);

    coot_check_cuda_error(result, "coot::cuda::find(): cuLaunchKernel() failed for find kernel");
    }
  else if (find_type == 0)
    {
    // Get first `k` nonzero elements.
    CUfunction find_k = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::find_first);

    const void* find_args[] = {
        &(A.cuda_mem_ptr),
        &(counts_mem.cuda_mem_ptr),
        &(out.cuda_mem_ptr),
        (uword*) &k,
        (uword*) &n_elem };

    result = coot_wrapper(cuLaunchKernel)(
        find_k,
        1, 1, 1, pow2_num_threads, 1, 1,
        0, NULL, (void**) find_args, 0);

    coot_check_cuda_error(result, "coot::cuda::find(): cuLaunchKernel() failed for find_first kernel");
    }
  else
    {
    // Get last `k` nonzero elements.
    CUfunction find_k = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::find_last);

    const uword m = total_nonzeros - k;

    const void* find_args[] = {
        &(A.cuda_mem_ptr),
        &(counts_mem.cuda_mem_ptr),
        &(out.cuda_mem_ptr),
        (uword*) &m,
        (uword*) &n_elem };

    result = coot_wrapper(cuLaunchKernel)(
        find_k,
        1, 1, 1, pow2_num_threads, 1, 1,
        0, NULL, (void**) find_args, 0);

    coot_check_cuda_error(result, "coot::cuda::find(): cuLaunchKernel() failed for find_last kernel");
    }

  get_rt().cuda_rt.release_memory(counts_mem.cuda_mem_ptr);
  }
