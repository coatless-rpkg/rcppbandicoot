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



// Utility to run a full reduce in an efficient and generic way.
//
// Note that all kernels that are used with generic uword-aux reduce are expected to compute an eT,
// and the first seven arguments should be:
//  - const eT1* mem
//  - const UWORD* uword_mem
//  - const UWORD use_uword_mem
//  - const UWORD n_elem
//  - eT1* out_mem
//  - UWORD* out_uwords
//  - const UWORD uword_aux_mem_start
// Additional arguments are fine, so long as those first seven are the same.

template<typename eT, typename aux_eT, typename... A1, typename... A2>
inline
uword
generic_reduce_uword_aux(const dev_mem_t<eT> mem,
                         const uword n_elem,
                         const char* kernel_name,
                         eT* reduce_result,
                         CUfunction& first_kernel,
                         CUfunction& first_kernel_small,
                         const std::tuple<A1...>& first_kernel_extra_args,
                         CUfunction& second_kernel,
                         CUfunction& second_kernel_small,
                         const std::tuple<A2...>& second_kernel_extra_args)
  {
  // Do first pass, hand off to appropriate smaller reduce if needed.
  // The first pass will use the first kernel; subsequent passes use the second kernel.
  const size_t n_elem_per_thread = std::ceil(n_elem / std::max(1.0, (2 * std::ceil(std::log2(std::max((double) n_elem, 2.0))))));
  const size_t mtpb = (size_t) get_rt().cuda_rt.dev_prop.maxThreadsPerBlock;

  // Compute size of auxiliary memory.
  const size_t first_aux_mem_size = (n_elem_per_thread + mtpb - 1) / mtpb;
  const size_t second_aux_mem_size = (first_aux_mem_size == 1) ? 0 : (first_aux_mem_size + mtpb - 1) / mtpb;
  const size_t aux_mem_size = first_aux_mem_size + second_aux_mem_size;
  Col<aux_eT> aux_mem(aux_mem_size);
  Col<uword> aux_uword_mem(aux_mem_size);

  dev_mem_t<aux_eT> aux_mem_ptr = aux_mem.get_dev_mem(false);
  dev_mem_t<uword> aux_uword_mem_ptr = aux_uword_mem.get_dev_mem(false);
  // Create offset for secondary buffer.
  dev_mem_t<aux_eT> second_aux_mem_ptr = aux_mem_ptr;
  dev_mem_t<uword> second_aux_uword_mem_ptr = aux_uword_mem_ptr;
  second_aux_mem_ptr.cuda_mem_ptr += first_aux_mem_size;
  second_aux_uword_mem_ptr.cuda_mem_ptr += first_aux_mem_size;
  const bool first_buffer = generic_reduce_uword_aux_inner(mem,
                                                           second_aux_uword_mem_ptr /* dummy */,
                                                           0 /* don't use uword memory */,
                                                           n_elem,
                                                           aux_mem_ptr,
                                                           aux_uword_mem_ptr,
                                                           kernel_name,
                                                           first_kernel,
                                                           first_kernel_small,
                                                           first_kernel_extra_args,
                                                           second_kernel,
                                                           second_kernel_small,
                                                           second_kernel_extra_args,
                                                           second_aux_mem_ptr,
                                                           second_aux_uword_mem_ptr);

  // Extract the reduce result, if needed.
  if (reduce_result != nullptr)
    {
    (*reduce_result) = (first_buffer ? eT(aux_mem[0]) : eT(aux_mem[first_aux_mem_size]));
    }

  return first_buffer ? uword(aux_uword_mem[0]) : uword(aux_uword_mem[first_aux_mem_size]);
  }



// This version uses the same kernel for all reduce passes.
template<typename eT, typename aux_eT, typename... Args>
inline
uword
generic_reduce_uword_aux(const dev_mem_t<eT> mem,
                         const uword n_elem,
                         const char* kernel_name,
                         eT* reduce_result,
                         CUfunction& kernel,
                         CUfunction& kernel_small,
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
                               CUfunction& first_kernel,
                               CUfunction& first_kernel_small,
                               const std::tuple<A1...>& first_kernel_extra_args,
                               CUfunction& second_kernel,
                               CUfunction& second_kernel_small,
                               const std::tuple<A2...>& second_kernel_extra_args,
                               dev_mem_t<aux_eT> second_aux_mem,
                               dev_mem_t<uword> second_aux_uword_mem)
  {
  const size_t mtpb = (size_t) get_rt().cuda_rt.dev_prop.maxThreadsPerBlock;
  const size_t max_num_threads = std::ceil(n_elem / std::max(1.0, (2 * std::ceil(std::log2(std::max((double) n_elem, 2.0))))));

  if (max_num_threads <= mtpb)
    {
    // If the data is small enough, we can do the work in a single pass.
    generic_reduce_uword_aux_inner_small(mem,
                                         uword_mem,
                                         use_uword_mem,
                                         n_elem,
                                         max_num_threads,
                                         aux_mem,
                                         aux_uword_mem,
                                         kernel_name,
                                         first_kernel,
                                         first_kernel_small,
                                         first_kernel_extra_args);
    return true;
    }
  else
    {
    // Here, we will have to do multiple reduces.
    const size_t block_size = (max_num_threads + mtpb - 1) / mtpb;

    // Compute the size of auxiliary memory.  The start of each memory section must be aligned to the size of the type, so we may require padding at the end of the aux_eTs.
    const size_t aux_eT_mem_size = (2 * mtpb * sizeof(aux_eT));
    const size_t uword_mem_start = ((aux_eT_mem_size + sizeof(uword) - 1) / sizeof(uword)) * sizeof(uword);
    const size_t uword_mem_size = (2 * mtpb * sizeof(uword));

    const void* args[7 + sizeof...(A1)];
    args[0] = &mem.cuda_mem_ptr;
    args[1] = &uword_mem.cuda_mem_ptr;
    args[2] = &use_uword_mem;
    args[3] = &n_elem;
    args[4] = &aux_mem.cuda_mem_ptr;
    args[5] = &aux_uword_mem.cuda_mem_ptr;
    args[6] = &uword_mem_start;
    unpack_args<sizeof...(A1), 7, A1...>::apply(args, first_kernel_extra_args);

    CUresult result = coot_wrapper(cuLaunchKernel)(
        first_kernel,
        block_size, 1, 1, mtpb, 1, 1,
        uword_mem_start + uword_mem_size,
        NULL,
        (void**) args,
        0);

    coot_check_cuda_error(result, std::string("coot::cuda::") + std::string(kernel_name) + std::string(": cuLaunchKernel() failed"));

    // Now, take subsequent passes.
    // We use the second kernel for all subsequent passes.
    // The '!' on the return value flips whether or not the final result is in the first/second aux mem buffer.
    return !generic_reduce_uword_aux_inner(aux_mem,
                                           aux_uword_mem,
                                           1 /* now use uword indexing */,
                                           block_size,
                                           second_aux_mem,
                                           second_aux_uword_mem,
                                           kernel_name,
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
                                     const uword max_num_threads,
                                     dev_mem_t<aux_eT> aux_mem, // must have at least one element
                                     dev_mem_t<uword> aux_uword_mem,
                                     const char* kernel_name,
                                     CUfunction& kernel,
                                     CUfunction& kernel_small, // for 32 threads or fewer
                                     const std::tuple<Args...>& kernel_extra_args)
  {
  const uword pow2_num_threads = next_pow2(max_num_threads);

  // Compute the size of auxiliary memory.  The start of each memory section must be aligned to the size of the type, so we may require padding at the end of the aux_eTs.
  const size_t aux_eT_mem_size = (2 * pow2_num_threads * sizeof(aux_eT));
  const size_t uword_mem_start = ((aux_eT_mem_size + sizeof(uword) - 1) / sizeof(uword)) * sizeof(uword);
  const size_t uword_mem_size = (2 * pow2_num_threads * sizeof(uword));

  const void* args[7 + sizeof...(Args)];
  args[0] = &mem.cuda_mem_ptr;
  args[1] = &uword_mem.cuda_mem_ptr;
  args[2] = &use_uword_mem;
  args[3] = &n_elem;
  args[4] = &aux_mem.cuda_mem_ptr;
  args[5] = &aux_uword_mem.cuda_mem_ptr;
  args[6] = &uword_mem_start;
  unpack_args<sizeof...(Args), 7, Args...>::apply(args, kernel_extra_args);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      pow2_num_threads <= 32 ? kernel_small : kernel, // if we have fewer threads than a single warp, we can use a more optimized version of the kernel
      1, 1, 1, pow2_num_threads, 1, 1,
      uword_mem_start + uword_mem_size,
      NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, std::string("coot::cuda::") + std::string(kernel_name) + std::string(": cuLaunchKernel() failed"));
  }
