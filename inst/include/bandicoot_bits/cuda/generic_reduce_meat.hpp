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
// Note that all kernels that are used with generic reduce are expected to compute an eT,
// and the first three arguments should be:
//  - const eT1* mem
//  - const UWORD n_elem
//  - aux_eT* out_mem
// Additional arguments are fine, so long as those first three are the same.

template<typename eT, typename aux_eT, typename... A1, typename... A2>
inline
aux_eT
generic_reduce(const dev_mem_t<eT> mem,
               const uword n_elem,
               const char* kernel_name,
               CUfunction& first_kernel,
               CUfunction& first_kernel_small,
               const std::tuple<A1...>& first_kernel_extra_args,
               CUfunction& second_kernel,
               CUfunction& second_kernel_small,
               const std::tuple<A2...>& second_kernel_extra_args)
  {
  // Do first pass, hand off to appropriate smaller reduce if needed.
  // The first pass will use the first kernel; subsequent passes use the second kernel.
  const size_t n_elem_per_thread = std::ceil(n_elem / std::max(1.0, (2 * std::ceil(std::log2(n_elem)))));
  const size_t mtpb = (size_t) get_rt().cuda_rt.dev_prop.maxThreadsPerBlock;

  // Compute size of auxiliary memory.
  const size_t first_aux_mem_size = (n_elem_per_thread + mtpb - 1) / mtpb;
  const size_t second_aux_mem_size = (first_aux_mem_size == 1) ? 0 : (first_aux_mem_size + mtpb - 1) / mtpb;
  const size_t aux_mem_size = first_aux_mem_size + second_aux_mem_size;
  Col<aux_eT> aux_mem(aux_mem_size);

  dev_mem_t<aux_eT> aux_mem_ptr = aux_mem.get_dev_mem(false);
  // Create offset for secondary buffer.
  dev_mem_t<aux_eT> second_aux_mem_ptr = aux_mem_ptr;
  second_aux_mem_ptr.cuda_mem_ptr += first_aux_mem_size;
  const bool first_buffer = generic_reduce_inner(mem,
                                                 n_elem,
                                                 aux_mem_ptr,
                                                 kernel_name,
                                                 first_kernel,
                                                 first_kernel_small,
                                                 first_kernel_extra_args,
                                                 second_kernel,
                                                 second_kernel_small,
                                                 second_kernel_extra_args,
                                                 second_aux_mem_ptr);
  return first_buffer ? aux_eT(aux_mem[0]) : aux_eT(aux_mem[first_aux_mem_size]);
  }



// This version uses the same kernel for all reduce passes.
template<typename eT, typename aux_eT, typename... Args>
inline
aux_eT
generic_reduce(const dev_mem_t<eT> mem,
               const uword n_elem,
               const char* kernel_name,
               CUfunction& kernel,
               CUfunction& kernel_small,
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



// unpack_args is a metaprogramming utility to recursively iterate over the extra arguments for a kernel

template<size_t i, typename... Args>
struct
unpack_args
  {
  inline static void apply(const void** args, const std::tuple<Args...>& args_tuple)
    {
    args[3 + i] = &std::get<i - 1>(args_tuple);
    unpack_args<i - 1, Args...>::apply(args, args_tuple);
    }
  };



template<typename... Args>
struct
unpack_args<1, Args...>
  {
  inline static void apply(const void** args, const std::tuple<Args...>& args_tuple)
    {
    // This is the last iteration of the recursion.
    args[3] = &std::get<0>(args_tuple);
    }
  };



template<typename... Args>
struct
unpack_args<0, Args...>
  {
  inline static void apply(const void** args, const std::tuple<Args...>& args_tuple)
    {
    // This gets called when there are no arguments at all.
    coot_ignore(args);
    coot_ignore(args_tuple);
    }
  };



template<typename eT, typename aux_eT, typename... A1, typename... A2>
inline
bool
generic_reduce_inner(const dev_mem_t<eT> mem,
                     const uword n_elem,
                     dev_mem_t<aux_eT> aux_mem,
                     const char* kernel_name,
                     CUfunction& first_kernel,
                     CUfunction& first_kernel_small,
                     const std::tuple<A1...>& first_kernel_extra_args,
                     CUfunction& second_kernel,
                     CUfunction& second_kernel_small,
                     const std::tuple<A2...>& second_kernel_extra_args,
                     dev_mem_t<aux_eT> second_aux_mem)
  {
  const size_t mtpb = (size_t) get_rt().cuda_rt.dev_prop.maxThreadsPerBlock;
  const size_t max_num_threads = std::ceil(n_elem / std::max(1.0, (2 * std::ceil(std::log2(n_elem)))));

  if (max_num_threads <= mtpb)
    {
    // If the data is small enough, we can do the work in a single pass.
    generic_reduce_inner_small(mem,
                               n_elem,
                               max_num_threads,
                               aux_mem,
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

    const void* args[3 + sizeof...(A1)];
    args[0] = &mem.cuda_mem_ptr;
    args[1] = &n_elem;
    args[2] = &aux_mem.cuda_mem_ptr;
    unpack_args<sizeof...(A1), A1...>::apply(args, first_kernel_extra_args);

    CUresult result = coot_wrapper(cuLaunchKernel)(
        first_kernel,
        block_size, 1, 1, mtpb, 1, 1,
        2 * mtpb * sizeof(aux_eT), // shared mem should have size equal to number of threads times 2
        NULL,
        (void**) args,
        0);

    coot_check_cuda_error(result, std::string("coot::cuda::") + std::string(kernel_name) + std::string(": cuLaunchKernel() failed"));

    // Now, take subsequent passes.
    // We use the second kernel for all subsequent passes.
    // The '!' on the return value flips whether or not the final result is in the first/second aux mem buffer.
    return !generic_reduce_inner(aux_mem,
                                 block_size,
                                 second_aux_mem,
                                 kernel_name,
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
                           const uword max_num_threads,
                           dev_mem_t<aux_eT> aux_mem, // must have at least one element
                           const char* kernel_name,
                           CUfunction& kernel,
                           CUfunction& kernel_small, // for 32 threads or fewer
                           const std::tuple<Args...>& kernel_extra_args)
  {
  // TODO: can this be made more efficient?
  const uword pow2_num_threads = (uword) std::pow(2.0f, std::ceil(std::log2((float) max_num_threads)));

  const void* args[3 + sizeof...(Args)];
  args[0] = &mem.cuda_mem_ptr;
  args[1] = &n_elem;
  args[2] = &aux_mem.cuda_mem_ptr;
  unpack_args<sizeof...(Args), Args...>::apply(args, kernel_extra_args);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      pow2_num_threads <= 32 ? kernel_small : kernel, // if we have fewer threads than a single warp, we can use a more optimized version of the kernel
      1, 1, 1, pow2_num_threads, 1, 1,
      2 * pow2_num_threads * sizeof(aux_eT), // shared mem should have size equal to number of threads times 2
      NULL,
      (void**) args,
      0);

  coot_check_cuda_error(result, std::string("coot::cuda::") + std::string(kernel_name) + std::string(": cuLaunchKernel() failed"));
  }
