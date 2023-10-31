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
//  - eT1* out_mem
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
               const std::tuple<A2...>& second_kernel_extra_args);



// This version uses the same kernel for all reduce passes.
template<typename eT, typename aux_eT, typename... Args>
inline
aux_eT
generic_reduce(const dev_mem_t<eT> mem,
               const uword n_elem,
               const char* kernel_name,
               CUfunction& kernel,
               CUfunction& kernel_small,
               const std::tuple<Args...>& kernel_extra_args);



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
                     dev_mem_t<aux_eT> second_aux_mem);



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
                           const std::tuple<Args...>& kernel_extra_args);
