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



// Utility to run a full reduce in an efficient and generic way, where the reduce
// must also maintain a vector of auxiliary uwords.
//
// Note that all kernels that are used with generic uword-aux reduce are expected to compute a UWORD,
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
                         const std::tuple<A2...>& second_kernel_extra_args);



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
                         const std::tuple<Args...>& kernel_extra_args);



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
                               dev_mem_t<uword> second_aux_uword_mem);



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
                                     const std::tuple<Args...>& kernel_extra_args);
