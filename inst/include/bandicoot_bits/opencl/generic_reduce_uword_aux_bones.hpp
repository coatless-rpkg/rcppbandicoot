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



// Utility to run a full reduce in an efficient way.
//
// Note that all kernels that are used with generic_reduce_uword_aux() are expected to return a uword in the first element of `out_uword_mem`,,
// and the first eight arguments should be:
//  - const __global eT1* mem
//  - const __global UWORD* mem_uword
//  - const bool use_uword_mem
//  - const UWORD n_elem
//  - __global eT1* out_mem
//  - __global UWORD* out_uword_mem
//  - __local volatile eT1* aux_mem
//  - __local volatile UWORD* aux_uword_mem
// Additional arguments are fine, so long as those first eight are the same.



template<typename eT, typename aux_eT, typename... A1, typename... A2>
inline
uword
generic_reduce_uword_aux(const dev_mem_t<eT> mem,
                         const uword n_elem,
                         const char* kernel_name,
                         eT* reduce_result,
                         cl_kernel& first_kernel,
                         cl_kernel& first_kernel_small,
                         const std::tuple<A1...>& first_kernel_extra_args,
                         cl_kernel& second_kernel,
                         cl_kernel& second_kernel_small,
                         const std::tuple<A2...>& second_kernel_extra_args);



template<typename eT, typename aux_eT, typename... Args>
inline
uword
generic_reduce_uword_aux(const dev_mem_t<eT> mem,
                         const uword n_elem,
                         const char* kernel_name,
                         eT* reduce_result,
                         cl_kernel& kernel,
                         cl_kernel& kernel_small,
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
                               const size_t total_num_threads,
                               const size_t local_group_size,
                               cl_kernel& first_kernel,
                               cl_kernel& first_kernel_small,
                               const std::tuple<A1...>& first_kernel_extra_args,
                               cl_kernel& second_kernel,
                               cl_kernel& second_kernel_small,
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
                                     dev_mem_t<aux_eT> aux_mem,
                                     dev_mem_t<uword> aux_uword_mem,
                                     const char* kernel_name,
                                     const size_t total_num_threads,
                                     const size_t local_group_size,
                                     cl_kernel& kernel,
                                     cl_kernel& kernel_small,
                                     const std::tuple<Args...>& first_kernel_extra_args);
