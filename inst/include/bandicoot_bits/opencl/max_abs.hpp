// Copyright 2021 Ryan Curtin (http://www.ratml.org)
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
 * Compute the maximum of absolute values of all elements in `mem`.
 * This is basically the same as accu(), which is also a reduction.
 */
template<typename eT>
inline
eT
max_abs(dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot::opencl::max_abs(): OpenCL runtime not valid" );

  cl_kernel k = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::max_abs);
  cl_kernel k_small = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::max_abs_small);

  return generic_reduce<eT, eT>(mem, n_elem, "max_abs", k, k_small, std::make_tuple(/* no extra args */));
  }
