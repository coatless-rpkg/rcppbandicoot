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
eT
vec_norm_1(dev_mem_t<eT> mem, const uword n_elem, const typename coot_real_only<eT>::result* junk = 0)
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT>(oneway_real_kernel_id::vec_norm_1);
  cl_kernel kernel_small = get_rt().cl_rt.get_kernel<eT>(oneway_real_kernel_id::vec_norm_1_small);

  cl_kernel accu_kernel = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::accu);
  cl_kernel accu_kernel_small = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::accu_small);

  const eT result = generic_reduce<eT, eT>(mem,
                                           n_elem,
                                           "vec_norm_1",
                                           kernel,
                                           kernel_small,
                                           std::make_tuple(/* no extra args */),
                                           accu_kernel,
                                           accu_kernel_small,
                                           std::make_tuple(/* no extra args */));
  return result;
  }



template<typename eT>
inline
eT
vec_norm_2(dev_mem_t<eT> mem, const uword n_elem, const typename coot_real_only<eT>::result* junk = 0)
  {
  // We don't use CLBLAS despite the fact that it has a NRM2 implementation,
  // because that implementation requires a scratch buffer of size 2 * n_elem
  // and I don't like that idea at all!
  //
  // So, we just use the powk_norm kernel and check for underflow or overflow,
  // to match the behavior of Armadillo and the CUDA backend.
  // Note that Armadillo does not check for underflow or overflow for the
  // general k-norm case (where k != 2), so we don't do that either.

  coot_extra_debug_sigprint();
  coot_ignore(junk);

  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT>(oneway_real_kernel_id::vec_norm_2);
  cl_kernel kernel_small = get_rt().cl_rt.get_kernel<eT>(oneway_real_kernel_id::vec_norm_2_small);

  cl_kernel accu_kernel = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::accu);
  cl_kernel accu_kernel_small = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::accu_small);

  const eT result = generic_reduce<eT, eT>(mem,
                                           n_elem,
                                           "vec_norm_2",
                                           kernel,
                                           kernel_small,
                                           std::make_tuple(/* no extra args */),
                                           accu_kernel,
                                           accu_kernel_small,
                                           std::make_tuple(/* no extra args */));

  if (result == eT(0) || !coot_isfinite(result))
    {
    // We detected overflow or underflow---try again.
    const eT max_elem = max_abs(mem, n_elem);
    if (max_elem == eT(0))
      {
      // False alarm, the norm is actually zero.
      return eT(0);
      }

    cl_kernel robust_kernel = get_rt().cl_rt.get_kernel<eT>(oneway_real_kernel_id::vec_norm_2_robust);
    cl_kernel robust_kernel_small = get_rt().cl_rt.get_kernel<eT>(oneway_real_kernel_id::vec_norm_2_robust_small);

    const eT robust_result = generic_reduce<eT, eT>(mem,
                                                    n_elem,
                                                    "vec_norm_2_robust",
                                                    robust_kernel,
                                                    robust_kernel_small,
                                                    std::make_tuple(max_elem),
                                                    accu_kernel,
                                                    accu_kernel_small,
                                                    std::make_tuple(/* no extra args */));

    return std::sqrt(robust_result) * max_elem;
    }
  else
    {
    return std::sqrt(result);
    }
  }




template<typename eT>
inline
eT
vec_norm_k(dev_mem_t<eT> mem, const uword n_elem, const uword k, const typename coot_real_only<eT>::result* junk = 0)
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT>(oneway_real_kernel_id::vec_norm_k);
  cl_kernel kernel_small = get_rt().cl_rt.get_kernel<eT>(oneway_real_kernel_id::vec_norm_k_small);

  cl_kernel accu_kernel = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::accu);
  cl_kernel accu_kernel_small = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::accu_small);

  const eT result = generic_reduce<eT, eT>(mem,
                                           n_elem,
                                           "vec_norm_k",
                                           kernel,
                                           kernel_small,
                                           std::make_tuple(k),
                                           accu_kernel,
                                           accu_kernel_small,
                                           std::make_tuple(/* no extra args */));
  return std::pow(result, eT(1) / eT(k));
  }




template<typename eT>
inline
eT
vec_norm_min(dev_mem_t<eT> mem, const uword n_elem, const typename coot_real_only<eT>::result* junk = 0)
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT>(oneway_real_kernel_id::vec_norm_min);
  cl_kernel kernel_small = get_rt().cl_rt.get_kernel<eT>(oneway_real_kernel_id::vec_norm_min_small);

  cl_kernel min_kernel = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::min);
  cl_kernel min_kernel_small = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::min_small);

  const eT result = generic_reduce<eT, eT>(mem,
                                           n_elem,
                                           "vec_norm_min",
                                           kernel,
                                           kernel_small,
                                           std::make_tuple(/* no extra args */),
                                           min_kernel,
                                           min_kernel_small,
                                           std::make_tuple(/* no extra args */));
  return result;
  }
