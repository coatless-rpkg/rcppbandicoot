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

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_real_kernel_id::vec_norm_1);
  CUfunction kernel_small = get_rt().cuda_rt.get_kernel<eT>(oneway_real_kernel_id::vec_norm_1_small);

  CUfunction accu_kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::accu);
  CUfunction accu_kernel_small = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::accu_small);

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



inline
float
vec_norm_2(dev_mem_t<float> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "cuda runtime not valid" );

  float result;
  cublasStatus_t status = coot_wrapper(cublasSnrm2)(get_rt().cuda_rt.cublas_handle, n_elem, mem.cuda_mem_ptr, 1, &result);

  coot_check_cublas_error( status, "coot::cuda::vec_norm_2(): call to cublasSnrm2() failed" );

  return result;
  }



inline
double
vec_norm_2(dev_mem_t<double> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "cuda runtime not valid" );

  double result;
  cublasStatus_t status = coot_wrapper(cublasDnrm2)(get_rt().cuda_rt.cublas_handle, n_elem, mem.cuda_mem_ptr, 1, &result);

  coot_check_cublas_error( status, "coot::cuda::vec_norm_2(): call to cublasDnrm2() failed" );

  return result;
  }



template<typename eT>
inline
eT
vec_norm_2(dev_mem_t<eT> mem, const uword n_elem, const typename coot_real_only<eT>::result* junk = 0)
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "cuda runtime not valid" );

  // For floating-point types, we perform a power-k accumulation.
  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_real_kernel_id::vec_norm_2);
  CUfunction kernel_small = get_rt().cuda_rt.get_kernel<eT>(oneway_real_kernel_id::vec_norm_2_small);

  CUfunction accu_kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::accu);
  CUfunction accu_kernel_small = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::accu_small);

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

    CUfunction robust_kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_real_kernel_id::vec_norm_2_robust);
    CUfunction robust_kernel_small = get_rt().cuda_rt.get_kernel<eT>(oneway_real_kernel_id::vec_norm_2_robust_small);

    const eT robust_result = generic_reduce<eT, eT>(mem,
                                                    n_elem,
                                                    "vec_norm_2_robust",
                                                    robust_kernel,
                                                    robust_kernel_small,
                                                    std::make_tuple(to_cuda_type(max_elem)),
                                                    accu_kernel,
                                                    accu_kernel_small,
                                                    std::make_tuple(/* no extra args */));

    return coot_sqrt(robust_result) * max_elem;
    }
  else
    {
    // The kernel returns just the accumulated result, so we still need to take the k'th root.
    return coot_sqrt(result);
    }
  }



template<typename eT>
inline
eT
vec_norm_k(dev_mem_t<eT> mem, const uword n_elem, const uword k, const typename coot_real_only<eT>::result* junk = 0)
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "cuda runtime not valid" );

  // For floating-point types, we perform a power-k accumulation.
  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_real_kernel_id::vec_norm_k);
  CUfunction kernel_small = get_rt().cuda_rt.get_kernel<eT>(oneway_real_kernel_id::vec_norm_k_small);

  CUfunction accu_kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::accu);
  CUfunction accu_kernel_small = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::accu_small);

  const eT result = generic_reduce<eT, eT>(mem,
                                           n_elem,
                                           "vec_norm_k",
                                           kernel,
                                           kernel_small,
                                           std::make_tuple(k),
                                           accu_kernel,
                                           accu_kernel_small,
                                           std::make_tuple(/* no extra args */));

  // The kernel returns just the accumulated result, so we still need to take the k'th root.
  return coot_pow(result, eT(1 / eT(k)));
  }



template<typename eT>
inline
eT
vec_norm_min(dev_mem_t<eT> mem, const uword n_elem, const typename coot_real_only<eT>::result* junk = 0)
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  // For floating-point types, we perform a power-k accumulation.
  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_real_kernel_id::vec_norm_min);
  CUfunction kernel_small = get_rt().cuda_rt.get_kernel<eT>(oneway_real_kernel_id::vec_norm_min_small);

  CUfunction min_kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::min);
  CUfunction min_kernel_small = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::min_small);

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



// vec_norm_max() is not needed---max_abs() is equivalent.
