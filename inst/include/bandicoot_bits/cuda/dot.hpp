// Copyright 2020 Ryan Curtin (http://www.ratml.org)
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
 * Compute a dot product between two vectors.
 */
template<typename eT1, typename eT2>
inline
typename promote_type<eT1, eT2>::result
dot(dev_mem_t<eT1> mem1, dev_mem_t<eT2> mem2, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  typedef typename promote_type<eT1, eT2>::result promoted_eT;

  coot_debug_check( (get_rt().cuda_rt.is_valid() == false), "coot::cuda::dot(): cuda runtime not valid" );

  // If we can, try to use cuBLAS.
  if (std::is_same<eT1, eT2>::value && std::is_same<eT1, float>::value)
    {
    float result;
    cublasStatus_t status = coot_wrapper(cublasSdot)(get_rt().cuda_rt.cublas_handle, n_elem, (float*) mem1.cuda_mem_ptr, 1, (float*) mem2.cuda_mem_ptr, 1, &result);

    coot_check_cublas_error( status, "coot::cuda::dot(): call to cublasSdot() failed" );
    return result;
    }
  else if (std::is_same<eT1, eT2>::value && std::is_same<eT1, double>::value)
    {
    double result;
    cublasStatus_t status = coot_wrapper(cublasDdot)(get_rt().cuda_rt.cublas_handle, n_elem, (double*) mem1.cuda_mem_ptr, 1, (double*) mem2.cuda_mem_ptr, 1, &result);

    coot_check_cublas_error( status, "coot::cuda::dot(): call to cublasDdot() failed" );
    return result;
    }
  else
    {
    // In any other situation, we'll use our own kernels.
    CUfunction k = get_rt().cuda_rt.get_kernel<eT2, eT1>(twoway_kernel_id::dot);
    CUfunction k_small = get_rt().cuda_rt.get_kernel<eT2, eT1>(twoway_kernel_id::dot_small);

    // Compute grid size; ideally we want to use the maximum possible number of threads per block.
    kernel_dims dims = one_dimensional_grid_dims(std::ceil(n_elem / (2 * std::ceil(std::log2(std::max((double) n_elem, 2.0))))));

    // Create auxiliary memory, with size equal to the number of blocks.
    Mat<promoted_eT> aux(dims.d[0], 1);
    dev_mem_t<promoted_eT> aux_mem = aux.get_dev_mem(false);

    // We'll only run once with the dot kernel, and if this still needs further reduction, we can use accu().

    // Ensure we always use a power of 2 for the number of threads.
    const uword num_threads = next_pow2(dims.d[3]);

    const void* args[] = {
        &(aux_mem.cuda_mem_ptr),
        &(mem1.cuda_mem_ptr),
        &(mem2.cuda_mem_ptr),
        (uword*) &n_elem };

    CUresult result = coot_wrapper(cuLaunchKernel)(
        num_threads <= 32 ? k_small : k, // if we have fewer threads than a single warp, we can use a more optimized version of the kernel
        dims.d[0], dims.d[1], dims.d[2],
        num_threads, dims.d[4], dims.d[5],
        2 * num_threads * sizeof(promoted_eT), // shared mem should have size equal to number of threads times 2
        NULL,
        (void**) args,
        0);

    coot_check_cuda_error(result, "coot::cuda::dot(): cuLaunchKernel() failed");

    if (aux.n_elem == 1)
      {
      return promoted_eT(aux[0]);
      }
    else
      {
      return accu(aux_mem, aux.n_elem);
      }
    }
  }
