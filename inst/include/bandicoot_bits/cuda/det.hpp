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
 * Compute the LU factorisation using OpenCL.
 */
template<typename eT>
inline
std::tuple<bool, std::string>
det(dev_mem_t<eT> in, const uword n_rows, eT& out_val)
  {
  coot_extra_debug_sigprint();

  if (get_rt().cuda_rt.is_valid() == false)
    {
    return std::make_tuple(false, "CUDA runtime not valid");
    }

  cusolverStatus_t status;
  cudaError_t status2;

  cudaDataType data_type;
  if (is_float<eT>::value)
    {
    data_type = CUDA_R_32F;
    }
  else if (is_double<eT>::value)
    {
    data_type = CUDA_R_64F;
    }
  else
    {
    return std::make_tuple(false, "unknown data type, must be float or double");
    }

  // This is an additional error code for cusolverDn; but it is an error code on the device...
  int* dev_info = NULL;
  status2 = coot_wrapper(cudaMalloc)((void**) &dev_info, sizeof(int));
  if (status2 != cudaSuccess)
    {
    return std::make_tuple(false, "couldn't cudaMalloc() device info holder");
    }

  size_t host_workspace_size = 0;
  size_t gpu_workspace_size = 0;
  status = coot_wrapper(cusolverDnXgetrf_bufferSize)(get_rt().cuda_rt.cusolver_handle,
                                                     NULL,
                                                     n_rows,
                                                     n_rows,
                                                     data_type,
                                                     in.cuda_mem_ptr,
                                                     n_rows,
                                                     data_type,
                                                     &gpu_workspace_size,
                                                     &host_workspace_size);
  if (status != CUSOLVER_STATUS_SUCCESS)
    {
    coot_wrapper(cudaFree)(dev_info);
    return std::make_tuple(false, "couldn't compute workspace size with cusolverDnXgetrf_bufferSize()");
    }

  // Allocate space for pivots.
  s64* ipiv = NULL;
  status2 = coot_wrapper(cudaMalloc)((void**) &ipiv, sizeof(s64) * n_rows);
  if (status2 != cudaSuccess)
    {
    coot_wrapper(cudaFree)(dev_info);
    return std::make_tuple(false, "couldn't cudaMalloc() pivot array");
    }

  void* gpu_workspace = NULL;
  status2 = coot_wrapper(cudaMalloc)((void**) &gpu_workspace, gpu_workspace_size);
  if (status2 != cudaSuccess)
    {
    coot_wrapper(cudaFree)(dev_info);
    coot_wrapper(cudaFree)(ipiv);
    return std::make_tuple(false, "couldn't cudaMalloc() GPU workspace memory");
    }

  char* host_workspace = cpu_memory::acquire<char>(host_workspace_size);

  status = coot_wrapper(cusolverDnXgetrf)(get_rt().cuda_rt.cusolver_handle,
                                          NULL,
                                          n_rows,
                                          n_rows,
                                          data_type,
                                          in.cuda_mem_ptr,
                                          n_rows,
                                          ipiv,
                                          data_type,
                                          gpu_workspace,
                                          gpu_workspace_size,
                                          (void*) host_workspace,
                                          host_workspace_size,
                                          dev_info);

  coot_wrapper(cudaFree)(gpu_workspace);
  cpu_memory::release(host_workspace);

  if (status != CUSOLVER_STATUS_SUCCESS)
    {
    coot_wrapper(cudaFree)(dev_info);
    coot_wrapper(cudaFree)(ipiv);
    return std::make_tuple(false, "factorisation via cusolverDnXgetrf() failed");
    }

  // Check whether the factorisation was successful.
  int info_result;
  status2 = coot_wrapper(cudaMemcpy)(&info_result, dev_info, sizeof(int), cudaMemcpyDeviceToHost);
  coot_wrapper(cudaFree)(dev_info);
  if (status2 != cudaSuccess)
    {
    coot_wrapper(cudaFree)(ipiv);
    return std::make_tuple(false, "couldn't copy device info holder to host");
    }

  if (info_result < 0)
    {
    std::ostringstream oss;
    oss << "parameter " << -info_result << " was incorrect in call to cusolverDnXgetrf()";
    return std::make_tuple(false, oss.str());
    }
  else if (info_result > 0 && info_result < (int) n_rows)
    {
    // Technically any positive info_result indicates a failed decomposition, but it looks like it randomly sometimes returns very large (invalid) numbers.
    // So... we ignore those.
    std::ostringstream oss;
    oss << "decomposition failed, U(" << (info_result - 1) << ", " << (info_result - 1) << ") was found to be 0";
    return std::make_tuple(false, oss.str());
    }

  // Now the determinant is det(L) * det(U) * det(P);
  // since L and U are triangular, then the determinant is the product of the diagonal values.
  // Also, since L's diagonal is 1, then det(L) = 1.
  // The determination of a permutation matrix is either 1 or -1,
  // depending on whether the number of permutations is even or odd (respectively).
  // Thus, the primary operation is just to compute det(U).

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_real_kernel_id::diag_prod);
  CUfunction kernel_small = get_rt().cuda_rt.get_kernel<eT>(oneway_real_kernel_id::diag_prod_small);

  CUfunction second_kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::prod);
  CUfunction second_kernel_small = get_rt().cuda_rt.get_kernel<eT>(oneway_kernel_id::prod_small);

  const eT U_det = generic_reduce<eT, eT>(in, n_rows, "det", kernel, kernel_small, std::make_tuple(/* no extra args */), second_kernel, second_kernel_small, std::make_tuple(/* no extra args */));

  CUfunction p_kernel = get_rt().cuda_rt.get_kernel<s64>(oneway_integral_kernel_id::ipiv_det);
  CUfunction p_kernel_small = get_rt().cuda_rt.get_kernel<s64>(oneway_integral_kernel_id::ipiv_det_small);

  CUfunction p_second_kernel = get_rt().cuda_rt.get_kernel<s64>(oneway_kernel_id::prod);
  CUfunction p_second_kernel_small = get_rt().cuda_rt.get_kernel<s64>(oneway_kernel_id::prod_small);

  dev_mem_t<s64> ipiv_gpu;
  ipiv_gpu.cuda_mem_ptr = ipiv;
  const s64 P_det = generic_reduce<s64, s64>(ipiv_gpu, n_rows, "det", p_kernel, p_kernel_small, std::make_tuple(/* no extra args */), p_second_kernel, p_second_kernel_small, std::make_tuple(/* no extra args */));

  get_rt().cuda_rt.synchronise();
  get_rt().cuda_rt.release_memory(ipiv_gpu.cuda_mem_ptr);

  out_val = U_det * (eT) P_det;

  return std::make_tuple(true, "");
  }
