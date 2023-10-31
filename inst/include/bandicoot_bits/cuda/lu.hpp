// Copyright 2019 Ryan Curtin (http://www.ratml.org)
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
 * Compute the LU factorisation using CUDA.
 */
template<typename eT>
inline
std::tuple<bool, std::string>
lu(dev_mem_t<eT> L, dev_mem_t<eT> U, dev_mem_t<eT> in, const bool pivoting, dev_mem_t<eT> P, const uword n_rows, const uword n_cols)
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
                                                     n_cols,
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

  s64* ipiv = NULL;
  const uword ipiv_size = std::min(n_rows, n_cols);

  // Allocate space for pivots.
  status2 = coot_wrapper(cudaMalloc)((void**) &ipiv, sizeof(s64) * ipiv_size);
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
                                          n_cols,
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
  else if (info_result > 0 && info_result < (int) std::max(n_rows, n_cols))
    {
    // Technically any positive info_result indicates a failed decomposition, but it looks like it randomly sometimes returns very large (invalid) numbers.
    // So... we ignore those.
    std::ostringstream oss;
    oss << "decomposition failed, U(" << (info_result - 1) << ", " << (info_result - 1) << ") was found to be 0";
    return std::make_tuple(false, oss.str());
    }

  // First the pivoting needs to be "unwound" into a way where we can make P.
  uword* ipiv2 = cpu_memory::acquire<uword>(n_rows);
  for (uword i = 0; i < n_rows; ++i)
    {
    ipiv2[i] = i;
    }

  s64* ipiv_cpu = cpu_memory::acquire<s64>(ipiv_size);
  status2 = coot_wrapper(cudaMemcpy)(ipiv_cpu, ipiv, ipiv_size * sizeof(s64), cudaMemcpyDeviceToHost);
  coot_wrapper(cudaFree)(ipiv);

  if (status2 != cudaSuccess)
    {
    cpu_memory::release(ipiv2);
    cpu_memory::release(ipiv_cpu);
    return std::make_tuple(false, "couldn't copy pivot array from GPU");
    }

  for (uword i = 0; i < ipiv_size; ++i)
    {
    const int k = ipiv_cpu[i] - 1; // cusolverDnXgetrf() returns one-indexed results

    if (ipiv2[i] != ipiv2[k])
      {
      std::swap( ipiv2[i], ipiv2[k] );
      }
    }

  dev_mem_t<uword> ipiv_gpu;
  ipiv_gpu.cuda_mem_ptr = get_rt().cuda_rt.acquire_memory<uword>(n_rows);
  copy_into_dev_mem(ipiv_gpu, ipiv2, n_rows);
  cpu_memory::release(ipiv_cpu);
  cpu_memory::release(ipiv2);

  // Now extract the lower triangular part (excluding diagonal).  This is done with a custom kernel.
  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT>(pivoting ? oneway_real_kernel_id::lu_extract_l : oneway_real_kernel_id::lu_extract_pivoted_l);

  const void* pivot_args[] = { &(L.cuda_mem_ptr),
                               &(U.cuda_mem_ptr),
                               &(in.cuda_mem_ptr),
                               (uword*) &n_rows,
                               (uword*) &n_cols };

  const void* nopivot_args[] = { &(L.cuda_mem_ptr),
                                 &(U.cuda_mem_ptr),
                                 &(in.cuda_mem_ptr),
                                 (uword*) &n_rows,
                                 (uword*) &n_cols,
                                 &(ipiv_gpu.cuda_mem_ptr) };

  const size_t max_rc = std::max(n_rows, n_cols);
  const kernel_dims dims = two_dimensional_grid_dims(n_rows, max_rc);

  CUresult status3 = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2], dims.d[3], dims.d[4], dims.d[5],
      0, NULL, (pivoting ? (void**) pivot_args : (void**) nopivot_args), 0);

  if (status3 != CUDA_SUCCESS)
    {
    get_rt().cuda_rt.release_memory(ipiv_gpu.cuda_mem_ptr);
    return std::make_tuple(false, "cuLaunchKernel() failed for " + (pivoting ? std::string("lu_extract_l") : std::string("lu_extract_pivoted_l")) + " kernel");
    }

  // If pivoting was allowed, extract the permutation matrix.
  if (pivoting)
    {
    kernel = get_rt().cuda_rt.get_kernel<eT>(oneway_real_kernel_id::lu_extract_p);

    const void* args2[] = {
        &(P.cuda_mem_ptr),
        &(ipiv_gpu.cuda_mem_ptr),
        (uword*) &n_rows };

    const kernel_dims dims2 = one_dimensional_grid_dims(n_rows);

    status3 = coot_wrapper(cuLaunchKernel)(
        kernel,
        dims2.d[0], dims2.d[1], dims2.d[2], dims2.d[3], dims2.d[4], dims2.d[5],
        0, NULL, (void**) args2, 0);

    if (status3 != CUDA_SUCCESS)
      {
      get_rt().cuda_rt.release_memory(ipiv_gpu.cuda_mem_ptr);
      return std::make_tuple(false, "cuLaunchKernel() failed for lu_extract_p kernel");
      }
    }

  get_rt().cuda_rt.synchronise();
  get_rt().cuda_rt.release_memory(ipiv_gpu.cuda_mem_ptr);

  return std::make_tuple(true, "");
  }
