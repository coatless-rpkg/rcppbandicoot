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
 * Solve the system A * X = B or A^T * X = B for square A using the LU factorisation using CUDA.
 *
 * A is of size n_rows x n_rows, and is destroyed.
 * B is of size n_rows x n_cols, and is replaced with X.
 */
template<typename eT>
inline
std::tuple<bool, std::string>
solve_square_fast(dev_mem_t<eT> A, const bool trans_A, dev_mem_t<eT> B, const uword n_rows, const uword n_cols)
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
                                                     A.cuda_mem_ptr,
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
  const uword ipiv_size = n_rows;

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
                                          n_rows,
                                          data_type,
                                          A.cuda_mem_ptr,
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

  // Now use the LU-factorised A as the input to getrs() to solve the system.

  status = coot_wrapper(cusolverDnXgetrs)(get_rt().cuda_rt.cusolver_handle,
                                          NULL,
                                          (trans_A ? CUBLAS_OP_T : CUBLAS_OP_N),
                                          n_rows,
                                          n_cols,
                                          data_type,
                                          A.cuda_mem_ptr,
                                          n_rows,
                                          ipiv,
                                          data_type,
                                          B.cuda_mem_ptr,
                                          n_rows,
                                          dev_info);

  // no longer needed
  coot_wrapper(cudaFree)(ipiv);

  if (status != CUSOLVER_STATUS_SUCCESS)
    {
    coot_wrapper(cudaFree)(dev_info);
    return std::make_tuple(false, "solving via cusolverDnXgetrs() failed");
    }

  // Check whether the operation was successful.
  status2 = coot_wrapper(cudaMemcpy)(&info_result, dev_info, sizeof(int), cudaMemcpyDeviceToHost);
  coot_wrapper(cudaFree)(dev_info);
  if (status2 != cudaSuccess)
    {
    return std::make_tuple(false, "couldn't copy device info holder to host");
    }

  if (info_result < 0)
    {
    std::ostringstream oss;
    oss << "parameter " << -info_result << " was incorrect in call to cusolverDnXgetrs()";
    return std::make_tuple(false, oss.str());
    }

  get_rt().cuda_rt.synchronise();

  return std::make_tuple(true, "");
  }
