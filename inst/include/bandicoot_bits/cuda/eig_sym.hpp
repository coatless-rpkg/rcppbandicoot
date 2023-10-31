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



// Utility cleanup function to clean up any allocated memory.
inline
void
eig_sym_cleanup(void* gpu_workspace, char* host_workspace, int* dev_info)
  {
  coot_wrapper(cudaFree)(gpu_workspace);
  cpu_memory::release(host_workspace);
  coot_wrapper(cudaFree)(dev_info);
  }



/**
 * Compute the eigendecomposition using CUDA.
 */
template<typename eT>
inline
std::tuple<bool, std::string>
eig_sym(dev_mem_t<eT> mem, const uword n_rows, const bool eigenvectors, dev_mem_t<eT> eigenvalues)
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

  cusolverEigMode_t jobz = (eigenvectors) ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;

  size_t host_workspace_size = 0;
  size_t gpu_workspace_size = 0;
  status = coot_wrapper(cusolverDnXsyevd_bufferSize)(get_rt().cuda_rt.cusolver_handle,
                                                     NULL, // no special parameters
                                                     jobz,
                                                     CUBLAS_FILL_MODE_UPPER,
                                                     (int64_t) n_rows,
                                                     data_type,
                                                     (void*) mem.cuda_mem_ptr,
                                                     (int64_t) n_rows,
                                                     data_type,
                                                     (void*) eigenvalues.cuda_mem_ptr,
                                                     data_type,
                                                     &gpu_workspace_size,
                                                     &host_workspace_size);
  if (status != CUSOLVER_STATUS_SUCCESS)
    {
    return std::make_tuple(false, "cusolverDnXsyevd_bufferSize() failed with error " + error_as_string(status));
    }

  void* gpu_workspace = NULL;
  status2 = coot_wrapper(cudaMalloc)((void**) &gpu_workspace, gpu_workspace_size);
  if (status2 != cudaSuccess)
    {
    return std::make_tuple(false, "couldn't cudaMalloc() device workspace: "  + error_as_string(status2));
    }

  char* host_workspace = cpu_memory::acquire<char>(host_workspace_size);

  // This is an additional error code for cusolverDn; but it is an error code on the device.
  int* dev_info = NULL;
  status2 = coot_wrapper(cudaMalloc)((void**) &dev_info, sizeof(int));
  if (status2 != cudaSuccess)
    {
    eig_sym_cleanup(gpu_workspace, host_workspace, NULL);
    return std::make_tuple(false, "couldn't cudaMalloc() device info holder: " + error_as_string(status2));
    }

  status = coot_wrapper(cusolverDnXsyevd)(get_rt().cuda_rt.cusolver_handle,
                                          NULL, // no special parameters
                                          jobz,
                                          CUBLAS_FILL_MODE_UPPER,
                                          (int64_t) n_rows,
                                          data_type,
                                          (void*) mem.cuda_mem_ptr,
                                          (int64_t) n_rows,
                                          data_type,
                                          (void*) eigenvalues.cuda_mem_ptr,
                                          data_type,
                                          gpu_workspace,
                                          gpu_workspace_size,
                                          (void*) host_workspace,
                                          host_workspace_size,
                                          dev_info);
  if (status != CUSOLVER_STATUS_SUCCESS)
    {
    eig_sym_cleanup(gpu_workspace, host_workspace, NULL);
    return std::make_tuple(false, "couldn't run cusolverDnXsyevd(): " + error_as_string(status));
    }

  // It seems that CUSOLVER_STATUS_SUCCESS gets returned even when
  // eigendecomposition fails!  So we have to process dev_info more carefully.
  int info;
  status2 = coot_wrapper(cudaMemcpy)(&info, dev_info, sizeof(int), cudaMemcpyDeviceToHost);
  eig_sym_cleanup(gpu_workspace, host_workspace, dev_info);

  if (status2 != cudaSuccess)
    {
    return std::make_tuple(false, "couldn't copy status code from GPU after eigendecomposition: " + error_as_string(status2));
    }

  if (info < 0)
    {
    std::ostringstream oss;
    oss << "cusolverDnXsyevd() failed: parameter " << (-info) << " was incorrect on entry";
    return std::make_tuple(false, oss.str());
    }
  else if (info > 0)
    {
    std::ostringstream oss;
    oss << "eigendecomposition failed: " << info << " off-diagonal elements of an intermediate tridiagonal form did not converge to 0";
    return std::make_tuple(false, oss.str());
    }

  return std::make_tuple(true, "");
  }
