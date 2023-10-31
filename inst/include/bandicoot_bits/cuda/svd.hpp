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



// Utility to free memory before returning.
inline
void
svd_cleanup(char* device_buffer, char* host_buffer, int* dev_info)
  {
  coot_wrapper(cudaFree)(device_buffer);
  cpu_memory::release(host_buffer);
  coot_wrapper(cudaFree)(dev_info);
  }



/**
 * Compute the singular value decomposition using CUDA (cuSolverDn).
 *
 * This expects that n_rows < n_cols; if that's not true, transpose A (and handle the slightly different results).
 *
 * Note that this function will not throw but instead will return a bool indicating success or failure.
 */
template<typename eT>
inline
std::tuple<bool, std::string>
svd(dev_mem_t<eT> U,
    dev_mem_t<eT> S,
    dev_mem_t<eT> V,
    dev_mem_t<eT> A,
    const uword n_rows,
    const uword n_cols,
    const bool compute_u_vt)
  {
  coot_extra_debug_sigprint();

  if (get_rt().cuda_rt.is_valid() == false)
    {
    return std::make_tuple(false, "CUDA runtime not valid");
    }

  if (n_rows < n_cols)
    {
    return std::make_tuple(false, "n_rows must be greater than or equal to n_cols");
    }

  cusolverStatus_t status;
  cudaError_t status2;

  const char jobuvt = (compute_u_vt) ? 'A' : 'N';

  // Compute the workspace sizes necessary on the host and the device.
  size_t device_buffer_size;
  size_t host_buffer_size;

  status = coot_wrapper(cusolverDnXgesvd_bufferSize)(get_rt().cuda_rt.cusolver_handle,
                                                     NULL, // it appears to be possible to pass NULL for the parameters
                                                     jobuvt,
                                                     jobuvt,
                                                     n_rows,
                                                     n_cols,
                                                     cuda_data_type<eT>::type,
                                                     (void*) A.cuda_mem_ptr,
                                                     n_rows,
                                                     cuda_data_type<eT>::type,
                                                     (void*) S.cuda_mem_ptr,
                                                     cuda_data_type<eT>::type,
                                                     (void*) U.cuda_mem_ptr,
                                                     // If compute_u_vt is false, we assume the user passed a 1x1 matrix for each,
                                                     // and we also assume those matrices won't be referenced.
                                                     (compute_u_vt) ? n_rows : 1,
                                                     cuda_data_type<eT>::type,
                                                     (void*) V.cuda_mem_ptr,
                                                     (compute_u_vt) ? n_cols : 1,
                                                     cuda_data_type<eT>::type,
                                                     &device_buffer_size,
                                                     &host_buffer_size);
  if (status != CUSOLVER_STATUS_SUCCESS)
    {
    return std::make_tuple(false, "couldn't calculate workspace sizes with cusolverDnXgesvd_bufferSize(): " + error_as_string(status));
    }

  char* device_buffer = NULL;
  status2 = coot_wrapper(cudaMalloc)((void**) &device_buffer, device_buffer_size);
  if (status2 != cudaSuccess)
    {
    return std::make_tuple(false, "couldn't cudaMalloc() device workspace: " + error_as_string(status2));
    }
  char* host_buffer = cpu_memory::acquire<char>(host_buffer_size);

  // This is an additional error code for cusolverDn; but it is an error code on the device...
  int* dev_info = NULL;
  status2 = coot_wrapper(cudaMalloc)((void**) &dev_info, sizeof(int));
  if (status2 != cudaSuccess)
    {
    svd_cleanup(device_buffer, host_buffer, NULL);
    return std::make_tuple(false, "couldn't cudaMalloc() status value: " + error_as_string(status2));
    }

  status = coot_wrapper(cusolverDnXgesvd)(get_rt().cuda_rt.cusolver_handle,
                                          NULL, // it appears to be possible to pass NULL for the parameters
                                          jobuvt,
                                          jobuvt,
                                          n_rows,
                                          n_cols,
                                          cuda_data_type<eT>::type,
                                          (void*) A.cuda_mem_ptr,
                                          n_rows,
                                          cuda_data_type<eT>::type,
                                          (void*) S.cuda_mem_ptr,
                                          cuda_data_type<eT>::type,
                                          (void*) U.cuda_mem_ptr,
                                          (compute_u_vt) ? n_rows : 1,
                                          cuda_data_type<eT>::type,
                                          (void*) V.cuda_mem_ptr,
                                          (compute_u_vt) ? n_cols : 1,
                                          cuda_data_type<eT>::type,
                                          (void*) device_buffer,
                                          device_buffer_size,
                                          (void*) host_buffer,
                                          host_buffer_size,
                                          dev_info);
  if (status != CUSOLVER_STATUS_SUCCESS)
    {
    svd_cleanup(device_buffer, host_buffer, dev_info);
    return std::make_tuple(false, "cusolverDnXgesvd() failed: " + error_as_string(status));
    }

  // It seems that CUSOLVER_STATUS_SUCCESS gets returned even when the SVD
  // failed!  So we have to process dev_info more carefully.
  int info;
  status2 = coot_wrapper(cudaMemcpy)(&info, dev_info, sizeof(int), cudaMemcpyDeviceToHost);

  // We don't need any of the allocated memory now.
  svd_cleanup(device_buffer, host_buffer, dev_info);

  if (status2 != cudaSuccess)
    {
    return std::make_tuple(false, "couldn't copy status code from GPU after factorisation: " + error_as_string(status2));
    }

  if (info < 0)
    {
    std::ostringstream oss;
    oss << "cusolverDnXgesvd() failed: parameter " << (-info) << " was incorrect on entry";
    return std::make_tuple(false, oss.str());
    }
  else if (info > 0)
    {
    std::ostringstream oss;
    oss << "cusolverDnXgesvd() failed: " << status2 << " superdiagonals of an intermediate bidiagonal form did not converge to 0";
    return std::make_tuple(false, oss.str());
    }

  return std::make_tuple(true, "");
  }
