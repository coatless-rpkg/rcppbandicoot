// Copyright 2019 Ryan Curtin (https://www.ratml.org)
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



template<typename T1>
coot_hot
inline
void
coot_check_cuda_error(const cudaError_t error_code, const T1& x)
  {
  if (error_code != cudaSuccess)
    {
    coot_stop_runtime_error( x, cuda::error_as_string(error_code) );
    }
  }



template<typename T1>
coot_hot
inline
void
coot_check_cuda_error(const CUresult error_code, const T1& x)
  {
  if (error_code != CUDA_SUCCESS)
    {
    coot_stop_runtime_error( x, cuda::error_as_string(error_code) );
    }
  }



template<typename T1>
coot_hot
inline
void
coot_check_nvrtc_error(const nvrtcResult error_code, const T1& x)
  {
  if (error_code != NVRTC_SUCCESS)
    {
    coot_stop_runtime_error( x, cuda::error_as_string(error_code) );
    }
  }



template<typename T1>
coot_hot
inline
void
coot_check_cusolver_error(const cusolverStatus_t error_code, const T1& x)
  {
  if (error_code != CUSOLVER_STATUS_SUCCESS)
    {
    coot_stop_runtime_error( x, cuda::error_as_string(error_code) );
    }
  }



template<typename T1>
coot_hot
inline
void
coot_check_cublas_error(const cublasStatus_t error_code, const T1& x)
  {
  if (error_code != CUBLAS_STATUS_SUCCESS)
    {
    coot_stop_runtime_error( x, cuda::error_as_string(error_code) );
    }
  }



template<typename T1>
coot_hot
inline
void
coot_check_curand_error(const curandStatus_t error_code, const T1& x)
  {
  if (error_code != CURAND_STATUS_SUCCESS)
    {
    coot_stop_runtime_error( x, cuda::error_as_string(error_code) );
    }
  }
