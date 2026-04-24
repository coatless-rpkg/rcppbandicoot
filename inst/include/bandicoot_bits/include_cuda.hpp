// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2023 Ryan Curtin (http://www.ratml.org)
// Copyright 2026 Conrad Sanderson (https://conradsanderson.id.au)
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



#if defined(COOT_USE_CUDA)
  #if defined(__has_include)
    #if __has_include(<cuda.h>)
      #include <cuda.h>
    #else
      #undef COOT_USE_CUDA
      #pragma message ("WARNING: use of CUDA disabled; cuda.h header not found")
    #endif
  #else
    #include <cuda.h>
  #endif
#endif


#if defined(COOT_USE_CUDA)

  #include <cuda_fp16.h>
  #include <cuda_runtime_api.h>
  #include <cuda_runtime.h>
  #include <cublas_v2.h>
  #include <nvrtc.h>
  #include <curand.h>
  #include <cusolverDn.h>

  #if CUDART_VERSION < 12020
    // Some older versions of CUDA do not have the definitions we need available.
    namespace coot
      {
      inline __half coot_cuda_ushort_as_half(const unsigned short int i)
        {
        __half h;
        *(reinterpret_cast<unsigned short*>(&h)) = i;
        return h;
        }
      }

    #define CUDART_INF_FP16            coot::coot_cuda_ushort_as_half((unsigned short)0x7C00U)
    #define CUDART_NAN_FP16            coot::coot_cuda_ushort_as_half((unsigned short)0x7FFFU)
    #define CUDART_MIN_DENORM_FP16     coot::coot_cuda_ushort_as_half((unsigned short)0x0001U)
    #define CUDART_MAX_NORMAL_FP16     coot::coot_cuda_ushort_as_half((unsigned short)0x7BFFU)
    #define CUDART_NEG_ZERO_FP16       coot::coot_cuda_ushort_as_half((unsigned short)0x8000U)
    #define CUDART_ZERO_FP16           coot::coot_cuda_ushort_as_half((unsigned short)0x0000U)
    #define CUDART_ONE_FP16            coot::coot_cuda_ushort_as_half((unsigned short)0x3C00U)

    namespace coot
      {
      inline bool coot_cuda_half_isinf(const __half x)
        {
        return ((*(reinterpret_cast<const unsigned short*>(&x))) == 0x7C00U) ||
               ((*(reinterpret_cast<const unsigned short*>(&x))) == 0xFC00U);
        }

      inline bool coot_cuda_half_isnan(const __half x)
        {
        return (*(reinterpret_cast<const unsigned short*>(&x))) == 0x7FFFU;
        }
      }
  #endif

#endif
