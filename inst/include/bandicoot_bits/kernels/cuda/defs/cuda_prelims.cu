// Copyright 2025 Ryan Curtin (http://www.ratml.org/)
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

// These statically-compiled definitions are available in any Bandicoot kernel.
#define uchar  unsigned char
#define ushort unsigned short
#define uint   unsigned int

#define COOT_FN2(ARG1, ARG2)  ARG1 ## ARG2
#define COOT_FN(ARG1, ARG2) COOT_FN2(ARG1, ARG2)

#define UWORD  size_t

// For older CUDA toolkit versions, we must manually make FP16 limit macros
// available.
#if CUDA_VERSION < 12020
  #define CUDART_INF_FP16            __ushort_as_half((unsigned short)0x7C00U)
  #define CUDART_NAN_FP16            __ushort_as_half((unsigned short)0x7FFFU)
  #define CUDART_MIN_DENORM_FP16     __ushort_as_half((unsigned short)0x0001U)
  #define CUDART_MAX_NORMAL_FP16     __ushort_as_half((unsigned short)0x7BFFU)
  #define CUDART_NEG_ZERO_FP16       __ushort_as_half((unsigned short)0x8000U)
  #define CUDART_ZERO_FP16           __ushort_as_half((unsigned short)0x0000U)
  #define CUDART_ONE_FP16            __ushort_as_half((unsigned short)0x3C00U)
#endif

extern __shared__ char aux_shared_mem[]; // this may be used in some kernels
