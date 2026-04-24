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

// Utility functions for fp16 elements.
__device__ inline bool   coot_is_fp(const __half)       { return true; }
__device__ inline bool   coot_is_signed(const __half)   { return true; }
__device__ inline __half coot_type_min(const __half)    { return -HALF_MAX; }
__device__ inline __half coot_type_minpos(const __half) { return HALF_MIN; }
__device__ inline __half coot_type_max(const __half)    { return HALF_MAX; }
__device__ inline bool   coot_isnan(const __half x)     { return __hisnan(x); }
__device__ inline bool   coot_isinf(const __half x)     { return __hisinf(x); }
__device__ inline bool   coot_isfinite(const __half x)  { return !__hisnan(x) && !__hisinf(x); }

// Conversion functions for fp16 elements.
#if CUDA_VERSION < 12020
__device__ inline __half coot_to___half(const  uchar& x) { return (__half) ((ushort) x); }
__device__ inline __half coot_to___half(const   char& x) { return (__half) ((short) x); }
#else
__device__ inline __half coot_to___half(const  uchar& x) { return (__half) x; }
__device__ inline __half coot_to___half(const   char& x) { return (__half) x; }
#endif
__device__ inline __half coot_to___half(const ushort& x) { return (__half) x; }
__device__ inline __half coot_to___half(const  short& x) { return (__half) x; }
__device__ inline __half coot_to___half(const   uint& x) { return (__half) x; }
__device__ inline __half coot_to___half(const    int& x) { return (__half) x; }
#if CUDA_VERSION < 12020
__device__ inline __half coot_to___half(const size_t& x) { return (__half) ((unsigned long long) x); }
__device__ inline __half coot_to___half(const   long& x) { return (__half) ((long long) x); }
#else
__device__ inline __half coot_to___half(const size_t& x) { return (__half) x; }
__device__ inline __half coot_to___half(const   long& x) { return (__half) x; }
#endif
__device__ inline __half coot_to___half(const __half& x) { return (__half) x; }
__device__ inline __half coot_to___half(const  float& x) { return __float2half(x); }
__device__ inline __half coot_to___half(const double& x) { return __float2half((double) x); }

// CUDA FP16 support does not include some arithmetic operators that we need for volatile elements so we add them ourselves...
#if CUDA_VERSION < 12040
__device__ inline volatile __half& operator+=(volatile __half& a, const volatile __half& b) { a = __hadd((__half) a, (__half) b); return a; }
__device__ inline volatile __half& operator-=(volatile __half& a, const volatile __half& b) { a = __hsub((__half) a, (__half) b); return a; }
__device__ inline volatile __half& operator*=(volatile __half& a, const volatile __half& b) { a = __hmul((__half) a, (__half) b); return a; }
__device__ inline volatile __half& operator/=(volatile __half& a, const volatile __half& b) { a = __hdiv((__half) a, (__half) b); return a; }
#else
__device__ inline volatile __half& operator+=(volatile __half& a, const volatile __half& b) { a = __hadd(a, b); return a; }
__device__ inline volatile __half& operator-=(volatile __half& a, const volatile __half& b) { a = __hsub(a, b); return a; }
__device__ inline volatile __half& operator*=(volatile __half& a, const volatile __half& b) { a = __hmul(a, b); return a; }
__device__ inline volatile __half& operator/=(volatile __half& a, const volatile __half& b) { a = __hdiv(a, b); return a; }
#endif
__device__ inline __half abs(const __half a) { return __habs(a); }
__device__ inline __half pow(const __half a, const __half b) { return hexp2(b * hlog2(a)); }
__device__ inline __half min(const __half a, const __half b) { return __hmin_nan(a, b); }
__device__ inline __half max(const __half a, const __half b) { return __hmax_nan(a, b); }

// Utility mathematical functions.
__device__ inline __half coot_absdiff(const __half x, const __half y) { return fabs(x - y); }
