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

// Utility functions for u64 elements.
__device__ inline bool   coot_is_fp(const size_t)       { return false; }
__device__ inline bool   coot_is_signed(const size_t)   { return false; }
__device__ inline size_t coot_type_min(const size_t)    { return 0; }
__device__ inline size_t coot_type_minpos(const size_t) { return 1; }
__device__ inline size_t coot_type_max(const size_t)    { return COOT_U64_MAX; }
__device__ inline bool   coot_isnan(const size_t)       { return false; }
__device__ inline bool   coot_isinf(const size_t)       { return false; }
__device__ inline bool   coot_isfinite(const size_t)    { return true; }

// Conversion functions for u64 elements.
__device__ inline size_t coot_to_size_t(const  uchar& x) { return (size_t) x; }
__device__ inline size_t coot_to_size_t(const   char& x) { return (size_t) x; }
__device__ inline size_t coot_to_size_t(const ushort& x) { return (size_t) x; }
__device__ inline size_t coot_to_size_t(const  short& x) { return (size_t) x; }
__device__ inline size_t coot_to_size_t(const   uint& x) { return (size_t) x; }
__device__ inline size_t coot_to_size_t(const    int& x) { return (size_t) x; }
__device__ inline size_t coot_to_size_t(const size_t& x) { return (size_t) x; }
__device__ inline size_t coot_to_size_t(const   long& x) { return (size_t) x; }
#if defined(COOT_HAVE_FP16)
#if CUDA_VERSION < 12020
__device__ inline size_t coot_to_size_t(const __half& x) { return (size_t) ((unsigned long long) x); }
#else
__device__ inline size_t coot_to_size_t(const __half& x) { return (size_t) x; }
#endif
#endif
__device__ inline size_t coot_to_size_t(const  float& x) { return (size_t) x; }
__device__ inline size_t coot_to_size_t(const double& x) { return (size_t) x; }

// Utility mathematical functions.
__device__ inline size_t coot_absdiff(const size_t x, const size_t y) { return (x > y) ? (x - y) : (y - x); }
