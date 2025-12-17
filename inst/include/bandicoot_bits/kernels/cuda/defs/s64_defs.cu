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

// Utility functions for s64 elements.
__device__ inline bool coot_is_fp(const long)       { return false; }
__device__ inline bool coot_is_signed(const long)   { return true; }
__device__ inline long coot_type_min(const long)    { return COOT_S64_MIN; }
__device__ inline long coot_type_minpos(const long) { return 1; }
__device__ inline long coot_type_max(const long)    { return COOT_S64_MAX; }
__device__ inline bool coot_isnan(const long)       { return false; }
__device__ inline bool coot_isinf(const long)       { return false; }
__device__ inline bool coot_isfinite(const long)    { return true; }

// Conversion functions for s64 elements.
__device__ inline long coot_to_long(const  uchar& x) { return (long) x; }
__device__ inline long coot_to_long(const   char& x) { return (long) x; }
__device__ inline long coot_to_long(const ushort& x) { return (long) x; }
__device__ inline long coot_to_long(const  short& x) { return (long) x; }
__device__ inline long coot_to_long(const   uint& x) { return (long) x; }
__device__ inline long coot_to_long(const    int& x) { return (long) x; }
__device__ inline long coot_to_long(const size_t& x) { return (long) x; }
__device__ inline long coot_to_long(const   long& x) { return (long) x; }
#if defined(COOT_HAVE_FP16)
#if CUDA_VERSION < 12020
__device__ inline long coot_to_long(const __half& x) { return (long) ((long long) x); }
#else
__device__ inline long coot_to_long(const __half& x) { return (long) x; }
#endif
#endif
__device__ inline long coot_to_long(const  float& x) { return (long) x; }
__device__ inline long coot_to_long(const double& x) { return (long) x; }

// Utility mathematical functions.
__device__ inline long coot_absdiff(const long x, const long y) { return abs(x - y); }
