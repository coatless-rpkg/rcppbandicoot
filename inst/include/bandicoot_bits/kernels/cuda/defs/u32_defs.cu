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

// Utility functions for u32 elements.
__device__ inline bool coot_is_fp(const uint)       { return false; }
__device__ inline bool coot_is_signed(const uint)   { return false; }
__device__ inline uint coot_type_min(const uint)    { return 0; }
__device__ inline uint coot_type_minpos(const uint) { return 1; }
__device__ inline uint coot_type_max(const uint)    { return COOT_U32_MAX; }
__device__ inline bool coot_isnan(const uint)       { return false; }
__device__ inline bool coot_isinf(const uint)       { return false; }
__device__ inline bool coot_isfinite(const uint)    { return true; }

// Conversion functions for u32 elements.
__device__ inline uint coot_to_uint(const  uchar& x) { return (uint) x; }
__device__ inline uint coot_to_uint(const   char& x) { return (uint) x; }
__device__ inline uint coot_to_uint(const ushort& x) { return (uint) x; }
__device__ inline uint coot_to_uint(const  short& x) { return (uint) x; }
__device__ inline uint coot_to_uint(const   uint& x) { return (uint) x; }
__device__ inline uint coot_to_uint(const    int& x) { return (uint) x; }
__device__ inline uint coot_to_uint(const size_t& x) { return (uint) x; }
__device__ inline uint coot_to_uint(const   long& x) { return (uint) x; }
#if defined(COOT_HAVE_FP16)
__device__ inline uint coot_to_uint(const __half& x) { return (uint) x; }
#endif
__device__ inline uint coot_to_uint(const  float& x) { return (uint) x; }
__device__ inline uint coot_to_uint(const double& x) { return (uint) x; }

// Utility mathematical functions.
__device__ inline uint coot_absdiff(const uint x, const uint y) { return (x > y) ? (x - y) : (y - x); }
