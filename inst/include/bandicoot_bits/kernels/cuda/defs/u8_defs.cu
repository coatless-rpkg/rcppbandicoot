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

// Utility functions for u8 elements.
__device__ inline bool  coot_is_fp(const uchar)       { return false; }
__device__ inline bool  coot_is_signed(const uchar)   { return false; }
__device__ inline uchar coot_type_min(const uchar)    { return 0; }
__device__ inline uchar coot_type_minpos(const uchar) { return 1; }
__device__ inline uchar coot_type_max(const uchar)    { return COOT_U8_MAX; }
__device__ inline bool  coot_isnan(const uchar)       { return false; }
__device__ inline bool  coot_isinf(const uchar)       { return false; }
__device__ inline bool  coot_isfinite(const uchar)    { return true; }

// Conversion functions for u8 elements.
__device__ inline uchar coot_to_uchar(const  uchar& x) { return (uchar) x; }
__device__ inline uchar coot_to_uchar(const   char& x) { return (uchar) x; }
__device__ inline uchar coot_to_uchar(const ushort& x) { return (uchar) x; }
__device__ inline uchar coot_to_uchar(const  short& x) { return (uchar) x; }
__device__ inline uchar coot_to_uchar(const   uint& x) { return (uchar) x; }
__device__ inline uchar coot_to_uchar(const    int& x) { return (uchar) x; }
__device__ inline uchar coot_to_uchar(const size_t& x) { return (uchar) x; }
__device__ inline uchar coot_to_uchar(const   long& x) { return (uchar) x; }
#if defined(COOT_HAVE_FP16)
#if CUDA_VERSION < 12020
__device__ inline uchar coot_to_uchar(const __half& x) { return (uchar) ((ushort) x); }
#else
__device__ inline uchar coot_to_uchar(const __half& x) { return (uchar) x; }
#endif
#endif
__device__ inline uchar coot_to_uchar(const  float& x) { return (uchar) x; }
__device__ inline uchar coot_to_uchar(const double& x) { return (uchar) x; }

// Utility mathematical functions.
__device__ inline uchar coot_absdiff(const uchar x, const uchar y) { return (x > y) ? (x - y) : (y - x); }
