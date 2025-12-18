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

// Utility functions for s8 elements.
__device__ inline bool coot_is_fp(const char)       { return false; }
__device__ inline bool coot_is_signed(const char)   { return true; }
__device__ inline bool coot_type_min(const char)    { return COOT_S8_MIN; }
__device__ inline bool coot_type_minpos(const char) { return 1; }
__device__ inline bool coot_type_max(const char)    { return COOT_S8_MAX; }
__device__ inline bool coot_isnan(const char)       { return false; }
__device__ inline bool coot_isinf(const char)       { return false; }
__device__ inline bool coot_isfinite(const char)    { return true; }

// Conversion functions for s8 elements.
__device__ inline char coot_to_char(const  uchar& x) { return (char) x; }
__device__ inline char coot_to_char(const   char& x) { return (char) x; }
__device__ inline char coot_to_char(const ushort& x) { return (char) x; }
__device__ inline char coot_to_char(const  short& x) { return (char) x; }
__device__ inline char coot_to_char(const   uint& x) { return (char) x; }
__device__ inline char coot_to_char(const    int& x) { return (char) x; }
__device__ inline char coot_to_char(const size_t& x) { return (char) x; }
__device__ inline char coot_to_char(const   long& x) { return (char) x; }
#if defined(COOT_HAVE_FP16)
#if CUDA_VERSION < 12020
__device__ inline char coot_to_char(const __half& x) { return (char) ((short) x); }
#else
__device__ inline char coot_to_char(const __half& x) { return (char) x; }
#endif
#endif
__device__ inline char coot_to_char(const  float& x) { return (char) x; }
__device__ inline char coot_to_char(const double& x) { return (char) x; }

// Utility mathematical functions.
__device__ inline char coot_absdiff(const char x, const char y) { return abs(x - y); }
