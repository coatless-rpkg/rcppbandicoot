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

// Utility functions for float elements.
__device__ inline bool  coot_is_fp(const float)       { return true; }
__device__ inline bool  coot_is_signed(const float)   { return true; }
__device__ inline float coot_type_min(const float)    { return -FLT_MAX; }
__device__ inline float coot_type_minpos(const float) { return FLT_MIN; }
__device__ inline float coot_type_max(const float)    { return FLT_MAX; }
__device__ inline bool  coot_isnan(const float x)     { return isnan(x); }
__device__ inline bool  coot_isinf(const float x)     { return isinf(x); }
__device__ inline bool  coot_isfinite(const float x)  { return isfinite(x); }

// Conversion functions for float elements.
__device__ inline float coot_to_float(const  uchar& x) { return (float) x; }
__device__ inline float coot_to_float(const   char& x) { return (float) x; }
__device__ inline float coot_to_float(const ushort& x) { return (float) x; }
__device__ inline float coot_to_float(const  short& x) { return (float) x; }
__device__ inline float coot_to_float(const   uint& x) { return (float) x; }
__device__ inline float coot_to_float(const    int& x) { return (float) x; }
__device__ inline float coot_to_float(const size_t& x) { return (float) x; }
__device__ inline float coot_to_float(const   long& x) { return (float) x; }
#if defined(COOT_HAVE_FP16)
__device__ inline float coot_to_float(const __half& x) { return __half2float(x); }
#endif
__device__ inline float coot_to_float(const  float& x) { return (float) x; }
__device__ inline float coot_to_float(const double& x) { return (float) x; }

// Utility mathematical functions.
__device__ inline float coot_absdiff(const float x, const float y) { return fabs(x - y); }
