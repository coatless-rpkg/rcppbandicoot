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

// Utility functions for double elements.
__device__ inline bool   coot_is_fp(const double)       { return true; }
__device__ inline bool   coot_is_signed(const double)   { return true; }
__device__ inline double coot_type_min(const double)    { return -DBL_MAX; }
__device__ inline double coot_type_minpos(const double) { return DBL_MIN; }
__device__ inline double coot_type_max(const double)    { return DBL_MAX; }
__device__ inline bool   coot_isnan(const double x)     { return isnan(x); }
__device__ inline bool   coot_isinf(const double x)     { return isinf(x); }
__device__ inline bool   coot_isfinite(const double x)  { return isfinite(x); }

// Conversion functions for double elements.
__device__ inline double coot_to_double(const  uchar& x) { return (double) x; }
__device__ inline double coot_to_double(const   char& x) { return (double) x; }
__device__ inline double coot_to_double(const ushort& x) { return (double) x; }
__device__ inline double coot_to_double(const  short& x) { return (double) x; }
__device__ inline double coot_to_double(const   uint& x) { return (double) x; }
__device__ inline double coot_to_double(const    int& x) { return (double) x; }
__device__ inline double coot_to_double(const size_t& x) { return (double) x; }
__device__ inline double coot_to_double(const   long& x) { return (double) x; }
#if defined(COOT_HAVE_FP16)
__device__ inline double coot_to_double(const __half& x) { return (double) __half2float(x); }
#endif
__device__ inline double coot_to_double(const  float& x) { return (double) x; }
__device__ inline double coot_to_double(const double& x) { return (double) x; }

// Utility mathematical functions.
__device__ inline double coot_absdiff(const double x, const double y) { return fabs(x - y); }
