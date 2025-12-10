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

// Utility functions for s32 elements.
__device__ inline bool coot_is_fp(const int)       { return false; }
__device__ inline bool coot_is_signed(const int)   { return true; }
__device__ inline int  coot_type_min(const int)    { return COOT_S32_MIN; }
__device__ inline int  coot_type_minpos(const int) { return 1; }
__device__ inline int  coot_type_max(const int)    { return COOT_S32_MAX; }
__device__ inline bool coot_isnan(const int)       { return false; }
__device__ inline bool coot_isinf(const int)       { return false; }
__device__ inline bool coot_isfinite(const int)    { return true; }

// Conversion functions for s32 elements.
__device__ inline int coot_to_int(const  uchar& x) { return (int) x; }
__device__ inline int coot_to_int(const   char& x) { return (int) x; }
__device__ inline int coot_to_int(const ushort& x) { return (int) x; }
__device__ inline int coot_to_int(const  short& x) { return (int) x; }
__device__ inline int coot_to_int(const   uint& x) { return (int) x; }
__device__ inline int coot_to_int(const    int& x) { return (int) x; }
__device__ inline int coot_to_int(const size_t& x) { return (int) x; }
__device__ inline int coot_to_int(const   long& x) { return (int) x; }
#if defined(COOT_HAVE_FP16)
__device__ inline int coot_to_int(const __half& x) { return (int) x; }
#endif
__device__ inline int coot_to_int(const  float& x) { return (int) x; }
__device__ inline int coot_to_int(const double& x) { return (int) x; }

// Utility mathematical functions.
__device__ inline int coot_absdiff(const int x, const int y) { return abs(x - y); }
