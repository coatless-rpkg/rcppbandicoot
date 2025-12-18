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

// Utility functions for u16 elements.
__device__ inline bool  coot_is_fp(const short)       { return false; }
__device__ inline bool  coot_is_signed(const short)   { return true; }
__device__ inline short coot_type_min(const short)    { return COOT_S16_MIN; }
__device__ inline short coot_type_minpos(const short) { return 1; }
__device__ inline short coot_type_max(const short)    { return COOT_S16_MAX; }
__device__ inline bool  coot_isnan(const short)       { return false; }
__device__ inline bool  coot_isinf(const short)       { return false; }
__device__ inline bool  coot_isfinite(const short)    { return true; }

// Conversion functions for u16 elements.
__device__ inline short coot_to_short(const  uchar& x) { return (short) x; }
__device__ inline short coot_to_short(const   char& x) { return (short) x; }
__device__ inline short coot_to_short(const ushort& x) { return (short) x; }
__device__ inline short coot_to_short(const  short& x) { return (short) x; }
__device__ inline short coot_to_short(const   uint& x) { return (short) x; }
__device__ inline short coot_to_short(const    int& x) { return (short) x; }
__device__ inline short coot_to_short(const size_t& x) { return (short) x; }
__device__ inline short coot_to_short(const   long& x) { return (short) x; }
#if defined(COOT_HAVE_FP16)
__device__ inline short coot_to_short(const __half& x) { return (short) x; }
#endif
__device__ inline short coot_to_short(const  float& x) { return (short) x; }
__device__ inline short coot_to_short(const double& x) { return (short) x; }

// Utility mathematical functions.
__device__ inline short coot_absdiff(const short x, const short y) { return abs(x - y); }
