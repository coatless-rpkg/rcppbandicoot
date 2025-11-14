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
__device__ inline bool   coot_is_fp(const ushort)       { return false; }
__device__ inline bool   coot_is_signed(const ushort)   { return false; }
__device__ inline ushort coot_type_min(const ushort)    { return 0; }
__device__ inline ushort coot_type_minpos(const ushort) { return 1; }
__device__ inline ushort coot_type_max(const ushort)    { return COOT_U16_MAX; }
__device__ inline bool   coot_isnan(const ushort)       { return false; }
__device__ inline bool   coot_isinf(const ushort)       { return false; }
__device__ inline bool   coot_isfinite(const ushort)    { return true; }

// Conversion functions for u16 elements.
__device__ inline ushort coot_to_ushort(const  uchar& x) { return (ushort) x; }
__device__ inline ushort coot_to_ushort(const   char& x) { return (ushort) x; }
__device__ inline ushort coot_to_ushort(const ushort& x) { return (ushort) x; }
__device__ inline ushort coot_to_ushort(const  short& x) { return (ushort) x; }
__device__ inline ushort coot_to_ushort(const   uint& x) { return (ushort) x; }
__device__ inline ushort coot_to_ushort(const    int& x) { return (ushort) x; }
__device__ inline ushort coot_to_ushort(const size_t& x) { return (ushort) x; }
__device__ inline ushort coot_to_ushort(const   long& x) { return (ushort) x; }
#if defined(COOT_HAVE_FP16)
__device__ inline ushort coot_to_ushort(const __half& x) { return (ushort) x; }
#endif
__device__ inline ushort coot_to_ushort(const  float& x) { return (ushort) x; }
__device__ inline ushort coot_to_ushort(const double& x) { return (ushort) x; }

// Utility mathematical functions.
__device__ inline ushort coot_absdiff(const ushort x, const ushort y) { return (x > y) ? (x - y) : (y - x); }
