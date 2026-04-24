// Copyright 2026 Ryan Curtin (http://www.ratml.org/)
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



// Utility functions for cx_double elements.
__device__ inline bool      coot_is_fp(const cx_double)       { return true; }
__device__ inline bool      coot_is_signed(const cx_double)   { return true; }
__device__ inline cx_double coot_type_min(const cx_double)    { return make_cuDoubleComplex(-DBL_MAX, -DBL_MAX); }
__device__ inline cx_double coot_type_minpos(const cx_double) { return make_cuDoubleComplex(DBL_MIN, DBL_MIN); }
__device__ inline cx_double coot_type_max(const cx_double)    { return make_cuDoubleComplex(DBL_MAX, DBL_MAX); }
__device__ inline bool      coot_isnan(const cx_double x)     { return isnan(x.x) || isnan(x.y); }
__device__ inline bool      coot_isinf(const cx_double x)     { return isinf(x.x) || isinf(x.y); }
__device__ inline bool      coot_isfinite(const cx_double x)  { return isfinite(x.x) && isfinite(x.y); }

// Conversion functions for cx_double elements.
__device__ inline cx_double coot_to_cx_double(const cx_float&  x) { return make_cuDoubleComplex((double) x.x, (double) x.y); }
__device__ inline cx_double coot_to_cx_double(const cx_double& x) { return x; }

// Utility mathematical functions.
__device__ inline cx_double coot_absdiff(const cx_double x, const cx_double y) { return make_cuDoubleComplex(fabs(x.x - y.x), fabs(x.y - y.y)); }

// Overload arithmetic operators so we can use complex numbers as though they
// are primitive types.
__device__ inline cx_double operator+(const cx_double& a, const cx_double& b) { return cuCadd(a, b); }
__device__ inline cx_double operator-(const cx_double& a, const cx_double& b) { return cuCsub(a, b); }
__device__ inline cx_double operator*(const cx_double& a, const cx_double& b) { return cuCmul(a, b); }
__device__ inline cx_double operator/(const cx_double& a, const cx_double& b) { return cuCdiv(a, b); }
