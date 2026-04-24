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



// Utility functions for cx_float elements.
__device__ inline bool      coot_is_fp(const cx_float)       { return true; }
__device__ inline bool      coot_is_signed(const cx_float)   { return true; }
__device__ inline cx_float coot_type_min(const cx_float)    { return make_cuFloatComplex(-FLT_MAX, -FLT_MAX); }
__device__ inline cx_float coot_type_minpos(const cx_float) { return make_cuFloatComplex(FLT_MIN, FLT_MIN); }
__device__ inline cx_float coot_type_max(const cx_float)    { return make_cuFloatComplex(FLT_MAX, FLT_MAX); }
__device__ inline bool      coot_isnan(const cx_float x)     { return isnan(x.x) || isnan(x.y); }
__device__ inline bool      coot_isinf(const cx_float x)     { return isinf(x.x) || isinf(x.y); }
__device__ inline bool      coot_isfinite(const cx_float x)  { return isfinite(x.x) && isfinite(x.y); }

// Conversion functions for cx_float elements.
__device__ inline cx_float coot_to_cx_float(const cx_float&  x) { return x; }
__device__ inline cx_float coot_to_cx_float(const cx_double& x) { return make_cuFloatComplex((float) x.x, (float) x.y); }

// Utility mathematical functions.
__device__ inline cx_float coot_absdiff(const cx_float x, const cx_float y) { return make_cuFloatComplex(fabs(x.x - y.x), fabs(x.y - y.y)); }

// Overload arithmetic operators so we can use complex numbers as though they
// are primitive types.
__device__ inline cx_float operator+(const cx_float& a, const cx_float& b) { return cuCaddf(a, b); }
__device__ inline cx_float operator-(const cx_float& a, const cx_float& b) { return cuCsubf(a, b); }
__device__ inline cx_float operator*(const cx_float& a, const cx_float& b) { return cuCmulf(a, b); }
__device__ inline cx_float operator/(const cx_float& a, const cx_float& b) { return cuCdivf(a, b); }
