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
R"(

inline half coot_type_min_half(const half x)    { return -HALF_MAX; }
inline half coot_type_minpos_half(const half x) { return HALF_MIN; }
inline half coot_type_max_half(const half x)    { return HALF_MAX; }

inline bool coot_is_fp_half()             { return true; }
inline bool coot_is_signed_half()         { return true; }
inline bool coot_isnan_half(const half x) { return isnan(x); }
inline bool coot_isinf_half(const half x) { return isinf(x); }

inline half coot_absdiff_half(const half x, const half y) { return fabs(x - y); }
inline half coot_conj_half(const half x) { return x; }
inline half coot_abs_half(const half x) { return fabs(x); }
inline half coot_min_half(const half x, const half y) { return fmin(x, y); }
inline half coot_max_half(const half x, const half y) { return fmax(x, y); }

// Basic mathematical operators.
inline half coot_plus_half(const half x, const half y)  { return x + y; }
inline half coot_minus_half(const half x, const half y) { return x - y; }
inline half coot_times_half(const half x, const half y) { return x * y; }
inline half coot_div_half(const half x, const half y)   { return x / y; }

)"
