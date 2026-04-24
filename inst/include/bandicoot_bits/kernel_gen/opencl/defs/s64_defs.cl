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

inline long coot_type_min_long(const long x)    { return COOT_S64_MIN; }
inline long coot_type_minpos_long(const long x) { return 1; }
inline long coot_type_max_long(const long x)    { return COOT_S64_MAX; }

inline bool coot_is_fp_long()             { return false; }
inline bool coot_is_signed_long()         { return true; }
inline bool coot_isnan_long(const long x) { return false; }
inline bool coot_isinf_long(const long x) { return false; }

inline long coot_absdiff_long(const long x, const long y) { return abs(x - y); }
inline long coot_conj_long(const long x) { return x; }
inline long coot_abs_long(const long x) { return abs(x); }
inline long coot_min_long(const long x, const long y) { return (x < y) ? x : y; }
inline long coot_max_long(const long x, const long y) { return (x > y) ? x : y; }

// Basic mathematical operators.
inline long coot_plus_long(const long x, const long y)  { return x + y; }
inline long coot_minus_long(const long x, const long y) { return x - y; }
inline long coot_times_long(const long x, const long y) { return x * y; }
inline long coot_div_long(const long x, const long y)   { return x / y; }

)"
