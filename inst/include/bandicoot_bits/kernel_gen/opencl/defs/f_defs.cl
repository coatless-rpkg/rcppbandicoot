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

inline float coot_type_min_float(const float x)    { return -FLT_MAX; }
inline float coot_type_minpos_float(const float x) { return FLT_MIN; }
inline float coot_type_max_float(const float x)    { return FLT_MAX; }

inline bool coot_is_fp_float()              { return true; }
inline bool coot_is_signed_float()          { return true; }
inline bool coot_isnan_float(const float x) { return isnan(x); }
inline bool coot_isinf_float(const float x) { return isinf(x); }

inline float coot_absdiff_float(const float x, const float y) { return fabs(x - y); }
inline float coot_conj_float(const float x) { return x; }
inline float coot_abs_float(const float x) { return fabs(x); }
inline float coot_min_float(const float x, const float y) { return fmin(x, y); }
inline float coot_max_float(const float x, const float y) { return fmax(x, y); }

// Basic mathematical operators.
inline float coot_plus_float(const float x, const float y)  { return x + y; }
inline float coot_minus_float(const float x, const float y) { return x - y; }
inline float coot_times_float(const float x, const float y) { return x * y; }
inline float coot_div_float(const float x, const float y)   { return x / y; }

)"
