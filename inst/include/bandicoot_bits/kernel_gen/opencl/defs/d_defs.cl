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

inline double coot_type_min_double(const double x)    { return -DBL_MAX; }
inline double coot_type_minpos_double(const double x) { return DBL_MIN; }
inline double coot_type_max_double(const double x)    { return DBL_MAX; }

inline bool coot_is_fp_double()               { return true; }
inline bool coot_is_signed_double()           { return true; }
inline bool coot_isnan_double(const double x) { return isnan(x); }
inline bool coot_isinf_double(const double x) { return isinf(x); }

inline double coot_absdiff_double(const double x, const double y) { return fabs(x - y); }
inline double coot_conj_double(const double x) { return x; }
inline double coot_abs_double(const double x) { return fabs(x); }
inline double coot_min_double(const double x, const double y) { return fmin(x, y); }
inline double coot_max_double(const double x, const double y) { return fmax(x, y); }

// Basic mathematical operators.
inline double coot_plus_double(const double x, const double y)  { return x + y; }
inline double coot_minus_double(const double x, const double y) { return x - y; }
inline double coot_times_double(const double x, const double y) { return x * y; }
inline double coot_div_double(const double x, const double y)   { return x / y; }

)"
