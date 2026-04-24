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

inline ulong coot_type_min_ulong(const ulong x)    { return 0; }
inline ulong coot_type_minpos_ulong(const ulong x) { return 1; }
inline ulong coot_type_max_ulong(const ulong x)    { return COOT_U64_MAX; }

inline bool coot_is_fp_ulong()              { return false; }
inline bool coot_is_signed_ulong()          { return false; }
inline bool coot_isnan_ulong(const ulong x) { return false; }
inline bool coot_isinf_ulong(const ulong x) { return false; }

inline ulong coot_absdiff_ulong(const ulong x, const ulong y) { return (x > y) ? (x - y) : (y - x); }
inline ulong coot_conj_ulong(const ulong x) { return x; }
inline ulong coot_abs_ulong(const ulong x) { return x; }
inline ulong coot_min_ulong(const ulong x, const ulong y) { return (x < y) ? x : y; }
inline ulong coot_max_ulong(const ulong x, const ulong y) { return (x > y) ? x : y; }

// Basic mathematical operators.
inline ulong coot_plus_ulong(const ulong x, const ulong y)  { return x + y; }
inline ulong coot_minus_ulong(const ulong x, const ulong y) { return x - y; }
inline ulong coot_times_ulong(const ulong x, const ulong y) { return x * y; }
inline ulong coot_div_ulong(const ulong x, const ulong y)   { return x / y; }

)"
