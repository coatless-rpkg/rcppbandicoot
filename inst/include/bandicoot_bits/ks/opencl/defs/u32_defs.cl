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

inline uint coot_type_min_uint()    { return 0; }
inline uint coot_type_minpos_uint() { return 1; }
inline uint coot_type_max_uint()    { return COOT_U32_MAX; }

inline bool coot_is_fp_uint()             { return false; }
inline bool coot_is_signed_uint()         { return false; }
inline bool coot_isnan_uint(const uint x) { return false; }

inline uint coot_absdiff_uint(const uint x, const uint y) { return (x > y) ? (x - y) : (y - x); }

inline uint coot_conj_uint(const uint x) { return x; }
