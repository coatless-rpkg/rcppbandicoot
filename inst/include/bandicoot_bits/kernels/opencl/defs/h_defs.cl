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

inline half coot_type_min_half()    { return -HALF_MAX; }
inline half coot_type_minpos_half() { return HALF_MIN; }
inline half coot_type_max_half()    { return HALF_MAX; }

inline bool coot_is_fp_half()             { return true; }
inline bool coot_is_signed_half()         { return true; }
inline bool coot_isnan_half(const half x) { return isnan(x); }

inline half coot_absdiff_half(const half x, const half y) { return fabs(x - y); }

inline half coot_conj_half(const half x) { return x; }
