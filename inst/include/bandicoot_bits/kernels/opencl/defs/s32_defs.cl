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

inline int coot_type_min_int()    { return COOT_S32_MIN; }
inline int coot_type_minpos_int() { return 1; }
inline int coot_type_max_int()    { return COOT_S32_MAX; }

inline bool coot_is_fp_int()            { return false; }
inline bool coot_is_signed_int()        { return true; }
inline bool coot_isnan_int(const int x) { return false; }

inline int coot_absdiff_int(const int x, const int y) { return abs(x - y); }

inline int coot_conj_int(const int x) { return x; }
