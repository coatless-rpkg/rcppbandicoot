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

inline char coot_type_min_char()    { return COOT_S8_MIN; }
inline char coot_type_minpos_char() { return 1; }
inline char coot_type_max_char()    { return COOT_S8_MAX; }

inline bool coot_is_fp_char()             { return false; }
inline bool coot_is_signed_char()         { return true; }
inline bool coot_isnan_char(const char x) { return false; }

inline char coot_absdiff_char(const char x, const char y) { return abs(x - y); }

inline char coot_conj_char(const char x) { return x; }
