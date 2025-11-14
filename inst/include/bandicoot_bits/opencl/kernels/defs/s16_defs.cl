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

inline short coot_type_min_short()    { return COOT_S16_MIN; }
inline short coot_type_minpos_short() { return 1; }
inline short coot_type_max_short()    { return COOT_S16_MAX; }

inline bool coot_is_fp_short()              { return false; }
inline bool coot_is_signed_short()          { return true; }
inline bool coot_isnan_short(const short x) { return false; }

inline short coot_absdiff_short(const short x, const short y) { return abs(x - y); }

inline short coot_conj_short(const short x) { return x; }
