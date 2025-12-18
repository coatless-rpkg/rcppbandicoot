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

inline float coot_type_min_float()    { return -FLT_MAX; }
inline float coot_type_minpos_float() { return FLT_MIN; }
inline float coot_type_max_float()    { return FLT_MAX; }

inline bool coot_is_fp_float()              { return true; }
inline bool coot_is_signed_float()          { return true; }
inline bool coot_isnan_float(const float x) { return isnan(x); }

inline float coot_absdiff_float(const float x, const float y) { return fabs(x - y); }

inline    float coot_conj_float(const float x)       { return x; }
//inline cx_float coot_conj_cx_float(const cx_float x) { return cx_float(x.x, -x.y); }
