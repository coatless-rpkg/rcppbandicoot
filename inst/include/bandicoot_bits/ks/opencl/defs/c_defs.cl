// Copyright 2026 Ryan Curtin (http://www.ratml.org/)
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

inline cx_float coot_type_min_cx_float()    { return (cx_float)(-FLT_MAX, -FLT_MAX); }
inline cx_float coot_type_minpos_cx_float() { return (cx_float)(FLT_MIN, FLT_MIN); }
inline cx_float coot_type_max_cx_float()    { return (cx_float)(FLT_MAX, FLT_MAX); }

inline bool coot_is_fp_cx_float()                 { return true; }
inline bool coot_is_signed_cx_float()             { return true; }
inline bool coot_isnan_cx_float(const cx_float x) { return isnan(x.x) || isnan(x.y); }

inline cx_float coot_absdiff_cx_float(const cx_float x, const cx_float y) { return (cx_float)(fabs(x.x - y.x), fabs(x.y - y.y)); }

inline cx_float coot_conj_cx_float(const cx_float x) { return (cx_float)(x.x, -x.y); }

inline cx_float coot_plus_cx_float(const cx_float x, const cx_float y)  { return (cx_float)(x.x + y.x, x.y + y.y); }
inline cx_float coot_minus_cx_float(const cx_float x, const cx_float y) { return (cx_float)(x.x - y.x, x.y - y.y); }
inline cx_float coot_mul_cx_float(const cx_float x, const cx_float y)   { return (cx_float)(x.x * y.x - x.y * y.y, x.x * y.y + x.y * y.x); }
inline cx_float coot_div_cx_float(const cx_float x, const cx_float y)   { return (cx_float)((x.x * y.x + x.y * y.y) / (y.x * y.x + y.y * y.y), (x.y * y.x - x.x * y.y) / (y.x * y.x + y.y * y.y)); }
