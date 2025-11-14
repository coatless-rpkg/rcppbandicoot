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

inline uchar coot_type_min_uchar()    { return 0; }
inline uchar coot_type_minpos_uchar() { return 1; }
inline uchar coot_type_max_uchar()    { return COOT_U8_MAX; }

inline bool coot_is_fp_uchar()              { return false; }
inline bool coot_is_signed_uchar()          { return false; }
inline bool coot_isnan_uchar(const uchar x) { return false; }

inline uchar coot_absdiff_uchar(const uchar x, const uchar y) { return (x > y) ? (x - y) : (y - x); }

inline uchar coot_conj_uchar(const uchar x) { return x; }
