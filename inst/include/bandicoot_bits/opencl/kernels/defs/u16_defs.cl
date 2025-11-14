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

inline ushort coot_type_min_ushort()    { return 0; }
inline ushort coot_type_minpos_ushort() { return 1; }
inline ushort coot_type_max_ushort()    { return COOT_U16_MAX; }

inline bool coot_is_fp_ushort()               { return false; }
inline bool coot_is_signed_ushort()           { return false; }
inline bool coot_isnan_ushort(const ushort x) { return false; }

inline ushort coot_absdiff_ushort(const ushort x, const ushort y) { return (x > y) ? (x - y) : (y - x); }

inline ushort coot_conj_ushort(const ushort x) { return x; }
