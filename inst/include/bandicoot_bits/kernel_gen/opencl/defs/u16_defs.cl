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

inline ushort coot_type_min_ushort(const ushort x)    { return 0; }
inline ushort coot_type_minpos_ushort(const ushort x) { return 1; }
inline ushort coot_type_max_ushort(const ushort x)    { return COOT_U16_MAX; }

inline bool coot_is_fp_ushort()               { return false; }
inline bool coot_is_signed_ushort()           { return false; }
inline bool coot_isnan_ushort(const ushort x) { return false; }
inline bool coot_isinf_ushort(const ushort x) { return false; }

// Conversion operators.
inline ushort coot_to_ushort_uchar(const         uchar x) { return (ushort) x;   }
inline ushort coot_to_ushort_ushort(const       ushort x) { return x;            }
inline ushort coot_to_ushort_uint(const           uint x) { return (ushort) x;   }
inline ushort coot_to_ushort_ulong(const         ulong x) { return (ushort) x;   }
inline ushort coot_to_ushort_char(const           char x) { return (ushort) x;   }
inline ushort coot_to_ushort_short(const         short x) { return (ushort) x;   }
inline ushort coot_to_ushort_int(const             int x) { return (ushort) x;   }
inline ushort coot_to_ushort_long(const           long x) { return (ushort) x;   }
#ifdef COOT_HAVE_FP16
inline ushort coot_to_ushort_half(const           half x) { return (ushort) x;   }
#endif
inline ushort coot_to_ushort_float(const         float x) { return (ushort) x;   }
inline ushort coot_to_ushort_cx_float(const   cx_float x) { return (ushort) x.x; }
#ifdef COOT_HAVE_FP64
inline ushort coot_to_ushort_double(const       double x) { return (ushort) x;   }
inline ushort coot_to_ushort_cx_double(const cx_double x) { return (ushort) x.x; }
#endif

inline ushort coot_absdiff_ushort(const ushort x, const ushort y) { return (x > y) ? (x - y) : (y - x); }
inline ushort coot_conj_ushort(const ushort x) { return x; }
inline ushort coot_abs_ushort(const ushort x) { return x; }
inline ushort coot_min_ushort(const ushort x, const ushort y) { return (x < y) ? x : y; }
inline ushort coot_max_ushort(const ushort x, const ushort y) { return (x > y) ? x : y; }

// Basic mathematical operators.
inline ushort coot_plus_ushort(const ushort x, const ushort y)  { return x + y; }
inline ushort coot_minus_ushort(const ushort x, const ushort y) { return x - y; }
inline ushort coot_times_ushort(const ushort x, const ushort y) { return x * y; }
inline ushort coot_div_ushort(const ushort x, const ushort y)   { return x / y; }

)"
