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

inline char coot_type_min_char(const char x)    { return COOT_S8_MIN; }
inline char coot_type_minpos_char(const char x) { return 1; }
inline char coot_type_max_char(const char x)    { return COOT_S8_MAX; }

inline bool coot_is_fp_char()             { return false; }
inline bool coot_is_signed_char()         { return true; }
inline bool coot_isnan_char(const char x) { return false; }
inline bool coot_isinf_char(const char x) { return false; }

// Conversion operators.
inline char coot_to_char_uchar(const         uchar x) { return (char) x;   }
inline char coot_to_char_ushort(const       ushort x) { return (char) x;   }
inline char coot_to_char_uint(const           uint x) { return (char) x;   }
inline char coot_to_char_ulong(const         ulong x) { return (char) x;   }
inline char coot_to_char_char(const           char x) { return x;          }
inline char coot_to_char_short(const         short x) { return (char) x;   }
inline char coot_to_char_int(const             int x) { return (char) x;   }
inline char coot_to_char_long(const           long x) { return (char) x;   }
#ifdef COOT_HAVE_FP16
inline char coot_to_char_half(const           half x) { return (char) x;   }
#endif
inline char coot_to_char_float(const         float x) { return (char) x;   }
inline char coot_to_char_cx_float(const   cx_float x) { return (char) x.x; }
#ifdef COOT_HAVE_FP64
inline char coot_to_char_double(const       double x) { return (char) x;   }
inline char coot_to_char_cx_double(const cx_double x) { return (char) x.x; }
#endif

inline char coot_absdiff_char(const char x, const char y) { return abs(x - y); }
inline char coot_conj_char(const char x) { return x; }
inline char coot_abs_char(const char x) { return abs(x); }
inline char coot_min_char(const char x, const char y) { return (x < y) ? x : y; }
inline char coot_max_char(const char x, const char y) { return (x > y) ? x : y; }

// Basic mathematical operators.
inline char coot_plus_char(const char x, const char y)  { return x + y; }
inline char coot_minus_char(const char x, const char y) { return x - y; }
inline char coot_times_char(const char x, const char y) { return x * y; }
inline char coot_div_char(const char x, const char y)   { return x / y; }

)"
