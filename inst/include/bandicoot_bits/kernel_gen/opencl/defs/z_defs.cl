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
R"(

inline cx_double coot_type_min_cx_double(const cx_double x)    { return (cx_double)(-DBL_MAX, -DBL_MAX); }
inline cx_double coot_type_minpos_cx_double(const cx_double x) { return (cx_double)(DBL_MIN, DBL_MIN); }
inline cx_double coot_type_max_cx_double(const cx_double x)    { return (cx_double)(DBL_MAX, DBL_MAX); }

inline bool coot_is_fp_cx_double()                 { return true; }
inline bool coot_is_signed_cx_double()             { return true; }
inline bool coot_isnan_cx_double(const cx_double x) { return isnan(x.x) || isnan(x.y); }
inline bool coot_isinf_cx_double(const cx_double x) { return isnan(x.x) || isnan(x.y); }

// Conversion operators.
inline cx_double coot_to_cx_double_uchar(const         uchar x) { return (cx_double)((double) x, (double) 0);     }
inline cx_double coot_to_cx_double_ushort(const       ushort x) { return (cx_double)((double) x, (double) 0);     }
inline cx_double coot_to_cx_double_uint(const           uint x) { return (cx_double)((double) x, (double) 0);     }
inline cx_double coot_to_cx_double_ulong(const         ulong x) { return (cx_double)((double) x, (double) 0);     }
inline cx_double coot_to_cx_double_char(const           char x) { return (cx_double)((double) x, (double) 0);     }
inline cx_double coot_to_cx_double_short(const         short x) { return (cx_double)((double) x, (double) 0);     }
inline cx_double coot_to_cx_double_int(const             int x) { return (cx_double)((double) x, (double) 0);     }
inline cx_double coot_to_cx_double_long(const           long x) { return (cx_double)((double) x, (double) 0);     }
#ifdef COOT_HAVE_FP16
inline cx_double coot_to_cx_double_half(const           half x) { return (cx_double)((double) x, (double) 0);     }
#endif
inline cx_double coot_to_cx_double_float(const         float x) { return (cx_double)((double) x, (double) 0);     }
inline cx_double coot_to_cx_double_cx_float(const   cx_float x) { return (cx_double)((double) x.x, (double) x.y); }
inline cx_double coot_to_cx_double_double(const       double x) { return (cx_double)(x, (double) 0);              }
inline cx_double coot_to_cx_double_cx_double(const cx_double x) { return x;                                       }

inline cx_double coot_absdiff_cx_double(const cx_double x, const cx_double y) { return (cx_double)(fabs(x.x - y.x), fabs(x.y - y.y)); }
// semi-hack: use squared norm instead of norm for magnitude check
inline cx_double coot_min_cx_double(const cx_double x, const cx_double y) { return ((x.x * x.x + x.y * x.y) < (y.x * y.x + y.y * y.y)) ? x : y; }
inline cx_double coot_max_cx_double(const cx_double x, const cx_double y) { return ((x.x * x.x + x.y * x.y) > (y.x * y.x + y.y * y.y)) ? x : y; }
inline cx_double coot_conj_cx_double(const cx_double x) { return (cx_double)(x.x, -x.y); }
inline cx_double coot_abs_cx_double(const cx_double x) { return (cx_double)(fabs(x.x), fabs(x.y)); }

// Basic mathematical operators.
inline cx_double coot_plus_cx_double(const cx_double x, const cx_double y)  { return (cx_double)(x.x + y.x, x.y + y.y); }
inline cx_double coot_minus_cx_double(const cx_double x, const cx_double y) { return (cx_double)(x.x - y.x, x.y - y.y); }
inline cx_double coot_times_cx_double(const cx_double x, const cx_double y) { return (cx_double)(x.x * y.x - x.y * y.y, x.x * y.y + x.y * y.x); }
inline cx_double coot_div_cx_double(const cx_double x, const cx_double y)   { return (cx_double)(cx_double)((x.x * y.x + x.y * y.y) / (y.x * y.x + y.y * y.y), (x.y * y.x - x.x * y.y) / (y.x * y.x + y.y * y.y)); }
inline cx_double coot_neg_cx_double(const cx_double x)                      { return (cx_double)(-x.x, -x.y); }

)"
