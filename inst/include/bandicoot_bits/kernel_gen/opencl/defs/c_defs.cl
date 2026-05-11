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

inline cx_float coot_type_min_cx_float(const cx_float x)    { return (cx_float)( -FLT_MAX, -FLT_MAX); }
inline cx_float coot_type_minpos_cx_float(const cx_float x) { return (cx_float)( FLT_MIN, FLT_MIN); }
inline cx_float coot_type_max_cx_float(const cx_float x)    { return (cx_float)( FLT_MAX, FLT_MAX); }

inline bool coot_is_fp_cx_float()                 { return true; }
inline bool coot_is_signed_cx_float()             { return true; }
inline bool coot_isnan_cx_float(const cx_float x) { return isnan(x.x) || isnan(x.y); }
inline bool coot_isinf_cx_float(const cx_float x) { return isnan(x.x) || isnan(x.y); }

// Conversion operators.
inline cx_float coot_to_cx_float_uchar(const         uchar x) { return (cx_float)((float) x, (float) 0);     }
inline cx_float coot_to_cx_float_ushort(const       ushort x) { return (cx_float)((float) x, (float) 0);     }
inline cx_float coot_to_cx_float_uint(const           uint x) { return (cx_float)((float) x, (float) 0);     }
inline cx_float coot_to_cx_float_ulong(const         ulong x) { return (cx_float)((float) x, (float) 0);     }
inline cx_float coot_to_cx_float_char(const           char x) { return (cx_float)((float) x, (float) 0);     }
inline cx_float coot_to_cx_float_short(const         short x) { return (cx_float)((float) x, (float) 0);     }
inline cx_float coot_to_cx_float_int(const             int x) { return (cx_float)((float) x, (float) 0);     }
inline cx_float coot_to_cx_float_long(const           long x) { return (cx_float)((float) x, (float) 0);     }
#ifdef COOT_HAVE_FP16
inline cx_float coot_to_cx_float_half(const           half x) { return (cx_float)((float) x, (float) 0);     }
#endif
inline cx_float coot_to_cx_float_float(const         float x) { return (cx_float)(x, (float) 0);             }
inline cx_float coot_to_cx_float_cx_float(const   cx_float x) { return x;                                    }
#ifdef COOT_HAVE_FP64
inline cx_float coot_to_cx_float_double(const       double x) { return (cx_float)((float) x, (float) 0);     }
inline cx_float coot_to_cx_float_cx_double(const cx_double x) { return (cx_float)((float) x.x, (float) x.y); }
#endif

inline cx_float coot_absdiff_cx_float(const cx_float x, const cx_float y) { return (cx_float)( fabs(x.x - y.x), fabs(x.y - y.y) ); }
// semi-hack: use squared norm instead of norm for magnitude check
inline cx_float coot_min_cx_float(const cx_float x, const cx_float y) { return ((x.x * x.x + x.y * x.y) < (y.x * y.x + y.y * y.y)) ? x : y; }
inline cx_float coot_max_cx_float(const cx_float x, const cx_float y) { return ((x.x * x.x + x.y * x.y) > (y.x * y.x + y.y * y.y)) ? x : y; }
inline cx_float coot_conj_cx_float(const cx_float x) { return (cx_float)(x.x, -x.y); }
inline cx_float coot_abs_cx_float(const cx_float x) { return (cx_float)(fabs(x.x), fabs(x.y)); }

// Basic mathematical operators.
inline cx_float coot_plus_cx_float(const cx_float x, const cx_float y)  { return (cx_float)(x.x + y.x, x.y + y.y); }
inline cx_float coot_minus_cx_float(const cx_float x, const cx_float y) { return (cx_float)(x.x - y.x, x.y - y.y); }
inline cx_float coot_times_cx_float(const cx_float x, const cx_float y) { return (cx_float)(x.x * y.x - x.y * y.y, x.x * y.y + x.y * y.x); }
inline cx_float coot_div_cx_float(const cx_float x, const cx_float y)   { return (cx_float)(cx_float)((x.x * y.x + x.y * y.y) / (y.x * y.x + y.y * y.y), (x.y * y.x - x.x * y.y) / (y.x * y.x + y.y * y.y)); }
inline cx_float coot_neg_cx_float(const cx_float x)                     { return (cx_float)(-x.x, -x.y); }

)"
