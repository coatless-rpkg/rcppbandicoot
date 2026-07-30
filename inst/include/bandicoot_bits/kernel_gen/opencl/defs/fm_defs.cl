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
//
// "fm" is "f-max-precision": this resolves to float on OpenCL devices that only support FP32,
// and double on devices that support FP64.  The "floatmax" type should be used in kernels to use this type.
R"(

#ifdef COOT_HAVE_FP64
inline floatmax coot_type_min_floatmax(const floatmax x)    { return -DBL_MAX; }
inline floatmax coot_type_minpos_floatmax(const floatmax x) { return DBL_MIN; }
inline floatmax coot_type_max_floatmax(const floatmax x)    { return DBL_MAX; }
#else
inline floatmax coot_type_min_floatmax(const floatmax x)    { return -FLT_MAX; }
inline floatmax coot_type_minpos_floatmax(const floatmax x) { return FLT_MIN; }
inline floatmax coot_type_max_floatmax(const floatmax x)    { return FLT_MAX; }
#endif

inline bool coot_is_fp_floatmax()                 { return true; }
inline bool coot_is_signed_floatmax()             { return true; }
inline bool coot_isnan_floatmax(const floatmax x) { return isnan(x); }
inline bool coot_isinf_floatmax(const floatmax x) { return isinf(x); }

// Conversion operators.
inline floatmax coot_to_floatmax_uchar(const         uchar x) { return (floatmax) x;   }
inline floatmax coot_to_floatmax_ushort(const       ushort x) { return (floatmax) x;   }
inline floatmax coot_to_floatmax_uint(const           uint x) { return (floatmax) x;   }
inline floatmax coot_to_floatmax_ulong(const         ulong x) { return (floatmax) x;   }
inline floatmax coot_to_floatmax_char(const           char x) { return (floatmax) x;   }
inline floatmax coot_to_floatmax_short(const         short x) { return (floatmax) x;   }
inline floatmax coot_to_floatmax_int(const             int x) { return (floatmax) x;   }
inline floatmax coot_to_floatmax_long(const           long x) { return (floatmax) x;   }
#ifdef COOT_HAVE_FP16
inline floatmax coot_to_floatmax_half(const           half x) { return (floatmax) x;   }
#endif
inline floatmax coot_to_floatmax_float(const         float x) { return (floatmax) x;   }
inline floatmax coot_to_floatmax_cx_float(const   cx_float x) { return (floatmax) x.x; }
#ifdef COOT_HAVE_FP64
inline floatmax coot_to_floatmax_double(const       double x) { return (floatmax) x;   }
inline floatmax coot_to_floatmax_cx_double(const cx_double x) { return (floatmax) x.x; }
#endif
inline floatmax coot_to_floatmax_floatmax(const   floatmax x) { return (floatmax) x;   }

inline floatmax coot_absdiff_floatmax(const floatmax x, const floatmax y) { return fabs(x - y); }
inline floatmax coot_conj_floatmax(const floatmax x) { return x; }
inline floatmax coot_abs_floatmax(const floatmax x) { return fabs(x); }
inline floatmax coot_min_floatmax(const floatmax x, const floatmax y) { return fmin(x, y); }
inline floatmax coot_max_floatmax(const floatmax x, const floatmax y) { return fmax(x, y); }
#ifdef COOT_HAVE_FP64
inline floatmax coot_sign_floatmax(const floatmax x) { return (x > 0.0) ? 1.0 : ( (x < 0.0) ? -1.0 : ( (x == 0.0) ? 0.0 : x ) );       }
#else
inline floatmax coot_sign_floatmax(const floatmax x) { return (x > 0.0f) ? 1.0f : ( (x < 0.0f) ? -1.0f : ( (x == 0.0f) ? 0.0f : x ) ); }

#endif

// Basic mathematical operators.
inline floatmax coot_plus_floatmax(const floatmax x, const floatmax y)  { return x + y; }
inline floatmax coot_minus_floatmax(const floatmax x, const floatmax y) { return x - y; }
inline floatmax coot_times_floatmax(const floatmax x, const floatmax y) { return x * y; }
inline floatmax coot_div_floatmax(const floatmax x, const floatmax y)   { return x / y; }

)"
