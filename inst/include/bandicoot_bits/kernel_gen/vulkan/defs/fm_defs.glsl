// Copyright 2026 Marcus Edel (http://www.kurg.org/)
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

#if (defined(COOT_FLOATMAX_IS_DOUBLE) && !defined(COOT_HAS_D_DEFS)) \
    (defined(COOT_FLOATMAX_IS_FLOAT)  && !defined(COOT_HAS_F_DEFS))

#ifdef COOT_FLOATMAX_IS_DOUBLE
#define COOT_HAS_D_DEFS
#else
#define COOT_HAS_F_DEFS
#endif

bool coot_is_fp(const floatmax x)                   { return true; }
bool coot_is_signed(const floatmax x)               { return true; }
bool coot_isnan(const floatmax x)                   { return isnan(x); }
bool coot_isinf(const floatmax x)                   { return isinf(x); }
bool coot_isfinite(const floatmax x)                { return !isnan(x) && !isinf(x); }

// Conversion functions for floatmax elements.
floatmax coot_to_floatmax(const      bool x) { return floatmax(x); }
// floatmax coot_to_floatmax(const     uchar x) { return floatmax(x); }
// floatmax coot_to_floatmax(const      char x) { return floatmax(x); }
// floatmax coot_to_floatmax(const    ushort x) { return floatmax(x); }
// floatmax coot_to_floatmax(const     short x) { return floatmax(x); }
floatmax coot_to_floatmax(const      uint x) { return floatmax(x); }
floatmax coot_to_floatmax(const       int x) { return floatmax(x); }
floatmax coot_to_floatmax(const  uint64_t x) { return floatmax(x); }
// floatmax coot_to_floatmax(const      long x) { return floatmax(x); }
// #if defined(COOT_HAVE_FP16)
// floatmax coot_to_floatmax(const    __half x) { return floatmax(x); }
// #endif
floatmax coot_to_floatmax(const     float x) { return x;        }
floatmax coot_to_floatmax(const    double x) { return floatmax(x); }
// floatmax coot_to_floatmax(const  cx_float x) { return x.x;        }
// floatmax coot_to_floatmax(const cx_double x) { return float(x.x); }

floatmax coot_absdiff(const floatmax x, const floatmax y) { return abs(x - y); }
floatmax coot_min(const floatmax x, const floatmax y)     { return min(x, y); }
floatmax coot_max(const floatmax x, const floatmax y)     { return max(x, y); }
floatmax coot_conj(const floatmax x)                   { return x; }
floatmax coot_abs(const floatmax x)                    { return abs(x); }

floatmax coot_plus(const floatmax x, const floatmax y)    { return x + y; }
floatmax coot_minus(const floatmax x, const floatmax y)   { return x - y; }
floatmax coot_times(const floatmax x, const floatmax y)   { return x * y; }
floatmax coot_div(const floatmax x, const floatmax y)     { return x / y; }

#endif

)"
