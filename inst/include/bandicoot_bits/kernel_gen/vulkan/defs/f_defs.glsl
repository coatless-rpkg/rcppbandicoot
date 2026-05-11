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
bool coot_is_fp(const float x)                   { return true; }
bool coot_is_signed(const float x)               { return true; }
bool coot_isnan(const float x)                   { return isnan(x); }
bool coot_isinf(const float x)                   { return isinf(x); }
bool coot_isfinite(const float x)                { return !isnan(x) && !isinf(x); }

// Conversion functions for float elements.
float coot_to_float(const      bool x) { return float(x); }
// float coot_to_float(const     uchar x) { return float(x); }
// float coot_to_float(const      char x) { return float(x); }
// float coot_to_float(const    ushort x) { return float(x); }
// float coot_to_float(const     short x) { return float(x); }
float coot_to_float(const      uint x) { return float(x); }
float coot_to_float(const       int x) { return float(x); }
float coot_to_float(const  uint64_t x) { return float(x); }
// float coot_to_float(const      long x) { return float(x); }
// #if defined(COOT_HAVE_FP16)
// float coot_to_float(const    __half x) { return float(x); }
// #endif
float coot_to_float(const     float x) { return x;        }
float coot_to_float(const    double x) { return float(x); }
// float coot_to_float(const  cx_float x) { return x.x;        }
// float coot_to_float(const cx_double x) { return float(x.x); }

float coot_absdiff(const float x, const float y) { return abs(x - y); }
float coot_min(const float x, const float y)     { return min(x, y); }
float coot_max(const float x, const float y)     { return max(x, y); }
float coot_conj(const float x)                   { return x; }
float coot_abs(const float x)                    { return abs(x); }

float coot_plus(const float x, const float y)    { return x + y; }
float coot_minus(const float x, const float y)   { return x - y; }
float coot_times(const float x, const float y)   { return x * y; }
float coot_div(const float x, const float y)     { return x / y; }
)"
