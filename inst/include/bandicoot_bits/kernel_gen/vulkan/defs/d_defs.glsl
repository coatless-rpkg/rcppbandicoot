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
bool coot_is_fp(const double x)                     { return true; }
bool coot_is_signed(const double x)                 { return true; }
bool coot_isnan(const double x)                     { return isnan(x); }
bool coot_isinf(const double x)                     { return isinf(x); }
bool coot_isfinite(const double x)                  { return !isnan(x) && !isinf(x); }

double coot_absdiff(const double x, const double y) { return abs(x - y); }
double coot_min(const double x, const double y)     { return min(x, y); }
double coot_max(const double x, const double y)     { return max(x, y); }
double coot_conj(const double x)                    { return x; }
double coot_abs(const double x)                     { return abs(x); }

double coot_plus(const double x, const double y)    { return x + y; }
double coot_minus(const double x, const double y)   { return x - y; }
double coot_times(const double x, const double y)   { return x * y; }
double coot_div(const double x, const double y)     { return x / y; }
)"
