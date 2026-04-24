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
