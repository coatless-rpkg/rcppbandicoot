// Copyright 2023 Ryan Curtin (http://www.ratml.org/)
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



// Utility functions for conj(x) for different types.

inline uint      conj_uint(const uint x)           { return x; }
inline int       conj_int(const int x)             { return x; }
inline ulong     conj_ulong(const ulong x)         { return x; }
inline long      conj_long(const long x)           { return x; }
inline float     conj_float(const float x)         { return x; }
//inline cx_float  conj_cx_float(const cx_float x)   { return cx_float(x.x, -x.y); }

#ifdef COOT_HAS_FP64
inline double    conj_double(const double x)       { return x; }
//inline cx_double conj_cx_double(const cx_double x) { return cx_double(x.x, -x.y); }
#endif
