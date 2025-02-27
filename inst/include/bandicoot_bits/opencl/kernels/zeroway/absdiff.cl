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



// Utility functions for safe abs(x - y) with different types.



inline uint coot_absdiff_uint(const uint x, const uint y)
  {
  if (x > y) { return (x - y); }
  else       { return (y - x); }
  }



inline int coot_absdiff_int(const int x, const int y)
  {
  return abs(x - y);
  }



inline ulong coot_absdiff_ulong(const ulong x, const ulong y)
  {
  if (x > y) { return (x - y); }
  else       { return (y - x); }
  }



inline long coot_absdiff_long(const long x, const long y)
  {
  return abs(x - y);
  }



inline float coot_absdiff_float(const float x, const float y)
  {
  return fabs(x - y);
  }



#ifdef COOT_HAS_FP64
inline double coot_absdiff_double(const double x, const double y)
  {
  return fabs(x - y);
  }
#endif
