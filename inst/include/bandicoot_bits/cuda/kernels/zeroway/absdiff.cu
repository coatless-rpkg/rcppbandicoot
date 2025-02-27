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



inline __device__ uint u32_absdiff(const uint x, const uint y)
  {
  if (x > y) { return (x - y); }
  else       { return (y - x); }
  }



inline __device__ int s32_absdiff(const int x, const int y)
  {
  return abs(x - y);
  }



inline __device__ size_t u64_absdiff(const size_t x, const size_t y)
  {
  if (x > y) { return (x - y); }
  else       { return (y - x); }
  }



inline __device__ long s64_absdiff(const long x, const long y)
  {
  return abs(x - y);
  }



inline __device__ float f_absdiff(const float x, const float y)
  {
  return fabs(x - y);
  }



inline __device__ double d_absdiff(const double x, const double y)
  {
  return fabs(x - y);
  }

