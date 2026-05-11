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

// Conversion functions for uint64_t elements.
uint64_t coot_to_uint64_t(const      bool x) { return uint64_t(x); }
// uint64_t coot_to_uint64_t(const     uchar x) { return uint64_t(x); }
// uint64_t coot_to_uint64_t(const      char x) { return uint64_t(x); }
// uint64_t coot_to_uint64_t(const    ushort x) { return uint64_t(x); }
// uint64_t coot_to_uint64_t(const     short x) { return uint64_t(x); }
uint64_t coot_to_uint64_t(const      uint x) { return uint64_t(x); }
uint64_t coot_to_uint64_t(const       int x) { return uint64_t(x); }
uint64_t coot_to_uint64_t(const  uint64_t x) { return x;           }
// uint64_t coot_to_uint64_t(const      long x) { return uint64_t(x); }
// #if defined(COOT_HAVE_FP16)
// uint64_t coot_to_uint64_t(const    __half x) { return uint64_t(x); }
// #endif
uint64_t coot_to_uint64_t(const     float x) { return uint64_t(x); }
uint64_t coot_to_uint64_t(const    double x) { return uint64_t(x); }
// uint64_t coot_to_uint64_t(const  cx_float x) { return x.x;           }
// uint64_t coot_to_uint64_t(const cx_double x) { return uint64_t(x.x); }

)"
