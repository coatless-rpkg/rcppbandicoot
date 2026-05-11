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

// Conversion functions for uint elements.
int coot_to_int(const      bool x) { return int(x); }
// int coot_to_int(const     uchar x) { return int(x); }
// int coot_to_int(const      char x) { return int(x); }
// int coot_to_int(const    ushort x) { return int(x); }
// int coot_to_int(const     short x) { return int(x); }
int coot_to_int(const      uint x) { return int(x); }
int coot_to_int(const       int x) { return x;      }
int coot_to_int(const  uint64_t x) { return int(x); }
// int coot_to_int(const      long x) { return int(x); }
// #if defined(COOT_HAVE_FP16)
// int coot_to_int(const    __half x) { return int(x); }
// #endif
int coot_to_int(const     float x) { return int(x); }
int coot_to_int(const    double x) { return int(x); }
// int coot_to_int(const  cx_float x) { return x.x;       }
// int coot_to_int(const cx_double x) { return int(x.x); }

)"
