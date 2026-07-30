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


// Utility struct to represent a floating-point GPU type of the maximum precision
// allowed by that GPU.
//
// This is important for OpenCL, where the device may only support 32-bit floats.
//
// This type is never used locally but only inside OpenCL kernels that need to up-cast to floating
// point to perform an operation.  Therefore, the type holds nothing and is just a placeholder.

struct floatmax { };

template<typename T>
struct safe_type
  {
  typedef T result;
  };

template<>
struct safe_type<double>
  {
  typedef floatmax result;
  };
