// Copyright 2025 Ryan Curtin (http://www.ratml.org)
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



//
// logical_name provides a compile-time string definition of the logical
// name of a kernel (e.g. no prefix indicating the type).
//
// The `kernel_name` specialization for the given kernel `num` must be
// specified in `kernels.hpp`.
//

template<kernel_id::enum_id num>
struct logical_name
  {
  static inline constexpr size_t len() { return arg_len_impl(kernel_name<num>::str()); }
  static inline constexpr char_array<len() + 1> str() { return to_array::array(kernel_name<num>::str()); }
  };
