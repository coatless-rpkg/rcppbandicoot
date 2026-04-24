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
// full_name<T> provides a compile-time string definition of the full name of
// a kernel, including its prefix for its argument type `T`.
//
// The `kernel_name` specialization for the given kernel `num` must be
// specified in `kernels.hpp`.
//

template<typename T>
using full_name_prefix = concat_str< prefix<T>, underscore >;



template<kernel_id::enum_id num, typename... Ts>
using full_name = concat_str< full_name_prefix<Ts>..., logical_name<num> >;
