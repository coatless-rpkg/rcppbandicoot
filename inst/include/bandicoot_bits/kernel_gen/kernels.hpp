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
// The `kernel_name` struct defines the name of all supported kernels.
// It must be specialized for each supported kernel, and each specialization
// must have the function:
//
// `static inline constexpr const auto& str() { return "name"; }`
//

template<kernel_id::enum_id num>
struct kernel_name { };

template<> struct kernel_name< kernel_id::fill         > { static inline constexpr const auto& str() { return "fill";         } };
template<> struct kernel_name< kernel_id::copy         > { static inline constexpr const auto& str() { return "copy";         } };
template<> struct kernel_name< kernel_id::sum_colwise  > { static inline constexpr const auto& str() { return "sum_colwise";  } };
template<> struct kernel_name< kernel_id::sum_rowwise  > { static inline constexpr const auto& str() { return "sum_rowwise";  } };
template<> struct kernel_name< kernel_id::mean_colwise > { static inline constexpr const auto& str() { return "mean_colwise"; } };
template<> struct kernel_name< kernel_id::mean_rowwise > { static inline constexpr const auto& str() { return "mean_rowwise"; } };
template<> struct kernel_name< kernel_id::max_colwise  > { static inline constexpr const auto& str() { return "max_colwise";  } };
template<> struct kernel_name< kernel_id::max_rowwise  > { static inline constexpr const auto& str() { return "max_rowwise";  } };
template<> struct kernel_name< kernel_id::min_colwise  > { static inline constexpr const auto& str() { return "min_colwise";  } };
template<> struct kernel_name< kernel_id::min_rowwise  > { static inline constexpr const auto& str() { return "min_rowwise";  } };
