// Copyright 2026 Ryan Curtin (http://www.ratml.org)
// Copyright 2026 Marcus Edel (http://www.kurg.org)
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



struct sum_init_op_str
  {
  static inline constexpr size_t len() { return 7; }
  static inline constexpr char_array<8> str() { return char_array<8>{ "acc=val" }; }
  };

struct sum_inner_op_str
  {
  static inline constexpr size_t len() { return 8; }
  static inline constexpr char_array<9> str() { return char_array<9>{ "acc+=val" }; }
  };

struct sum_final_op_str
  {
  static inline constexpr size_t len() { return 3; }
  static inline constexpr char_array<4> str() { return char_array<4>{ "acc" }; }
  };



struct mean_final_op_str
  {
  static inline constexpr size_t len() { return 18; }
  static inline constexpr char_array<19> str() { return char_array<19>{ "acc/COOT_TO_ET0(n)" }; }
  };



struct max_init_op_str
  {
  static inline constexpr size_t len() { return 7; }
  static inline constexpr char_array<8> str() { return char_array<8>{ "acc=val" }; }
  };

struct max_inner_op_str
  {
  static inline constexpr size_t len() { return 21; }
  static inline constexpr char_array<22> str() { return char_array<22>{ "acc=(acc>val)?acc:val" }; }
  };

struct max_final_op_str
  {
  static inline constexpr size_t len() { return 3; }
  static inline constexpr char_array<4> str() { return char_array<4>{ "acc" }; }
  };



struct min_inner_op_str
  {
  static inline constexpr size_t len() { return 21; }
  static inline constexpr char_array<22> str() { return char_array<22>{ "acc=(acc<val)?acc:val" }; }
  };



struct init_op_macro_name
  {
  static inline constexpr size_t len() { return 22; }
  static inline constexpr char_array<23> str() { return char_array<23>{ "COOT_INIT_OP(acc,val)=" }; }
  };

struct inner_op_macro_name
  {
  static inline constexpr size_t len() { return 23; }
  static inline constexpr char_array<24> str() { return char_array<24>{ "COOT_INNER_OP(acc,val)=" }; }
  };

struct final_op_macro_name
  {
  static inline constexpr size_t len() { return 21; }
  static inline constexpr char_array<22> str() { return char_array<22>{ "COOT_FINAL_OP(acc,n)=" }; }
  };



template<kernel_id::enum_id num, coot_backend_t backend>
struct reduction_op_init_macro;

template<kernel_id::enum_id num, coot_backend_t backend>
struct reduction_op_inner_macro;

template<kernel_id::enum_id num, coot_backend_t backend>
struct reduction_op_final_macro;



template<coot_backend_t backend>
struct reduction_op_init_macro< kernel_id::sum_colwise, backend > : public concat_str< typename macro_defn<backend>::prefix, init_op_macro_name, sum_init_op_str, typename macro_defn<backend>::suffix > { };
template<coot_backend_t backend>
struct reduction_op_inner_macro< kernel_id::sum_colwise, backend > : public concat_str< typename macro_defn<backend>::prefix, inner_op_macro_name, sum_inner_op_str, typename macro_defn<backend>::suffix > { };
template<coot_backend_t backend>
struct reduction_op_final_macro< kernel_id::sum_colwise, backend > : public concat_str< typename macro_defn<backend>::prefix, final_op_macro_name, sum_final_op_str, typename macro_defn<backend>::suffix > { };



template<coot_backend_t backend>
struct reduction_op_init_macro< kernel_id::sum_rowwise, backend > : public concat_str< typename macro_defn<backend>::prefix, init_op_macro_name, sum_init_op_str, typename macro_defn<backend>::suffix > { };
template<coot_backend_t backend>
struct reduction_op_inner_macro< kernel_id::sum_rowwise, backend > : public concat_str< typename macro_defn<backend>::prefix, inner_op_macro_name, sum_inner_op_str, typename macro_defn<backend>::suffix > { };
template<coot_backend_t backend>
struct reduction_op_final_macro< kernel_id::sum_rowwise, backend > : public concat_str< typename macro_defn<backend>::prefix, final_op_macro_name, sum_final_op_str, typename macro_defn<backend>::suffix > { };



template<coot_backend_t backend>
struct reduction_op_init_macro< kernel_id::mean_colwise, backend > : public concat_str< typename macro_defn<backend>::prefix, init_op_macro_name, sum_init_op_str, typename macro_defn<backend>::suffix > { };
template<coot_backend_t backend>
struct reduction_op_inner_macro< kernel_id::mean_colwise, backend > : public concat_str< typename macro_defn<backend>::prefix, inner_op_macro_name, sum_inner_op_str, typename macro_defn<backend>::suffix > { };
template<coot_backend_t backend>
struct reduction_op_final_macro< kernel_id::mean_colwise, backend > : public concat_str< typename macro_defn<backend>::prefix, final_op_macro_name, mean_final_op_str, typename macro_defn<backend>::suffix > { };



template<coot_backend_t backend>
struct reduction_op_init_macro< kernel_id::mean_rowwise, backend > : public concat_str< typename macro_defn<backend>::prefix, init_op_macro_name, sum_init_op_str, typename macro_defn<backend>::suffix > { };
template<coot_backend_t backend>
struct reduction_op_inner_macro< kernel_id::mean_rowwise, backend > : public concat_str< typename macro_defn<backend>::prefix, inner_op_macro_name, sum_inner_op_str, typename macro_defn<backend>::suffix > { };
template<coot_backend_t backend>
struct reduction_op_final_macro< kernel_id::mean_rowwise, backend > : public concat_str< typename macro_defn<backend>::prefix, final_op_macro_name, mean_final_op_str, typename macro_defn<backend>::suffix > { };



template<coot_backend_t backend>
struct reduction_op_init_macro< kernel_id::max_colwise, backend > : public concat_str< typename macro_defn<backend>::prefix, init_op_macro_name, max_init_op_str, typename macro_defn<backend>::suffix > { };
template<coot_backend_t backend>
struct reduction_op_inner_macro< kernel_id::max_colwise, backend > : public concat_str< typename macro_defn<backend>::prefix, inner_op_macro_name, max_inner_op_str, typename macro_defn<backend>::suffix > { };
template<coot_backend_t backend>
struct reduction_op_final_macro< kernel_id::max_colwise, backend > : public concat_str< typename macro_defn<backend>::prefix, final_op_macro_name, max_final_op_str, typename macro_defn<backend>::suffix > { };



template<coot_backend_t backend>
struct reduction_op_init_macro< kernel_id::max_rowwise, backend > : public concat_str< typename macro_defn<backend>::prefix, init_op_macro_name, max_init_op_str, typename macro_defn<backend>::suffix > { };
template<coot_backend_t backend>
struct reduction_op_inner_macro< kernel_id::max_rowwise, backend > : public concat_str< typename macro_defn<backend>::prefix, inner_op_macro_name, max_inner_op_str, typename macro_defn<backend>::suffix > { };
template<coot_backend_t backend>
struct reduction_op_final_macro< kernel_id::max_rowwise, backend > : public concat_str< typename macro_defn<backend>::prefix, final_op_macro_name, max_final_op_str, typename macro_defn<backend>::suffix > { };



template<coot_backend_t backend>
struct reduction_op_init_macro< kernel_id::min_colwise, backend > : public concat_str< typename macro_defn<backend>::prefix, init_op_macro_name, max_init_op_str, typename macro_defn<backend>::suffix > { };
template<coot_backend_t backend>
struct reduction_op_inner_macro< kernel_id::min_colwise, backend > : public concat_str< typename macro_defn<backend>::prefix, inner_op_macro_name, min_inner_op_str, typename macro_defn<backend>::suffix > { };
template<coot_backend_t backend>
struct reduction_op_final_macro< kernel_id::min_colwise, backend > : public concat_str< typename macro_defn<backend>::prefix, final_op_macro_name, max_final_op_str, typename macro_defn<backend>::suffix > { };



template<coot_backend_t backend>
struct reduction_op_init_macro< kernel_id::min_rowwise, backend > : public concat_str< typename macro_defn<backend>::prefix, init_op_macro_name, max_init_op_str, typename macro_defn<backend>::suffix > { };
template<coot_backend_t backend>
struct reduction_op_inner_macro< kernel_id::min_rowwise, backend > : public concat_str< typename macro_defn<backend>::prefix, inner_op_macro_name, min_inner_op_str, typename macro_defn<backend>::suffix > { };
template<coot_backend_t backend>
struct reduction_op_final_macro< kernel_id::min_rowwise, backend > : public concat_str< typename macro_defn<backend>::prefix, final_op_macro_name, max_final_op_str, typename macro_defn<backend>::suffix > { };



template<kernel_id::enum_id num> struct is_reduction_kernel { static constexpr bool value = false; };
template<> struct is_reduction_kernel<kernel_id::sum_colwise> { static constexpr bool value = true; };
template<> struct is_reduction_kernel<kernel_id::sum_rowwise> { static constexpr bool value = true; };
template<> struct is_reduction_kernel<kernel_id::mean_colwise> { static constexpr bool value = true; };
template<> struct is_reduction_kernel<kernel_id::mean_rowwise> { static constexpr bool value = true; };
template<> struct is_reduction_kernel<kernel_id::max_colwise> { static constexpr bool value = true; };
template<> struct is_reduction_kernel<kernel_id::max_rowwise> { static constexpr bool value = true; };
template<> struct is_reduction_kernel<kernel_id::min_colwise> { static constexpr bool value = true; };
template<> struct is_reduction_kernel<kernel_id::min_rowwise> { static constexpr bool value = true; };
