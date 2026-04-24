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
// `prefix<T>` provides a compile-time string definition of the prefix for a
// kernel for a given type `T`.  Prefixes can be variable length but generally
// follow a simple formula.
//
// Any allowable argument to a kernel must have a specialization in this file.
//



//
// type_prefix converts a given type into a single-character prefix value
// (unsigned types are capitalized)
//

template<typename eT>
struct type_prefix { };

template<> struct type_prefix< u8 >        { static inline constexpr char prefix_char() { return 'B'; } }; // byte
template<> struct type_prefix< s8 >        { static inline constexpr char prefix_char() { return 'b'; } };
template<> struct type_prefix< u16 >       { static inline constexpr char prefix_char() { return 'S'; } }; // short
template<> struct type_prefix< s16 >       { static inline constexpr char prefix_char() { return 's'; } };
template<> struct type_prefix< u32 >       { static inline constexpr char prefix_char() { return 'I'; } }; // int
template<> struct type_prefix< s32 >       { static inline constexpr char prefix_char() { return 'i'; } };
template<> struct type_prefix< u64 >       { static inline constexpr char prefix_char() { return 'L'; } }; // long
template<> struct type_prefix< s64 >       { static inline constexpr char prefix_char() { return 'l'; } };
template<> struct type_prefix< fp16 >      { static inline constexpr char prefix_char() { return 'h'; } };
template<> struct type_prefix< float >     { static inline constexpr char prefix_char() { return 'f'; } };
template<> struct type_prefix< double >    { static inline constexpr char prefix_char() { return 'd'; } };
template<> struct type_prefix< cx_float >  { static inline constexpr char prefix_char() { return 'c'; } }; // complex single
template<> struct type_prefix< cx_double > { static inline constexpr char prefix_char() { return 'z'; } }; // complex double



template<typename T, typename... Ts>
struct prefix { };



// Mat<eT>: `mX` for eT -> `X`

template<typename eT>
struct prefix< Mat<eT> >
  {
  static inline constexpr size_t len() { return 2; }
  static inline constexpr char_array<len() + 1> str() { return { 'm', type_prefix< eT >::prefix_char(), '\0' }; }
  };



// Col<eT>: `cX` for eT -> `X`

template<typename eT>
struct prefix< Col<eT> >
  {
  static inline constexpr size_t len() { return 2; }
  static inline constexpr char_array<len() + 1> str() { return { 'c', type_prefix< eT >::prefix_char(), '\0' }; }
  };



// Row<eT>: `rX` for eT -> `X`

template<typename eT>
struct prefix< Row<eT> >
  {
  static inline constexpr size_t len() { return 2; }
  static inline constexpr char_array<len() + 1> str() { return { 'r', type_prefix< eT >::prefix_char(), '\0' }; }
  };



// subview<eT>: `sX` for eT -> `X`

template<typename eT>
struct prefix< subview<eT> >
  {
  static inline constexpr size_t len() { return 2; }
  static inline constexpr char_array<len() + 1> str() { return { 's', type_prefix<eT>::prefix_char(), '\0' }; }
  };



// subview_col<eT>: `scX` for eT -> `X`

template<typename eT>
struct prefix< subview_col<eT> >
  {
  static inline constexpr size_t len() { return 3; }
  static inline constexpr char_array<len() + 1> str() { return { 's', 'c', type_prefix<eT>::prefix_char(), '\0' }; }
  };



// subview_row<eT>: `srX` for eT -> `X`

template<typename eT>
struct prefix< subview_row<eT> >
  {
  static inline constexpr size_t len() { return 3; }
  static inline constexpr char_array<len() + 1> str() { return { 's', 'r', type_prefix<eT>::prefix_char(), '\0' }; }
  };



// diagview<eT> -> `dX` for eT -> `X`

template<typename eT>
struct prefix< diagview<eT> >
  {
  static inline constexpr size_t len() { return 2; }
  static inline constexpr char_array<len() + 1> str() { return { 'd', type_prefix<eT>::prefix_char(), '\0' }; }
  };



// Cube<eT> -> `CX` for eT -> `X`

template<typename eT>
struct prefix< Cube<eT> >
  {
  static inline constexpr size_t len() { return 2; }
  static inline constexpr char_array<len() + 1> str() { return { 'C', type_prefix<eT>::prefix_char(), '\0' }; }
  };



// subview_cube<eT> -> `SX` for eT -> `X`

template<typename eT>
struct prefix< subview_cube<eT> >
  {
  static inline constexpr size_t len() { return 2; }
  static inline constexpr char_array<len() + 1> str() { return { 'S', type_prefix<eT>::prefix_char(), '\0' }; }
  };



// subview_elem1<eT, T1> -> `eXY` for eT -> `X` and `T1` -> `Y`
// (so, e.g., subview_elem1<float, Mat<u32>> -> `efmI`)

template<typename eT>
struct sve1_prefix
  {
  static inline constexpr size_t len() { return 2; }
  static inline constexpr char_array<len() + 1> str() { return { 'e', type_prefix<eT>::prefix_char(), '\0' }; }
  };

template<typename eT, typename T1>
struct prefix< subview_elem1< eT, T1 > > : public concat_str
  <
  sve1_prefix<eT>,
  prefix<T1>
  > { };



// subview_elem2<eT, subview_elem2_both<eT, T1, T2>> -> `EXYZ` for eT -> `X` and `T1` -> `Y` and `T2` -> `Z`
//
template<typename eT>
struct sve2_both_prefix
  {
  static inline constexpr size_t len() { return 2; }
  static inline constexpr char_array<len() + 1> str() { return { 'E', type_prefix<eT>::prefix_char(), '\0' }; }
  };

template<typename eT, typename T1, typename T2>
struct prefix< subview_elem2< eT, subview_elem2_both<eT, T1, T2> > > : public concat_str
  <
  sve2_both_prefix<eT>,
  prefix<T1>,
  prefix<T2>
  > { };



// subview_elem2<eT, subview_elem2_all_cols<eT, T1>> -> `E1XYZ` for eT -> `X` and `T1` -> `Y` and `T2` -> `Z`

template<typename eT>
struct sve2_all_cols_prefix
  {
  static inline constexpr size_t len() { return 3; }
  static inline constexpr char_array<len() + 1> str() { return { 'E', '1', type_prefix<eT>::prefix_char(), '\0' }; }
  };

template<typename eT, typename T1>
struct prefix< subview_elem2< eT, subview_elem2_all_cols<eT, T1> > > : public concat_str
  <
  sve2_all_cols_prefix<eT>,
  prefix<T1>
  > { };



// subview_elem2<eT, subview_elem2_all_rows<eT, T2>> -> `E2XYZ` for eT -> `X` and `T1` -> `Y` and `T2` -> `Z`

template<typename eT>
struct sve2_all_rows_prefix
  {
  static inline constexpr size_t len() { return 3; }
  static inline constexpr char_array<len() + 1> str() { return { 'E', '2', type_prefix<eT>::prefix_char(), '\0' }; }
  };

template<typename eT, typename T2>
struct prefix< subview_elem2< eT, subview_elem2_all_rows<eT, T2> > > : public concat_str
  <
  sve2_all_rows_prefix<eT>,
  prefix<T2>
  > { };



// ProxyColCast<T1> -> "P1" then T1 prefix

struct proxy_col_cast_prefix { static inline constexpr auto& str() { return "P1"; } };

template<typename T1, size_t src_dims, bool proxy_uses_ref>
struct prefix< ProxyColCast<T1, src_dims, proxy_uses_ref> > : public concat_str
  <
  proxy_col_cast_prefix,
  prefix<T1>
  > { };



// ProxyMatCast<T1> -> "P2" then T1 prefix

struct proxy_mat_cast_prefix { static inline constexpr auto& str() { return "P2"; } };

template<typename T1, size_t src_dims, bool proxy_uses_ref>
struct prefix< ProxyMatCast<T1, src_dims, proxy_uses_ref> > : public concat_str
  <
  proxy_mat_cast_prefix,
  prefix<T1>
  > { };



// ProxyCubeCast<T1> -> "P3" then T1 prefix

struct proxy_cube_cast_prefix { static inline constexpr auto& str() { return "P3"; } };

template<typename T1, size_t src_dims, bool proxy_uses_ref>
struct prefix< ProxyCubeCast<T1, src_dims, proxy_uses_ref> > : public concat_str
  <
  proxy_cube_cast_prefix,
  prefix<T1>
  > { };



// eOp<T1, eop_type> -> "eo<X>_" then T1 prefix

struct eop_prefix { static inline constexpr auto& str() { return "eo"; } };

template<typename T1, typename eop_type>
struct prefix< eOp<T1, eop_type> > : public concat_str
  <
  eop_prefix,
  typename eop_type::prefix,
  underscore,
  prefix<T1>
  > { };

template<typename T1, typename eop_type>
struct prefix< eOpCube<T1, eop_type> > : public prefix< eOp<T1, eop_type> > { };



// eGlue<T1, T2, eglue_type> -> "eg<X>_" then T1 prefix and T2 prefix

struct eglue_prefix { static inline constexpr auto& str() { return "eg"; } };

template<typename T1, typename T2, typename eglue_type>
struct prefix< eGlue<T1, T2, eglue_type> > : public concat_str
  <
  eglue_prefix,
  typename eglue_type::prefix,
  underscore,
  prefix<T1>,
  underscore,
  prefix<T2>
  > { };

template<typename T1, typename T2, typename eglue_type>
struct prefix< eGlueCube<T1, T2, eglue_type> > : public prefix< eGlue<T1, T2, eglue_type> > { };



// mtOp<out_eT, T1, mtop_conv_to> -> "mtc<X>_" then T1 prefix

template<typename eT>
struct mtop_conv_to_prefix
  {
  static inline constexpr size_t len() { return 4; }
  static inline constexpr char_array<len() + 1> str() { return { 'm', 't', 'c', type_prefix<eT>::prefix_char(), '\0' }; }
  };

template<typename out_eT, typename T1>
struct prefix< mtOp<out_eT, T1, mtop_conv_to> > : public concat_str
  <
  mtop_conv_to_prefix< out_eT >,
  underscore,
  prefix<T1>
  > { };

template<typename out_eT, typename T1>
struct prefix< mtOpCube<out_eT, T1, mtop_conv_to> > : public prefix< mtOp<out_eT, T1, mtop_conv_to> > { };



// mtOp<out_eT, T1, mtop_rel_core<mtop_type> > -> "mtr<X><Y>_" then T1 prefix

template<typename eT>
struct mtop_rel_core_prefix
  {
  static inline constexpr size_t len() { return 4; }
  static inline constexpr char_array<len() + 1> str() { return { 'm', 't', 'r', type_prefix<eT>::prefix_char(), '\0' }; }
  };

template<typename out_eT, typename T1, typename mtop_type>
struct prefix< mtOp<out_eT, T1, mtop_rel_core<mtop_type> > > : public concat_str
  <
  mtop_rel_core_prefix< out_eT >,
  typename mtop_type::prefix,
  underscore,
  prefix<T1>
  > { };

template<typename out_eT, typename T1, typename mtop_type>
struct prefix< mtOpCube<out_eT, T1, mtop_rel_core<mtop_type> > > : public prefix< mtOp<out_eT, T1, mtop_rel_core<mtop_type> > > { };



// mtGlue<out_eT, T1, T2, mtglue_mixed_core> -> "mg<X><op>_" then T1 prefix and T2 prefix

template<typename eT>
struct mtglue_prefix
  {
  static inline constexpr size_t len() { return 3; }
  static inline constexpr char_array<len() + 1> str() { return { 'm', 'g', type_prefix<eT>::prefix_char(), '\0' }; }
  };

template<typename out_eT, typename T1, typename T2, typename mtglue_type>
struct prefix< mtGlue<out_eT, T1, T2, mtglue_mixed_core<mtglue_type> > > : public concat_str
  <
  mtglue_prefix< out_eT >,
  typename mtglue_type::prefix,
  underscore,
  prefix<T1>,
  underscore,
  prefix<T2>
  > { };

template<typename out_eT, typename T1, typename T2, typename mtglue_type>
struct prefix< mtGlueCube<out_eT, T1, T2, mtglue_mixed_core<mtglue_type> > > : public prefix< mtGlue<out_eT, T1, T2, mtglue_mixed_core<mtglue_type> > > { };

template<typename out_eT, typename T1, typename T2, typename mtglue_type>
struct prefix< mtGlue<out_eT, T1, T2, mtglue_rel_core<mtglue_type> > > : public prefix< mtGlue<out_eT, T1, T2, mtglue_mixed_core<mtglue_type> > > { };

template<typename out_eT, typename T1, typename T2, typename mtglue_type>
struct prefix< mtGlueCube<out_eT, T1, T2, mtglue_rel_core<mtglue_type> > > : public prefix< mtGlue<out_eT, T1, T2, mtglue_mixed_core<mtglue_type> > > { };



// Op<T1, op_htrans> -> "ht_" then T1 prefix

struct op_htrans_prefix { static inline constexpr auto& str() { return "ht_"; } };

template<typename T1>
struct prefix< Op<T1, op_htrans> > : public concat_str
  <
  op_htrans_prefix,
  prefix<T1>
  > { };
