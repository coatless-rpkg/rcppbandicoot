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
// template metaprogramming utilities to get a unique list of types in an expression
//



//
// concatenate types
//
template<typename T1, typename... Ts>
struct concat_types { };

template<typename T>
struct concat_types< T >
  {
  typedef T result;
  };

template<typename T1, typename... T2>
struct concat_types< T1, std::tuple<T2...> >
  {
  typedef typename std::tuple<T1, T2...> result;
  };

template<typename... T1, typename T2>
struct concat_types< std::tuple<T1...>, T2 >
  {
  typedef typename std::tuple<T1..., T2> result;
  };

template<typename... T1, typename... T2>
struct concat_types< std::tuple<T1...>, std::tuple<T2...> >
  {
  typedef typename std::tuple<T1..., T2...> result;
  };

template<typename T1, typename T2, typename... Ts>
struct concat_types< T1, T2, Ts... >
  {
  typedef typename concat_types< typename concat_types<T1, T2>::result, Ts... >::result result;
  };



//
// remove a type from a list
//

template<typename X, typename T>
struct strip_type { };

template<typename X, typename T, typename... Ts>
struct strip_type<X, std::tuple<T, Ts...> >
  {
  typedef typename concat_types<T, typename strip_type<X, std::tuple<Ts...> >::result>::result result;
  };

template<typename T, typename... Ts>
struct strip_type<T, std::tuple<T, Ts...> >
  {
  typedef typename strip_type<T, std::tuple<Ts...> >::result result;
  };

template<typename X, typename T>
struct strip_type<X, std::tuple<T> >
  {
  typedef typename std::tuple<T> result;
  };

template<typename T>
struct strip_type<T, std::tuple<T> >
  {
  typedef typename std::tuple<> result;
  };

template<typename T>
struct strip_type<T, std::tuple<> >
  {
  typedef typename std::tuple<> result;
  };


//
// remove duplicates from a list of types
//

template<typename T>
struct remove_duplicates { };

template<typename T, typename... Ts>
struct remove_duplicates< std::tuple<T, Ts...> >
  {
  typedef typename concat_types<T, typename strip_type<T, typename remove_duplicates< std::tuple<Ts...> >::result>::result>::result result;
  };

template<>
struct remove_duplicates< std::tuple<> >
  {
  typedef typename std::tuple<> result;
  };


//
// this shim will add the source type for a complex type
//

template<typename T>
struct expand_cx_type { typedef std::tuple<T> result; };

template<typename T>
struct expand_cx_type<std::complex<T>> { typedef std::tuple<T, std::complex<T>> result; };



//
// now the actual implementations of elem_types for each object
//

template<typename T>
struct elem_types_inner { };

template<typename eT>
struct elem_types_inner< Mat<eT> >
  {
  typedef typename expand_cx_type<eT>::result result;
  };

template<typename eT>
struct elem_types_inner< Row<eT> >
  {
  typedef typename expand_cx_type<eT>::result result;
  };

template<typename eT>
struct elem_types_inner< Col<eT> >
  {
  typedef typename expand_cx_type<eT>::result result;
  };

template<typename eT>
struct elem_types_inner< subview<eT> >
  {
  typedef typename expand_cx_type<eT>::result result;
  };

template<typename eT>
struct elem_types_inner< subview_row<eT> >
  {
  typedef typename expand_cx_type<eT>::result result;
  };

template<typename eT>
struct elem_types_inner< subview_col<eT> >
  {
  typedef typename expand_cx_type<eT>::result result;
  };

template<typename eT>
struct elem_types_inner< diagview<eT> >
  {
  typedef typename expand_cx_type<eT>::result result;
  };

template<typename eT>
struct elem_types_inner< Cube<eT> >
  {
  typedef typename expand_cx_type<eT>::result result;
  };

template<typename eT>
struct elem_types_inner< subview_cube<eT> >
  {
  typedef typename expand_cx_type<eT>::result result;
  };

template<typename eT, typename T1>
struct elem_types_inner< subview_elem1< eT, T1 > >
  {
  typedef typename concat_types< typename expand_cx_type<eT>::result, typename elem_types_inner< T1 >::result >::result result;
  };

template<typename eT, typename T1, typename T2>
struct elem_types_inner< subview_elem2< eT, subview_elem2_both<eT, T1, T2> > >
  {
  typedef typename concat_types< typename expand_cx_type<eT>::result,
      typename concat_types<
          typename elem_types_inner< T1 >::result,
          typename elem_types_inner< T2 >::result
      >::result >::result result;
  };

template<typename eT, typename T1>
struct elem_types_inner< subview_elem2< eT, subview_elem2_all_cols<eT, T1> > >
  {
  typedef typename concat_types< typename expand_cx_type<eT>::result, typename elem_types_inner< T1 >::result >::result result;
  };

template<typename eT, typename T2>
struct elem_types_inner< subview_elem2< eT, subview_elem2_all_rows<eT, T2> > >
  {
  typedef typename concat_types< typename expand_cx_type<eT>::result, typename elem_types_inner< T2 >::result >::result result;
  };

template<typename T1, size_t src_dims, bool proxy_uses_ref>
struct elem_types_inner< ProxyColCast<T1, src_dims, proxy_uses_ref> >
  {
  typedef typename elem_types_inner<T1>::result result;
  };

template<typename T1, size_t src_dims, bool proxy_uses_ref>
struct elem_types_inner< ProxyMatCast<T1, src_dims, proxy_uses_ref> >
  {
  typedef typename elem_types_inner<T1>::result result;
  };

template<typename T1, size_t src_dims, bool proxy_uses_ref>
struct elem_types_inner< ProxyCubeCast<T1, src_dims, proxy_uses_ref> >
  {
  typedef typename elem_types_inner<T1>::result result;
  };

template<typename T1, typename eop_type>
struct elem_types_inner< eOp<T1, eop_type> >
  {
  typedef typename concat_types< typename elem_types_inner<T1>::result, typename eop_type::template extra_kernel_types<T1> >::result result;
  };

template<typename T1, typename eop_type>
struct elem_types_inner< eOpCube<T1, eop_type> >
  {
  typedef typename concat_types< typename elem_types_inner<T1>::result, typename eop_type::template extra_kernel_types<T1> >::result result;
  };

template<typename T1, typename T2, typename eglue_type>
struct elem_types_inner< eGlue<T1, T2, eglue_type> >
  {
  typedef typename concat_types< typename elem_types_inner<T1>::result, typename elem_types_inner<T2>::result >::result result;
  };

template<typename T1, typename T2, typename eglue_type>
struct elem_types_inner< eGlueCube<T1, T2, eglue_type> >
  {
  typedef typename concat_types< typename elem_types_inner<T1>::result, typename elem_types_inner<T2>::result >::result result;
  };

template<typename out_eT, typename T1, typename mtop_type>
struct elem_types_inner< mtOp<out_eT, T1, mtop_type> >
  {
  typedef typename concat_types< out_eT, typename elem_types_inner<T1>::result >::result result;
  };

template<typename out_eT, typename T1, typename mtop_type>
struct elem_types_inner< mtOpCube<out_eT, T1, mtop_type> >
  {
  typedef typename concat_types< out_eT, typename elem_types_inner<T1>::result >::result result;
  };

template<typename out_eT, typename T1, typename T2, typename mtglue_type>
struct elem_types_inner< mtGlue<out_eT, T1, T2, mtglue_mixed_core<mtglue_type> > >
  {
  typedef typename concat_types< out_eT, typename elem_types_inner<T1>::result, typename elem_types_inner<T2>::result >::result result;
  };

template<typename out_eT, typename T1, typename T2, typename mtglue_type>
struct elem_types_inner< mtGlueCube<out_eT, T1, T2, mtglue_mixed_core<mtglue_type> > >
  {
  typedef typename concat_types< out_eT, typename elem_types_inner<T1>::result, typename elem_types_inner<T2>::result >::result result;
  };

template<typename out_eT, typename T1, typename T2, typename mtglue_type>
struct elem_types_inner< mtGlue<out_eT, T1, T2, mtglue_rel_core<mtglue_type> > >
  {
  typedef typename concat_types< out_eT, typename elem_types_inner<T1>::result, typename elem_types_inner<T2>::result >::result result;
  };

template<typename out_eT, typename T1, typename T2, typename mtglue_type>
struct elem_types_inner< mtGlueCube<out_eT, T1, T2, mtglue_rel_core<mtglue_type> > >
  {
  typedef typename concat_types< out_eT, typename elem_types_inner<T1>::result, typename elem_types_inner<T2>::result >::result result;
  };

template<typename T1>
struct elem_types_inner< Op<T1, op_htrans> >
  {
  typedef typename elem_types_inner<T1>::result result;
  };



//
// `elem_types::result` is a `std::tuple<>` containing a unique list of types necessary to represent a `T`;
// so e.g. if `T` is `Mat<eT>` then this is just `std::tuple<eT>`; if `T` is `subview_elem1<eT, Col<u32>>`
// then this is `std::tuple<eT, u32>`.
//

template<typename... Ts>
struct elem_types
  {
  typedef typename remove_duplicates< typename concat_types< typename elem_types_inner< Ts >::result... >::result >::result result;
  };
