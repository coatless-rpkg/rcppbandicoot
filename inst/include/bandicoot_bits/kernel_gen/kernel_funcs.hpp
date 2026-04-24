// Copyright 2026 Ryan Curtin (http://www.ratml.org)
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
// template metaprogramming utilities that construct any utility functions needed
// by a type that is passed to a kernel
//



template<typename T>
struct kernel_func_types_inner
  {
  typedef std::tuple<> result;
  };

// eOps may have utility functions
template<typename T1, typename eop_type>
struct kernel_func_types_inner< eOp<T1, eop_type> >
  {
  typedef typename concat_types<
      std::tuple< std::pair<typename T1::elem_type, eop_type> >,
      typename kernel_func_types_inner< T1 >::result
  >::result result;
  };

template<typename T1, typename eop_type>
struct kernel_func_types_inner< eOpCube<T1, eop_type> >
  {
  typedef typename concat_types<
      std::tuple< std::pair<typename T1::elem_type, eop_type> >,
      typename kernel_func_types_inner< T1 >::result
  >::result result;
  };

// eGlues may have utility functions
template<typename T1, typename T2, typename eglue_type>
struct kernel_func_types_inner< eGlue<T1, T2, eglue_type> >
  {
  typedef typename concat_types<
      std::tuple< std::pair<typename T1::elem_type, eglue_type> >,
      typename kernel_func_types_inner< T1 >::result,
      typename kernel_func_types_inner< T2 >::result
  >::result result;
  };

template<typename T1, typename T2, typename eglue_type>
struct kernel_func_types_inner< eGlueCube<T1, T2, eglue_type> >
  {
  typedef typename concat_types<
      std::tuple< std::pair<typename T1::elem_type, eglue_type> >,
      typename kernel_func_types_inner< T1 >::result,
      typename kernel_func_types_inner< T2 >::result
  >::result result;
  };

template<typename eT, typename T1>
struct kernel_func_types_inner< mtOp<eT, T1, mtop_conv_to> >
  {
  typedef typename kernel_func_types_inner<T1>::result result;
  };

template<typename eT, typename T1 >
struct kernel_func_types_inner< mtOpCube<eT, T1, mtop_conv_to> >
  {
  typedef typename kernel_func_types_inner<T1>::result result;
  };

template<typename eT, typename T1, typename mtop_type>
struct kernel_func_types_inner< mtOp<eT, T1, mtop_rel_core<mtop_type>> >
  {
  typedef typename concat_types<
      std::tuple< std::pair< std::pair<eT, typename T1::elem_type>, typename mtop_type::equiv_mtglue > >,
      typename kernel_func_types_inner< T1 >::result
  >::result result;
  };

template<typename eT, typename T1, typename mtop_type>
struct kernel_func_types_inner< mtOpCube<eT, T1, mtop_rel_core<mtop_type>> >
  {
  typedef typename concat_types<
      std::tuple< std::pair< std::pair<eT, typename T1::elem_type>, typename mtop_type::equiv_mtglue > >,
      typename kernel_func_types_inner< T1 >::result
  >::result result;
  };

template<typename eT, typename T1, typename T2, typename mtglue_type>
struct kernel_func_types_inner< mtGlue<eT, T1, T2, mtglue_mixed_core<mtglue_type>> >
  {
  typedef typename concat_types< typename kernel_func_types_inner<T1>::result, typename kernel_func_types_inner<T2>::result>::result result;
  };

template<typename eT, typename T1, typename T2, typename mtglue_type>
struct kernel_func_types_inner< mtGlueCube<eT, T1, T2, mtglue_mixed_core<mtglue_type>> >
  {
  typedef typename concat_types< typename kernel_func_types_inner<T1>::result, typename kernel_func_types_inner<T2>::result>::result result;
  };

template<typename eT, typename T1, typename T2, typename mtglue_type>
struct kernel_func_types_inner< mtGlue<eT, T1, T2, mtglue_rel_core<mtglue_type>> >
  {
  typedef typename concat_types<
      std::tuple< std::pair< std::pair<eT, typename T1::elem_type>, mtglue_type> >,
      typename kernel_func_types_inner<T1>::result,
      typename kernel_func_types_inner<T2>::result
  >::result result;
  };

template<typename eT, typename T1, typename T2, typename mtglue_type>
struct kernel_func_types_inner< mtGlueCube<eT, T1, T2, mtglue_rel_core<mtglue_type>> >
  {
  typedef typename concat_types<
      std::tuple< std::pair< std::pair<eT, typename T1::elem_type>, mtglue_type> >,
      typename kernel_func_types_inner<T1>::result,
      typename kernel_func_types_inner<T2>::result
  >::result result;
  };

// for other types, we must recurse into the operation
template<typename eT, typename T1>
struct kernel_func_types_inner< subview_elem1<eT, T1> >
  {
  typedef typename kernel_func_types_inner<T1>::result result;
  };

template<typename eT, typename T1, typename T2>
struct kernel_func_types_inner< subview_elem2<eT, subview_elem2_both<eT, T1, T2>> >
  {
  typedef typename concat_types<
      typename kernel_func_types_inner<T1>::result,
      typename kernel_func_types_inner<T2>::result
  >::result result;
  };

template<typename eT, typename T1>
struct kernel_func_types_inner< subview_elem2<eT, subview_elem2_all_cols<eT, T1>> >
  {
  typedef typename kernel_func_types_inner<T1>::result result;
  };

template<typename eT, typename T2>
struct kernel_func_types_inner< subview_elem2<eT, subview_elem2_all_rows<eT, T2>> >
  {
  typedef typename kernel_func_types_inner<T2>::result result;
  };

template<typename T1, size_t src_dims, bool proxy_uses_ref>
struct kernel_func_types_inner< ProxyColCast<T1, src_dims, proxy_uses_ref> >
  {
  typedef typename kernel_func_types_inner<T1>::result result;
  };

template<typename T1, size_t src_dims, bool proxy_uses_ref>
struct kernel_func_types_inner< ProxyMatCast<T1, src_dims, proxy_uses_ref> >
  {
  typedef typename kernel_func_types_inner<T1>::result result;
  };

template<typename T1, size_t src_dims, bool proxy_uses_ref>
struct kernel_func_types_inner< ProxyCubeCast<T1, src_dims, proxy_uses_ref> >
  {
  typedef typename kernel_func_types_inner<T1>::result result;
  };

template<typename T1, typename op_type>
struct kernel_func_types_inner< Op<T1, op_type> >
  {
  typedef typename kernel_func_types_inner<T1>::result result;
  };



template<coot_backend_t backend, typename... Ts>
struct kernel_func_types
  {
  typedef typename remove_duplicates<
      typename concat_types< typename kernel_func_types_inner< Ts >::result... >::result
  >::result result;
  };



template<coot_backend_t backend, typename T>
struct kernel_func_inner_applier { };

template<coot_backend_t backend, typename eT, typename T>
struct kernel_func_inner_applier< backend, std::pair<eT, T> >
  {
  typedef typename T::template aux_functions<eT, backend> result;
  };

// for mtGlue functions
template<coot_backend_t backend, typename out_eT, typename in_eT, typename T>
struct kernel_func_inner_applier< backend, std::pair< std::pair<out_eT, in_eT>, T> >
  {
  typedef typename T::template aux_functions<out_eT, in_eT, backend> result;
  };



template<coot_backend_t backend, typename T>
struct kernel_func_applier { };

template<coot_backend_t backend, typename... Ts>
struct kernel_func_applier< backend, std::tuple<Ts...> > : public concat_str
  <
  typename kernel_func_inner_applier<backend, Ts>::result...
  > { };



template<coot_backend_t backend, typename... Ts>
struct kernel_funcs : public kernel_func_applier< backend, typename kernel_func_types<backend, Ts...>::result > { };
