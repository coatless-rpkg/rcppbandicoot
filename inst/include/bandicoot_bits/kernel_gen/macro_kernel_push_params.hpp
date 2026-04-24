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



//
// Vulkan: `macro_kernel_push_params` emits the Vulkan macro `COOT_OBJECT_i_PARAMS(name)`.
// Each expansion produces one `layout(set=0, binding=N, std430) buffer BufN { eT data[]; } name_buf`
// declaration per `dev_mem_t` leaf found in the Proxy's expression tree.
//

template<typename T> struct vk_is_dev_mem_ref : std::false_type {};
template<typename eT> struct vk_is_dev_mem_ref< dev_mem_t<eT> > : std::true_type {};
template<typename eT> struct vk_is_dev_mem_ref< const dev_mem_t<eT> > : std::true_type {};
template<typename eT> struct vk_is_dev_mem_ref< dev_mem_t<eT>& > : std::true_type {};
template<typename eT> struct vk_is_dev_mem_ref< const dev_mem_t<eT>& > : std::true_type {};

template<typename Tuple> struct vk_count_dev_mem_in_tuple;

template<typename... Ts>
struct vk_count_dev_mem_in_tuple< std::tuple<Ts...> >
  {
  static constexpr size_t value = ((vk_is_dev_mem_ref<Ts>::value ? size_t(1) : size_t(0)) + ... + size_t(0));
  };

template<typename T>
struct vk_leaf_count
  {
  static constexpr size_t value = vk_count_dev_mem_in_tuple< typename Proxy<T>::arg_types >::value;
  };



struct buf_decl_open { static inline constexpr auto& str() { return " { "; } };
struct buf_decl_data_decl { static inline constexpr auto& str() { return " data[]; } COOT_CONCAT(name,"; } };
struct buf_decl_close { static inline constexpr auto& str() { return "_buf)"; } };



template<typename eT, size_t binding, typename arg_name_prefix>
using vk_buffer_decl_leaf = concat_str
  <
  vk_buf_prefix,                         // layout(set=0, binding=
  index_to_str<binding>,                 // <binding>
  vk_buf_std430,                         // , std430) buffer Buf
  index_to_str<binding>,                 // <binding>
  buf_decl_open,                         // " { "
  elem_type_str<eT, VULKAN_BACKEND>,     // <eT>
  buf_decl_data_decl,                    // " data[]; } COOT_CONCAT(name,"
  arg_name_prefix,                       // <prefix>
  buf_decl_close                         // "_buf)"
  >;



template<typename T, size_t start_binding, typename arg_name_prefix = empty_str>
struct vk_buffer_decl;

template<typename T> struct vk_is_simple_leaf : std::false_type {};
template<typename eT> struct vk_is_simple_leaf< Mat<eT> > : std::true_type {};
template<typename eT> struct vk_is_simple_leaf< Col<eT> > : std::true_type {};
template<typename eT> struct vk_is_simple_leaf< Row<eT> > : std::true_type {};
template<typename eT> struct vk_is_simple_leaf< subview<eT> > : std::true_type {};
template<typename eT> struct vk_is_simple_leaf< subview_col<eT> > : std::true_type {};
template<typename eT> struct vk_is_simple_leaf< subview_row<eT> > : std::true_type {};
template<typename eT> struct vk_is_simple_leaf< diagview<eT> > : std::true_type {};
template<typename eT> struct vk_is_simple_leaf< Cube<eT> > : std::true_type {};
template<typename eT> struct vk_is_simple_leaf< subview_cube<eT> > : std::true_type {};

template<typename T, size_t start_binding, typename arg_name_prefix, typename = void>
struct vk_buffer_decl_impl;

template<typename T, size_t start_binding, typename arg_name_prefix>
struct vk_buffer_decl_impl<T, start_binding, arg_name_prefix,
                           typename std::enable_if<vk_is_simple_leaf<T>::value>::type>
  : public vk_buffer_decl_leaf<typename T::elem_type, start_binding, arg_name_prefix> { };

template<typename T, size_t start_binding, typename arg_name_prefix>
struct vk_buffer_decl : public vk_buffer_decl_impl<T, start_binding, arg_name_prefix> { };

template<typename eT, typename T1, size_t start_binding, typename arg_name_prefix>
struct vk_buffer_decl< subview_elem1<eT, T1>, start_binding, arg_name_prefix > : public nested_concat_str
  <
  vk_buffer_decl_leaf<eT, start_binding, arg_name_prefix>,
  concat_str< space_semicolon >,
  vk_buffer_decl< typename elem_vectorised_arg_type<T1>::result, start_binding + 1, concat_str< arg_name_prefix, elem1_index_prefix > >
  > { };


template<typename eT, typename T1, typename T2, size_t start_binding, typename arg_name_prefix>
struct vk_buffer_decl< subview_elem2<eT, subview_elem2_both<eT, T1, T2>>, start_binding, arg_name_prefix > : public nested_concat_str
  <
  vk_buffer_decl_leaf<eT, start_binding, arg_name_prefix>,
  concat_str< space_semicolon >,
  vk_buffer_decl< typename elem_vectorised_arg_type<T1>::result, start_binding + 1, concat_str< arg_name_prefix, elem2_row_index_prefix > >,
  concat_str< space_semicolon >,
  vk_buffer_decl< typename elem_vectorised_arg_type<T2>::result,
                  start_binding + 1 + vk_leaf_count< typename elem_vectorised_arg_type<T1>::result >::value,
                  concat_str< arg_name_prefix, elem2_col_index_prefix > >
  > { };

template<typename eT, typename T1, size_t start_binding, typename arg_name_prefix>
struct vk_buffer_decl< subview_elem2<eT, subview_elem2_all_cols<eT, T1>>, start_binding, arg_name_prefix > : public nested_concat_str
  <
  vk_buffer_decl_leaf<eT, start_binding, arg_name_prefix>,
  concat_str< space_semicolon >,
  vk_buffer_decl< typename elem_vectorised_arg_type<T1>::result, start_binding + 1, concat_str< arg_name_prefix, elem2_row_index_prefix > >
  > { };

template<typename eT, typename T2, size_t start_binding, typename arg_name_prefix>
struct vk_buffer_decl< subview_elem2<eT, subview_elem2_all_rows<eT, T2>>, start_binding, arg_name_prefix > : public nested_concat_str
  <
  vk_buffer_decl_leaf<eT, start_binding, arg_name_prefix>,
  concat_str< space_semicolon >,
  vk_buffer_decl< typename elem_vectorised_arg_type<T2>::result, start_binding + 1, concat_str< arg_name_prefix, elem2_col_index_prefix > >
  > { };

template<typename T1, size_t src_dims, bool proxy_uses_ref, size_t start_binding, typename arg_name_prefix>
struct vk_buffer_decl< ProxyColCast<T1, src_dims, proxy_uses_ref>, start_binding, arg_name_prefix > : public vk_buffer_decl<T1, start_binding, arg_name_prefix> { };

template<typename T1, size_t src_dims, bool proxy_uses_ref, size_t start_binding, typename arg_name_prefix>
struct vk_buffer_decl< ProxyMatCast<T1, src_dims, proxy_uses_ref>, start_binding, arg_name_prefix > : public vk_buffer_decl<T1, start_binding, arg_name_prefix> { };

template<typename T1, size_t src_dims, bool proxy_uses_ref, size_t start_binding, typename arg_name_prefix>
struct vk_buffer_decl< ProxyCubeCast<T1, src_dims, proxy_uses_ref>, start_binding, arg_name_prefix > : public vk_buffer_decl<T1, start_binding, arg_name_prefix> { };

template<typename T1, typename eop_type, size_t num_args, size_t start_binding, typename arg_name_prefix>
struct vk_buffer_decl_eop_helper : public vk_buffer_decl<T1, start_binding, arg_name_prefix> { };

template<typename T1, typename eop_type, size_t start_binding, typename arg_name_prefix>
struct vk_buffer_decl_eop_helper<T1, eop_type, 1, start_binding, arg_name_prefix>
  : public vk_buffer_decl<T1, start_binding, concat_str<arg_name_prefix, arg_prefix_name>> { };

template<typename T1, typename eop_type, size_t start_binding, typename arg_name_prefix>
struct vk_buffer_decl_eop_helper<T1, eop_type, 2, start_binding, arg_name_prefix>
  : public vk_buffer_decl<T1, start_binding, concat_str<arg_name_prefix, arg_prefix_name>> { };

template<typename T1, typename eop_type, size_t start_binding, typename arg_name_prefix>
struct vk_buffer_decl< eOp<T1, eop_type>, start_binding, arg_name_prefix >
  : public vk_buffer_decl_eop_helper<T1, eop_type, eop_type::num_args, start_binding, arg_name_prefix> { };

template<typename T1, typename eop_type, size_t start_binding, typename arg_name_prefix>
struct vk_buffer_decl< eOpCube<T1, eop_type>, start_binding, arg_name_prefix >
  : public vk_buffer_decl_eop_helper<T1, eop_type, eop_type::num_args, start_binding, arg_name_prefix> { };

template<typename T1, typename T2, typename eglue_type, size_t start_binding, typename arg_name_prefix>
struct vk_buffer_decl< eGlue<T1, T2, eglue_type>, start_binding, arg_name_prefix > : public nested_concat_str
  <
  vk_buffer_decl< T1, start_binding, concat_str< arg_name_prefix, eglue_arg1_name > >,
  concat_str< space_semicolon >,
  vk_buffer_decl< T2, start_binding + vk_leaf_count<T1>::value, concat_str< arg_name_prefix, eglue_arg2_name > >
  > { };

template<typename T1, typename T2, typename eglue_type, size_t start_binding, typename arg_name_prefix>
struct vk_buffer_decl< eGlueCube<T1, T2, eglue_type>, start_binding, arg_name_prefix > : public nested_concat_str
  <
  vk_buffer_decl< T1, start_binding, concat_str< arg_name_prefix, eglue_arg1_name > >,
  concat_str< space_semicolon >,
  vk_buffer_decl< T2, start_binding + vk_leaf_count<T1>::value, concat_str< arg_name_prefix, eglue_arg2_name > >
  > { };

template<typename out_eT, typename T1, size_t start_binding, typename arg_name_prefix>
struct vk_buffer_decl< mtOp<out_eT, T1, mtop_conv_to>, start_binding, arg_name_prefix > : public vk_buffer_decl<T1, start_binding, arg_name_prefix> { };

template<typename out_eT, typename T1, size_t start_binding, typename arg_name_prefix>
struct vk_buffer_decl< mtOpCube<out_eT, T1, mtop_conv_to>, start_binding, arg_name_prefix > : public vk_buffer_decl<T1, start_binding, arg_name_prefix> { };

template<typename out_eT, typename T1, typename mtop_type, size_t start_binding, typename arg_name_prefix>
struct vk_buffer_decl< mtOp<out_eT, T1, mtop_rel_core<mtop_type>>, start_binding, arg_name_prefix >
  : public vk_buffer_decl<T1, start_binding, concat_str<arg_name_prefix, arg_prefix_name>> { };

template<typename out_eT, typename T1, typename mtop_type, size_t start_binding, typename arg_name_prefix>
struct vk_buffer_decl< mtOpCube<out_eT, T1, mtop_rel_core<mtop_type>>, start_binding, arg_name_prefix >
  : public vk_buffer_decl<T1, start_binding, arg_name_prefix> { };

template<typename out_eT, typename T1, typename T2, typename mtglue_type, size_t start_binding, typename arg_name_prefix>
struct vk_buffer_decl< mtGlue<out_eT, T1, T2, mtglue_mixed_core<mtglue_type>>, start_binding, arg_name_prefix > : public nested_concat_str
  <
  vk_buffer_decl< T1, start_binding, concat_str< arg_name_prefix, eglue_arg1_name > >,
  concat_str< space_semicolon >,
  vk_buffer_decl< T2, start_binding + vk_leaf_count<T1>::value, concat_str< arg_name_prefix, eglue_arg2_name > >
  > { };

template<typename out_eT, typename T1, typename T2, typename mtglue_type, size_t start_binding, typename arg_name_prefix>
struct vk_buffer_decl< mtGlueCube<out_eT, T1, T2, mtglue_mixed_core<mtglue_type>>, start_binding, arg_name_prefix > : public nested_concat_str
  <
  vk_buffer_decl< T1, start_binding, concat_str< arg_name_prefix, eglue_arg1_name > >,
  concat_str< space_semicolon >,
  vk_buffer_decl< T2, start_binding + vk_leaf_count<T1>::value, concat_str< arg_name_prefix, eglue_arg2_name > >
  > { };

template<typename out_eT, typename T1, typename T2, typename mtglue_type, size_t start_binding, typename arg_name_prefix>
struct vk_buffer_decl< mtGlue<out_eT, T1, T2, mtglue_rel_core<mtglue_type>>, start_binding, arg_name_prefix > : public nested_concat_str
  <
  vk_buffer_decl< T1, start_binding, concat_str< arg_name_prefix, eglue_arg1_name > >,
  concat_str< space_semicolon >,
  vk_buffer_decl< T2, start_binding + vk_leaf_count<T1>::value, concat_str< arg_name_prefix, eglue_arg2_name > >
  > { };

template<typename out_eT, typename T1, typename T2, typename mtglue_type, size_t start_binding, typename arg_name_prefix>
struct vk_buffer_decl< mtGlueCube<out_eT, T1, T2, mtglue_rel_core<mtglue_type>>, start_binding, arg_name_prefix > : public nested_concat_str
  <
  vk_buffer_decl< T1, start_binding, concat_str< arg_name_prefix, eglue_arg1_name > >,
  concat_str< space_semicolon >,
  vk_buffer_decl< T2, start_binding + vk_leaf_count<T1>::value, concat_str< arg_name_prefix, eglue_arg2_name > >
  > { };



template<typename T1, size_t start_binding, typename arg_name_prefix>
struct vk_buffer_decl< Op<T1, op_htrans>, start_binding, arg_name_prefix > : public vk_buffer_decl<T1, start_binding, arg_name_prefix> { };

template<typename T1, size_t start_binding, typename arg_name_prefix>
struct vk_buffer_decl< Op<T1, op_strans>, start_binding, arg_name_prefix > : public vk_buffer_decl<T1, start_binding, arg_name_prefix> { };



template<typename T, size_t i, size_t start_binding, typename arg_name_prefix = empty_str>
struct macro_kernel_push_params : public nested_concat_str
  <
  concat_str
    <
    typename macro_defn<VULKAN_BACKEND>::prefix,   // -D
    coot_object_prefix,                            // COOT_OBJECT_
    index_to_str<i>,                               // <i>
    params_suffix                                  // _PARAMS(name)=
    >,
  vk_buffer_decl<T, start_binding, arg_name_prefix>,
  concat_str
    <
    typename macro_defn<VULKAN_BACKEND>::suffix
    >
  > { };
