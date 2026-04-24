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



struct at_default_arg_names
  {
  using arg1 = macro_arg1;
  using arg2 = macro_arg2;
  using arg3 = macro_arg3;
  };



//
// macro_elem_access provides a compile-time string definition of a macro function
// COOT_OBJECT_i_AT(name, params...) that is used inside of kernels to provide the
// kernel parameters necessary for a given type T (where e.g. T is a Mat<eT>,
// subview<eT>, etc.)
//
// Each specialization must provide a function len() and str() (returning a char_array<len() + 1>);
// to add a new specialization, adapt the others.
//

template<typename T, size_t i, coot_backend_t backend, typename arg_name_prefix = empty_str, typename arg_names = at_default_arg_names>
struct elem_access_str;

template<typename T, size_t i, coot_backend_t backend, typename arg_name_prefix = empty_str, typename arg_names = at_default_arg_names>
using macro_elem_access = nested_concat_str
  <
  concat_str
    <
    typename macro_defn<backend>::prefix,
    coot_object_prefix,
    index_to_str<i>,
    at_arg_macro
    >,
  elem_access_str<T, i, backend, arg_name_prefix, arg_names>,
  concat_str
    <
    typename macro_defn<backend>::suffix
    >
  >;



//
// Mat<eT>: regular access with row/col
//

template<typename arg_name_prefix = empty_str, typename arg_names = at_default_arg_names>
using at_two_arg_index = concat_str
  <
  typename arg_names::arg1,
  spaced_plus,
  typename arg_names::arg2,
  spaced_mul,
  coot_concat_name,
  arg_name_prefix,
  n_rows_name,
  close_paren
  >;



template<typename eT, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str< Mat<eT>, i, backend, arg_name_prefix, arg_names > : public concat_str
  <
  typename coot_pointer_access<backend, arg_name_prefix>::prefix, // name_ptr[
  at_two_arg_index<arg_name_prefix, arg_names>,                   // row + col * name_n_rows
  typename coot_pointer_access<backend, arg_name_prefix>::suffix  // ]
  > { };



//
// Col<eT>: regular access with element index
//

template<typename eT, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str< Col<eT>, i, backend, arg_name_prefix, arg_names > : public concat_str
  <
  typename coot_pointer_access<backend, arg_name_prefix>::prefix, // name_ptr[
  typename arg_names::arg1,                                       // row
  typename coot_pointer_access<backend, arg_name_prefix>::suffix  // ]
  > { };



//
// Row<eT>: same as Col<eT>
//

template<typename eT, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str< Row<eT>, i, backend, arg_name_prefix, arg_names > : public elem_access_str< Col<eT>, i, backend, arg_name_prefix, arg_names > { };



//
// subview<eT>: two-dimensional access using parent matrix number of rows
//

template<typename arg_name_prefix = empty_str, typename arg_names = at_default_arg_names>
using at_two_arg_subview_index = concat_str
  <
  typename arg_names::arg1, // row
  spaced_plus,              // +
  typename arg_names::arg2, // col
  spaced_mul,               // *
  coot_concat_name,         // COOT_CONCAT(name,
  arg_name_prefix,          //
  M_n_rows_name,            // _M_n_rows
  close_paren               // )
  >;



template<typename eT, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str< subview<eT>, i, backend, arg_name_prefix, arg_names > : public concat_str
  <
  typename coot_pointer_access<backend, arg_name_prefix>::prefix, // name_ptr[
  at_two_arg_subview_index<arg_name_prefix, arg_names>,           // row + col * name_M_n_rows
  typename coot_pointer_access<backend, arg_name_prefix>::suffix  // ]
  > { };



//
// subview_col<eT>: same as Col<eT> (elements are contiguous and offset is handled by pointer)
//

template<typename eT, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str< subview_col<eT>, i, backend, arg_name_prefix, arg_names > : public elem_access_str< Col<eT>, i, backend, arg_name_prefix, arg_names > { };



//
// subview_row<eT>: one-dimensional strided access
//

template<typename arg_name_prefix = empty_str, typename arg_names = at_default_arg_names>
using at_one_arg_strided_index = concat_str
  <
  typename arg_names::arg1, // row
  spaced_mul,               // *
  coot_concat_name,         // COOT_CONCAT(name,
  arg_name_prefix,          //
  incr_name,                // _incr
  close_paren               // )
  >;



template<typename eT, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str< subview_row<eT>, i, backend, arg_name_prefix, arg_names > : public concat_str
  <
  typename coot_pointer_access<backend, arg_name_prefix>::prefix, // name_ptr[
  at_one_arg_strided_index<arg_name_prefix, arg_names>,           // row * name_incr
  typename coot_pointer_access<backend, arg_name_prefix>::suffix  // ]
  > { };



//
// diagview<eT>: one-dimensional strided access, same as subview_row<eT>
//

template<typename eT, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str< diagview<eT>, i, backend, arg_name_prefix, arg_names > : public elem_access_str< subview_row<eT>, i, backend, arg_name_prefix, arg_names > { };



//
// subview_elem1<eT>: uses parameters recursively
//

template<typename eT, typename T1, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str< subview_elem1<eT, T1>, i, backend, arg_name_prefix, arg_names > : public nested_concat_str
  <
  concat_str
    <
    typename coot_pointer_access<backend, arg_name_prefix>::prefix  // name_ptr[
    >,
  elem_access_str< typename elem_vectorised_arg_type<T1>::result, i, backend, concat_str< arg_name_prefix, elem1_index_prefix >, arg_names >, // whatever the indirect lookup is
  concat_str
    <
    typename coot_pointer_access<backend, arg_name_prefix>::suffix  // ]
    >
  > { };



//
// subview_elem2<eT>: uses parameters recursively for row and column indices
//

template<typename arg_names>
struct at_elem2_col_arg_names
  {
  using arg1 = typename arg_names::arg2;
  using arg2 = typename arg_names::arg2; // should never be used anyway
  using arg3 = typename arg_names::arg3; // should never be used anyway
  };



template<typename eT, typename T1, typename T2, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str< subview_elem2<eT, subview_elem2_both<eT, T1, T2>>, i, backend, arg_name_prefix, arg_names > : public nested_concat_str
  <
  concat_str
    <
    typename coot_pointer_access<backend, arg_name_prefix>::prefix // name_ptr[
    >,

  // something like...
  //   name_row_index_ptr[row]
  elem_access_str< typename elem_vectorised_arg_type<T1>::result, i, backend, concat_str< arg_name_prefix, elem2_row_index_prefix >, arg_names >,

  concat_str< spaced_plus, open_paren >, //  + (
  // something like...
  //   (name_row_index_ptr[col] * name_n_rows))
  // note that T2 is expected to have vector type, and we will use the second index argument to access into it
  elem_access_str< typename elem_vectorised_arg_type<T2>::result, i, backend, concat_str< arg_name_prefix, elem2_col_index_prefix >, at_elem2_col_arg_names<arg_names> >,

  concat_str
    <
    spaced_mul,       //  *
    coot_concat_name, // COOT_CONCAT(name,
    arg_name_prefix,  //
    elem2_src_prefix, // _src
    n_rows_name,      // _n_rows
    double_arr_close  // ))]
    >
  > { };



//
// subview_elem2<eT, subview_elem2_all_cols<eT, T1>>: indirect access for rows, direct access for cols
//

template<typename eT, typename T1, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str< subview_elem2<eT, subview_elem2_all_cols<eT, T1>>, i, backend, arg_name_prefix, arg_names > : public nested_concat_str
  <
  concat_str
    <
    typename coot_pointer_access<backend, arg_name_prefix>::prefix  // name_ptr[
    >,

  // something like...
  //   name_row_index_ptr[row]
  elem_access_str< typename elem_vectorised_arg_type<T1>::result, i, backend, concat_str< arg_name_prefix, elem2_row_index_prefix >, arg_names >,

  concat_str<
    spaced_plus,              // +
    open_paren,               // (
    typename arg_names::arg2, // col
    spaced_mul,               //  *
    coot_concat_name,         // COOT_CONCAT(name,
    arg_name_prefix,          //
    elem2_src_prefix,         // _src
    n_rows_name,              // _n_rows
    double_arr_close          // ))]
    >
  > { };



//
// subview_elem2<eT, subview_elem2_all_rows<eT, T2>>: indirect access for rows, direct access for cols
//

template<typename eT, typename T2, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str< subview_elem2<eT, subview_elem2_all_rows<eT, T2>>, i, backend, arg_name_prefix, arg_names > : public nested_concat_str
  <
  concat_str
    <
    typename coot_pointer_access<backend, arg_name_prefix>::prefix, // name_ptr[
    typename arg_names::arg1,                                       // row
    spaced_plus,                                                    // +
    open_paren                                                      // (
    >,

  // something like...
  //   name_row_index_ptr[col] * name_n_rows))
  // note that T2 is expected to have vector type, and we will use the second index argument to access into it
  elem_access_str< typename elem_vectorised_arg_type<T2>::result, i, backend, concat_str< arg_name_prefix, elem2_col_index_prefix >, at_elem2_col_arg_names<arg_names> >,

  concat_str
    <
    spaced_mul,       //  *
    coot_concat_name, // COOT_CONCAT(name,
    arg_name_prefix,  //
    n_rows_name,      // _n_rows
    double_arr_close  // ))]
    >
  > { };



//
// Cube<eT>: uses 3 parameters
//

template<typename arg_name_prefix = empty_str, typename arg_names = at_default_arg_names>
using at_three_arg_index = concat_str
  <
  typename arg_names::arg1, // row
  spaced_plus,              //  +
  typename arg_names::arg2, // col
  spaced_mul,               //  *
  coot_concat_name,         // COOT_CONCAT(name,"
  arg_name_prefix,          //
  n_rows_name,              // _n_rows
  close_paren,              // )
  spaced_plus,              //  +
  typename arg_names::arg3, // slice
  spaced_mul,               //  *
  coot_concat_name,         // COOT_CONCAT(name,"
  arg_name_prefix,          //
  n_rows_name,              // _n_rows
  close_paren,              // )
  spaced_mul,               //  *
  coot_concat_name,         // COOT_CONCAT(name,"
  arg_name_prefix,          //
  n_cols_name,              // _n_cols
  close_paren               // )
  >;



template<typename eT, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str< Cube<eT>, i, backend, arg_name_prefix, arg_names > : public concat_str
  <
  typename coot_pointer_access<backend, arg_name_prefix>::prefix, // name_ptr[
  at_three_arg_index<arg_name_prefix, arg_names>,                 // row + col * name_n_rows + slice * name_n_rows * name_n_cols
  typename coot_pointer_access<backend, arg_name_prefix>::suffix  // ]
  > { };



//
// subview_cube<eT>: uses 3 parameters
//

template<typename arg_name_prefix = empty_str, typename arg_names = at_default_arg_names>
using at_three_arg_subview_index = concat_str
  <
  typename arg_names::arg1, // row
  spaced_plus,              //  +
  typename arg_names::arg2, // col
  spaced_mul,               //  *
  coot_concat_name,         // COOT_CONCAT(name,"
  arg_name_prefix,          //
  M_n_rows_name,            // _M_n_rows
  close_paren,              // )
  spaced_plus,              //  +
  typename arg_names::arg3, // slice
  spaced_mul,               //  *
  coot_concat_name,         // COOT_CONCAT(name,"
  arg_name_prefix,          //
  M_n_elem_slice_name,      // _M_n_elem_slice
  close_paren               // )
  >;



template<typename eT, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str< subview_cube<eT>, i, backend, arg_name_prefix, arg_names > : public concat_str
  <
  typename coot_pointer_access<backend, arg_name_prefix>::prefix, // name_ptr[
  at_three_arg_subview_index<arg_name_prefix, arg_names>,         // row + col * name_M_n_rows + slice * name_M_n_elem_slice
  typename coot_pointer_access<backend, arg_name_prefix>::suffix  // ]
  > { };



//
// ProxyColCast<T1>: reinterpret the T1 as a column vector
// we need two different implementations here, one for a 2D input and one for a 3D input
//

template<typename T1, bool proxy_uses_ref, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str< ProxyColCast<T1, 2, proxy_uses_ref>, i, backend, arg_name_prefix, arg_names > : public elem_access_str< T1, i, backend, arg_name_prefix, col_cast_2d_to_1d_arg_names<T1, arg_names, arg_name_prefix> > { };

template<typename T1, bool proxy_uses_ref, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str< ProxyColCast<T1, 3, proxy_uses_ref>, i, backend, arg_name_prefix, arg_names > : public elem_access_str< T1, i, backend, arg_name_prefix, col_cast_3d_to_1d_arg_names<T1, arg_names, arg_name_prefix> > { };



//
// ProxyMatCast<T1>: reinterpret the T1 as a matrix
// we have the extra arguments name_target_n_rows and name_target_n_cols available to us
//

template<typename T1, bool proxy_uses_ref, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str< ProxyMatCast<T1, 1, proxy_uses_ref>, i, backend, arg_name_prefix, arg_names > : public elem_access_str< T1, i, backend, arg_name_prefix, mat_cast_1d_to_2d_arg_names<arg_names, arg_name_prefix> > { };

template<typename T1, bool proxy_uses_ref, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str< ProxyMatCast<T1, 3, proxy_uses_ref>, i, backend, arg_name_prefix, arg_names > : public elem_access_str< T1, i, backend, arg_name_prefix, mat_cast_3d_to_2d_arg_names<T1, arg_names, arg_name_prefix> > { };



//
// ProxyCubeCast<T1>: reinterpret the T1 as a cube
// we also have the name_target_n_rows, name_target_n_cols, and name_target_n_slices variables available to us
//

template<typename T1, bool proxy_uses_ref, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str< ProxyCubeCast<T1, 1, proxy_uses_ref>, i, backend, arg_name_prefix, arg_names > : public elem_access_str< T1, i, backend, arg_name_prefix, cube_cast_1d_to_3d_arg_names<arg_names, arg_name_prefix> > { };

template<typename T1, bool proxy_uses_ref, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str< ProxyCubeCast<T1, 2, proxy_uses_ref>, i, backend, arg_name_prefix, arg_names > : public elem_access_str< T1, i, backend, arg_name_prefix, cube_cast_2d_to_3d_arg_names<T1, arg_names, arg_name_prefix> > { };



//
// eOp<T1, eop_type>: call the eOp's utility function with the right number of arguments
//

// specialization for num_args == 0
template<typename T1, typename eop_type, size_t num_args, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str_eop_helper : public nested_concat_str
  <
  concat_str
    <
    typename eop_type::func_name,
    func_name_suffix<typename T1::elem_type, backend>,
    open_paren
    >,
  elem_access_str< T1, i, backend, arg_name_prefix, arg_names >, // however we access element (row, col, slice) of a T1
  concat_str< close_paren >
  > { };



// specialization for num_args == 1
template<typename T1, typename eop_type, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str_eop_helper<T1, eop_type, 1, i, backend, arg_name_prefix, arg_names> : public nested_concat_str
  <
  concat_str
    <
    typename eop_type::func_name,
    func_name_suffix<typename T1::elem_type, backend>,
    open_paren
    >,

  elem_access_str< T1, i, backend, concat_str<arg_name_prefix, arg_prefix_name>, arg_names >, // however we access element (row, col, slice) of a T1

  concat_str
    <
    space_sep,
    coot_concat_name,
    arg_name_prefix,
    eop_aux1_name,
    double_close_paren
    >
  > { };



// specialization for num_args == 2
template<typename T1, typename eop_type, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str_eop_helper<T1, eop_type, 2, i, backend, arg_name_prefix, arg_names> : public nested_concat_str
  <
  concat_str
    <
    typename eop_type::func_name,
    func_name_suffix<typename T1::elem_type, backend>,
    open_paren
    >,

  elem_access_str< T1, i, backend, concat_str<arg_name_prefix, arg_prefix_name>, arg_names >, // however we access element (row, col, slice) of a T1

  concat_str
    <
    space_sep,
    coot_concat_name,
    arg_name_prefix,
    eop_aux1_name,
    close_space_sep, // ),
    coot_concat_name,
    arg_name_prefix,
    eop_aux2_name,
    double_close_paren
    >
  > { };



template<typename T1, typename eop_type, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str< eOp<T1, eop_type>, i, backend, arg_name_prefix, arg_names > : public elem_access_str_eop_helper< T1, eop_type, eop_type::num_args, i, backend, arg_name_prefix, arg_names > { };

template<typename T1, typename eop_type, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str< eOpCube<T1, eop_type>, i, backend, arg_name_prefix, arg_names > : public elem_access_str_eop_helper< T1, eop_type, eop_type::num_args, i, backend, arg_name_prefix, arg_names > { };



//
// eGlue<T1, T2, eglue_type>: perform the operation between the two arguments
//

template<typename T1, typename T2, typename eglue_type, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str< eGlue<T1, T2, eglue_type>, i, backend, arg_name_prefix, arg_names > : public nested_concat_str
  <
  concat_str
    <
    typename eglue_type::func_name,
    func_name_suffix<typename T1::elem_type, backend>,
    open_paren
    >,

  elem_access_str< T1, i, backend, concat_str<arg_name_prefix, eglue_arg1_name>, arg_names >,

  concat_str< space_sep >,

  elem_access_str< T2, i, backend, concat_str<arg_name_prefix, eglue_arg2_name>, arg_names >,

  concat_str< close_paren >
  > { };

template<typename T1, typename T2, typename eglue_type, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str< eGlueCube<T1, T2, eglue_type>, i, backend, arg_name_prefix, arg_names > : public elem_access_str< eGlue<T1, T2, eglue_type>, i, backend, arg_name_prefix, arg_names > { };



//
// mtop_conv_to: convert the inner element to an out_eT
//

template<typename out_eT, typename T1, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str< mtOp<out_eT, T1, mtop_conv_to>, i, backend, arg_name_prefix, arg_names > : public nested_concat_str
  <
  concat_str
    <
    conv_elem_type_str<out_eT, backend>,
    open_paren
    >,
  elem_access_str< T1, i, backend, arg_name_prefix, arg_names >,
  concat_str< close_paren >
  > { };

template<typename out_eT, typename T1, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str< mtOpCube<out_eT, T1, mtop_conv_to>, i, backend, arg_name_prefix, arg_names > : public elem_access_str< mtOp<out_eT, T1, mtop_conv_to>, i, backend, arg_name_prefix, arg_names > { };



//
// mtop_rel_core: apply the relational operation
//

template<typename out_eT, typename T1, typename mtop_type, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str< mtOp<out_eT, T1, mtop_rel_core<mtop_type> >, i, backend, arg_name_prefix, arg_names > : public nested_concat_str
  <
  concat_str
    <
    typename mtop_type::equiv_mtglue::func_name,
    func_name_suffix<out_eT, backend>,
    func_name_suffix<typename T1::elem_type, backend>,
    open_paren
    >,
  elem_access_str< T1, i, backend, concat_str< arg_name_prefix, arg_prefix_name >, arg_names >,
  concat_str
    <
    space_sep,
    coot_concat_name,
    arg_name_prefix,
    eop_aux1_name,
    double_close_paren
    >
  > { };

template<typename out_eT, typename T1, typename mtop_type, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str< mtOpCube<out_eT, T1, mtop_rel_core<mtop_type> >, i, backend, arg_name_prefix, arg_names > : public elem_access_str< mtOp<out_eT, T1, mtop_conv_to>, i, backend, arg_name_prefix, arg_names > { };



//
// mtGlue<out_eT, T1, T2, mtglue_mixed_core<mtglue_type> >: coerce each inner argument to an out_eT and then perform the operation
//

template<typename out_eT, typename T1, typename T2, typename mtglue_type, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str< mtGlue<out_eT, T1, T2, mtglue_mixed_core<mtglue_type> >, i, backend, arg_name_prefix, arg_names > : public nested_concat_str
  <
  concat_str
    <
    typename mtglue_type::func_name,
    func_name_suffix<out_eT, backend>,
    open_paren,
    conv_elem_type_str<out_eT, backend>,
    open_paren
    >,
  elem_access_str< T1, i, backend, concat_str<arg_name_prefix, eglue_arg1_name>, arg_names >,
  concat_str
    <
    close_space_sep, // ),
    conv_elem_type_str<out_eT, backend>,
    open_paren
    >,
  elem_access_str< T2, i, backend, concat_str<arg_name_prefix, eglue_arg2_name>, arg_names >,
  concat_str< double_close_paren >
  > { };

template<typename out_eT, typename T1, typename T2, typename mtglue_type, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str< mtGlueCube<out_eT, T1, T2, mtglue_mixed_core<mtglue_type> >, i, backend, arg_name_prefix, arg_names > : public elem_access_str< mtGlue<out_eT, T1, T2, mtglue_mixed_core<mtglue_type> >, i, backend, arg_name_prefix, arg_names > { };



//
// mtGlue<out_eT, T1, T2, mtglue_rel_core<mtglue_type> >: the auxiliary function will return an out_eT
//

template<typename out_eT, typename T1, typename T2, typename mtglue_type, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str< mtGlue<out_eT, T1, T2, mtglue_rel_core<mtglue_type> >, i, backend, arg_name_prefix, arg_names > : public nested_concat_str
  <
  concat_str
    <
    typename mtglue_type::func_name,
    func_name_suffix<out_eT, backend>,
    func_name_suffix<typename T1::elem_type, backend>,
    open_paren
    >,
  elem_access_str< T1, i, backend, concat_str<arg_name_prefix, eglue_arg1_name>, arg_names >,
  concat_str< space_sep >,
  elem_access_str< T2, i, backend, concat_str<arg_name_prefix, eglue_arg2_name>, arg_names >,
  concat_str< close_paren >
  > { };

template<typename out_eT, typename T1, typename T2, typename mtglue_type, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str< mtGlueCube<out_eT, T1, T2, mtglue_rel_core<mtglue_type> >, i, backend, arg_name_prefix, arg_names > : public elem_access_str< mtGlue<out_eT, T1, T2, mtglue_rel_core<mtglue_type> >, i, backend, arg_name_prefix, arg_names > { };



//
// Op<T1, op_strans>: swap row/col arguments
//

template<typename T1, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str< Op<T1, op_strans>, i, backend, arg_name_prefix, arg_names > : public elem_access_str< T1, i, backend, arg_name_prefix, trans_arg_names<arg_names> > { };



//
// Op<T1, op_htrans>: swap row/col arguments and take complex conjugate
//

template<typename T1, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct elem_access_str< Op<T1, op_htrans>, i, backend, arg_name_prefix, arg_names > : public nested_concat_str
  <
  concat_str
    <
    conj_str, // coot_conj
    func_name_suffix< typename T1::elem_type, backend >, // optional suffix to disambiguate function names
    open_paren
    >,
  elem_access_str< T1, i, backend, arg_name_prefix, trans_arg_names<arg_names> >, // transposed access of T1
  concat_str< close_paren >
  > { };
