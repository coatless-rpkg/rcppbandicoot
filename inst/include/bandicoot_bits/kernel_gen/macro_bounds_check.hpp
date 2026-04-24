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



struct bounds_check_default_arg_names
  {
  using arg1 = macro_arg1;
  using arg2 = macro_arg2;
  using arg3 = macro_arg3;
  };



template<typename arg_name_prefix = empty_str, typename arg_names = bounds_check_default_arg_names>
using bounds_check_rows = concat_str
  <
  open_paren,               // (
  typename arg_names::arg1, // row
  spaced_lt,                //  <
  coot_concat_name,         // COOT_CONCAT(name,
  arg_name_prefix,          //
  n_rows_name,              // _n_rows
  double_close_paren        // ))
  >;



template<typename arg_name_prefix = empty_str, typename arg_names = bounds_check_default_arg_names>
using bounds_check_cols = concat_str
  <
  open_paren,               // (
  typename arg_names::arg2, // col
  spaced_lt,                //  <
  coot_concat_name,         // COOT_CONCAT(name,
  arg_name_prefix,          //
  n_cols_name,              // _n_cols
  double_close_paren        // ))
  >;



template<typename arg_name_prefix = empty_str, typename arg_names = bounds_check_default_arg_names>
using bounds_check_slices = concat_str
  <
  open_paren,               // (
  typename arg_names::arg3, // slice
  spaced_lt,                //  <
  coot_concat_name,         // COOT_CONCAT(name,
  arg_name_prefix,          //
  n_slices_name,            // _n_slices
  double_close_paren        // ))
  >;



template<typename arg_name_prefix = empty_str, typename arg_names = bounds_check_default_arg_names>
using bounds_check_elem = concat_str
  <
  open_paren,               // (
  typename arg_names::arg1, // row
  spaced_lt,                //  <
  coot_concat_name,         // COOT_CONCAT(name,
  arg_name_prefix,          //
  n_elem_name,              // _n_elem
  double_close_paren        // ))
  >;




//
// macro_bounds_check provides a compile-time string definition of a macro
// function COOT_OBJECT_i_BOUNDS_CHECK(name, params...) that is used inside of
// kernels to provide a bounds check and ensure that the given parameters are
// inside the bounds of the given T (where e.g. T is a Mat<eT>, subview<eT>, etc.).
//
// Each specialization must provide a function len() and str() (returning a char_array<len() + 1>);
// to add a new specialization, adapt the others.
//

template<typename T, size_t i, coot_backend_t backend, typename arg_name_prefix = empty_str, typename arg_names = bounds_check_default_arg_names>
struct bounds_check_str;


template<typename T, size_t i, coot_backend_t backend, typename arg_name_prefix = empty_str, typename arg_names = bounds_check_default_arg_names>
using macro_bounds_check = nested_concat_str
  <
  concat_str
    <
    typename macro_defn<backend>::prefix,
    coot_object_prefix,
    index_to_str<i>,
    bounds_check_arg_macro
    >,
  bounds_check_str<T, i, backend, arg_name_prefix, arg_names >,
  concat_str< typename macro_defn<backend>::suffix >
  >;



//
// Mat<eT>
//   COOT_OBJECT_i_BOUNDS_CHECK(name, row, col, _)=(row < name_n_rows && col < name_n_cols)
//

template<typename eT, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct bounds_check_str< Mat<eT>, i, backend, arg_name_prefix, arg_names > : public concat_str
  <
  open_paren,
  bounds_check_rows<arg_name_prefix, arg_names>, // (row < name_n_rows)
  spaced_and,                                    // &&
  bounds_check_cols<arg_name_prefix, arg_names>, // (col < name_n_cols)
  close_paren
  > { };



//
// Col<eT>
//   COOT_OBJECT_i_BOUNDS_CHECK(name, i)="(i < name_n_elem)"
//

template<typename eT, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct bounds_check_str< Col<eT>, i, backend, arg_name_prefix, arg_names > : public bounds_check_elem<arg_name_prefix, arg_names> { };



//
// Row<eT>
//   COOT_OBJECT_i_BOUNDS_CHECK(name, i, _, __)=(i < name_n_elem)
//   (same as Col<eT>)
//

template<typename eT, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct bounds_check_str< Row<eT>, i, backend, arg_name_prefix, arg_names > : public bounds_check_str< Col<eT>, i, backend, arg_name_prefix, arg_names > { };



//
// subview<eT>: same as Mat<eT>
//

template<typename eT, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct bounds_check_str< subview<eT>, i, backend, arg_name_prefix, arg_names > : public bounds_check_str< Mat<eT>, i, backend, arg_name_prefix, arg_names > { };



//
// subview_col<eT> and subview_row<eT>: same as Col<eT> and Row<eT>
//

template<typename eT, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct bounds_check_str< subview_col<eT>, i, backend, arg_name_prefix, arg_names > : public bounds_check_str< Col<eT>, i, backend, arg_name_prefix, arg_names > { };

template<typename eT, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct bounds_check_str< subview_row<eT>, i, backend, arg_name_prefix, arg_names > : public bounds_check_str< Row<eT>, i, backend, arg_name_prefix, arg_names > { };



//
// diagview<eT>: like a vector so we can reuse Row<eT>/Col<eT> implementation
//

template<typename eT, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct bounds_check_str< diagview<eT>, i, backend, arg_name_prefix, arg_names > : public bounds_check_str< Col<eT>, i, backend, arg_name_prefix, arg_names > { };



//
// subview_elem1<eT, T1>: indirect bounds check for the T1
//

template<typename eT, typename T1, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct bounds_check_str< subview_elem1< eT, T1 >, i, backend, arg_name_prefix, arg_names > : public bounds_check_str< typename elem_vectorised_arg_type<T1>::result, i, backend, concat_str< arg_name_prefix, elem1_index_prefix >, arg_names > { };



//
// subview_elem2<eT, subview_elem2_both<eT, T1, T2>>: indirect two-dimensional bounds check
//

template<typename arg_names>
struct bounds_check_elem2_col_arg_names
  {
  using arg1 = typename arg_names::arg2;
  using arg2 = typename arg_names::arg2; // should never be used
  using arg3 = typename arg_names::arg3; // should never be used
  };

template<typename eT, typename T1, typename T2, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct bounds_check_str< subview_elem2< eT, subview_elem2_both<eT, T1, T2> >, i, backend, arg_name_prefix, arg_names > : public nested_concat_str
  <
  concat_str< open_paren >,
  // child 1 bounds check, e.g., (row < name_row_index_n_elem)
  bounds_check_str< typename elem_vectorised_arg_type<T1>::result, i, backend, concat_str< arg_name_prefix, elem2_row_index_prefix >, arg_names >,
  concat_str< spaced_and >, // &&
  // child 2 bounds check, e.g., (col < name_col_index_n_elem)
  bounds_check_str< typename elem_vectorised_arg_type<T2>::result, i, backend, concat_str< arg_name_prefix, elem2_col_index_prefix >, bounds_check_elem2_col_arg_names< arg_names > >,
  concat_str< close_paren >
  > { };



//
// subview_elem2<eT, subview_elem2_all_cols<eT, T1>>: indirect bounds check on rows, direct check on cols
//

template<typename eT, typename T1, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct bounds_check_str< subview_elem2< eT, subview_elem2_all_cols<eT, T1> >, i, backend, arg_name_prefix, arg_names > : public concat_str
  <
  concat_str< open_paren >,
  // child 1 bounds check, e.g., (row < name_row_index_n_elem)
  bounds_check_str< typename elem_vectorised_arg_type<T1>::result, i, backend, concat_str< arg_name_prefix, elem2_row_index_prefix >, arg_names >,
  concat_str
    <
    spaced_and, // &&
    // direct bounds check for columns, e.g., (col < COOT_CONCAT(name,_n_cols))
    open_paren,
    typename arg_names::arg2,
    spaced_lt, //  <
    coot_concat_name,
    arg_name_prefix,
    n_cols_name,
    triple_close_paren // )))
    >
  > { };



//
// subview_elem2<eT, subview_elem2_all_rows<eT, T2>>: direct bounds check on rows, indirect check on cols
//

template<typename eT, typename T2, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct bounds_check_str< subview_elem2< eT, subview_elem2_all_rows<eT, T2> >, i, backend, arg_name_prefix, arg_names > : public concat_str
  <
  concat_str
    <
    double_open_paren,
    // direct bounds check for rows, e.g., (row < COOT_CONCAT(name,_n_rows))
    typename arg_names::arg1,           // |
    spaced_lt,                          // |
    coot_concat_name,                   // |
    arg_name_prefix,                    // +-> row < COOT_CONCAT(name,_n_rows)) :
    n_rows_name,                        // |
    double_close_paren,                 // |
    spaced_and                          // &&
    >,
  // child 2 bounds check, e.g., (col < name_col_index_n_elem)
  bounds_check_str< typename elem_vectorised_arg_type<T2>::result, i, backend, concat_str< arg_name_prefix, elem2_col_index_prefix >, bounds_check_elem2_col_arg_names< arg_names > >,
  concat_str< close_paren >
  > { };



//
// Cube<eT>
//   COOT_OBJECT_i_BOUNDS_CHECK(name, row, col, slice)=(row < name_n_rows && col < name_n_cols && slice < name_n_slices)
//

template<typename eT, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct bounds_check_str< Cube<eT>, i, backend, arg_name_prefix, arg_names > : public concat_str
  <
  open_paren,
  bounds_check_rows<arg_name_prefix, arg_names>,   // (row < name_n_rows)
  spaced_and,                                      // &&
  bounds_check_cols<arg_name_prefix, arg_names>,   // (col < name_n_cols)
  spaced_and,                                      // &&
  bounds_check_slices<arg_name_prefix, arg_names>, // (col < name_n_slices)
  close_paren
  > { };



//
// subview_cube<eT>: same as Cube<eT>
//

template<typename eT, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct bounds_check_str< subview_cube<eT>, i, backend, arg_name_prefix, arg_names > : public bounds_check_str< Cube<eT>, i, backend, arg_name_prefix, arg_names > { };



//
// ProxyColCast<T1>: reinterpret the T1 as a column vector
// we need two different implementations here, one for a 2D input and one for a 3D input
//

// Utility to extract the number of rows or columns from an argument

template<typename T1, typename arg_name_prefix>
struct proxycast_n_rows_name : public concat_str
  <
  coot_concat_name,
  arg_name_prefix,
  n_rows_name,
  close_paren
  > { };

template<typename T1, typename T2, typename eglue_type, typename arg_name_prefix>
struct proxycast_n_rows_name< eGlue<T1, T2, eglue_type>, arg_name_prefix > : public proxycast_n_rows_name< T1, concat_str<arg_name_prefix, eglue_arg1_name> > { };

template<typename T1, typename arg_name_prefix>
struct proxycast_n_cols_name : public concat_str
  <
  coot_concat_name,
  arg_name_prefix,
  n_cols_name,
  close_paren
  > { };

template<typename T1, typename T2, typename eglue_type, typename arg_name_prefix>
struct proxycast_n_cols_name< eGlue<T1, T2, eglue_type>, arg_name_prefix > : public proxycast_n_cols_name< T1, concat_str<arg_name_prefix, eglue_arg1_name> > { };



template<typename T1, typename arg_names, typename arg_name_prefix>
struct col_cast_2d_to_1d_arg_names
  {
  using arg1 = concat_str<
    open_paren,                                 // (
    typename arg_names::arg1,                   // row
    spaced_mod,                                 //  %
    proxycast_n_rows_name<T1, arg_name_prefix>, // COOT_CONCAT(name,_n_rows)
    close_paren                                 // )
    >;

  using arg2 = concat_str<
    open_paren,                                 // (
    typename arg_names::arg1,                   // row
    spaced_div,                                 //  /
    proxycast_n_rows_name<T1, arg_name_prefix>, // COOT_CONCAT(name,_n_rows)
    close_paren                                 // )
    >;

  using arg3 = typename arg_names::arg3;
  };

template<typename T1, bool proxy_uses_ref, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct bounds_check_str< ProxyColCast<T1, 2, proxy_uses_ref>, i, backend, arg_name_prefix, arg_names > : public bounds_check_str< T1, i, backend, arg_name_prefix, col_cast_2d_to_1d_arg_names<T1, arg_names, arg_name_prefix> > { };



template<typename T1, typename arg_names, typename arg_name_prefix>
struct col_cast_3d_to_1d_arg_names
  {
  using arg1 = concat_str<
    open_paren,                                 // (
    typename arg_names::arg1,                   // row
    spaced_mod,                                 //  %
    proxycast_n_rows_name<T1, arg_name_prefix>, // COOT_CONCAT(name,_n_rows)
    close_paren                                 // ))
    >;

  using arg2 = concat_str<
    double_open_paren,                          // ((
    typename arg_names::arg1,                   // row
    spaced_div,                                 //  /
    proxycast_n_rows_name<T1, arg_name_prefix>, // COOT_CONCAT(name,_n_rows)
    close_paren,                                // )
    spaced_mod,                                 //  %
    proxycast_n_cols_name<T1, arg_name_prefix>, // COOT_CONCAT(name,_n_cols)
    close_paren                                 // )
    >;

  using arg3 = concat_str<
    open_paren,                                 // (
    typename arg_names::arg1,                   // row
    spaced_div,                                 //  /
    open_paren,                                 // (
    proxycast_n_rows_name<T1, arg_name_prefix>, // COOT_CONCAT(name,_n_rows)
    spaced_mul,                                 //  *
    proxycast_n_cols_name<T1, arg_name_prefix>, // COOT_CONCAT(name,_n_cols)
    double_close_paren                          // ))
    >;
  };



template<typename T1, bool proxy_uses_ref, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct bounds_check_str< ProxyColCast<T1, 3, proxy_uses_ref>, i, backend, arg_name_prefix, arg_names > : public bounds_check_str< T1, i, backend, arg_name_prefix, col_cast_3d_to_1d_arg_names<T1, arg_names, arg_name_prefix> > { };



//
// ProxyMatCast<T1>: reinterpret the T1 as a matrix
// we also have the name_target_n_rows and name_target_n_cols variables available to us
//

template<typename arg_names, typename arg_name_prefix>
using mat_cast_to_linear_index = concat_str
  <
  open_paren,                 // (
  typename arg_names::arg1,   // row
  spaced_plus,                //  +
  typename arg_names::arg2,   // col
  spaced_mul,                 //  *
  coot_concat_name,           // COOT_CONCAT(name,
  arg_name_prefix,            //
  target_name,                // _target
  n_rows_name,                // _n_rows
  double_close_paren          // ))
  >;



template<typename arg_names, typename arg_name_prefix>
struct mat_cast_1d_to_2d_arg_names
  {
  using arg1 = mat_cast_to_linear_index<arg_names, arg_name_prefix>;

  // should be unused
  using arg2 = typename arg_names::arg2;
  using arg3 = typename arg_names::arg3;
  };



template<typename T1, bool proxy_uses_ref, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct bounds_check_str< ProxyMatCast<T1, 1, proxy_uses_ref>, i, backend, arg_name_prefix, arg_names > : public bounds_check_str< T1, i, backend, arg_name_prefix, mat_cast_1d_to_2d_arg_names<arg_names, arg_name_prefix> > { };



template<typename T1, typename arg_names, typename arg_name_prefix>
struct mat_cast_3d_to_2d_arg_names
  {
  using arg1 = concat_str
    <
    open_paren,                                           // (
    mat_cast_to_linear_index<arg_names, arg_name_prefix>, // (row + col * COOT_CONCAT(name,_target_n_rows))
    spaced_mod,                                           //  %
    proxycast_n_rows_name<T1, arg_name_prefix>,           // COOT_CONCAT(name,_n_rows)
    close_paren                                           // )
    >;

  using arg2 = concat_str
    <
    double_open_paren,                                    // ((
    mat_cast_to_linear_index<arg_names, arg_name_prefix>, // (row + col * COOT_CONCAT(name,_target_n_rows))
    spaced_div,                                           //  /
    proxycast_n_rows_name<T1, arg_name_prefix>,           // COOT_CONCAT(name,_n_rows)
    close_paren,                                          // )
    spaced_mod,                                           //  %
    proxycast_n_cols_name<T1, arg_name_prefix>,           // COOT_CONCAT(name,_n_cols)
    close_paren                                           // )
    >;

  using arg3 = concat_str
    <
    open_paren,                                           // (
    mat_cast_to_linear_index<arg_names, arg_name_prefix>, // (row + col * COOT_CONCAT(name,_target_n_rows))
    spaced_div,                                           //  %
    open_paren,                                           // (
    proxycast_n_rows_name<T1, arg_name_prefix>,           // COOT_CONCAT(name,_n_rows)
    spaced_mul,                                           //  *
    proxycast_n_cols_name<T1, arg_name_prefix>,           // COOT_CONCAT(name,_n_cols)
    double_close_paren                                    // )))
    >;
  };



template<typename T1, bool proxy_uses_ref, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct bounds_check_str< ProxyMatCast<T1, 3, proxy_uses_ref>, i, backend, arg_name_prefix, arg_names > : public bounds_check_str< T1, i, backend, arg_name_prefix, mat_cast_3d_to_2d_arg_names<T1, arg_names, arg_name_prefix> > { };



//
// ProxyCubeCast<T1>: reinterpret the T1 as a cube
// we also have the name_target_n_rows, name_target_n_cols, and name_target_n_slices variables available to us
//


template<typename arg_names, typename arg_name_prefix>
using cube_cast_to_linear_index = concat_str
  <
  open_paren,               // (
  typename arg_names::arg1, // row
  spaced_plus,              //  +
  typename arg_names::arg2, // col
  spaced_mul,               //  *
  coot_concat_name,         // COOT_CONCAT(name,
  arg_name_prefix,          //
  target_name,              // _target
  n_rows_name,              // _n_rows
  close_paren,              // )
  spaced_plus,              //  +
  typename arg_names::arg3, // slice
  spaced_mul,               //  *
  coot_concat_name,         // COOT_CONCAT(name,
  arg_name_prefix,          //
  target_name,              // _target
  n_rows_name,              // _n_rows
  close_paren,              // )
  spaced_mul,               //  *
  coot_concat_name,         // COOT_CONCAT(name,
  arg_name_prefix,          //
  target_name,              // _target
  n_cols_name,              // _n_cols
  double_close_paren        // ))
  >;

template<typename arg_names, typename arg_name_prefix>
struct cube_cast_1d_to_3d_arg_names
  {
  using arg1 = cube_cast_to_linear_index<arg_names, arg_name_prefix>;

  // should be unused
  using arg2 = typename arg_names::arg2;
  using arg3 = typename arg_names::arg3;
  };



template<typename T1, typename arg_names, typename arg_name_prefix>
struct cube_cast_2d_to_3d_arg_names
  {
  using arg1 = concat_str
    <
    open_paren,                                            // (
    cube_cast_to_linear_index<arg_names, arg_name_prefix>, // (row + col * COOT_CONCAT(name,_target_n_rows) + slice * COOT_CONCAT(name,_target_n_rows) * COOT_CONCAT(name,_target_n_cols))
    spaced_mod,                                            //  %
    proxycast_n_rows_name<T1, arg_name_prefix>,            // COOT_CONCAT(name,_n_rows)
    close_paren                                            // )
    >;

  using arg2 = concat_str
    <
    open_paren,                                            // (
    cube_cast_to_linear_index<arg_names, arg_name_prefix>, // (row + col * COOT_CONCAT(name,_target_n_rows) + slice * COOT_CONCAT(name,_target_n_rows) * COOT_CONCAT(name,_target_n_cols))
    spaced_div,                                            //  /
    proxycast_n_rows_name<T1, arg_name_prefix>,            // COOT_CONCAT(name,_n_rows)
    close_paren                                            // )
    >;

  // should never be used
  using arg3 = typename arg_names::arg3;
  };



template<typename T1, bool proxy_uses_ref, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct bounds_check_str< ProxyCubeCast<T1, 1, proxy_uses_ref>, i, backend, arg_name_prefix, arg_names > : public bounds_check_str< T1, i, backend, arg_name_prefix, cube_cast_1d_to_3d_arg_names<arg_names, arg_name_prefix> > { };

template<typename T1, bool proxy_uses_ref, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct bounds_check_str< ProxyCubeCast<T1, 2, proxy_uses_ref>, i, backend, arg_name_prefix, arg_names > : public bounds_check_str< T1, i, backend, arg_name_prefix, cube_cast_2d_to_3d_arg_names<T1, arg_names, arg_name_prefix> > { };



//
// eOp<T1, eop_type> (and related): we can use the regular bounds check, but for any child arguments we have to append the right prefix
//

// apply the arg_ prefix to any inner arguments, since the eOp has more than one argument
template<typename T1, typename eop_type, size_t num_args, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct bounds_check_str_eop_helper : public bounds_check_str< T1, i, backend, concat_str<arg_name_prefix, arg_prefix_name>, arg_names > { };

// specialize for 0 where we don't need any extra arg_ prefix on the argument names
template<typename T1, typename eop_type, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct bounds_check_str_eop_helper< T1, eop_type, 0, i, backend, arg_name_prefix, arg_names > : public bounds_check_str< T1, i, backend, arg_name_prefix, arg_names > { };

template<typename T1, typename eop_type, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct bounds_check_str< eOp<T1, eop_type>, i, backend, arg_name_prefix, arg_names > : public bounds_check_str_eop_helper< T1, eop_type, eop_type::num_args, i, backend, arg_name_prefix, arg_names > { };

template<typename T1, typename eop_type, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct bounds_check_str< eOpCube<T1, eop_type>, i, backend, arg_name_prefix, arg_names > : public bounds_check_str_eop_helper< T1, eop_type, eop_type::num_args, i, backend, arg_name_prefix, arg_names > { };



//
// eGlue: just check bounds in the first object
//

template<typename T1, typename T2, typename eglue_type, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct bounds_check_str< eGlue<T1, T2, eglue_type>, i, backend, arg_name_prefix, arg_names > : public bounds_check_str< T1, i, backend, concat_str< arg_name_prefix, eglue_arg1_name >, arg_names > { };

template<typename T1, typename T2, typename eglue_type, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct bounds_check_str< eGlueCube<T1, T2, eglue_type>, i, backend, arg_name_prefix, arg_names > : public bounds_check_str< T1, i, backend, concat_str< arg_name_prefix, eglue_arg1_name >, arg_names > { };



//
// mtop_conv_to: just the bounds check of the argument
//

template<typename out_eT, typename T1, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct bounds_check_str< mtOp<out_eT, T1, mtop_conv_to>, i, backend, arg_name_prefix, arg_names > : public bounds_check_str< T1, i, backend, arg_name_prefix, arg_names > { };

template<typename out_eT, typename T1, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct bounds_check_str< mtOpCube<out_eT, T1, mtop_conv_to>, i, backend, arg_name_prefix, arg_names > : public bounds_check_str< T1, i, backend, arg_name_prefix, arg_names > { };



//
// mtop_rel_core: just the bounds check of the argument
//

template<typename out_eT, typename T1, typename mtop_type, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct bounds_check_str< mtOp<out_eT, T1, mtop_rel_core<mtop_type> >, i, backend, arg_name_prefix, arg_names > : public bounds_check_str< T1, i, backend, concat_str< arg_name_prefix, arg_prefix_name >, arg_names > { };

template<typename out_eT, typename T1, typename mtop_type, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct bounds_check_str< mtOpCube<out_eT, T1, mtop_rel_core<mtop_type> >, i, backend, arg_name_prefix, arg_names > : public bounds_check_str< T1, i, backend, concat_str< arg_name_prefix, arg_prefix_name >, arg_names > { };



//
// mtGlue: just check bounds in the first object
//

template<typename out_eT, typename T1, typename T2, typename mtglue_type, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct bounds_check_str< mtGlue<out_eT, T1, T2, mtglue_mixed_core<mtglue_type> >, i, backend, arg_name_prefix, arg_names > : public bounds_check_str< T1, i, backend, concat_str< arg_name_prefix, eglue_arg1_name >, arg_names > { };

template<typename out_eT, typename T1, typename T2, typename mtglue_type, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct bounds_check_str< mtGlueCube<out_eT, T1, T2, mtglue_mixed_core<mtglue_type> >, i, backend, arg_name_prefix, arg_names > : public bounds_check_str< T1, i, backend, concat_str< arg_name_prefix, eglue_arg1_name >, arg_names > { };

template<typename out_eT, typename T1, typename T2, typename mtglue_type, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct bounds_check_str< mtGlue<out_eT, T1, T2, mtglue_rel_core<mtglue_type> >, i, backend, arg_name_prefix, arg_names > : public bounds_check_str< T1, i, backend, concat_str< arg_name_prefix, eglue_arg1_name >, arg_names > { };

template<typename out_eT, typename T1, typename T2, typename mtglue_type, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct bounds_check_str< mtGlueCube<out_eT, T1, T2, mtglue_rel_core<mtglue_type> >, i, backend, arg_name_prefix, arg_names > : public bounds_check_str< T1, i, backend, concat_str< arg_name_prefix, eglue_arg1_name >, arg_names > { };



//
// Op<T1, op_htrans>: swap row and column names in T1 bounds check
//

template<typename arg_names>
struct trans_arg_names
  {
  using arg1 = typename arg_names::arg2;
  using arg2 = typename arg_names::arg1;
  using arg3 = typename arg_names::arg3;
  };

template<typename T1, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct bounds_check_str< Op<T1, op_htrans>, i, backend, arg_name_prefix, arg_names > : public bounds_check_str< T1, i, backend, arg_name_prefix, trans_arg_names<arg_names> > { };

template<typename T1, size_t i, coot_backend_t backend, typename arg_name_prefix, typename arg_names>
struct bounds_check_str< Op<T1, op_strans>, i, backend, arg_name_prefix, arg_names > : public bounds_check_str< T1, i, backend, arg_name_prefix, trans_arg_names<arg_names> > { };
