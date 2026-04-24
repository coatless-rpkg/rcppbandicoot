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



template<typename name, typename prefix = empty_str>
using uword_arg = concat_str
  <
  uword_arg_prefix,
  prefix,
  name,
  close_paren
  >;



template<typename name, typename eT, coot_backend_t backend, typename prefix = empty_str>
using et_arg = concat_str
  <
  elem_type_str<eT, backend>,
  et_arg_name,
  prefix,
  name,
  close_paren
  >;



//
// macro_kernel_params provides a compile-time string definition of a macro function COOT_OBJECT_i(name)
// that is used inside of kernels to provide the kernel parameters necessary for a given type T
// (where e.g. T is a Mat<eT>, subview<eT>, etc.)
//
// Each specialization must provide a function len() and str() (returning a char_array<len() + 1>);
// to add a new specialization, adapt the others.
//
// It is expected that:
//  * any 1D object will have the member name_n_elem defined
//  * any 2D object will have the members name_n_rows and name_n_cols defined
//  * any 3D object will have the members name_n_rows, name_n_cols, and name_n_slices defined
//

//
// Transform an entire kernel_param_tuple to push constant types.
//

// This should be overloaded for each type as a std::tuple<> of arguments.
template<typename T, size_t i, coot_backend_t backend, typename arg_name_prefix = empty_str, typename sep = space_sep >
struct kernel_param_str;

//
// `macro_kernel_params` expands `COOT_OBJECT_i(name)` to a separator-joined
// argument list for a given type T.  For CUDA/OpenCL this is the kernel
// function-argument list (separator ", ").  For Vulkan it is the list of
// push-constant-struct fields (separator "; "); buffer declarations are
// emitted separately via `macro_kernel_push_params`.
//

template<typename T, size_t i, coot_backend_t backend, typename arg_name_prefix = empty_str >
struct macro_kernel_params : public nested_concat_str
  <
  concat_str
    <
    typename macro_defn<backend>::prefix,                         // -D
    coot_object_prefix,                                           // COOT_OBJECT_
    index_to_str<i>,                                              // <i>
    coot_name_arg                                                 // (name)=
    >,
  kernel_param_str<T, i, backend, arg_name_prefix, space_sep >, // <list of arguments>
  concat_str
    <
    typename macro_defn<backend>::suffix
    >
  > { };



template<typename T, size_t i, typename arg_name_prefix>
struct macro_kernel_params<T, i, VULKAN_BACKEND, arg_name_prefix> : public nested_concat_str
  <
  concat_str
    <
    typename macro_defn<VULKAN_BACKEND>::prefix,                                  // -D
    coot_object_prefix,                                                           // COOT_OBJECT_
    index_to_str<i>,                                                              // <i>
    coot_name_arg                                                                 // (name)=
    >,
  kernel_param_str<T, i, VULKAN_BACKEND, arg_name_prefix, space_semicolon>,       // <list of push-constant fields>
  concat_str
    <
    typename macro_defn<VULKAN_BACKEND>::suffix
    >
  > { };



//
// Mat<eT>
// 3 parameters: pointer, n_rows, n_cols
//

template<typename eT, size_t i, coot_backend_t backend, typename arg_name_prefix, typename sep>
struct kernel_param_str< Mat<eT>, i, backend, arg_name_prefix, sep > : public concat_str
  <
  coot_pointer_arg<eT, backend, arg_name_prefix>,   // eT* name_ptr
  sep, uword_arg<n_rows_name, arg_name_prefix>,     // UWORD name_n_rows
  sep, uword_arg<n_cols_name, arg_name_prefix>      // UWORD name_n_cols
  > { };



//
// Col<eT>
// 2 parameters: pointer, n_elem
//

template<typename eT, size_t i, coot_backend_t backend, typename arg_name_prefix, typename sep>
struct kernel_param_str< Col<eT>, i, backend, arg_name_prefix, sep > : public concat_str
  <
  coot_pointer_arg<eT, backend, arg_name_prefix>, // eT* name_ptr
  sep, uword_arg<n_elem_name, arg_name_prefix>    // const UWORD name_n_elem
  > { };



//
// Row<eT>
// 2 parameters: pointer, n_elem
//

template<typename eT, size_t i, coot_backend_t backend, typename arg_name_prefix, typename sep>
struct kernel_param_str< Row<eT>, i, backend, arg_name_prefix, sep > : public concat_str
  <
  coot_pointer_arg<eT, backend, arg_name_prefix>, // eT* name_ptr
  sep, uword_arg<n_elem_name, arg_name_prefix>    // UWORD name_n_elem
  > { };



//
// subview<eT>
// 4 parameters: pointer, n_rows, n_cols, M_n_rows
//

template<typename eT, size_t i, coot_backend_t backend, typename arg_name_prefix, typename sep>
struct kernel_param_str< subview<eT>, i, backend, arg_name_prefix, sep > : public concat_str
  <
  coot_pointer_arg<eT, backend, arg_name_prefix>, // eT* name_ptr
  sep, uword_arg<n_rows_name, arg_name_prefix>,   // UWORD name_n_rows
  sep, uword_arg<n_cols_name, arg_name_prefix>,   // UWORD name_n_cols
  sep, uword_arg<M_n_rows_name, arg_name_prefix>  // UWORD name_M_n_rows
  > { };



//
// subview_col<eT>: same as subview
//

template<typename eT, size_t i, coot_backend_t backend, typename arg_name_prefix, typename sep>
struct kernel_param_str< subview_col<eT>, i, backend, arg_name_prefix, sep > : public kernel_param_str< Col<eT>, i, backend, arg_name_prefix, sep > { };



//
// subview_row<eT>
// 3 parameters: pointer, n_elem, M_n_rows
//

template<typename eT, size_t i, coot_backend_t backend, typename arg_name_prefix, typename sep>
struct kernel_param_str< subview_row<eT>, i, backend, arg_name_prefix, sep > : public concat_str
  <
  coot_pointer_arg<eT, backend, arg_name_prefix>, // eT* name_ptr
  sep, uword_arg<n_elem_name, arg_name_prefix>,   // UWORD name_n_elem
  sep, uword_arg<incr_name, arg_name_prefix>      // UWORD name_incr
  > { };



//
// diagview<eT>:
// 3 parameters: pointer, n_elem, incr
//

template<typename eT, size_t i, coot_backend_t backend, typename arg_name_prefix, typename sep>
struct kernel_param_str< diagview<eT>, i, backend, arg_name_prefix, sep > : public concat_str
  <
  coot_pointer_arg<eT, backend, arg_name_prefix>, // eT* name_ptr
  sep, uword_arg<n_elem_name, arg_name_prefix>,   // UWORD name_n_elem
  sep, uword_arg<incr_name, arg_name_prefix>      // UWORD name_incr
  > { };



//
// subview_elem1<eT, T1>:
// 2 (or more) parameters: pointer, parameters for T1 (including bounds)
//

template<typename T>
struct elem_vectorised_arg_type
  {
  typedef T result;
  };

template<typename eT>
struct elem_vectorised_arg_type< Mat<eT> >
  {
  typedef Col<eT> result;
  };

template<typename eT>
struct elem_vectorised_arg_type< subview<eT> >
  {
  typedef subview_row<eT> result;
  };



template<typename eT, typename T1, size_t i, coot_backend_t backend, typename arg_name_prefix, typename sep>
struct kernel_param_str< subview_elem1< eT, T1 >, i, backend, arg_name_prefix, sep > : public nested_concat_str
  <
  concat_str
    <
    coot_pointer_arg<eT, backend, arg_name_prefix>, // eT* name_ptr
    sep
    >,
  kernel_param_str< typename elem_vectorised_arg_type<T1>::result, i, backend, concat_str< arg_name_prefix, elem1_index_prefix >, sep >
  > { };



//
// subview_elem2<eT, subview_elem2_both<eT, T1, T2>>:
// 6 (or more) parameters: pointer, n_rows, n_cols, src_n_rows, src_n_cols, parameters for T1 (including bounds), parameters for T2 (including bounds)
// note that n_rows and n_cols are always the size of the resulting matrix,
// but src_n_rows and src_n_cols are the size of the original matrix
//

template<typename eT, typename T1, typename T2, size_t i, coot_backend_t backend, typename arg_name_prefix, typename sep>
struct kernel_param_str< subview_elem2< eT, subview_elem2_both<eT, T1, T2> >, i, backend, arg_name_prefix, sep > : public nested_concat_str
  <
  concat_str
    <
    coot_pointer_arg<eT, backend, arg_name_prefix>,                         // eT* name_ptr
    sep,
    uword_arg<n_rows_name, arg_name_prefix>,                                // UWORD name_n_rows
    sep,
    uword_arg<n_cols_name, arg_name_prefix>,                                // UWORD name_n_cols
    sep,
    uword_arg<n_rows_name, concat_str<arg_name_prefix, elem2_src_prefix> >, // UWORD name_src_n_rows
    sep,
    uword_arg<n_cols_name, concat_str<arg_name_prefix, elem2_src_prefix> >, // UWORD name_src_n_cols
    sep
    >,
  kernel_param_str< typename elem_vectorised_arg_type<T1>::result, i, backend, concat_str< arg_name_prefix, elem2_row_index_prefix >, sep >,
  concat_str< sep >,
  kernel_param_str< typename elem_vectorised_arg_type<T2>::result, i, backend, concat_str< arg_name_prefix, elem2_col_index_prefix >, sep >
  > { };



//
// subview_elem2<eT, subview_elem2_all_cols<eT, T1>>:
// 6 (or more) parameters: pointer, n_rows, n_cols, src_n_rows, parameters for T1 (including bounds)
//

template<typename eT, typename T1, size_t i, coot_backend_t backend, typename arg_name_prefix, typename sep>
struct kernel_param_str< subview_elem2< eT, subview_elem2_all_cols<eT, T1> >, i, backend, arg_name_prefix, sep > : public nested_concat_str
  <
  concat_str
    <
    coot_pointer_arg<eT, backend, arg_name_prefix>,                         // eT* name_ptr
    sep,
    uword_arg<n_rows_name, arg_name_prefix>,                                // UWORD name_n_rows
    sep,
    uword_arg<n_cols_name, arg_name_prefix>,                                // UWORD name_n_cols
    sep,
    uword_arg<n_rows_name, concat_str<arg_name_prefix, elem2_src_prefix> >, // UWORD name_src_n_rows
    sep
    >,
  kernel_param_str< typename elem_vectorised_arg_type<T1>::result, i, backend, concat_str< arg_name_prefix, elem2_row_index_prefix >, sep >
  > { };



//
// subview_elem2<eT, subview_elem2_all_rows<eT, T2>>:
// 4 (or more) parameters: pointer, n_rows, n_cols, parameters for T2 (including bounds)
//
// Note that src_n_rows and src_n_cols aren't actually needed for bounds checks
// or element access.
//

template<typename eT, typename T2, size_t i, coot_backend_t backend, typename arg_name_prefix, typename sep>
struct kernel_param_str< subview_elem2< eT, subview_elem2_all_rows<eT, T2> >, i, backend, arg_name_prefix, sep > : public nested_concat_str
  <
  concat_str
    <
    coot_pointer_arg<eT, backend, arg_name_prefix>, // eT* name_ptr
    sep,
    uword_arg<n_rows_name, arg_name_prefix>,        // UWORD name_n_rows
    sep,
    uword_arg<n_cols_name, arg_name_prefix>,        // UWORD name_n_cols
    sep
    >,
  kernel_param_str< typename elem_vectorised_arg_type<T2>::result, i, backend, concat_str< arg_name_prefix, elem2_col_index_prefix >, sep >
  > { };



//
// Cube<eT>
// 4 parameters: pointer, n_rows, n_cols, n_slices
//

template<typename eT, size_t i, coot_backend_t backend, typename arg_name_prefix, typename sep>
struct kernel_param_str< Cube<eT>, i, backend, arg_name_prefix, sep > : public concat_str
  <
  coot_pointer_arg<eT, backend, arg_name_prefix>, // eT* name_ptr
  sep, uword_arg<n_rows_name, arg_name_prefix>,   // UWORD name_n_rows
  sep, uword_arg<n_cols_name, arg_name_prefix>,   // UWORD name_n_cols
  sep, uword_arg<n_slices_name, arg_name_prefix>  // UWORD name_n_slices
  > { };



//
// subview_cube<eT>
// 6 parameters: pointer, n_rows, n_cols, n_slices, M_n_rows, M_n_elem_slice
//

template<typename eT, size_t i, coot_backend_t backend, typename arg_name_prefix, typename sep>
struct kernel_param_str< subview_cube<eT>, i, backend, arg_name_prefix, sep > : public concat_str
  <
  coot_pointer_arg<eT, backend, arg_name_prefix>,      // eT* name_ptr
  sep, uword_arg<n_rows_name, arg_name_prefix>,        // UWORD name_n_rows
  sep, uword_arg<n_cols_name, arg_name_prefix>,        // UWORD name_n_cols
  sep, uword_arg<n_slices_name, arg_name_prefix>,      // UWORD name_n_slices
  sep, uword_arg<M_n_rows_name, arg_name_prefix>,      // UWORD name_M_n_rows
  sep, uword_arg<M_n_elem_slice_name, arg_name_prefix> // UWORD name_M_n_elem_slice
  > { };



//
// ProxyColCast<T1>: the T1 should be reinterpreted as a vector;
// no extra parameters are needed for this, so just use the T1's
//

template<typename T1, size_t src_dims, bool proxy_uses_ref, size_t i, coot_backend_t backend, typename arg_name_prefix, typename sep>
struct kernel_param_str< ProxyColCast<T1, src_dims, proxy_uses_ref>, i, backend, arg_name_prefix, sep > : public kernel_param_str< T1, i, backend, arg_name_prefix, sep > { };



//
// ProxyMatCast<T1>: the T1 should be reinterpreted as a matrix;
// we must also specify the target number of rows and columns
//

template<typename arg_name_prefix>
using target_prefix = concat_str<arg_name_prefix, target_name>;



template<typename T1, size_t src_dims, bool proxy_uses_ref, size_t i, coot_backend_t backend, typename arg_name_prefix, typename sep>
struct kernel_param_str< ProxyMatCast<T1, src_dims, proxy_uses_ref>, i, backend, arg_name_prefix, sep > : public nested_concat_str
  <
  kernel_param_str<T1, i, backend, arg_name_prefix, sep>,
  concat_str
    <
    sep,
    uword_arg<n_rows_name, target_prefix<arg_name_prefix>>, // UWORD name_target_n_rows
    sep,
    uword_arg<n_cols_name, target_prefix<arg_name_prefix>>  // UWORD name_target_n_cols
    >
  > { };



//
// ProxyCubeCast<T1>: the T1 should be reinterpreted as a cube;
// we must also specify the target number of rows, columns, and slices
//

template<typename T1, size_t src_dims, bool proxy_uses_ref, size_t i, coot_backend_t backend, typename arg_name_prefix, typename sep>
struct kernel_param_str< ProxyCubeCast<T1, src_dims, proxy_uses_ref>, i, backend, arg_name_prefix, sep > : public nested_concat_str
  <
  kernel_param_str<T1, i, backend, arg_name_prefix, sep>,
  concat_str
    <
    sep,
    uword_arg<n_rows_name, target_prefix<arg_name_prefix>>,  // UWORD name_target_n_rows
    sep,
    uword_arg<n_cols_name, target_prefix<arg_name_prefix>>,  // UWORD name_target_n_cols
    sep,
    uword_arg<n_slices_name, target_prefix<arg_name_prefix>> // UWORD name_target_n_slices
    >
  > { };



//
// eOp with no extra arguments
//

template<typename T1, typename eop_type, size_t num_args, size_t i, coot_backend_t backend, typename arg_name_prefix, typename sep>
struct kernel_param_str_eop_helper : public kernel_param_str< T1, i, backend, arg_name_prefix, sep > { };



//
// eOp with one extra argument
//

template<typename T1, typename eop_type, size_t i, coot_backend_t backend, typename arg_name_prefix, typename sep>
struct kernel_param_str_eop_helper<T1, eop_type, 1, i, backend, arg_name_prefix, sep> : public nested_concat_str
  <
  kernel_param_str< T1, i, backend, concat_str< arg_name_prefix, arg_prefix_name >, sep >,
  concat_str
    <
    sep,
    et_arg< eop_aux1_name, typename T1::elem_type, backend, arg_name_prefix >
    >
  > { };



//
// eOp with two extra arguments
//

template<typename T1, typename eop_type, size_t i, coot_backend_t backend, typename arg_name_prefix, typename sep>
struct kernel_param_str_eop_helper<T1, eop_type, 2, i, backend, arg_name_prefix, sep> : public nested_concat_str
  <
  kernel_param_str< T1, i, backend, concat_str< arg_name_prefix, arg_prefix_name >, sep >,
  concat_str
    <
    sep,
    et_arg< eop_aux1_name, typename T1::elem_type, backend, arg_name_prefix >,
    sep,
    et_arg< eop_aux2_name, typename T1::elem_type, backend, arg_name_prefix >
    >
  > { };



//
// dispatch to the correct eOp overload given the number of arguments the eOp takes
//

template<typename T1, typename eop_type, size_t i, coot_backend_t backend, typename arg_name_prefix, typename sep>
struct kernel_param_str< eOp<T1, eop_type>, i, backend, arg_name_prefix, sep > : public kernel_param_str_eop_helper< T1, eop_type, eop_type::num_args, i, backend, arg_name_prefix, sep > { };

template<typename T1, typename eop_type, size_t i, coot_backend_t backend, typename arg_name_prefix, typename sep>
struct kernel_param_str< eOpCube<T1, eop_type>, i, backend, arg_name_prefix, sep > : public kernel_param_str_eop_helper< T1, eop_type, eop_type::num_args, i, backend, arg_name_prefix, sep > { };



//
// eGlue parameters are just the parameters of the two arguments, prefixed with _a or _b depending on the argument
//

template<typename T1, typename T2, typename eglue_type, size_t i, coot_backend_t backend, typename arg_name_prefix, typename sep>
struct kernel_param_str< eGlue<T1, T2, eglue_type>, i, backend, arg_name_prefix, sep > : public nested_concat_str
  <
  kernel_param_str< T1, i, backend, concat_str< arg_name_prefix, eglue_arg1_name >, sep >,
  concat_str< sep >,
  kernel_param_str< T2, i, backend, concat_str< arg_name_prefix, eglue_arg2_name >, sep >
  > { };

template<typename T1, typename T2, typename eglue_type, size_t i, coot_backend_t backend, typename arg_name_prefix, typename sep>
struct kernel_param_str< eGlueCube<T1, T2, eglue_type>, i, backend, arg_name_prefix, sep > : public nested_concat_str
  <
  kernel_param_str< T1, i, backend, concat_str< arg_name_prefix, eglue_arg1_name >, sep >,
  concat_str< sep >,
  kernel_param_str< T2, i, backend, concat_str< arg_name_prefix, eglue_arg2_name >, sep >
  > { };



//
// mtOp<out_eT, T1, mtop_conv_to> parameters are just the parameters of the T1
//

template<typename out_eT, typename T1, size_t i, coot_backend_t backend, typename arg_name_prefix, typename sep>
struct kernel_param_str< mtOp<out_eT, T1, mtop_conv_to>, i, backend, arg_name_prefix, sep > : public kernel_param_str< T1, i, backend, arg_name_prefix, sep > { };


template<typename out_eT, typename T1, size_t i, coot_backend_t backend, typename arg_name_prefix, typename sep>
struct kernel_param_str< mtOpCube<out_eT, T1, mtop_conv_to>, i, backend, arg_name_prefix, sep > : public kernel_param_str< T1, i, backend, arg_name_prefix, sep > { };



//
// mtOp<out_eT, T1, mtop_rel_core<mtop_type> > parameters are just the parameters of the T1, plus an _arg parameter for the scalar argument
//

template<typename out_eT, typename T1, typename mtop_type, size_t i, coot_backend_t backend, typename arg_name_prefix, typename sep>
struct kernel_param_str< mtOp<out_eT, T1, mtop_rel_core<mtop_type> >, i, backend, arg_name_prefix, sep > : public nested_concat_str
  <
  kernel_param_str< T1, i, backend, concat_str< arg_name_prefix, arg_prefix_name >, sep >,
  concat_str
    <
    sep,
    et_arg< eop_aux1_name, typename T1::elem_type, backend, arg_name_prefix >
    >
  > { };

template<typename out_eT, typename T1, typename mtop_type, size_t i, coot_backend_t backend, typename arg_name_prefix, typename sep>
struct kernel_param_str< mtOpCube<out_eT, T1, mtop_rel_core<mtop_type> >, i, backend, arg_name_prefix, sep > : public kernel_param_str< T1, i, backend, arg_name_prefix, sep > { };



//
// mtGlue parameters are just the parameters of the two arguments, prefixed with _a or _b depending on the argument
//

template<typename out_eT, typename T1, typename T2, typename mtglue_type, size_t i, coot_backend_t backend, typename arg_name_prefix, typename sep>
struct kernel_param_str< mtGlue<out_eT, T1, T2, mtglue_mixed_core<mtglue_type> >, i, backend, arg_name_prefix, sep > : public nested_concat_str
  <
  kernel_param_str< T1, i, backend, concat_str< arg_name_prefix, eglue_arg1_name >, sep >,
  concat_str < sep >,
  kernel_param_str< T2, i, backend, concat_str< arg_name_prefix, eglue_arg2_name >, sep >
  > { };

template<typename out_eT, typename T1, typename T2, typename mtglue_type, size_t i, coot_backend_t backend, typename arg_name_prefix, typename sep>
struct kernel_param_str< mtGlueCube<out_eT, T1, T2, mtglue_mixed_core<mtglue_type> >, i, backend, arg_name_prefix, sep > : public nested_concat_str
  <
  kernel_param_str< T1, i, backend, concat_str< arg_name_prefix, eglue_arg1_name >, sep >,
  concat_str< sep >,
  kernel_param_str< T2, i, backend, concat_str< arg_name_prefix, eglue_arg2_name >, sep >
  > { };

template<typename out_eT, typename T1, typename T2, typename mtglue_type, size_t i, coot_backend_t backend, typename arg_name_prefix, typename sep>
struct kernel_param_str< mtGlue<out_eT, T1, T2, mtglue_rel_core<mtglue_type> >, i, backend, arg_name_prefix, sep > : public nested_concat_str
  <
  kernel_param_str< T1, i, backend, concat_str< arg_name_prefix, eglue_arg1_name >, sep >,
  concat_str< sep >,
  kernel_param_str< T2, i, backend, concat_str< arg_name_prefix, eglue_arg2_name >, sep >
  > { };

template<typename out_eT, typename T1, typename T2, typename mtglue_type, size_t i, coot_backend_t backend, typename arg_name_prefix, typename sep>
struct kernel_param_str< mtGlueCube<out_eT, T1, T2, mtglue_rel_core<mtglue_type> >, i, backend, arg_name_prefix, sep > : public nested_concat_str
  <
  kernel_param_str< T1, i, backend, concat_str< arg_name_prefix, eglue_arg1_name >, sep >,
  concat_str< sep >,
  kernel_param_str< T2, i, backend, concat_str< arg_name_prefix, eglue_arg2_name >, sep >
  > { };



//
// Op<T1, op_htrans>: no extra arguments needed
//

template<typename T1, size_t i, coot_backend_t backend, typename arg_name_prefix, typename sep>
struct kernel_param_str< Op<T1, op_htrans>, i, backend, arg_name_prefix, sep > : public kernel_param_str< T1, i, backend, arg_name_prefix, sep > { };

template<typename T1, size_t i, coot_backend_t backend, typename arg_name_prefix, typename sep>
struct kernel_param_str< Op<T1, op_strans>, i, backend, arg_name_prefix, sep > : public kernel_param_str< T1, i, backend, arg_name_prefix, sep > { };

