// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2017-2023 Ryan Curtin (https://www.ratml.org)
// Copyright 2017      Conrad Sanderson (https://conradsanderson.id.au)
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


using std::ios;
using std::size_t;

template<typename elem_type, typename derived> struct Base;
template<typename elem_type, typename derived> struct BaseCube;

template<typename eT> class MatValProxy;

template<typename eT> class Mat;
template<typename eT> class Col;
template<typename eT> class Row;
template<typename eT> class Cube;

template<typename eT> class subview;
template<typename eT> class subview_col;
template<typename eT> class subview_row;
template<typename eT> class subview_cube;

template<typename eT, typename T1>              class subview_elem1;
template<typename eT, typename T1, typename T2> class subview_elem2;

template<typename parent, unsigned int mode>              class subview_each1;
template<typename parent, unsigned int mode, typename TB> class subview_each2;

template<typename eT> class diagview;


class SizeMat;
// class SizeCube;

class coot_empty_class {};

template<const bool, const bool, const bool, const bool> class gemm;
template<const bool, const bool, const bool>             class gemv;


template<                 typename eT, typename gen_type> class  Gen;

template<typename, typename, bool> class Op_traits;

template<typename T1, typename  op_type> class   Op;
template<typename T1, typename eop_type> class  eOp;

template<typename out_eT, typename T1, typename mtop_type> class mtOp;

template<typename T1, typename T2, typename  glue_type> class   Glue;
template<typename T1, typename T2, typename eglue_type> class  eGlue;

template<typename out_eT, typename T1, typename T2, typename mtglue_type> class mtGlue;

template<                 typename eT, typename gen_type> class  GenCube;

template<                 typename T1, typename   op_type> class   OpCube;
template<                 typename T1, typename  eop_type> class  eOpCube;
template<typename out_eT, typename T1, typename mtop_type> class mtOpCube;

template<                 typename T1, typename T2, typename  glue_type> class   GlueCube;
template<                 typename T1, typename T2, typename eglue_type> class  eGlueCube;

template<                 typename T1, typename op_type> class CubeToMatOp;

template<typename T1> class SizeProxy;
template<typename T1> class SizeProxyCube;


struct coot_vec_indicator {};

template<typename eT> struct conv_to;

class op_sum;
class op_inv;
class op_strans;
class op_htrans;
class op_htrans2;
class op_repmat;
class op_resize;
class op_reshape;
class op_vectorise_col;
class op_vectorise_row;
class op_vectorise_all;
class op_vectorise_cube_col;
class op_clamp;
class op_diagmat;
class op_diagmat2;
class op_diagvec;
class op_normalise_vec;
class op_normalise_mat;
class op_mean;
class op_median;
class op_stddev;
class op_var;
class op_range;
class op_cov;
class op_cor;
class op_sort;
class op_sort_vec;
class op_det;
class op_symmat;
class op_pinv;

class op_row_as_mat;
class op_col_as_mat;

class mtop_conv_to;
class mtop_all;
class mtop_rel_lt_pre;
class mtop_rel_lt_post;
class mtop_rel_gt_pre;
class mtop_rel_gt_post;
class mtop_rel_lteq_pre;
class mtop_rel_lteq_post;
class mtop_rel_gteq_pre;
class mtop_rel_gteq_post;
class mtop_rel_eq;
class mtop_rel_noteq;
class mtop_sort_index;
class mtop_find;
class mtop_find_finite;
class mtop_find_nonfinite;
class mtop_find_nan;
class mtop_index_min;
class mtop_index_max;
class mtop_real;
class mtop_imag;

class eop_scalar_plus;
class eop_neg;
class eop_scalar_minus_pre;
class eop_scalar_minus_post;
class eop_scalar_div_pre;
class eop_scalar_div_post;

class eglue_plus;
class eglue_minus;
class eglue_schur;
class eglue_div;

class glue_times;
class glue_times_diag;
class glue_cov;
class glue_cor;
class glue_join_cols;
class glue_join_rows;
class glue_cross;
class glue_conv;
class glue_conv2;
class glue_solve;
class glue_min;
class glue_max;

class glue_mixed_plus;
class glue_mixed_minus;
class glue_mixed_schur;
class glue_mixed_div;
class glue_mixed_times;

class mtglue_rel_lt;
class mtglue_rel_gt;
class mtglue_rel_lteq;
class mtglue_rel_gteq;
class mtglue_rel_eq;
class mtglue_rel_noteq;
class mtglue_rel_and;
class mtglue_rel_or;
