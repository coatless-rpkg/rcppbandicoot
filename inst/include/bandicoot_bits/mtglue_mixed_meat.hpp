// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2008-2016 Conrad Sanderson (https://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
// Copyright 2025      Ryan Curtin (https://ratml.org)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------


//
// mtglue_mixed_times
//



// matrix multiplication with different element types
template<typename out_eT, typename T1, typename T2>
inline
void
mtglue_mixed_times::apply(Mat<out_eT>& out, const mtGlue<out_eT, T1, T2, mtglue_mixed_times>& X)
  {
  coot_debug_sigprint();

  typedef typename T1::elem_type in_eT1;
  typedef typename T2::elem_type in_eT2;

  // For mixed matrix multiplication, we have to convert both results to the output type.
  const partial_unwrap<mtOp<out_eT, T1, mtop_conv_to>> tmp1(mtOp<out_eT, T1, mtop_conv_to>(X.A));
  const partial_unwrap<mtOp<out_eT, T2, mtop_conv_to>> tmp2(mtOp<out_eT, T2, mtop_conv_to>(X.B));

  typedef typename partial_unwrap<mtOp<out_eT, T1, mtop_conv_to>>::stored_type PT1;
  typedef typename partial_unwrap<mtOp<out_eT, T2, mtop_conv_to>>::stored_type PT2;

  const PT1& A = tmp1.M;
  const PT2& B = tmp2.M;

  const bool use_alpha = partial_unwrap<T1>::do_times || partial_unwrap<T2>::do_times;
  const out_eT   alpha = use_alpha ? (tmp1.get_val() * tmp2.get_val()) : out_eT(0);

  alias_wrapper<Mat<out_eT>, PT1, PT2> W(out, A, B);
  glue_times::apply
    <
    out_eT,
    PT1,
    PT2,
    partial_unwrap<mtOp<out_eT, T1, mtop_conv_to>>::do_trans,
    partial_unwrap<mtOp<out_eT, T2, mtop_conv_to>>::do_trans,
    (partial_unwrap<mtOp<out_eT, T1, mtop_conv_to>>::do_times || partial_unwrap<mtOp<out_eT, T2, mtop_conv_to>>::do_times)
    >
    (W.use, A, B, alpha);
  }



template<typename out_eT, typename T1, typename T2>
inline
uword
mtglue_mixed_times::compute_n_rows(const mtGlue<out_eT, T1, T2, mtglue_mixed_times>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_cols);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  return A_n_rows;
  }



template<typename out_eT, typename T1, typename T2>
inline
uword
mtglue_mixed_times::compute_n_cols(const mtGlue<out_eT, T1, T2, mtglue_mixed_times>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  return A_n_cols;
  }



//
// mtglue_mixed_base
//



template<typename mtglue_type>
template<typename out_eT, typename T1, typename T2>
inline
void
mtglue_mixed_core<mtglue_type>::apply(Mat<out_eT>& out, const mtGlue<out_eT, T1, T2, mtglue_mixed_core<mtglue_type> >& x)
  {
  coot_debug_sigprint();

  const Proxy<mtGlue<out_eT, T1, T2, mtglue_mixed_core<mtglue_type>>> P_in(x);
  const inexact_alias_wrapper<Mat<out_eT>, Proxy<mtGlue<out_eT, T1, T2, mtglue_mixed_core<mtglue_type>>>> A(out, P_in);

  A.use.set_size(P_in.get_n_rows(), P_in.get_n_cols());
  coot_rt_t::copy(make_proxy(A.use), P_in);
  }



template<typename mtglue_type>
template<typename out_eT, typename T1, typename T2>
inline
void
mtglue_mixed_core<mtglue_type>::apply(Cube<out_eT>& out, const mtGlueCube<out_eT, T1, T2, mtglue_mixed_core<mtglue_type> >& x)
  {
  coot_debug_sigprint();

  const Proxy<mtGlueCube<out_eT, T1, T2, mtglue_mixed_core<mtglue_type>>> P_in(x);
  const inexact_alias_wrapper<Cube<out_eT>, Proxy<mtGlueCube<out_eT, T1, T2, mtglue_mixed_core<mtglue_type>>>> A(out, P_in);

  A.use.set_size(P_in.get_n_rows(), P_in.get_n_cols(), P_in.get_n_slices());
  coot_rt_t::copy(make_proxy(A.use), P_in);
  }




template<typename mtglue_type>
template<typename out_eT, typename T1, typename T2>
inline
uword
mtglue_mixed_core<mtglue_type>::compute_n_rows(const mtGlue<out_eT, T1, T2, mtglue_mixed_core<mtglue_type>>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_cols);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  return A_n_rows;
  }



template<typename mtglue_type>
template<typename out_eT, typename T1, typename T2>
inline
uword
mtglue_mixed_core<mtglue_type>::compute_n_cols(const mtGlue<out_eT, T1, T2, mtglue_mixed_core<mtglue_type>>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);

  return A_n_cols;
  }



template<typename mtglue_type>
template<typename out_eT, typename T1, typename T2>
inline
uword
mtglue_mixed_core<mtglue_type>::compute_n_rows(const mtGlueCube<out_eT, T1, T2, mtglue_mixed_core<mtglue_type>>& glue, const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices)
  {
  coot_ignore(glue);
  coot_ignore(A_n_cols);
  coot_ignore(A_n_slices);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);
  coot_ignore(B_n_slices);

  return A_n_rows;
  }



template<typename mtglue_type>
template<typename out_eT, typename T1, typename T2>
inline
uword
mtglue_mixed_core<mtglue_type>::compute_n_cols(const mtGlueCube<out_eT, T1, T2, mtglue_mixed_core<mtglue_type>>& glue, const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(A_n_slices);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);
  coot_ignore(B_n_slices);

  return A_n_cols;
  }



template<typename mtglue_type>
template<typename out_eT, typename T1, typename T2>
inline
uword
mtglue_mixed_core<mtglue_type>::compute_n_slices(const mtGlueCube<out_eT, T1, T2, mtglue_mixed_core<mtglue_type>>& glue, const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices)
  {
  coot_ignore(glue);
  coot_ignore(A_n_rows);
  coot_ignore(A_n_cols);
  coot_ignore(B_n_rows);
  coot_ignore(B_n_cols);
  coot_ignore(B_n_slices);

  return A_n_slices;
  }
