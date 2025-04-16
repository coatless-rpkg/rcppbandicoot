// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
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



template<typename parent, unsigned int mode>
inline
subview_each_common<parent, mode>::subview_each_common(const parent& in_P)
  : P(in_P)
  {
  coot_extra_debug_sigprint();
  }



template<typename parent, unsigned int mode>
coot_inline
const Mat<typename parent::elem_type>&
subview_each_common<parent, mode>::get_mat_ref_helper(const Mat<typename parent::elem_type>& X) const
  {
  return X;
  }



template<typename parent, unsigned int mode>
coot_inline
const Mat<typename parent::elem_type>&
subview_each_common<parent, mode>::get_mat_ref_helper(const subview<typename parent::elem_type>& X) const
  {
  return X.m;
  }



template<typename parent, unsigned int mode>
coot_inline
const Mat<typename parent::elem_type>&
subview_each_common<parent, mode>::get_mat_ref() const
  {
  return get_mat_ref_helper(P);
  }



template<typename parent, unsigned int mode>
template<typename T2>
inline
void
subview_each_common<parent, mode>::check_size(const T2& A) const
  {
  if(mode == 0)
    {
    if( (A.n_rows != P.n_rows) || (A.n_cols != 1) )
      {
      coot_stop_logic_error( incompat_size_string(A) );
      }
    }
  else
    {
    if( (A.n_rows != 1) || (A.n_cols != P.n_cols) )
      {
      coot_stop_logic_error( incompat_size_string(A) );
      }
    }
  }



template<typename parent, unsigned int mode>
template<typename T2>
inline
const std::string
subview_each_common<parent, mode>::incompat_size_string(const T2& A) const
  {
  std::ostringstream tmp;

  if(mode == 0)
    {
    tmp << "each_col(): incompatible size; expected " << P.n_rows << "x1" << ", got " << A.n_rows << 'x' << A.n_cols;
    }
  else
    {
    tmp << "each_row(): incompatible size; expected 1x" << P.n_cols << ", got " << A.n_rows << 'x' << A.n_cols;
    }

  return tmp.str();
  }



//
// subview_each1
//



template<typename parent, unsigned int mode>
inline
subview_each1<parent, mode>::~subview_each1()
  {
  coot_extra_debug_sigprint();
  }



template<typename parent, unsigned int mode>
inline
subview_each1<parent, mode>::subview_each1(const parent& in_P)
  : subview_each_common<parent, mode>::subview_each_common(in_P)
  {
  coot_extra_debug_sigprint();
  }



template<typename parent, unsigned int mode>
template<typename T1>
inline
void
subview_each1<parent, mode>::inplace_op(twoway_kernel_id::enum_id op,
                                        const Base<eT, T1>& in)
  {
  parent& p = access::rw(subview_each_common<parent, mode>::P);

  // This will provide an interface to get the offsets for a subview, if parent is a subview.
  const unwrap<parent> PU(p);
  const no_conv_unwrap<T1> U(in.get_ref());
  alias_wrapper<parent, typename no_conv_unwrap<T1>::stored_type> W(p, U.M);

  const uword copies_per_row = (mode == 1) ? p.n_rows : 1;
  const uword copies_per_col = (mode == 0) ? p.n_cols : 1;

  if (U.M.n_rows == 0 || U.M.n_cols == 0 || copies_per_row == 0 || copies_per_col == 0)
    {
    return;
    }

  subview_each_common<parent, mode>::check_size(U.M);

  // If we are using an alias, make sure it contains the same data as the
  // parent, if the operation is not just overwriting.
  if (op != twoway_kernel_id::broadcast_set && W.using_aux)
    {
    W.aux = p;
    }

  coot_rt_t::broadcast_op(op,
                          W.get_dev_mem(false), W.get_dev_mem(false), U.get_dev_mem(false),
                          U.M.n_rows, U.M.n_cols,
                          copies_per_row, copies_per_col,
                          W.get_row_offset(), W.get_col_offset(), W.get_M_n_rows(),
                          W.get_row_offset(), W.get_col_offset(), W.get_M_n_rows(),
                          U.get_row_offset(), U.get_col_offset(), U.get_M_n_rows());
  }



template<typename parent, unsigned int mode>
template<typename T1>
inline
void
subview_each1<parent, mode>::operator=(const Base<eT, T1>& in)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::broadcast_set, in);
  }




template<typename parent, unsigned int mode>
template<typename T1>
inline
void
subview_each1<parent, mode>::operator+=(const Base<eT, T1>& in)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::broadcast_plus, in);
  }



template<typename parent, unsigned int mode>
template<typename T1>
inline
void
subview_each1<parent, mode>::operator-=(const Base<eT, T1>& in)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::broadcast_minus_post, in);
  }



template<typename parent, unsigned int mode>
template<typename T1>
inline
void
subview_each1<parent, mode>::operator%=(const Base<eT, T1>& in)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::broadcast_schur, in);
  }



template<typename parent, unsigned int mode>
template<typename T1>
inline
void
subview_each1<parent, mode>::operator/=(const Base<eT, T1>& in)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::broadcast_div_post, in);
  }



//
// subview_each2
//



template<typename parent, unsigned int mode, typename TB>
inline
subview_each2<parent, mode, TB>::~subview_each2()
  {
  coot_extra_debug_sigprint();
  }



template<typename parent, unsigned int mode, typename TB>
inline
subview_each2<parent, mode, TB>::subview_each2(const parent& in_P, const Base<uword, TB>& in_indices)
  : subview_each_common<parent, mode>::subview_each_common(in_P)
  , base_indices(in_indices)
  {
  coot_extra_debug_sigprint();
  }



template<typename parent, unsigned int mode, typename TB>
inline
void
subview_each2<parent, mode, TB>::check_indices(const Mat<uword>& indices) const
  {
  if(mode == 0)
    {
    coot_check( ((indices.is_vec() == false) && (indices.is_empty() == false)), "each_col(): list of indices must be a vector" );
    }
  else
    {
    coot_check( ((indices.is_vec() == false) && (indices.is_empty() == false)), "each_row(): list of indices must be a vector" );
    }
  }



template<typename parent, unsigned int mode, typename TB>
template<typename T1>
inline
void
subview_each2<parent, mode, TB>::inplace_op(twoway_kernel_id::enum_id op,
                                            const Base<eT, T1>& in)
  {
  parent& p = access::rw(subview_each_common<parent, mode>::P);

  // This will provide an interface to get the offsets for a subview, if parent is a subview.
  const unwrap<parent> PU(p);
  const unwrap<TB> IU(base_indices.get_ref());
  const no_conv_unwrap<T1> U(in.get_ref());
  alias_wrapper<parent, typename no_conv_unwrap<T1>::stored_type, typename unwrap<TB>::stored_type> W(p, U.M, IU.M);

  const uword copies_per_row = (mode == 1) ? IU.M.n_elem : 1;
  const uword copies_per_col = (mode == 0) ? IU.M.n_elem : 1;

  if (U.M.n_rows == 0 || U.M.n_cols == 0 || copies_per_row == 0 || copies_per_col == 0)
    {
    return;
    }

  subview_each_common<parent, mode>::check_size(U.M);

  // If we need the original values for the operation to be computed correctly,
  // and we are using a temporary result, then copy the parent to the temporary
  // matrix.
  if (W.using_aux)
    {
    W.aux = p;
    }

  coot_rt_t::broadcast_subset_op(op,
                                 W.get_dev_mem(false), W.get_dev_mem(false), U.get_dev_mem(false), IU.get_dev_mem(false),
                                 mode, U.M.n_rows, U.M.n_cols,
                                 copies_per_row, copies_per_col,
                                 W.get_row_offset(), W.get_col_offset(), W.get_M_n_rows(),
                                 W.get_row_offset(), W.get_col_offset(), W.get_M_n_rows(),
                                 U.get_row_offset(), U.get_col_offset(), U.get_M_n_rows(),
                                 IU.get_row_offset() + IU.get_col_offset() * IU.get_M_n_rows(), (IU.M.n_rows == 1) ? IU.get_M_n_rows() : 1);
  }



template<typename parent, unsigned int mode, typename TB>
template<typename T1>
inline
void
subview_each2<parent, mode, TB>::operator=(const Base<eT,T1>& in)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::broadcast_subset_set, in);
  }



template<typename parent, unsigned int mode, typename TB>
template<typename T1>
inline
void
subview_each2<parent, mode, TB>::operator+=(const Base<eT,T1>& in)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::broadcast_subset_plus, in);
  }



template<typename parent, unsigned int mode, typename TB>
template<typename T1>
inline
void
subview_each2<parent, mode, TB>::operator-=(const Base<eT,T1>& in)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::broadcast_subset_minus_post, in);
  }



template<typename parent, unsigned int mode, typename TB>
template<typename T1>
inline
void
subview_each2<parent, mode, TB>::operator%=(const Base<eT,T1>& in)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::broadcast_subset_schur, in);
  }



template<typename parent, unsigned int mode, typename TB>
template<typename T1>
inline
void
subview_each2<parent, mode, TB>::operator/=(const Base<eT,T1>& in)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::broadcast_subset_div_post, in);
  }



//
// subview_each1_aux
//



template<typename parent, unsigned int mode, typename T2>
inline
Mat<typename parent::elem_type>
subview_each1_aux::call_op
  (
  const twoway_kernel_id::enum_id op,
  const subview_each1<parent, mode>& X,
  const Base<typename parent::elem_type, T2>& Y
  )
  {
  // This will provide an interface to get the offsets for a subview, if parent is a subview.
  const unwrap<parent> PU(X.P);
  const no_conv_unwrap<T2> U(Y.get_ref());

  const uword copies_per_row = (mode == 1) ? X.P.n_rows : 1;
  const uword copies_per_col = (mode == 0) ? X.P.n_cols : 1;

  if (U.M.n_rows == 0 || U.M.n_cols == 0 || copies_per_row == 0 || copies_per_col == 0)
    {
    return Mat<typename parent::elem_type>(U.M.n_rows * copies_per_row, U.M.n_cols * copies_per_col);
    }

  X.check_size(U.M);

  Mat<typename parent::elem_type> out;
  out.set_size(U.M.n_rows * copies_per_row, U.M.n_cols * copies_per_col);

  coot_rt_t::broadcast_op(op,
                          out.get_dev_mem(false), PU.get_dev_mem(false), U.get_dev_mem(false),
                          U.M.n_rows, U.M.n_cols,
                          copies_per_row, copies_per_col,
                          0, 0, out.n_rows,
                          PU.get_row_offset(), PU.get_col_offset(), PU.get_M_n_rows(),
                          U.get_row_offset(), U.get_col_offset(), U.get_M_n_rows());

  return out;
  }



template<typename parent, unsigned int mode, typename T2>
inline
Mat<typename parent::elem_type>
subview_each1_aux::operator_plus(const subview_each1<parent, mode>& X, const Base<typename parent::elem_type, T2>& Y)
  {
  coot_extra_debug_sigprint();

  return call_op(twoway_kernel_id::broadcast_plus, X, Y);
  }



template<typename parent, unsigned int mode, typename T2>
inline
Mat<typename parent::elem_type>
subview_each1_aux::operator_minus(const subview_each1<parent, mode>& X, const Base<typename parent::elem_type, T2>& Y)
  {
  coot_extra_debug_sigprint();

  return call_op(twoway_kernel_id::broadcast_minus_post, X, Y);
  }



template<typename T1, typename parent, unsigned int mode>
inline
Mat<typename parent::elem_type>
subview_each1_aux::operator_minus(const Base<typename parent::elem_type, T1>& X, const subview_each1<parent, mode>& Y)
  {
  coot_extra_debug_sigprint();

  return call_op(twoway_kernel_id::broadcast_minus_pre, Y, X);
  }



template<typename parent, unsigned int mode, typename T2>
inline
Mat<typename parent::elem_type>
subview_each1_aux::operator_schur(const subview_each1<parent, mode>& X, const Base<typename parent::elem_type, T2>& Y)
  {
  coot_extra_debug_sigprint();

  return call_op(twoway_kernel_id::broadcast_schur, X, Y);
  }



template<typename parent, unsigned int mode, typename T2>
inline
Mat<typename parent::elem_type>
subview_each1_aux::operator_div(const subview_each1<parent, mode>& X, const Base<typename parent::elem_type, T2>& Y)
  {
  coot_extra_debug_sigprint();

  return call_op(twoway_kernel_id::broadcast_div_post, X, Y);
  }



template<typename T1, typename parent, unsigned int mode>
inline
Mat<typename parent::elem_type>
subview_each1_aux::operator_div(const Base<typename parent::elem_type, T1>& X, const subview_each1<parent, mode>& Y)
  {
  coot_extra_debug_sigprint();

  return call_op(twoway_kernel_id::broadcast_div_pre, Y, X);
  }



template<typename parent, unsigned int mode, typename TB, typename T2>
inline
Mat<typename parent::elem_type>
subview_each2_aux::call_op
  (
  const twoway_kernel_id::enum_id op,
  const subview_each2<parent, mode, TB>& X,
  const Base<typename parent::elem_type, T2>& Y
  )
  {
  // This will provide an interface to get the offsets for a subview, if parent is a subview.
  const unwrap<parent> PU(X.P);
  const unwrap<TB> IU(X.base_indices.get_ref());
  const no_conv_unwrap<T2> U(Y.get_ref());

  const uword copies_per_row = (mode == 1) ? IU.M.n_elem : 1;
  const uword copies_per_col = (mode == 0) ? IU.M.n_elem : 1;

  if (U.M.n_rows == 0 || U.M.n_cols == 0 || copies_per_row == 0 || copies_per_col == 0)
    {
    return Mat<typename parent::elem_type>(U.M.n_rows * copies_per_row, U.M.n_cols * copies_per_col);
    }

  X.check_size(U.M);

  Mat<typename parent::elem_type> out(U.M.n_rows * copies_per_row, U.M.n_cols * copies_per_col);

  coot_rt_t::broadcast_subset_op(op,
                                 out.get_dev_mem(false), PU.get_dev_mem(false), U.get_dev_mem(false), IU.get_dev_mem(false),
                                 mode + 2, U.M.n_rows, U.M.n_cols,
                                 copies_per_row, copies_per_col,
                                 0, 0, out.n_rows,
                                 PU.get_row_offset(), PU.get_col_offset(), PU.get_M_n_rows(),
                                 U.get_row_offset(), U.get_col_offset(), U.get_M_n_rows(),
                                 IU.get_row_offset() + IU.get_col_offset() * IU.get_M_n_rows(), (IU.M.n_rows == 1) ? IU.get_M_n_rows() : 1);

  return out;
  }



template<typename parent, unsigned int mode, typename TB, typename T2>
inline
Mat<typename parent::elem_type>
subview_each2_aux::operator_plus(const subview_each2<parent, mode, TB>& X, const Base<typename parent::elem_type, T2>& Y)
  {
  coot_extra_debug_sigprint();

  return call_op(twoway_kernel_id::broadcast_subset_plus, X, Y);
  }



template<typename parent, unsigned int mode, typename TB, typename T2>
inline
Mat<typename parent::elem_type>
subview_each2_aux::operator_minus(const subview_each2<parent, mode, TB>& X, const Base<typename parent::elem_type, T2>& Y)
  {
  coot_extra_debug_sigprint();

  return call_op(twoway_kernel_id::broadcast_subset_minus_post, X, Y);
  }



template<typename T1, typename parent, unsigned int mode, typename TB>
inline
Mat<typename parent::elem_type>
subview_each2_aux::operator_minus(const Base<typename parent::elem_type, T1>& X, const subview_each2<parent, mode, TB>& Y)
  {
  coot_extra_debug_sigprint();

  return call_op(twoway_kernel_id::broadcast_subset_minus_pre, Y, X);
  }



template<typename parent, unsigned int mode, typename TB, typename T2>
inline
Mat<typename parent::elem_type>
subview_each2_aux::operator_schur(const subview_each2<parent, mode, TB>& X, const Base<typename parent::elem_type, T2>& Y)
  {
  coot_extra_debug_sigprint();

  return call_op(twoway_kernel_id::broadcast_subset_schur, X, Y);
  }



template<typename parent, unsigned int mode, typename TB, typename T2>
inline
Mat<typename parent::elem_type>
subview_each2_aux::operator_div(const subview_each2<parent, mode, TB>& X, const Base<typename parent::elem_type, T2>& Y)
  {
  coot_extra_debug_sigprint();

  return call_op(twoway_kernel_id::broadcast_subset_div_post, X, Y);
  }



template<typename T1, typename parent, unsigned int mode, typename TB>
inline
Mat<typename parent::elem_type>
subview_each2_aux::operator_div(const Base<typename parent::elem_type, T1>& X, const subview_each2<parent, mode, TB>& Y)
  {
  coot_extra_debug_sigprint();

  return call_op(twoway_kernel_id::broadcast_subset_div_pre, Y, X);
  }
