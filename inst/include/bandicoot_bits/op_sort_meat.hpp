// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2023 Ryan Curtin (https://www.ratml.org)
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



template<typename T1>
inline
void
op_sort::apply(Mat<typename T1::elem_type>& out, const Op<T1, op_sort>& in)
  {
  // Apply the inner operation, and then we'll sort in place.
  out = in.m;

  const uword sort_type = in.aux_uword_a;
  const uword dim       = in.aux_uword_b;

  coot_debug_check( (sort_type > 1), "sort(): parameter 'sort_type' must be 0 or 1" );
  coot_debug_check( (dim > 1),       "sort(): parameter 'dim' must be 0 or 1"       );
  // TODO: implement has_nan() and use it here
  //coot_debug_check( (X.has_nan()),   "sort(): detected NaN"                         );

  coot_rt_t::sort(out.get_dev_mem(false), out.n_rows, out.n_cols, sort_type, dim, 0, 0, out.n_rows);
  }



template<typename out_eT, typename T1>
inline
void
op_sort::apply(Mat<out_eT>& out, const Op<T1, op_sort>& in, const typename enable_if<!is_same_type<out_eT, typename T1::elem_type>::value>::result* junk)
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  typedef typename T1::elem_type eT;

  // Apply the sort into a temporary output of the same type.
  Mat<eT> tmp;
  apply(tmp, in);

  // Now perform the final conversion.
  out = conv_to<Mat<out_eT>>::from(tmp);
  }


template<typename T1>
inline
uword
op_sort::compute_n_rows(const Op<T1, op_sort>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(in_n_cols);

  const uword dim = op.aux_uword_b;
  if (dim == 0)
    {
    return std::min(in_n_rows, uword(1)); // either 0 or 1
    }
  else
    {
    return in_n_rows;
    }
  }



template<typename T1>
inline
uword
op_sort::compute_n_cols(const Op<T1, op_sort>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(in_n_rows);

  const uword dim = op.aux_uword_b;
  if (dim == 0)
    {
    return in_n_cols;
    }
  else
    {
    return std::min(in_n_cols, uword(0)); // either 0 or 1
    }
  }



template<typename T1>
inline
void
op_sort_vec::apply(Mat<typename T1::elem_type>& out, const Op<T1, op_sort_vec>& in)
  {
  coot_extra_debug_sigprint();

  // Compute the input, then sort in-place.
  out = in.m;

  const uword sort_type = in.aux_uword_a;
  coot_rt_t::sort_vec(out.get_dev_mem(false), out.n_elem, sort_type);
  }



template<typename out_eT, typename T1>
inline
void
op_sort_vec::apply(Mat<out_eT>& out, const Op<T1, op_sort_vec>& in, const typename enable_if<!is_same_type<out_eT, typename T1::elem_type>::value>::result* junk)
  {
  coot_extra_debug_sigprint();

  // Apply the sort to a temporary vector of the same type, then perform the conversion.
  typedef typename T1::elem_type eT;
  Mat<eT> tmp;
  apply(tmp, in);

  out = conv_to<Mat<out_eT>>::from(tmp);
  }



template<typename T1>
inline
uword
op_sort_vec::compute_n_rows(const Op<T1, op_sort_vec>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_cols);

  return in_n_rows;
  }



template<typename T1>
inline
uword
op_sort_vec::compute_n_cols(const Op<T1, op_sort_vec>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);

  return in_n_cols;
  }
