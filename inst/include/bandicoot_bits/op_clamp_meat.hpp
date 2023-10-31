// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2023      Ryan Curtin (http://ratml.org)
// Copyright 2021      Marcus Edel (http://kurg.org)
// Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
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



template<typename out_eT, typename T1>
inline
void
op_clamp::apply(Mat<out_eT>& out, const Op<T1, op_clamp>& in)
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT;

  const eT min_val = in.aux;
  const eT max_val = in.aux_b;

  coot_debug_check( (min_val > max_val), "clamp(): min_val must be less than max_val" );

  const unwrap<T1> U(in.m);
  if (U.is_alias(out))
    {
    Mat<eT> tmp;
    tmp.set_size(U.M.n_rows, U.M.n_cols);
    coot_rt_t::clamp(tmp.get_dev_mem(false), U.get_dev_mem(false),
                     min_val, max_val,
                     tmp.n_rows, tmp.n_cols,
                     0, 0, tmp.n_rows,
                     U.get_row_offset(), U.get_col_offset(), U.get_M_n_rows());
    steal_or_copy_mem(out, tmp);
    }
  else
    {
    out.set_size(U.M.n_rows, U.M.n_cols);
    coot_rt_t::clamp(out.get_dev_mem(false), U.get_dev_mem(false),
                     min_val, max_val,
                     out.n_rows, out.n_cols,
                     0, 0, out.n_rows,
                     U.get_row_offset(), U.get_col_offset(), U.get_M_n_rows());
    }
  }



template<typename T1>
inline
uword
op_clamp::compute_n_rows(const Op<T1, op_clamp>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_cols);
  return in_n_rows;
  }



template<typename T1>
inline
uword
op_clamp::compute_n_cols(const Op<T1, op_clamp>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);
  return in_n_cols;
  }
