// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2017-2023 Ryan Curtin (https://www.ratml.org)
// Copyright 2008-2017 Conrad Sanderson (https://conradsanderson.id.au)
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
op_strans::apply(Mat<out_eT>& out, const Op<T1, op_strans>& in)
  {
  coot_extra_debug_sigprint();

  const unwrap<T1> U(in.m);

  alias_wrapper<Mat<out_eT>, typename unwrap<T1>::stored_type> W(out, U.M);
  W.use.set_size(U.M.n_cols, U.M.n_rows);

  if (U.M.n_cols == 1 || U.M.n_rows == 1)
    {
    // Simply copying the data is sufficient.
    coot_rt_t::copy_mat(W.get_dev_mem(false), U.get_dev_mem(false),
                        // logically treat both as vectors
                        W.use.n_elem, 1,
                        0, 0, W.use.n_elem,
                        0, 0, U.M.n_elem);
    }
  else
    {
    // TODO: subview arguments
    coot_rt_t::strans(W.get_dev_mem(false), U.get_dev_mem(false), U.M.n_rows, U.M.n_cols);
    }
  }



template<typename T1>
inline
uword
op_strans::compute_n_rows(const Op<T1, op_strans>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);
  return in_n_cols;
  }



template<typename T1>
inline
uword
op_strans::compute_n_cols(const Op<T1, op_strans>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_cols);
  return in_n_rows;
  }
