// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2025 Ryan Curtin (https://www.ratml.org)
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
op_replace::apply(Mat<out_eT>& out, const Op<T1, op_replace>& in)
  {
  coot_extra_debug_sigprint();

  const unwrap<T1> U(in.m);

  // Aliasing actually does not matter for this kernel.
  out.set_size(U.M.n_rows, U.M.n_cols);
  if (out.n_elem == 0)
    {
    return;
    }

  coot_rt_t::replace(out.get_dev_mem(false), U.get_dev_mem(false),
                     in.aux, in.aux_b,
                     out.n_rows, out.n_cols, 1,
                     0, 0, 0, out.n_rows, out.n_cols,
                     U.get_row_offset(), U.get_col_offset(), 0, U.get_M_n_rows(), U.M.n_cols);
  }



template<typename out_eT, typename T1>
inline
void
op_replace::apply(Cube<out_eT>& out, const OpCube<T1, op_replace>& in)
  {
  coot_extra_debug_sigprint();

  const unwrap_cube<T1> U(in.m);

  // Aliasing actually does not matter for this kernel.
  out.set_size(U.M.n_rows, U.M.n_cols, U.M.n_slices);
  if (out.n_elem == 0)
    {
    return;
    }

  coot_rt_t::replace(out.get_dev_mem(false), U.get_dev_mem(false),
                     in.aux, in.aux_b,
                     out.n_rows, out.n_cols, out.n_slices,
                     0, 0, 0, out.n_rows, out.n_cols,
                     U.get_row_offset(), U.get_col_offset(), U.get_slice_offset(), U.get_M_n_rows(), U.get_M_n_cols());
  }



template<typename T1>
inline
uword
op_replace::compute_n_rows(const Op<T1, op_replace>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_cols);

  return in_n_rows;
  }



template<typename T1>
inline
uword
op_replace::compute_n_cols(const Op<T1, op_replace>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);

  return in_n_cols;
  }



template<typename T1>
inline
uword
op_replace::compute_n_rows(const OpCube<T1, op_replace>& op, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices)
  {
  coot_ignore(op);
  coot_ignore(in_n_cols);
  coot_ignore(in_n_slices);

  return in_n_rows;
  }



template<typename T1>
inline
uword
op_replace::compute_n_cols(const OpCube<T1, op_replace>& op, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);
  coot_ignore(in_n_slices);

  return in_n_cols;
  }



template<typename T1>
inline
uword
op_replace::compute_n_slices(const OpCube<T1, op_replace>& op, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);
  coot_ignore(in_n_cols);

  return in_n_slices;
  }
