// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
// Copyright 2025      Ryan Curtin (http://www.ratml.org)
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
op_col_as_mat::apply(Mat<typename T1::elem_type>& out, const CubeToMatOp<T1, op_col_as_mat>& expr)
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT;

  const unwrap_cube<T1> U(expr.m);
  const extract_subcube<typename unwrap_cube<T1>::stored_type> E(U.M);
  const Cube<eT>& A = E.M;

  const uword in_col = expr.aux_uword;

  coot_debug_check_bounds( (in_col >= A.n_cols), "Cube::col_as_mat(): index out of bounds" );

  const uword A_n_rows = A.n_rows;
  const uword A_n_cols = A.n_cols;
  const uword A_n_slices = A.n_slices;

  out.set_size(A_n_rows, A_n_slices);

  coot_rt_t::copy_mat(out.get_dev_mem(false),
                      A.get_dev_mem(false),
                      A_n_rows, A_n_slices,
                      // no offsets required for the output
                      0, 0, A_n_rows,
                      // pretend that the input cube is "flattened" into a 2D matrix
                      in_col * A_n_rows, 0, A_n_rows * A_n_cols);
  }



template<typename T1>
inline
uword
op_col_as_mat::compute_n_rows(const CubeToMatOp<T1, op_col_as_mat>& op, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices)
  {
  coot_ignore(op);
  coot_ignore(in_n_cols);
  coot_ignore(in_n_slices);
  return in_n_rows;
  }



template<typename T1>
inline
uword
op_col_as_mat::compute_n_cols(const CubeToMatOp<T1, op_col_as_mat>& op, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);
  coot_ignore(in_n_cols);
  return in_n_slices;
  }
