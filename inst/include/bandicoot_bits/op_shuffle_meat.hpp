// Copyright 2021 Marcus Edel (http://kurg.org)
// Copyright 2021 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2024 Ryan Curtin (http://www.ratml.org)
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
op_shuffle::apply(Mat<typename T1::elem_type>& out, const Op<T1,op_shuffle>& in)
  {
  coot_extra_debug_sigprint();

  const unwrap<T1> U(in.m);
  if (U.M.is_empty()) { return; }

  const uword dim = in.aux_uword_a;

  coot_debug_check( (dim > 1), "shuffle(): parameter 'dim' must be 0 or 1" );

  // If the output is an alias of the input, allocate a temporary matrix.
  if (U.is_alias(out))
    {
    Mat<typename T1::elem_type> tmp(U.M.n_rows, U.M.n_cols);
    coot_rt_t::shuffle(tmp.get_dev_mem(false), 0, 0, tmp.n_rows,
                       U.get_dev_mem(false), U.get_row_offset(), U.get_col_offset(), U.get_M_n_rows(),
                       U.M.n_rows, U.M.n_cols, dim);
    out.steal_mem(tmp);
    }
  else
    {
    out.set_size(U.M.n_rows, U.M.n_cols);

    coot_rt_t::shuffle(out.get_dev_mem(false), 0, 0, out.n_rows,
                       U.get_dev_mem(false), U.get_row_offset(), U.get_col_offset(), U.get_M_n_rows(),
                       U.M.n_rows, U.M.n_cols, dim);
    }
  }



template<typename T1>
inline
void
op_shuffle_vec::apply(Mat<typename T1::elem_type>& out, const Op<T1,op_shuffle_vec>& in)
  {
  coot_extra_debug_sigprint();

  const unwrap<T1> U(in.m);
  if (U.M.is_empty()) { return; }

  // If the output is an alias of the input, allocate a temporary matrix.
  if (U.is_alias(out))
    {
    Mat<typename T1::elem_type> tmp(U.M.n_rows, U.M.n_cols);
    coot_rt_t::shuffle(tmp.get_dev_mem(false), 0, 0, tmp.n_elem,
                       U.get_dev_mem(false), U.get_row_offset(), U.get_col_offset(), U.get_M_n_rows(),
                       U.M.n_elem, 1, 0 /* always dim = 0 for vectors by convention */);
    out.steal_mem(tmp);
    }
  else
    {
    out.set_size(U.M.n_rows, U.M.n_cols);
    coot_rt_t::shuffle(out.get_dev_mem(false), 0, 0, out.n_elem,
                       U.get_dev_mem(false), U.get_row_offset(), U.get_col_offset(), U.get_M_n_rows(),
                       U.M.n_elem, 1, 0 /* always dim = 0 for vectors by convention */);
    }
  }
