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



template<typename out_eT, typename T1, typename T2>
inline
void
glue_conv::apply(Mat<out_eT>& out, const Glue<T1, T2, glue_conv>& in)
  {
  coot_extra_debug_sigprint();

  const uword mode = in.aux_uword;

  // TODO: handle transposed input and other delayed inputs or optimizations
  unwrap<T1> UA(in.A);
  unwrap<T2> UB(in.B);

  typedef typename T1::elem_type eT;

  // Create aliases of the unwrapped objects that are column vectors.
  Mat<eT> A_col(UA.M.get_dev_mem(false), UA.M.n_elem, 1);
  Mat<eT> B_col(UB.M.get_dev_mem(false), UB.M.n_elem, 1);

  // We may need to make an alias of the output.
  Mat<out_eT> tmp;
  Mat<out_eT>& out_ref = (UA.is_alias(out) || UB.is_alias(out)) ? tmp : out;

  // Following Armadillo convention, if A is a column vector, then the result is a column vector.
  if (UA.M.n_cols == 1 || UA.M.is_col)
    {
    // If the result is a column vector, we can call glue_conv2 directly on the output.
    glue_conv2::apply_direct(out_ref, A_col, B_col, mode);
    }
  else
    {
    // Otherwise, the output from glue_conv2 will need to be reshaped.
    Mat<out_eT> out_tmp;
    glue_conv2::apply_direct(out_tmp, A_col, B_col, mode);
    out_tmp.reshape(1, out_tmp.n_elem);
    out_ref.set_size(out_tmp.n_rows, out_tmp.n_cols);
    out_ref.steal_mem(out_tmp);
    }

  if (UA.is_alias(out) || UB.is_alias(out))
    {
    out.set_size(tmp.n_rows, tmp.n_cols);
    out.steal_mem(tmp);
    }
  }



template<typename T1, typename T2>
inline
uword
glue_conv::compute_n_rows(const Glue<T1, T2, glue_conv>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  // (Imitating Armadillo convention) If A is a column vector, the result is a column vector; otherwise, it is a row vector.
  if (A_n_cols == 1 || T1::is_col)
    {
    const uword A_n_elem = A_n_rows * A_n_cols;
    const uword B_n_elem = B_n_rows * B_n_cols;

    if (glue.aux_uword == 0)
      {
      // "full"
      return A_n_elem + B_n_elem - 1;
      }
    else
      {
      // "same"
      return A_n_elem;
      }
    }
  else
    {
    return 1;
    }
  }



template<typename T1, typename T2>
inline
uword
glue_conv::compute_n_cols(const Glue<T1, T2, glue_conv>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  // (Imitating Armadillo convention) If A is a column vector, the result is a column vector; otherwise, it is a row vector.
  if (!(A_n_rows == 1 || T1::is_col))
    {
    const uword A_n_elem = A_n_rows * A_n_cols;
    const uword B_n_elem = B_n_rows * B_n_cols;

    if (glue.aux_uword == 0)
      {
      // "full"
      return A_n_elem + B_n_elem - 1;
      }
    else
      {
      // "same"
      return A_n_elem;
      }
    }
  else
    {
    return 1;
    }
  }
