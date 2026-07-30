// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2026 Andrew Furey (https://andrew.industries)
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
op_fliplr::apply(Mat<out_eT>& out, const Op<T1, op_fliplr>& in)
  {
    coot_debug_sigprint();

    const quasi_unwrap<T1> E(in.m);

    apply(out, E.M);
  }

template<typename eT, typename T1>
inline
void
op_fliplr::apply(Mat<eT>& out, const Op<Op<T1, op_flipud>, op_fliplr>& in)
  {
    coot_debug_sigprint();

    // NOTE: rotate_180 doesn't take in subviews, so use plain_unwrap instead of quasi.
    const plain_unwrap<T1> E(in.m.m);

    alias_wrapper<Mat<eT>, Mat<eT>> W(out, E.M);

    if (E.M.n_rows == 0 && E.M.n_cols == 0)
      {
      W.using_aux = false;
      return;
      }

    W.use.set_size(E.M.n_rows, E.M.n_cols);
    coot_rt_t::rotate_180(W.use.get_dev_mem(false), E.M.get_dev_mem(false), E.M.n_rows, E.M.n_cols);
  }


template<typename eT>
inline
void
op_fliplr::apply(Mat<eT>& out, const Mat<eT>& in)
  {
  coot_debug_sigprint();

  alias_wrapper<Mat<eT>, Mat<eT>> W(out, in);

  if (in.n_cols == 0)
    {
    W.using_aux = false;
    return;
    }

  W.use.set_size(in.n_rows, in.n_cols);
  coot_rt_t::fliplr(W.use.get_dev_mem(false), in.get_dev_mem(false), in.n_rows, in.n_cols, in.n_rows, 0, 0);
  }

template <typename eT>
inline
void
op_fliplr::apply(Mat<eT>& out, const subview<eT>& in)
  {
  coot_debug_sigprint();

  alias_wrapper<Mat<eT>, subview<eT>> W(out, in);

  if (in.n_cols == 0)
    {
    W.using_aux = false;
    return;
    }

  W.use.set_size(in.n_rows, in.n_cols);
  coot_rt_t::fliplr(W.use.get_dev_mem(false), in.m.get_dev_mem(false), in.n_rows, in.n_cols, in.m.n_rows, in.aux_row1, in.aux_col1);
  }


template<typename T1>
inline
uword
op_fliplr::compute_n_rows(const Op<T1, op_fliplr>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_cols);
  return in_n_rows;
  }


template <typename T1>
inline
uword
op_fliplr::compute_n_cols(const Op<T1, op_fliplr>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);
  return in_n_cols;
  }

///

template<typename out_eT, typename T1>
inline
void
op_flipud::apply(Mat<out_eT>& out, const Op<T1, op_flipud>& in)
  {
    coot_debug_sigprint();

    const quasi_unwrap<T1> E(in.m);

    apply(out, E.M);
  }

template<typename eT, typename T1>
inline
void
op_flipud::apply(Mat<eT>& out, const Op<Op<T1, op_fliplr>, op_flipud>& in)
  {
    coot_debug_sigprint();

    // NOTE: rotate_180 doesn't take in subviews, so use plain_unwrap instead of quasi.
    const plain_unwrap<T1> E(in.m.m);
    alias_wrapper<Mat<eT>, Mat<eT>> W(out, E.M);

    if (E.M.n_rows == 0 && E.M.n_cols == 0)
      {
      W.using_aux = false;
      return;
      }

    W.use.set_size(E.M.n_rows, E.M.n_cols);
    coot_rt_t::rotate_180(W.use.get_dev_mem(false), E.M.get_dev_mem(false), E.M.n_rows, E.M.n_cols);
  }

template<typename eT>
inline
void
op_flipud::apply(Mat<eT>& out, const Mat<eT>& in)
  {
  coot_debug_sigprint();

  alias_wrapper<Mat<eT>, Mat<eT>> W(out, in);

  if (in.n_cols == 0)
    {
    W.using_aux = false;
    return;
    }

  W.use.set_size(in.n_rows, in.n_cols);
  coot_rt_t::flipud(W.use.get_dev_mem(false), in.get_dev_mem(false), in.n_rows, in.n_cols, in.n_rows, 0, 0);
  }

template <typename eT>
inline
void
op_flipud::apply(Mat<eT>& out, const subview<eT>& in)
  {
  coot_debug_sigprint();

  alias_wrapper<Mat<eT>, subview<eT>> W(out, in);

  if (in.n_cols == 0)
    {
    W.using_aux = false;
    return;
    }

  W.use.set_size(in.n_rows, in.n_cols);
  coot_rt_t::flipud(W.use.get_dev_mem(false), in.m.get_dev_mem(false), in.n_rows, in.n_cols, in.m.n_rows, in.aux_row1, in.aux_col1);
  }


template<typename T1>
inline
uword
op_flipud::compute_n_rows(const Op<T1, op_flipud>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_cols);
  return in_n_rows;
  }


template <typename T1>
inline
uword
op_flipud::compute_n_cols(const Op<T1, op_flipud>& op, const uword in_n_rows, const uword in_n_cols)
  {
  coot_ignore(op);
  coot_ignore(in_n_rows);
  return in_n_cols;
  }

///
