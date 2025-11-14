// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2017-2023 Ryan Curtin (https://www.ratml.org)
// Copyright 2008-2016 Conrad Sanderson (https://conradsanderson.id.au)
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



template<const bool do_trans_A=false, const bool use_alpha=false, const bool use_beta=false>
class gemv
  {
  public:

  template<typename eT>
  inline
  static
  void
  apply( Mat<eT>& y, const Mat<eT>& A, const Mat<eT>& x, const eT alpha = eT(1), const eT beta = eT(0) )
    {
    coot_extra_debug_sigprint();

    const eT local_alpha = (use_alpha) ? alpha : eT(1);
    const eT local_beta  = (use_beta)  ? beta  : eT(0);

    coot_rt_t::gemv<eT, do_trans_A>(y.get_dev_mem(true),
                                    A.get_dev_mem(true),
                                    A.n_rows, A.n_cols,
                                    x.get_dev_mem(true),
                                    local_alpha, local_beta,
                                    0, 1,
                                    0, 0, A.n_rows,
                                    0, 1);
    }



  template<typename eT>
  inline
  static
  void
  apply( Mat<eT>& y, const subview<eT>& A, const Mat<eT>& x, const eT alpha = eT(1), const eT beta = eT(0) )
    {
    coot_extra_debug_sigprint();

    const eT local_alpha = (use_alpha) ? alpha : eT(1);
    const eT local_beta  = (use_beta)  ? beta  : eT(0);

    coot_rt_t::gemv<eT, do_trans_A>(y.get_dev_mem(true),
                                    A.m.get_dev_mem(true),
                                    A.n_rows, A.n_cols,
                                    x.get_dev_mem(true),
                                    local_alpha, local_beta,
                                    0, 1,
                                    A.aux_row1, A.aux_col1, A.m.n_rows,
                                    0, 1);
    }



  template<typename eT>
  inline
  static
  void
  apply( Mat<eT>& y, const Mat<eT>& A, const subview<eT>& x, const eT alpha = eT(1), const eT beta = eT(0) )
    {
    coot_extra_debug_sigprint();

    const eT local_alpha = (use_alpha) ? alpha : eT(1);
    const eT local_beta  = (use_beta)  ? beta  : eT(0);

    const uword x_mem_incr = (x.n_rows == 1) ? x.m.n_rows : 1;

    coot_rt_t::gemv<eT, do_trans_A>(y.get_dev_mem(true),
                                    A.get_dev_mem(true),
                                    A.n_rows, A.n_cols,
                                    x.m.get_dev_mem(true),
                                    local_alpha, local_beta,
                                    0, 1,
                                    0, 0, A.n_rows,
                                    x.aux_row1 + x.aux_col1 * x.m.n_rows, x_mem_incr);
    }



  template<typename eT>
  inline
  static
  void
  apply( Mat<eT>& y, const subview<eT>& A, const subview<eT>& x, const eT alpha = eT(1), const eT beta = eT(0) )
    {
    coot_extra_debug_sigprint();

    const eT local_alpha = (use_alpha) ? alpha : eT(1);
    const eT local_beta  = (use_beta)  ? beta  : eT(0);

    const uword x_mem_incr = (x.n_rows == 1) ? x.m.n_rows : 1;

    coot_rt_t::gemv<eT, do_trans_A>(y.get_dev_mem(true),
                                    A.m.get_dev_mem(true),
                                    A.n_rows, A.n_cols,
                                    x.m.get_dev_mem(true),
                                    local_alpha, local_beta,
                                    0, 1,
                                    A.aux_row1, A.aux_col1, A.m.n_rows,
                                    x.aux_row1 + x.aux_col1 * x.m.n_rows, x_mem_incr);
    }
  };
