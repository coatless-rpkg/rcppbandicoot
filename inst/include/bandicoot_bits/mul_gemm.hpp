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



template<const bool do_trans_A=false, const bool do_trans_B=false, const bool use_alpha=false, const bool use_beta=false>
class gemm
  {
  public:

  template<typename eT>
  inline
  static
  void
  apply( Mat<eT>& C, const Mat<eT>& A, const Mat<eT>& B, const eT alpha = eT(1), const eT beta = eT(0) )
    {
    coot_extra_debug_sigprint();

    const eT local_alpha = (use_alpha) ? alpha : eT(1);
    const eT local_beta  = (use_beta)  ? beta  : eT(0);

    coot_rt_t::gemm<eT, do_trans_A, do_trans_B>(C.get_dev_mem(true),
                                                C.n_rows, C.n_cols,
                                                A.get_dev_mem(true),
                                                A.n_rows, A.n_cols,
                                                B.get_dev_mem(true),
                                                local_alpha, local_beta,
                                                0, 0, C.n_rows,
                                                0, 0, A.n_rows,
                                                0, 0, B.n_rows);
    }



  template<typename eT>
  inline
  static
  void
  apply( Mat<eT>& C, const subview<eT>& A, const Mat<eT>& B, const eT alpha = eT(1), const eT beta = eT(0) )
    {
    coot_extra_debug_sigprint();

    const eT local_alpha = (use_alpha) ? alpha : eT(1);
    const eT local_beta  = (use_beta)  ? beta  : eT(0);

    coot_rt_t::gemm<eT, do_trans_A, do_trans_B>(C.get_dev_mem(true),
                                                C.n_rows, C.n_cols,
                                                A.m.get_dev_mem(true),
                                                A.n_rows, A.n_cols,
                                                B.get_dev_mem(true),
                                                local_alpha, local_beta,
                                                0, 0, C.n_rows,
                                                A.aux_row1, A.aux_col1, A.m.n_rows,
                                                0, 0, B.n_rows);
    }



  template<typename eT>
  inline
  static
  void
  apply( Mat<eT>& C, const Mat<eT>& A, const subview<eT>& B, const eT alpha = eT(1), const eT beta = eT(0) )
    {
    coot_extra_debug_sigprint();

    const eT local_alpha = (use_alpha) ? alpha : eT(1);
    const eT local_beta  = (use_beta)  ? beta  : eT(0);

    coot_rt_t::gemm<eT, do_trans_A, do_trans_B>(C.get_dev_mem(true),
                                                C.n_rows, C.n_cols,
                                                A.get_dev_mem(true),
                                                A.n_rows, A.n_cols,
                                                B.m.get_dev_mem(true),
                                                local_alpha, local_beta,
                                                0, 0, C.n_rows,
                                                0, 0, A.n_rows,
                                                B.aux_row1, B.aux_col1, B.m.n_rows);
    }



  template<typename eT>
  inline
  static
  void
  apply( Mat<eT>& C, const subview<eT>& A, const subview<eT>& B, const eT alpha = eT(1), const eT beta = eT(0) )
    {
    coot_extra_debug_sigprint();

    const eT local_alpha = (use_alpha) ? alpha : eT(1);
    const eT local_beta  = (use_beta)  ? beta  : eT(0);

    coot_rt_t::gemm<eT, do_trans_A, do_trans_B>(C.get_dev_mem(true),
                                                C.n_rows, C.n_cols,
                                                A.m.get_dev_mem(true),
                                                A.n_rows, A.n_cols,
                                                B.m.get_dev_mem(true),
                                                local_alpha, local_beta,
                                                0, 0, C.n_rows,
                                                A.aux_row1, A.aux_col1, A.m.n_rows,
                                                B.aux_row1, B.aux_col1, B.m.n_rows);
    }
  };
