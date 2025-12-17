// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2008-2016 Conrad Sanderson (https://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
// Copyright 2025      Ryan Curtin (http://ratml.org)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------


template<typename eT, typename T1, typename T2>
inline
subview_elem2<eT,T1,T2>::~subview_elem2()
  {
  coot_extra_debug_sigprint();
  }


template<typename eT, typename T1, typename T2>
coot_inline
subview_elem2<eT,T1,T2>::subview_elem2
  (
  const Mat<eT>&        in_m,
  const Base<uword,T1>& in_ri,
  const Base<uword,T2>& in_ci,
  const bool            in_all_rows,
  const bool            in_all_cols
  )
  : m        (in_m       )
  , base_ri  (in_ri      )
  , base_ci  (in_ci      )
  , all_rows (in_all_rows)
  , all_cols (in_all_cols)
  {
  coot_extra_debug_sigprint();
  }



template<typename eT, typename T1, typename T2>
inline
void
subview_elem2<eT,T1,T2>::inplace_op(const twoway_kernel_id::enum_id kernel_id,
                                    const eT val_pre,
                                    const eT val_post)
  {
  coot_extra_debug_sigprint();

  if (all_cols && all_rows)
    {
    // we are using... no indices at all?  This shouldn't happen...
    coot_rt_t::eop_scalar(kernel_id,
                          m.get_dev_mem(false), m.get_dev_mem(false),
                          val_pre, val_post,
                          m.n_rows, m.n_cols, 1,
                          0, 0, 0, m.n_rows, m.n_cols,
                          0, 0, 0, m.n_rows, m.n_cols);
    }
  else if (all_cols)
    {
    // we are only using the row indices
    const unwrap<T1> U(base_ri.get_ref());
    const extract_subview<typename unwrap<T1>::stored_type> E(U.M);

    coot_debug_check( E.M.n_rows != 1 && E.M.n_cols != 1, "Mat::rows(): column indices must be a vector" );

    // A bounds check is tedious but the only way that we can give the user an error.
    coot_debug_check( E.M.max() >= m.n_rows, "Mat::rows(): index out of bounds" );

    alias_wrapper<Mat<eT>, Mat<uword>> A(const_cast<Mat<eT>&>(m), E.M);

    coot_rt_t::eop_scalar_subview_elem2(kernel_id,
                                        A.get_dev_mem(false),
                                        E.M.get_dev_mem(false),
                                        dev_mem_t<uword>({{ nullptr, 0 }}), // no column indices
                                        A.get_dev_mem(false),
                                        E.M.get_dev_mem(false),
                                        dev_mem_t<uword>({{ nullptr, 0 }}), // no column indices
                                        val_pre, val_post,
                                        E.M.n_elem, A.get_n_cols(),
                                        A.get_n_rows(), A.get_n_rows());
    }
  else if (all_rows)
    {
    // we are only using the column indices
    const unwrap<T2> U(base_ci.get_ref());
    const extract_subview<typename unwrap<T2>::stored_type> E(U.M);

    coot_debug_check( E.M.n_rows != 1 && E.M.n_cols != 1, "Mat::cols(): row indices must be a vector" );

    // A bounds check is tedious but the only way that we can give the user an error.
    coot_debug_check( E.M.max() >= m.n_cols, "Mat::cols(): index out of bounds" );

    alias_wrapper<Mat<eT>, Mat<uword>> A(const_cast<Mat<eT>&>(m), E.M);

    coot_rt_t::eop_scalar_subview_elem2(kernel_id,
                                        A.get_dev_mem(false),
                                        dev_mem_t<uword>({{ nullptr, 0 }}), // no row indices
                                        E.M.get_dev_mem(false),
                                        A.get_dev_mem(false),
                                        dev_mem_t<uword>({{ nullptr, 0 }}), // no row indices
                                        E.M.get_dev_mem(false),
                                        val_pre, val_post,
                                        A.get_n_rows(), E.M.n_elem,
                                        A.get_n_rows(), A.get_n_rows());
    }
  else
    {
    // we have to unwrap both sets of indices
    const unwrap<T1> U1(base_ri.get_ref());
    const extract_subview<typename unwrap<T1>::stored_type> E1(U1.M);

    const unwrap<T2> U2(base_ci.get_ref());
    const extract_subview<typename unwrap<T2>::stored_type> E2(U2.M);

    coot_debug_check( E1.M.n_rows != 1 && E1.M.n_cols != 1, "Mat::elem(): row indices must be a vector" );
    coot_debug_check( E2.M.n_rows != 1 && E2.M.n_cols != 1, "Mat::elem(): column indices must be a vector" );

    // A bounds check is tedious but the only way that we can give the user an error.
    coot_debug_check( E1.M.max() >= m.n_rows, "Mat::elem(): row index out of bounds" );
    coot_debug_check( E2.M.max() >= m.n_cols, "Mat::elem(): column index out of bounds" );

    alias_wrapper<Mat<eT>, Mat<uword>, Mat<uword>> A(const_cast<Mat<eT>&>(m), E1.M, E2.M);

    coot_rt_t::eop_scalar_subview_elem2(kernel_id,
                                        A.get_dev_mem(false),
                                        E1.M.get_dev_mem(false),
                                        E2.M.get_dev_mem(false),
                                        A.get_dev_mem(false),
                                        E1.M.get_dev_mem(false),
                                        E2.M.get_dev_mem(false),
                                        val_pre, val_post,
                                        E1.M.n_elem, E2.M.n_elem,
                                        A.get_n_rows(), A.get_n_rows());
    }
  }



template<typename eT, typename T1, typename T2>
template<typename T3, typename T4>
inline
void
subview_elem2<eT,T1,T2>::inplace_op(const twoway_kernel_id::enum_id kernel_id, const subview_elem2<eT, T3, T4>& x)
  {
  coot_extra_debug_sigprint();

  // ouch, lots of possible cases...
  if (x.all_rows && x.all_cols)
    {
    (*this).operator=(x.m); // this shouldn't happen...
    }

  if (all_rows && all_cols)
    {
    const_cast<Mat<eT>&>(m).operator=(x.get_ref()); // this shouldn't happen...
    }
  else if (all_rows)
    {
    if (x.all_rows)
      {
      const unwrap<T2> U2(base_ci.get_ref());
      const unwrap<T4> U4(x.base_ci.get_ref());

      const extract_subview<typename unwrap<T2>::stored_type> E2(U2.M);
      const extract_subview<typename unwrap<T4>::stored_type> E4(U4.M);

      coot_debug_check( E2.M.n_rows != 1 && E2.M.n_cols != 1, "Mat::cols(): column indices must be a vector" );
      coot_debug_check( E4.M.n_rows != 1 && E4.M.n_cols != 1, "Mat::cols(): column indices must be a vector" );

      coot_debug_check( x.m.n_rows != m.n_rows && E4.M.n_elem != E2.M.n_elem, "Mat::cols(): size mismatch" );

      alias_wrapper<Mat<eT>, Mat<uword>, Mat<eT>, Mat<uword>> A(const_cast<Mat<eT>&>(m), E2.M, const_cast<Mat<eT>&>(x.m), E4.M);

      coot_rt_t::eop_subview_elem2(kernel_id,
                                   A.get_dev_mem(false),
                                   dev_mem_t<uword>({{ nullptr, 0 }}), // no row indices
                                   E2.M.get_dev_mem(false),
                                   x.m.get_dev_mem(false),
                                   dev_mem_t<uword>({{ nullptr, 0 }}), // no row indices
                                   E4.M.get_dev_mem(false),
                                   A.use.n_rows,
                                   E2.M.n_elem,
                                   A.use.n_rows,
                                   x.m.n_rows);
      }
    else if (x.all_cols)
      {
      const unwrap<T2> U2(base_ci.get_ref());
      const unwrap<T3> U3(x.base_ri.get_ref());

      const extract_subview<typename unwrap<T2>::stored_type> E2(U2.M);
      const extract_subview<typename unwrap<T3>::stored_type> E3(U3.M);

      coot_debug_check( E2.M.n_rows != 1 && E2.M.n_cols != 1, "Mat::cols(): column indices must be a vector" );
      coot_debug_check( E3.M.n_rows != 1 && E3.M.n_cols != 1, "Mat::rows(): row indices must be a vector" );

      coot_debug_check( E3.M.n_elem != m.n_rows && x.m.n_cols != E2.M.n_elem, "Mat::cols(): size mismatch" );

      alias_wrapper<Mat<eT>, Mat<uword>, Mat<eT>, Mat<uword>> A(const_cast<Mat<eT>&>(m), E2.M, const_cast<Mat<eT>&>(x.m), E3.M);

      coot_rt_t::eop_subview_elem2(kernel_id,
                                   A.get_dev_mem(false),
                                   dev_mem_t<uword>({{ nullptr, 0 }}), // no row indices
                                   E2.M.get_dev_mem(false),
                                   x.m.get_dev_mem(false),
                                   E3.M.get_dev_mem(false),
                                   dev_mem_t<uword>({{ nullptr, 0 }}), // no column indices
                                   A.use.n_rows,
                                   E2.M.n_elem,
                                   A.use.n_rows,
                                   x.m.n_rows);
      }
    else
      {
      const unwrap<T2> U2(base_ci.get_ref());
      const unwrap<T3> U3(x.base_ri.get_ref());
      const unwrap<T4> U4(x.base_ci.get_ref());

      const extract_subview<typename unwrap<T2>::stored_type> E2(U2.M);
      const extract_subview<typename unwrap<T3>::stored_type> E3(U3.M);
      const extract_subview<typename unwrap<T4>::stored_type> E4(U4.M);

      coot_debug_check( E2.M.n_rows != 1 && E2.M.n_cols != 1, "Mat::cols(): column indices must be a vector" );
      coot_debug_check( E3.M.n_rows != 1 && E3.M.n_cols != 1, "Mat::elem(): row indices must be a vector" );
      coot_debug_check( E4.M.n_rows != 1 && E4.M.n_cols != 1, "Mat::elem(): column indices must be a vector" );

      coot_debug_check( E3.M.n_elem != m.n_rows && E4.M.n_elem != E2.M.n_elem, "Mat::cols(): size mismatch" );

      alias_wrapper<Mat<eT>, Mat<uword>, Mat<eT>, Mat<uword>, Mat<uword>> A(const_cast<Mat<eT>&>(m), E2.M, const_cast<Mat<eT>&>(x.m), E3.M, E4.M);

      coot_rt_t::eop_subview_elem2(kernel_id,
                                   A.get_dev_mem(false),
                                   dev_mem_t<uword>({{ nullptr, 0 }}), // no row indices
                                   E2.M.get_dev_mem(false),
                                   x.m.get_dev_mem(false),
                                   E3.M.get_dev_mem(false),
                                   E4.M.get_dev_mem(false),
                                   A.use.n_rows,
                                   E2.M.n_elem,
                                   A.use.n_rows,
                                   x.m.n_rows);
      }
    }
  else if (all_cols)
    {
    if (x.all_rows)
      {
      const unwrap<T1> U1(base_ri.get_ref());
      const unwrap<T4> U4(x.base_ci.get_ref());

      const extract_subview<typename unwrap<T1>::stored_type> E1(U1.M);
      const extract_subview<typename unwrap<T4>::stored_type> E4(U4.M);

      coot_debug_check( E1.M.n_rows != 1 && E1.M.n_cols != 1, "Mat::rows(): row indices must be a vector" );
      coot_debug_check( E4.M.n_rows != 1 && E4.M.n_cols != 1, "Mat::elem(): column indices must be a vector" );

      coot_debug_check( x.m.n_rows != E1.M.n_elem && E4.M.n_elem != m.n_cols, "Mat::rows(): size mismatch" );

      alias_wrapper<Mat<eT>, Mat<uword>, Mat<eT>, Mat<uword>> A(const_cast<Mat<eT>&>(m), E1.M, const_cast<Mat<eT>&>(x.m), E4.M);

      coot_rt_t::eop_subview_elem2(kernel_id,
                                   A.get_dev_mem(false),
                                   E1.M.get_dev_mem(false),
                                   dev_mem_t<uword>({{ nullptr, 0 }}), // no column indices
                                   x.m.get_dev_mem(false),
                                   dev_mem_t<uword>({{ nullptr, 0 }}), // no row indices
                                   E4.M.get_dev_mem(false),
                                   E1.M.n_elem,
                                   A.use.n_cols,
                                   A.use.n_rows,
                                   x.m.n_rows);
      }
    else if (x.all_cols)
      {
      const unwrap<T1> U1(base_ri.get_ref());
      const unwrap<T3> U3(x.base_ri.get_ref());

      const extract_subview<typename unwrap<T1>::stored_type> E1(U1.M);
      const extract_subview<typename unwrap<T3>::stored_type> E3(U3.M);

      coot_debug_check( E1.M.n_rows != 1 && E1.M.n_cols != 1, "Mat::rows(): row indices must be a vector" );
      coot_debug_check( E3.M.n_rows != 1 && E3.M.n_cols != 1, "Mat::rows(): row indices must be a vector" );

      coot_debug_check( E3.M.n_elem != E1.M.n_elem && x.m.n_cols != m.n_cols, "Mat::rows(): size mismatch" );

      alias_wrapper<Mat<eT>, Mat<uword>, Mat<eT>, Mat<uword>> A(const_cast<Mat<eT>&>(m), E1.M, const_cast<Mat<eT>&>(x.m), E3.M);

      coot_rt_t::eop_subview_elem2(kernel_id,
                                   A.get_dev_mem(false),
                                   E1.M.get_dev_mem(false),
                                   dev_mem_t<uword>({{ nullptr, 0 }}), // no column indices
                                   x.m.get_dev_mem(false),
                                   E3.M.get_dev_mem(false),
                                   dev_mem_t<uword>({{ nullptr, 0 }}), // no column indices
                                   E1.M.n_elem,
                                   A.use.n_cols,
                                   A.use.n_rows,
                                   x.m.n_rows);
      }
    else
      {
      const unwrap<T1> U1(base_ri.get_ref());
      const unwrap<T3> U3(x.base_ri.get_ref());
      const unwrap<T4> U4(x.base_ci.get_ref());

      const extract_subview<typename unwrap<T1>::stored_type> E1(U1.M);
      const extract_subview<typename unwrap<T3>::stored_type> E3(U3.M);
      const extract_subview<typename unwrap<T4>::stored_type> E4(U4.M);

      coot_debug_check( E1.M.n_rows != 1 && E1.M.n_cols != 1, "Mat::rows(): row indices must be a vector" );
      coot_debug_check( E3.M.n_rows != 1 && E3.M.n_cols != 1, "Mat::elem(): row indices must be a vector" );
      coot_debug_check( E4.M.n_rows != 1 && E4.M.n_cols != 1, "Mat::elem(): column indices must be a vector" );

      coot_debug_check( E3.M.n_elem != E1.M.n_elem && E4.M.n_elem != m.n_cols, "Mat::rows(): size mismatch" );

      alias_wrapper<Mat<eT>, Mat<uword>, Mat<eT>, Mat<uword>, Mat<uword>> A(const_cast<Mat<eT>&>(m), E1.M, const_cast<Mat<eT>&>(x.m), E3.M, E4.M);

      coot_rt_t::eop_subview_elem2(kernel_id,
                                   A.get_dev_mem(false),
                                   E1.M.get_dev_mem(false),
                                   dev_mem_t<uword>({{ nullptr, 0 }}), // no column indices
                                   x.m.get_dev_mem(false),
                                   E3.M.get_dev_mem(false),
                                   E4.M.get_dev_mem(false),
                                   E1.M.n_elem,
                                   A.use.n_cols,
                                   A.use.n_rows,
                                   x.m.n_rows);
      }
    }
  else
    {
    if (x.all_rows)
      {
      const unwrap<T1> U1(base_ri.get_ref());
      const unwrap<T2> U2(base_ci.get_ref());
      const unwrap<T4> U4(x.base_ci.get_ref());

      const extract_subview<typename unwrap<T1>::stored_type> E1(U1.M);
      const extract_subview<typename unwrap<T2>::stored_type> E2(U2.M);
      const extract_subview<typename unwrap<T4>::stored_type> E4(U4.M);

      coot_debug_check( E1.M.n_rows != 1 && E1.M.n_cols != 1, "Mat::elem(): row indices must be a vector" );
      coot_debug_check( E2.M.n_rows != 1 && E2.M.n_cols != 1, "Mat::elem(): column indices must be a vector" );
      coot_debug_check( E4.M.n_rows != 1 && E4.M.n_cols != 1, "Mat::cols(): column indices must be a vector" );

      coot_debug_check( x.m.n_rows != E1.M.n_elem && E4.M.n_elem != E2.M.n_elem, "Mat::elem(): size mismatch" );

      alias_wrapper<Mat<eT>, Mat<uword>, Mat<uword>, Mat<eT>, Mat<uword>> A(const_cast<Mat<eT>&>(m), E1.M, E2.M, const_cast<Mat<eT>&>(x.m), E4.M);

      coot_rt_t::eop_subview_elem2(kernel_id,
                                   A.get_dev_mem(false),
                                   E1.M.get_dev_mem(false),
                                   E2.M.get_dev_mem(false),
                                   x.m.get_dev_mem(false),
                                   dev_mem_t<uword>({{ nullptr, 0 }}), // no row indices
                                   E4.M.get_dev_mem(false),
                                   E1.M.n_elem,
                                   E2.M.n_elem,
                                   A.use.n_rows,
                                   x.m.n_rows);
      }
    else if (x.all_cols)
      {
      const unwrap<T1> U1(base_ri.get_ref());
      const unwrap<T2> U2(base_ci.get_ref());
      const unwrap<T3> U3(x.base_ri.get_ref());

      const extract_subview<typename unwrap<T1>::stored_type> E1(U1.M);
      const extract_subview<typename unwrap<T2>::stored_type> E2(U2.M);
      const extract_subview<typename unwrap<T3>::stored_type> E3(U3.M);

      coot_debug_check( E1.M.n_rows != 1 && E1.M.n_cols != 1, "Mat::elem(): row indices must be a vector" );
      coot_debug_check( E2.M.n_rows != 1 && E2.M.n_cols != 1, "Mat::elem(): column indices must be a vector" );
      coot_debug_check( E3.M.n_rows != 1 && E3.M.n_cols != 1, "Mat::rows(): row indices must be a vector" );

      coot_debug_check( E3.M.n_elem != E1.M.n_elem && x.m.n_cols != E2.M.n_elem, "Mat::elem(): size mismatch" );

      alias_wrapper<Mat<eT>, Mat<uword>, Mat<uword>, Mat<eT>, Mat<uword>> A(const_cast<Mat<eT>&>(m), E1.M, E2.M, const_cast<Mat<eT>&>(x.m), E3.M);

      coot_rt_t::eop_subview_elem2(kernel_id,
                                   A.get_dev_mem(false),
                                   E1.M.get_dev_mem(false),
                                   E2.M.get_dev_mem(false),
                                   x.m.get_dev_mem(false),
                                   E3.M.get_dev_mem(false),
                                   dev_mem_t<uword>({{ nullptr, 0 }}), // no column indices
                                   E1.M.n_elem,
                                   E2.M.n_elem,
                                   A.use.n_rows,
                                   x.m.n_rows);
      }
    else
      {
      const unwrap<T1> U1(base_ri.get_ref());
      const unwrap<T2> U2(base_ci.get_ref());
      const unwrap<T3> U3(x.base_ri.get_ref());
      const unwrap<T4> U4(x.base_ci.get_ref());

      const extract_subview<typename unwrap<T1>::stored_type> E1(U1.M);
      const extract_subview<typename unwrap<T2>::stored_type> E2(U2.M);
      const extract_subview<typename unwrap<T3>::stored_type> E3(U3.M);
      const extract_subview<typename unwrap<T4>::stored_type> E4(U4.M);

      coot_debug_check( E1.M.n_rows != 1 && E1.M.n_cols != 1, "Mat::elem(): row indices must be a vector" );
      coot_debug_check( E2.M.n_rows != 1 && E2.M.n_cols != 1, "Mat::elem(): column indices must be a vector" );
      coot_debug_check( E3.M.n_rows != 1 && E3.M.n_cols != 1, "Mat::elem(): row indices must be a vector" );
      coot_debug_check( E4.M.n_rows != 1 && E4.M.n_cols != 1, "Mat::elem(): column indices must be a vector" );

      coot_debug_check( E3.M.n_elem != E1.M.n_elem && E4.M.n_elem != E2.M.n_elem, "Mat::elem(): size mismatch" );

      alias_wrapper<Mat<eT>, Mat<uword>, Mat<uword>, Mat<eT>, Mat<uword>, Mat<uword>> A(const_cast<Mat<eT>&>(m), E1.M, E2.M, const_cast<Mat<eT>&>(x.m), E3.M, E4.M);

      coot_rt_t::eop_subview_elem2(kernel_id,
                                   A.get_dev_mem(false),
                                   E1.M.get_dev_mem(false),
                                   E2.M.get_dev_mem(false),
                                   x.m.get_dev_mem(false),
                                   E3.M.get_dev_mem(false),
                                   E4.M.get_dev_mem(false),
                                   E1.M.n_elem,
                                   E2.M.n_elem,
                                   A.use.n_rows,
                                   x.m.n_rows);
      }
    }
  }



template<typename eT, typename T1, typename T2>
template<typename expr>
inline
void
subview_elem2<eT,T1,T2>::inplace_op(const twoway_kernel_id::enum_id kernel_id, const Base<eT,expr>& x)
  {
  coot_extra_debug_sigprint();

  if (all_rows && all_cols)
    {
    const_cast<Mat<eT>&>(m).operator=(x.get_ref()); // this shouldn't happen...
    }
  else if (all_rows)
    {
    const unwrap<T2> U2(base_ci.get_ref());
    const unwrap<expr> U3(x.get_ref());

    const extract_subview<typename unwrap<T2>::stored_type> E2(U2.M);
    const extract_subview<typename unwrap<expr>::stored_type> E3(U3.M);

    coot_debug_check( E2.M.n_rows != 1 && E2.M.n_cols != 1, "Mat::rows(): column indices must be a vector" );

    coot_debug_check( E3.M.n_rows != m.n_rows && E3.M.n_cols != E2.M.n_elem, "Mat::rows(): size mismatch" );

    alias_wrapper<Mat<eT>, Mat<uword>, Mat<eT>> A(const_cast<Mat<eT>&>(m), E2.M, E3.M);

    coot_rt_t::eop_subview_elem2_array(kernel_id,
                                       A.get_dev_mem(false),
                                       dev_mem_t<uword>({{ nullptr, 0 }}), // no row indices
                                       E2.M.get_dev_mem(false),
                                       E3.M.get_dev_mem(false),
                                       A.use.n_rows,
                                       E2.M.n_elem,
                                       A.use.n_rows);
    }
  else if (all_cols)
    {
    const unwrap<T1> U1(base_ri.get_ref());
    const unwrap<expr> U3(x.get_ref());

    const extract_subview<typename unwrap<T1>::stored_type> E1(U1.M);
    const extract_subview<typename unwrap<expr>::stored_type> E3(U3.M);

    coot_debug_check( E1.M.n_rows != 1 && E1.M.n_cols != 1, "Mat::cols(): row indices must be a vector" );

    coot_debug_check( E3.M.n_rows != E1.M.n_elem && E3.M.n_cols != m.n_cols, "Mat::cols(): size mismatch" );

    alias_wrapper<Mat<eT>, Mat<uword>, Mat<eT>> A(const_cast<Mat<eT>&>(m), E1.M, E3.M);

    coot_rt_t::eop_subview_elem2_array(kernel_id,
                                       A.get_dev_mem(false),
                                       E1.M.get_dev_mem(false),
                                       dev_mem_t<uword>({{ nullptr, 0 }}), // no column indices
                                       E3.M.get_dev_mem(false),
                                       E1.M.n_elem,
                                       A.use.n_cols,
                                       A.use.n_rows);
    }
  else
    {
    const unwrap<T1> U1(base_ri.get_ref());
    const unwrap<T2> U2(base_ci.get_ref());
    const unwrap<expr> U3(x.get_ref());

    const extract_subview<typename unwrap<T1>::stored_type> E1(U1.M);
    const extract_subview<typename unwrap<T2>::stored_type> E2(U2.M);
    const extract_subview<typename unwrap<expr>::stored_type> E3(U3.M);

    coot_debug_check( E1.M.n_rows != 1 && E1.M.n_cols != 1, "Mat::elem(): row indices must be a vector" );
    coot_debug_check( E2.M.n_rows != 1 && E2.M.n_cols != 1, "Mat::elem(): column indices must be a vector" );

    coot_debug_check( E3.M.n_rows != E1.M.n_elem && E3.M.n_cols != E2.M.n_elem, "Mat::elem(): size mismatch" );

    alias_wrapper<Mat<eT>, Mat<uword>, Mat<uword>, Mat<eT>> A(const_cast<Mat<eT>&>(m), E1.M, E2.M, E3.M);

    coot_rt_t::eop_subview_elem2_array(kernel_id,
                                       A.get_dev_mem(false),
                                       E1.M.get_dev_mem(false),
                                       E2.M.get_dev_mem(false),
                                       E3.M.get_dev_mem(false),
                                       E1.M.n_elem,
                                       E2.M.n_elem,
                                       A.use.n_rows);
    }
  }



template<typename eT, typename T1, typename T2>
inline
void
subview_elem2<eT,T1,T2>::randu()
  {
  coot_extra_debug_sigprint();

  // until we are able to index subviews in any arbitrary way in a generated kernel,
  // we use a slow implementation where we generate all the random numbers and
  // then insert them.
  if (all_cols && all_rows)
    {
    Mat<eT> tmp(m.n_rows, m.n_cols, fill::randu);
    (*this).operator=(tmp);
    }
  else if (all_cols)
    {
    unwrap<T1> U1(base_ri.get_ref());

    extract_subview<typename unwrap<T1>::stored_type> E1(U1.M);

    coot_debug_check( E1.M.n_rows != 1 && E1.M.n_cols != 1, "Mat::rows(): row indices must be a vector" );

    Mat<eT> tmp(E1.M.n_elem, m.n_cols, fill::randu);
    (*this).operator=(tmp);
    }
  else if (all_rows)
    {
    unwrap<T2> U2(base_ci.get_ref());

    extract_subview<typename unwrap<T2>::stored_type> E2(U2.M);

    coot_debug_check( E2.M.n_rows != 1 && E2.M.n_cols != 1, "Mat::cols(): column indices must be a vector" );

    Mat<eT> tmp(m.n_rows, E2.M.n_elem, fill::randu);
    (*this).operator=(tmp);
    }
  else
    {
    unwrap<T1> U1(base_ri.get_ref());
    unwrap<T2> U2(base_ci.get_ref());

    extract_subview<typename unwrap<T1>::stored_type> E1(U1.M);
    extract_subview<typename unwrap<T2>::stored_type> E2(U2.M);

    coot_debug_check( E1.M.n_rows != 1 && E1.M.n_cols != 1, "Mat::elem(): row indices must be a vector" );
    coot_debug_check( E2.M.n_rows != 1 && E2.M.n_cols != 1, "Mat::elem(): column indices must be a vector" );

    Mat<eT> tmp(E1.M.n_elem, E2.M.n_elem, fill::randu);
    (*this).operator=(tmp);
    }
  }



template<typename eT, typename T1, typename T2>
inline
void
subview_elem2<eT,T1,T2>::randn()
  {
  coot_extra_debug_sigprint();

  // until we are able to index subviews in any arbitrary way in a generated kernel,
  // we use a slow implementation where we generate all the random numbers and
  // then insert them.
  if (all_cols && all_rows)
    {
    Mat<eT> tmp(m.n_rows, m.n_cols, fill::randn);
    (*this).operator=(tmp);
    }
  else if (all_cols)
    {
    unwrap<T1> U1(base_ri.get_ref());

    extract_subview<typename unwrap<T1>::stored_type> E1(U1.M);

    coot_debug_check( E1.M.n_rows != 1 && E1.M.n_cols != 1, "Mat::rows(): row indices must be a vector" );

    Mat<eT> tmp(E1.M.n_elem, m.n_cols, fill::randn);
    (*this).operator=(tmp);
    }
  else if (all_rows)
    {
    unwrap<T2> U2(base_ci.get_ref());

    extract_subview<typename unwrap<T2>::stored_type> E2(U2.M);

    coot_debug_check( E2.M.n_rows != 1 && E2.M.n_cols != 1, "Mat::cols(): column indices must be a vector" );

    Mat<eT> tmp(m.n_rows, E2.M.n_elem, fill::randn);
    (*this).operator=(tmp);
    }
  else
    {
    unwrap<T1> U1(base_ri.get_ref());
    unwrap<T2> U2(base_ci.get_ref());

    extract_subview<typename unwrap<T1>::stored_type> E1(U1.M);
    extract_subview<typename unwrap<T2>::stored_type> E2(U2.M);

    coot_debug_check( E1.M.n_rows != 1 && E1.M.n_cols != 1, "Mat::elem(): row indices must be a vector" );
    coot_debug_check( E2.M.n_rows != 1 && E2.M.n_cols != 1, "Mat::elem(): column indices must be a vector" );

    Mat<eT> tmp(E1.M.n_elem, E2.M.n_elem, fill::randn);
    (*this).operator=(tmp);
    }
  }



//
//



template<typename eT, typename T1, typename T2>
inline
void
subview_elem2<eT,T1,T2>::replace(const eT old_val, const eT new_val)
  {
  coot_extra_debug_sigprint();

  Mat<eT> tmp(*this);

  // ugly, slow implementation until we have kernels that can handle any kind of subview addressing
  tmp.replace(old_val, new_val);

  (*this).operator=(tmp);
  }



//template<typename eT, typename T1, typename T2>
//inline
//void
//subview_elem2<eT,T1,T2>::clean(const pod_type threshold)
//  {
//  coot_extra_debug_sigprint();
//
//  Mat<eT> tmp(*this);
//
//  tmp.clean(threshold);
//
//  (*this).operator=(tmp);
//  }



template<typename eT, typename T1, typename T2>
inline
void
subview_elem2<eT,T1,T2>::clamp(const eT min_val, const eT max_val)
  {
  coot_extra_debug_sigprint();

  Mat<eT> tmp(*this);

  // ugly, slow implementation until we have kernels that can handle any kind of subview addressing
  tmp.clamp(min_val, max_val);

  (*this).operator=(tmp);
  }



template<typename eT, typename T1, typename T2>
inline
void
subview_elem2<eT,T1,T2>::fill(const eT val)
  {
  coot_extra_debug_sigprint();

  if (all_rows && all_cols)
    {
    // no subview at all?
    coot_rt_t::fill(m.get_dev_mem(false), val, m.n_rows, m.n_cols, 0, 0, m.n_rows);
    }
  else if (all_cols)
    {
    // we are only using the row indices
    const unwrap<T1> U(base_ri.get_ref());
    const extract_subview<typename unwrap<T1>::stored_type> E(U.M);

    coot_debug_check( E.M.n_rows != 1 && E.M.n_cols != 1, "Mat::rows(): column indices must be a vector" );

    // A bounds check is tedious but the only way that we can give the user an error.
    coot_debug_check( E.M.max() >= m.n_rows, "Mat::rows(): index out of bounds" );

    alias_wrapper<Mat<eT>, Mat<uword>> A(const_cast<Mat<eT>&>(m), E.M);

    coot_rt_t::fill_subview_elem2(A.get_dev_mem(false),
                                  E.M.get_dev_mem(false),
                                  dev_mem_t<uword>({{ nullptr, 0 }}),
                                  val,
                                  E.M.n_elem,
                                  A.get_n_cols(),
                                  A.get_n_rows());
    }
  else if (all_rows)
    {
    // we are only using the column indices
    const unwrap<T2> U(base_ci.get_ref());
    const extract_subview<typename unwrap<T2>::stored_type> E(U.M);

    coot_debug_check( E.M.n_rows != 1 && E.M.n_cols != 1, "Mat::cols(): column indices must be a vector" );

    // A bounds check is tedious but the only way that we can give the user an error.
    coot_debug_check( E.M.max() >= m.n_rows, "Mat::cols(): index out of bounds" );

    alias_wrapper<Mat<eT>, Mat<uword>> A(const_cast<Mat<eT>&>(m), E.M);

    coot_rt_t::fill_subview_elem2(A.get_dev_mem(false),
                                  dev_mem_t<uword>({{ nullptr, 0 }}),
                                  E.M.get_dev_mem(false),
                                  val,
                                  A.get_n_rows(),
                                  E.M.n_elem,
                                  A.get_n_rows());
    }
  else
    {
    // we have to unwrap both sets of indices
    const unwrap<T1> U1(base_ri.get_ref());
    const extract_subview<typename unwrap<T1>::stored_type> E1(U1.M);

    const unwrap<T2> U2(base_ci.get_ref());
    const extract_subview<typename unwrap<T2>::stored_type> E2(U2.M);

    coot_debug_check( E1.M.n_rows != 1 && E1.M.n_cols != 1, "Mat::elem(): row indices must be a vector" );
    coot_debug_check( E2.M.n_rows != 1 && E2.M.n_cols != 1, "Mat::elem(): column indices must be a vector" );

    // A bounds check is tedious but the only way that we can give the user an error.
    coot_debug_check( E1.M.max() >= m.n_rows, "Mat::elem(): row index out of bounds" );
    coot_debug_check( E2.M.max() >= m.n_cols, "Mat::elem(): column index out of bounds" );

    alias_wrapper<Mat<eT>, Mat<uword>, Mat<uword>> A(const_cast<Mat<eT>&>(m), E1.M, E2.M);

    coot_rt_t::fill_subview_elem2(A.get_dev_mem(false),
                                  E1.M.get_dev_mem(false),
                                  E2.M.get_dev_mem(false),
                                  val,
                                  E1.M.n_elem,
                                  E2.M.n_elem,
                                  A.get_n_rows());
    }
  }



template<typename eT, typename T1, typename T2>
inline
void
subview_elem2<eT,T1,T2>::zeros()
  {
  coot_extra_debug_sigprint();

  (*this).fill(eT(0));
  }



template<typename eT, typename T1, typename T2>
inline
void
subview_elem2<eT,T1,T2>::ones()
  {
  coot_extra_debug_sigprint();

  (*this).fill(eT(1));
  }



template<typename eT, typename T1, typename T2>
inline
void
subview_elem2<eT,T1,T2>::operator+= (const eT val)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::equ_array_plus_scalar_sve2, val, (eT) 0);
  }



template<typename eT, typename T1, typename T2>
inline
void
subview_elem2<eT,T1,T2>::operator-= (const eT val)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::equ_array_minus_scalar_post_sve2, val, (eT) 0);
  }



template<typename eT, typename T1, typename T2>
inline
void
subview_elem2<eT,T1,T2>::operator*= (const eT val)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::equ_array_mul_scalar_sve2, val, (eT) 1);
  }



template<typename eT, typename T1, typename T2>
inline
void
subview_elem2<eT,T1,T2>::operator/= (const eT val)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::equ_array_div_scalar_post_sve2, val, (eT) 1);
  }



//
//



template<typename eT, typename T1, typename T2>
template<typename T3, typename T4>
inline
void
subview_elem2<eT,T1,T2>::operator= (const subview_elem2<eT,T3,T4>& x)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::inplace_sve2_eq_sve2, x);
  }



// ! work around compiler bugs
template<typename eT, typename T1, typename T2>
inline
void
subview_elem2<eT,T1,T2>::operator= (const subview_elem2<eT,T1,T2>& x)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::inplace_sve2_eq_sve2, x);
  }



template<typename eT, typename T1, typename T2>
template<typename T3, typename T4>
inline
void
subview_elem2<eT,T1,T2>::operator+= (const subview_elem2<eT,T3,T4>& x)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::inplace_sve2_plus_sve2, x);
  }



template<typename eT, typename T1, typename T2>
template<typename T3, typename T4>
inline
void
subview_elem2<eT,T1,T2>::operator-= (const subview_elem2<eT,T3,T4>& x)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::inplace_sve2_minus_sve2, x);
  }



template<typename eT, typename T1, typename T2>
template<typename T3, typename T4>
inline
void
subview_elem2<eT,T1,T2>::operator%= (const subview_elem2<eT,T3,T4>& x)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::inplace_sve2_mul_sve2, x);
  }



template<typename eT, typename T1, typename T2>
template<typename T3, typename T4>
inline
void
subview_elem2<eT,T1,T2>::operator/= (const subview_elem2<eT,T3,T4>& x)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::inplace_sve2_div_sve2, x);
  }



template<typename eT, typename T1, typename T2>
template<typename expr>
inline
void
subview_elem2<eT,T1,T2>::operator= (const Base<eT,expr>& x)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::inplace_sve2_eq_array, x);
  }



template<typename eT, typename T1, typename T2>
template<typename expr>
inline
void
subview_elem2<eT,T1,T2>::operator+= (const Base<eT,expr>& x)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::inplace_sve2_plus_array, x);
  }



template<typename eT, typename T1, typename T2>
template<typename expr>
inline
void
subview_elem2<eT,T1,T2>::operator-= (const Base<eT,expr>& x)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::inplace_sve2_minus_array, x);
  }



template<typename eT, typename T1, typename T2>
template<typename expr>
inline
void
subview_elem2<eT,T1,T2>::operator%= (const Base<eT,expr>& x)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::inplace_sve2_mul_array, x);
  }



template<typename eT, typename T1, typename T2>
template<typename expr>
inline
void
subview_elem2<eT,T1,T2>::operator/= (const Base<eT,expr>& x)
  {
  coot_extra_debug_sigprint();

  inplace_op(twoway_kernel_id::inplace_sve2_div_array, x);
  }



//
//



template<typename eT, typename T1, typename T2>
inline
void
subview_elem2<eT,T1,T2>::extract(Mat<eT>& actual_out, const subview_elem2<eT,T1,T2>& in)
  {
  coot_extra_debug_sigprint();

  if (in.all_rows && in.all_cols)
    {
    // not a subview at all?
    actual_out = in.m;
    }
  else if (in.all_cols)
    {
    // we are only using the row indices
    const unwrap<T1> U(in.base_ri.get_ref());
    const extract_subview<typename unwrap<T1>::stored_type> E(U.M);

    coot_debug_check( E.M.n_rows != 1 && E.M.n_cols != 1, "Mat::rows(): column indices must be a vector" );

    // A bounds check is tedious but the only way that we can give the user an error.
    coot_debug_check( E.M.max() >= in.m.n_rows, "Mat::rows(): index out of bounds" );

    alias_wrapper<Mat<eT>, Mat<eT>, Mat<uword>> A(actual_out, const_cast<Mat<eT>&>(in.m), E.M);

    A.use.set_size(E.M.n_elem, in.m.n_cols);

    coot_rt_t::extract_subview_elem2(A.get_dev_mem(false),
                                     in.m.get_dev_mem(false),
                                     E.M.get_dev_mem(false),
                                     dev_mem_t<uword>({{ nullptr, 0 }}), // no column indices
                                     E.M.n_elem,
                                     in.m.n_cols,
                                     A.use.n_rows,
                                     in.m.n_rows);
    }
  else if (in.all_rows)
    {
    // we are only using the column indices
    const unwrap<T2> U(in.base_ci.get_ref());
    const extract_subview<typename unwrap<T2>::stored_type> E(U.M);

    coot_debug_check( E.M.n_rows != 1 && E.M.n_cols != 1, "Mat::cols(): row indices must be a vector" );

    // A bounds check is tedious but the only way that we can give the user an error.
    coot_debug_check( E.M.max() >= in.m.n_cols, "Mat::cols(): index out of bounds" );

    alias_wrapper<Mat<eT>, Mat<eT>, Mat<uword>> A(actual_out, const_cast<Mat<eT>&>(in.m), E.M);

    A.use.set_size(in.m.n_rows, E.M.n_elem);

    coot_rt_t::extract_subview_elem2(A.get_dev_mem(false),
                                     in.m.get_dev_mem(false),
                                     dev_mem_t<uword>({{ nullptr, 0 }}), // no row indices
                                     E.M.get_dev_mem(false),
                                     in.m.n_rows,
                                     E.M.n_elem,
                                     A.use.n_rows,
                                     in.m.n_rows);
    }
  else
    {
    // we have to unwrap both sets of indices
    const unwrap<T1> U1(in.base_ri.get_ref());
    const extract_subview<typename unwrap<T1>::stored_type> E1(U1.M);

    const unwrap<T2> U2(in.base_ci.get_ref());
    const extract_subview<typename unwrap<T2>::stored_type> E2(U2.M);

    coot_debug_check( E1.M.n_rows != 1 && E1.M.n_cols != 1, "Mat::elem(): row indices must be a vector" );
    coot_debug_check( E2.M.n_rows != 1 && E2.M.n_cols != 1, "Mat::elem(): column indices must be a vector" );

    // A bounds check is tedious but the only way that we can give the user an error.
    coot_debug_check( E1.M.max() >= in.m.n_rows, "Mat::elem(): row index out of bounds" );
    coot_debug_check( E2.M.max() >= in.m.n_cols, "Mat::elem(): column index out of bounds" );

    alias_wrapper<Mat<eT>, Mat<eT>, Mat<uword>, Mat<uword>> A(actual_out, const_cast<Mat<eT>&>(in.m), E1.M, E2.M);

    A.use.set_size(E1.M.n_elem, E2.M.n_elem);

    coot_rt_t::extract_subview_elem2(A.get_dev_mem(false),
                                     in.m.get_dev_mem(false),
                                     E1.M.get_dev_mem(false),
                                     E2.M.get_dev_mem(false),
                                     E1.M.n_elem,
                                     E2.M.n_elem,
                                     A.use.n_rows,
                                     in.m.n_rows);
    }
  }
