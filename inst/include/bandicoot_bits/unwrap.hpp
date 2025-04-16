// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2017-2023 Ryan Curtin (https://www.ratml.org)
// Copyright 2008-2023 Conrad Sanderson (https://conradsanderson.id.au)
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



//
// The unwrap<> struct is a tool to actually evaluate expressions.
// Construction of an unwrap<T1> object called `U` for some expression T1
// will cause `U.M` to hold the result of the expression being evaluated:
//
// unwrap<T1> U(expr); // expr has type T1
//
// Importantly, `U.M` may have type `Mat` or `subview`!  That is, `unwrap`
// does *not* extract subviews.  Make sure your code can handle that,
// and also that you test both situations.  Extracting subviews can be done
// with the `extract_subview` struct (see extract_subview.hpp).
//
// If you are going to call a backend function that accepts subviews (such
// as eop_scalar()), you can use the convenience functions below to compute
// offsets agnostic to whether `U.M` is a `Mat` or `subview`:
//
//  - get_row_offset()  (0 for `Mat`, `U.M.aux_row1` for `subview`s)
//  - get_col_offset()  (0 for `Mat`, `U.M.aux_col1` for `subview`s)
//  - get_M_n_rows()    (`U.M.n_rows` for `Mat`, `U.M.m.n_rows` for `subview`s)
//  - get_dev_mem(sync) (`U.M.get_dev_mem(sync)` for `Mat`, `U.M.m.get_dev_mem(sync)` for `subview`s)
//
template<typename T1>
struct unwrap
  {
  typedef typename T1::elem_type eT;
  typedef Mat<eT>                stored_type;

  inline
  unwrap(const T1& A)
    : M(A)
    {
    coot_extra_debug_sigprint();
    }

  Mat<eT> M;

  template<typename T2>
  coot_inline bool is_alias(const T2& X) const { return coot::is_alias(X, M); }

  constexpr inline uword         get_row_offset()                    const { return 0; }
  constexpr inline uword         get_col_offset()                    const { return 0; }
            inline uword         get_M_n_rows()                      const { return M.n_rows; }
            inline dev_mem_t<eT> get_dev_mem(const bool synchronise) const { return M.get_dev_mem(synchronise); }
  };



template<typename eT>
struct unwrap< Mat<eT> >
  {
  typedef Mat<eT> stored_type;

  inline
  unwrap(const Mat<eT>& A)
    : M(A)
    {
    coot_extra_debug_sigprint();
    }

  const Mat<eT>& M;

  template<typename T2>
  coot_inline bool is_alias(const T2& X) const { return coot::is_alias(X, M); }

  constexpr inline uword         get_row_offset()                    const { return 0; }
  constexpr inline uword         get_col_offset()                    const { return 0; }
            inline uword         get_M_n_rows()                      const { return M.n_rows; }
            inline dev_mem_t<eT> get_dev_mem(const bool synchronise) const { return M.get_dev_mem(synchronise); }
  };



template<typename eT>
struct unwrap< Row<eT> >
  {
  typedef Row<eT> stored_type;

  inline
  unwrap(const Row<eT>& A)
    : M(A)
    {
    coot_extra_debug_sigprint();
    }

  const Row<eT>& M;

  template<typename T2>
  coot_inline bool is_alias(const T2& X) const { return coot::is_alias(X, M); }

  constexpr inline uword         get_row_offset()                    const { return 0; }
  constexpr inline uword         get_col_offset()                    const { return 0; }
            inline uword         get_M_n_rows()                      const { return M.n_rows; }
            inline dev_mem_t<eT> get_dev_mem(const bool synchronise) const { return M.get_dev_mem(synchronise); }
  };



template<typename eT>
struct unwrap< Col<eT> >
  {
  typedef Col<eT> stored_type;

  inline
  unwrap(const Col<eT>& A)
    : M(A)
    {
    coot_extra_debug_sigprint();
    }

  const Col<eT>& M;

  template<typename T2>
  coot_inline bool is_alias(const T2& X) const { return coot::is_alias(X, M); }

  constexpr inline uword         get_row_offset()                    const { return 0; }
  constexpr inline uword         get_col_offset()                    const { return 0; }
            inline uword         get_M_n_rows()                      const { return M.n_rows; }
            inline dev_mem_t<eT> get_dev_mem(const bool synchronise) const { return M.get_dev_mem(synchronise); }
  };



template<typename eT>
struct unwrap< subview<eT> >
  {
  typedef subview<eT> stored_type;

  inline
  unwrap(const subview<eT>& A)
    : M(A)
    {
    coot_extra_debug_sigprint();
    }

  const subview<eT>& M;

  template<typename T2>
  coot_inline bool is_alias(const T2& X) const { return coot::is_alias(X, M); }

  inline uword get_row_offset() const { return M.aux_row1; }
  inline uword get_col_offset() const { return M.aux_col1; }
  inline uword get_M_n_rows()   const { return M.m.n_rows; }
  inline dev_mem_t<eT> get_dev_mem(const bool synchronise) const { return M.m.get_dev_mem(synchronise); }
  };



// Since this is not no_conv_unwrap, we have to ensure that the stored_type has the correct out_eT.
template<typename out_eT, typename T1>
struct unwrap< mtOp<out_eT, T1, mtop_conv_to> >
  {
  typedef Mat<out_eT> stored_type;

  inline
  unwrap(const mtOp<out_eT, T1, mtop_conv_to>& A)
    : M(A)
    {
    coot_extra_debug_sigprint();
    }

  Mat<out_eT> M;

  template<typename T2>
  coot_inline bool is_alias(const T2& X) const { return coot::is_alias(X, M); }

  constexpr inline uword             get_row_offset()                    const { return 0; }
  constexpr inline uword             get_col_offset()                    const { return 0; }
            inline uword             get_M_n_rows()                      const { return M.n_rows; }
            inline dev_mem_t<out_eT> get_dev_mem(const bool synchronise) const { return M.get_dev_mem(synchronise); }
  };



template<typename out_eT, typename T1, typename T2, typename mtglue_type>
struct unwrap< mtGlue<out_eT, T1, T2, mtglue_type> >
  {
  typedef Mat<out_eT> stored_type;

  inline
  unwrap(const mtGlue<out_eT, T1, T2, mtglue_type>& A)
    : M(A)
    {
    coot_extra_debug_sigprint();
    }

  Mat<out_eT> M;

  template<typename T3>
  coot_inline bool is_alias(const T3& X) const { return coot::is_alias(X, M); }

  constexpr inline uword             get_row_offset()                    const { return 0; }
  constexpr inline uword             get_col_offset()                    const { return 0; }
            inline uword             get_M_n_rows()                      const { return M.n_rows; }
            inline dev_mem_t<out_eT> get_dev_mem(const bool synchronise) const { return M.get_dev_mem(synchronise); }
  };



//
//
//



template<typename T1>
struct partial_unwrap
  {
  typedef typename T1::elem_type eT;
  typedef Mat<eT>                stored_type;

  inline
  partial_unwrap(const T1& A)
    : M(A)
    {
    coot_extra_debug_sigprint();
    }

  constexpr eT get_val() const { return eT(1); }

  template<typename T2>
  constexpr bool is_alias(const T2&) const { return false; }

  static constexpr bool do_trans = false;
  static constexpr bool do_times = false;

  const Mat<eT> M;
  };



template<typename eT>
struct partial_unwrap< Mat<eT> >
  {
  typedef Mat<eT> stored_type;

  inline
  partial_unwrap(const Mat<eT>& A)
    : M(A)
    {
    coot_extra_debug_sigprint();
    }

  constexpr eT get_val() const { return eT(1); }

  template<typename T2>
  constexpr bool is_alias(const T2&) const { return false; }

  static constexpr bool do_trans = false;
  static constexpr bool do_times = false;

  const Mat<eT>& M;
  };



template<typename eT>
struct partial_unwrap< subview<eT> >
  {
  typedef subview<eT> stored_type;

  inline
  partial_unwrap(const subview<eT>& A)
    : M(A)
    {
    coot_extra_debug_sigprint();
    }

  constexpr eT get_val() const { return eT(1); }

  template<typename T2>
  constexpr bool is_alias(const T2&) const { return false; }

  static constexpr bool do_trans = false;
  static constexpr bool do_times = false;

  const subview<eT>& M;
  };



template<typename T1>
struct partial_unwrap< Op<T1, op_htrans> >
  {
  typedef typename T1::elem_type eT;
  typedef Mat<eT>                stored_type;

  inline
  partial_unwrap(const Op<T1, op_htrans>& A)
    : M(A.m)
    {
    coot_extra_debug_sigprint();
    }

  constexpr eT get_val() const { return eT(1); }

  template<typename T2>
  constexpr bool is_alias(const T2&) const { return false; }

  static constexpr bool do_trans = true;
  static constexpr bool do_times = false;

  const Mat<eT> M;
  };



template<typename eT>
struct partial_unwrap< Op< Mat<eT>, op_htrans> >
  {
  typedef Mat<eT> stored_type;

  inline
  partial_unwrap(const Op< Mat<eT>, op_htrans>& A)
    : M(A.m)
    {
    coot_extra_debug_sigprint();
    }

  constexpr eT get_val() const { return eT(1); }

  template<typename T2>
  coot_inline bool is_alias(const T2& X) const { return coot::is_alias(X, M); }

  static constexpr bool do_trans = true;
  static constexpr bool do_times = false;

  const Mat<eT>& M;
  };



template<typename eT>
struct partial_unwrap< Op< subview<eT>, op_htrans> >
  {
  typedef subview<eT> stored_type;

  inline
  partial_unwrap(const Op< subview<eT>, op_htrans>& A)
    : M(A.m)
    {
    coot_extra_debug_sigprint();
    }

  constexpr eT get_val() const { return eT(1); }

  template<typename T2>
  coot_inline bool is_alias(const T2& X) const { return coot::is_alias(X, M); }

  static constexpr bool do_trans = true;
  static constexpr bool do_times = false;

  const subview<eT>& M;
  };



template<typename T1>
struct partial_unwrap< Op<T1, op_htrans2> >
  {
  typedef typename T1::elem_type eT;
  typedef Mat<eT> stored_type;

  inline
  partial_unwrap(const Op<T1, op_htrans2>& A)
    : val(A.aux)
    , M  (A.m)
    {
    coot_extra_debug_sigprint();
    }

  coot_inline eT get_val() const { return val; }

  template<typename T2>
  coot_inline bool is_alias(const T2& X) const { return coot::is_alias(X, M); }

  static constexpr bool do_trans = true;
  static constexpr bool do_times = true;

  const eT      val;
  const Mat<eT> M;
  };



template<typename eT>
struct partial_unwrap< Op< Mat<eT>, op_htrans2> >
  {
  typedef Mat<eT> stored_type;

  inline
  partial_unwrap(const Op<Mat<eT>, op_htrans2>& A)
    : val(A.aux)
    , M  (A.m)
    {
    coot_extra_debug_sigprint();
    }

  coot_inline eT get_val() const { return val; }

  template<typename T2>
  coot_inline bool is_alias(const T2& X) const { return coot::is_alias(X, M); }

  static constexpr bool do_trans = true;
  static constexpr bool do_times = true;

  const eT       val;
  const Mat<eT>& M;
  };



template<typename eT>
struct partial_unwrap< Op< subview<eT>, op_htrans2> >
  {
  typedef subview<eT> stored_type;

  inline
  partial_unwrap(const Op<subview<eT>, op_htrans2>& A)
    : val(A.aux)
    , M  (A.m)
    {
    coot_extra_debug_sigprint();
    }

  coot_inline eT get_val() const { return val; }

  template<typename T2>
  coot_inline bool is_alias(const T2& X) const { return coot::is_alias(X, M); }

  static constexpr bool do_trans = true;
  static constexpr bool do_times = true;

  const eT           val;
  const subview<eT>& M;
  };



template<typename eT>
struct partial_unwrap< eOp<Mat<eT>, eop_scalar_times> >
  {
  typedef Mat<eT> stored_type;

  inline
  partial_unwrap(const eOp<Mat<eT>, eop_scalar_times>& A)
    : val(A.aux)
    , M  (A.m.Q)
    {
    coot_extra_debug_sigprint();
    }

  coot_inline eT get_val() const { return val; }

  template<typename T2>
  coot_inline bool is_alias(const T2& X) const { return coot::is_alias(X, M); }

  static constexpr bool do_trans = false;
  static constexpr bool do_times = true;

  const eT       val;
  const Mat<eT>& M;
  };



template<typename eT>
struct partial_unwrap< eOp<subview<eT>, eop_scalar_times> >
  {
  typedef subview<eT> stored_type;

  inline
  partial_unwrap(const eOp<subview<eT>, eop_scalar_times>& A)
    : val(A.aux)
    , M  (A.m.Q)
    {
    coot_extra_debug_sigprint();
    }

  coot_inline eT get_val() const { return val; }

  template<typename T2>
  coot_inline bool is_alias(const T2& X) const { return coot::is_alias(X, M); }

  static constexpr bool do_trans = false;
  static constexpr bool do_times = true;

  const eT           val;
  const subview<eT>& M;
  };



template<typename eT>
struct partial_unwrap< eOp<Mat<eT>, eop_neg> >
  {
  typedef Mat<eT> stored_type;

  inline
  partial_unwrap(const eOp<Mat<eT>, eop_neg>& A)
    : M(A.m.Q)
    {
    coot_extra_debug_sigprint();
    }

  constexpr eT get_val() const { return eT(-1); }

  template<typename T2>
  coot_inline bool is_alias(const T2& X) const { return coot::is_alias(X, M); }

  static constexpr bool do_trans = false;
  static constexpr bool do_times = true;

  const Mat<eT>& M;
  };



template<typename eT>
struct partial_unwrap< eOp<subview<eT>, eop_neg> >
  {
  typedef subview<eT> stored_type;

  inline
  partial_unwrap(const eOp<subview<eT>, eop_neg>& A)
    : M(A.m.Q)
    {
    coot_extra_debug_sigprint();
    }

  constexpr eT get_val() const { return eT(-1); }

  template<typename T2>
  coot_inline bool is_alias(const T2& X) const { return coot::is_alias(X, M); }

  static constexpr bool do_trans = false;
  static constexpr bool do_times = true;

  const subview<eT>& M;
  };



// To partially unwrap a conversion operation, perform only the conversion---and partially unwrap everything else.
template<typename eT, typename T1, typename mtop_type>
struct partial_unwrap< mtOp<eT, T1, mtop_type> >
  {
  typedef Mat<eT> stored_type;

  inline
  partial_unwrap(const mtOp<eT, T1, mtop_type>& X)
    : Q(X.q)
    // It's possible this can miss some opportunities to merge the conversion with the operation,
    // but we don't currently have a great way to capture the not-yet-unwrapped type held in any T1.
    , M(mtOp<eT, typename partial_unwrap<T1>::stored_type, mtop_type>(Q.M))
    {
    coot_extra_debug_sigprint();
    }

  coot_inline eT get_val() const { return Q.get_val(); }

  template<typename T2>
  constexpr bool is_alias(const T2&) const { return false; }

  static constexpr bool do_trans = partial_unwrap<T1>::do_trans;
  static constexpr bool do_times = partial_unwrap<T1>::do_times;

  const partial_unwrap<T1> Q;
  Mat<eT> M;
  };
