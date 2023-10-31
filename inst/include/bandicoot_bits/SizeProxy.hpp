// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2020 Ryan Curtin (http://www.ratml.org/)
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


// The SizeProxy class is meant, for now, to provide an interface to partially unwrap types for operations,
// so that the sizes of the operand(s) can be known when the operations are created.  Operations should never be
// unwrapped when a SizeProxy is created.  (If you want to audit this, check the stored_type for each struct
// specialization.)  The underlying object Q should be used for any actual operations.
//
// This differs from Armadillo's Proxy because all GPU operations work directly on blocks of memory,
// which will generally force a full unwrap when any actual operation happens.
//
// The SizeProxy class defines the following types and methods:
//
// elem_type        = the type of the elements obtained from object Q
// pod_type         = the underlying type of elements if elem_type is std::complex
// stored_type      = the type of the Q object
//
// is_row           = boolean indicating whether the Q object can be treated a row vector
// is_col           = boolean indicating whether the Q object can be treated a column vector
//
// Q                = object that can be unwrapped via the unwrap family of classes (ie. Q must be convertible to Mat)
//
// get_n_rows()     = return the number of rows in Q
// get_n_cols()     = return the number of columns in Q
// get_n_elem()     = return the number of elements in Q
//
// Note that Q may not necessarily have the same row/col dimensions as the SizeProxy object!
// So, do all size checks with get_n_rows()/get_n_cols()/get_n_elem(), then use Q to get a memory pointer to do operations with.

template<typename eT>
class SizeProxy< Mat<eT> >
  {
  public:

  typedef eT                                       elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef Mat<eT>                                  stored_type;

  static constexpr bool is_row = false;
  static constexpr bool is_col = false;

  coot_aligned const Mat<eT>& Q;

  inline explicit SizeProxy(const Mat<eT>& A)
    : Q(A)
    {
    coot_extra_debug_sigprint();
    }

  coot_inline uword get_n_rows() const { return Q.n_rows; }
  coot_inline uword get_n_cols() const { return Q.n_cols; }
  coot_inline uword get_n_elem() const { return Q.n_elem; }
  };



template<typename eT>
class SizeProxy< Col<eT> >
  {
  public:

  typedef eT                                       elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef Col<eT>                                  stored_type;

  static constexpr bool is_row = false;
  static constexpr bool is_col = true;

  coot_aligned const Col<eT>& Q;

  inline explicit SizeProxy(const Col<eT>& A)
    : Q(A)
    {
    coot_extra_debug_sigprint();
    }

  coot_inline uword get_n_rows() const { return Q.n_rows; }
  constexpr   uword get_n_cols() const { return uword(1); }
  coot_inline uword get_n_elem() const { return Q.n_elem; }
  };



template<typename eT>
class SizeProxy< Row<eT> >
  {
  public:

  typedef eT                                       elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef Row<eT>                                  stored_type;

  static constexpr bool is_row = true;
  static constexpr bool is_col = false;

  coot_aligned const Row<eT>& Q;

  inline explicit SizeProxy(const Row<eT>& A)
    : Q(A)
    {
    coot_extra_debug_sigprint();
    }

  constexpr   uword get_n_rows() const { return uword(1); }
  coot_inline uword get_n_cols() const { return Q.n_cols; }
  coot_inline uword get_n_elem() const { return Q.n_elem; }
  };



template<typename eT>
class SizeProxy< subview<eT> >
  {
  public:

  typedef eT                                       elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef subview<eT>                              stored_type;

  static constexpr bool is_row = false;
  static constexpr bool is_col = false;

  coot_aligned const subview<eT>& Q;

  inline explicit SizeProxy(const subview<eT>& A)
    : Q(A)
    {
    coot_extra_debug_sigprint();
    }

  coot_inline uword get_n_rows() const { return Q.n_rows; }
  coot_inline uword get_n_cols() const { return Q.n_cols; }
  coot_inline uword get_n_elem() const { return Q.n_elem; }
  };



template<typename eT>
class SizeProxy< subview_col<eT> >
  {
  public:

  typedef eT                                       elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef subview_col<eT>                          stored_type;

  static constexpr bool is_row = false;
  static constexpr bool is_col = true;

  coot_aligned const subview_col<eT>& Q;

  inline explicit SizeProxy(const subview_col<eT>& A)
    : Q(A)
    {
    coot_extra_debug_sigprint();
    }

  coot_inline uword get_n_rows() const { return Q.n_rows; }
  constexpr   uword get_n_cols() const { return uword(1); }
  coot_inline uword get_n_elem() const { return Q.n_elem; }
  };



template<typename eT>
class SizeProxy< subview_row<eT> >
  {
  public:

  typedef eT                                       elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef subview_row<eT>                          stored_type;

  static constexpr bool is_row = true;
  static constexpr bool is_col = false;

  coot_aligned const subview_row<eT>& Q;

  inline explicit SizeProxy(const subview_row<eT>& A)
    : Q(A)
    {
    coot_extra_debug_sigprint();
    }

  constexpr   uword get_n_rows() const { return uword(1); }
  coot_inline uword get_n_cols() const { return Q.n_cols; }
  coot_inline uword get_n_elem() const { return Q.n_elem; }
  };



template<typename eT>
class SizeProxy< diagview<eT> >
  {
  public:

  typedef eT                                       elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef diagview<eT>                             stored_type;

  static constexpr bool is_row = false;
  static constexpr bool is_col = true;

  coot_aligned const diagview<eT>& Q;

  inline explicit SizeProxy(const diagview<eT>& A)
    : Q(A)
    {
    coot_extra_debug_sigprint();
    }

  coot_inline uword get_n_rows() const { return Q.n_rows; }
  constexpr   uword get_n_cols() const { return uword(1); }
  coot_inline uword get_n_elem() const { return Q.n_elem; }
  };



// eOp
template<typename T1, typename eop_type>
class SizeProxy< eOp<T1, eop_type> >
  {
  public:

  typedef typename T1::elem_type                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef eOp<T1, eop_type>                        stored_type;

  static constexpr bool is_row = eOp<T1, eop_type>::is_row;
  static constexpr bool is_col = eOp<T1, eop_type>::is_col;

  coot_aligned const eOp<T1, eop_type>& Q;

  inline explicit SizeProxy(const eOp<T1, eop_type>& A)
    : Q(A)
    {
    coot_extra_debug_sigprint();
    }

  coot_inline uword get_n_rows() const { return Q.get_n_rows(); }
  coot_inline uword get_n_cols() const { return Q.get_n_cols(); }
  coot_inline uword get_n_elem() const { return Q.get_n_elem(); }
  };



// eGlue
template<typename T1, typename T2, typename eglue_type>
class SizeProxy< eGlue<T1, T2, eglue_type> >
  {
  public:

  typedef typename T1::elem_type                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef eGlue<T1, T2, eglue_type>                stored_type;

  static constexpr bool is_row = eGlue<T1, T2, eglue_type>::is_row;
  static constexpr bool is_col = eGlue<T1, T2, eglue_type>::is_col;

  coot_aligned const eGlue<T1, T2, eglue_type>& Q;

  inline explicit SizeProxy(const eGlue<T1, T2, eglue_type>& A)
    : Q(A)
    {
    coot_extra_debug_sigprint();
    }

  coot_inline uword get_n_rows() const { return Q.get_n_rows(); }
  coot_inline uword get_n_cols() const { return Q.get_n_cols(); }
  coot_inline uword get_n_elem() const { return Q.get_n_elem(); }
  };



// We expect that each Op can compute its own size based on its arguments.
template<typename T1, typename op_type>
class SizeProxy< Op<T1, op_type> >
  {
  public:

  typedef typename T1::elem_type                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef Op<T1, op_type>                          stored_type;

  static constexpr bool is_row = Op<T1, op_type>::is_row;
  static constexpr bool is_col = Op<T1, op_type>::is_col;

  coot_aligned const SizeProxy<T1> S;
  coot_aligned const Op<T1, op_type>& Q;

  inline explicit SizeProxy(const Op<T1, op_type>& A)
    : S(A.m)
    , Q(A)
    {
    coot_extra_debug_sigprint();
    }

  coot_inline uword get_n_rows() const { return op_type::compute_n_rows(Q, S.get_n_rows(), S.get_n_cols()); }
  coot_inline uword get_n_cols() const { return op_type::compute_n_cols(Q, S.get_n_rows(), S.get_n_cols()); }
  coot_inline uword get_n_elem() const { return get_n_rows() * get_n_cols(); }
  };



template<typename out_eT, typename T1, typename mtop_type>
class SizeProxy< mtOp<out_eT, T1, mtop_type> >
  {
  public:

  typedef out_eT                                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef mtOp<out_eT, T1, mtop_type>              stored_type;

  static constexpr bool is_row = mtOp<out_eT, T1, mtop_type>::is_row;
  static constexpr bool is_col = mtOp<out_eT, T1, mtop_type>::is_col;

  coot_aligned const SizeProxy<T1> S;
  coot_aligned const mtOp<out_eT, T1, mtop_type>& Q;

  inline explicit SizeProxy(const mtOp<out_eT, T1, mtop_type>& A)
    : S(A.q)
    , Q(A)
    {
    coot_extra_debug_sigprint();
    }

  coot_inline uword get_n_rows() const { return mtop_type::compute_n_rows(Q, S.get_n_rows(), S.get_n_cols()); }
  coot_inline uword get_n_cols() const { return mtop_type::compute_n_cols(Q, S.get_n_rows(), S.get_n_cols()); }
  coot_inline uword get_n_elem() const { return get_n_rows() * get_n_cols(); }
  };



template<typename T1, typename T2, typename glue_type>
class SizeProxy< Glue<T1, T2, glue_type> >
  {
  public:

  typedef typename T1::elem_type                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef Glue<T1, T2, glue_type>                  stored_type;

  static constexpr bool is_row = Glue<T1, T2, glue_type>::is_row;
  static constexpr bool is_col = Glue<T1, T2, glue_type>::is_col;

  coot_aligned const SizeProxy<T1> S1;
  coot_aligned const SizeProxy<T2> S2;
  coot_aligned const Glue<T1, T2, glue_type>& Q;

  inline explicit SizeProxy(const Glue<T1, T2, glue_type>& A)
    : S1(A.A)
    , S2(A.B)
    , Q(A)
    {
    coot_extra_debug_sigprint();
    }

  // Each glue_type must implement compute_n_rows() and compute_n_cols()
  coot_inline uword get_n_rows() const { return glue_type::compute_n_rows(Q, S1.get_n_rows(), S1.get_n_cols(), S2.get_n_rows(), S2.get_n_cols()); }
  coot_inline uword get_n_cols() const { return glue_type::compute_n_cols(Q, S1.get_n_rows(), S1.get_n_cols(), S2.get_n_rows(), S2.get_n_cols()); }
  coot_inline uword get_n_elem() const { return get_n_rows() * get_n_cols(); }
  };



template<typename out_eT, typename T1, typename T2, typename mtglue_type>
class SizeProxy< mtGlue<out_eT, T1, T2, mtglue_type> >
  {
  public:

  typedef out_eT                                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef mtGlue<out_eT, T1, T2, mtglue_type>      stored_type;

  static constexpr bool is_row = mtGlue<out_eT, T1, T2, mtglue_type>::is_row;
  static constexpr bool is_col = mtGlue<out_eT, T1, T2, mtglue_type>::is_col;

  coot_aligned const SizeProxy<T1> S1;
  coot_aligned const SizeProxy<T2> S2;
  coot_aligned const mtGlue<out_eT, T1, T2, mtglue_type>& Q;

  inline explicit SizeProxy(const mtGlue<out_eT, T1, T2, mtglue_type>& A)
    : S1(A.A)
    , S2(A.B)
    , Q(A)
    {
    coot_extra_debug_sigprint();
    }

  // Each mtglue_type must implement compute_n_rows() and compute_n_cols()
  coot_inline uword get_n_rows() const { return mtglue_type::compute_n_rows(Q, S1.get_n_rows(), S1.get_n_cols(), S2.get_n_rows(), S2.get_n_cols()); }
  coot_inline uword get_n_cols() const { return mtglue_type::compute_n_cols(Q, S1.get_n_rows(), S1.get_n_cols(), S2.get_n_rows(), S2.get_n_cols()); }
  coot_inline uword get_n_elem() const { return get_n_rows() * get_n_cols(); }
  };
