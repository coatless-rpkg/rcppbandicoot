// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2025 Ryan Curtin (http://www.ratml.org/)
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


// The SizeProxyCube class is meant, for now, to provide an interface to partially unwrap types for operations,
// so that the sizes of the operand(s) can be known when the operations are created.  Operations should never be
// unwrapped when a SizeProxyCube is created.  (If you want to audit this, check the stored_type for each struct
// specialization.)  The underlying object Q should be used for any actual operations.
//
// This differs from Armadillo's Proxy because all GPU operations work directly on blocks of memory,
// which will generally force a full unwrap when any actual operation happens.
//
// The SizeProxyCube class defines the following types and methods:
//
// elem_type        = the type of the elements obtained from object Q
// pod_type         = the underlying type of elements if elem_type is std::complex
// stored_type      = the type of the Q object
//
// Q                = object that can be unwrapped via the unwrap family of classes (ie. Q must be convertible to Mat)
//
// get_n_rows()       = return the number of rows in Q
// get_n_cols()       = return the number of columns in Q
// get_n_slices()     = return the number of slices in Q
// get_n_elem()       = return the number of elements in Q
// get_n_elem_slice() = return the number of elements per slice in Q
//
// Note that Q may not necessarily have the same row/col/slice dimensions as the SizeProxyCube object!
// So, do all size checks with get_n_rows()/get_n_cols()/get_n_slices()/get_n_elem()/get_n_elem_slice(), then use Q to get a memory pointer to do operations with.

template<typename eT>
class SizeProxyCube< Cube<eT> >
  {
  public:

  typedef eT                                       elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef Cube<eT>                                 stored_type;

  coot_aligned const Cube<eT>& Q;

  inline explicit SizeProxyCube(const Cube<eT>& A)
    : Q(A)
    {
    coot_extra_debug_sigprint();
    }

  coot_inline uword get_n_rows() const       { return Q.n_rows; }
  coot_inline uword get_n_cols() const       { return Q.n_cols; }
  coot_inline uword get_n_slices() const     { return Q.n_slices; }
  coot_inline uword get_n_elem() const       { return Q.n_elem; }
  coot_inline uword get_n_elem_slice() const { return Q.n_elem_slice; }
  };


template<typename eT>
class SizeProxyCube< subview_cube<eT> >
  {
  public:

  typedef eT                                       elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef subview_cube<eT>                         stored_type;

  coot_aligned const subview_cube<eT>& Q;

  inline explicit SizeProxyCube(const subview_cube<eT>& A)
    : Q(A)
    {
    coot_extra_debug_sigprint();
    }

  coot_inline uword get_n_rows() const       { return Q.n_rows; }
  coot_inline uword get_n_cols() const       { return Q.n_cols; }
  coot_inline uword get_n_slices() const     { return Q.n_slices; }
  coot_inline uword get_n_elem() const       { return Q.n_elem; }
  coot_inline uword get_n_elem_slice() const { return Q.n_elem_slice; }
  };



// eOpCube
template<typename T1, typename eop_type>
class SizeProxyCube< eOpCube<T1, eop_type> >
  {
  public:

  typedef typename T1::elem_type                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef eOpCube<T1, eop_type>                    stored_type;

  coot_aligned const eOpCube<T1, eop_type>& Q;

  inline explicit SizeProxyCube(const eOpCube<T1, eop_type>& A)
    : Q(A)
    {
    coot_extra_debug_sigprint();
    }

  coot_inline uword get_n_rows() const       { return Q.get_n_rows(); }
  coot_inline uword get_n_cols() const       { return Q.get_n_cols(); }
  coot_inline uword get_n_slices() const     { return Q.get_n_slices(); }
  coot_inline uword get_n_elem() const       { return Q.get_n_elem(); }
  coot_inline uword get_n_elem_slice() const { return Q.get_n_elem_slice(); }
  };



// eGlueCube
template<typename T1, typename T2, typename eglue_type>
class SizeProxyCube< eGlueCube<T1, T2, eglue_type> >
  {
  public:

  typedef typename T1::elem_type                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef eGlueCube<T1, T2, eglue_type>            stored_type;

  coot_aligned const eGlueCube<T1, T2, eglue_type>& Q;

  inline explicit SizeProxyCube(const eGlueCube<T1, T2, eglue_type>& A)
    : Q(A)
    {
    coot_extra_debug_sigprint();
    }

  coot_inline uword get_n_rows() const       { return Q.get_n_rows(); }
  coot_inline uword get_n_cols() const       { return Q.get_n_cols(); }
  coot_inline uword get_n_slices() const     { return Q.get_n_slices(); }
  coot_inline uword get_n_elem() const       { return Q.get_n_elem(); }
  coot_inline uword get_n_elem_slice() const { return Q.get_n_elem_slice(); }
  };



// mtOpCube
template<typename out_eT, typename T1, typename mtop_type>
class SizeProxyCube< mtOpCube<out_eT, T1, mtop_type> >
  {
  public:

  typedef out_eT                                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef mtOpCube<out_eT, T1, mtop_type>          stored_type;

  static constexpr bool is_row = mtOp<out_eT, T1, mtop_type>::is_row;
  static constexpr bool is_col = mtOp<out_eT, T1, mtop_type>::is_col;

  coot_aligned const SizeProxyCube<T1> S;
  coot_aligned const mtOpCube<out_eT, T1, mtop_type>& Q;

  inline explicit SizeProxyCube(const mtOpCube<out_eT, T1, mtop_type>& A)
    : S(A.q)
    , Q(A)
    {
    coot_extra_debug_sigprint();
    }

  coot_inline uword get_n_rows() const { return mtop_type::compute_n_rows(Q, S.get_n_rows(), S.get_n_cols(), S.get_n_slices()); }
  coot_inline uword get_n_cols() const { return mtop_type::compute_n_cols(Q, S.get_n_rows(), S.get_n_cols(), S.get_n_slices()); }
  coot_inline uword get_n_slices() const { return mtop_type::compute_n_slices(Q, S.get_n_rows(), S.get_n_cols(), S.get_n_slices()); }
  coot_inline uword get_n_elem() const { return get_n_rows() * get_n_cols() * get_n_slices(); }
  coot_inline uword get_n_elem_slice() const { return get_n_rows() * get_n_cols(); }
  };



// mtGlueCube
template<typename out_eT, typename T1, typename T2, typename mtglue_type>
class SizeProxyCube< mtGlueCube<out_eT, T1, T2, mtglue_type> >
  {
  public:

  typedef typename T1::elem_type                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;
  typedef mtGlueCube<out_eT, T1, T2, mtglue_type>  stored_type;

  // take the size of the first only
  coot_aligned const SizeProxyCube<T1> S;
  coot_aligned const mtGlueCube<out_eT, T1, T2, mtglue_type>& Q;

  inline explicit SizeProxyCube(const mtGlueCube<out_eT, T1, T2, mtglue_type>& A)
    : S(A.A)
    , Q(A)
    {
    coot_extra_debug_sigprint();
    }

  coot_inline uword get_n_rows() const       { return S.get_n_rows(); }
  coot_inline uword get_n_cols() const       { return S.get_n_cols(); }
  coot_inline uword get_n_slices() const     { return S.get_n_slices(); }
  coot_inline uword get_n_elem() const       { return S.get_n_elem(); }
  coot_inline uword get_n_elem_slice() const { return S.get_n_elem_slice(); }
  };
