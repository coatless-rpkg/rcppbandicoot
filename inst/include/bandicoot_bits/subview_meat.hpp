// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2017-2023 Ryan Curtin (https://www.ratml.org)
// Copyright 2008-2017 Conrad Sanderson (https://conradsanderson.id.au)
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


template<typename eT>
coot_inline
uword
subview<eT>::get_n_rows() const
  {
  return n_rows;
  }



template<typename eT>
coot_inline
uword
subview<eT>::get_n_cols() const
  {
  return n_cols;
  }



template<typename eT>
coot_inline
uword
subview<eT>::get_n_elem() const
  {
  return n_elem;
  }



template<typename eT>
inline
subview<eT>::~subview()
  {
  coot_debug_sigprint();
  }



template<typename eT>
inline
subview<eT>::subview(const Mat<eT>& in_m, const uword in_row1, const uword in_col1, const uword in_n_rows, const uword in_n_cols)
  : m(in_m)
  , aux_row1(in_row1)
  , aux_col1(in_col1)
  , n_rows(in_n_rows)
  , n_cols(in_n_cols)
  , n_elem(in_n_rows*in_n_cols)
  {
  coot_debug_sigprint();
  }



template<typename eT>
inline
subview<eT>::subview(const subview<eT>& x)
  : m(x.m)
  , aux_row1(x.aux_row1)
  , aux_col1(x.aux_col1)
  , n_rows(x.n_rows)
  , n_cols(x.n_cols)
  , n_elem(x.n_elem)
  {
  coot_debug_sigprint();
  }



template<typename eT>
inline
subview<eT>::subview(subview<eT>&& x)
  : m(x.m)
  , aux_row1(x.aux_row1)
  , aux_col1(x.aux_col1)
  , n_rows(x.n_rows)
  , n_cols(x.n_cols)
  , n_elem(x.n_elem)
  {
  coot_debug_sigprint();

  // for paranoia
  access::rw(x.aux_row1) = 0;
  access::rw(x.aux_col1) = 0;
  access::rw(x.n_rows) = 0;
  access::rw(x.n_cols) = 0;
  access::rw(x.n_elem) = 0;
  }



template<typename eT>
inline
void
subview<eT>::operator= (const subview<eT>& x)
  {
  coot_debug_sigprint();

  // TODO: this is currently a "better-than-nothing" solution; replace with code using a dedicated kernel

  const Mat<eT> tmp(x);

  (*this).operator=(tmp);


  // if(check_overlap(x))
  //   {
  //   const Mat<eT> tmp(x);
  //
  //   (*this).operator=(tmp);
  //   }
  // else
  //   {
  //   // TODO: implement kernel to copy from submatrix to submatrix
  //   }
  }



template<typename eT>
inline
void
subview<eT>::operator= (subview<eT>&& x)
  {
  coot_debug_sigprint();

  // TODO: this is currently a "better-than-nothing" solution; replace with code using a dedicated kernel

  const Mat<eT> tmp(x);

  (*this).operator=(tmp);


  // if(check_overlap(x))
  //   {
  //   const Mat<eT> tmp(x);
  //
  //   (*this).operator=(tmp);
  //   }
  // else
  //   {
  //   // TODO: implement kernel to copy from submatrix to submatrix
  //   }
  }



template<typename eT>
inline
void
subview<eT>::operator= (const eT val)
  {
  coot_debug_sigprint();

  if(n_elem == 1)
    {
    Mat<eT>& X = const_cast< Mat<eT>& >(m);

    X.at(aux_row1, aux_col1) = val;
    }
  else
    {
    coot_conform_assert_same_size(n_rows, n_cols, 1, 1, "subview::operator=");
    }
  }



template<typename eT>
inline
void
subview<eT>::operator+= (const eT val)
  {
  coot_debug_sigprint();

  coot_rt_t::copy(make_proxy(*this), make_proxy(eOp<subview<eT>, eop_scalar_plus>(*this, val)));
  }



template<typename eT>
inline
void
subview<eT>::operator-= (const eT val)
  {
  coot_debug_sigprint();

  coot_rt_t::copy(make_proxy(*this), make_proxy(eOp<subview<eT>, eop_scalar_minus_post>(*this, val)));
  }



template<typename eT>
inline
void
subview<eT>::operator*= (const eT val)
  {
  coot_debug_sigprint();

  coot_rt_t::copy(make_proxy(*this), make_proxy(eOp<subview<eT>, eop_scalar_times>(*this, val)));
  }



template<typename eT>
inline
void
subview<eT>::operator/= (const eT val)
  {
  coot_debug_sigprint();

  coot_rt_t::copy(make_proxy(*this), make_proxy(eOp<subview<eT>, eop_scalar_div_post>(*this, val)));
  }



template<typename eT>
template<typename T1>
inline
void
subview<eT>::operator= (const Base<eT, T1>& in)
  {
  coot_debug_sigprint();

  no_conv_unwrap<T1> U(in.get_ref());

  coot_assert_same_size(n_rows, n_cols, U.M.n_rows, U.M.n_cols, "subview::operator=");

  coot_rt_t::copy(make_proxy(*this), make_proxy(U.M));
  }



template<typename eT>
template<typename T1>
inline
void
subview<eT>::operator+= (const Base<eT, T1>& in)
  {
  coot_debug_sigprint();

  // Create the Proxy locally so we can do a size check.
  const eGlue<subview<eT>, T1, eglue_plus> G(*this, in.get_ref());
  const Proxy<eGlue<subview<eT>, T1, eglue_plus>> P(G);
  coot_assert_same_size(n_rows, n_cols, P.get_n_rows(), P.get_n_cols(), "subview::operator+=");

  inexact_alias_wrapper<subview<eT>, Proxy<eGlue<subview<eT>, T1, eglue_plus>>> A(*this, P);
  if (A.using_aux)
    {
    coot_rt_t::copy(make_proxy(A.aux), P);
    A.using_aux = false;
    *this = A.aux;
    }
  else
    {
    coot_rt_t::copy(make_proxy(*this), P);
    }
  }



template<typename eT>
template<typename T1>
inline
void
subview<eT>::operator-= (const Base<eT, T1>& in)
  {
  coot_debug_sigprint();

  // Create the Proxy locally so we can do a size check.
  const eGlue<subview<eT>, T1, eglue_minus> G(*this, in.get_ref());
  const Proxy<eGlue<subview<eT>, T1, eglue_minus>> P(G);
  coot_assert_same_size(n_rows, n_cols, P.get_n_rows(), P.get_n_cols(), "subview::operator-=");

  inexact_alias_wrapper<subview<eT>, Proxy<eGlue<subview<eT>, T1, eglue_minus>>> A(*this, P);
  if (A.using_aux)
    {
    coot_rt_t::copy(make_proxy(A.aux), P);
    A.using_aux = false;
    *this = A.aux;
    }
  else
    {
    coot_rt_t::copy(make_proxy(*this), P);
    }
  }



template<typename eT>
template<typename T1>
inline
void
subview<eT>::operator%= (const Base<eT, T1>& in)
  {
  coot_debug_sigprint();

  // Create the Proxy locally so we can do a size check.
  const eGlue<subview<eT>, T1, eglue_schur> G(*this, in.get_ref());
  const Proxy<eGlue<subview<eT>, T1, eglue_schur>> P(G);
  coot_assert_same_size(n_rows, n_cols, P.get_n_rows(), P.get_n_cols(), "subview::operator%=");

  inexact_alias_wrapper<subview<eT>, Proxy<eGlue<subview<eT>, T1, eglue_schur>>> A(*this, P);
  if (A.using_aux)
    {
    coot_rt_t::copy(make_proxy(A.aux), P);
    A.using_aux = false;
    *this = A.aux;
    }
  else
    {
    coot_rt_t::copy(make_proxy(*this), P);
    }
  }



template<typename eT>
template<typename T1>
inline
void
subview<eT>::operator/= (const Base<eT, T1>& in)
  {
  coot_debug_sigprint();

  // Create the Proxy locally so we can do a size check.
  const eGlue<subview<eT>, T1, eglue_div> G(*this, in.get_ref());
  const Proxy<eGlue<subview<eT>, T1, eglue_div>> P(G);
  coot_assert_same_size(n_rows, n_cols, P.get_n_rows(), P.get_n_cols(), "subview::operator/=");

  inexact_alias_wrapper<subview<eT>, Proxy<eGlue<subview<eT>, T1, eglue_div>>> A(*this, P);
  if (A.using_aux)
    {
    coot_rt_t::copy(make_proxy(A.aux), P);
    A.using_aux = false;
    *this = A.aux;
    }
  else
    {
    coot_rt_t::copy(make_proxy(*this), P);
    }
  }



template<typename eT>
inline
subview<eT>::operator arma::Mat<eT> () const
  {
  coot_debug_sigprint();

  #if defined(COOT_HAVE_ARMA)
    {
    arma::Mat<eT> out(n_rows, n_cols);

    coot_rt_t::copy_from_dev_mem(out.memptr(), m.get_dev_mem(false),
                                 n_rows, n_cols,
                                 aux_row1, aux_col1, m.n_rows);

    return out;
    }
  #else
    {
    coot_stop_logic_error("#include <armadillo> must be before #include <bandicoot>");

    return arma::Mat<eT>();
    }
  #endif
  }



template<typename eT>
coot_inline
diagview<eT>
subview<eT>::diag(const sword in_id)
  {
  coot_debug_sigprint();

  const uword row_offset = ((in_id < 0) ? uword(-in_id) : 0);
  const uword col_offset = ((in_id > 0) ? uword( in_id) : 0);

  coot_conform_check_bounds
    (
    ((row_offset > 0) && (row_offset >= n_rows)) || ((col_offset > 0) && (col_offset >= n_cols)),
    "subview::diag(): requested diagonal out of bounds"
    );

  const uword len = (std::min)(n_rows - row_offset, n_cols - col_offset);

  return diagview<eT>(m, row_offset + aux_row1, col_offset + aux_col1, len);
  }



template<typename eT>
coot_inline
const diagview<eT>
subview<eT>::diag(const sword in_id) const
  {
  coot_debug_sigprint();

  const uword row_offset = ((in_id < 0) ? uword(-in_id) : 0);
  const uword col_offset = ((in_id > 0) ? uword( in_id) : 0);

  coot_conform_check_bounds
    (
    ((row_offset > 0) && (row_offset >= n_rows)) || ((col_offset > 0) && (col_offset >= n_cols)),
    "subview::diag(): requested diagonal out of bounds"
    );

  const uword len = (std::min)(n_rows - row_offset, n_cols - col_offset);

  return diagview<eT>(m, row_offset + aux_row1, col_offset + aux_col1, len);
  }



template<typename eT>
inline
void
subview<eT>::clamp(const eT min_val, const eT max_val)
  {
  coot_debug_sigprint();

  coot_conform_check( (min_val > max_val), "clamp(): min_val must be less than max_val" );

  const eOp<subview<eT>, eop_clamp> E(*this, 'j', min_val, max_val);
  coot_rt_t::copy(make_proxy(*this), make_proxy(E));
  }



template<typename eT>
inline
void
subview<eT>::fill(const eT val)
  {
  coot_debug_sigprint();

  coot_rt_t::fill(make_proxy(*this), val);
  }



template<typename eT>
inline
void
subview<eT>::zeros()
  {
  coot_debug_sigprint();

  (*this).fill(eT(0));
  }



template<typename eT>
inline
void
subview<eT>::ones()
  {
  coot_debug_sigprint();

  (*this).fill(eT(1));
  }



template<typename eT>
inline
void
subview<eT>::eye()
  {
  coot_debug_sigprint();

  // TODO: this is currently a "better-than-nothing" solution; replace with code using a dedicated kernel

  Mat<eT> tmp(n_rows, n_cols);
  tmp.eye();

  (*this).operator=(tmp);
  }



template<typename eT>
inline
MatValProxy<eT>
subview<eT>::operator[](const uword ii)
  {
  const uword in_col = ii / n_rows;
  const uword in_row = ii % n_rows;

  const uword index = (in_col + aux_col1)*m.n_rows + aux_row1 + in_row;

  return MatValProxy<eT>(access::rw(this->m), index);
  }



template<typename eT>
inline
eT
subview<eT>::operator[](const uword ii) const
  {
  const uword in_col = ii / n_rows;
  const uword in_row = ii % n_rows;

  const uword index = (in_col + aux_col1)*m.n_rows + aux_row1 + in_row;

  return MatValProxy<eT>::get_val(this->m, index);
  }



template<typename eT>
inline
MatValProxy<eT>
subview<eT>::at(const uword ii)
  {
  const uword in_col = ii / n_rows;
  const uword in_row = ii % n_rows;

  const uword index = (in_col + aux_col1)*m.n_rows + aux_row1 + in_row;

  return MatValProxy<eT>(access::rw(this->m), index);
  }



template<typename eT>
inline
eT
subview<eT>::at(const uword ii) const
  {
  const uword in_col = ii / n_rows;
  const uword in_row = ii % n_rows;

  const uword index = (in_col + aux_col1)*m.n_rows + aux_row1 + in_row;

  return MatValProxy<eT>::get_val(this->m, index);
  }



template<typename eT>
inline
MatValProxy<eT>
subview<eT>::operator()(const uword ii)
  {
  coot_conform_check_bounds( (ii >= n_elem), "subview::operator(): index out of bounds");

  const uword in_col = ii / n_rows;
  const uword in_row = ii % n_rows;

  const uword index = (in_col + aux_col1)*m.n_rows + aux_row1 + in_row;

  return MatValProxy<eT>(access::rw(this->m), index);
  }



template<typename eT>
inline
eT
subview<eT>::operator()(const uword ii) const
  {
  coot_conform_check_bounds( (ii >= n_elem), "subview::operator(): index out of bounds");

  const uword in_col = ii / n_rows;
  const uword in_row = ii % n_rows;

  const uword index = (in_col + aux_col1)*m.n_rows + aux_row1 + in_row;

  return MatValProxy<eT>::get_val(this->m, index);
  }



template<typename eT>
inline
MatValProxy<eT>
subview<eT>::at(const uword in_row, const uword in_col)
  {
  return MatValProxy<eT>(access::rw(this->m), in_row + aux_row1 + (in_col + aux_col1) * m.n_rows);
  }



template<typename eT>
inline
eT
subview<eT>::at(const uword in_row, const uword in_col) const
  {
  return MatValProxy<eT>::get_val(this->m, in_row + aux_row1 + (in_col + aux_col1) * m.n_rows);
  }



template<typename eT>
inline
MatValProxy<eT>
subview<eT>::operator()(const uword in_row, const uword in_col)
  {
  coot_conform_check_bounds( ((in_row >= n_rows) || (in_col >= n_cols)), "subview::operator(): index out of bounds");

  return MatValProxy<eT>(access::rw(this->m), in_row + aux_row1 + (in_col + aux_col1) * m.n_rows);
  }



template<typename eT>
inline
eT
subview<eT>::operator()(const uword in_row, const uword in_col) const
  {
  coot_conform_check_bounds( ((in_row >= n_rows) || (in_col >= n_cols)), "subview::operator(): index out of bounds");

  return MatValProxy<eT>::get_val(this->m, in_row + aux_row1 + (in_col + aux_col1) * m.n_rows);
  }



template<typename eT>
inline
eT
subview<eT>::front() const
  {
  coot_conform_check( (n_elem == 0), "subview::front(): matrix is empty" );

  return m.at(aux_row1, aux_col1);
  }



template<typename eT>
inline
eT
subview<eT>::back() const
  {
  coot_conform_check( (n_elem == 0), "subview::back(): matrix is empty" );

  return m.at(aux_row1 + n_rows - 1, aux_col1 + n_cols - 1);
  }



template<typename eT>
inline
bool
subview<eT>::check_overlap(const subview<eT>& x) const
  {
  const subview<eT>& s = *this;

  if(&s.m != &x.m)
    {
    return false;
    }
  else
    {
    if( (s.n_elem == 0) || (x.n_elem == 0) )
      {
      return false;
      }
    else
      {
      const uword s_row_start  = s.aux_row1;
      const uword s_row_end_p1 = s_row_start + s.n_rows;

      const uword s_col_start  = s.aux_col1;
      const uword s_col_end_p1 = s_col_start + s.n_cols;


      const uword x_row_start  = x.aux_row1;
      const uword x_row_end_p1 = x_row_start + x.n_rows;

      const uword x_col_start  = x.aux_col1;
      const uword x_col_end_p1 = x_col_start + x.n_cols;


      const bool outside_rows = ( (x_row_start >= s_row_end_p1) || (s_row_start >= x_row_end_p1) );
      const bool outside_cols = ( (x_col_start >= s_col_end_p1) || (s_col_start >= x_col_end_p1) );

      return ( (outside_rows == false) && (outside_cols == false) );
      }
    }
  }



template<typename eT>
inline
bool
subview<eT>::is_vec() const
  {
  return ( (n_rows == 1) || (n_cols == 1) );
  }



template<typename eT>
inline
bool
subview<eT>::is_colvec() const
  {
  return (n_cols == 1);
  }



template<typename eT>
inline
bool
subview<eT>::is_rowvec() const
  {
  return (n_rows == 1);
  }



template<typename eT>
inline
bool
subview<eT>::is_square() const
  {
  return (n_rows == n_cols);
  }



template<typename eT>
inline
bool
subview<eT>::is_empty() const
  {
  return (n_elem == 0);
  }



// X = Y.submat(...)
template<typename eT>
template<typename eT1>
inline
void
subview<eT>::extract(Mat<eT1>& out, const subview<eT>& in)
  {
  coot_debug_sigprint();

  // NOTE: we're assuming that the matrix has already been set to the correct size and there is no aliasing;
  // size setting and alias checking is done by either the Mat contructor or operator=()

  coot_debug_print(coot_str::format("out.n_rows: %u; out.n_cols: %u; in.m.n_rows: %u; in.m.n_cols: %u") % out.n_rows % out.n_cols % in.m.n_rows % in.m.n_cols );

  if(in.n_elem == 0)  { return; }

  coot_rt_t::copy(make_proxy(out), make_proxy(in));
  }



//
// each_col and each_row


template<typename eT>
coot_inline
subview_each1<subview<eT>, 0>
subview<eT>::each_col()
  {
  coot_debug_sigprint();

  return subview_each1<subview<eT>, 0>(*this);
  }



template<typename eT>
coot_inline
subview_each1<subview<eT>, 1>
subview<eT>::each_row()
  {
  coot_debug_sigprint();

  return subview_each1<subview<eT>, 1>(*this);
  }



template<typename eT>
coot_inline
const subview_each1<subview<eT>, 0>
subview<eT>::each_col() const
  {
  coot_debug_sigprint();

  return subview_each1<subview<eT>, 0>(*this);
  }



template<typename eT>
coot_inline
const subview_each1<subview<eT>, 1>
subview<eT>::each_row() const
  {
  coot_debug_sigprint();

  return subview_each1<subview<eT>, 1>(*this);
  }



template<typename eT>
template<typename T1>
inline
subview_each2<subview<eT>, 0, T1>
subview<eT>::each_col(const Base<uword, T1>& indices)
  {
  coot_debug_sigprint();

  return subview_each2<subview<eT>, 0, T1>(*this, indices);
  }



template<typename eT>
template<typename T1>
inline
subview_each2<subview<eT>, 1, T1>
subview<eT>::each_row(const Base<uword, T1>& indices)
  {
  coot_debug_sigprint();

  return subview_each2<subview<eT>, 1, T1>(*this, indices);
  }



template<typename eT>
template<typename T1>
inline
const subview_each2<subview<eT>, 0, T1>
subview<eT>::each_col(const Base<uword, T1>& indices) const
  {
  coot_debug_sigprint();

  return subview_each2<subview<eT>, 0, T1>(*this, indices);
  }



template<typename eT>
template<typename T1>
inline
const subview_each2<subview<eT>, 1, T1>
subview<eT>::each_row(const Base<uword, T1>& indices) const
  {
  coot_debug_sigprint();

  return subview_each2<subview<eT>, 1, T1>(*this, indices);
  }



//
// subview_col


template<typename eT>
coot_inline
uword
subview_col<eT>::get_n_cols() const
  {
  return uword(1);
  }



template<typename eT>
inline
subview_col<eT>::subview_col(const Mat<eT>& in_m, const uword in_col)
  : subview<eT>(in_m, 0, in_col, in_m.n_rows, 1)
  {
  coot_debug_sigprint();
  }



template<typename eT>
inline
subview_col<eT>::subview_col(const Mat<eT>& in_m, const uword in_col, const uword in_row1, const uword in_n_rows)
  : subview<eT>(in_m, in_row1, in_col, in_n_rows, 1)
  {
  coot_debug_sigprint();
  }



template<typename eT>
inline
subview_col<eT>::subview_col(const char junk, const Mat<eT>& in_m, const uword in_aux_row1, const uword in_n_rows)
  : subview<eT>(in_m, in_aux_row1, 0, in_n_rows, 1)
  {
  coot_debug_sigprint();
  coot_ignore(junk);
  }



template<typename eT>
inline
void
subview_col<eT>::operator=(const subview<eT>& X)
  {
  coot_debug_sigprint();

  subview<eT>::operator=(X);
  }



template<typename eT>
inline
void
subview_col<eT>::operator=(const subview_col<eT>& X)
  {
  coot_debug_sigprint();

  subview<eT>::operator=(X); // interprets 'subview_col' as 'subview'
  }



template<typename eT>
inline
void
subview_col<eT>::operator=(const eT val)
  {
  coot_debug_sigprint();

  subview<eT>::operator=(val); // interprets 'subview_col' as 'subview'
  }



template<typename eT>
template<typename T1>
inline
void
subview_col<eT>::operator=(const Base<eT,T1>& X)
  {
  coot_debug_sigprint();

  subview<eT>::operator=(X); // interprets 'subview_col' as 'subview'
  }



template<typename eT>
coot_inline
const Op<subview_col<eT>, op_htrans>
subview_col<eT>::t() const
  {
  return Op<subview_col<eT>, op_htrans>(*this);
  }



template<typename eT>
coot_inline
const Op<subview_col<eT>, op_htrans>
subview_col<eT>::ht() const
  {
  return Op<subview_col<eT>, op_htrans>(*this);
  }



template<typename eT>
coot_inline
const Op<subview_col<eT>, op_strans>
subview_col<eT>::st() const
  {
  return Op<subview_col<eT>, op_strans>(*this);
  }



//
// subview_row


template<typename eT>
coot_inline
uword
subview_row<eT>::get_n_rows() const
  {
  return uword(1);
  }



template<typename eT>
inline
subview_row<eT>::subview_row(const Mat<eT>& in_m, const uword in_row)
  : subview<eT>(in_m, in_row, 0, 1, in_m.n_cols)
  {
  coot_debug_sigprint();
  }



template<typename eT>
inline
subview_row<eT>::subview_row(const Mat<eT>& in_m, const uword in_row, const uword in_col1, const uword in_n_cols)
  : subview<eT>(in_m, in_row, in_col1, 1, in_n_cols)
  {
  coot_debug_sigprint();
  }



template<typename eT>
inline
void
subview_row<eT>::operator=(const subview<eT>& X)
  {
  coot_debug_sigprint();

  subview<eT>::operator=(X);
  }



template<typename eT>
inline
void
subview_row<eT>::operator=(const subview_row<eT>& X)
  {
  coot_debug_sigprint();

  subview<eT>::operator=(X); // interprets 'subview_row' as 'subview'
  }



template<typename eT>
inline
void
subview_row<eT>::operator=(const eT val)
  {
  coot_debug_sigprint();

  subview<eT>::operator=(val); // interprets 'subview_row' as 'subview'
  }



template<typename eT>
template<typename T1>
inline
void
subview_row<eT>::operator=(const Base<eT,T1>& X)
  {
  coot_debug_sigprint();

  subview<eT>::operator=(X);
  }



template<typename eT>
coot_inline
const Op<subview_row<eT>, op_htrans>
subview_row<eT>::t() const
  {
  return Op<subview_row<eT>, op_htrans>(*this);
  }



template<typename eT>
coot_inline
const Op<subview_row<eT>, op_htrans>
subview_row<eT>::ht() const
  {
  return Op<subview_row<eT>, op_htrans>(*this);
  }



template<typename eT>
coot_inline
const Op<subview_row<eT>, op_strans>
subview_row<eT>::st() const
  {
  return Op<subview_row<eT>, op_strans>(*this);
  }
