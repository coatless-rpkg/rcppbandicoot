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



template<typename eT>
inline
Col<eT>::Col()
  : Mat<eT>()
  {
  coot_extra_debug_sigprint();

  access::rw(Mat<eT>::vec_state) = 1;
  }



template<typename eT>
inline
Col<eT>::Col(const uword N)
  : Mat<eT>(N, 1)
  {
  coot_extra_debug_sigprint();

  access::rw(Mat<eT>::vec_state) = 1;
  }



template<typename eT>
inline
Col<eT>::Col(const uword in_rows, const uword in_cols)
  : Mat<eT>()
  {
  coot_extra_debug_sigprint();

  access::rw(Mat<eT>::vec_state) = 1;

  Mat<eT>::init(in_rows, in_cols);

  Mat<eT>::zeros();  // fill with zeros by default
  }



template<typename eT>
inline
Col<eT>::Col(const SizeMat& s)
  : Mat<eT>()
  {
  coot_extra_debug_sigprint();

  access::rw(Mat<eT>::vec_state) = 1;

  Mat<eT>::init(s.n_rows, s.n_cols);

  Mat<eT>::zeros();  // fill with zeros by default
  }



template<typename eT>
template<typename fill_type>
inline
Col<eT>::Col(const uword N, const fill::fill_class<fill_type>& f)
  : Mat<eT>(N, 1)
  {
  coot_extra_debug_sigprint();

  access::rw(Mat<eT>::vec_state) = 1;

  Mat<eT>::fill(f);
  }



template<typename eT>
template<typename fill_type>
inline
Col<eT>::Col(const uword in_rows, const uword in_cols, const fill::fill_class<fill_type>& f)
  : Mat<eT>()
  {
  coot_extra_debug_sigprint();

  access::rw(Mat<eT>::vec_state) = 1;

  Mat<eT>::init(in_rows, in_cols);

  Mat<eT>::fill(f);
  }



template<typename eT>
template<typename fill_type>
inline
Col<eT>::Col(const SizeMat& s, const fill::fill_class<fill_type>& f)
  : Mat<eT>()
  {
  coot_extra_debug_sigprint();

  access::rw(Mat<eT>::vec_state) = 1;

  Mat<eT>::init(s.n_rows, s.n_cols);

  Mat<eT>::fill(f);
  }



template<typename eT>
inline
Col<eT>::Col(dev_mem_t<eT> aux_dev_mem, const uword N)
  : Mat<eT>(aux_dev_mem, N, 1)
  {
  coot_extra_debug_sigprint();

  access::rw(Mat<eT>::vec_state) = 1;
  }



template<typename eT>
inline
Col<eT>::Col(cl_mem aux_dev_mem, const uword N)
  : Mat<eT>(aux_dev_mem, N, 1)
  {
  coot_extra_debug_sigprint();

  access::rw(Mat<eT>::vec_state) = 1;
  }



template<typename eT>
inline
Col<eT>::Col(typename cuda_type<eT>::type* aux_dev_mem, const uword N)
  : Mat<eT>(aux_dev_mem, N, 1)
  {
  coot_extra_debug_sigprint();

  access::rw(Mat<eT>::vec_state) = 1;
  }



template<typename eT>
inline
Col<eT>::Col(const Col<eT>& X)
  : Mat<eT>(X.n_rows, 1)
  {
  coot_extra_debug_sigprint();

  access::rw(Mat<eT>::vec_state) = 1;
  coot_rt_t::copy_mat(this->get_dev_mem(), X.get_dev_mem(),
                      Mat<eT>::n_rows, 1,
                      0, 0, Mat<eT>::n_rows,
                      0, 0, X.n_rows);
  }



template<typename eT>
inline
Col<eT>&
Col<eT>::operator=(const Col<eT>& X)
  {
  coot_extra_debug_sigprint();

  Mat<eT>::init(X.n_rows, 1);
  coot_rt_t::copy_mat(this->get_dev_mem(), X.get_dev_mem(),
                      Mat<eT>::n_rows, 1,
                      0, 0, Mat<eT>::n_rows,
                      0, 0, X.n_rows);

  return *this;
  }



template<typename eT>
inline
Col<eT>::Col(Col<eT>&& X)
  : Mat<eT>()
  {
  coot_extra_debug_sigprint();

  Mat<eT>::steal_mem(X);
  // Make sure to restore the other Col's vec_state.
  access::rw(X.vec_state) = 1;
  }



template<typename eT>
inline
Col<eT>&
Col<eT>::operator=(Col<eT>&& X)
  {
  coot_extra_debug_sigprint();

  // Clean up old memory, if required.
  coot_rt_t::synchronise();
  Mat<eT>::cleanup();

  Mat<eT>::steal_mem(X);
  // Make sure to restore the other Col's vec_state.
  access::rw(X.vec_state) = 1;

  return *this;
  }



template<typename eT>
inline
Col<eT>::Col(const char* text)
  : Mat<eT>()
  {
  coot_extra_debug_sigprint_this(this);

  (*this).operator=(text);
  }



template<typename eT>
inline
Col<eT>&
Col<eT>::operator=(const char* text)
  {
  coot_extra_debug_sigprint();

  Mat<eT> tmp(text);

  coot_debug_check( ((tmp.n_elem > 0) && (tmp.is_vec() == false)), "Mat::init(): requested size is not compatible with column vector layout" );

  access::rw(tmp.n_rows) = tmp.n_elem;
  access::rw(tmp.n_cols) = 1;

  (*this).steal_mem(tmp);

  return *this;
  }



template<typename eT>
inline
Col<eT>::Col(const std::string& text)
  : Mat<eT>()
  {
  coot_extra_debug_sigprint_this(this);

  (*this).operator=(text);
  }



template<typename eT>
inline
Col<eT>&
Col<eT>::operator=(const std::string& text)
  {
  coot_extra_debug_sigprint();

  Mat<eT> tmp(text);

  coot_debug_check( ((tmp.n_elem > 0) && (tmp.is_vec() == false)), "Mat::init(): requested size is not compatible with column vector layout" );

  access::rw(tmp.n_rows) = tmp.n_elem;
  access::rw(tmp.n_cols) = 1;

  (*this).steal_mem(tmp);

  return *this;
  }



template<typename eT>
inline
Col<eT>::Col(const std::vector<eT>& x)
  : Mat<eT>()
  {
  coot_extra_debug_sigprint_this(this);

  const uword N = uword(x.size());

  Mat<eT>::set_size(N, 1);

  if (N > 0)
    {
    coot_rt_t::copy_into_dev_mem(this->get_dev_mem(false), &(x[0]), N);
    // Force synchronisation in case x goes out of scope.
    coot_rt_t::synchronise();
    }
  }



template<typename eT>
inline
Col<eT>&
Col<eT>::operator=(const std::vector<eT>& x)
  {
  coot_extra_debug_sigprint();

  const uword N = uword(x.size());

  Mat<eT>::set_size(N, 1);

  if (N > 0)
    {
    coot_rt_t::copy_into_dev_mem(this->get_dev_mem(false), &(x[0]), N);
    // Force synchronisation in case x goes out of scope.
    coot_rt_t::synchronise();
    }

  return *this;
  }



template<typename eT>
inline
Col<eT>::Col(const std::initializer_list<eT>& list)
  : Mat<eT>()
  {
  coot_extra_debug_sigprint_this(this);

  const uword N = uword(list.size());

  Mat<eT>::set_size(N, 1);

  if (N > 0)
    {
    coot_rt_t::copy_into_dev_mem(this->get_dev_mem(false), list.begin(), N);
    // Force synchronisation in case the list goes out of scope.
    coot_rt_t::synchronise();
    }
  }



template<typename eT>
inline
Col<eT>&
Col<eT>::operator=(const std::initializer_list<eT>& list)
  {
  coot_extra_debug_sigprint();

  const uword N = uword(list.size());

  Mat<eT>::set_size(N, 1);

  if (N > 0)
    {
    coot_rt_t::copy_into_dev_mem(this->get_dev_mem(false), list.begin(), N);
    // Force synchronisation in case the list goes out of scope.
    coot_rt_t::synchronise();
    }

  return *this;
  }



template<typename eT>
template<typename T1>
inline
Col<eT>::Col(const Base<eT, T1>& X)
  : Mat<eT>()
  {
  coot_extra_debug_sigprint();

  access::rw(Mat<eT>::vec_state) = 1;

  Mat<eT>::operator=(X.get_ref());
  }



template<typename eT>
template<typename T1>
inline
Col<eT>&
Col<eT>::operator=(const Base<eT, T1>& X)
  {
  coot_extra_debug_sigprint();

  Mat<eT>::operator=(X.get_ref());

  return *this;
  }



template<typename eT>
inline
Col<eT>::Col(const arma::Col<eT>& X)
  : Mat<eT>((const arma::Mat<eT>&) X)
  {
  coot_extra_debug_sigprint_this(this);
  }



template<typename eT>
inline
Col<eT>&
Col<eT>::operator=(const arma::Col<eT>& X)
  {
  coot_extra_debug_sigprint();

  #if defined(COOT_HAVE_ARMA)
    {
    (*this).set_size(X.n_rows, X.n_cols);

    (*this).copy_into_dev_mem(X.memptr(), (*this).n_elem);
    }
  #else
    {
    coot_stop_logic_error("#include <armadillo> must be before #include <bandicoot>");
    }
  #endif

  return *this;
  }



template<typename eT>
inline
Col<eT>::operator arma::Col<eT>() const
  {
  coot_extra_debug_sigprint();

  #if defined(COOT_HAVE_ARMA)
    {
    arma::Col<eT> out(Mat<eT>::n_rows, 1);

    (*this).copy_from_dev_mem(out.memptr(), (*this).n_elem);

    return out;
    }
  #else
    {
    coot_stop_logic_error("#include <armadillo> must be before #include <bandicoot>");

    return arma::Col<eT>();
    }
  #endif
  }



template<typename eT>
inline
const Op<Col<eT>, op_htrans>
Col<eT>::t() const
  {
  return Op<Col<eT>, op_htrans>(*this);
  }



template<typename eT>
inline
const Op<Col<eT>, op_htrans>
Col<eT>::ht() const
  {
  return Op<Col<eT>, op_htrans>(*this);
  }



template<typename eT>
inline
const Op<Col<eT>, op_strans>
Col<eT>::st() const
  {
  return Op<Col<eT>, op_strans>(*this);
  }



template<typename eT>
coot_inline
subview_col<eT>
Col<eT>::rows(const uword in_row1, const uword in_row2)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( ((in_row1 > in_row2) || (in_row2 >= Mat<eT>::n_rows) ), "Col::rows(): indices out of bounds or incorrectly used");

  const uword subview_n_rows = in_row2 - in_row1 + 1;

  return subview_col<eT>(*this, 0, in_row1, subview_n_rows);
  }



template<typename eT>
coot_inline
const subview_col<eT>
Col<eT>::rows(const uword in_row1, const uword in_row2) const
  {
  coot_extra_debug_sigprint();

  coot_debug_check( ((in_row1 > in_row2) || (in_row2 >= Mat<eT>::n_rows) ), "Col::rows(): indices out of bounds or incorrectly used");

  const uword subview_n_rows = in_row2 - in_row1 + 1;

  return subview_col<eT>(*this, 0, in_row1, subview_n_rows);
  }



template<typename eT>
coot_inline
subview_col<eT>
Col<eT>::subvec(const uword in_row1, const uword in_row2)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( ((in_row1 > in_row2) || (in_row2 >= Mat<eT>::n_rows) ), "Col::rows(): indices out of bounds or incorrectly used");

  const uword subview_n_rows = in_row2 - in_row1 + 1;

  return subview_col<eT>(*this, 0, in_row1, subview_n_rows);
  }



template<typename eT>
coot_inline
const subview_col<eT>
Col<eT>::subvec(const uword in_row1, const uword in_row2) const
  {
  coot_extra_debug_sigprint();

  coot_debug_check( ((in_row1 > in_row2) || (in_row2 >= Mat<eT>::n_rows) ), "Col::rows(): indices out of bounds or incorrectly used");

  const uword subview_n_rows = in_row2 - in_row1 + 1;

  return subview_col<eT>(*this, 0, in_row1, subview_n_rows);
  }



template<typename eT>
coot_inline
subview_col<eT>
Col<eT>::subvec(const uword start_row, const SizeMat& s)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (s.n_cols != 1), "Col::subvec(): given size does not specify a column vector" );

  coot_debug_check_bounds( ( (start_row >= Mat<eT>::n_rows) || ((start_row + s.n_rows) > Mat<eT>::n_rows) ), "Col::subvec(): size out of bounds" );

  return subview_col<eT>(*this, 0, start_row, s.n_rows);
  }



template<typename eT>
coot_inline
const subview_col<eT>
Col<eT>::subvec(const uword start_row, const SizeMat& s) const
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (s.n_cols != 1), "Col::subvec(): given size does not specify a column vector" );

  coot_debug_check_bounds( ( (start_row >= Mat<eT>::n_rows) || ((start_row + s.n_rows) > Mat<eT>::n_rows) ), "Col::subvec(): size out of bounds" );

  return subview_col<eT>(*this, 0, start_row, s.n_rows);
  }



#ifdef COOT_EXTRA_COL_MEAT
  #include COOT_INCFILE_WRAP(COOT_EXTRA_COL_MEAT)
#endif
