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
Row<eT>::Row()
  : Mat<eT>()
  {
  coot_extra_debug_sigprint();

  access::rw(Mat<eT>::vec_state) = 2;
  }



template<typename eT>
inline
Row<eT>::Row(const uword N)
  : Mat<eT>(1, N)
  {
  coot_extra_debug_sigprint();

  access::rw(Mat<eT>::vec_state) = 2;
  }



template<typename eT>
inline
Row<eT>::Row(const uword in_rows, const uword in_cols)
  : Mat<eT>()
  {
  coot_extra_debug_sigprint();

  access::rw(Mat<eT>::vec_state) = 2;

  Mat<eT>::init(in_rows, in_cols);

  Mat<eT>::zeros();  // fill with zeros by default
  }



template<typename eT>
inline
Row<eT>::Row(const SizeMat& s)
  : Mat<eT>()
  {
  coot_extra_debug_sigprint();

  access::rw(Mat<eT>::vec_state) = 2;

  Mat<eT>::init(s.n_rows, s.n_cols);

  Mat<eT>::zeros();  // fill with zeros by default
  }



template<typename eT>
template<typename fill_type>
inline
Row<eT>::Row(const uword N, const fill::fill_class<fill_type>& f)
  : Mat<eT>(1, N)
  {
  coot_extra_debug_sigprint();
  
  access::rw(Mat<eT>::vec_state) = 2;
  
  Mat<eT>::fill(f);
  }



template<typename eT>
template<typename fill_type>
inline
Row<eT>::Row(const uword in_rows, const uword in_cols, const fill::fill_class<fill_type>& f)
  : Mat<eT>()
  {
  coot_extra_debug_sigprint();
  
  access::rw(Mat<eT>::vec_state) = 2;
  
  Mat<eT>::init(in_rows, in_cols);
  
  Mat<eT>::fill(f);
  }



template<typename eT>
template<typename fill_type>
inline
Row<eT>::Row(const SizeMat& s, const fill::fill_class<fill_type>& f)
  : Mat<eT>()
  {
  coot_extra_debug_sigprint();
  
  access::rw(Mat<eT>::vec_state) = 2;
  
  Mat<eT>::init(s.n_rows, s.n_cols);
  
  Mat<eT>::fill(f);
  }



template<typename eT>
inline
Row<eT>::Row(dev_mem_t<eT> aux_dev_mem, const uword N)
  : Mat<eT>(aux_dev_mem, 1, N)
  {
  coot_extra_debug_sigprint();

  access::rw(Mat<eT>::vec_state) = 2;
  }



template<typename eT>
inline
Row<eT>::Row(cl_mem aux_dev_mem, const uword N)
  : Mat<eT>(aux_dev_mem, 1, N)
  {
  coot_extra_debug_sigprint();

  access::rw(Mat<eT>::vec_state) = 2;
  }



template<typename eT>
inline
Row<eT>::Row(eT* aux_dev_mem, const uword N)
  : Mat<eT>(aux_dev_mem, 1, N)
  {
  coot_extra_debug_sigprint();

  access::rw(Mat<eT>::vec_state) = 2;
  }



template<typename eT>
inline
Row<eT>::Row(const Row<eT>& X)
  : Mat<eT>(1, X.n_cols)
  {
  coot_extra_debug_sigprint();

  access::rw(Mat<eT>::vec_state) = 2;
  coot_rt_t::copy_mat(this->get_dev_mem(), X.get_dev_mem(),
                      1, Mat<eT>::n_cols,
                      0, 0, Mat<eT>::n_rows,
                      0, 0, X.n_rows);
  }



template<typename eT>
inline
const Row<eT>&
Row<eT>::operator=(const Row<eT>& X)
  {
  coot_extra_debug_sigprint();

  Mat<eT>::init(1, X.n_cols);
  coot_rt_t::copy_mat(this->get_dev_mem(), X.get_dev_mem(),
                      1, Mat<eT>::n_cols,
                      0, 0, Mat<eT>::n_rows,
                      0, 0, X.n_rows);

  return *this;
  }



template<typename eT>
inline
Row<eT>::Row(Row<eT>&& X)
  : Mat<eT>()
  {
  coot_extra_debug_sigprint();

  Mat<eT>::steal_mem(X);
  // Make sure to restore the other Row's vec_state.
  access::rw(X.vec_state) = 2;
  }



template<typename eT>
inline
const Row<eT>&
Row<eT>::operator=(Row<eT>&& X)
  {
  coot_extra_debug_sigprint();

  // Clean up old memory, if required.
  coot_rt_t::synchronise();
  Mat<eT>::cleanup();

  Mat<eT>::steal_mem(X);
  // Make sure to restore the other Row's vec_state.
  access::rw(X.vec_state) = 2;

  return *this;
  }



template<typename eT>
inline
Row<eT>::Row(const char* text)
  : Mat<eT>()
  {
  coot_extra_debug_sigprint_this(this);

  (*this).operator=(text);
  }



template<typename eT>
inline
Row<eT>&
Row<eT>::operator=(const char* text)
  {
  coot_extra_debug_sigprint();

  Mat<eT> tmp(text);

  coot_debug_check( ((tmp.n_elem > 0) && (tmp.is_vec() == false)), "Mat::init(): requested size is not compatible with row vector layout" );

  access::rw(tmp.n_rows) = 1;
  access::rw(tmp.n_cols) = tmp.n_elem;

  (*this).steal_mem(tmp);

  return *this;
  }



template<typename eT>
inline
Row<eT>::Row(const std::string& text)
  : Mat<eT>()
  {
  coot_extra_debug_sigprint_this(this);

  (*this).operator=(text);
  }



template<typename eT>
inline
Row<eT>&
Row<eT>::operator=(const std::string& text)
  {
  coot_extra_debug_sigprint();

  Mat<eT> tmp(text);

  coot_debug_check( ((tmp.n_elem > 0) && (tmp.is_vec() == false)), "Mat::init(): requested size is not compatible with row vector layout" );

  access::rw(tmp.n_rows) = 1;
  access::rw(tmp.n_cols) = tmp.n_elem;

  (*this).steal_mem(tmp);

  return *this;
  }



template<typename eT>
inline
Row<eT>::Row(const std::vector<eT>& x)
  : Mat<eT>()
  {
  coot_extra_debug_sigprint_this(this);

  const uword N = uword(x.size());

  Mat<eT>::set_size(1, N);

  if (N > 0)
    {
    coot_rt_t::copy_into_dev_mem(this->get_dev_mem(false), &(x[0]), N);
    // Force synchronisation in case x goes out of scope.
    coot_rt_t::synchronise();
    }
  }



template<typename eT>
inline
Row<eT>&
Row<eT>::operator=(const std::vector<eT>& x)
  {
  coot_extra_debug_sigprint();

  const uword N = uword(x.size());

  Mat<eT>::set_size(1, N);

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
Row<eT>::Row(const std::initializer_list<eT>& list)
  : Mat<eT>()
  {
  coot_extra_debug_sigprint_this(this);

  const uword N = uword(list.size());

  Mat<eT>::set_size(1, N);

  if (N > 0)
    {
    coot_rt_t::copy_into_dev_mem(this->get_dev_mem(false), list.begin(), N);
    // Force synchronisation in case the list goes out of scope.
    coot_rt_t::synchronise();
    }
  }



template<typename eT>
inline
Row<eT>&
Row<eT>::operator=(const std::initializer_list<eT>& list)
  {
  coot_extra_debug_sigprint();

  const uword N = uword(list.size());

  Mat<eT>::set_size(1, N);

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
Row<eT>::Row(const Base<eT, T1>& X)
  : Mat<eT>()
  {
  coot_extra_debug_sigprint();

  access::rw(Mat<eT>::vec_state) = 2;

  Mat<eT>::operator=(X.get_ref());
  }



template<typename eT>
template<typename T1>
inline
Row<eT>&
Row<eT>::operator=(const Base<eT, T1>& X)
  {
  coot_extra_debug_sigprint();

  Mat<eT>::operator=(X.get_ref());

  return *this;
  }



template<typename eT>
inline
Row<eT>::Row(const arma::Row<eT>& X)
  : Mat<eT>((const arma::Mat<eT>&) X)
  {
  coot_extra_debug_sigprint_this(this);
  }



template<typename eT>
inline
const Row<eT>&
Row<eT>::operator=(const arma::Row<eT>& X)
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
Row<eT>::operator arma::Row<eT>() const
  {
  coot_extra_debug_sigprint();

  #if defined(COOT_HAVE_ARMA)
    {
    arma::Row<eT> out(1, Mat<eT>::n_cols);

    (*this).copy_from_dev_mem(out.memptr(), (*this).n_elem);

    return out;
    }
  #else
    {
    coot_stop_logic_error("#include <armadillo> must be before #include <bandicoot>");

    return arma::Row<eT>();
    }
  #endif
  }



template<typename eT>
inline
const Op<Row<eT>, op_htrans>
Row<eT>::t() const
  {
  return Op<Row<eT>, op_htrans>(*this);
  }



template<typename eT>
inline
const Op<Row<eT>, op_htrans>
Row<eT>::ht() const
  {
  return Op<Row<eT>, op_htrans>(*this);
  }



template<typename eT>
inline
const Op<Row<eT>, op_strans>
Row<eT>::st() const
  {
  return Op<Row<eT>, op_strans>(*this);
  }



template<typename eT>
coot_inline
subview_row<eT>
Row<eT>::cols(const uword in_col1, const uword in_col2)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( ((in_col1 > in_col2) || (in_col2 >= Mat<eT>::n_cols) ), "Row::cols(): indices out of bounds or incorrectly used");

  const uword subview_n_cols = in_col2 - in_col1 + 1;

  return subview_row<eT>(*this, 0, in_col1, subview_n_cols);
  }



template<typename eT>
coot_inline
const subview_row<eT>
Row<eT>::cols(const uword in_col1, const uword in_col2) const
  {
  coot_extra_debug_sigprint();

  coot_debug_check( ((in_col1 > in_col2) || (in_col2 >= Mat<eT>::n_cols) ), "Row::cols(): indices out of bounds or incorrectly used");

  const uword subview_n_cols = in_col2 - in_col1 + 1;

  return subview_row<eT>(*this, 0, in_col1, subview_n_cols);
  }



template<typename eT>
coot_inline
subview_row<eT>
Row<eT>::subvec(const uword in_col1, const uword in_col2)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( ((in_col1 > in_col2) || (in_col2 >= Mat<eT>::n_cols) ), "Row::cols(): indices out of bounds or incorrectly used");

  const uword subview_n_cols = in_col2 - in_col1 + 1;

  return subview_row<eT>(*this, 0, in_col1, subview_n_cols);
  }



template<typename eT>
coot_inline
const subview_row<eT>
Row<eT>::subvec(const uword in_col1, const uword in_col2) const
  {
  coot_extra_debug_sigprint();

  coot_debug_check( ((in_col1 > in_col2) || (in_col2 >= Mat<eT>::n_cols) ), "Row::cols(): indices out of bounds or incorrectly used");

  const uword subview_n_cols = in_col2 - in_col1 + 1;

  return subview_row<eT>(*this, 0, in_col1, subview_n_cols);
  }



template<typename eT>
coot_inline
subview_row<eT>
Row<eT>::subvec(const uword start_col, const SizeMat& s)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (s.n_rows != 1), "Row::subvec(): given size does not specify a row vector" );

  coot_debug_check_bounds( ( (start_col >= Mat<eT>::n_cols) || ((start_col + s.n_cols) > Mat<eT>::n_cols) ), "Row::subvec(): size out of bounds" );

  return subview_row<eT>(*this, 0, start_col, s.n_cols);
  }



template<typename eT>
coot_inline
const subview_row<eT>
Row<eT>::subvec(const uword start_col, const SizeMat& s) const
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (s.n_rows != 1), "Row::subvec(): given size does not specify a row vector" );

  coot_debug_check_bounds( ( (start_col >= Mat<eT>::n_cols) || ((start_col + s.n_cols) > Mat<eT>::n_cols) ), "Row::subvec(): size out of bounds" );

  return subview_row<eT>(*this, 0, start_col, s.n_cols);
  }



#ifdef COOT_EXTRA_ROW_MEAT
  #include COOT_INCFILE_WRAP(COOT_EXTRA_ROW_MEAT)
#endif
