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
Mat<eT>::~Mat()
  {
  coot_extra_debug_sigprint_this(this);

  coot_rt_t::synchronise();

  cleanup();

  coot_type_check(( is_supported_elem_type<eT>::value == false ));
  }



template<typename eT>
inline
Mat<eT>::Mat()
  : n_rows    (0)
  , n_cols    (0)
  , n_elem    (0)
  , vec_state (0)
  , mem_state (0)
  , dev_mem({ NULL, 0 })
  {
  coot_extra_debug_sigprint_this(this);
  }



// construct the matrix to have user specified dimensions
template<typename eT>
inline
Mat<eT>::Mat(const uword in_n_rows, const uword in_n_cols)
  : n_rows    (0)
  , n_cols    (0)
  , n_elem    (0)
  , vec_state (0)
  , mem_state (0)
  , dev_mem({ NULL, 0 })
  {
  coot_extra_debug_sigprint_this(this);

  init(in_n_rows, in_n_cols);

  zeros();  // fill with zeros by default
  }



template<typename eT>
inline
Mat<eT>::Mat(const SizeMat& s)
  : n_rows    (0)
  , n_cols    (0)
  , n_elem    (0)
  , vec_state (0)
  , mem_state (0)
  , dev_mem({ NULL, 0 })
  {
  coot_extra_debug_sigprint_this(this);

  init(s.n_rows, s.n_cols);

  zeros();  // fill with zeros by default
  }



template<typename eT>
template<typename fill_type>
inline
Mat<eT>::Mat(const uword in_n_rows, const uword in_n_cols, const fill::fill_class<fill_type>& f)
  : n_rows    (0)
  , n_cols    (0)
  , n_elem    (0)
  , vec_state (0)
  , mem_state (0)
  , dev_mem({ NULL, 0 })
  {
  coot_extra_debug_sigprint_this(this);

  init(in_n_rows, in_n_cols);

  (*this).fill(f);
  }



template<typename eT>
template<typename fill_type>
inline
Mat<eT>::Mat(const SizeMat& s, const fill::fill_class<fill_type>& f)
  : n_rows    (0)
  , n_cols    (0)
  , n_elem    (0)
  , vec_state (0)
  , mem_state (0)
  , dev_mem({ NULL, 0 })
  {
  coot_extra_debug_sigprint_this(this);

  init(s.n_rows, s.n_cols);

  (*this).fill(f);
  }



template<typename eT>
inline
Mat<eT>::Mat(dev_mem_t<eT> aux_dev_mem, const uword in_n_rows, const uword in_n_cols)
  : n_rows    (in_n_rows)
  , n_cols    (in_n_cols)
  , n_elem    (in_n_rows*in_n_cols)  // TODO: need to check whether the result fits
  , vec_state (0)
  , mem_state (1)
  , dev_mem(aux_dev_mem)
  {
  coot_extra_debug_sigprint_this(this);
  }



template<typename eT>
inline
Mat<eT>::Mat(cl_mem aux_dev_mem, const uword in_n_rows, const uword in_n_cols)
  : n_rows    (in_n_rows)
  , n_cols    (in_n_cols)
  , n_elem    (in_n_rows*in_n_cols)  // TODO: need to check whether the result fits
  , vec_state (0)
  , mem_state (1)
  {
  this->dev_mem.cl_mem_ptr.ptr = aux_dev_mem;
  this->dev_mem.cl_mem_ptr.offset = 0;

  coot_debug_check( get_rt().backend != CL_BACKEND, "Mat(): cannot wrap OpenCL memory when not using OpenCL backend");

  coot_extra_debug_sigprint_this(this);
  }



template<typename eT>
inline
Mat<eT>::Mat(eT* aux_dev_mem, const uword in_n_rows, const uword in_n_cols)
  : n_rows    (in_n_rows)
  , n_cols    (in_n_cols)
  , n_elem    (in_n_rows*in_n_cols)  // TODO: need to check whether the result fits
  , vec_state (0)
  , mem_state (1)
  {
  this->dev_mem.cuda_mem_ptr = aux_dev_mem;

  coot_debug_check( get_rt().backend != CUDA_BACKEND, "Mat(): cannot wrap CUDA memory when not using CUDA backend");

  coot_extra_debug_sigprint_this(this);
  }



template<typename eT>
inline
dev_mem_t<eT>
Mat<eT>::get_dev_mem(const bool sync) const
  {
  coot_extra_debug_sigprint();

  if (sync) { get_rt().synchronise(); }

  return dev_mem;
  }



template<typename eT>
inline
void
Mat<eT>::copy_from_dev_mem(eT* dest_cpu_memptr, const uword N) const
  {
  coot_extra_debug_sigprint();

  if( (n_elem == 0) || (N == 0) )  { return; }

  const uword n_elem_mod = (std::min)(n_elem, N);

  // Treat our device memory as a column vector.
  coot_rt_t::copy_from_dev_mem(dest_cpu_memptr, dev_mem, n_elem_mod, 1, 0, 0, n_elem_mod);
  }



template<typename eT>
inline
void
Mat<eT>::copy_into_dev_mem(const eT* src_cpu_memptr, const uword N)
  {
  coot_extra_debug_sigprint();

  if( (n_elem == 0) || (N == 0) )  { return; }

  const uword n_elem_mod = (std::min)(n_elem, N);

  coot_rt_t::copy_into_dev_mem(dev_mem, src_cpu_memptr, n_elem_mod);
  }



template<typename eT>
inline
Mat<eT>::Mat(const arma::Mat<eT>& X)
  : n_rows   (0)
  , n_cols   (0)
  , n_elem   (0)
  , vec_state(0)
  , mem_state(0)
  , dev_mem({ NULL, 0 })
  {
  coot_extra_debug_sigprint_this(this);

  (*this).operator=(X);
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::operator=(const arma::Mat<eT>& X)
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
Mat<eT>::operator arma::Mat<eT> () const
  {
  coot_extra_debug_sigprint();

  #if defined(COOT_HAVE_ARMA)
    {
    arma::Mat<eT> out(n_rows, n_cols);

    (*this).copy_from_dev_mem(out.memptr(), (*this).n_elem);

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
inline
Mat<eT>::Mat(const char junk, dev_mem_t<eT> aux_dev_mem, const uword in_n_rows, const uword in_n_cols)
  : n_rows    (in_n_rows)
  , n_cols    (in_n_cols)
  , n_elem    (in_n_rows * in_n_cols)  // TODO: need to check whether the result fits
  , vec_state (0)
  , mem_state (3) // fixed memory size
  , dev_mem(aux_dev_mem)
  {
  coot_extra_debug_sigprint_this(this);
  coot_ignore(junk);
  }



template<typename eT>
inline
void
Mat<eT>::cleanup()
  {
  coot_extra_debug_sigprint();

  if((dev_mem.cl_mem_ptr.ptr != NULL) && (mem_state == 0) && (n_elem > 0))
    {
    get_rt().release_memory(dev_mem);
    }

  dev_mem.cl_mem_ptr.ptr = NULL;  // for paranoia
  dev_mem.cl_mem_ptr.offset = 0;
  }



template<typename eT>
inline
void
Mat<eT>::init(const uword new_n_rows, const uword new_n_cols)
  {
  coot_extra_debug_sigprint( coot_str::format("new_n_rows = %d, new_n_cols = %d") % new_n_rows % new_n_cols );

  if( (n_rows == new_n_rows) && (n_cols == new_n_cols) )  { return; }

  uword in_n_rows = new_n_rows;
  uword in_n_cols = new_n_cols;

  // ensure that n_elem can hold the result of (n_rows * n_cols)
  coot_debug_check( ((double(new_n_rows)*double(new_n_cols)) > double(std::numeric_limits<uword>::max())), "Mat::init(): requested size is too large" );

  const uword t_mem_state = mem_state;
  const uword t_vec_state = vec_state;

  bool err_state = false;
  char* err_msg = nullptr;
  const char* error_message_1 = "Mat::init(): size is fixed and hence cannot be changed";
  const char* error_message_2 = "Mat::init(): requested size is not compatible with column vector layout";
  const char* error_message_3 = "Mat::init(): requested size is not compatible with row vector layout";

  coot_debug_set_error( err_state, err_msg, (t_mem_state == 3), error_message_1 );

  if (vec_state > 0)
    {
    if ((in_n_rows == 0) && (in_n_cols == 0))
      {
      if (t_vec_state == 1) { in_n_cols = 1; }
      if (t_vec_state == 2) { in_n_rows = 1; }
      }
    else
      {
      if (t_vec_state == 1) { coot_debug_set_error( err_state, err_msg, (in_n_cols != 1), error_message_2 ); }
      if (t_vec_state == 2) { coot_debug_set_error( err_state, err_msg, (in_n_rows != 1), error_message_3 ); }
      }
    }

  coot_debug_check( err_state, err_msg );

  const uword old_n_elem = n_elem;
  const uword in_n_elem = in_n_rows*in_n_cols;

  if(old_n_elem == in_n_elem)
    {
    coot_extra_debug_print("Mat::init(): reusing memory");
    access::rw(n_rows) = in_n_rows;
    access::rw(n_cols) = in_n_cols;
    }
  else  // condition: old_n_elem != in_n_elem
    {
    if(in_n_elem == 0)
      {
      coot_extra_debug_print("Mat::init(): releasing memory");
      cleanup();
      }
    else
    if(in_n_elem < old_n_elem)
      {
      coot_extra_debug_print("Mat::init(): reusing memory");
      }
    else  // condition: in_n_elem > old_n_elem
      {
      if(old_n_elem > 0)
        {
        coot_extra_debug_print("Mat::init(): releasing memory");
        cleanup();
        }

      coot_extra_debug_print("Mat::init(): acquiring memory");
      dev_mem = get_rt().acquire_memory<eT>(in_n_elem);
      }

    access::rw(n_rows) = in_n_rows;
    access::rw(n_cols) = in_n_cols;
    access::rw(n_elem) = in_n_elem;
    }
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::operator=(const eT val)
  {
  coot_extra_debug_sigprint();

  set_size(1,1);

  coot_rt_t::fill(dev_mem, val, n_rows, n_cols, 0, 0, n_rows);

  return *this;
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::operator+=(const eT val)
  {
  coot_extra_debug_sigprint();

  coot_rt_t::eop_scalar(twoway_kernel_id::equ_array_plus_scalar,
                        dev_mem, dev_mem,
                        val, (eT) 0,
                        n_rows, n_cols, 1,
                        0, 0, 0, n_rows, n_cols,
                        0, 0, 0, n_rows, n_cols);

  return *this;
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::operator-=(const eT val)
  {
  coot_extra_debug_sigprint();

  coot_rt_t::eop_scalar(twoway_kernel_id::equ_array_minus_scalar_post,
                        dev_mem, dev_mem,
                        val, (eT) 0,
                        n_rows, n_cols, 1,
                        0, 0, 0, n_rows, n_cols,
                        0, 0, 0, n_rows, n_cols);

  return *this;
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::operator*=(const eT val)
  {
  coot_extra_debug_sigprint();

  coot_rt_t::eop_scalar(twoway_kernel_id::equ_array_mul_scalar,
                        dev_mem, dev_mem,
                        val, (eT) 1,
                        n_rows, n_cols, 1,
                        0, 0, 0, n_rows, n_cols,
                        0, 0, 0, n_rows, n_cols);

  return *this;
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::operator/=(const eT val)
  {
  coot_extra_debug_sigprint();

  coot_rt_t::eop_scalar(twoway_kernel_id::equ_array_div_scalar_post,
                        dev_mem, dev_mem,
                        val, (eT) 1,
                        n_rows, n_cols, 1,
                        0, 0, 0, n_rows, n_cols,
                        0, 0, 0, n_rows, n_cols);

  return *this;
  }



template<typename eT>
inline
Mat<eT>::Mat(const Mat<eT>& X)
  : n_rows   (0)
  , n_cols   (0)
  , n_elem   (0)
  , vec_state(0)
  , mem_state(0)
  , dev_mem({ NULL, 0 })
  {
  coot_extra_debug_sigprint_this(this);

  (*this).operator=(X);
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::operator=(const Mat<eT>& X)
  {
  coot_extra_debug_sigprint();

  if(this != &X)
    {
    (*this).set_size(X.n_rows, X.n_cols);

    coot_rt_t::copy_mat(dev_mem, X.dev_mem,
                        n_rows, n_cols,
                        0, 0, n_rows,
                        0, 0, X.n_rows);
    }

  return *this;
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::operator+=(const Mat<eT>& X)
  {
  coot_extra_debug_sigprint();

  coot_assert_same_size((*this), X, "Mat::operator+=" );

  coot_rt_t::eop_mat(threeway_kernel_id::equ_array_plus_array,
                     dev_mem, dev_mem, X.dev_mem,
                     n_rows, n_cols,
                     0, 0, n_rows,
                     0, 0, n_rows,
                     0, 0, X.n_rows);

  return *this;
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::operator-=(const Mat<eT>& X)
  {
  coot_extra_debug_sigprint();

  coot_assert_same_size((*this), X, "Mat::operator-=" );

  coot_rt_t::eop_mat(threeway_kernel_id::equ_array_minus_array,
                     dev_mem, dev_mem, X.dev_mem,
                     n_rows, n_cols,
                     0, 0, n_rows,
                     0, 0, n_rows,
                     0, 0, X.n_rows);

  return *this;
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::operator*=(const Mat<eT>& X)
  {
  coot_extra_debug_sigprint();

  Mat<eT> tmp = (*this) * X;

  (*this).steal_mem(tmp);

  return *this;
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::operator%=(const Mat<eT>& X)
  {
  coot_extra_debug_sigprint();

  coot_assert_same_size((*this), X, "Mat::operator%=" );

  coot_rt_t::eop_mat(threeway_kernel_id::equ_array_mul_array,
                     dev_mem, dev_mem, X.dev_mem,
                     n_rows, n_cols,
                     0, 0, n_rows,
                     0, 0, n_rows,
                     0, 0, X.n_rows);

  return *this;
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::operator/=(const Mat<eT>& X)
  {
  coot_extra_debug_sigprint();

  coot_assert_same_size((*this), X, "Mat::operator/=" );

  coot_rt_t::eop_mat(threeway_kernel_id::equ_array_div_array,
                     dev_mem, dev_mem, X.dev_mem,
                     n_rows, n_cols,
                     0, 0, n_rows,
                     0, 0, n_rows,
                     0, 0, X.n_rows);

  return *this;
  }



template<typename eT>
inline
Mat<eT>::Mat(Mat&& X)
  : n_rows   (0)
  , n_cols   (0)
  , n_elem   (0)
  , vec_state(0)
  , mem_state(0)
  {
  coot_extra_debug_sigprint_this(this);

  (*this).steal_mem(X);
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::operator=(Mat<eT>&& X)
  {
  coot_extra_debug_sigprint();

  (*this).steal_mem(X);

  return *this;
  }



template<typename eT>
inline
void
Mat<eT>::steal_mem(Mat<eT>& X)
  {
  coot_extra_debug_sigprint();

  if(this == &X) { return; }

  if (X.mem_state == 0)
    {
    reset(); // clear any existing memory

    access::rw(n_rows)    = X.n_rows;
    access::rw(n_cols)    = X.n_cols;
    access::rw(n_elem)    = X.n_elem;
    access::rw(vec_state) = X.vec_state;
    access::rw(mem_state) = X.mem_state;
    access::rw(dev_mem)   = X.dev_mem;

    access::rw(X.n_rows)             = 0;
    access::rw(X.n_cols)             = 0;
    access::rw(X.n_elem)             = 0;
    access::rw(X.vec_state)          = 0;
    access::rw(X.mem_state)          = 0;
    access::rw(X.dev_mem.cl_mem_ptr) = { NULL, 0 };
    }
  else
    {
    (*this).operator=(X);
    }
  }



template<typename eT>
inline
Mat<eT>::Mat(const subview<eT>& X)
  : n_rows   (0)
  , n_cols   (0)
  , n_elem   (0)
  , vec_state(0)
  , mem_state(0)
  , dev_mem({ NULL, 0 })
  {
  coot_extra_debug_sigprint_this(this);

  (*this).operator=(X);
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::operator=(const subview<eT>& X)
  {
  coot_extra_debug_sigprint();

  const bool alias = (this == &(X.m));

  if(alias == false)
    {
    set_size(X.n_rows, X.n_cols);

    subview<eT>::extract(*this, X);
    }
  else
    {
    Mat<eT> tmp(X);

    steal_mem(tmp);
    }

  return *this;
  }


template<typename eT>
inline
const Mat<eT>&
Mat<eT>::operator+=(const subview<eT>& X)
  {
  coot_extra_debug_sigprint();

  coot_debug_assert_same_size(n_rows, n_cols, X.n_rows, X.n_cols, "Mat::operator+=");

  subview<eT>::plus_inplace(*this, X);

  return *this;
  }


template<typename eT>
inline
const Mat<eT>&
Mat<eT>::operator-=(const subview<eT>& X)
  {
  coot_extra_debug_sigprint();

  coot_debug_assert_same_size(n_rows, n_cols, X.n_rows, X.n_cols, "Mat::operator-=");

  subview<eT>::minus_inplace(*this, X);

  return *this;
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::operator*=(const subview<eT>& X)
  {
  coot_extra_debug_sigprint();

  // TODO: improve this implementation (maybe?)
  Mat<eT> tmp(X);
  return operator*=(tmp);
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::operator%=(const subview<eT>& X)
  {
  coot_extra_debug_sigprint();

  coot_debug_assert_same_size(n_rows, n_cols, X.n_rows, X.n_cols, "Mat::operator%=");

  subview<eT>::schur_inplace(*this, X);

  return *this;
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::operator/=(const subview<eT>& X)
  {
  coot_extra_debug_sigprint();

  coot_debug_assert_same_size(n_rows, n_cols, X.n_rows, X.n_cols, "Mat::operator/=");

  subview<eT>::div_inplace(*this, X);

  return *this;
  }



template<typename eT>
inline
Mat<eT>::Mat(const diagview<eT>& X)
  : n_rows   (0)
  , n_cols   (0)
  , n_elem   (0)
  , vec_state(0)
  , mem_state(0)
  , dev_mem({ NULL, 0 })
  {
  coot_extra_debug_sigprint_this(this);

  diagview<eT>::extract(*this, X);
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::operator=(const diagview<eT>& X)
  {
  coot_extra_debug_sigprint();

  const bool alias = (this == &(X.m));

  if (alias == false)
    {
    diagview<eT>::extract(*this, X);
    }
  else
    {
    Mat<eT> tmp(X);
    steal_mem(tmp);
    }

  return *this;
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::operator+=(const diagview<eT>& X)
  {
  coot_extra_debug_sigprint();

  // Extract the diagview, and then add.
  Mat<eT> diag(X);
  return operator+=(diag);
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::operator-=(const diagview<eT>& X)
  {
  coot_extra_debug_sigprint();

  // Extract the diagview, and then subtract.
  Mat<eT> diag(X);
  return operator-=(diag);
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::operator*=(const diagview<eT>& X)
  {
  coot_extra_debug_sigprint();

  // Extract the diagview, and then multiply.
  Mat<eT> diag(X);
  return operator*=(diag);
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::operator%=(const diagview<eT>& X)
  {
  coot_extra_debug_sigprint();

  // Extract the diagview, and then multiply.
  Mat<eT> diag(X);
  return operator%=(diag);
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::operator/=(const diagview<eT>& X)
  {
  coot_extra_debug_sigprint();

  // Extract the diagview, and then divide.
  Mat<eT> diag(X);
  return operator/=(diag);
  }



template<typename eT>
template<typename T1, typename eop_type>
inline
Mat<eT>::Mat(const eOp<T1, eop_type>& X)
  : n_rows   (0)
  , n_cols   (0)
  , n_elem   (0)
  , vec_state(0)
  , mem_state(0)
  , dev_mem({ NULL, 0 })
  {
  coot_extra_debug_sigprint_this(this);

  (*this).operator=(X);
  }



template<typename eT>
template<typename T1, typename eop_type>
inline
const Mat<eT>&
Mat<eT>::operator=(const eOp<T1, eop_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));

  // eop_core currently forcefully unwraps submatrices to matrices,
  // so currently there can't be dangerous aliasing with the out matrix

  set_size(X.get_n_rows(), X.get_n_cols());

  eop_type::apply(*this, X);

  return *this;
  }



template<typename eT>
template<typename T1, typename eop_type>
inline
const Mat<eT>&
Mat<eT>::operator+=(const eOp<T1, eop_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));

  coot_assert_same_size(n_rows, n_cols, X.get_n_rows(), X.get_n_cols(), "Mat::operator+=");

  eop_type::apply_inplace_plus(*this, X);

  return *this;
  }



template<typename eT>
template<typename T1, typename eop_type>
inline
const Mat<eT>&
Mat<eT>::operator-=(const eOp<T1, eop_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));

  coot_assert_same_size(n_rows, n_cols, X.get_n_rows(), X.get_n_cols(), "Mat::operator-=");

  eop_type::apply_inplace_minus(*this, X);

  return *this;
  }



template<typename eT>
template<typename T1, typename eop_type>
inline
const Mat<eT>&
Mat<eT>::operator*=(const eOp<T1, eop_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));

  Mat<eT> tmp = (*this) * X;

  (*this).steal_mem(tmp);

  return *this;
  }



template<typename eT>
template<typename T1, typename eop_type>
inline
const Mat<eT>&
Mat<eT>::operator%=(const eOp<T1, eop_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));

  coot_assert_same_size(n_rows, n_cols, X.get_n_rows(), X.get_n_cols(), "Mat::operator%=");

  eop_type::apply_inplace_schur(*this, X);

  return *this;
  }



template<typename eT>
template<typename T1, typename eop_type>
inline
const Mat<eT>&
Mat<eT>::operator/=(const eOp<T1, eop_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));

  coot_assert_same_size(n_rows, n_cols, X.get_n_rows(), X.get_n_cols(), "Mat::operator/=");

  eop_type::apply_inplace_div(*this, X);

  return *this;
  }



template<typename eT>
template<typename T1, typename T2, typename eglue_type>
inline
Mat<eT>::Mat(const eGlue<T1, T2, eglue_type>& X)
  : n_rows   (0)
  , n_cols   (0)
  , n_elem   (0)
  , vec_state(0)
  , mem_state(0)
  , dev_mem({ NULL, 0 })
  {
  coot_extra_debug_sigprint_this(this);

  (*this).operator=(X);
  }



template<typename eT>
template<typename T1, typename T2, typename eglue_type>
inline
const Mat<eT>&
Mat<eT>::operator=(const eGlue<T1, T2, eglue_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  coot_type_check(( is_same_type< eT, typename T2::elem_type >::no ));

  // eglue_core currently forcefully unwraps submatrices to matrices,
  // so currently there can't be dangerous aliasing with the out matrix

  set_size(X.get_n_rows(), X.get_n_cols());

  eglue_type::apply(*this, X);

  return *this;
  }



template<typename eT>
template<typename T1, typename T2, typename eglue_type>
inline
const Mat<eT>&
Mat<eT>::operator+=(const eGlue<T1, T2, eglue_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  coot_type_check(( is_same_type< eT, typename T2::elem_type >::no ));

  coot_assert_same_size(n_rows, n_cols, X.get_n_rows(), X.get_n_cols(), "Mat::operator+=");

  eglue_type::apply_inplace_plus(*this, X);

  return *this;
  }



template<typename eT>
template<typename T1, typename T2, typename eglue_type>
inline
const Mat<eT>&
Mat<eT>::operator-=(const eGlue<T1, T2, eglue_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  coot_type_check(( is_same_type< eT, typename T2::elem_type >::no ));

  coot_assert_same_size(n_rows, n_cols, X.get_n_rows(), X.get_n_cols(), "Mat::operator-=");

  eglue_type::apply_inplace_minus(*this, X);

  return *this;
  }



template<typename eT>
template<typename T1, typename T2, typename eglue_type>
inline
const Mat<eT>&
Mat<eT>::operator*=(const eGlue<T1, T2, eglue_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  coot_type_check(( is_same_type< eT, typename T2::elem_type >::no ));

  Mat<eT> tmp = (*this) * X;

  (*this).steal_mem(tmp);

  return *this;
  }



template<typename eT>
template<typename T1, typename T2, typename eglue_type>
inline
const Mat<eT>&
Mat<eT>::operator%=(const eGlue<T1, T2, eglue_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  coot_type_check(( is_same_type< eT, typename T2::elem_type >::no ));

  coot_assert_same_size(n_rows, n_cols, X.get_n_rows(), X.get_n_cols(), "Mat::operator%=");

  eglue_type::apply_inplace_schur(*this, X);

  return *this;
  }



template<typename eT>
template<typename T1, typename T2, typename eglue_type>
inline
const Mat<eT>&
Mat<eT>::operator/=(const eGlue<T1, T2, eglue_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  coot_type_check(( is_same_type< eT, typename T2::elem_type >::no ));

  coot_assert_same_size(n_rows, n_cols, X.get_n_rows(), X.get_n_cols(), "Mat::operator/=");

  eglue_type::apply_inplace_div(*this, X);

  return *this;
  }



template<typename eT>
template<typename T1, typename mtop_type>
inline
Mat<eT>::Mat(const mtOp<eT, T1, mtop_type>& X)
  : n_rows   (0)
  , n_cols   (0)
  , n_elem   (0)
  , vec_state(0)
  , mem_state(0)
  , dev_mem({ NULL, 0 })
  {
  coot_extra_debug_sigprint_this(this);

  (*this).operator=(X);
  }



template<typename eT>
template<typename T1, typename mtop_type>
inline
const Mat<eT>&
Mat<eT>::operator=(const mtOp<eT, T1, mtop_type>& X)
  {
  coot_extra_debug_sigprint();

  mtop_type::apply(*this, X);

  return *this;
  }



template<typename eT>
template<typename T1, typename mtop_type>
inline
const Mat<eT>&
Mat<eT>::operator+=(const mtOp<eT, T1, mtop_type>& X)
  {
  coot_extra_debug_sigprint();

  SizeProxy<mtOp<eT, T1, mtop_type>> S(X);
  coot_assert_same_size(n_rows, n_cols, S.get_n_rows(), S.get_n_cols(), "Mat::operator+=");

  mtop_type::apply_inplace_plus(*this, X);

  return (*this);
  }



template<typename eT>
template<typename T1, typename mtop_type>
inline
const Mat<eT>&
Mat<eT>::operator-=(const mtOp<eT, T1, mtop_type>& X)
  {
  coot_extra_debug_sigprint();

  SizeProxy<mtOp<eT, T1, mtop_type>> S(X);
  coot_assert_same_size(n_rows, n_cols, S.get_n_rows(), S.get_n_cols(), "Mat::operator-=");

  mtop_type::apply_inplace_minus(*this, X);

  return *this;
  }



template<typename eT>
template<typename T1, typename mtop_type>
inline
const Mat<eT>&
Mat<eT>::operator*=(const mtOp<eT, T1, mtop_type>& X)
  {
  coot_extra_debug_sigprint();

  SizeProxy<mtOp<eT, T1, mtop_type>> S(X);
  coot_assert_same_size(n_rows, n_cols, S.get_n_rows(), S.get_n_cols(), "Mat::operator*=");

  mtop_type::apply_inplace_times(*this, X);

  return *this;
  }



template<typename eT>
template<typename T1, typename mtop_type>
inline
const Mat<eT>&
Mat<eT>::operator%=(const mtOp<eT, T1, mtop_type>& X)
  {
  coot_extra_debug_sigprint();

  SizeProxy<mtOp<eT, T1, mtop_type>> S(X);
  coot_assert_same_size(n_rows, n_cols, S.get_n_rows(), S.get_n_cols(), "Mat::operator%=");

  mtop_type::apply_inplace_schur(*this, X);

  return *this;
  }



template<typename eT>
template<typename T1, typename mtop_type>
inline
const Mat<eT>&
Mat<eT>::operator/=(const mtOp<eT, T1, mtop_type>& X)
  {
  coot_extra_debug_sigprint();

  SizeProxy<mtOp<eT, T1, mtop_type>> S(X);
  coot_assert_same_size(n_rows, n_cols, S.get_n_rows(), S.get_n_cols(), "Mat::operator/=");

  mtop_type::apply_inplace_div(*this, X);

  return *this;
  }




template<typename eT>
template<typename T1, typename op_type>
inline
Mat<eT>::Mat(const Op<T1, op_type>& X)
  : n_rows   (0)
  , n_cols   (0)
  , n_elem   (0)
  , vec_state(0)
  , mem_state(0)
  , dev_mem({ NULL, 0 })
  {
  coot_extra_debug_sigprint_this(this);

  (*this).operator=(X);
  }



template<typename eT>
template<typename T1, typename op_type>
inline
const Mat<eT>&
Mat<eT>::operator=(const Op<T1, op_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));

  op_type::apply(*this, X);

  return *this;
  }



template<typename eT>
template<typename T1, typename op_type>
inline
const Mat<eT>&
Mat<eT>::operator+=(const Op<T1, op_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));

  const unwrap<Op<T1, op_type>> U(X);

  return (*this).operator+=(U.M);
  }



template<typename eT>
template<typename T1, typename op_type>
inline
const Mat<eT>&
Mat<eT>::operator-=(const Op<T1, op_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));

  const unwrap<Op<T1, op_type>> U(X);

  return (*this).operator-=(U.M);
  }



template<typename eT>
template<typename T1, typename op_type>
inline
const Mat<eT>&
Mat<eT>::operator*=(const Op<T1, op_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));

  Mat<eT> tmp = (*this) * X;

  (*this).steal_mem(tmp);

  return *this;
  }



template<typename eT>
template<typename T1, typename op_type>
inline
const Mat<eT>&
Mat<eT>::operator%=(const Op<T1, op_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));

  const unwrap<Op<T1, op_type>> U(X);

  return (*this).operator*=(U.M);
  }



template<typename eT>
template<typename T1, typename op_type>
inline
const Mat<eT>&
Mat<eT>::operator/=(const Op<T1, op_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));

  const unwrap<Op<T1, op_type>> U(X);

  return (*this).operator/=(U.M);
  }



template<typename eT>
template<typename T1, typename T2, typename glue_type>
inline
Mat<eT>::Mat(const Glue<T1, T2, glue_type>& X)
  : n_rows   (0)
  , n_cols   (0)
  , n_elem   (0)
  , vec_state(0)
  , mem_state(0)
  , dev_mem({ NULL, 0 })
  {
  coot_extra_debug_sigprint_this(this);

  (*this).operator=(X);
  }



template<typename eT>
template<typename T1, typename T2, typename glue_type>
inline
const Mat<eT>&
Mat<eT>::operator=(const Glue<T1, T2, glue_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  coot_type_check(( is_same_type< eT, typename T2::elem_type >::no ));

  glue_type::apply(*this, X);

  return *this;
  }



template<typename eT>
template<typename T1, typename T2, typename glue_type>
inline
const Mat<eT>&
Mat<eT>::operator+=(const Glue<T1, T2, glue_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  coot_type_check(( is_same_type< eT, typename T2::elem_type >::no ));

  const Mat<eT> m(X);

  return (*this).operator+=(m);
  }



template<typename eT>
template<typename T1, typename T2, typename glue_type>
inline
const Mat<eT>&
Mat<eT>::operator-=(const Glue<T1, T2, glue_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  coot_type_check(( is_same_type< eT, typename T2::elem_type >::no ));

  const Mat<eT> m(X);

  return (*this).operator-=(m);
  }



template<typename eT>
template<typename T1, typename T2, typename glue_type>
inline
const Mat<eT>&
Mat<eT>::operator*=(const Glue<T1, T2, glue_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  coot_type_check(( is_same_type< eT, typename T2::elem_type >::no ));

  Mat<eT> tmp = (*this) * X;

  (*this).steal_mem(tmp);

  return *this;
  }



template<typename eT>
template<typename T1, typename T2, typename glue_type>
inline
const Mat<eT>&
Mat<eT>::operator%=(const Glue<T1, T2, glue_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  coot_type_check(( is_same_type< eT, typename T2::elem_type >::no ));

  const Mat<eT> m(X);

  return (*this).operator%=(m);
  }



template<typename eT>
template<typename T1, typename T2, typename glue_type>
inline
const Mat<eT>&
Mat<eT>::operator/=(const Glue<T1, T2, glue_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  coot_type_check(( is_same_type< eT, typename T2::elem_type >::no ));

  const Mat<eT> m(X);

  return (*this).operator/=(m);
  }



template<typename eT>
template<typename T1, typename T2, typename mtglue_type>
inline
Mat<eT>::Mat(const mtGlue<eT, T1, T2, mtglue_type>& X)
  : n_rows   (0)
  , n_cols   (0)
  , n_elem   (0)
  , vec_state(0)
  , mem_state(0)
  , dev_mem({ NULL, 0 })
  {
  coot_extra_debug_sigprint_this(this);

  (*this).operator=(X);
  }



template<typename eT>
template<typename T1, typename T2, typename mtglue_type>
inline
const Mat<eT>&
Mat<eT>::operator=(const mtGlue<eT, T1, T2, mtglue_type>& X)
  {
  coot_extra_debug_sigprint();

  mtglue_type::apply(*this, X);

  return *this;
  }



template<typename eT>
template<typename T1, typename T2, typename mtglue_type>
inline
const Mat<eT>&
Mat<eT>::operator+=(const mtGlue<eT, T1, T2, mtglue_type>& X)
  {
  coot_extra_debug_sigprint();

  const Mat<eT> m(X);

  return (*this).operator+=(m);
  }



template<typename eT>
template<typename T1, typename T2, typename mtglue_type>
inline
const Mat<eT>&
Mat<eT>::operator-=(const mtGlue<eT, T1, T2, mtglue_type>& X)
  {
  coot_extra_debug_sigprint();

  const Mat<eT> m(X);

  return (*this).operator-=(m);
  }



template<typename eT>
template<typename T1, typename T2, typename mtglue_type>
inline
const Mat<eT>&
Mat<eT>::operator*=(const mtGlue<eT, T1, T2, mtglue_type>& X)
  {
  coot_extra_debug_sigprint();

  Mat<eT> tmp = (*this) * X;

  (*this).steal_mem(tmp);

  return *this;
  }



template<typename eT>
template<typename T1, typename T2, typename mtglue_type>
inline
const Mat<eT>&
Mat<eT>::operator%=(const mtGlue<eT, T1, T2, mtglue_type>& X)
  {
  coot_extra_debug_sigprint();

  const Mat<eT> m(X);

  return (*this).operator%=(m);
  }



template<typename eT>
template<typename T1, typename T2, typename mtglue_type>
inline
const Mat<eT>&
Mat<eT>::operator/=(const mtGlue<eT, T1, T2, mtglue_type>& X)
  {
  coot_extra_debug_sigprint();

  const Mat<eT> m(X);

  return (*this).operator/=(m);
  }



template<typename eT>
template<typename T1, typename op_type>
inline
Mat<eT>::Mat(const CubeToMatOp<T1, op_type>& X)
  : n_rows   (0)
  , n_cols   (0)
  , n_elem   (0)
  , vec_state(0)
  , mem_state(0)
  , dev_mem({ NULL, 0 })
  {
  coot_extra_debug_sigprint_this(this);

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));

  op_type::apply(*this, X);
  }



template<typename eT>
template<typename T1, typename op_type>
inline
Mat<eT>&
Mat<eT>::operator=(const CubeToMatOp<T1, op_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));

  op_type::apply(*this, X);
  }



template<typename eT>
template<typename T1, typename op_type>
inline
Mat<eT>&
Mat<eT>::operator+=(const CubeToMatOp<T1, op_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));

  (*this) = (*this) + X;

  return (*this);
  }
  


template<typename eT>
template<typename T1, typename op_type>
inline
Mat<eT>&
Mat<eT>::operator-=(const CubeToMatOp<T1, op_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));

  (*this) = (*this) - X;

  return (*this);
  }



template<typename eT>
template<typename T1, typename op_type>
inline
Mat<eT>&
Mat<eT>::operator*=(const CubeToMatOp<T1, op_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));

  Mat<eT> tmp = (*this) * X;

  (*this).steal_mem(tmp);

  return *this;
  }



template<typename eT>
template<typename T1, typename op_type>
inline
Mat<eT>&
Mat<eT>::operator%=(const CubeToMatOp<T1, op_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));

  (*this) = (*this) % X;

  return (*this);
  }



template<typename eT>
template<typename T1, typename op_type>
inline
Mat<eT>&
Mat<eT>::operator/=(const CubeToMatOp<T1, op_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));

  (*this) = (*this) / X;

  return (*this);
  }



template<typename eT>
coot_inline
diagview<eT>
Mat<eT>::diag(const sword in_id)
  {
  coot_extra_debug_sigprint();

  const uword row_offset = (in_id < 0) ? uword(-in_id) : 0;
  const uword col_offset = (in_id > 0) ? uword( in_id) : 0;

  coot_debug_check_bounds
      (
      ((row_offset > 0) && (row_offset >= n_rows)) || ((col_offset > 0) && (col_offset >= n_cols)),
      "Mat::diag(): requested diagonal out of bounds"
      );

  const uword len = (std::min)(n_rows - row_offset, n_cols - col_offset);

  return diagview<eT>(*this, row_offset, col_offset, len);
  }



template<typename eT>
coot_inline
const diagview<eT>
Mat<eT>::diag(const sword in_id) const
  {
  coot_extra_debug_sigprint();

  const uword row_offset = (in_id < 0) ? uword(-in_id) : 0;
  const uword col_offset = (in_id > 0) ? uword( in_id) : 0;

  coot_debug_check_bounds
      (
      ((row_offset > 0) && (row_offset >= n_rows)) || ((col_offset > 0) && (col_offset >= n_cols)),
      "Mat::diag(): requested diagonal out of bounds"
      );

  const uword len = (std::min)(n_rows - row_offset, n_cols - col_offset);

  return diagview<eT>(*this, row_offset, col_offset, len);
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::clamp(const eT min_val, const eT max_val)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (min_val > max_val), "clamp(): min_val must be less than max_val" );

  coot_rt_t::clamp(get_dev_mem(false), get_dev_mem(false),
                   min_val, max_val,
                   n_rows, n_cols,
                   0, 0, n_rows,
                   0, 0, n_rows);

  return *this;
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::fill(const eT val)
  {
  coot_extra_debug_sigprint();

  coot_rt_t::fill(dev_mem, val, n_rows, n_cols, 0, 0, n_rows);

  return *this;
  }



template<typename eT>
template<typename fill_type>
inline
const Mat<eT>&
Mat<eT>::fill(const fill::fill_class<fill_type>&)
  {
  coot_extra_debug_sigprint();
  
  if(is_same_type<fill_type, fill::fill_zeros>::yes)  { (*this).zeros(); }
  if(is_same_type<fill_type, fill::fill_ones >::yes)  { (*this).ones();  }
  if(is_same_type<fill_type, fill::fill_eye  >::yes)  { (*this).eye();   }
  if(is_same_type<fill_type, fill::fill_randu>::yes)  { (*this).randu(); }
  if(is_same_type<fill_type, fill::fill_randn>::yes)  { (*this).randn(); }
  
  return *this;
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::zeros()
  {
  coot_extra_debug_sigprint();

  (*this).fill(eT(0));

  return *this;
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::zeros(const uword new_n_elem)
  {
  coot_extra_debug_sigprint();

  (*this).set_size(new_n_elem);
  (*this).fill(eT(0));

  return *this;
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::zeros(const uword new_n_rows, const uword new_n_cols)
  {
  coot_extra_debug_sigprint();

  (*this).set_size(new_n_rows, new_n_cols);
  (*this).fill(eT(0));

  return *this;
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::zeros(const SizeMat& s)
  {
  coot_extra_debug_sigprint();

  (*this).set_size(s);
  (*this).fill(eT(0));

  return *this;
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::ones()
  {
  coot_extra_debug_sigprint();

  (*this).fill(eT(1));

  return *this;
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::ones(const uword new_n_elem)
  {
  coot_extra_debug_sigprint();

  (*this).set_size(new_n_elem);
  (*this).fill(eT(1));

  return *this;
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::ones(const uword new_n_rows, const uword new_n_cols)
  {
  coot_extra_debug_sigprint();

  (*this).set_size(new_n_rows, new_n_cols);
  (*this).fill(eT(1));

  return *this;
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::ones(const SizeMat& s)
  {
  coot_extra_debug_sigprint();

  (*this).set_size(s);
  (*this).fill(eT(1));

  return *this;
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::randu()
  {
  coot_extra_debug_sigprint();

  coot_rng::fill_randu(dev_mem, n_elem);

  return *this;
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::randu(const uword new_n_elem)
  {
  coot_extra_debug_sigprint();

  set_size(new_n_elem);

  return (*this).randu();
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::randu(const uword new_n_rows, const uword new_n_cols)
  {
  coot_extra_debug_sigprint();

  set_size(new_n_rows, new_n_cols);

  return (*this).randu();
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::randu(const SizeMat& s)
  {
  coot_extra_debug_sigprint();

  set_size(s);

  return (*this).randu();
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::randn()
  {
  coot_extra_debug_sigprint();

  coot_rng::fill_randn(dev_mem, n_elem);

  return *this;
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::randn(const uword new_n_elem)
  {
  coot_extra_debug_sigprint();

  set_size(new_n_elem);

  return (*this).randn();
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::randn(const uword new_n_rows, const uword new_n_cols)
  {
  coot_extra_debug_sigprint();

  set_size(new_n_rows, new_n_cols);

  return (*this).randn();
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::randn(const SizeMat& s)
  {
  coot_extra_debug_sigprint();

  set_size(s);

  return (*this).randn();
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::eye()
  {
  coot_extra_debug_sigprint();

  if (n_elem == 0)
    {
    return *this;
    }

  coot_rt_t::eye(dev_mem, n_rows, n_cols);

  return *this;
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::eye(const uword new_n_rows, const uword new_n_cols)
  {
  coot_extra_debug_sigprint();

  (*this).set_size(new_n_rows, new_n_cols);
  (*this).eye();

  return *this;
  }



template<typename eT>
inline
const Mat<eT>&
Mat<eT>::eye(const SizeMat& s)
  {
  coot_extra_debug_sigprint();

  (*this).set_size(s.n_rows, s.n_cols);
  (*this).eye();

  return *this;
  }



template<typename eT>
inline
void
Mat<eT>::reset()
  {
  coot_extra_debug_sigprint();

  uword new_n_rows = 0;
  uword new_n_cols = 0;

  switch(vec_state)
    {
    case  0:                 break;
    case  1: new_n_cols = 1; break;
    case  2: new_n_rows = 1; break;
    default: ;
    }

  init(new_n_rows, new_n_cols);
  }



template<typename eT>
inline
void
Mat<eT>::set_size(const uword new_n_elem)
  {
  coot_extra_debug_sigprint();

  uword new_n_rows = 0;
  uword new_n_cols = 0;

  switch(vec_state)
    {
    case  0: new_n_rows = new_n_elem; new_n_cols = 1;          break;
    case  1: new_n_rows = new_n_elem; new_n_cols = 1;          break;
    case  2: new_n_rows =          1; new_n_cols = new_n_elem; break;
    default: ;
    }

  init(new_n_rows, new_n_cols);
  }



template<typename eT>
inline
void
Mat<eT>::set_size(const uword new_n_rows, const uword new_n_cols)
  {
  coot_extra_debug_sigprint();

  init(new_n_rows, new_n_cols);
  }



template<typename eT>
inline
void
Mat<eT>::set_size(const SizeMat& s)
  {
  coot_extra_debug_sigprint();

  init(s.n_rows, s.n_cols);
  }



template<typename eT>
inline
void
Mat<eT>::resize(const uword new_n_elem)
  {
  coot_extra_debug_sigprint();

  switch(vec_state)
    {
    case 0:
      // fallthrough
    case 1:
      (*this).resize(new_n_elem, 1);
      break;

    case 2:
      (*this).resize(1, new_n_elem);
      break;

    default:
      ;
    }
  }



template<typename eT>
inline
void
Mat<eT>::reshape(const uword new_n_rows, const uword new_n_cols)
  {
  coot_extra_debug_sigprint();

  if (new_n_rows == 0 || new_n_cols == 0)
    {
    // Shortcut: just clear the memory.
    set_size(new_n_rows, new_n_cols);
    }
  else
    {
    op_reshape::apply_direct(*this, *this, new_n_rows, new_n_cols);
    }
  }



template<typename eT>
inline
void
Mat<eT>::reshape(const SizeMat& s)
  {
  coot_extra_debug_sigprint();

  reshape(s.n_rows, s.n_cols);
  }



template<typename eT>
inline
void
Mat<eT>::resize(const uword new_n_rows, const uword new_n_cols)
  {
  coot_extra_debug_sigprint();

  op_resize::apply_mat_inplace((*this), new_n_rows, new_n_cols);
  }



template<typename eT>
inline
void
Mat<eT>::resize(const SizeMat& s)
  {
  coot_extra_debug_sigprint();

  op_resize::apply_mat_inplace((*this), s.n_rows, s.n_cols);
  }



template<typename eT>
inline
eT
Mat<eT>::min() const
  {
  coot_extra_debug_sigprint();

  return coot_rt_t::min_vec(dev_mem, n_elem);
  }



template<typename eT>
inline
eT
Mat<eT>::max() const
  {
  coot_extra_debug_sigprint();

  return coot_rt_t::max_vec(dev_mem, n_elem);
  }



template<typename eT>
inline
eT
Mat<eT>::min(uword& index_of_min_val) const
  {
  coot_extra_debug_sigprint();

  eT result = eT(0);
  index_of_min_val = coot_rt_t::index_min_vec(dev_mem, n_elem, &result);
  return result;
  }



template<typename eT>
inline
eT
Mat<eT>::max(uword& index_of_max_val) const
  {
  coot_extra_debug_sigprint();

  eT result = eT(0);
  index_of_max_val = coot_rt_t::index_max_vec(dev_mem, n_elem, &result);
  return result;
  }



template<typename eT>
inline
eT
Mat<eT>::min(uword& row_of_min_val, uword& col_of_min_val) const
  {
  coot_extra_debug_sigprint();

  eT result = eT(0);
  uword index = coot_rt_t::index_min_vec(dev_mem, n_elem, &result);

  row_of_min_val = index % n_rows;
  col_of_min_val = index / n_rows;

  return result;
  }



template<typename eT>
inline
eT
Mat<eT>::max(uword& row_of_max_val, uword& col_of_max_val) const
  {
  coot_extra_debug_sigprint();

  eT result = eT(0);
  uword index = coot_rt_t::index_max_vec(dev_mem, n_elem, &result);

  row_of_max_val = index % n_rows;
  col_of_max_val = index / n_rows;

  return result;
  }



template<typename eT>
inline
bool
Mat<eT>::is_vec() const
  {
  return ((n_rows == 1) || (n_cols == 1));
  }



template<typename eT>
inline
bool
Mat<eT>::is_colvec() const
  {
  return (n_cols == 1);
  }



template<typename eT>
inline
bool
Mat<eT>::is_rowvec() const
  {
  return (n_rows == 1);
  }



template<typename eT>
inline
bool
Mat<eT>::is_square() const
  {
  return (n_rows == n_cols);
  }



template<typename eT>
inline
bool
Mat<eT>::is_empty() const
  {
  return (n_elem == 0);
  }



//! resets this matrix to an empty matrix
template<typename eT>
inline
void
Mat<eT>::clear()
  {
  reset();
  }



//! returns true if the matrix has no elements
template<typename eT>
inline
bool
Mat<eT>::empty() const
  {
  return (n_elem == 0);
  }



//! return the number of elements in the matrix
template<typename eT>
inline
uword
Mat<eT>::size() const
  {
  return n_elem;
  }



template<typename eT>
inline
eT
Mat<eT>::front() const
  {
  coot_debug_check( (n_elem == 0), "Mat::front(): matrix is empty" );

  return at(0);
  }



template<typename eT>
inline
eT
Mat<eT>::back() const
  {
  coot_debug_check( (n_elem == 0), "Mat::back(): matrix is empty" );

  return at(n_elem - 1);
  }



template<typename eT>
coot_inline
uword
Mat<eT>::get_n_rows() const
  {
  return n_rows;
  }



template<typename eT>
coot_inline
uword
Mat<eT>::get_n_cols() const
  {
  return n_cols;
  }



template<typename eT>
coot_inline
uword
Mat<eT>::get_n_elem() const
  {
  return n_elem;
  }



// linear element accessor without bounds check; this is very slow - do not use it unless absolutely necessary
template<typename eT>
inline
MatValProxy<eT>
Mat<eT>::operator[] (const uword ii)
  {
  return MatValProxy<eT>(*this, ii);
  }



// linear element accessor without bounds check; this is very slow - do not use it unless absolutely necessary
template<typename eT>
inline
eT
Mat<eT>::operator[] (const uword ii) const
  {
  return MatValProxy<eT>::get_val(*this, ii);
  }



// linear element accessor without bounds check; this is very slow - do not use it unless absolutely necessary
template<typename eT>
inline
MatValProxy<eT>
Mat<eT>::at(const uword ii)
  {
  return MatValProxy<eT>(*this, ii);
  }



// linear element accessor without bounds check; this is very slow - do not use it unless absolutely necessary
template<typename eT>
inline
eT
Mat<eT>::at(const uword ii) const
  {
  return MatValProxy<eT>::get_val(*this, ii);
  }



// linear element accessor with bounds check; this is very slow - do not use it unless absolutely necessary
template<typename eT>
inline
MatValProxy<eT>
Mat<eT>::operator() (const uword ii)
  {
  coot_debug_check( (ii >= n_elem), "Mat::operator(): index out of bounds");

  return MatValProxy<eT>(*this, ii);
  }



// linear element accessor with bounds check; this is very slow - do not use it unless absolutely necessary
template<typename eT>
inline
eT
Mat<eT>::operator() (const uword ii) const
  {
  coot_debug_check( (ii >= n_elem), "Mat::operator(): index out of bounds");

  return MatValProxy<eT>::get_val(*this, ii);
  }



template<typename eT>
inline
MatValProxy<eT>
Mat<eT>::at(const uword in_row, const uword in_col)
  {
  return MatValProxy<eT>(*this, (in_row + in_col*n_rows));
  }



template<typename eT>
inline
eT
Mat<eT>::at(const uword in_row, const uword in_col) const
  {
  return MatValProxy<eT>::get_val(*this, (in_row + in_col*n_rows));
  }



template<typename eT>
inline
MatValProxy<eT>
Mat<eT>::operator() (const uword in_row, const uword in_col)
  {
  coot_debug_check( ((in_row >= n_rows) || (in_col >= n_cols)), "Mat::operator(): index out of bounds");

  return MatValProxy<eT>(*this, (in_row + in_col*n_rows));
  }



template<typename eT>
inline
eT
Mat<eT>::operator() (const uword in_row, const uword in_col) const
  {
  coot_debug_check( ((in_row >= n_rows) || (in_col >= n_cols)), "Mat::operator(): index out of bounds");

  return MatValProxy<eT>::get_val(*this, (in_row + in_col*n_rows));
  }



template<typename eT>
coot_inline
subview_row<eT>
Mat<eT>::row(const uword row_num)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( row_num >= n_rows, "Mat::row(): index out of bounds" );

  return subview_row<eT>(*this, row_num);
  }



template<typename eT>
coot_inline
const subview_row<eT>
Mat<eT>::row(const uword row_num) const
  {
  coot_extra_debug_sigprint();

  coot_debug_check( row_num >= n_rows, "Mat::row(): index out of bounds" );

  return subview_row<eT>(*this, row_num);
  }



template<typename eT>
inline
subview_row<eT>
Mat<eT>::operator()(const uword row_num, const span& col_span)
  {
  coot_extra_debug_sigprint();

  const bool col_all = col_span.whole;

  const uword local_n_cols = n_cols;

  const uword in_col1       = col_all ? 0            : col_span.a;
  const uword in_col2       =                          col_span.b;
  const uword submat_n_cols = col_all ? local_n_cols : in_col2 - in_col1 + 1;

  coot_debug_check
    (
    (row_num >= n_rows)
    ||
    ( col_all ? false : ((in_col1 > in_col2) || (in_col2 >= local_n_cols)) )
    ,
    "Mat::operator(): indices out of bounds or incorrectly used"
    );

  return subview_row<eT>(*this, row_num, in_col1, submat_n_cols);
  }



template<typename eT>
inline
const subview_row<eT>
Mat<eT>::operator()(const uword row_num, const span& col_span) const
  {
  coot_extra_debug_sigprint();

  const bool col_all = col_span.whole;

  const uword local_n_cols = n_cols;

  const uword in_col1       = col_all ? 0            : col_span.a;
  const uword in_col2       =                          col_span.b;
  const uword submat_n_cols = col_all ? local_n_cols : in_col2 - in_col1 + 1;

  coot_debug_check
    (
    (row_num >= n_rows)
    ||
    ( col_all ? false : ((in_col1 > in_col2) || (in_col2 >= local_n_cols)) )
    ,
    "Mat::operator(): indices out of bounds or incorrectly used"
    );

  return subview_row<eT>(*this, row_num, in_col1, submat_n_cols);
  }



template<typename eT>
coot_inline
subview_col<eT>
Mat<eT>::col(const uword col_num)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( col_num >= n_cols, "Mat::col(): index out of bounds");

  return subview_col<eT>(*this, col_num);
  }



template<typename eT>
coot_inline
const subview_col<eT>
Mat<eT>::col(const uword col_num) const
  {
  coot_extra_debug_sigprint();

  coot_debug_check( col_num >= n_cols, "Mat::col(): index out of bounds");

  return subview_col<eT>(*this, col_num);
  }



template<typename eT>
inline
subview_col<eT>
Mat<eT>::operator()(const span& row_span, const uword col_num)
  {
  coot_extra_debug_sigprint();

  const bool row_all = row_span.whole;

  const uword local_n_rows = n_rows;

  const uword in_row1       = row_all ? 0            : row_span.a;
  const uword in_row2       =                          row_span.b;
  const uword submat_n_rows = row_all ? local_n_rows : in_row2 - in_row1 + 1;

  coot_debug_check
    (
    (col_num >= n_cols)
    ||
    ( row_all ? false : ((in_row1 > in_row2) || (in_row2 >= local_n_rows)) )
    ,
    "Mat::operator(): indices out of bounds or incorrectly used"
    );

  return subview_col<eT>(*this, col_num, in_row1, submat_n_rows);
  }



template<typename eT>
inline
const subview_col<eT>
Mat<eT>::operator()(const span& row_span, const uword col_num) const
  {
  coot_extra_debug_sigprint();

  const bool row_all = row_span.whole;

  const uword local_n_rows = n_rows;

  const uword in_row1       = row_all ? 0            : row_span.a;
  const uword in_row2       =                          row_span.b;
  const uword submat_n_rows = row_all ? local_n_rows : in_row2 - in_row1 + 1;

  coot_debug_check
    (
    (col_num >= n_cols)
    ||
    ( row_all ? false : ((in_row1 > in_row2) || (in_row2 >= local_n_rows)) )
    ,
    "Mat::operator(): indices out of bounds or incorrectly used"
    );

  return subview_col<eT>(*this, col_num, in_row1, submat_n_rows);
  }



template<typename eT>
coot_inline
subview<eT>
Mat<eT>::rows(const uword in_row1, const uword in_row2)
  {
  coot_extra_debug_sigprint();

  coot_debug_check
    (
    (in_row1 > in_row2) || (in_row2 >= n_rows),
    "Mat::rows(): indices out of bounds or incorrectly used"
    );

  const uword subview_n_rows = in_row2 - in_row1 + 1;

  return subview<eT>(*this, in_row1, 0, subview_n_rows, n_cols );
  }



template<typename eT>
coot_inline
const subview<eT>
Mat<eT>::rows(const uword in_row1, const uword in_row2) const
  {
  coot_extra_debug_sigprint();

  coot_debug_check
    (
    (in_row1 > in_row2) || (in_row2 >= n_rows),
    "Mat::rows(): indices out of bounds or incorrectly used"
    );

  const uword subview_n_rows = in_row2 - in_row1 + 1;

  return subview<eT>(*this, in_row1, 0, subview_n_rows, n_cols );
  }



template<typename eT>
coot_inline
subview<eT>
Mat<eT>::cols(const uword in_col1, const uword in_col2)
  {
  coot_extra_debug_sigprint();

  coot_debug_check
    (
    (in_col1 > in_col2) || (in_col2 >= n_cols),
    "Mat::cols(): indices out of bounds or incorrectly used"
    );

  const uword subview_n_cols = in_col2 - in_col1 + 1;

  return subview<eT>(*this, 0, in_col1, n_rows, subview_n_cols);
  }



template<typename eT>
coot_inline
const subview<eT>
Mat<eT>::cols(const uword in_col1, const uword in_col2) const
  {
  coot_extra_debug_sigprint();

  coot_debug_check
    (
    (in_col1 > in_col2) || (in_col2 >= n_cols),
    "Mat::cols(): indices out of bounds or incorrectly used"
    );

  const uword subview_n_cols = in_col2 - in_col1 + 1;

  return subview<eT>(*this, 0, in_col1, n_rows, subview_n_cols);
  }



template<typename eT>
inline
subview<eT>
Mat<eT>::rows(const span& row_span)
  {
  coot_extra_debug_sigprint();

  const bool row_all = row_span.whole;

  const uword local_n_rows = n_rows;

  const uword in_row1       = row_all ? 0            : row_span.a;
  const uword in_row2       =                          row_span.b;
  const uword submat_n_rows = row_all ? local_n_rows : in_row2 - in_row1 + 1;

  coot_debug_check
    (
    ( row_all ? false : ((in_row1 > in_row2) || (in_row2 >= local_n_rows)) )
    ,
    "Mat::rows(): indices out of bounds or incorrectly used"
    );

  return subview<eT>(*this, in_row1, 0, submat_n_rows, n_cols);
  }



template<typename eT>
inline
const subview<eT>
Mat<eT>::rows(const span& row_span) const
  {
  coot_extra_debug_sigprint();

  const bool row_all = row_span.whole;

  const uword local_n_rows = n_rows;

  const uword in_row1       = row_all ? 0            : row_span.a;
  const uword in_row2       =                          row_span.b;
  const uword submat_n_rows = row_all ? local_n_rows : in_row2 - in_row1 + 1;

  coot_debug_check
    (
    ( row_all ? false : ((in_row1 > in_row2) || (in_row2 >= local_n_rows)) )
    ,
    "Mat::rows(): indices out of bounds or incorrectly used"
    );

  return subview<eT>(*this, in_row1, 0, submat_n_rows, n_cols);
  }



template<typename eT>
coot_inline
subview<eT>
Mat<eT>::cols(const span& col_span)
  {
  coot_extra_debug_sigprint();

  const bool col_all = col_span.whole;

  const uword local_n_cols = n_cols;

  const uword in_col1       = col_all ? 0            : col_span.a;
  const uword in_col2       =                          col_span.b;
  const uword submat_n_cols = col_all ? local_n_cols : in_col2 - in_col1 + 1;

  coot_debug_check
    (
    ( col_all ? false : ((in_col1 > in_col2) || (in_col2 >= local_n_cols)) )
    ,
    "Mat::cols(): indices out of bounds or incorrectly used"
    );

  return subview<eT>(*this, 0, in_col1, n_rows, submat_n_cols);
  }



template<typename eT>
coot_inline
const subview<eT>
Mat<eT>::cols(const span& col_span) const
  {
  coot_extra_debug_sigprint();

  const bool col_all = col_span.whole;

  const uword local_n_cols = n_cols;

  const uword in_col1       = col_all ? 0            : col_span.a;
  const uword in_col2       =                          col_span.b;
  const uword submat_n_cols = col_all ? local_n_cols : in_col2 - in_col1 + 1;

  coot_debug_check
    (
    ( col_all ? false : ((in_col1 > in_col2) || (in_col2 >= local_n_cols)) )
    ,
    "Mat::cols(): indices out of bounds or incorrectly used"
    );

  return subview<eT>(*this, 0, in_col1, n_rows, submat_n_cols);
  }



template<typename eT>
coot_inline
subview_each1<Mat<eT>, 0>
Mat<eT>::each_col()
  {
  coot_extra_debug_sigprint();

  return subview_each1<Mat<eT>, 0>(*this);
  }



template<typename eT>
coot_inline
subview_each1<Mat<eT>, 1>
Mat<eT>::each_row()
  {
  coot_extra_debug_sigprint();

  return subview_each1<Mat<eT>, 1>(*this);
  }



template<typename eT>
coot_inline
const subview_each1<Mat<eT>, 0>
Mat<eT>::each_col() const
  {
  coot_extra_debug_sigprint();

  return subview_each1<Mat<eT>, 0>(*this);
  }



template<typename eT>
coot_inline
const subview_each1<Mat<eT>, 1>
Mat<eT>::each_row() const
  {
  coot_extra_debug_sigprint();

  return subview_each1<Mat<eT>, 1>(*this);
  }



template<typename eT>
template<typename T1>
inline
subview_each2<Mat<eT>, 0, T1>
Mat<eT>::each_col(const Base<uword, T1>& indices)
  {
  coot_extra_debug_sigprint();

  return subview_each2<Mat<eT>, 0, T1>(*this, indices);
  }



template<typename eT>
template<typename T1>
inline
subview_each2<Mat<eT>, 1, T1>
Mat<eT>::each_row(const Base<uword, T1>& indices)
  {
  coot_extra_debug_sigprint();

  return subview_each2<Mat<eT>, 1, T1>(*this, indices);
  }



template<typename eT>
template<typename T1>
inline
const subview_each2<Mat<eT>, 0, T1>
Mat<eT>::each_col(const Base<uword, T1>& indices) const
  {
  coot_extra_debug_sigprint();

  return subview_each2<Mat<eT>, 0, T1>(*this, indices);
  }



template<typename eT>
template<typename T1>
inline
const subview_each2<Mat<eT>, 1, T1>
Mat<eT>::each_row(const Base<uword, T1>& indices) const
  {
  coot_extra_debug_sigprint();

  return subview_each2<Mat<eT>, 1, T1>(*this, indices);
  }



template<typename eT>
coot_inline
subview<eT>
Mat<eT>::submat(const uword in_row1, const uword in_col1, const uword in_row2, const uword in_col2)
  {
  coot_extra_debug_sigprint();

  coot_debug_check
    (
    (in_row1 > in_row2) || (in_col1 >  in_col2) || (in_row2 >= n_rows) || (in_col2 >= n_cols),
    "Mat::submat(): indices out of bounds or incorrectly used"
    );

  const uword subview_n_rows = in_row2 - in_row1 + 1;
  const uword subview_n_cols = in_col2 - in_col1 + 1;

  return subview<eT>(*this, in_row1, in_col1, subview_n_rows, subview_n_cols);
  }



template<typename eT>
coot_inline
const subview<eT>
Mat<eT>::submat(const uword in_row1, const uword in_col1, const uword in_row2, const uword in_col2) const
  {
  coot_extra_debug_sigprint();

  coot_debug_check
    (
    (in_row1 > in_row2) || (in_col1 >  in_col2) || (in_row2 >= n_rows) || (in_col2 >= n_cols),
    "Mat::submat(): indices out of bounds or incorrectly used"
    );

  const uword subview_n_rows = in_row2 - in_row1 + 1;
  const uword subview_n_cols = in_col2 - in_col1 + 1;

  return subview<eT>(*this, in_row1, in_col1, subview_n_rows, subview_n_cols);
  }



template<typename eT>
coot_inline
subview<eT>
Mat<eT>::submat(const uword in_row1, const uword in_col1, const SizeMat& s)
  {
  coot_extra_debug_sigprint();

  const uword l_n_rows = n_rows;
  const uword l_n_cols = n_cols;

  const uword s_n_rows = s.n_rows;
  const uword s_n_cols = s.n_cols;

  coot_debug_check
    (
    ((in_row1 >= l_n_rows) || (in_col1 >= l_n_cols) || ((in_row1 + s_n_rows) > l_n_rows) || ((in_col1 + s_n_cols) > l_n_cols)),
    "Mat::submat(): indices or size out of bounds"
    );

  return subview<eT>(*this, in_row1, in_col1, s_n_rows, s_n_cols);
  }



template<typename eT>
coot_inline
const subview<eT>
Mat<eT>::submat(const uword in_row1, const uword in_col1, const SizeMat& s) const
  {
  coot_extra_debug_sigprint();

  const uword l_n_rows = n_rows;
  const uword l_n_cols = n_cols;

  const uword s_n_rows = s.n_rows;
  const uword s_n_cols = s.n_cols;

  coot_debug_check
    (
    ((in_row1 >= l_n_rows) || (in_col1 >= l_n_cols) || ((in_row1 + s_n_rows) > l_n_rows) || ((in_col1 + s_n_cols) > l_n_cols)),
    "Mat::submat(): indices or size out of bounds"
    );

  return subview<eT>(*this, in_row1, in_col1, s_n_rows, s_n_cols);
  }



template<typename eT>
inline
subview<eT>
Mat<eT>::submat(const span& row_span, const span& col_span)
  {
  coot_extra_debug_sigprint();

  const bool row_all = row_span.whole;
  const bool col_all = col_span.whole;

  const uword local_n_rows = n_rows;
  const uword local_n_cols = n_cols;

  const uword in_row1       = row_all ? 0            : row_span.a;
  const uword in_row2       =                          row_span.b;
  const uword submat_n_rows = row_all ? local_n_rows : in_row2 - in_row1 + 1;

  const uword in_col1       = col_all ? 0            : col_span.a;
  const uword in_col2       =                          col_span.b;
  const uword submat_n_cols = col_all ? local_n_cols : in_col2 - in_col1 + 1;

  coot_debug_check
    (
    ( row_all ? false : ((in_row1 > in_row2) || (in_row2 >= local_n_rows)) )
    ||
    ( col_all ? false : ((in_col1 > in_col2) || (in_col2 >= local_n_cols)) )
    ,
    "Mat::submat(): indices out of bounds or incorrectly used"
    );

  return subview<eT>(*this, in_row1, in_col1, submat_n_rows, submat_n_cols);
  }



template<typename eT>
inline
const subview<eT>
Mat<eT>::submat(const span& row_span, const span& col_span) const
  {
  coot_extra_debug_sigprint();

  const bool row_all = row_span.whole;
  const bool col_all = col_span.whole;

  const uword local_n_rows = n_rows;
  const uword local_n_cols = n_cols;

  const uword in_row1       = row_all ? 0            : row_span.a;
  const uword in_row2       =                          row_span.b;
  const uword submat_n_rows = row_all ? local_n_rows : in_row2 - in_row1 + 1;

  const uword in_col1       = col_all ? 0            : col_span.a;
  const uword in_col2       =                          col_span.b;
  const uword submat_n_cols = col_all ? local_n_cols : in_col2 - in_col1 + 1;

  coot_debug_check
    (
    ( row_all ? false : ((in_row1 > in_row2) || (in_row2 >= local_n_rows)) )
    ||
    ( col_all ? false : ((in_col1 > in_col2) || (in_col2 >= local_n_cols)) )
    ,
    "Mat::submat(): indices out of bounds or incorrectly used"
    );

  return subview<eT>(*this, in_row1, in_col1, submat_n_rows, submat_n_cols);
  }



template<typename eT>
inline
subview<eT>
Mat<eT>::operator()(const span& row_span, const span& col_span)
  {
  coot_extra_debug_sigprint();

  return (*this).submat(row_span, col_span);
  }



template<typename eT>
inline
const subview<eT>
Mat<eT>::operator()(const span& row_span, const span& col_span) const
  {
  coot_extra_debug_sigprint();

  return (*this).submat(row_span, col_span);
  }



template<typename eT>
inline
subview<eT>
Mat<eT>::operator()(const uword in_row1, const uword in_col1, const SizeMat& s)
  {
  coot_extra_debug_sigprint();

  return (*this).submat(in_row1, in_col1, s);
  }



template<typename eT>
inline
const subview<eT>
Mat<eT>::operator()(const uword in_row1, const uword in_col1, const SizeMat& s) const
  {
  coot_extra_debug_sigprint();

  return (*this).submat(in_row1, in_col1, s);
  }



template<typename eT>
inline
subview<eT>
Mat<eT>::head_rows(const uword N)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (N > n_rows), "Mat::head_rows(): size out of bounds");

  return subview<eT>(*this, 0, 0, N, n_cols);
  }



template<typename eT>
inline
const subview<eT>
Mat<eT>::head_rows(const uword N) const
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (N > n_rows), "Mat::head_rows(): size out of bounds");

  return subview<eT>(*this, 0, 0, N, n_cols);
  }



template<typename eT>
inline
subview<eT>
Mat<eT>::tail_rows(const uword N)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (N > n_rows), "Mat::tail_rows(): size out of bounds");

  const uword start_row = n_rows - N;

  return subview<eT>(*this, start_row, 0, N, n_cols);
  }



template<typename eT>
inline
const subview<eT>
Mat<eT>::tail_rows(const uword N) const
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (N > n_rows), "Mat::tail_rows(): size out of bounds");

  const uword start_row = n_rows - N;

  return subview<eT>(*this, start_row, 0, N, n_cols);
  }



template<typename eT>
inline
subview<eT>
Mat<eT>::head_cols(const uword N)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (N > n_cols), "Mat::head_cols(): size out of bounds");

  return subview<eT>(*this, 0, 0, n_rows, N);
  }



template<typename eT>
inline
const subview<eT>
Mat<eT>::head_cols(const uword N) const
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (N > n_cols), "Mat::head_cols(): size out of bounds");

  return subview<eT>(*this, 0, 0, n_rows, N);
  }



template<typename eT>
inline
subview<eT>
Mat<eT>::tail_cols(const uword N)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (N > n_cols), "Mat::tail_cols(): size out of bounds");

  const uword start_col = n_cols - N;

  return subview<eT>(*this, 0, start_col, n_rows, N);
  }



template<typename eT>
inline
const subview<eT>
Mat<eT>::tail_cols(const uword N) const
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (N > n_cols), "Mat::tail_cols(): size out of bounds");

  const uword start_col = n_cols - N;

  return subview<eT>(*this, 0, start_col, n_rows, N);
  }



#ifdef COOT_EXTRA_MAT_MEAT
  #include COOT_INCFILE_WRAP(COOT_EXTRA_MAT_MEAT)
#endif
