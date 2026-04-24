// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2017-2023 Ryan Curtin (https://www.ratml.org)
// Copyright 2017      Conrad Sanderson (http://conradsanderson.id.au)
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
MatValProxy<eT>::MatValProxy(Mat<eT>& in_M, const uword in_index)
  : dev_mem(in_M.get_dev_mem(false))
  , index  (in_index)
  {
  coot_debug_sigprint();
  }



template<typename eT>
coot_inline
MatValProxy<eT>::MatValProxy(Cube<eT>& in_M, const uword in_index)
  : dev_mem(in_M.get_dev_mem(false))
  , index  (in_index)
  {
  coot_debug_sigprint();
  }



template<typename eT>
coot_inline
MatValProxy<eT>::operator eT() const
  {
  coot_debug_sigprint();

  return coot_rt_t::get_val(dev_mem, index);
  }



template<typename eT>
inline
eT
MatValProxy<eT>::get_val(const Mat<eT>& M, const uword index)
  {
  coot_debug_sigprint();

  return coot_rt_t::get_val(M.dev_mem, index);
  }



template<typename eT>
inline
eT
MatValProxy<eT>::get_val(const Cube<eT>& M, const uword index)
  {
  coot_debug_sigprint();

  return coot_rt_t::get_val(M.dev_mem, index);
  }



template<typename eT>
inline
void
MatValProxy<eT>::operator=(const MatValProxy<eT>& in_val)
  {
  coot_rt_t::copy(make_proxy(*this), make_proxy(in_val));
  }



template<typename eT>
inline
void
MatValProxy<eT>::operator+=(const MatValProxy<eT>& in_val)
  {
  // TODO: use eGlue to perform a fully on-device operation
  *this += eT(in_val);
  }



template<typename eT>
inline
void
MatValProxy<eT>::operator-=(const MatValProxy<eT>& in_val)
  {
  // TODO: use eGlue to perform a fully on-device operation
  *this = eT(in_val);
  }



template<typename eT>
inline
void
MatValProxy<eT>::operator*=(const MatValProxy<eT>& in_val)
  {
  // TODO: use eGlue to perform a fully on-device operation
  *this = eT(in_val);
  }



template<typename eT>
inline
void
MatValProxy<eT>::operator/=(const MatValProxy<eT>& in_val)
  {
  // TODO: use eGlue to perform a fully on-device operation
  *this = eT(in_val);
  }



template<typename eT>
inline
void
MatValProxy<eT>::operator=(const eT in_val)
  {
  coot_debug_sigprint();

  coot_rt_t::set_val(dev_mem, index, in_val);
  }



template<typename eT>
inline
void
MatValProxy<eT>::operator+=(const eT in_val)
  {
  coot_debug_sigprint();

  coot_rt_t::copy(make_proxy(*this), Proxy< eOp<MatValProxy<eT>, eop_scalar_plus> >(*this, in_val) );
  }



template<typename eT>
inline
void
MatValProxy<eT>::operator-=(const eT in_val)
  {
  coot_debug_sigprint();

  coot_rt_t::copy(make_proxy(*this), Proxy< eOp<MatValProxy<eT>, eop_scalar_minus_post> >(*this, in_val) );
  }



template<typename eT>
inline
void
MatValProxy<eT>::operator*=(const eT in_val)
  {
  coot_debug_sigprint();

  coot_rt_t::copy(make_proxy(*this), Proxy< eOp<MatValProxy<eT>, eop_scalar_times> >(*this, in_val) );
  }



template<typename eT>
inline
void
MatValProxy<eT>::operator/=(const eT in_val)
  {
  coot_debug_sigprint();

  coot_rt_t::copy(make_proxy(*this), Proxy< eOp<MatValProxy<eT>, eop_scalar_div_post> >(*this, in_val) );
  }
