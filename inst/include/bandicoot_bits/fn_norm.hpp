// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2023      Marcus Edel (http://kurg.org)
// Copyright 2008-2023 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
//
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


//! \addtogroup fn_norm
//! @{



template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value, typename T1::pod_type >::result
norm
  (
  const T1&   X,
  const uword k = uword(2),
  const typename coot_real_or_cx_only<typename T1::elem_type>::result* junk = nullptr
  )
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  typedef typename T1::pod_type T;

  const SizeProxy<T1> S(X);
  if(S.get_n_elem() == 0)
    {
    return T(0);
    }

  const bool is_vec = (S.get_n_rows() == 1) || (S.get_n_cols() == 1);

  // At this point, unwrapping is unavoidable, since we perform the norm computation directly and immediately.
  const unwrap<T1> U(X);

  if(is_vec)
    {
    if(k == uword(1))  { return op_norm::vec_norm_1(U.M); }
    if(k == uword(2))  { return op_norm::vec_norm_2(U.M); }

    coot_debug_check( (k == 0), "norm(): k must be greater than zero" );

    return op_norm::vec_norm_k(U.M, int(k));
    }
  else
    {
    if(k == uword(1))  { return op_norm::mat_norm_1(U.M); }
    if(k == uword(2))  { return op_norm::mat_norm_2(U.M); }

    coot_stop_logic_error("norm(): unsupported matrix norm type");
    }

  return T(0);
  }



template<typename T1>
coot_warn_unused
inline
typename enable_if2< is_coot_type<T1>::value, typename T1::pod_type >::result
norm
  (
  const T1&   X,
  const char* method,
  const typename coot_real_or_cx_only<typename T1::elem_type>::result* junk = nullptr
  )
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  typedef typename T1::pod_type T;

  const SizeProxy<T1> S(X);
  if (S.get_n_elem() == 0)
    {
    return T(0);
    }

  const char sig    = (method != nullptr) ? method[0] : char(0);
  const bool is_vec = (S.get_n_rows() == 1) || (S.get_n_cols() == 1);

  // At this point, unwrapping is unavoidable, since we perform the norm computation directly and immediately.
  const unwrap<T1> U(X);

  if (is_vec)
    {
    if( (sig == 'i') || (sig == 'I') || (sig == '+') )  { return op_norm::vec_norm_max(U.M); }
    if( (sig == '-')                                 )  { return op_norm::vec_norm_min(U.M); }
    if( (sig == 'f') || (sig == 'F')                 )  { return op_norm::vec_norm_2(U.M);   }

    coot_stop_logic_error("norm(): unsupported vector norm type");
    }
  else
    {
    if( (sig == 'i') || (sig == 'I') || (sig == '+') )   // inf norm
      {
      return op_norm::mat_norm_inf(U.M);
      }
    else
    if( (sig == 'f') || (sig == 'F') )
      {
      return op_norm::vec_norm_2(U.M);
      }

    coot_stop_logic_error("norm(): unsupported matrix norm type");
    }

  return T(0);
  }
