// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
// Copyright 2025      Ryan Curtin (http://www.ratml.org)
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



template<typename vec_type, typename eT>
coot_warn_unused
inline
typename
enable_if2
  <
  is_Mat<vec_type>::value,
  vec_type
  >::result
regspace
  (
  const typename vec_type::pod_type start,
  const          eT                 delta,
  const typename vec_type::pod_type end
  )
  {
  coot_extra_debug_sigprint();

  if( ((start < end) && (delta < eT(0))) || ((start > end) && (delta > eT(0))) || (delta == eT(0)) )  { return vec_type(); }

  const bool ascend = (start <= end);

  const eT inc = (delta < eT(0)) ? eT(-delta) : eT(delta);

  const eT M = ((ascend) ? eT(end-start) : eT(start-end)) / eT(inc);

  const uword N = uword(1) + ( (is_non_integral<eT>::value) ? uword(std::floor(double(M))) : uword(M) );

  vec_type x;
  x.set_size(N);

  if(x.n_elem == 0)
    {
    if(is_Mat_only<vec_type>::value)  { x.set_size(1, 0); }
    }
  else
    {
    coot_rt_t::regspace(x.get_dev_mem(false), 1, start, typename vec_type::pod_type(inc), end, N, !ascend);
    }

  return x;
  }



template<typename vec_type>
coot_warn_unused
inline
typename
enable_if2
  <
  is_Mat<vec_type>::value,
  vec_type
  >::result
regspace
  (
  const typename vec_type::pod_type start,
  const typename vec_type::pod_type end
  )
  {
  coot_extra_debug_sigprint();

  typedef typename vec_type::pod_type eT;

  const bool ascend = (start <= end);
  const uword N = uword(1) + uword((ascend) ? (end-start) : (start-end));

  vec_type x;
  x.set_size(N);

  if(x.n_elem == 0)
    {
    if(is_Mat_only<vec_type>::value)  { x.set_size(1, 0); }
    }
  else
    {
    coot_rt_t::regspace(x.get_dev_mem(false), 1, start, eT(1), end, N, !ascend);
    }

  return x;
  }



coot_warn_unused
inline
vec
regspace(const double start, const double delta, const double end)
  {
  coot_extra_debug_sigprint();

  return regspace<vec>(start, delta, end);
  }



coot_warn_unused
inline
vec
regspace(const double start, const double end)
  {
  coot_extra_debug_sigprint();

  return regspace<vec>(start, end);
  }
