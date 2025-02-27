// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2024 Ryan Curtin (http://www.ratml.org/)
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


template<typename out_type>
coot_warn_unused
inline
typename
enable_if2
  <
  is_Mat<out_type>::value,
  out_type
  >::result
logspace
  (
  const typename out_type::pod_type start,
  const typename out_type::pod_type end,
  const uword                       num = 50u
  )
  {
  coot_extra_debug_sigprint();

  out_type x;
  x.set_size(num);

  coot_rt_t::logspace(x.get_dev_mem(false), 1, start, end, num);
  return x;
  }



coot_warn_unused
inline
vec
logspace(const double start, const double end, const uword num = 50u)
  {
  coot_extra_debug_sigprint();
  return logspace<vec>(start, end, num);
  }
