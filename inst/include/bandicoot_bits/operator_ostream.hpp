// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2017-2023 Ryan Curtin (https://www.ratml.org)
// Copyright 2008-2016 Conrad Sanderson (https://conradsanderson.id.au)
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



template<typename eT, typename T1>
inline
std::ostream&
operator<< (std::ostream& o, const Base<eT, T1>& X)
  {
  coot_extra_debug_sigprint();

  const unwrap<T1> tmp(X.get_ref());

  coot_ostream::print(o, tmp.M, true);

  return o;
  }



template<typename eT, typename T1>
inline
std::ostream&
operator<< (std::ostream& o, const BaseCube<eT, T1>& X)
  {
  coot_extra_debug_sigprint();

  const unwrap_cube<T1> tmp(X.get_ref());

  coot_ostream::print(o, tmp.M, true);

  return o;
  }



inline
std::ostream&
operator<< (std::ostream& o, const SizeMat& S)
  {
  coot_extra_debug_sigprint();

  coot_ostream::print(o, S);

  return o;
  }



inline
std::ostream&
operator<< (std::ostream& o, const SizeCube& S)
  {
  coot_extra_debug_sigprint();

  coot_ostream::print(o, S);

  return o;
  }
