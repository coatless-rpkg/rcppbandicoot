// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2023 Ryan Curtin (http://www.ratml.org)
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



// Note that Armadillo has also an op_diagvec2 class that handles diagonals that
// are not the main diagonal; but, that gives us no extra optimization
// opportunities for Bandicoot, so we only use op_diagvec here to capture both
// cases.

template<typename T1>
coot_warn_unused
inline
typename
enable_if2
  <
  is_coot_type<T1>::value,
  const Op<T1, op_diagvec>
  >::result
diagvec(const T1& X, const sword k = 0)
  {
  coot_extra_debug_sigprint();

  const uword a = (std::abs)(k);
  const uword b = (k < 0) ? 1 : 0;

  return Op<T1, op_diagvec>(X, a, b);
  }
