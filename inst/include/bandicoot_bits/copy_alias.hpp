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



// Utility to ensure that an alias is copied into a new matrix.

template<typename eT>
struct copy_alias
  {
  explicit inline copy_alias(const Mat<eT>& in, const Mat<eT>& out)
    : M_internal((&in == &out) ? new Mat<eT>(in) : NULL)
    , M         ((&in == &out) ? *M_internal     : in)
    {
    coot_extra_debug_sigprint();
    }

  template<typename eT2>
  explicit inline copy_alias(const Mat<eT>& in, const Mat<eT2>& /* out */)
    : M_internal(NULL)
    , M(in)
    {
    coot_extra_debug_sigprint();
    }

  ~copy_alias()
    {
    if (M_internal)
      {
      delete M_internal;
      }
    }

  Mat<eT>* M_internal;
  const Mat<eT>& M;
  };
