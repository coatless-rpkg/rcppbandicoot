// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2023 Ryan Curtin (http://ratml.org)
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



template<typename T>
coot_warn_unused
inline
T
eye(const uword n_rows, const uword n_cols, const typename coot_Mat_Col_Row_only<T>::result* junk = nullptr)
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  if (is_Col<T>::value)
    {
    coot_debug_check( (n_cols != 1), "eye(): incompatible size" );
    }
  else if (is_Row<T>::value)
    {
    coot_debug_check( (n_rows != 1), "eye(): incompatible size" );
    }

  T out(n_rows, n_cols);
  out.eye();
  return out;
  }



template<typename T>
coot_warn_unused
inline
T
eye(const SizeMat& s, const typename coot_Mat_Col_Row_only<T>::result* junk = nullptr)
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  return eye<T>(s.n_rows, s.n_cols);
  }
