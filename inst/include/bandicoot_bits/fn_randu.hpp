// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2022      Marcus Edel (http://kurg.org)
// Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
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



template<typename T>
coot_warn_unused
inline
T
randu(const uword n_rows, const uword n_cols, const typename coot_Mat_Col_Row_only<T>::result* junk = nullptr)
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  if (is_Col<T>::value)
    {
    coot_debug_check( (n_cols != 1), "randu(): incompatible size" );
    }
  else if (is_Row<T>::value)
    {
    coot_debug_check( (n_rows != 1), "randu(): incompatible size" );
    }

  T out(n_rows, n_cols);
  coot_rng::fill_randu(out.get_dev_mem(false), out.n_elem);
  return out;
  }



template<typename T>
coot_warn_unused
inline
T
randu(const uword n_elem, const typename coot_Mat_Col_Row_only<T>::result* junk = nullptr)
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  const uword n_rows = (is_Row<T>::value) ? uword(1) : n_elem;
  const uword n_cols = (is_Row<T>::value) ? n_elem   : uword(1);

  T out(n_rows, n_cols);
  coot_rng::fill_randu(out.get_dev_mem(false), out.n_elem);
  return out;
  }



template<typename T>
coot_warn_unused
inline
T
randu(const SizeMat& s, const typename coot_Mat_Col_Row_only<T>::result* junk = nullptr)
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  return randu<T>(s.n_rows, s.n_cols);
  }



template<typename T>
coot_warn_unused
inline
T
randu(const uword n_rows, const uword n_cols, const uword n_slices, const typename coot_Cube_only<T>::result* junk = nullptr)
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  T out(n_rows, n_cols, n_slices);
  coot_rng::fill_randu(out.get_dev_mem(false), out.n_elem);
  return out;
  }



template<typename T>
coot_warn_unused
inline
T
randu(const SizeCube& s, const typename coot_Cube_only<T>::result* junk = nullptr)
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  return randu<T>(s.n_rows, s.n_cols, s.n_slices);
  }
