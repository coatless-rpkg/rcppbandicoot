// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Ryan Curtin (http://www.ratml.org/)
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



// The alias_wrapper is a utility struct that should be used by any kernel (e.g.
// coot_rt:: backend function) where the input and output cannot be the same
// memory.
//
// The structure is created with the two (or more) arguments that should be
// checked for aliasing.  If any of the second and later arguments are aliases
// of the first argument, an internal matrix is initialized to have the same
// size as the output matrix.
//
// Whether or not the arguments are aliases, the operation should then be run
// using the .get_dev_mem(), .get_row_offset(), and .get_col_offset() members.
//
// When the alias_wrapper object is destructed, it will have the original output
// matrix take control of any memory that was initialized (via .steal_mem()).



template<typename T1>
struct is_any_alias
  {
  inline is_any_alias(const T1& in_p) : in(in_p) { }

  template<typename T2, typename... Args>
  coot_inline
  typename enable_if2< is_Mat<T2>::value || is_subview<T2>::value || is_diagview<T2>::value, bool >::result
  check(const T2& t2, const Args&... args)
    {
    return is_alias(in, t2) || check(args...);
    }

  template<typename T2, typename... Args>
  coot_inline
  typename enable_if2< !is_Mat<T2>::value && !is_subview<T2>::value && !is_diagview<T2>::value, bool >::result
  check(const T2&, const Args&... args)
    {
    return check(args...);
    }

  template<typename T2>
  coot_inline
  typename enable_if2< is_Mat<T2>::value || is_subview<T2>::value || is_diagview<T2>::value, bool >::result
  check(const T2& t2)
    {
    return is_alias(in, t2);
    }

  template<typename T2>
  constexpr static
  coot_inline
  typename enable_if2< !is_Mat<T2>::value && !is_subview<T2>::value && !is_diagview<T2>::value, bool >::result
  check(const T2&)
    {
    return false;
    }

  const T1& in;
  };



template<typename TDest, typename... TArgs>
struct alias_wrapper
  {
  // dest: the output matrix to be used for the operation
  // arg: the input, which may be an alias of the output
  alias_wrapper(TDest& dest_in, const TArgs&... args)
    : dest(dest_in)
    , using_aux(is_any_alias<TDest>(dest).check(args...))
    , use(using_aux ? aux : dest)
    {
    if (using_aux)
      {
      aux.set_size(dest.n_rows, dest.n_cols);
      }
    }

  ~alias_wrapper()
    {
    if (using_aux)
      {
      dest.steal_mem(aux);
      }
    }

  typedef typename TDest::elem_type elem_type;
  typedef TDest                     stored_type;

  inline uword get_n_rows() const { return use.n_rows; }
  inline uword get_n_cols() const { return use.n_cols; }
  inline uword get_n_elem() const { return use.n_elem; }

  coot_inline dev_mem_t<elem_type> get_dev_mem(const bool synchronise = false) { return use.get_dev_mem(synchronise); }
  coot_inline uword                get_row_offset() const                      { return 0;                            }
  coot_inline uword                get_col_offset() const                      { return 0;                            }
  coot_inline uword                get_M_n_rows() const                        { return use.n_rows;                   }
  coot_inline uword                get_incr() const                            { return 1;                            }

  TDest& dest;
  TDest  aux;
  bool   using_aux;
  TDest& use;
  };



template<typename eT, typename... TArgs>
struct alias_wrapper<subview<eT>, TArgs...>
  {
  // dest: the output matrix to be used for the operation
  // arg: the input, which may be an alias of the output
  alias_wrapper(subview<eT>& dest_in, const TArgs&... args)
    : dest(dest_in)
    , using_aux(is_any_alias<subview<eT>>(dest).check(args...))
    {
    if (using_aux)
      {
      aux.set_size(dest.n_rows, dest.n_cols);
      }
    }

  ~alias_wrapper()
    {
    if (using_aux)
      {
      dest = aux; // a copy is needed unfortunately
      }
    }

  typedef eT          elem_type;
  typedef subview<eT> stored_type;

  inline uword get_n_rows() const { return using_aux ? aux.n_rows : dest.n_rows; }
  inline uword get_n_cols() const { return using_aux ? aux.n_cols : dest.n_cols; }
  inline uword get_n_elem() const { return using_aux ? aux.n_elem : dest.n_elem; }

  coot_inline dev_mem_t<eT> get_dev_mem(const bool synchronise = false) { return using_aux ? aux.get_dev_mem(synchronise) : dest.m.get_dev_mem(synchronise); }
  coot_inline uword         get_row_offset() const                      { return using_aux ? 0 : dest.aux_row1;                                            }
  coot_inline uword         get_col_offset() const                      { return using_aux ? 0 : dest.aux_col1;                                            }
  coot_inline uword         get_M_n_rows() const                        { return using_aux ? aux.n_rows : dest.m.n_rows;                                   }
  coot_inline uword         get_incr() const                            { return 1;                                                                        }

  subview<eT>& dest;
  Mat<eT>      aux;
  bool         using_aux;
  };



template<typename eT, typename... TArgs>
struct alias_wrapper<diagview<eT>, TArgs...>
  {
  // dest: the output matrix to be used for the operation
  // arg: the input, which may be an alias of the output
  alias_wrapper(diagview<eT>& dest_in, const TArgs&... args)
    : dest(dest_in)
    , using_aux(is_any_alias<diagview<eT>>(dest).check(args...))
    {
    if (using_aux)
      {
      aux.set_size(dest.n_rows, dest.n_cols);
      }
    }

  ~alias_wrapper()
    {
    if (using_aux)
      {
      dest = aux; // a copy is needed unfortunately
      }
    }

  typedef eT           elem_type;
  typedef diagview<eT> stored_type;

  inline uword get_n_rows() const { return using_aux ? aux.n_rows : dest.n_rows; }
  inline uword get_n_cols() const { return using_aux ? aux.n_cols : dest.n_cols; }
  inline uword get_n_elem() const { return using_aux ? aux.n_elem : dest.n_elem; }

  // diagviews are vecs
  coot_inline dev_mem_t<eT> get_dev_mem(const bool synchronise = false) { return using_aux ? aux.get_dev_mem(synchronise) : dest.m.get_dev_mem(synchronise); }
  coot_inline uword         get_row_offset() const                      { return using_aux ? 0 : dest.mem_offset;                                            }
  coot_inline uword         get_col_offset() const                      { return 0;                                                                          }
  coot_inline uword         get_M_n_rows() const                        { return using_aux ? aux.n_rows : dest.m.n_rows;                                     }
  coot_inline uword         get_incr() const                            { return using_aux ? 1 : dest.m.n_rows + 1;                                          }

  diagview<eT>& dest;
  Mat<eT>       aux;
  bool          using_aux;
  };
