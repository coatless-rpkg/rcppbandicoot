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



template<typename T1>
struct unwrap_cube
  {
  typedef typename T1::elem_type eT;
  typedef Cube<eT>               stored_type;

  inline
  unwrap_cube(const T1& A)
    : M(A)
    {
    coot_extra_debug_sigprint();
    }

  const Cube<eT> M;

  template<typename T2>
  constexpr bool is_alias(const T2& X) const { return coot::is_alias(X, M); }

  constexpr inline uword         get_row_offset()                    const { return 0; }
  constexpr inline uword         get_col_offset()                    const { return 0; }
  constexpr inline uword         get_slice_offset()                  const { return 0; }
            inline uword         get_M_n_rows()                      const { return M.n_rows; }
            inline uword         get_M_n_cols()                      const { return M.n_cols; }
            inline dev_mem_t<eT> get_dev_mem(const bool synchronise) const { return M.get_dev_mem(synchronise); }
  };



template<typename eT>
struct unwrap_cube< Cube<eT> >
  {
  typedef Cube<eT> stored_type;

  inline
  unwrap_cube(const Cube<eT>& A)
    : M(A)
    {
    coot_extra_debug_sigprint();
    }

  const Cube<eT>& M;

  template<typename T2>
  constexpr bool is_alias(const T2& X) const { return coot::is_alias(X, M); }

  constexpr inline uword         get_row_offset()                    const { return 0; }
  constexpr inline uword         get_col_offset()                    const { return 0; }
  constexpr inline uword         get_slice_offset()                  const { return 0; }
            inline uword         get_M_n_rows()                      const { return M.n_rows; }
            inline uword         get_M_n_cols()                      const { return M.n_cols; }
            inline dev_mem_t<eT> get_dev_mem(const bool synchronise) const { return M.get_dev_mem(synchronise); }
  };



template<typename eT>
struct unwrap_cube< subview_cube<eT> >
  {
  typedef subview_cube<eT> stored_type;

  inline
  unwrap_cube(const subview_cube<eT>& A)
    : M(A)
    {
    coot_extra_debug_sigprint();
    }

  const subview_cube<eT>& M;

  template<typename T2>
  constexpr bool is_alias(const T2& X) const { return coot::is_alias(X, M); }

  inline uword         get_row_offset()                    const { return M.aux_row1; }
  inline uword         get_col_offset()                    const { return M.aux_col1; }
  inline uword         get_slice_offset()                  const { return M.aux_slice1; }
  inline uword         get_M_n_rows()                      const { return M.m.n_rows; }
  inline uword         get_M_n_cols()                      const { return M.m.n_cols; }
  inline dev_mem_t<eT> get_dev_mem(const bool synchronise) const { return M.m.get_dev_mem(synchronise); }
  };


//
//
//



template<typename T1>
struct unwrap_cube_check
  {
  typedef typename T1::elem_type eT;

  inline
  unwrap_cube_check(const T1& A, const Cube<eT>&)
    : M(A)
    {
    coot_extra_debug_sigprint();

    coot_type_check(( is_coot_cube_type<T1>::value == false ));
    }

  inline
  unwrap_cube_check(const T1& A, const bool)
    : M(A)
    {
    coot_extra_debug_sigprint();

    coot_type_check(( is_coot_cube_type<T1>::value == false ));
    }

  const Cube<eT> M;
  };



template<typename eT>
struct unwrap_cube_check< Cube<eT> >
  {
  inline
  unwrap_cube_check(const Cube<eT>& A, const Cube<eT>& B)
    : M_local( (&A == &B) ? new Cube<eT>(A) : nullptr )
    , M      ( (&A == &B) ? (*M_local)      : A       )
    {
    coot_extra_debug_sigprint();
    }


  inline
  unwrap_cube_check(const Cube<eT>& A, const bool is_alias)
    : M_local( is_alias ? new Cube<eT>(A) : nullptr )
    , M      ( is_alias ? (*M_local)      : A       )
    {
    coot_extra_debug_sigprint();
    }


  inline
  ~unwrap_cube_check()
    {
    coot_extra_debug_sigprint();

    if(M_local)  { delete M_local; }
    }


  // the order below is important
  const Cube<eT>* M_local;
  const Cube<eT>& M;
  };
