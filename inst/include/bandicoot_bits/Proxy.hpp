// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Ryan Curtin (https://www.ratml.org)
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



//
// The base Proxy class will unwrap any expression into a type that we can use Proxy
// with (e.g. a matrix, vector, cube, or subview).
//
// Note that some memory handling care needs to be considered here: avoid non-trivial
// copies of Proxy objects (e.g. copies of return values from make_proxy() are fine);
// and ensure that no argument given to a Proxy will go out of scope before the Proxy
// does, because it may hold references.  These are generally not too onerous conditions to meet.
//

// If the type is a cube type, we have to use unwrap_cube.
template<typename T1, bool is_cube> struct Proxy_unwrap_type           { typedef unwrap<T1> type; };
template<typename T1>               struct Proxy_unwrap_type<T1, true> { typedef unwrap_cube<T1> type; };

template<typename T1>
struct Proxy
  {
  typedef typename Proxy_unwrap_type<T1, is_coot_cube_type<T1>::value>::type::stored_type held_type;
  typedef typename T1::elem_type elem_type;

  const typename Proxy_unwrap_type<T1, is_coot_cube_type<T1>::value>::type U;
  const Proxy<held_type> P;
  const held_type& Q;

  inline Proxy(const T1& in_Q)
    : U(in_Q)
    , P(U.M)
    , Q(U.M)
    {
    coot_type_check(( is_coot_type<T1>::value == false && is_coot_cube_type<T1>::value == false ));
    }

  static constexpr const size_t num_args = Proxy<held_type>::num_args;
  static constexpr const size_t num_dims = Proxy<held_type>::num_dims;

  typedef typename Proxy<held_type>::arg_types arg_types;

  inline arg_types args() const { return P.args(); }

  template<typename T2> inline bool         is_alias(const T2& t) const { return coot::is_alias(t, U.M);         }
  template<typename T2> inline bool is_inexact_alias(const T2& t) const { return coot::is_inexact_alias(t, U.M); }

  inline uword get_n_rows() const          { return P.get_n_rows();          }
  inline uword get_M_n_rows() const        { return P.get_M_n_rows();        }
  inline uword get_n_cols() const          { return P.get_n_cols();          }
  inline uword get_n_slices() const        { return P.get_n_slices();        }
  inline uword get_n_elem() const          { return P.get_n_elem();          }
  inline uword get_M_n_elem_slice() const  { return P.get_M_n_elem_slice();  }

  inline bool is_empty() const { return P.is_empty(); }
  };



template<typename eT>
struct Proxy< Mat<eT> >
  {
  typedef Mat<eT> held_type;
  typedef eT      elem_type;

  const Mat<eT>& Q;

  inline Proxy(const Mat<eT>& in_Q) : Q(in_Q) { }

  static constexpr const size_t num_args = 3;
  static constexpr const size_t num_dims = 2;

  typedef std::tuple< dev_mem_t<eT>&, const uword&, const uword& > arg_types;

  inline arg_types args() const
    {
    // can't use std::make_tuple() because we need the tuple to hold lvalue references
    return std::tie< dev_mem_t<eT>&, const uword& , const uword& >( access::rw(Q.dev_mem), Q.n_rows, Q.n_cols );
    }

  template<typename T2> inline bool         is_alias(const T2& t) const { return coot::is_alias(t, Q);         }
  template<typename T2> inline bool is_inexact_alias(const T2& t) const { return coot::is_inexact_alias(t, Q); }

  inline uword get_n_rows() const   { return Q.n_rows; }
  inline uword get_M_n_rows() const { return Q.n_rows; }
  inline uword get_n_cols() const   { return Q.n_cols; }
  inline uword get_n_slices() const { return 1;        }
  inline uword get_n_elem() const   { return Q.n_elem; }

  inline bool is_empty() const { return Q.is_empty(); }
  };



template<typename eT>
struct Proxy< Row<eT> >
  {
  typedef Row<eT> held_type;
  typedef eT      elem_type;

  const Row<eT>& Q;

  inline Proxy(const Row<eT>& in_Q) : Q(in_Q) { }

  static constexpr const size_t num_args = 2;
  static constexpr const size_t num_dims = 1;

  typedef std::tuple< dev_mem_t<eT>&, const uword& > arg_types;

  inline arg_types args() const
    {
    // can't use std::make_tuple() because we need the tuple to hold lvalue references
    return std::tie< dev_mem_t<eT>&, const uword& >( access::rw(Q.dev_mem), Q.n_elem );
    }

  template<typename T2> inline bool         is_alias(const T2& t) const { return coot::is_alias(t, Q);         }
  template<typename T2> inline bool is_inexact_alias(const T2& t) const { return coot::is_inexact_alias(t, Q); }

  inline uword get_n_rows() const   { return Q.n_rows; }
  inline uword get_M_n_rows() const { return Q.n_rows; }
  inline uword get_n_cols() const   { return Q.n_cols; }
  inline uword get_n_slices() const { return 1;        }
  inline uword get_n_elem() const   { return Q.n_elem; }

  inline bool is_empty() const { return Q.is_empty(); }
  };



// This can also be created from a Mat or a Cube, since all three hold contiguous memory
template<typename eT>
struct Proxy< Col<eT> >
  {
  typedef Col<eT> held_type;
  typedef eT      elem_type;

  const dev_mem_t<eT> mem;
  const uword         n_elem;

  inline Proxy(const Col<eT>& in_Q)  : mem(in_Q.get_dev_mem(false)), n_elem(in_Q.n_elem) { }
  // Allow a "fake" column vector proxy so we can treat Mats and Cubes as vectors
  inline Proxy(const Mat<eT>& in_Q)  : mem(in_Q.get_dev_mem(false)), n_elem(in_Q.n_elem) { }
  inline Proxy(const Cube<eT>& in_Q) : mem(in_Q.get_dev_mem(false)), n_elem(in_Q.n_elem) { }
  // Allow a fake alias proxy for MatValProxy.
  inline Proxy(const dev_mem_t<eT>& in_mem) : mem(in_mem), n_elem(1) { }

  static constexpr const size_t num_args = 2;
  static constexpr const size_t num_dims = 1;

  typedef std::tuple< dev_mem_t<eT>&, const uword& > arg_types;

  inline arg_types args() const
    {
    // can't use std::make_tuple() because we need the tuple to hold lvalue references
    return std::tie< dev_mem_t<eT>&, const uword& >( access::rw(mem), n_elem );
    }

  template<typename T2>
  inline bool is_alias(const T2& t) const
    {
    return mem_overlaps(alias_details<T2>::get_dev_mem(t),
                        alias_details<T2>::get_offset(t),
                        alias_details<T2>::get_n_elem(t),
                        mem,
                        0,
                        n_elem);
    }

  template<typename T2>
  inline bool is_inexact_alias(const T2& t) const
    {
    if (alias_details<T2>::get_dev_mem(t) == mem &&
        alias_details<T2>::get_offset(t) == 0 &&
        alias_details<T2>::get_n_elem(t) == n_elem)
      {
      return false;
      }

    return this->is_alias(t);
    }

  inline uword get_n_rows() const   { return n_elem; }
  inline uword get_M_n_rows() const { return n_elem; }
  inline uword get_n_cols() const   { return 1;      }
  inline uword get_n_slices() const { return 1;      }
  inline uword get_n_elem() const   { return n_elem; }

  inline bool is_empty() const { return (n_elem == 0); }
  };



template<typename eT>
struct Proxy< subview<eT> >
  {
  typedef subview<eT> held_type;
  typedef eT          elem_type;

  // we hold the elements manually, so that it is possible to make a "fake" subview Proxy for internal convenience

  const dev_mem_t<eT> offset_mem; // memory with subview offset applied
  const uword n_rows;
  const uword n_cols;
  const uword M_n_rows;

  inline Proxy(const subview<eT>& in_Q)
    : offset_mem(in_Q.m.dev_mem + in_Q.aux_row1 + in_Q.aux_col1 * in_Q.m.n_rows), n_rows(in_Q.n_rows), n_cols(in_Q.n_cols), M_n_rows(in_Q.m.n_rows) { }
  // special constructor for "fake" subviews
  inline Proxy(const dev_mem_t<eT> in_mem, const uword aux_row1, const uword aux_col1, const uword in_n_rows, const uword in_n_cols, const uword in_M_n_rows)
    : offset_mem(in_mem + aux_row1 + aux_col1 * in_M_n_rows), n_rows(in_n_rows), n_cols(in_n_cols), M_n_rows(in_M_n_rows) { }

  static constexpr const size_t num_args = 4;
  static constexpr const size_t num_dims = 2;

  typedef std::tuple< dev_mem_t<eT>&, const uword&, const uword&, const uword& > arg_types;

  inline arg_types args() const
    {
    return std::tie< dev_mem_t<eT>&, const uword& , const uword&, const uword& >( access::rw(offset_mem), n_rows, n_cols, M_n_rows );
    }

  template<typename T2>
  inline bool is_alias(const T2& t) const
    {
    return mem_overlaps(alias_details<T2>::get_dev_mem(t),
                        alias_details<T2>::get_offset(t),
                        alias_details<T2>::get_n_elem(t),
                        offset_mem,
                        0,
                        n_rows + n_cols * M_n_rows);
    }

  template<typename T2>
  inline bool is_inexact_alias(const T2& t) const
    {
    if (alias_details<T2>::get_dev_mem(t) == offset_mem &&
        alias_details<T2>::get_offset(t) == 0 &&
        alias_details<T2>::get_n_elem(t) == (n_rows + n_cols * M_n_rows))
      {
      return false;
      }

    return this->is_alias(t);
    }

  inline uword get_n_rows() const   { return n_rows;          }
  inline uword get_M_n_rows() const { return M_n_rows;        }
  inline uword get_n_cols() const   { return n_cols;          }
  inline uword get_n_slices() const { return 1;               }
  inline uword get_n_elem() const   { return n_rows * n_cols; }

  inline bool is_empty() const { return (n_rows == 0) || (n_cols == 0); }
  };



template<typename eT>
struct Proxy< subview_row<eT> >
  {
  typedef subview_row<eT> held_type;
  typedef eT              elem_type;

  const subview_row<eT>& Q;
  const dev_mem_t<eT>    offset_mem; // memory with subview offset applied

  inline Proxy(const subview_row<eT>& in_Q) : Q(in_Q), offset_mem(Q.m.dev_mem + Q.aux_row1 + Q.aux_col1 * Q.m.n_rows) { }

  static constexpr const size_t num_args = 3;
  static constexpr const size_t num_dims = 1;

  typedef std::tuple< dev_mem_t<eT>&, const uword&, const uword& > arg_types;

  inline arg_types args() const
    {
    return std::tie< dev_mem_t<eT>&, const uword&, const uword& >( access::rw(offset_mem), Q.n_elem, Q.m.n_rows );
    }

  template<typename T2>
  inline bool is_alias(const T2& t) const
    {
    return mem_overlaps(alias_details<T2>::get_dev_mem(t),
                        alias_details<T2>::get_offset(t),
                        alias_details<T2>::get_n_elem(t),
                        offset_mem,
                        0,
                        Q.n_elem * Q.m.n_rows);
    }

  template<typename T2>
  inline bool is_inexact_alias(const T2& t) const
    {
    if (alias_details<T2>::get_dev_mem(t) == offset_mem &&
        alias_details<T2>::get_offset(t) == 0 &&
        alias_details<T2>::get_n_elem(t) == (Q.n_elem * Q.m.n_rows))
      {
      return false;
      }

    return this->is_alias(t);
    }

  inline uword get_n_rows() const   { return Q.n_rows; }
  inline uword get_M_n_rows() const { return Q.n_rows; }
  inline uword get_n_cols() const   { return Q.n_cols; }
  inline uword get_n_slices() const { return 1;        }
  inline uword get_n_elem() const   { return Q.n_elem; }

  inline bool is_empty() const { return Q.is_empty(); }
  };



template<typename eT>
struct Proxy< subview_col<eT> >
  {
  typedef subview_col<eT> held_type;
  typedef eT              elem_type;

  const subview_col<eT>& Q;
  const dev_mem_t<eT>    offset_mem; // memory with subview offset applied

  inline Proxy(const subview_col<eT>& in_Q) : Q(in_Q), offset_mem(Q.m.dev_mem + Q.aux_row1 + Q.aux_col1 * Q.m.n_rows) { }

  static constexpr const size_t num_args = 2;
  static constexpr const size_t num_dims = 1;

  typedef std::tuple< dev_mem_t<eT>&, const uword& > arg_types;

  inline arg_types args() const
    {
    return std::tie< dev_mem_t<eT>&, const uword& >( access::rw(offset_mem), Q.n_elem );
    }

  template<typename T2>
  inline bool is_alias(const T2& t) const
    {
    return mem_overlaps(alias_details<T2>::get_dev_mem(t),
                        alias_details<T2>::get_offset(t),
                        alias_details<T2>::get_n_elem(t),
                        offset_mem,
                        0,
                        Q.n_elem);
    }

  template<typename T2>
  inline bool is_inexact_alias(const T2& t) const
    {
    if (alias_details<T2>::get_dev_mem(t) == offset_mem &&
        alias_details<T2>::get_offset(t) == 0 &&
        alias_details<T2>::get_n_elem(t) == Q.n_elem)
      {
      return false;
      }

    return this->is_alias(t);
    }

  inline uword get_n_rows() const   { return Q.n_rows; }
  inline uword get_M_n_rows() const { return Q.n_rows; }
  inline uword get_n_cols() const   { return Q.n_cols; }
  inline uword get_n_slices() const { return 1;        }
  inline uword get_n_elem() const   { return Q.n_elem; }

  inline bool is_empty() const { return Q.is_empty(); }
  };



template<typename eT>
struct Proxy< diagview<eT> >
  {
  typedef diagview<eT> held_type;
  typedef eT           elem_type;

  const diagview<eT>& Q;
  const dev_mem_t<eT> offset_mem;
  const uword         mem_incr;

  inline Proxy(const diagview<eT>& in_Q) : Q(in_Q), offset_mem(Q.m.dev_mem + Q.mem_offset), mem_incr(Q.m.n_rows + 1) { }

  static constexpr const size_t num_args = 3;
  static constexpr const size_t num_dims = 1;

  typedef std::tuple< dev_mem_t<eT>&, const uword&, const uword& > arg_types;

  inline arg_types args() const
    {
    return std::tie< dev_mem_t<eT>&, const uword&, const uword&>( access::rw(offset_mem), Q.n_elem, mem_incr );
    }

  template<typename T2>
  inline bool is_alias(const T2& t) const
    {
    return mem_overlaps(alias_details<T2>::get_dev_mem(t),
                        alias_details<T2>::get_offset(t),
                        alias_details<T2>::get_n_elem(t),
                        offset_mem,
                        0,
                        Q.n_elem * mem_incr);
    }

  template<typename T2>
  inline bool is_inexact_alias(const T2& t) const
    {
    if (alias_details<T2>::get_dev_mem(t) == offset_mem &&
        alias_details<T2>::get_offset(t) == 0 &&
        alias_details<T2>::get_n_elem(t) == (Q.n_elem * mem_incr))
      {
      return false;
      }

    return this->is_alias(t);
    }

  inline uword get_n_rows() const   { return Q.n_rows; }
  inline uword get_M_n_rows() const { return Q.n_rows; }
  inline uword get_n_cols() const   { return Q.n_cols; }
  inline uword get_n_slices() const { return 1;        }
  inline uword get_n_elem() const   { return Q.n_elem; }

  inline bool is_empty() const { return Q.is_empty(); }
  };



// temporary max() shim until max() supports Proxy arguments
// TODO: remove
template<typename T1>
inline
typename T1::elem_type
proxy_max_shim(const Proxy<T1>& P)
  {
  return P.Q.max();
  }



template<typename eT>
inline
eT
proxy_max_shim(const Proxy< Col<eT> >& P)
  {
  return coot_rt_t::max_vec(P.mem, P.get_n_elem());
  }



template<typename eT>
inline
eT
proxy_max_shim(const Proxy< subview<eT> >& P)
  {
  // ugly: extract into a column vector...
  Col<eT> tmp(P.get_n_elem());
  coot_rt_t::copy(make_proxy(tmp), P);
  return tmp.max();
  }



template<typename T1>
inline
typename T1::elem_type
proxy_max_shim(const Proxy< ProxyColCast< T1 > >& P)
  {
  return proxy_max_shim(P.P);
  }



template<typename T1>
inline
typename T1::elem_type
proxy_max_shim(const Proxy< Op< T1, op_vectorise_col > >& P)
  {
  return proxy_max_shim(P.P);
  }



template<typename T1>
inline
typename T1::elem_type
proxy_max_shim(const Proxy< eOp<T1, eop_scalar_plus> >& P)
  {
  return proxy_max_shim(P.P) + P.Q.aux_a;
  }



template<typename eT, typename T1>
struct Proxy< subview_elem1< eT, T1 > >
  {
  typedef subview_elem1<eT, typename proxy_col_type<T1>::type::held_type> held_type;
  typedef eT                                                   elem_type;

  const subview_elem1<eT, T1>&            Q;
  const typename proxy_col_type<T1>::type P;

  inline Proxy(const subview_elem1<eT, T1>& in_Q) : Q(in_Q), P(Q.a.get_ref())
    {
    // Perform bounds checks.  This is the only way we can give a user an error because we can't check this on the GPU.
    coot_conform_check_bounds( proxy_max_shim(P) >= Q.m.n_elem, "Mat::elem(): index out of bounds" );
    }

  static constexpr const size_t num_args = 1 + proxy_col_type<T1>::type::num_args;
  static constexpr const size_t num_dims = 1;

  typedef typename merge_tuple< std::tuple< dev_mem_t<eT>& >, typename proxy_col_type<T1>::type::arg_types >::result arg_types;

  inline arg_types args() const
    {
    return std::tuple_cat( std::tie< dev_mem_t<eT>& >( access::rw(Q.m.dev_mem) ), P.args() );
    }

  // We have to check both the main object and the indices in the T1.
  template<typename T2> inline bool         is_alias(const T2& t) const { return coot::is_alias(t, Q.m) || P.is_alias(t);                 }
  template<typename T2> inline bool is_inexact_alias(const T2& t) const { return coot::is_inexact_alias(t, Q.m) || P.is_inexact_alias(t); }

  inline uword get_n_rows() const   { return P.get_n_elem(); }
  inline uword get_M_n_rows() const { return P.get_n_elem(); }
  inline uword get_n_cols() const   { return 1;              }
  inline uword get_n_slices() const { return 1;              }
  inline uword get_n_elem() const   { return P.get_n_elem(); }

  inline bool is_empty() const { return Q.is_empty(); }
  };



template<typename eT, typename T1, typename T2>
struct Proxy< subview_elem2< eT, subview_elem2_both<eT, T1, T2> > >
  {
  typedef subview_elem2<eT, subview_elem2_both<eT, typename proxy_col_type<T1>::type::held_type, typename proxy_col_type<T2>::type::held_type>> held_type;
  typedef eT                                                                                                                                    elem_type;

  const subview_elem2<eT, subview_elem2_both<eT, T1, T2>>& Q;
  const typename proxy_col_type<T1>::type                  P1;
  const typename proxy_col_type<T2>::type                  P2;
  const uword out_n_rows;
  const uword out_n_cols;

  inline Proxy(const subview_elem2<eT, subview_elem2_both<eT, T1, T2>>& in_Q)
    : Q(in_Q)
    , P1(Q.r.base_ri.get_ref())
    , P2(Q.r.base_ci.get_ref())
    , out_n_rows(P1.get_n_elem())
    , out_n_cols(P2.get_n_elem())
    {
    // Perform bounds checks.  This is the only way we can give a user an error because we can't check this on the GPU.
    coot_conform_check( P1.get_n_rows() != 1 && P1.get_n_cols() != 1, "Mat::elem(): row indices must be a vector" );
    coot_conform_check( P2.get_n_rows() != 1 && P2.get_n_cols() != 1, "Mat::elem(): row indices must be a vector" );

    coot_conform_check_bounds( proxy_max_shim(P1) >= Q.m.n_rows, "Mat::elem(): row index out of bounds" );
    coot_conform_check_bounds( proxy_max_shim(P2) >= Q.m.n_cols, "Mat::elem(): column index out of bounds" );
    }

  static constexpr const size_t num_args = 5 + proxy_col_type<T1>::type::num_args + proxy_col_type<T2>::type::num_args;
  static constexpr const size_t num_dims = 2;

  typedef typename merge_tuple
    <
    std::tuple< dev_mem_t<eT>&, const uword&, const uword&, const uword&, const uword& >,
    typename proxy_col_type<T1>::type::arg_types,
    typename proxy_col_type<T2>::type::arg_types
    >::result arg_types;

  inline arg_types args() const
    {
    return std::tuple_cat(
        std::tie< dev_mem_t<eT>&, const uword&, const uword&, const uword&, const uword& >( access::rw(Q.m.dev_mem), out_n_rows, out_n_cols, Q.m.n_rows, Q.m.n_cols ),
        P1.args(),
        P2.args());
    }

  // We have to check both the main object and the indices in the T1.
  template<typename T3> inline bool         is_alias(const T3& t) const { return coot::is_alias(t, Q.m) || P1.is_alias(t) || P2.is_alias(t);                         }
  template<typename T3> inline bool is_inexact_alias(const T3& t) const { return coot::is_inexact_alias(t, Q.m) || P1.is_inexact_alias(t) || P2.is_inexact_alias(t); }

  inline uword get_n_rows() const   { return out_n_rows;              }
  inline uword get_M_n_rows() const { return out_n_rows;              }
  inline uword get_n_cols() const   { return out_n_cols;              }
  inline uword get_n_slices() const { return 1;                       }
  inline uword get_n_elem() const   { return out_n_rows * out_n_cols; }

  inline bool is_empty() const { return Q.is_empty(); }
  };



template<typename eT, typename T1>
struct Proxy< subview_elem2< eT, subview_elem2_all_cols<eT, T1> > >
  {
  typedef subview_elem2<eT, subview_elem2_all_cols<eT, typename proxy_col_type<T1>::type::held_type>> held_type;
  typedef eT                                                                                          elem_type;

  const subview_elem2<eT, subview_elem2_all_cols<eT, T1>>& Q;
  const typename proxy_col_type<T1>::type                  P1;
  const uword out_n_rows;

  inline Proxy(const subview_elem2<eT, subview_elem2_all_cols<eT, T1>>& in_Q)
    : Q(in_Q)
    , P1(Q.r.base_ri.get_ref())
    , out_n_rows(P1.get_n_elem())
    {
    // Perform bounds checks.  This is the only way we can give a user an error because we can't check this on the GPU.
    coot_conform_check( P1.get_n_rows() != 1 && P1.get_n_cols() != 1, "Mat::cols(): row indices must be a vector" );
    coot_conform_check_bounds( proxy_max_shim(P1) >= Q.m.n_rows, "Mat::cols(): row index out of bounds" );
    }

  static constexpr const size_t num_args = 4 + proxy_col_type<T1>::type::num_args;
  static constexpr const size_t num_dims = 2;

  typedef typename merge_tuple
    <
    std::tuple< dev_mem_t<eT>&, const uword&, const uword&, const uword& >,
    typename proxy_col_type<T1>::type::arg_types
    >::result arg_types;

  inline arg_types args() const
    {
    return std::tuple_cat(
        std::tie< dev_mem_t<eT>&, const uword&, const uword&, const uword& >( access::rw(Q.m.dev_mem), out_n_rows, Q.m.n_cols, Q.m.n_rows ),
        P1.args());
    }

  // We have to check both the main object and the indices in the T1.
  template<typename T3> inline bool         is_alias(const T3& t) const { return coot::is_alias(t, Q.m) || P1.is_alias(t);                   }
  template<typename T3> inline bool is_inexact_alias(const T3& t) const { return coot::is_inexact_alias(t, Q.m) || P1.is_inexact_alias(t); }

  inline uword get_n_rows() const   { return out_n_rows;              }
  inline uword get_M_n_rows() const { return out_n_rows;              }
  inline uword get_n_cols() const   { return Q.m.n_cols;              }
  inline uword get_n_slices() const { return 1;                       }
  inline uword get_n_elem() const   { return out_n_rows * Q.m.n_cols; }

  inline bool is_empty() const { return Q.is_empty(); }
  };



template<typename eT, typename T2>
struct Proxy< subview_elem2< eT, subview_elem2_all_rows<eT, T2> > >
  {
  typedef subview_elem2<eT, subview_elem2_all_rows<eT, typename proxy_col_type<T2>::type::held_type>> held_type;
  typedef eT                                                                                          elem_type;

  const subview_elem2<eT, subview_elem2_all_rows<eT, T2>>& Q;
  const typename proxy_col_type<T2>::type                  P2;
  const uword out_n_cols;

  inline Proxy(const subview_elem2<eT, subview_elem2_all_rows<eT, T2>>& in_Q)
    : Q(in_Q)
    , P2(Q.r.base_ci.get_ref())
    , out_n_cols(P2.get_n_elem())
    {
    // Perform bounds checks.  This is the only way we can give a user an error because we can't check this on the GPU.
    coot_conform_check( P2.get_n_rows() != 1 && P2.get_n_cols() != 1, "Mat::rows(): column indices must be a vector" );
    coot_conform_check_bounds( proxy_max_shim(P2) >= Q.m.n_cols, "Mat::rows(): column index out of bounds" );
    }

  static constexpr const size_t num_args = 3 + proxy_col_type<T2>::type::num_args;
  static constexpr const size_t num_dims = 2;

  typedef typename merge_tuple
    <
    std::tuple< dev_mem_t<eT>&, const uword&, const uword& >,
    typename proxy_col_type<T2>::type::arg_types
    >::result arg_types;

  inline arg_types args() const
    {
    return std::tuple_cat(
        std::tie< dev_mem_t<eT>&, const uword&, const uword& >( access::rw(Q.m.dev_mem), Q.m.n_rows, out_n_cols ),
        P2.args());
    }

  // We have to check both the main object and the indices in the T2.
  template<typename T3> inline bool         is_alias(const T3& t) const { return coot::is_alias(t, Q.m) || P2.is_alias(t);                   }
  template<typename T3> inline bool is_inexact_alias(const T3& t) const { return coot::is_inexact_alias(t, Q.m) || P2.is_inexact_alias(t); }

  inline uword get_n_rows() const   { return Q.m.n_rows;              }
  inline uword get_M_n_rows() const { return Q.m.n_rows;              }
  inline uword get_n_cols() const   { return out_n_cols;              }
  inline uword get_n_slices() const { return 1;                       }
  inline uword get_n_elem() const   { return Q.m.n_rows * out_n_cols; }

  inline bool is_empty() const { return Q.is_empty(); }
  };



template<typename eT>
struct Proxy< Cube<eT> >
  {
  typedef Cube<eT> held_type;
  typedef eT       elem_type;

  const Cube<eT>& Q;

  inline Proxy(const Cube<eT>& in_Q) : Q(in_Q) { }

  static constexpr const size_t num_args = 4;
  static constexpr const size_t num_dims = 3;

  typedef std::tuple< dev_mem_t<eT>&, const uword&, const uword&, const uword& > arg_types;

  inline arg_types args() const
    {
    return std::tie< dev_mem_t<eT>&, const uword&, const uword&, const uword& >( access::rw(Q.dev_mem), Q.n_rows, Q.n_cols, Q.n_slices );
    }

  template<typename T2> inline bool         is_alias(const T2& t) const { return coot::is_alias(t, Q);         }
  template<typename T2> inline bool is_inexact_alias(const T2& t) const { return coot::is_inexact_alias(t, Q); }

  inline uword get_n_rows() const          { return Q.n_rows;       }
  inline uword get_M_n_rows() const        { return Q.n_rows;       }
  inline uword get_n_cols() const          { return Q.n_cols;       }
  inline uword get_n_slices() const        { return Q.n_slices;     }
  inline uword get_n_elem() const          { return Q.n_elem;       }
  inline uword get_M_n_elem_slice() const  { return Q.n_elem_slice; }

  inline bool is_empty() const { return Q.is_empty(); }
  };



template<typename eT>
struct Proxy< subview_cube<eT> >
  {
  typedef subview_cube<eT> held_type;
  typedef eT               elem_type;

  const subview_cube<eT>& Q;
  const dev_mem_t<eT>     offset_mem;

  inline Proxy(const subview_cube<eT>& in_Q) : Q(in_Q), offset_mem(Q.m.dev_mem + Q.aux_row1 + Q.aux_col1 * Q.m.n_rows + Q.aux_slice1 * Q.m.n_elem_slice) { }

  static constexpr const size_t num_args = 6;
  static constexpr const size_t num_dims = 3;

  typedef std::tuple< dev_mem_t<eT>&, const uword&, const uword&, const uword&, const uword&, const uword& > arg_types;

  inline arg_types args() const
    {
    return std::tie< dev_mem_t<eT>&, const uword&, const uword&, const uword&, const uword&, const uword& >(
        access::rw(offset_mem), Q.n_rows, Q.n_cols, Q.n_slices, Q.m.n_rows, Q.m.n_elem_slice);
    }

  template<typename T2>
  inline bool is_alias(const T2& t) const
    {
    return mem_overlaps(alias_details<T2>::get_dev_mem(t),
                        alias_details<T2>::get_offset(t),
                        alias_details<T2>::get_n_elem(t),
                        offset_mem,
                        0,
                        Q.m.n_elem_slice * (Q.n_slices - 1) + Q.n_elem_slice);
    }

  template<typename T2>
  inline bool is_inexact_alias(const T2& t) const
    {
    if (alias_details<T2>::get_dev_mem(t) == offset_mem &&
        alias_details<T2>::get_offset(t) == 0 &&
        alias_details<T2>::get_n_elem(t) == (Q.m.n_elem_slice * (Q.n_slices - 1) + Q.n_elem_slice))
      {
      return false;
      }

    return this->is_alias(t);
    }

  inline uword get_n_rows() const          { return Q.n_rows;         }
  inline uword get_M_n_rows() const        { return Q.m.n_rows;       }
  inline uword get_n_cols() const          { return Q.n_cols;         }
  inline uword get_n_slices() const        { return Q.n_slices;       }
  inline uword get_n_elem() const          { return Q.n_elem;         }
  inline uword get_M_n_elem_slice() const  { return Q.m.n_elem_slice; }

  inline bool is_empty() const { return Q.is_empty(); }
  };



// shim proxy to operate on a single value of a matrix
template<typename eT>
struct Proxy< MatValProxy<eT> >
  {
  typedef Col<eT> held_type; // generated kernels will use Col types
  typedef eT      elem_type;

  const dev_mem_t<eT> mem;
  const uword         n_elem; // always 1

  inline Proxy(const MatValProxy<eT>& P) : mem(P.dev_mem + P.index), n_elem(1) { }

  static constexpr const size_t num_args = 2;
  static constexpr const size_t num_dims = 1;

  typedef std::tuple< dev_mem_t<eT>&, const uword& > arg_types;

  inline arg_types args() const { return std::tie< dev_mem_t<eT>&, const uword& >( access::rw(mem), n_elem ); }

  inline const dev_mem_t<eT>& get_dev_mem() const { return mem; }

  inline constexpr uword get_n_rows() const   { return 1; }
  inline constexpr uword get_M_n_rows() const { return 1; }
  inline constexpr uword get_n_cols() const   { return 1; }
  inline constexpr uword get_n_slices() const { return 1; }
  inline constexpr uword get_n_elem() const   { return 1; }

  inline constexpr bool is_empty() const { return false; }
  };



// shim proxy to perform an elementwise operation on a single value of a matrix
template<typename eT, typename eop_type>
struct Proxy< eOp<MatValProxy<eT>, eop_type> >
  {
  typedef eOp<Col<eT>, eop_type> held_type; // generated kernels will use Col types
  typedef eT                     elem_type;

  const dev_mem_t<eT> mem;
  const uword         n_elem; // always 1
  const eT            val;

  inline Proxy(const MatValProxy<eT>& P, const eT val) : mem(P.dev_mem + P.index), n_elem(1), val(val) { }

  static constexpr const size_t num_args = 3;
  static constexpr const size_t num_dims = 1;

  typedef std::tuple< dev_mem_t<eT>&, const uword&, const eT& > arg_types;

  inline arg_types args() const { return std::tie< dev_mem_t<eT>&, const uword&, const eT& >( access::rw(mem), n_elem, val ); }

  inline const dev_mem_t<eT>& get_dev_mem() const { return mem; }

  inline constexpr uword get_n_rows() const   { return 1; }
  inline constexpr uword get_M_n_rows() const { return 1; }
  inline constexpr uword get_n_cols() const   { return 1; }
  inline constexpr uword get_n_slices() const { return 1; }
  inline constexpr uword get_n_elem() const   { return 1; }

  inline constexpr bool is_empty() const { return false; }
  };



//
// Proxy shims for casts to other dimensions.
//
// The has_ref template parameter controls whether the internally-held proxy
// is a reference or an object.  It should be set to `true` when shimming an
// already-existing Proxy.
//

template<typename T1, size_t src_dims>
struct Proxy< ProxyColCast<T1, src_dims, false> >
  {
  typedef ProxyColCast<typename Proxy<T1>::held_type, src_dims, false> held_type;
  typedef typename Proxy<T1>::elem_type                                elem_type;

  const Proxy<T1> P;

  inline Proxy(const T1& in_Q) : P(in_Q) { }

  // no extra arguments necessary to reinterpret something as a vector
  static constexpr const size_t num_args = Proxy<T1>::num_args;
  static constexpr const size_t num_dims = 1;

  typedef typename Proxy<T1>::arg_types arg_types;

  inline arg_types args() const { return P.args(); }

  template<typename T2> inline bool         is_alias(const T2& t) const { return P.is_alias(t);         }
  template<typename T2> inline bool is_inexact_alias(const T2& t) const { return P.is_inexact_alias(t); }

  inline uword get_n_rows() const   { return P.get_n_elem(); }
  inline uword get_M_n_rows() const { return P.get_n_elem(); }
  inline uword get_n_cols() const   { return 1;              }
  inline uword get_n_slices() const { return 1;              }
  inline uword get_n_elem() const   { return P.get_n_elem(); }

  inline bool is_empty() const { return P.is_empty(); }
  };



template<typename T1, size_t src_dims>
struct Proxy< ProxyColCast<T1, src_dims, true> >
  {
  typedef ProxyColCast<typename Proxy<T1>::held_type, src_dims, true> held_type;
  typedef typename Proxy<T1>::elem_type                               elem_type;

  const Proxy<T1>& P;

  inline Proxy(const Proxy<T1>& in_P) : P(in_P) { }

  // no extra arguments necessary to reinterpret something as a vector
  static constexpr const size_t num_args = Proxy<T1>::num_args;
  static constexpr const size_t num_dims = 1;

  typedef typename Proxy<T1>::arg_types arg_types;

  inline arg_types args() const { return P.args(); }

  template<typename T2> inline bool         is_alias(const T2& t) const { return P.is_alias(t);         }
  template<typename T2> inline bool is_inexact_alias(const T2& t) const { return P.is_inexact_alias(t); }

  inline uword get_n_rows() const   { return P.get_n_elem(); }
  inline uword get_M_n_rows() const       { return P.get_n_elem(); }
  inline uword get_n_cols() const   { return 1; }
  inline uword get_n_slices() const { return 1; }
  inline uword get_n_elem() const   { return P.get_n_elem(); }

  inline bool is_empty() const { return P.is_empty(); }
  };



template<typename T1, size_t src_dims>
struct Proxy< ProxyMatCast<T1, src_dims, false> >
  {
  typedef ProxyMatCast<typename Proxy<T1>::held_type, src_dims, false> held_type;
  typedef typename Proxy<T1>::elem_type                                elem_type;

  const Proxy<T1> P;
  const uword     n_rows;
  const uword     n_cols;

  inline Proxy(const T1& in_Q, const uword in_n_rows, const uword in_n_cols) : P(in_Q), n_rows(in_n_rows),      n_cols(in_n_cols)      { }
  inline Proxy(const T1& in_Q)                                               : P(in_Q), n_rows(P.get_n_rows()), n_cols(P.get_n_cols()) { }

  // two extra arguments necessary to reinterpret something as a matrix
  static constexpr const size_t num_args = Proxy<T1>::num_args + 2;
  static constexpr const size_t num_dims = 2;

  typedef typename merge_tuple< typename Proxy<T1>::arg_types, std::tuple< const uword&, const uword& > >::result arg_types;

  inline arg_types args() const { return std::tuple_cat( P.args(), std::tie< const uword&, const uword& >(n_rows, n_cols) ); }

  template<typename T2> inline bool         is_alias(const T2& t) const { return P.is_alias(t);         }
  template<typename T2> inline bool is_inexact_alias(const T2& t) const { return P.is_inexact_alias(t); }

  inline uword get_n_rows() const   { return n_rows; }
  inline uword get_M_n_rows() const { return n_rows; }
  inline uword get_n_cols() const   { return n_cols; }
  inline uword get_n_slices() const { return 1; }
  inline uword get_n_elem() const   { return n_rows * n_cols; }

  inline bool is_empty() const { return P.is_empty(); }
  };



template<typename T1, size_t src_dims>
struct Proxy< ProxyMatCast<T1, src_dims, true> >
  {
  typedef ProxyMatCast<typename Proxy<T1>::held_type, src_dims, true> held_type;
  typedef typename Proxy<T1>::elem_type                               elem_type;

  const Proxy<T1>& P;
  const uword      n_rows;
  const uword      n_cols;

  // only one constructor type is enabled depending on proxy_uses_ref
  inline Proxy(const Proxy<T1>& in_P, const uword in_n_rows, const uword in_n_cols) : P(in_P), n_rows(in_n_rows),      n_cols(in_n_cols)      { }
  inline Proxy(const Proxy<T1>& in_P)                                               : P(in_P), n_rows(P.get_n_rows()), n_cols(P.get_n_cols()) { }

  // two extra arguments necessary to reinterpret something as a matrix
  static constexpr const size_t num_args = Proxy<T1>::num_args + 2;
  static constexpr const size_t num_dims = 2;

  typedef typename merge_tuple< typename Proxy<T1>::arg_types, std::tuple< const uword&, const uword& > >::result arg_types;

  inline arg_types args() const { return std::tuple_cat( P.args(), std::tie< const uword&, const uword& >(n_rows, n_cols) ); }

  template<typename T2> inline bool         is_alias(const T2& t) const { return P.is_alias(t);         }
  template<typename T2> inline bool is_inexact_alias(const T2& t) const { return P.is_inexact_alias(t); }

  inline uword get_n_rows() const   { return n_rows; }
  inline uword get_M_n_rows() const       { return n_rows; }
  inline uword get_n_cols() const   { return n_cols; }
  inline uword get_n_slices() const { return 1; }
  inline uword get_n_elem() const   { return n_rows * n_cols; }

  inline bool is_empty() const { return P.is_empty(); }
  };



template<typename T1, size_t src_dims>
struct Proxy< ProxyCubeCast<T1, src_dims, false> >
  {
  typedef ProxyCubeCast<typename Proxy<T1>::held_type, src_dims, false> held_type;
  typedef typename Proxy<T1>::elem_type                                 elem_type;

  const Proxy<T1> P;
  const uword     n_rows;
  const uword     n_cols;
  const uword     n_slices;

  inline Proxy(const T1& in_Q, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices) : P(in_Q), n_rows(in_n_rows),      n_cols(in_n_cols),      n_slices(in_n_slices)      { }
  inline Proxy(const T1& in_Q)                                                                        : P(in_Q), n_rows(P.get_n_rows()), n_cols(P.get_n_cols()), n_slices(P.get_n_slices()) { }

  // two extra arguments necessary to reinterpret something as a matrix
  static constexpr const size_t num_args = Proxy<T1>::num_args + 3;
  static constexpr const size_t num_dims = 3;

  typedef typename merge_tuple< typename Proxy<T1>::arg_types, std::tuple< const uword&, const uword&, const uword& > >::result arg_types;

  inline arg_types args() const { return std::tuple_cat( P.args(), std::tie< const uword&, const uword&, const uword& >(n_rows, n_cols, n_slices) ); }

  template<typename T2> inline bool         is_alias(const T2& t) const { return P.is_alias(t);         }
  template<typename T2> inline bool is_inexact_alias(const T2& t) const { return P.is_inexact_alias(t); }

  inline uword get_n_rows() const         { return n_rows; }
  inline uword get_M_n_rows() const       { return n_rows; }
  inline uword get_n_cols() const         { return n_cols; }
  inline uword get_n_slices() const       { return n_slices; }
  inline uword get_n_elem() const         { return n_rows * n_cols * n_slices; }
  inline uword get_M_n_elem_slice() const { return n_rows * n_cols; }

  inline bool is_empty() const { return P.is_empty(); }
  };



template<typename T1, size_t src_dims>
struct Proxy< ProxyCubeCast<T1, src_dims, true> >
  {
  typedef ProxyCubeCast<typename Proxy<T1>::held_type, src_dims, true> held_type;
  typedef typename Proxy<T1>::elem_type                                elem_type;

  const Proxy<T1>& P;
  const uword      n_rows;
  const uword      n_cols;
  const uword      n_slices;

  inline Proxy(const Proxy<T1>& in_P, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices) : P(in_P), n_rows(in_n_rows),      n_cols(in_n_cols),      n_slices(in_n_slices)      { }
  inline Proxy(const Proxy<T1>& in_P)                                                                        : P(in_P), n_rows(P.get_n_rows()), n_cols(P.get_n_cols()), n_slices(P.get_n_slices()) { }

  // two extra arguments necessary to reinterpret something as a matrix
  static constexpr const size_t num_args = Proxy<T1>::num_args + 3;
  static constexpr const size_t num_dims = 3;

  typedef typename merge_tuple< typename Proxy<T1>::arg_types, std::tuple< const uword&, const uword&, const uword& > >::result arg_types;

  inline arg_types args() const { return std::tuple_cat( P.args(), std::tie< const uword&, const uword&, const uword& >(n_rows, n_cols, n_slices) ); }

  template<typename T2> inline bool         is_alias(const T2& t) const { return P.is_alias(t);         }
  template<typename T2> inline bool is_inexact_alias(const T2& t) const { return P.is_inexact_alias(t); }

  inline uword get_n_rows() const          { return n_rows; }
  inline uword get_M_n_rows() const        { return n_rows; }
  inline uword get_n_cols() const          { return n_cols; }
  inline uword get_n_slices() const        { return n_slices; }
  inline uword get_n_elem() const          { return n_rows * n_cols * n_slices; }
  inline uword get_M_n_elem_slice() const  { return n_rows * n_cols; }

  inline bool is_empty() const { return P.is_empty(); }
  };



template<typename eT, size_t num_args>
struct Proxy_eop_arg_tuple_type
  {
  typedef std::tuple<> type;

  template<typename T1, typename eop_type>
  static std::tuple<> args(const eOp<T1, eop_type>& Q) { return std::tuple<>(); }

  template<typename T1, typename eop_type>
  static std::tuple<> args(const eOpCube<T1, eop_type>& Q) { return std::tuple<>(); }
  };

template<typename eT>
struct Proxy_eop_arg_tuple_type<eT, 1>
  {
  typedef std::tuple< const eT& > type;

  template<typename T1, typename eop_type>
  static std::tuple< const eT& > args(const eOp<T1, eop_type>& Q) { return std::tie<const eT&>(Q.aux_a); }

  template<typename T1, typename eop_type>
  static std::tuple< const eT& > args(const eOpCube<T1, eop_type>& Q) { return std::tie<const eT&>(Q.aux_a); }
  };

template<typename eT>
struct Proxy_eop_arg_tuple_type<eT, 2>
  {
  typedef std::tuple< const eT&, const eT& > type;

  template<typename T1, typename eop_type>
  static std::tuple< const eT&, const eT& > args(const eOp<T1, eop_type>& Q) { return std::tie<const eT&>(Q.aux_a, Q.aux_b); }

  template<typename T1, typename eop_type>
  static std::tuple< const eT&, const eT& > args(const eOpCube<T1, eop_type>& Q) { return std::tie<const eT&>(Q.aux_a, Q.aux_b); }
  };



template<typename T1, typename eop_type>
struct Proxy< eOp<T1, eop_type> >
  {
  typedef eOp<typename Proxy<T1>::held_type, eop_type> held_type;
  typedef typename Proxy<T1>::elem_type                elem_type;

  const Proxy<T1> P;
  const eOp<T1, eop_type>& Q;

  inline Proxy(const eOp<T1, eop_type>& in_Q) : P(in_Q.m.Q), Q(in_Q) { }

  // up to two extra arguments depending on the eop
  static constexpr const size_t num_args = Proxy<T1>::num_args + eop_type::num_args;
  static constexpr const size_t num_dims = Proxy<T1>::num_dims;

  typedef typename merge_tuple< typename Proxy<T1>::arg_types, typename Proxy_eop_arg_tuple_type<elem_type, eop_type::num_args>::type >::result arg_types;

  inline arg_types args() const { return std::tuple_cat( P.args(), Proxy_eop_arg_tuple_type<elem_type, eop_type::num_args>::args(Q) ); }

  template<typename T2> inline bool         is_alias(const T2& t) const { return P.is_alias(t);         }
  template<typename T2> inline bool is_inexact_alias(const T2& t) const { return P.is_inexact_alias(t); }

  inline uword get_n_rows() const   { return P.get_n_rows();   }
  inline uword get_M_n_rows() const { return P.get_M_n_rows(); }
  inline uword get_n_cols() const   { return P.get_n_cols();   }
  inline uword get_n_slices() const { return P.get_n_slices(); }
  inline uword get_n_elem() const   { return P.get_n_elem();   }

  inline bool is_empty() const { return P.is_empty(); }
  };



template<typename T1, typename eop_type>
struct Proxy< eOpCube<T1, eop_type> >
  {
  typedef eOpCube<typename Proxy<T1>::held_type, eop_type> held_type;
  typedef typename Proxy<T1>::elem_type                    elem_type;

  const Proxy<T1> P;
  const eOpCube<T1, eop_type>& Q;

  inline Proxy(const eOpCube<T1, eop_type>& in_Q) : P(in_Q.m.Q), Q(in_Q) { }

  // up to two extra arguments depending on the eop
  static constexpr const size_t num_args = Proxy<T1>::num_args + eop_type::num_args;
  static constexpr const size_t num_dims = Proxy<T1>::num_dims;

  typedef typename merge_tuple< typename Proxy<T1>::arg_types, typename Proxy_eop_arg_tuple_type<elem_type, eop_type::num_args>::type >::result arg_types;

  inline arg_types args() const { return std::tuple_cat( P.args(), Proxy_eop_arg_tuple_type<elem_type, eop_type::num_args>::args(Q) ); }

  template<typename T2> inline bool         is_alias(const T2& t) const { return P.is_alias(t);         }
  template<typename T2> inline bool is_inexact_alias(const T2& t) const { return P.is_inexact_alias(t); }

  inline uword get_n_rows() const          { return P.get_n_rows();          }
  inline uword get_M_n_rows() const        { return P.get_M_n_rows();        }
  inline uword get_n_cols() const          { return P.get_n_cols();          }
  inline uword get_n_slices() const        { return P.get_n_slices();        }
  inline uword get_n_elem() const          { return P.get_n_elem();          }
  inline uword get_M_n_elem_slice() const  { return P.get_M_n_elem_slice();  }

  inline bool is_empty() const { return P.is_empty(); }
  };



template<typename T, size_t in_dims, size_t out_dims>
struct Proxy_glue_helper { };

template<typename T, size_t dims>
struct Proxy_glue_helper<T, dims, dims>
  {
  typedef T result;
  };

template<typename T>
struct Proxy_glue_helper<T, 1, 2>
  {
  typedef ProxyMatCast<T, 1, false> result;
  };

template<typename T>
struct Proxy_glue_helper<T, 1, 3>
  {
  typedef ProxyCubeCast<T, 1, false> result;
  };

template<typename T>
struct Proxy_glue_helper<T, 2, 3>
  {
  typedef ProxyCubeCast<T, 2, false> result;
  };

template<typename T, typename T2>
struct Proxy_glue_type : public Proxy_glue_helper<T, Proxy<T>::num_dims, std::max(Proxy<T>::num_dims, Proxy<T2>::num_dims)> { };

template<typename T1, typename T2, typename eglue_type>
struct Proxy< eGlue<T1, T2, eglue_type> >
  {
  // promote T1 and T2 to have the same dimensions, if needed
  typedef typename Proxy_glue_type<T1, T2>::result P1_type;
  typedef typename Proxy_glue_type<T2, T1>::result P2_type;

  typedef eGlue<typename Proxy<P1_type>::held_type, typename Proxy<P2_type>::held_type, eglue_type> held_type;
  typedef typename Proxy<T1>::elem_type                                                             elem_type;

  const Proxy<P1_type> P1;
  const Proxy<P2_type> P2;
  const eGlue<T1, T2, eglue_type>& Q;

  inline Proxy(const eGlue<T1, T2, eglue_type>& in_Q) : P1(in_Q.A.Q), P2(in_Q.B.Q), Q(in_Q) { }

  // no extra arguments
  static constexpr const size_t num_args = Proxy<P1_type>::num_args + Proxy<P2_type>::num_args;
  // use the dimensionality of the largest of the two children
  static constexpr const size_t num_dims = Proxy<P1_type>::num_dims;

  typedef typename merge_tuple< typename Proxy<P1_type>::arg_types, typename Proxy<P2_type>::arg_types >::result arg_types;

  inline arg_types args() const { return std::tuple_cat( P1.args(), P2.args() ); }

  template<typename T3> inline bool         is_alias(const T3& t) const { return P1.is_alias(t) || P2.is_alias(t);                 }
  template<typename T3> inline bool is_inexact_alias(const T3& t) const { return P1.is_inexact_alias(t) || P2.is_inexact_alias(t); }

  // We assume that the size has already been checked and the T1/T2 have the same size.
  inline uword get_n_rows() const   { return P1.get_n_rows(); }
  inline uword get_M_n_rows() const { return P1.get_M_n_rows(); }
  inline uword get_n_cols() const   { return P1.get_n_cols(); }
  inline uword get_n_slices() const { return P1.get_n_slices(); }
  inline uword get_n_elem() const   { return P1.get_n_elem(); }

  inline bool is_empty() const { return P1.is_empty(); }
  };



template<typename T1, typename T2, typename eglue_type>
struct Proxy< eGlueCube<T1, T2, eglue_type> >
  {
  // promote T1 and T2 to have the same dimensions, if needed
  typedef typename Proxy_glue_type<T1, T2>::result P1_type;
  typedef typename Proxy_glue_type<T2, T1>::result P2_type;

  typedef eGlueCube<typename Proxy<P1_type>::held_type, typename Proxy<P2_type>::held_type, eglue_type> held_type;
  typedef typename Proxy<T1>::elem_type                                                                 elem_type;

  const Proxy<P1_type> P1;
  const Proxy<P2_type> P2;
  const eGlueCube<T1, T2, eglue_type>& Q;

  inline Proxy(const eGlueCube<T1, T2, eglue_type>& in_Q) : P1(in_Q.A.Q), P2(in_Q.B.Q), Q(in_Q) { }

  // no extra arguments
  static constexpr const size_t num_args = Proxy<P1_type>::num_args + Proxy<P2_type>::num_args;
  // use the dimensionality of the largest of the two children
  static constexpr const size_t num_dims = Proxy<P1_type>::num_dims;

  typedef typename merge_tuple< typename Proxy<P1_type>::arg_types, typename Proxy<P2_type>::arg_types >::result arg_types;

  inline arg_types args() const { return std::tuple_cat( P1.args(), P2.args() ); }

  template<typename T3> inline bool         is_alias(const T3& t) const { return P1.is_alias(t) || P2.is_alias(t);                 }
  template<typename T3> inline bool is_inexact_alias(const T3& t) const { return P1.is_inexact_alias(t) || P2.is_inexact_alias(t); }

  // We assume that the size has already been checked and the T1/T2 have the same size.
  inline uword get_n_rows() const         { return P1.get_n_rows(); }
  inline uword get_M_n_rows() const       { return P1.get_M_n_rows(); }
  inline uword get_n_cols() const         { return P1.get_n_cols(); }
  inline uword get_n_slices() const       { return P1.get_n_slices(); }
  inline uword get_n_elem() const         { return P1.get_n_elem(); }
  inline uword get_M_n_elem_slice() const { return P1.get_M_n_elem_slice(); }

  inline bool is_empty() const { return P1.is_empty(); }
  };



//
// mtOp<out_eT, T1, mtop_conv_to>
//

template<typename out_eT, typename T1>
struct Proxy< mtOp<out_eT, T1, mtop_conv_to> >
  {
  typedef mtOp<out_eT, typename Proxy<T1>::held_type, mtop_conv_to> held_type;
  typedef out_eT                                                    elem_type;

  const Proxy<T1> P;
  const mtOp<out_eT, T1, mtop_conv_to>& Q;

  inline Proxy(const mtOp<out_eT, T1, mtop_conv_to>& in_Q) : P(in_Q.q), Q(in_Q) { }

  // no extra arguments
  static constexpr const size_t num_args = Proxy<T1>::num_args;
  static constexpr const size_t num_dims = Proxy<T1>::num_dims;

  typedef typename Proxy<T1>::arg_types arg_types;

  inline arg_types args() const { return P.args(); }

  template<typename T3> inline bool         is_alias(const T3& t) const { return P.is_alias(t);         }
  template<typename T3> inline bool is_inexact_alias(const T3& t) const { return P.is_inexact_alias(t); }

  inline uword get_n_rows() const   { return P.get_n_rows(); }
  inline uword get_n_cols() const   { return P.get_n_cols(); }
  inline uword get_n_slices() const { return P.get_n_slices(); }
  inline uword get_n_elem() const   { return P.get_n_elem(); }

  inline bool is_empty() const { return P.is_empty(); }
  };



//
// mtOpCube<out_eT, T1, mtop_conv_to>
//

template<typename out_eT, typename T1>
struct Proxy< mtOpCube<out_eT, T1, mtop_conv_to> >
  {
  typedef mtOpCube<out_eT, typename Proxy<T1>::held_type, mtop_conv_to> held_type;
  typedef out_eT                                                        elem_type;

  const Proxy<T1> P;
  const mtOpCube<out_eT, T1, mtop_conv_to>& Q;

  inline Proxy(const mtOpCube<out_eT, T1, mtop_conv_to>& in_Q) : P(in_Q.q), Q(in_Q) { }

  // no extra arguments
  static constexpr const size_t num_args = Proxy<T1>::num_args;
  static constexpr const size_t num_dims = Proxy<T1>::num_dims;

  typedef typename Proxy<T1>::arg_types arg_types;

  inline arg_types args() const { return P.args(); }

  template<typename T3> inline bool         is_alias(const T3& t) const { return P.is_alias(t);         }
  template<typename T3> inline bool is_inexact_alias(const T3& t) const { return P.is_inexact_alias(t); }

  inline uword get_n_rows() const   { return P.get_n_rows(); }
  inline uword get_n_cols() const   { return P.get_n_cols(); }
  inline uword get_n_slices() const { return P.get_n_slices(); }
  inline uword get_n_elem() const   { return P.get_n_elem(); }

  inline bool is_empty() const { return P.is_empty(); }
  };



//
// mtOp<out_eT, T1, mtop_rel_core<mtop_type> >
//

template<typename out_eT, typename T1, typename mtop_type>
struct Proxy< mtOp<out_eT, T1, mtop_rel_core<mtop_type> > >
  {
  typedef mtOp<out_eT, typename Proxy<T1>::held_type, mtop_rel_core<mtop_type> > held_type;
  typedef out_eT                                                                 elem_type;

  const Proxy<T1> P;
  const mtOp<out_eT, T1, mtop_rel_core<mtop_type> >& Q;

  inline Proxy(const mtOp<out_eT, T1, mtop_rel_core<mtop_type> >& in_Q) : P(in_Q.q), Q(in_Q) { }

  // one extra argument for the scalar
  static constexpr const size_t num_args = Proxy<T1>::num_args + 1;
  static constexpr const size_t num_dims = Proxy<T1>::num_dims;

  typedef typename merge_tuple< typename Proxy<T1>::arg_types, std::tuple< const typename T1::elem_type& > >::result arg_types;

  inline arg_types args() const { return std::tuple_cat( P.args(), std::tie< const typename T1::elem_type& >(Q.aux) ); }

  template<typename T3> inline bool         is_alias(const T3& t) const { return P.is_alias(t);         }
  template<typename T3> inline bool is_inexact_alias(const T3& t) const { return P.is_inexact_alias(t); }

  inline uword get_n_rows() const   { return P.get_n_rows(); }
  inline uword get_n_cols() const   { return P.get_n_cols(); }
  inline uword get_n_slices() const { return P.get_n_slices(); }
  inline uword get_n_elem() const   { return P.get_n_elem(); }

  inline bool is_empty() const { return P.is_empty(); }
  };



//
// mtOpCube<out_eT, T1, mtop_rel_core<mtop_type> >
//

template<typename out_eT, typename T1, typename mtop_type>
struct Proxy< mtOpCube<out_eT, T1, mtop_rel_core<mtop_type> > >
  {
  typedef mtOpCube<out_eT, typename Proxy<T1>::held_type, mtop_rel_core<mtop_type> > held_type;
  typedef out_eT                                                                     elem_type;

  const Proxy<T1> P;
  const mtOpCube<out_eT, T1, mtop_rel_core<mtop_type> >& Q;

  inline Proxy(const mtOpCube<out_eT, T1, mtop_rel_core<mtop_type> >& in_Q) : P(in_Q.q), Q(in_Q) { }

  // one extra argument for the scalar
  static constexpr const size_t num_args = Proxy<T1>::num_args;
  static constexpr const size_t num_dims = Proxy<T1>::num_dims;

  typedef typename merge_tuple< typename Proxy<T1>::arg_types, std::tuple< const typename T1::elem_type& > >::result arg_types;

  inline arg_types args() const { return std::tuple_cat( P.args(), std::tie< const typename T1::elem_type& >(Q.aux) ); }

  template<typename T3> inline bool         is_alias(const T3& t) const { return P.is_alias(t);         }
  template<typename T3> inline bool is_inexact_alias(const T3& t) const { return P.is_inexact_alias(t); }

  inline uword get_n_rows() const   { return P.get_n_rows(); }
  inline uword get_n_cols() const   { return P.get_n_cols(); }
  inline uword get_n_slices() const { return P.get_n_slices(); }
  inline uword get_n_elem() const   { return P.get_n_elem(); }

  inline bool is_empty() const { return P.is_empty(); }
  };



//
// mtGlue<out_eT, T1, T2, mtglue_mixed_core<mtglue_type> >
//

template<typename out_eT, typename T1, typename T2, typename mtglue_type>
struct Proxy< mtGlue<out_eT, T1, T2, mtglue_mixed_core<mtglue_type> > >
  {
  // promote T1 and T2 to have the same dimensions, if needed
  typedef typename Proxy_glue_type<T1, T2>::result P1_type;
  typedef typename Proxy_glue_type<T2, T1>::result P2_type;

  typedef mtGlue<out_eT, typename Proxy<P1_type>::held_type, typename Proxy<P2_type>::held_type, mtglue_mixed_core<mtglue_type> > held_type;
  typedef out_eT                                                                                                                  elem_type;

  const Proxy<P1_type> P1;
  const Proxy<P2_type> P2;
  const mtGlue<out_eT, T1, T2, mtglue_mixed_core<mtglue_type> >& Q;

  inline Proxy(const mtGlue<out_eT, T1, T2, mtglue_mixed_core<mtglue_type> >& in_Q) : P1(in_Q.A), P2(in_Q.B), Q(in_Q) { }

  // no extra arguments
  static constexpr const size_t num_args = Proxy<P1_type>::num_args + Proxy<P2_type>::num_args;
  // use the dimensionality of the largest of the two children
  static constexpr const size_t num_dims = Proxy<P1_type>::num_dims;

  typedef typename merge_tuple< typename Proxy<P1_type>::arg_types, typename Proxy<P2_type>::arg_types >::result arg_types;

  inline arg_types args() const { return std::tuple_cat( P1.args(), P2.args() ); }

  template<typename T3> inline bool         is_alias(const T3& t) const { return P1.is_alias(t) || P2.is_alias(t);                 }
  template<typename T3> inline bool is_inexact_alias(const T3& t) const { return P1.is_inexact_alias(t) || P2.is_inexact_alias(t); }

  // We assume that the size has already been checked and the T1/T2 have the same size.
  inline uword get_n_rows() const   { return P1.get_n_rows(); }
  inline uword get_n_cols() const   { return P1.get_n_cols(); }
  inline uword get_n_slices() const { return P1.get_n_slices(); }
  inline uword get_n_elem() const   { return P1.get_n_elem(); }

  inline bool is_empty() const { return P1.is_empty(); }
  };



//
// mtGlueCube<out_eT, T1, T2, mtglue_mixed_core<mtglue_type> >
//

template<typename out_eT, typename T1, typename T2, typename mtglue_type>
struct Proxy< mtGlueCube<out_eT, T1, T2, mtglue_mixed_core<mtglue_type> > >
  {
  // promote T1 and T2 to have the same dimensions, if needed
  typedef typename Proxy_glue_type<T1, T2>::result P1_type;
  typedef typename Proxy_glue_type<T2, T1>::result P2_type;

  typedef mtGlueCube<out_eT, typename Proxy<P1_type>::held_type, typename Proxy<P2_type>::held_type, mtglue_mixed_core<mtglue_type> > held_type;
  typedef out_eT                                                                                                                      elem_type;

  const Proxy<P1_type> P1;
  const Proxy<P2_type> P2;
  const mtGlueCube<out_eT, T1, T2, mtglue_mixed_core<mtglue_type> >& Q;

  inline Proxy(const mtGlueCube<out_eT, T1, T2, mtglue_mixed_core<mtglue_type> >& in_Q) : P1(in_Q.A), P2(in_Q.B), Q(in_Q) { }

  // no extra arguments
  static constexpr const size_t num_args = Proxy<P1_type>::num_args + Proxy<P2_type>::num_args;
  // use the dimensionality of the largest of the two children
  static constexpr const size_t num_dims = Proxy<P1_type>::num_dims;

  typedef typename merge_tuple< typename Proxy<P1_type>::arg_types, typename Proxy<P2_type>::arg_types >::result arg_types;

  inline arg_types args() const { return std::tuple_cat( P1.args(), P2.args() ); }

  template<typename T3> inline bool         is_alias(const T3& t) const { return P1.is_alias(t) || P2.is_alias(t);                 }
  template<typename T3> inline bool is_inexact_alias(const T3& t) const { return P1.is_inexact_alias(t) || P2.is_inexact_alias(t); }

  // We assume that the size has already been checked and the T1/T2 have the same size.
  inline uword get_n_rows() const   { return P1.get_n_rows(); }
  inline uword get_n_cols() const   { return P1.get_n_cols(); }
  inline uword get_n_slices() const { return P1.get_n_slices(); }
  inline uword get_n_elem() const   { return P1.get_n_elem(); }

  inline bool is_empty() const { return P1.is_empty(); }
  };



//
// mtGlue<out_eT, T1, T2, mtglue_rel_core<mtglue_type> >
//

template<typename out_eT, typename T1, typename T2, typename mtglue_type>
struct Proxy< mtGlue<out_eT, T1, T2, mtglue_rel_core<mtglue_type> > >
  {
  // promote T1 and T2 to have the same dimensions, if needed
  typedef typename Proxy_glue_type<T1, T2>::result P1_type;
  typedef typename Proxy_glue_type<T2, T1>::result P2_type;

  typedef mtGlue<out_eT, typename Proxy<P1_type>::held_type, typename Proxy<P2_type>::held_type, mtglue_rel_core<mtglue_type> > held_type;
  typedef out_eT                                                                                                                elem_type;

  const Proxy<P1_type> P1;
  const Proxy<P2_type> P2;
  const mtGlue<out_eT, T1, T2, mtglue_rel_core<mtglue_type> >& Q;

  inline Proxy(const mtGlue<out_eT, T1, T2, mtglue_rel_core<mtglue_type> >& in_Q) : P1(in_Q.A), P2(in_Q.B), Q(in_Q) { }

  // no extra arguments
  static constexpr const size_t num_args = Proxy<P1_type>::num_args + Proxy<P2_type>::num_args;
  // use the dimensionality of the largest of the two children
  static constexpr const size_t num_dims = Proxy<P1_type>::num_dims;

  typedef typename merge_tuple< typename Proxy<P1_type>::arg_types, typename Proxy<P2_type>::arg_types >::result arg_types;

  inline arg_types args() const { return std::tuple_cat( P1.args(), P2.args() ); }

  template<typename T3> inline bool         is_alias(const T3& t) const { return P1.is_alias(t) || P2.is_alias(t);                 }
  template<typename T3> inline bool is_inexact_alias(const T3& t) const { return P1.is_inexact_alias(t) || P2.is_inexact_alias(t); }

  // We assume that the size has already been checked and the T1/T2 have the same size.
  inline uword get_n_rows() const   { return P1.get_n_rows(); }
  inline uword get_n_cols() const   { return P1.get_n_cols(); }
  inline uword get_n_slices() const { return P1.get_n_slices(); }
  inline uword get_n_elem() const   { return P1.get_n_elem(); }

  inline bool is_empty() const { return P1.is_empty(); }
  };



//
// mtGlueCube<out_eT, T1, T2, mtglue_rel_core<mtglue_type> >
//

template<typename out_eT, typename T1, typename T2, typename mtglue_type>
struct Proxy< mtGlueCube<out_eT, T1, T2, mtglue_rel_core<mtglue_type> > >
  {
  // promote T1 and T2 to have the same dimensions, if needed
  typedef typename Proxy_glue_type<T1, T2>::result P1_type;
  typedef typename Proxy_glue_type<T2, T1>::result P2_type;

  typedef mtGlueCube<out_eT, typename Proxy<P1_type>::held_type, typename Proxy<P2_type>::held_type, mtglue_rel_core<mtglue_type> > held_type;
  typedef out_eT                                                                                                                    elem_type;

  const Proxy<P1_type> P1;
  const Proxy<P2_type> P2;
  const mtGlueCube<out_eT, T1, T2, mtglue_rel_core<mtglue_type> >& Q;

  inline Proxy(const mtGlueCube<out_eT, T1, T2, mtglue_rel_core<mtglue_type> >& in_Q) : P1(in_Q.A), P2(in_Q.B), Q(in_Q) { }

  // no extra arguments
  static constexpr const size_t num_args = Proxy<P1_type>::num_args + Proxy<P2_type>::num_args;
  // use the dimensionality of the largest of the two children
  static constexpr const size_t num_dims = Proxy<P1_type>::num_dims;

  typedef typename merge_tuple< typename Proxy<P1_type>::arg_types, typename Proxy<P2_type>::arg_types >::result arg_types;

  inline arg_types args() const { return std::tuple_cat( P1.args(), P2.args() ); }

  template<typename T3> inline bool         is_alias(const T3& t) const { return P1.is_alias(t) || P2.is_alias(t);                 }
  template<typename T3> inline bool is_inexact_alias(const T3& t) const { return P1.is_inexact_alias(t) || P2.is_inexact_alias(t); }

  // We assume that the size has already been checked and the T1/T2 have the same size.
  inline uword get_n_rows() const   { return P1.get_n_rows(); }
  inline uword get_n_cols() const   { return P1.get_n_cols(); }
  inline uword get_n_slices() const { return P1.get_n_slices(); }
  inline uword get_n_elem() const   { return P1.get_n_elem(); }

  inline bool is_empty() const { return P1.is_empty(); }
  };



// op_vectorise_col is the same as a ProxyColCast;
// however, to avoid unnecessary use of ProxyColCast,
// we inherit from T1 directly when possible

template<typename T1>
struct Proxy< Op<T1, op_vectorise_col> > : public proxy_col_type<T1>::type
  {
  inline Proxy(const Op<T1, op_vectorise_col>& in_Q) : proxy_col_type<T1>::type(in_Q.m) { }
  };



//
// op_htrans / op_strans:
// the T1 is forced to be two dimensions
//

template<typename T1, size_t dims>
struct Proxy_trans_type
  {
  typedef ProxyMatCast<T1, dims> type;
  };


template<typename T1>
struct Proxy_trans_type<T1, 2>
  {
  typedef T1 type;
  };

template<typename T1>
struct Proxy< Op<T1, op_htrans> >
  {
  typedef Op<typename Proxy<typename Proxy_trans_type<T1, Proxy<T1>::num_dims>::type>::held_type, op_htrans> held_type;
  typedef typename Proxy<typename Proxy_trans_type<T1, Proxy<T1>::num_dims>::type>::elem_type                elem_type;

  const Proxy<typename Proxy_trans_type<T1, Proxy<T1>::num_dims>::type> P;
  const Op<T1, op_htrans>& Q;

  inline Proxy(const Op<T1, op_htrans>& in_Q) : P(in_Q.m), Q(in_Q) { }

  // no extra arguments
  static constexpr const size_t num_args = Proxy<typename Proxy_trans_type<T1, Proxy<T1>::num_dims>::type>::num_args;
  static constexpr const size_t num_dims = Proxy<typename Proxy_trans_type<T1, Proxy<T1>::num_dims>::type>::num_dims;

  typedef typename Proxy<typename Proxy_trans_type<T1, Proxy<T1>::num_dims>::type>::arg_types arg_types;

  inline arg_types args() const { return P.args(); }

  template<typename T2> inline bool         is_alias(const T2& t) const { return P.is_alias(t);         }
  template<typename T2> inline bool is_inexact_alias(const T2& t) const { return P.is_inexact_alias(t); }

  inline uword get_n_rows() const   { return P.get_n_cols();   }
  inline uword get_M_n_rows() const { return get_n_rows();     }
  inline uword get_n_cols() const   { return P.get_n_rows();   }
  inline uword get_n_slices() const { return P.get_n_slices(); }
  inline uword get_n_elem() const   { return P.get_n_elem();   }

  inline bool is_empty() const { return P.is_empty(); }
  };



template<typename T1>
struct Proxy< Op<T1, op_strans> >
  {
  typedef Op<typename Proxy<typename Proxy_trans_type<T1, Proxy<T1>::num_dims>::type>::held_type, op_htrans> held_type;
  typedef typename Proxy<typename Proxy_trans_type<T1, Proxy<T1>::num_dims>::type>::elem_type                elem_type;

  const Proxy<typename Proxy_trans_type<T1, Proxy<T1>::num_dims>::type> P;
  const Op<T1, op_strans>& Q;

  inline Proxy(const Op<T1, op_strans>& in_Q) : P(in_Q.m), Q(in_Q) { }

  // no extra arguments
  static constexpr const size_t num_args = Proxy<typename Proxy_trans_type<T1, Proxy<T1>::num_dims>::type>::num_args;
  static constexpr const size_t num_dims = Proxy<typename Proxy_trans_type<T1, Proxy<T1>::num_dims>::type>::num_dims;

  typedef typename Proxy<typename Proxy_trans_type<T1, Proxy<T1>::num_dims>::type>::arg_types arg_types;

  inline arg_types args() const { return P.args(); }

  template<typename T2> inline bool         is_alias(const T2& t) const { return P.is_alias(t);         }
  template<typename T2> inline bool is_inexact_alias(const T2& t) const { return P.is_inexact_alias(t); }

  inline uword get_n_rows() const   { return P.get_n_cols();   }
  inline uword get_M_n_rows() const { return get_n_rows();     }
  inline uword get_n_cols() const   { return P.get_n_rows();   }
  inline uword get_n_slices() const { return P.get_n_slices(); }
  inline uword get_n_elem() const   { return P.get_n_elem();   }

  inline bool is_empty() const { return P.is_empty(); }
  };
