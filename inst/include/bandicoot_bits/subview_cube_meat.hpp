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



template<typename eT>
inline
subview_cube<eT>::~subview_cube()
  {
  coot_extra_debug_sigprint_this(this);
  }



template<typename eT>
coot_inline
subview_cube<eT>::subview_cube
  (
  const Cube<eT>& in_m,
  const uword     in_row1,
  const uword     in_col1,
  const uword     in_slice1,
  const uword     in_n_rows,
  const uword     in_n_cols,
  const uword     in_n_slices
  )
  : m           (in_m)
  , aux_row1    (in_row1)
  , aux_col1    (in_col1)
  , aux_slice1  (in_slice1)
  , n_rows      (in_n_rows)
  , n_cols      (in_n_cols)
  , n_elem_slice(in_n_rows * in_n_cols)
  , n_slices    (in_n_slices)
  , n_elem      (n_elem_slice * in_n_slices)
  {
  coot_extra_debug_sigprint_this(this);
  }



template<typename eT>
inline
subview_cube<eT>::subview_cube(const subview_cube<eT>& in)
  : m           (in.m           )
  , aux_row1    (in.aux_row1    )
  , aux_col1    (in.aux_col1    )
  , aux_slice1  (in.aux_slice1  )
  , n_rows      (in.n_rows      )
  , n_cols      (in.n_cols      )
  , n_elem_slice(in.n_elem_slice)
  , n_slices    (in.n_slices    )
  , n_elem      (in.n_elem      )
  {
  coot_extra_debug_sigprint_this(this);
  }



template<typename eT>
inline
subview_cube<eT>::subview_cube(subview_cube<eT>&& in)
  : m           (in.m           )
  , aux_row1    (in.aux_row1    )
  , aux_col1    (in.aux_col1    )
  , aux_slice1  (in.aux_slice1  )
  , n_rows      (in.n_rows      )
  , n_cols      (in.n_cols      )
  , n_elem_slice(in.n_elem_slice)
  , n_slices    (in.n_slices    )
  , n_elem      (in.n_elem      )
  {
  coot_extra_debug_sigprint_this(this);

  // for paranoia

  access::rw(in.aux_row1    ) = 0;
  access::rw(in.aux_col1    ) = 0;
  access::rw(in.aux_slice1  ) = 0;
  access::rw(in.n_rows      ) = 0;
  access::rw(in.n_cols      ) = 0;
  access::rw(in.n_elem_slice) = 0;
  access::rw(in.n_slices    ) = 0;
  access::rw(in.n_elem      ) = 0;
  }



template<typename eT>
inline
subview_cube<eT>::operator arma::Cube<eT> () const
  {
  coot_extra_debug_sigprint();

  #if defined(COOT_HAVE_ARMA)
    {
    // TODO: improve this implementation.  For now we extract the subview.
    Cube<eT> v(*this);

    arma::Cube<eT> out(v.n_rows, v.n_cols, v.n_slices);

    v.copy_from_dev_mem(out.memptr(), v.n_elem);

    return out;
    }
  #else
    {
    coot_stop_logic_error("#include <armadillo> must be before #include <bandicoot>");

    return arma::Cube<eT>();
    }
  #endif
  }



template<typename eT>
inline
void
subview_cube<eT>::operator=(const eT val)
  {
  coot_extra_debug_sigprint();

  if (n_elem == 1)
    {
    Cube<eT>& X = const_cast< Cube<eT>& >(m);

    X.at(aux_row1, aux_col1, aux_slice1) = val;
    }
  else
    {
    coot_debug_assert_same_size(n_rows, n_cols, n_slices, 1, 1, 1, "subview_cube::operator=");
    }
  }



template<typename eT>
inline
void
subview_cube<eT>::operator+= (const eT val)
  {
  coot_extra_debug_sigprint();

  coot_rt_t::eop_scalar(twoway_kernel_id::equ_array_plus_scalar,
                        m.dev_mem, m.dev_mem,
                        (eT) val, (eT) 0,
                        n_rows, n_cols, n_slices,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols);
  }



template<typename eT>
inline
void
subview_cube<eT>::operator-= (const eT val)
  {
  coot_extra_debug_sigprint();

  coot_rt_t::eop_scalar(twoway_kernel_id::equ_array_minus_scalar_post,
                        m.dev_mem, m.dev_mem,
                        (eT) val, (eT) 0,
                        n_rows, n_cols, n_slices,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols);
  }



template<typename eT>
inline
void
subview_cube<eT>::operator*= (const eT val)
  {
  coot_extra_debug_sigprint();

  coot_rt_t::eop_scalar(twoway_kernel_id::equ_array_mul_scalar,
                        m.dev_mem, m.dev_mem,
                        (eT) val, (eT) 1,
                        n_rows, n_cols, n_slices,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols);
  }



template<typename eT>
inline
void
subview_cube<eT>::operator/= (const eT val)
  {
  coot_extra_debug_sigprint();

  coot_rt_t::eop_scalar(twoway_kernel_id::equ_array_div_scalar_post,
                        m.dev_mem, m.dev_mem,
                        (eT) val, (eT) 1,
                        n_rows, n_cols, n_slices,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols);
  }



template<typename eT>
template<typename T1>
inline
void
subview_cube<eT>::operator=(const BaseCube<eT, T1>& x)
  {
  coot_extra_debug_sigprint();

  no_conv_unwrap_cube<T1> U(x.get_ref());

  coot_rt_t::copy_cube(m.dev_mem, U.get_dev_mem(false),
                       n_rows, n_cols, n_slices,
                       aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                       U.get_row_offset(), U.get_col_offset(), U.get_slice_offset(), U.get_M_n_rows(), U.get_M_n_cols());
  }



template<typename eT>
template<typename T1>
inline
void
subview_cube<eT>::operator+=(const BaseCube<eT, T1>& x)
  {
  coot_extra_debug_sigprint();

  no_conv_unwrap_cube<T1> U(x.get_ref());

  coot_rt_t::eop_cube(twoway_kernel_id::equ_array_plus_array_cube,
                      m.dev_mem, m.dev_mem, U.get_dev_mem(false),
                      n_rows, n_cols, n_slices,
                      aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                      aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                      U.get_row_offset(), U.get_col_offset(), U.get_slice_offset(), U.get_M_n_rows(), U.get_M_n_cols());
  }



template<typename eT>
template<typename T1>
inline
void
subview_cube<eT>::operator-=(const BaseCube<eT, T1>& x)
  {
  coot_extra_debug_sigprint();

  no_conv_unwrap_cube<T1> U(x.get_ref());

  coot_rt_t::eop_cube(twoway_kernel_id::equ_array_minus_array_cube,
                      m.dev_mem, m.dev_mem, U.get_dev_mem(false),
                      n_rows, n_cols, n_slices,
                      aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                      aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                      U.get_row_offset(), U.get_col_offset(), U.get_slice_offset(), U.get_M_n_rows(), U.get_M_n_cols());
  }



template<typename eT>
template<typename T1>
inline
void
subview_cube<eT>::operator%=(const BaseCube<eT, T1>& x)
  {
  coot_extra_debug_sigprint();

  no_conv_unwrap_cube<T1> U(x.get_ref());

  coot_rt_t::eop_cube(twoway_kernel_id::equ_array_mul_array_cube,
                      m.dev_mem, m.dev_mem, U.get_dev_mem(false),
                      n_rows, n_cols, n_slices,
                      aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                      aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                      U.get_row_offset(), U.get_col_offset(), U.get_slice_offset(), U.get_M_n_rows(), U.get_M_n_cols());
  }



template<typename eT>
template<typename T1>
inline
void
subview_cube<eT>::operator/=(const BaseCube<eT, T1>& x)
  {
  coot_extra_debug_sigprint();

  no_conv_unwrap_cube<T1> U(x.get_ref());

  coot_rt_t::eop_cube(twoway_kernel_id::equ_array_div_array_cube,
                      m.dev_mem, m.dev_mem, U.get_dev_mem(false),
                      n_rows, n_cols, n_slices,
                      aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                      aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                      U.get_row_offset(), U.get_col_offset(), U.get_slice_offset(), U.get_M_n_rows(), U.get_M_n_cols());
  }



template<typename eT>
template<typename T1>
inline
void
subview_cube<eT>::operator=(const Base<eT, T1>& x)
  {
  coot_extra_debug_sigprint();

  no_conv_unwrap<T1> U(x.get_ref());

  // if this subview can be interpreted as an object with the same dimensions as x, we can use it
  const uword t_n_rows   = this->n_rows;
  const uword t_n_cols   = this->n_cols;
  const uword t_n_slices = this->n_slices;

  const uword x_n_rows   = U.M.n_rows;
  const uword x_n_cols   = U.M.n_cols;

  if( ((x_n_rows == 1) || (x_n_cols == 1)) && (t_n_rows == 1) && (t_n_cols == 1) && (x.n_elem == t_n_slices) )
    {
    // interpret the matrix as a scalar that will apply to every (scalar) slice
    coot_rt_t::broadcast_op(twoway_kernel_id::broadcast_set,
                            m.dev_mem, m.dev_mem /* ignored */, U.get_dev_mem(false),
                            1, 1,
                            1, t_n_slices,
                            U.get_row_offset(), U.get_col_offset(), U.get_M_n_rows(),
                            aux_row1 + aux_col1 * m.n_rows + aux_slice1 * m.n_rows * m.n_cols, 0, m.slice_elem);
    }
  else
  if( (t_n_rows == x_n_rows) && (t_n_cols == x_n_cols) && (t_n_slices == 1) )
    {
    // interpret the matrix as a cube with one slice
    coot_rt_t::eop_cube(twoway_kernel_id::convert_type_cube,
                        m.dev_mem, m.dev_mem, U.get_dev_mem(false),
                        t_n_rows, t_n_cols, t_n_slices,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                        U.get_row_offset(), U.get_col_offset(), 0, U.get_M_n_rows(), x_n_rows * x_n_cols);
    }
  else
  if( (t_n_rows == x_n_rows) && (t_n_cols == 1) && (t_n_slices == x_n_cols) )
    {
    // interpret the matrix as a rows x 1 x slices tube
    coot_rt_t::eop_cube(twoway_kernel_id::convert_type_cube,
                        m.dev_mem, m.dev_mem, U.get_dev_mem(false),
                        t_n_rows, t_n_cols, t_n_slices,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                        U.get_row_offset(), 0, U.get_col_offset(), U.get_M_n_rows(), 1);
    }
  else
  if( (t_n_rows == 1) && (t_n_cols == x_n_rows) && (t_n_slices == x_n_cols) )
    {
    // intepret the matrix as a 1 x cols x slices tube
    coot_rt_t::eop_cube(twoway_kernel_id::convert_type_cube,
                        m.dev_mem, m.dev_mem, U.get_dev_mem(false),
                        t_n_rows, t_n_cols, t_n_slices,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                        0, U.get_row_offset(), U.get_col_offset(), 1, U.get_M_n_rows());
    }
  else
    {
    coot_stop_logic_error( coot_incompat_size_string(*this, x, "copy into subcube") );
    }
  }



template<typename eT>
template<typename T1>
inline
void
subview_cube<eT>::operator+=(const Base<eT, T1>& x)
  {
  coot_extra_debug_sigprint();

  no_conv_unwrap<T1> U(x.get_ref());

  // if this subview can be interpreted as an object with the same dimensions as x, we can use it
  const uword t_n_rows   = this->n_rows;
  const uword t_n_cols   = this->n_cols;
  const uword t_n_slices = this->n_slices;

  const uword x_n_rows   = U.M.n_rows;
  const uword x_n_cols   = U.M.n_cols;

  if( ((x_n_rows == 1) || (x_n_cols == 1)) && (t_n_rows == 1) && (t_n_cols == 1) && (x.n_elem == t_n_slices) )
    {
    // interpret the matrix as a scalar that will apply to every (scalar) slice
    coot_rt_t::broadcast_op(twoway_kernel_id::broadcast_plus,
                            m.dev_mem, m.dev_mem, U.get_dev_mem(false),
                            1, 1,
                            1, t_n_slices,
                            U.get_row_offset(), U.get_col_offset(), U.get_M_n_rows(),
                            aux_row1 + aux_col1 * m.n_rows + aux_slice1 * m.n_rows * m.n_cols, 0, m.slice_elem);
    }
  else
  if( (t_n_rows == x_n_rows) && (t_n_cols == x_n_cols) && (t_n_slices == 1) )
    {
    // interpret the matrix as a cube with one slice
    coot_rt_t::eop_cube(twoway_kernel_id::equ_array_plus_array_cube,
                        m.dev_mem, m.dev_mem, U.get_dev_mem(false),
                        t_n_rows, t_n_cols, t_n_slices,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                        U.get_row_offset(), U.get_col_offset(), 0, U.get_M_n_rows(), x_n_rows * x_n_cols);
    }
  else
  if( (t_n_rows == x_n_rows) && (t_n_cols == 1) && (t_n_slices == x_n_cols) )
    {
    // interpret the matrix as a rows x 1 x slices tube
    coot_rt_t::eop_cube(twoway_kernel_id::equ_array_plus_array_cube,
                        m.dev_mem, m.dev_mem, U.get_dev_mem(false),
                        t_n_rows, t_n_cols, t_n_slices,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                        U.get_row_offset(), 0, U.get_col_offset(), U.get_M_n_rows(), 1);
    }
  else
  if( (t_n_rows == 1) && (t_n_cols == x_n_rows) && (t_n_slices == x_n_cols) )
    {
    // intepret the matrix as a 1 x cols x slices tube
    coot_rt_t::eop_cube(twoway_kernel_id::equ_array_plus_array_cube,
                        m.dev_mem, m.dev_mem, U.get_dev_mem(false),
                        t_n_rows, t_n_cols, t_n_slices,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                        0, U.get_row_offset(), U.get_col_offset(), 1, U.get_M_n_rows());
    }
  else
    {
    coot_stop_logic_error( coot_incompat_size_string(*this, x, "addition") );
    }
  }



template<typename eT>
template<typename T1>
inline
void
subview_cube<eT>::operator-=(const Base<eT, T1>& x)
  {
  coot_extra_debug_sigprint();

  no_conv_unwrap<T1> U(x.get_ref());

  // if this subview can be interpreted as an object with the same dimensions as x, we can use it
  const uword t_n_rows   = this->n_rows;
  const uword t_n_cols   = this->n_cols;
  const uword t_n_slices = this->n_slices;

  const uword x_n_rows   = U.M.n_rows;
  const uword x_n_cols   = U.M.n_cols;

  if( ((x_n_rows == 1) || (x_n_cols == 1)) && (t_n_rows == 1) && (t_n_cols == 1) && (x.n_elem == t_n_slices) )
    {
    // interpret the matrix as a scalar that will apply to every (scalar) slice
    coot_rt_t::broadcast_op(twoway_kernel_id::broadcast_minus_post,
                            m.dev_mem, m.dev_mem, U.get_dev_mem(false),
                            1, 1,
                            1, t_n_slices,
                            U.get_row_offset(), U.get_col_offset(), U.get_M_n_rows(),
                            aux_row1 + aux_col1 * m.n_rows + aux_slice1 * m.n_rows * m.n_cols, 0, m.slice_elem);
    }
  else
  if( (t_n_rows == x_n_rows) && (t_n_cols == x_n_cols) && (t_n_slices == 1) )
    {
    // interpret the matrix as a cube with one slice
    coot_rt_t::eop_cube(twoway_kernel_id::equ_array_minus_array_cube,
                        m.dev_mem, m.dev_mem, U.get_dev_mem(false),
                        t_n_rows, t_n_cols, t_n_slices,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                        U.get_row_offset(), U.get_col_offset(), 0, U.get_M_n_rows(), x_n_rows * x_n_cols);
    }
  else
  if( (t_n_rows == x_n_rows) && (t_n_cols == 1) && (t_n_slices == x_n_cols) )
    {
    // interpret the matrix as a rows x 1 x slices tube
    coot_rt_t::eop_cube(twoway_kernel_id::equ_array_minus_array_cube,
                        m.dev_mem, m.dev_mem, U.get_dev_mem(false),
                        t_n_rows, t_n_cols, t_n_slices,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                        U.get_row_offset(), 0, U.get_col_offset(), U.get_M_n_rows(), 1);
    }
  else
  if( (t_n_rows == 1) && (t_n_cols == x_n_rows) && (t_n_slices == x_n_cols) )
    {
    // intepret the matrix as a 1 x cols x slices tube
    coot_rt_t::eop_cube(twoway_kernel_id::equ_array_minus_array_cube,
                        m.dev_mem, m.dev_mem, U.get_dev_mem(false),
                        t_n_rows, t_n_cols, t_n_slices,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                        0, U.get_row_offset(), U.get_col_offset(), 1, U.get_M_n_rows());
    }
  else
    {
    coot_stop_logic_error( coot_incompat_size_string(*this, x, "subtraction") );
    }
  }



template<typename eT>
template<typename T1>
inline
void
subview_cube<eT>::operator%=(const Base<eT, T1>& x)
  {
  coot_extra_debug_sigprint();

  no_conv_unwrap<T1> U(x.get_ref());

  // if this subview can be interpreted as an object with the same dimensions as x, we can use it
  const uword t_n_rows   = this->n_rows;
  const uword t_n_cols   = this->n_cols;
  const uword t_n_slices = this->n_slices;

  const uword x_n_rows   = U.M.n_rows;
  const uword x_n_cols   = U.M.n_cols;

  if( ((x_n_rows == 1) || (x_n_cols == 1)) && (t_n_rows == 1) && (t_n_cols == 1) && (x.n_elem == t_n_slices) )
    {
    // interpret the matrix as a scalar that will apply to every (scalar) slice
    coot_rt_t::broadcast_op(twoway_kernel_id::broadcast_schur,
                            m.dev_mem, m.dev_mem, U.get_dev_mem(false),
                            1, 1,
                            1, t_n_slices,
                            U.get_row_offset(), U.get_col_offset(), U.get_M_n_rows(),
                            aux_row1 + aux_col1 * m.n_rows + aux_slice1 * m.n_rows * m.n_cols, 0, m.slice_elem);
    }
  else
  if( (t_n_rows == x_n_rows) && (t_n_cols == x_n_cols) && (t_n_slices == 1) )
    {
    // interpret the matrix as a cube with one slice
    coot_rt_t::eop_cube(twoway_kernel_id::equ_array_mul_array_cube,
                        m.dev_mem, m.dev_mem, U.get_dev_mem(false),
                        t_n_rows, t_n_cols, t_n_slices,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                        U.get_row_offset(), U.get_col_offset(), 0, U.get_M_n_rows(), x_n_rows * x_n_cols);
    }
  else
  if( (t_n_rows == x_n_rows) && (t_n_cols == 1) && (t_n_slices == x_n_cols) )
    {
    // interpret the matrix as a rows x 1 x slices tube
    coot_rt_t::eop_cube(twoway_kernel_id::equ_array_mul_array_cube,
                        m.dev_mem, m.dev_mem, U.get_dev_mem(false),
                        t_n_rows, t_n_cols, t_n_slices,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                        U.get_row_offset(), 0, U.get_col_offset(), U.get_M_n_rows(), 1);
    }
  else
  if( (t_n_rows == 1) && (t_n_cols == x_n_rows) && (t_n_slices == x_n_cols) )
    {
    // intepret the matrix as a 1 x cols x slices tube
    coot_rt_t::eop_cube(twoway_kernel_id::equ_array_mul_array_cube,
                        m.dev_mem, m.dev_mem, U.get_dev_mem(false),
                        t_n_rows, t_n_cols, t_n_slices,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                        0, U.get_row_offset(), U.get_col_offset(), 1, U.get_M_n_rows());
    }
  else
    {
    coot_stop_logic_error( coot_incompat_size_string(*this, x, "element-wise multiplication") );
    }
  }



template<typename eT>
template<typename T1>
inline
void
subview_cube<eT>::operator/=(const Base<eT, T1>& x)
  {
  coot_extra_debug_sigprint();

  no_conv_unwrap<T1> U(x.get_ref());

  // if this subview can be interpreted as an object with the same dimensions as x, we can use it
  const uword t_n_rows   = this->n_rows;
  const uword t_n_cols   = this->n_cols;
  const uword t_n_slices = this->n_slices;

  const uword x_n_rows   = U.M.n_rows;
  const uword x_n_cols   = U.M.n_cols;

  if( ((x_n_rows == 1) || (x_n_cols == 1)) && (t_n_rows == 1) && (t_n_cols == 1) && (x.n_elem == t_n_slices) )
    {
    // interpret the matrix as a scalar that will apply to every (scalar) slice
    coot_rt_t::broadcast_op(twoway_kernel_id::broadcast_div_post,
                            m.dev_mem, m.dev_mem, U.get_dev_mem(false),
                            1, 1,
                            1, t_n_slices,
                            U.get_row_offset(), U.get_col_offset(), U.get_M_n_rows(),
                            aux_row1 + aux_col1 * m.n_rows + aux_slice1 * m.n_rows * m.n_cols, 0, m.slice_elem);
    }
  else
  if( (t_n_rows == x_n_rows) && (t_n_cols == x_n_cols) && (t_n_slices == 1) )
    {
    // interpret the matrix as a cube with one slice
    coot_rt_t::eop_cube(twoway_kernel_id::equ_array_div_array_cube,
                        m.dev_mem, m.dev_mem, U.get_dev_mem(false),
                        t_n_rows, t_n_cols, t_n_slices,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                        U.get_row_offset(), U.get_col_offset(), 0, U.get_M_n_rows(), x_n_rows * x_n_cols);
    }
  else
  if( (t_n_rows == x_n_rows) && (t_n_cols == 1) && (t_n_slices == x_n_cols) )
    {
    // interpret the matrix as a rows x 1 x slices tube
    coot_rt_t::eop_cube(twoway_kernel_id::equ_array_div_array_cube,
                        m.dev_mem, m.dev_mem, U.get_dev_mem(false),
                        t_n_rows, t_n_cols, t_n_slices,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                        U.get_row_offset(), 0, U.get_col_offset(), U.get_M_n_rows(), 1);
    }
  else
  if( (t_n_rows == 1) && (t_n_cols == x_n_rows) && (t_n_slices == x_n_cols) )
    {
    // intepret the matrix as a 1 x cols x slices tube
    coot_rt_t::eop_cube(twoway_kernel_id::equ_array_div_array_cube,
                        m.dev_mem, m.dev_mem, U.get_dev_mem(false),
                        t_n_rows, t_n_cols, t_n_slices,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                        aux_row1, aux_col1, aux_slice1, m.n_rows, m.n_cols,
                        0, U.get_row_offset(), U.get_col_offset(), 1, U.get_M_n_rows());
    }
  else
    {
    coot_stop_logic_error( coot_incompat_size_string(*this, x, "addition into subcube") );
    }
  }



template<typename eT>
inline
void
subview_cube<eT>::extract(Cube<eT>& out, const subview_cube<eT>& in)
  {
  coot_extra_debug_sigprint();

  // extract the subview into `out`
  coot_rt_t::eop_cube(twoway_kernel_id::convert_type_cube,
                      out.get_dev_mem(false), out.get_dev_mem(false), in.m.get_dev_mem(false),
                      out.n_rows, out.n_cols, out.n_slices,
                      0, 0, 0, out.n_rows, out.n_cols,
                      0, 0, 0, out.n_rows, out.n_cols,
                      in.aux_row1, in.aux_col1, in.aux_slice1, in.m.n_rows, in.m.n_cols);
  }



template<typename eT>
inline
void
subview_cube<eT>::plus_inplace(Cube<eT>& out, const subview_cube<eT>& in)
  {
  coot_extra_debug_sigprint();

  coot_debug_assert_same_size(out, in, "addition");

  // add the subview to `out`
  coot_rt_t::eop_cube(twoway_kernel_id::equ_array_plus_array_cube,
                      out.get_dev_mem(false), out.get_dev_mem(false), in.m.get_dev_mem(false),
                      out.n_rows, out.n_cols, out.n_slices,
                      0, 0, 0, out.n_rows, out.n_cols,
                      0, 0, 0, out.n_rows, out.n_cols,
                      in.aux_row1, in.aux_col1, in.aux_slice1, in.m.n_rows, in.m.n_cols);
  }



template<typename eT>
inline
void
subview_cube<eT>::minus_inplace(Cube<eT>& out, const subview_cube<eT>& in)
  {
  coot_extra_debug_sigprint();

  coot_debug_assert_same_size(out, in, "subtraction");

  // subtract the subview from `out`
  coot_rt_t::eop_cube(twoway_kernel_id::equ_array_minus_array_cube,
                      out.get_dev_mem(false), out.get_dev_mem(false), in.m.get_dev_mem(false),
                      out.n_rows, out.n_cols, out.n_slices,
                      0, 0, 0, out.n_rows, out.n_cols,
                      0, 0, 0, out.n_rows, out.n_cols,
                      in.aux_row1, in.aux_col1, in.aux_slice1, in.m.n_rows, in.m.n_cols);
  }



template<typename eT>
inline
void
subview_cube<eT>::schur_inplace(Cube<eT>& out, const subview_cube<eT>& in)
  {
  coot_extra_debug_sigprint();

  coot_debug_assert_same_size(out, in, "element-wise multiplication");

  // add the subview to `out`
  coot_rt_t::eop_cube(twoway_kernel_id::equ_array_mul_array_cube,
                      out.get_dev_mem(false), out.get_dev_mem(false), in.m.get_dev_mem(false),
                      out.n_rows, out.n_cols, out.n_slices,
                      0, 0, 0, out.n_rows, out.n_cols,
                      0, 0, 0, out.n_rows, out.n_cols,
                      in.aux_row1, in.aux_col1, in.aux_slice1, in.m.n_rows, in.m.n_cols);
  }



template<typename eT>
inline
void
subview_cube<eT>::div_inplace(Cube<eT>& out, const subview_cube<eT>& in)
  {
  coot_extra_debug_sigprint();

  coot_debug_assert_same_size(out, in, "division");

  // add the subview to `out`
  coot_rt_t::eop_cube(twoway_kernel_id::equ_array_div_array_cube,
                      out.get_dev_mem(false), out.get_dev_mem(false), in.m.get_dev_mem(false),
                      out.n_rows, out.n_cols, out.n_slices,
                      0, 0, 0, out.n_rows, out.n_cols,
                      0, 0, 0, out.n_rows, out.n_cols,
                      in.aux_row1, in.aux_col1, in.aux_slice1, in.m.n_rows, in.m.n_cols);
  }



template<typename eT>
inline
void
subview_cube<eT>::extract(Mat<eT>& out, const subview_cube<eT>& in)
  {
  coot_extra_debug_sigprint();

  coot_debug_assert_cube_as_mat(out, in, "copy into matrix", false);

  const uword in_n_rows   = in.n_rows;
  const uword in_n_cols   = in.n_cols;
  const uword in_n_slices = in.n_slices;

  const uword out_vec_state = out.vec_state;

  if(in_n_slices == 1)
    {
    out.set_size(in_n_rows, in_n_cols);

    coot_rt_t::copy_mat(out.get_dev_mem(false), in.get_dev_mem(false),
                        in_n_rows, in_n_cols,
                        0, 0, out.n_rows,
                        in.aux_row1, in.aux_col1, in.m.n_rows);
    }
  else
    {
    if (out_vec_state == 0)
      {
      // out is a matrix
      if (in_n_cols == 1)
        {
        out.set_size(in_n_rows, in_n_slices);

        coot_rt_t::copy_cube(out.get_dev_mem(false), in.get_dev_mem(false),
                             in_n_rows, in_n_cols, in_n_slices,
                             0, 0, 0, out.n_rows, 1,
                             in.aux_row1, in.aux_col1, in.aux_slice1, in.m.n_rows, in.m.n_cols);
        }
      else if (in_n_rows == 1)
        {
        out.set_size(in_n_cols, in_n_slices);

        coot_rt_t::copy_cube(out.get_dev_mem(false), in.get_dev_mem(false),
                             in_n_rows, in_n_cols, in_n_slices,
                             0, 0, 0, 1, out.n_rows,
                             in.aux_row1, in.aux_col1, in.aux_slice1, in.m.n_rows, in.m.n_cols);
        }
      }
    else
      {
      // out is a vector
      out.set_size(in_n_slices);

      coot_rt_t::copy_cube(out.get_dev_mem(false), in.get_dev_mem(false),
                           1, 1, in_n_slices,
                           0, 0, 0, 1, 1,
                           in.aux_row1, in.aux_col1, in.aux_slice1, in.m.n_rows, in.m.n_cols);
      }
    }
  }



template<typename eT>
inline
void
subview_cube<eT>::plus_inplace(Mat<eT>& out, const subview_cube<eT>& in)
  {
  coot_extra_debug_sigprint();

  coot_debug_assert_cube_as_mat(out, in, "addition", true);

  const uword in_n_rows   = in.n_rows;
  const uword in_n_cols   = in.n_cols;
  const uword in_n_slices = in.n_slices;

  const uword out_vec_state = out.vec_state;

  if(in_n_slices == 1)
    {
    coot_rt_t::eop_mat(threeway_kernel_id::equ_array_plus_array,
                       out.get_dev_mem(false), out.get_dev_mem(false), in.get_dev_mem(false),
                       in_n_rows, in_n_cols,
                       0, 0, out.n_rows,
                       0, 0, out.n_rows,
                       in.aux_row1, in.aux_col1, in.m.n_rows);
    }
  else
    {
    if (out_vec_state == 0)
      {
      // out is a matrix
      if (in_n_cols == 1)
        {
        coot_rt_t::eop_cube(twoway_kernel_id::equ_array_plus_array_cube,
                            out.get_dev_mem(false), in.get_dev_mem(false),
                            in_n_rows, in_n_cols, in_n_slices,
                            0, 0, 0, out.n_rows, 1,
                            in.aux_row1, in.aux_col1, in.aux_slice1, in.m.n_rows, in.m.n_cols);
        }
      else if (in_n_rows == 1)
        {
        coot_rt_t::eop_cube(twoway_kernel_id::equ_array_plus_array_cube,
                            out.get_dev_mem(false), in.get_dev_mem(false),
                            in_n_rows, in_n_cols, in_n_slices,
                            0, 0, 0, 1, out.n_rows,
                            in.aux_row1, in.aux_col1, in.aux_slice1, in.m.n_rows, in.m.n_cols);
        }
      }
    else
      {
      // out is a vector
      coot_rt_t::eop_cube(twoway_kernel_id::equ_array_plus_array_cube,
                          out.get_dev_mem(false), in.get_dev_mem(false),
                          1, 1, in_n_slices,
                          0, 0, 0, 1, 1,
                          in.aux_row1, in.aux_col1, in.aux_slice1, in.m.n_rows, in.m.n_cols);
      }
    }
  }



template<typename eT>
inline
void
subview_cube<eT>::minus_inplace(Mat<eT>& out, const subview_cube<eT>& in)
  {
  coot_extra_debug_sigprint();

  coot_debug_assert_cube_as_mat(out, in, "subtraction", true);

  const uword in_n_rows   = in.n_rows;
  const uword in_n_cols   = in.n_cols;
  const uword in_n_slices = in.n_slices;

  const uword out_vec_state = out.vec_state;

  if(in_n_slices == 1)
    {
    coot_rt_t::eop_mat(threeway_kernel_id::equ_array_minus_array,
                       out.get_dev_mem(false), out.get_dev_mem(false), in.get_dev_mem(false),
                       in_n_rows, in_n_cols,
                       0, 0, out.n_rows,
                       0, 0, out.n_rows,
                       in.aux_row1, in.aux_col1, in.m.n_rows);
    }
  else
    {
    if (out_vec_state == 0)
      {
      // out is a matrix
      if (in_n_cols == 1)
        {
        coot_rt_t::eop_cube(twoway_kernel_id::equ_array_minus_array_cube,
                            out.get_dev_mem(false), in.get_dev_mem(false),
                            in_n_rows, in_n_cols, in_n_slices,
                            0, 0, 0, out.n_rows, 1,
                            in.aux_row1, in.aux_col1, in.aux_slice1, in.m.n_rows, in.m.n_cols);
        }
      else if (in_n_rows == 1)
        {
        coot_rt_t::eop_cube(twoway_kernel_id::equ_array_minus_array_cube,
                            out.get_dev_mem(false), in.get_dev_mem(false),
                            in_n_rows, in_n_cols, in_n_slices,
                            0, 0, 0, 1, out.n_rows,
                            in.aux_row1, in.aux_col1, in.aux_slice1, in.m.n_rows, in.m.n_cols);
        }
      }
    else
      {
      // out is a vector
      coot_rt_t::eop_cube(twoway_kernel_id::equ_array_minus_array_cube,
                          out.get_dev_mem(false), in.get_dev_mem(false),
                          1, 1, in_n_slices,
                          0, 0, 0, 1, 1,
                          in.aux_row1, in.aux_col1, in.aux_slice1, in.m.n_rows, in.m.n_cols);
      }
    }
  }



template<typename eT>
inline
void
subview_cube<eT>::schur_inplace(Mat<eT>& out, const subview_cube<eT>& in)
  {
  coot_extra_debug_sigprint();

  coot_debug_assert_cube_as_mat(out, in, "element-wise multiplication", true);

  const uword in_n_rows   = in.n_rows;
  const uword in_n_cols   = in.n_cols;
  const uword in_n_slices = in.n_slices;

  const uword out_vec_state = out.vec_state;

  if(in_n_slices == 1)
    {
    coot_rt_t::eop_mat(threeway_kernel_id::equ_array_mul_array,
                       out.get_dev_mem(false), out.get_dev_mem(false), in.get_dev_mem(false),
                       in_n_rows, in_n_cols,
                       0, 0, out.n_rows,
                       0, 0, out.n_rows,
                       in.aux_row1, in.aux_col1, in.m.n_rows);
    }
  else
    {
    if (out_vec_state == 0)
      {
      // out is a matrix
      if (in_n_cols == 1)
        {
        coot_rt_t::eop_cube(twoway_kernel_id::equ_array_mul_array_cube,
                            out.get_dev_mem(false), in.get_dev_mem(false),
                            in_n_rows, in_n_cols, in_n_slices,
                            0, 0, 0, out.n_rows, 1,
                            in.aux_row1, in.aux_col1, in.aux_slice1, in.m.n_rows, in.m.n_cols);
        }
      else if (in_n_rows == 1)
        {
        coot_rt_t::eop_cube(twoway_kernel_id::equ_array_mul_array_cube,
                            out.get_dev_mem(false), in.get_dev_mem(false),
                            in_n_rows, in_n_cols, in_n_slices,
                            0, 0, 0, 1, out.n_rows,
                            in.aux_row1, in.aux_col1, in.aux_slice1, in.m.n_rows, in.m.n_cols);
        }
      }
    else
      {
      // out is a vector
      coot_rt_t::eop_cube(twoway_kernel_id::equ_array_mul_array_cube,
                          out.get_dev_mem(false), in.get_dev_mem(false),
                          1, 1, in_n_slices,
                          0, 0, 0, 1, 1,
                          in.aux_row1, in.aux_col1, in.aux_slice1, in.m.n_rows, in.m.n_cols);
      }
    }
  }



template<typename eT>
inline
void
subview_cube<eT>::div_inplace(Mat<eT>& out, const subview_cube<eT>& in)
  {
  coot_extra_debug_sigprint();

  coot_debug_assert_cube_as_mat(out, in, "division", true);

  const uword in_n_rows   = in.n_rows;
  const uword in_n_cols   = in.n_cols;
  const uword in_n_slices = in.n_slices;

  const uword out_vec_state = out.vec_state;

  if(in_n_slices == 1)
    {
    coot_rt_t::eop_mat(threeway_kernel_id::equ_array_div_array,
                       out.get_dev_mem(false), out.get_dev_mem(false), in.get_dev_mem(false),
                       in_n_rows, in_n_cols,
                       0, 0, out.n_rows,
                       0, 0, out.n_rows,
                       in.aux_row1, in.aux_col1, in.m.n_rows);
    }
  else
    {
    if (out_vec_state == 0)
      {
      // out is a matrix
      if (in_n_cols == 1)
        {
        coot_rt_t::eop_cube(twoway_kernel_id::equ_array_div_array_cube,
                            out.get_dev_mem(false), in.get_dev_mem(false),
                            in_n_rows, in_n_cols, in_n_slices,
                            0, 0, 0, out.n_rows, 1,
                            in.aux_row1, in.aux_col1, in.aux_slice1, in.m.n_rows, in.m.n_cols);
        }
      else if (in_n_rows == 1)
        {
        coot_rt_t::eop_cube(twoway_kernel_id::equ_array_div_array_cube,
                            out.get_dev_mem(false), in.get_dev_mem(false),
                            in_n_rows, in_n_cols, in_n_slices,
                            0, 0, 0, 1, out.n_rows,
                            in.aux_row1, in.aux_col1, in.aux_slice1, in.m.n_rows, in.m.n_cols);
        }
      }
    else
      {
      // out is a vector
      coot_rt_t::eop_cube(twoway_kernel_id::equ_array_div_array_cube,
                          out.get_dev_mem(false), in.get_dev_mem(false),
                          1, 1, in_n_slices,
                          0, 0, 0, 1, 1,
                          in.aux_row1, in.aux_col1, in.aux_slice1, in.m.n_rows, in.m.n_cols);
      }
    }
  }



template<typename eT>
inline
void
subview_cube<eT>::clamp(const eT min_val, const eT max_val)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (min_val > max_val), "clamp(): min_val must be less than max_val" );

  // iterate over slices individually (this is not the best possible approach!)
  for (uword s = 0; s < n_slices; ++s)
    {
    coot_rt_t::clamp(m.dev_mem, m.dev_mem,
                     min_val, max_val,
                     n_rows, n_cols,
                     (aux_slice1 + s) * m.n_elem_slice + aux_row1, aux_col1, m.n_rows,
                     (aux_slice1 + s) * m.n_elem_slice + aux_row1, aux_col1, m.n_rows);
    }
  }



template<typename eT>
inline
void
subview_cube<eT>::fill(const eT val)
  {
  coot_extra_debug_sigprint();

  // iterate over slices individually (this is not the best possible approach!)
  for (uword s = 0; s < n_slices; ++s)
    {
    coot_rt_t::fill(m.dev_mem,
                    val,
                    n_rows, n_cols,
                    (aux_slice1 + s) * m.n_elem_slice + aux_row1, aux_col1, m.n_rows);
    }
  }



template<typename eT>
inline
void
subview_cube<eT>::zeros()
  {
  coot_extra_debug_sigprint();

  fill(eT(0));
  }



template<typename eT>
inline
void
subview_cube<eT>::ones()
  {
  coot_extra_debug_sigprint();

  fill(eT(1));
  }



template<typename eT>
inline
MatValProxy<eT>
subview_cube<eT>::operator[](const uword i)
  {
  const uword in_slice = i / n_elem_slice;
  const uword offset   = in_slice * n_elem_slice;
  const uword j        = i - offset;

  const uword in_col   = j / n_rows;
  const uword in_row   = j % n_rows;

  const uword index = (in_slice + aux_slice1)*m.n_elem_slice + (in_col + aux_col1)*m.n_rows + aux_row1 + in_row;

  return MatValProxy<eT>(access::rw(this->m), index);
  }



template<typename eT>
inline
eT
subview_cube<eT>::operator[](const uword i) const
  {
  const uword in_slice = i / n_elem_slice;
  const uword offset   = in_slice * n_elem_slice;
  const uword j        = i - offset;

  const uword in_col   = j / n_rows;
  const uword in_row   = j % n_rows;

  const uword index = (in_slice + aux_slice1)*m.n_elem_slice + (in_col + aux_col1)*m.n_rows + aux_row1 + in_row;

  return MatValProxy<eT>::get_val(this->m, index);
  }



template<typename eT>
inline
MatValProxy<eT>
subview_cube<eT>::operator()(const uword i)
  {
  coot_debug_check( (i >= n_elem), "subview_cube::operator(): index out of bounds" );

  const uword in_slice = i / n_elem_slice;
  const uword offset   = in_slice * n_elem_slice;
  const uword j        = i - offset;

  const uword in_col   = j / n_rows;
  const uword in_row   = j % n_rows;

  const uword index = (in_slice + aux_slice1)*m.n_elem_slice + (in_col + aux_col1)*m.n_rows + aux_row1 + in_row;

  return MatValProxy<eT>(access::rw(this->m), index);
  }



template<typename eT>
inline
eT
subview_cube<eT>::operator()(const uword i) const
  {
  coot_debug_check( (i >= n_elem), "subview_cube::operator(): index out of bounds" );

  const uword in_slice = i / n_elem_slice;
  const uword offset   = in_slice * n_elem_slice;
  const uword j        = i - offset;

  const uword in_col   = j / n_rows;
  const uword in_row   = j % n_rows;

  const uword index = (in_slice + aux_slice1)*m.n_elem_slice + (in_col + aux_col1)*m.n_rows + aux_row1 + in_row;

  return MatValProxy<eT>::get_val(this->m, index);
  }



template<typename eT>
coot_inline
MatValProxy<eT>
subview_cube<eT>::operator()(const uword in_row, const uword in_col, const uword in_slice)
  {
  coot_debug_check( ((in_row >= n_rows) || (in_col >= n_cols) || (in_slice >= n_slices)), "subview_cube::operator(): index out of bounds" );

  const uword index = (in_slice + aux_slice1)*m.n_elem_slice + (in_col + aux_col1)*m.n_rows + aux_row1 + in_row;

  return MatValProxy<eT>(access::rw(this->m), index);
  }



template<typename eT>
coot_inline
eT
subview_cube<eT>::operator()(const uword in_row, const uword in_col, const uword in_slice) const
  {
  coot_debug_check( ((in_row >= n_rows) || (in_col >= n_cols) || (in_slice >= n_slices)), "subview_cube::operator(): index out of bounds" );

  const uword index = (in_slice + aux_slice1)*m.n_elem_slice + (in_col + aux_col1)*m.n_rows + aux_row1 + in_row;

  return MatValProxy<eT>::get_val(this->m, index);
  }



template<typename eT>
coot_inline
MatValProxy<eT>
subview_cube<eT>::at(const uword in_row, const uword in_col, const uword in_slice)
  {
  coot_debug_check( ((in_row >= n_rows) || (in_col >= n_cols) || (in_slice >= n_slices)), "subview_cube::operator(): index out of bounds" );

  const uword index = (in_slice + aux_slice1)*m.n_elem_slice + (in_col + aux_col1)*m.n_rows + aux_row1 + in_row;

  return MatValProxy<eT>(access::rw(this->m), index);
  }



template<typename eT>
coot_inline
eT
subview_cube<eT>::at(const uword in_row, const uword in_col, const uword in_slice) const
  {
  coot_debug_check( ((in_row >= n_rows) || (in_col >= n_cols) || (in_slice >= n_slices)), "subview_cube::operator(): index out of bounds" );

  const uword index = (in_slice + aux_slice1)*m.n_elem_slice + (in_col + aux_col1)*m.n_rows + aux_row1 + in_row;

  return MatValProxy<eT>::get_val(this->m, index);
  }



template<typename eT>
inline
eT
subview_cube<eT>::front() const
  {
  coot_debug_check( (n_elem == 0), "subview_cube::front(): matrix is empty" );

  return m.at(aux_row1, aux_col1, aux_slice1);
  }



template<typename eT>
inline
eT
subview_cube<eT>::back() const
  {
  coot_debug_check( (n_elem == 0), "subview:_cube:back(): matrix is empty" );

  return m.at(aux_row1 + n_rows - 1, aux_col1 + n_cols - 1, aux_slice1 + n_slices - 1);
  }



template<typename eT>
coot_inline
dev_mem_t<eT>
subview_cube<eT>::slice_get_dev_mem(const uword in_slice, const uword in_col)
  {
  return m.get_dev_mem(false) + (aux_row1 + (aux_col1 + in_col) * m.n_rows + (aux_slice1 + in_slice) * m.n_elem_slice);
  }



template<typename eT>
coot_inline
const dev_mem_t<eT>
subview_cube<eT>::slice_get_dev_mem(const uword in_slice, const uword in_col) const
  {
  return m.get_dev_mem(false) + (aux_row1 + (aux_col1 + in_col) * m.n_rows + (aux_slice1 + in_slice) * m.n_elem_slice);
  }



template<typename eT>
template<typename eT2>
inline
bool
subview_cube<eT>::check_overlap(const subview_cube<eT2>& x) const
  {
  return is_alias(*this, x);
  }



template<typename eT>
inline
bool
subview_cube<eT>::check_overlap(const Mat<eT>& x) const
  {
  return is_alias(*this, x);
  }
