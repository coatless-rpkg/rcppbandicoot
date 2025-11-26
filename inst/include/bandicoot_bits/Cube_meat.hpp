// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
// Copyright 2023      Marcus Edel (http://www.kurg.org)
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
Cube<eT>::~Cube()
  {
  coot_extra_debug_sigprint_this(this);

  coot_rt_t::synchronise();

  cleanup();

  delete_mat();

  coot_type_check(( is_supported_elem_type<eT>::value == false ));
  }



template<typename eT>
inline
Cube<eT>::Cube()
  : n_rows(0)
  , n_cols(0)
  , n_elem_slice(0)
  , n_slices(0)
  , n_elem(0)
  , mem_state(0)
  , dev_mem({{ NULL, 0 }})
  , mat_ptrs(nullptr)
  {
  coot_extra_debug_sigprint_this(this);
  }



//! construct the cube to have user specified dimensions
template<typename eT>
inline
Cube<eT>::Cube(const uword in_n_rows, const uword in_n_cols, const uword in_n_slices)
  : n_rows(0)
  , n_cols(0)
  , n_elem_slice(0)
  , n_slices(0)
  , n_elem(0)
  , mem_state(0)
  , dev_mem({{ NULL, 0 }})
  , mat_ptrs(nullptr)
  {
  coot_extra_debug_sigprint_this(this);

  init(in_n_rows, in_n_cols, in_n_slices);

  zeros(); // fill with zeros by default
  }



template<typename eT>
inline
Cube<eT>::Cube(const SizeCube& s)
  : n_rows(0)
  , n_cols(0)
  , n_elem_slice(0)
  , n_slices(0)
  , n_elem(0)
  , mem_state(0)
  , dev_mem({{ NULL, 0 }})
  , mat_ptrs(nullptr)
  {
  coot_extra_debug_sigprint_this(this);

  init(s.n_rows, s.n_cols, s.n_slices);

  zeros();
  }



/* //! internal use only */
/* template<typename eT> */
/* template<bool do_zeros> */
/* inline */
/* Cube<eT>::Cube(const uword in_n_rows, const uword in_n_cols, const uword in_n_slices, const arma_initmode_indicator<do_zeros>&) */
/*   : n_rows(in_n_rows) */
/*   , n_cols(in_n_cols) */
/*   , n_elem_slice(in_n_rows*in_n_cols) */
/*   , n_slices(in_n_slices) */
/*   , n_elem(in_n_rows*in_n_cols*in_n_slices) */
/*   , mem_state(0) */
/*   , mem() */
/*   , mat_ptrs(nullptr) */
/*   { */
/*   arma_extra_debug_sigprint_this(this); */

/*   init_cold(); */

/*   if(do_zeros) */
/*     { */
/*     arma_extra_debug_print("Cube::constructor: zeroing memory"); */
/*     arrayops::fill_zeros(memptr(), n_elem); */
/*     } */
/*   } */



/* //! internal use only */
/* template<typename eT> */
/* template<bool do_zeros> */
/* inline */
/* Cube<eT>::Cube(const SizeCube& s, const arma_initmode_indicator<do_zeros>&) */
/*   : n_rows(s.n_rows) */
/*   , n_cols(s.n_cols) */
/*   , n_elem_slice(s.n_rows*s.n_cols) */
/*   , n_slices(s.n_slices) */
/*   , n_elem(s.n_rows*s.n_cols*s.n_slices) */
/*   , mem_state(0) */
/*   , mem() */
/*   , mat_ptrs(nullptr) */
/*   { */
/*   arma_extra_debug_sigprint_this(this); */

/*   init_cold(); */

/*   if(do_zeros) */
/*     { */
/*     arma_extra_debug_print("Cube::constructor: zeroing memory"); */
/*     arrayops::fill_zeros(memptr(), n_elem); */
/*     } */
/*   } */



//! construct the cube to have user specified dimensions and fill with specified pattern
template<typename eT>
template<typename fill_type>
inline
Cube<eT>::Cube(const uword in_n_rows, const uword in_n_cols, const uword in_n_slices, const fill::fill_class<fill_type>& f)
  : n_rows(0)
  , n_cols(0)
  , n_elem_slice(0)
  , n_slices(0)
  , n_elem(0)
  , mem_state(0)
  , dev_mem({{ NULL, 0 }})
  , mat_ptrs(nullptr)
  {
  coot_extra_debug_sigprint_this(this);
  coot_ignore(f);

  init(in_n_rows, in_n_cols, in_n_slices);

  if(is_same_type<fill_type, fill::fill_zeros>::yes)  { (*this).zeros(); }
  if(is_same_type<fill_type, fill::fill_ones >::yes)  { (*this).ones();  }
  if(is_same_type<fill_type, fill::fill_randu>::yes)  { (*this).randu(); }
  if(is_same_type<fill_type, fill::fill_randn>::yes)  { (*this).randn(); }

  coot_static_check( (is_same_type<fill_type, fill::fill_eye>::yes), "Cube::Cube(): unsupported fill type" );
  }



template<typename eT>
template<typename fill_type>
inline
Cube<eT>::Cube(const SizeCube& s, const fill::fill_class<fill_type>& f)
  : n_rows(0)
  , n_cols(0)
  , n_elem_slice(0)
  , n_slices(0)
  , n_elem(0)
  , mem_state(0)
  , dev_mem({{ NULL, 0 }})
  , mat_ptrs(nullptr)
  {
  coot_extra_debug_sigprint_this(this);
  coot_ignore(f);

  init(s.n_rows, s.n_cols, s.n_slices);

  if(is_same_type<fill_type, fill::fill_zeros>::yes)  { (*this).zeros(); }
  if(is_same_type<fill_type, fill::fill_ones >::yes)  { (*this).ones();  }
  if(is_same_type<fill_type, fill::fill_randu>::yes)  { (*this).randu(); }
  if(is_same_type<fill_type, fill::fill_randn>::yes)  { (*this).randn(); }

  coot_static_check( (is_same_type<fill_type, fill::fill_eye>::yes), "Cube::Cube(): unsupported fill type" );
  }



/* //! construct the cube to have user specified dimensions and fill with specified value */
/* template<typename eT> */
/* inline */
/* Cube<eT>::Cube(const uword in_n_rows, const uword in_n_cols, const uword in_n_slices, const fill::scalar_holder<eT> f) */
/*   : n_rows(in_n_rows) */
/*   , n_cols(in_n_cols) */
/*   , n_elem_slice(in_n_rows*in_n_cols) */
/*   , n_slices(in_n_slices) */
/*   , n_elem(in_n_rows*in_n_cols*in_n_slices) */
/*   , mem_state(0) */
/*   , mem() */
/*   , mat_ptrs(nullptr) */
/*   { */
/*   arma_extra_debug_sigprint_this(this); */

/*   init_cold(); */

/*   (*this).fill(f.scalar); */
/*   } */



/* template<typename eT> */
/* inline */
/* Cube<eT>::Cube(const SizeCube& s, const fill::scalar_holder<eT> f) */
/*   : n_rows(s.n_rows) */
/*   , n_cols(s.n_cols) */
/*   , n_elem_slice(s.n_rows*s.n_cols) */
/*   , n_slices(s.n_slices) */
/*   , n_elem(s.n_rows*s.n_cols*s.n_slices) */
/*   , mem_state(0) */
/*   , mem() */
/*   , mat_ptrs(nullptr) */
/*   { */
/*   arma_extra_debug_sigprint_this(this); */

/*   init_cold(); */

/*   (*this).fill(f.scalar); */
/*   } */



template<typename eT>
inline
Cube<eT>::Cube(Cube<eT>&& in_cube)
  : n_rows(0)
  , n_cols(0)
  , n_elem_slice(0)
  , n_slices(0)
  , n_elem(0)
  , mem_state(0)
  , dev_mem({{ NULL, 0 }})
  , mat_ptrs(nullptr)
  {
  coot_extra_debug_sigprint_this(this);

  (*this).steal_mem(in_cube);
  }



template<typename eT>
inline
Cube<eT>&
Cube<eT>::operator=(Cube<eT>&& in_cube)
  {
  coot_extra_debug_sigprint();

  (*this).steal_mem(in_cube);

  return *this;
  }



template<typename eT>
inline
void
Cube<eT>::cleanup()
  {
  coot_extra_debug_sigprint();

  if ((dev_mem.cl_mem_ptr.ptr != NULL) && (mem_state == 0) && (n_elem > 0))
    {
    get_rt().release_memory(dev_mem);
    }

  dev_mem.cl_mem_ptr.ptr = NULL; // for paranoia
  dev_mem.cl_mem_ptr.offset = 0;
  }



template<typename eT>
inline
void
Cube<eT>::init(const uword new_n_rows, const uword new_n_cols, const uword new_n_slices)
  {
  if( (n_rows == new_n_rows) && (n_cols == new_n_cols) && (n_slices == new_n_slices) ) { return; }

  // ensure that n_elem can hold the result of (n_rows * n_cols * n_slices)
  coot_debug_check(
    ( ( (new_n_rows > 0x0FFF) || (new_n_cols > 0x0FFF) || (new_n_slices > 0xFF) )
      ? ( (double(new_n_rows) * double(new_n_cols) * double(new_n_slices)) > double(Datum<uword>::max) )
      : false
    ),
    "Cube::init(): requested size is too large"
    );

  const uword old_n_elem = n_elem;
  const uword new_n_elem = new_n_rows * new_n_cols * new_n_slices;

  if (old_n_elem == new_n_elem)
    {
    coot_extra_debug_print("Cube::init(): reusing memory");

    delete_mat();

    access::rw(n_rows)       = new_n_rows;
    access::rw(n_cols)       = new_n_cols;
    access::rw(n_elem_slice) = new_n_rows * new_n_cols;
    access::rw(n_slices)     = new_n_slices;

    create_mat();

    return;
    }

  delete_mat();

  if (new_n_elem == 0)
    {
    coot_extra_debug_print("Cube::init(): releasing memory");
    cleanup();
    }
  else if (new_n_elem < old_n_elem)
    {
    coot_extra_debug_print("Cube::init(): reusing memory");
    }
  else
    {
    if (old_n_elem > 0)
      {
      coot_extra_debug_print("Cube::init(): releasing memory");
      cleanup();
      }

    coot_extra_debug_print("Cube::init(): acquiring memory");
    dev_mem = get_rt().acquire_memory<eT>(new_n_elem);
    }

  access::rw(n_rows)       = new_n_rows;
  access::rw(n_cols)       = new_n_cols;
  access::rw(n_elem_slice) = new_n_rows * new_n_cols;
  access::rw(n_slices)     = new_n_slices;
  access::rw(n_elem)       = new_n_elem;
  access::rw(mem_state)    = 0;

  create_mat();
  }



template<typename eT>
inline
void
Cube<eT>::delete_mat()
  {
  coot_extra_debug_sigprint();

  if((n_slices > 0) && (mat_ptrs != nullptr))
    {
    for(uword uslice = 0; uslice < n_slices; ++uslice)
      {
      if(mat_ptrs[uslice] != nullptr)  { delete access::rw(mat_ptrs[uslice]); }
      }

    if( n_slices > Cube_prealloc::mat_ptrs_size )
      {
      delete [] mat_ptrs;
      access::rw(mat_ptrs) = nullptr;
      }
    }
  }



template<typename eT>
inline
void
Cube<eT>::create_mat()
  {
  coot_extra_debug_sigprint();

  if(n_slices == 0)
    {
    access::rw(mat_ptrs) = nullptr;
    }
  else
    {
    if(mem_state <= 2)
      {
      if(n_slices <= Cube_prealloc::mat_ptrs_size)
        {
        access::rw(mat_ptrs) = const_cast< Mat<eT>** >(mat_ptrs_local);
        }
      else
        {
        access::rw(mat_ptrs) = new(std::nothrow) Mat<eT>*[n_slices];

        coot_check_bad_alloc( (mat_ptrs == nullptr), "Cube::create_mat(): out of memory" );
        }
      }

    for(uword uslice = 0; uslice < n_slices; ++uslice)
      {
      mat_ptrs[uslice] = nullptr;
      }
    }
  }



// Set the cube to be equal to the specified scalar.
// NOTE: the size of the cube will be 1x1x1
template<typename eT>
inline
Cube<eT>&
Cube<eT>::operator=(const eT val)
  {
  coot_extra_debug_sigprint();

  init(1, 1, 1);

  coot_rt_t::fill(dev_mem, val, n_elem_slice, n_slices, 0, 0, n_elem_slice);

  return *this;
  }



//! In-place addition of a scalar to all elements of the cube
template<typename eT>
inline
Cube<eT>&
Cube<eT>::operator+=(const eT val)
  {
  coot_extra_debug_sigprint();

  coot_rt_t::eop_scalar(twoway_kernel_id::equ_array_plus_scalar,
                        dev_mem, dev_mem,
                        val, (eT) 0,
                        n_rows, n_cols, n_slices,
                        0, 0, 0, n_rows, n_cols,
                        0, 0, 0, n_rows, n_cols);

  return *this;
  }



//! In-place subtraction of a scalar from all elements of the cube
template<typename eT>
inline
Cube<eT>&
Cube<eT>::operator-=(const eT val)
  {
  coot_extra_debug_sigprint();

  coot_rt_t::eop_scalar(twoway_kernel_id::equ_array_minus_scalar_post,
                        dev_mem, dev_mem,
                        val, (eT) 0,
                        n_rows, n_cols, n_slices,
                        0, 0, 0, n_rows, n_cols,
                        0, 0, 0, n_rows, n_cols);

  return *this;
  }



//! In-place multiplication of all elements of the cube with a scalar
template<typename eT>
inline
Cube<eT>&
Cube<eT>::operator*=(const eT val)
  {
  coot_extra_debug_sigprint();

  coot_rt_t::eop_scalar(twoway_kernel_id::equ_array_mul_scalar,
                        dev_mem, dev_mem,
                        val, (eT) 1,
                        n_rows, n_cols, n_slices,
                        0, 0, 0, n_rows, n_cols,
                        0, 0, 0, n_rows, n_cols);

  return *this;
  }



//! In-place division of all elements of the cube with a scalar
template<typename eT>
inline
Cube<eT>&
Cube<eT>::operator/=(const eT val)
  {
  coot_extra_debug_sigprint();

  coot_rt_t::eop_scalar(twoway_kernel_id::equ_array_div_scalar_post,
                        dev_mem, dev_mem,
                        val, (eT) 1,
                        n_rows, n_cols, n_slices,
                        0, 0, 0, n_rows, n_cols,
                        0, 0, 0, n_rows, n_cols);

  return *this;
  }



//! construct a cube from a given cube
template<typename eT>
inline
Cube<eT>::Cube(const Cube<eT>& x)
  : n_rows(0)
  , n_cols(0)
  , n_elem_slice(0)
  , n_slices(0)
  , n_elem(0)
  , mem_state(0)
  , dev_mem({{ NULL, 0 }})
  , mat_ptrs(nullptr)
  {
  coot_extra_debug_sigprint_this(this);

  (*this).operator=(x);
  }



//! construct a cube from a given cube
template<typename eT>
inline
Cube<eT>&
Cube<eT>::operator=(const Cube<eT>& x)
  {
  coot_extra_debug_sigprint();

  if(this != &x)
    {
    (*this).set_size(x.n_rows, x.n_cols, x.n_slices);

    coot_rt_t::copy_mat(dev_mem, x.dev_mem,
                        n_elem_slice, n_slices,
                        0, 0, n_elem_slice,
                        0, 0, x.n_elem_slice);
    }

  return *this;
  }



template<typename eT>
inline
Cube<eT>::Cube(dev_mem_t<eT> aux_dev_mem, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices)
  : n_rows(in_n_rows)
  , n_cols(in_n_cols)
  , n_elem_slice(in_n_rows * in_n_cols)
  , n_slices(in_n_slices)
  , n_elem(in_n_rows * in_n_cols * in_n_slices)
  , mem_state(1)
  , dev_mem(aux_dev_mem)
  , mat_ptrs(nullptr)
  {
  coot_extra_debug_sigprint_this(this);

  create_mat();
  }



template<typename eT>
inline
Cube<eT>::Cube(cl_mem aux_dev_mem, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices)
  : n_rows(in_n_rows)
  , n_cols(in_n_cols)
  , n_elem_slice(in_n_rows * in_n_cols)
  , n_slices(in_n_slices)
  , n_elem(in_n_rows * in_n_cols * in_n_slices)
  , mem_state(1)
  , dev_mem({{ NULL, 0 }})
  , mat_ptrs(nullptr)
  {
  coot_extra_debug_sigprint_this(this);

  this->dev_mem.cl_mem_ptr.ptr = aux_dev_mem;
  this->dev_mem.cl_mem_ptr.offset = 0;

  coot_debug_check( get_rt().backend != CL_BACKEND, "Cube(): cannot wrap OpenCL memory when not using OpenCL backend");

  create_mat();
  }



template<typename eT>
inline
Cube<eT>::Cube(typename cuda_type<eT>::type* aux_dev_mem, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices)
  : n_rows(in_n_rows)
  , n_cols(in_n_cols)
  , n_elem_slice(in_n_rows * in_n_cols)
  , n_slices(in_n_slices)
  , n_elem(in_n_rows * in_n_cols * in_n_slices)
  , mem_state(1)
  , dev_mem({{ NULL, 0 }})
  , mat_ptrs(nullptr)
  {
  coot_extra_debug_sigprint_this(this);

  this->dev_mem.cuda_mem_ptr = aux_dev_mem;

  coot_debug_check( get_rt().backend != CUDA_BACKEND, "Cube(): cannot wrap CUDA memory when not using CUDA backend");
  }



//! in-place cube addition
template<typename eT>
inline
Cube<eT>&
Cube<eT>::operator+=(const Cube<eT>& m)
  {
  coot_extra_debug_sigprint();

  coot_assert_same_size((*this), m, "element-wise addition");

  coot_rt_t::eop_mat(threeway_kernel_id::equ_array_plus_array,
                     dev_mem, dev_mem, m.dev_mem,
                     n_elem_slice, n_slices,
                     0, 0, n_elem_slice,
                     0, 0, n_elem_slice,
                     0, 0, m.n_elem_slice);

  return *this;
  }



//! in-place cube subtraction
template<typename eT>
inline
Cube<eT>&
Cube<eT>::operator-=(const Cube<eT>& m)
  {
  coot_extra_debug_sigprint();

  coot_assert_same_size(*this, m, "element-wise subtraction");

  coot_rt_t::eop_mat(threeway_kernel_id::equ_array_minus_array,
                     dev_mem, dev_mem, m.dev_mem,
                     n_elem_slice, n_slices,
                     0, 0, n_elem_slice,
                     0, 0, n_elem_slice,
                     0, 0, m.n_elem_slice);

  return *this;
  }



//! in-place element-wise cube multiplication
template<typename eT>
inline
Cube<eT>&
Cube<eT>::operator%=(const Cube<eT>& m)
  {
  coot_extra_debug_sigprint();

  coot_assert_same_size(*this, m, "element-wise multiplication");

  coot_rt_t::eop_mat(threeway_kernel_id::equ_array_mul_array,
                     dev_mem, dev_mem, m.dev_mem,
                     n_elem_slice, n_slices,
                     0, 0, n_elem_slice,
                     0, 0, n_elem_slice,
                     0, 0, m.n_elem_slice);

  return *this;
  }



//! in-place element-wise cube division
template<typename eT>
inline
Cube<eT>&
Cube<eT>::operator/=(const Cube<eT>& m)
  {
  coot_extra_debug_sigprint();

  coot_assert_same_size(*this, m, "element-wise division");

  coot_rt_t::eop_mat(threeway_kernel_id::equ_array_div_array,
                     dev_mem, dev_mem, m.dev_mem,
                     n_elem_slice, n_slices,
                     0, 0, n_elem_slice,
                     0, 0, n_elem_slice,
                     0, 0, m.n_elem_slice);

  return *this;
  }



/* //! for constructing a complex cube out of two non-complex cubes */
/* template<typename eT> */
/* template<typename T1, typename T2> */
/* inline */
/* Cube<eT>::Cube */
/*   ( */
/*   const BaseCube<typename Cube<eT>::pod_type,T1>& A, */
/*   const BaseCube<typename Cube<eT>::pod_type,T2>& B */
/*   ) */
/*   : n_rows(0) */
/*   , n_cols(0) */
/*   , n_elem_slice(0) */
/*   , n_slices(0) */
/*   , n_elem(0) */
/*   , mem_state(0) */
/*   , mem() */
/*   , mat_ptrs(nullptr) */
/*   { */
/*   arma_extra_debug_sigprint_this(this); */

/*   init(A,B); */
/*   } */



//! construct a cube from a subview_cube instance (eg. construct a cube from a delayed subcube operation)
template<typename eT>
inline
Cube<eT>::Cube(const subview_cube<eT>& X)
  : n_rows(0)
  , n_cols(0)
  , n_elem_slice(0)
  , n_slices(0)
  , n_elem(0)
  , mem_state(0)
  , dev_mem({{ NULL, 0 }})
  , mat_ptrs(nullptr)
  {
  coot_extra_debug_sigprint_this(this);

  set_size(X.n_rows, X.n_cols, X.n_slices);

  subview_cube<eT>::extract(*this, X);
  }



//! construct a cube from a subview_cube instance (eg. construct a cube from a delayed subcube operation)
template<typename eT>
inline
Cube<eT>&
Cube<eT>::operator=(const subview_cube<eT>& X)
  {
  coot_extra_debug_sigprint();

  const bool alias = is_alias(*this, X.m);

  if(alias == false)
    {
    set_size(X.n_rows, X.n_cols, X.n_slices);

    subview_cube<eT>::extract(*this, X);
    }
  else
    {
    Cube<eT> tmp(X);

    steal_mem(tmp);
    }

  return *this;
  }



//! in-place cube addition (using a subcube on the right-hand-side)
template<typename eT>
inline
Cube<eT>&
Cube<eT>::operator+=(const subview_cube<eT>& X)
  {
  coot_extra_debug_sigprint();

  subview_cube<eT>::plus_inplace(*this, X);

  return *this;
  }



//! in-place cube subtraction (using a subcube on the right-hand-side)
template<typename eT>
inline
Cube<eT>&
Cube<eT>::operator-=(const subview_cube<eT>& X)
  {
  coot_extra_debug_sigprint();

  subview_cube<eT>::minus_inplace(*this, X);

  return *this;
  }



//! in-place element-wise cube mutiplication (using a subcube on the right-hand-side) */
template<typename eT>
inline
Cube<eT>&
Cube<eT>::operator%=(const subview_cube<eT>& X)
  {
  coot_extra_debug_sigprint();

  subview_cube<eT>::schur_inplace(*this, X);

  return *this;
  }



//! in-place element-wise cube division (using a subcube on the right-hand-side)
template<typename eT>
inline
Cube<eT>&
Cube<eT>::operator/=(const subview_cube<eT>& X)
  {
  coot_extra_debug_sigprint();

  subview_cube<eT>::div_inplace(*this, X);

  return *this;
  }



/* template<typename eT> */
/* template<typename T1> */
/* inline */
/* Cube<eT>::Cube(const subview_cube_slices<eT,T1>& X) */
/*   : n_rows(0) */
/*   , n_cols(0) */
/*   , n_elem_slice(0) */
/*   , n_slices(0) */
/*   , n_elem(0) */
/*   , mem_state(0) */
/*   , mem() */
/*   , mat_ptrs(nullptr) */
/*   { */
/*   arma_extra_debug_sigprint_this(this); */

/*   subview_cube_slices<eT,T1>::extract(*this, X); */
/*   } */



/* template<typename eT> */
/* template<typename T1> */
/* inline */
/* Cube<eT>& */
/* Cube<eT>::operator=(const subview_cube_slices<eT,T1>& X) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   const bool alias = (this == &(X.m)); */

/*   if(alias == false) */
/*     { */
/*     subview_cube_slices<eT,T1>::extract(*this, X); */
/*     } */
/*   else */
/*     { */
/*     Cube<eT> tmp(X); */

/*     steal_mem(tmp); */
/*     } */

/*   return *this; */
/*   } */



/* template<typename eT> */
/* template<typename T1> */
/* inline */
/* Cube<eT>& */
/* Cube<eT>::operator+=(const subview_cube_slices<eT,T1>& X) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   subview_cube_slices<eT,T1>::plus_inplace(*this, X); */

/*   return *this; */
/*   } */



/* template<typename eT> */
/* template<typename T1> */
/* inline */
/* Cube<eT>& */
/* Cube<eT>::operator-=(const subview_cube_slices<eT,T1>& X) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   subview_cube_slices<eT,T1>::minus_inplace(*this, X); */

/*   return *this; */
/*   } */



/* template<typename eT> */
/* template<typename T1> */
/* inline */
/* Cube<eT>& */
/* Cube<eT>::operator%=(const subview_cube_slices<eT,T1>& X) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   subview_cube_slices<eT,T1>::schur_inplace(*this, X); */

/*   return *this; */
/*   } */



/* template<typename eT> */
/* template<typename T1> */
/* inline */
/* Cube<eT>& */
/* Cube<eT>::operator/=(const subview_cube_slices<eT,T1>& X) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   subview_cube_slices<eT,T1>::div_inplace(*this, X); */

/*   return *this; */
/*   } */



//! creation of subview_cube (subcube comprised of specified row)
template<typename eT>
coot_inline
subview_cube<eT>
Cube<eT>::row(const uword in_row)
  {
  coot_extra_debug_sigprint();

  coot_debug_check_bounds( (in_row >= n_rows), "Cube::row(): index out of bounds" );

  return (*this).rows(in_row, in_row);
  }



//! creation of subview_cube (subcube comprised of specified row)
template<typename eT>
coot_inline
const subview_cube<eT>
Cube<eT>::row(const uword in_row) const
  {
  coot_extra_debug_sigprint();

  coot_debug_check_bounds( (in_row >= n_rows), "Cube::row(): index out of bounds" );

  return (*this).rows(in_row, in_row);
  }



//! creation of subview_cube (subcube comprised of specified column)
template<typename eT>
coot_inline
subview_cube<eT>
Cube<eT>::col(const uword in_col)
  {
  coot_extra_debug_sigprint();

  coot_debug_check_bounds( (in_col >= n_cols), "Cube::col(): index out of bounds" );

  return (*this).cols(in_col, in_col);
  }



//! creation of subview_cube (subcube comprised of specified column)
template<typename eT>
coot_inline
const subview_cube<eT>
Cube<eT>::col(const uword in_col) const
  {
  coot_extra_debug_sigprint();

  coot_debug_check_bounds( (in_col >= n_cols), "Cube::col(): index out of bounds" );

  return (*this).cols(in_col, in_col);
  }



//! provide the reference to the matrix representing a single slice
template<typename eT>
inline
Mat<eT>&
Cube<eT>::slice(const uword in_slice)
  {
  coot_extra_debug_sigprint();

  coot_debug_check_bounds( (in_slice >= n_slices), "Cube::slice(): index out of bounds" );

  if(mat_ptrs[in_slice] == nullptr)
    {
    const dev_mem_t<eT> ptr = (n_elem_slice > 0) ? slice_get_dev_mem(in_slice, false) : dev_mem_t<eT>({{ NULL, 0 }});

    mat_ptrs[in_slice] = new Mat<eT>('j', ptr, n_rows, n_cols);
    }

  return const_cast< Mat<eT>& >( *(mat_ptrs[in_slice]) );
  }



//! provide the reference to the matrix representing a single slice
template<typename eT>
inline
const Mat<eT>&
Cube<eT>::slice(const uword in_slice) const
  {
  coot_extra_debug_sigprint();

  coot_debug_check_bounds( (in_slice >= n_slices), "Cube::slice(): index out of bounds" );

  if(mat_ptrs[in_slice] == nullptr)
    {
    const dev_mem_t<eT> ptr = (n_elem_slice > 0) ? slice_get_dev_mem(in_slice, false) : dev_mem_t<eT>({{ NULL, 0 }});

    mat_ptrs[in_slice] = new Mat<eT>('j', ptr, n_rows, n_cols);
    }

  return *(mat_ptrs[in_slice]);
  }



//! creation of subview_cube (subcube comprised of specified rows)
template<typename eT>
coot_inline
subview_cube<eT>
Cube<eT>::rows(const uword in_row1, const uword in_row2)
  {
  coot_extra_debug_sigprint();

  coot_debug_check_bounds
    (
    (in_row1 > in_row2) || (in_row2 >= n_rows),
    "Cube::rows(): indices out of bounds or incorrectly used"
    );

  const uword subcube_n_rows = in_row2 - in_row1 + 1;

  return subview_cube<eT>(*this, in_row1, 0, 0, subcube_n_rows, n_cols, n_slices);
  }



//! creation of subview_cube (subcube comprised of specified rows)
template<typename eT>
coot_inline
const subview_cube<eT>
Cube<eT>::rows(const uword in_row1, const uword in_row2) const
  {
  coot_extra_debug_sigprint();

  coot_debug_check_bounds
    (
    (in_row1 > in_row2) || (in_row2 >= n_rows),
    "Cube::rows(): indices out of bounds or incorrectly used"
    );

  const uword subcube_n_rows = in_row2 - in_row1 + 1;

  return subview_cube<eT>(*this, in_row1, 0, 0, subcube_n_rows, n_cols, n_slices);
  }



//! creation of subview_cube (subcube comprised of specified columns)
template<typename eT>
coot_inline
subview_cube<eT>
Cube<eT>::cols(const uword in_col1, const uword in_col2)
  {
  coot_extra_debug_sigprint();

  coot_debug_check_bounds
    (
    (in_col1 > in_col2) || (in_col2 >= n_cols),
    "Cube::cols(): indices out of bounds or incorrectly used"
    );

  const uword subcube_n_cols = in_col2 - in_col1 + 1;

  return subview_cube<eT>(*this, 0, in_col1, 0, n_rows, subcube_n_cols, n_slices);
  }



//! creation of subview_cube (subcube comprised of specified columns)
template<typename eT>
coot_inline
const subview_cube<eT>
Cube<eT>::cols(const uword in_col1, const uword in_col2) const
  {
  coot_extra_debug_sigprint();

  coot_debug_check_bounds
    (
    (in_col1 > in_col2) || (in_col2 >= n_cols),
    "Cube::cols(): indices out of bounds or incorrectly used"
    );

  const uword subcube_n_cols = in_col2 - in_col1 + 1;

  return subview_cube<eT>(*this, 0, in_col1, 0, n_rows, subcube_n_cols, n_slices);
  }



//! creation of subview_cube (subcube comprised of specified slices)
template<typename eT>
coot_inline
subview_cube<eT>
Cube<eT>::slices(const uword in_slice1, const uword in_slice2)
  {
  coot_extra_debug_sigprint();

  coot_debug_check_bounds
    (
    (in_slice1 > in_slice2) || (in_slice2 >= n_slices),
    "Cube::slices(): indices out of bounds or incorrectly used"
    );

  const uword subcube_n_slices = in_slice2 - in_slice1 + 1;

  return subview_cube<eT>(*this, 0, 0, in_slice1, n_rows, n_cols, subcube_n_slices);
  }



//! creation of subview_cube (subcube comprised of specified slices)
template<typename eT>
coot_inline
const subview_cube<eT>
Cube<eT>::slices(const uword in_slice1, const uword in_slice2) const
  {
  coot_extra_debug_sigprint();

  coot_debug_check_bounds
    (
    (in_slice1 > in_slice2) || (in_slice2 >= n_slices),
    "Cube::slices(): indices out of bounds or incorrectly used"
    );

  const uword subcube_n_slices = in_slice2 - in_slice1 + 1;

  return subview_cube<eT>(*this, 0, 0, in_slice1, n_rows, n_cols, subcube_n_slices);
  }



//! creation of subview_cube (generic subcube)
template<typename eT>
coot_inline
subview_cube<eT>
Cube<eT>::subcube(const uword in_row1, const uword in_col1, const uword in_slice1, const uword in_row2, const uword in_col2, const uword in_slice2)
  {
  coot_extra_debug_sigprint();

  coot_debug_check_bounds
    (
    (in_row1 >  in_row2) || (in_col1 >  in_col2) || (in_slice1 >  in_slice2) ||
    (in_row2 >= n_rows)  || (in_col2 >= n_cols)  || (in_slice2 >= n_slices),
    "Cube::subcube(): indices out of bounds or incorrectly used"
    );

  const uword subcube_n_rows   = in_row2   - in_row1   + 1;
  const uword subcube_n_cols   = in_col2   - in_col1   + 1;
  const uword subcube_n_slices = in_slice2 - in_slice1 + 1;

  return subview_cube<eT>(*this, in_row1, in_col1, in_slice1, subcube_n_rows, subcube_n_cols, subcube_n_slices);
  }



//! creation of subview_cube (generic subcube)
template<typename eT>
coot_inline
const subview_cube<eT>
Cube<eT>::subcube(const uword in_row1, const uword in_col1, const uword in_slice1, const uword in_row2, const uword in_col2, const uword in_slice2) const
  {
  coot_extra_debug_sigprint();

  coot_debug_check_bounds
    (
    (in_row1 >  in_row2) || (in_col1 >  in_col2) || (in_slice1 >  in_slice2) ||
    (in_row2 >= n_rows)  || (in_col2 >= n_cols)  || (in_slice2 >= n_slices),
    "Cube::subcube(): indices out of bounds or incorrectly used"
    );

  const uword subcube_n_rows   = in_row2   - in_row1   + 1;
  const uword subcube_n_cols   = in_col2   - in_col1   + 1;
  const uword subcube_n_slices = in_slice2 - in_slice1 + 1;

  return subview_cube<eT>(*this, in_row1, in_col1, in_slice1, subcube_n_rows, subcube_n_cols, subcube_n_slices);
  }



//! creation of subview_cube (generic subcube)
template<typename eT>
inline
subview_cube<eT>
Cube<eT>::subcube(const uword in_row1, const uword in_col1, const uword in_slice1, const SizeCube& s)
  {
  coot_extra_debug_sigprint();

  const uword l_n_rows   = n_rows;
  const uword l_n_cols   = n_cols;
  const uword l_n_slices = n_slices;

  const uword s_n_rows   = s.n_rows;
  const uword s_n_cols   = s.n_cols;
  const uword s_n_slices = s.n_slices;

  coot_debug_check_bounds
    (
       ( in_row1             >= l_n_rows) || ( in_col1             >= l_n_cols) || ( in_slice1               >= l_n_slices)
    || ((in_row1 + s_n_rows) >  l_n_rows) || ((in_col1 + s_n_cols) >  l_n_cols) || ((in_slice1 + s_n_slices) >  l_n_slices),
    "Cube::subcube(): indices or size out of bounds"
    );

  return subview_cube<eT>(*this, in_row1, in_col1, in_slice1, s_n_rows, s_n_cols, s_n_slices);
  }



//! creation of subview_cube (generic subcube)
template<typename eT>
inline
const subview_cube<eT>
Cube<eT>::subcube(const uword in_row1, const uword in_col1, const uword in_slice1, const SizeCube& s) const
  {
  coot_extra_debug_sigprint();

  const uword l_n_rows   = n_rows;
  const uword l_n_cols   = n_cols;
  const uword l_n_slices = n_slices;

  const uword s_n_rows   = s.n_rows;
  const uword s_n_cols   = s.n_cols;
  const uword s_n_slices = s.n_slices;

  coot_debug_check_bounds
    (
       ( in_row1             >= l_n_rows) || ( in_col1             >= l_n_cols) || ( in_slice1               >= l_n_slices)
    || ((in_row1 + s_n_rows) >  l_n_rows) || ((in_col1 + s_n_cols) >  l_n_cols) || ((in_slice1 + s_n_slices) >  l_n_slices),
    "Cube::subcube(): indices or size out of bounds"
    );

  return subview_cube<eT>(*this, in_row1, in_col1, in_slice1, s_n_rows, s_n_cols, s_n_slices);
  }



//! creation of subview_cube (generic subcube) */
template<typename eT>
inline
subview_cube<eT>
Cube<eT>::subcube(const span& row_span, const span& col_span, const span& slice_span)
  {
  coot_extra_debug_sigprint();

  const bool row_all   = row_span.whole;
  const bool col_all   = col_span.whole;
  const bool slice_all = slice_span.whole;

  const uword local_n_rows   = n_rows;
  const uword local_n_cols   = n_cols;
  const uword local_n_slices = n_slices;

  const uword in_row1          = row_all   ? 0              : row_span.a;
  const uword in_row2          =                              row_span.b;
  const uword subcube_n_rows   = row_all   ? local_n_rows   : in_row2 - in_row1 + 1;

  const uword in_col1          = col_all   ? 0              : col_span.a;
  const uword in_col2          =                              col_span.b;
  const uword subcube_n_cols   = col_all   ? local_n_cols   : in_col2 - in_col1 + 1;

  const uword in_slice1        = slice_all ? 0              : slice_span.a;
  const uword in_slice2        =                              slice_span.b;
  const uword subcube_n_slices = slice_all ? local_n_slices : in_slice2 - in_slice1 + 1;

  coot_debug_check_bounds
    (
    ( row_all   ? false : ((in_row1   >  in_row2)   || (in_row2   >= local_n_rows))   )
    ||
    ( col_all   ? false : ((in_col1   >  in_col2)   || (in_col2   >= local_n_cols))   )
    ||
    ( slice_all ? false : ((in_slice1 >  in_slice2) || (in_slice2 >= local_n_slices)) )
    ,
    "Cube::subcube(): indices out of bounds or incorrectly used"
    );

  return subview_cube<eT>(*this, in_row1, in_col1, in_slice1, subcube_n_rows, subcube_n_cols, subcube_n_slices);
  }



//! creation of subview_cube (generic subcube)
template<typename eT>
inline
const subview_cube<eT>
Cube<eT>::subcube(const span& row_span, const span& col_span, const span& slice_span) const
  {
  coot_extra_debug_sigprint();

  const bool row_all   = row_span.whole;
  const bool col_all   = col_span.whole;
  const bool slice_all = slice_span.whole;

  const uword local_n_rows   = n_rows;
  const uword local_n_cols   = n_cols;
  const uword local_n_slices = n_slices;

  const uword in_row1          = row_all   ? 0              : row_span.a;
  const uword in_row2          =                              row_span.b;
  const uword subcube_n_rows   = row_all   ? local_n_rows   : in_row2 - in_row1 + 1;

  const uword in_col1          = col_all   ? 0              : col_span.a;
  const uword in_col2          =                              col_span.b;
  const uword subcube_n_cols   = col_all   ? local_n_cols   : in_col2 - in_col1 + 1;

  const uword in_slice1        = slice_all ? 0              : slice_span.a;
  const uword in_slice2        =                              slice_span.b;
  const uword subcube_n_slices = slice_all ? local_n_slices : in_slice2 - in_slice1 + 1;

  coot_debug_check_bounds
    (
    ( row_all   ? false : ((in_row1   >  in_row2)   || (in_row2   >= local_n_rows))   )
    ||
    ( col_all   ? false : ((in_col1   >  in_col2)   || (in_col2   >= local_n_cols))   )
    ||
    ( slice_all ? false : ((in_slice1 >  in_slice2) || (in_slice2 >= local_n_slices)) )
    ,
    "Cube::subcube(): indices out of bounds or incorrectly used"
    );

  return subview_cube<eT>(*this, in_row1, in_col1, in_slice1, subcube_n_rows, subcube_n_cols, subcube_n_slices);
  }



template<typename eT>
inline
subview_cube<eT>
Cube<eT>::operator()(const span& row_span, const span& col_span, const span& slice_span)
  {
  coot_extra_debug_sigprint();

  return (*this).subcube(row_span, col_span, slice_span);
  }



template<typename eT>
inline
const subview_cube<eT>
Cube<eT>::operator()(const span& row_span, const span& col_span, const span& slice_span) const
  {
  coot_extra_debug_sigprint();

  return (*this).subcube(row_span, col_span, slice_span);
  }



template<typename eT>
inline
subview_cube<eT>
Cube<eT>::operator()(const uword in_row1, const uword in_col1, const uword in_slice1, const SizeCube& s)
  {
  coot_extra_debug_sigprint();

  return (*this).subcube(in_row1, in_col1, in_slice1, s);
  }



template<typename eT>
inline
const subview_cube<eT>
Cube<eT>::operator()(const uword in_row1, const uword in_col1, const uword in_slice1, const SizeCube& s) const
  {
  coot_extra_debug_sigprint();

  return (*this).subcube(in_row1, in_col1, in_slice1, s);
  }



template<typename eT>
coot_inline
subview_cube<eT>
Cube<eT>::tube(const uword in_row1, const uword in_col1)
  {
  coot_extra_debug_sigprint();

  coot_debug_check_bounds
    (
    ((in_row1 >= n_rows) || (in_col1 >= n_cols)),
    "Cube::tube(): indices out of bounds"
    );

  return subview_cube<eT>(*this, in_row1, in_col1, 0, 1, 1, n_slices);
  }



template<typename eT>
coot_inline
const subview_cube<eT>
Cube<eT>::tube(const uword in_row1, const uword in_col1) const
  {
  coot_extra_debug_sigprint();

  coot_debug_check_bounds
    (
    ((in_row1 >= n_rows) || (in_col1 >= n_cols)),
    "Cube::tube(): indices out of bounds"
    );

  return subview_cube<eT>(*this, in_row1, in_col1, 0, 1, 1, n_slices);
  }



template<typename eT>
coot_inline
subview_cube<eT>
Cube<eT>::tube(const uword in_row1, const uword in_col1, const uword in_row2, const uword in_col2)
  {
  coot_extra_debug_sigprint();

  coot_debug_check_bounds
    (
    (in_row1 >  in_row2) || (in_col1 >  in_col2) ||
    (in_row2 >= n_rows)  || (in_col2 >= n_cols),
    "Cube::tube(): indices out of bounds or incorrectly used"
    );

  const uword subcube_n_rows = in_row2 - in_row1 + 1;
  const uword subcube_n_cols = in_col2 - in_col1 + 1;

  return subview_cube<eT>(*this, in_row1, in_col1, 0, subcube_n_rows, subcube_n_cols, n_slices);
  }



template<typename eT>
coot_inline
const subview_cube<eT>
Cube<eT>::tube(const uword in_row1, const uword in_col1, const uword in_row2, const uword in_col2) const
  {
  coot_extra_debug_sigprint();

  coot_debug_check_bounds
    (
    (in_row1 >  in_row2) || (in_col1 >  in_col2) ||
    (in_row2 >= n_rows)  || (in_col2 >= n_cols),
    "Cube::tube(): indices out of bounds or incorrectly used"
    );

  const uword subcube_n_rows = in_row2 - in_row1 + 1;
  const uword subcube_n_cols = in_col2 - in_col1 + 1;

  return subview_cube<eT>(*this, in_row1, in_col1, 0, subcube_n_rows, subcube_n_cols, n_slices);
  }



template<typename eT>
coot_inline
subview_cube<eT>
Cube<eT>::tube(const uword in_row1, const uword in_col1, const SizeMat& s)
  {
  coot_extra_debug_sigprint();

  const uword l_n_rows = n_rows;
  const uword l_n_cols = n_cols;

  const uword s_n_rows = s.n_rows;
  const uword s_n_cols = s.n_cols;

  coot_debug_check_bounds
    (
    ((in_row1 >= l_n_rows) || (in_col1 >= l_n_cols) || ((in_row1 + s_n_rows) > l_n_rows) || ((in_col1 + s_n_cols) > l_n_cols)),
    "Cube::tube(): indices or size out of bounds"
    );

  return subview_cube<eT>(*this, in_row1, in_col1, 0, s_n_rows, s_n_cols, n_slices);
  }



template<typename eT>
coot_inline
const subview_cube<eT>
Cube<eT>::tube(const uword in_row1, const uword in_col1, const SizeMat& s) const
  {
  coot_extra_debug_sigprint();

  const uword l_n_rows = n_rows;
  const uword l_n_cols = n_cols;

  const uword s_n_rows = s.n_rows;
  const uword s_n_cols = s.n_cols;

  coot_debug_check_bounds
    (
    ((in_row1 >= l_n_rows) || (in_col1 >= l_n_cols) || ((in_row1 + s_n_rows) > l_n_rows) || ((in_col1 + s_n_cols) > l_n_cols)),
    "Cube::tube(): indices or size out of bounds"
    );

  return subview_cube<eT>(*this, in_row1, in_col1, 0, s_n_rows, s_n_cols, n_slices);
  }



template<typename eT>
inline
subview_cube<eT>
Cube<eT>::tube(const span& row_span, const span& col_span)
  {
  coot_extra_debug_sigprint();

  const bool row_all = row_span.whole;
  const bool col_all = col_span.whole;

  const uword local_n_rows   = n_rows;
  const uword local_n_cols   = n_cols;

  const uword in_row1        = row_all   ? 0            : row_span.a;
  const uword in_row2        =                            row_span.b;
  const uword subcube_n_rows = row_all   ? local_n_rows : in_row2 - in_row1 + 1;

  const uword in_col1        = col_all   ? 0            : col_span.a;
  const uword in_col2        =                            col_span.b;
  const uword subcube_n_cols = col_all   ? local_n_cols : in_col2 - in_col1 + 1;

  coot_debug_check_bounds
    (
    ( row_all ? false : ((in_row1 > in_row2) || (in_row2 >= local_n_rows)) )
    ||
    ( col_all ? false : ((in_col1 > in_col2) || (in_col2 >= local_n_cols)) )
    ,
    "Cube::tube(): indices out of bounds or incorrectly used"
    );

  return subview_cube<eT>(*this, in_row1, in_col1, 0, subcube_n_rows, subcube_n_cols, n_slices);
  }



template<typename eT>
inline
const subview_cube<eT>
Cube<eT>::tube(const span& row_span, const span& col_span) const
  {
  coot_extra_debug_sigprint();

  const bool row_all = row_span.whole;
  const bool col_all = col_span.whole;

  const uword local_n_rows   = n_rows;
  const uword local_n_cols   = n_cols;

  const uword in_row1        = row_all   ? 0            : row_span.a;
  const uword in_row2        =                            row_span.b;
  const uword subcube_n_rows = row_all   ? local_n_rows : in_row2 - in_row1 + 1;

  const uword in_col1        = col_all   ? 0            : col_span.a;
  const uword in_col2        =                            col_span.b;
  const uword subcube_n_cols = col_all   ? local_n_cols : in_col2 - in_col1 + 1;

  coot_debug_check_bounds
    (
    ( row_all ? false : ((in_row1 > in_row2) || (in_row2 >= local_n_rows)) )
    ||
    ( col_all ? false : ((in_col1 > in_col2) || (in_col2 >= local_n_cols)) )
    ,
    "Cube::tube(): indices out of bounds or incorrectly used"
    );

  return subview_cube<eT>(*this, in_row1, in_col1, 0, subcube_n_rows, subcube_n_cols, n_slices);
  }



template<typename eT>
inline
subview_cube<eT>
Cube<eT>::head_slices(const uword N)
  {
  coot_extra_debug_sigprint();

  coot_debug_check_bounds( (N > n_slices), "Cube::head_slices(): size out of bounds" );

  return subview_cube<eT>(*this, 0, 0, 0, n_rows, n_cols, N);
  }



template<typename eT>
inline
const subview_cube<eT>
Cube<eT>::head_slices(const uword N) const
  {
  coot_extra_debug_sigprint();

  coot_debug_check_bounds( (N > n_slices), "Cube::head_slices(): size out of bounds" );

  return subview_cube<eT>(*this, 0, 0, 0, n_rows, n_cols, N);
  }



template<typename eT>
inline
subview_cube<eT>
Cube<eT>::tail_slices(const uword N)
  {
  coot_extra_debug_sigprint();

  coot_debug_check_bounds( (N > n_slices), "Cube::tail_slices(): size out of bounds" );

  const uword start_slice = n_slices - N;

  return subview_cube<eT>(*this, 0, 0, start_slice, n_rows, n_cols, N);
  }



template<typename eT>
inline
const subview_cube<eT>
Cube<eT>::tail_slices(const uword N) const
  {
  coot_extra_debug_sigprint();

  coot_debug_check_bounds( (N > n_slices), "Cube::tail_slices(): size out of bounds" );

  const uword start_slice = n_slices - N;

  return subview_cube<eT>(*this, 0, 0, start_slice, n_rows, n_cols, N);
  }



template<typename eT>
template<typename T1>
coot_inline
subview_elem1<eT,T1>
Cube<eT>::elem(const Base<uword,T1>& a)
  {
  coot_extra_debug_sigprint();

  return subview_elem1<eT,T1>(*this, a);
  }



template<typename eT>
template<typename T1>
coot_inline
const subview_elem1<eT,T1>
Cube<eT>::elem(const Base<uword,T1>& a) const
  {
  coot_extra_debug_sigprint();

  return subview_elem1<eT,T1>(*this, a);
  }



template<typename eT>
template<typename T1>
coot_inline
subview_elem1<eT,T1>
Cube<eT>::operator()(const Base<uword,T1>& a)
  {
  coot_extra_debug_sigprint();

  return subview_elem1<eT,T1>(*this, a);
  }



template<typename eT>
template<typename T1>
coot_inline
const subview_elem1<eT,T1>
Cube<eT>::operator()(const Base<uword,T1>& a) const
  {
  coot_extra_debug_sigprint();

  return subview_elem1<eT,T1>(*this, a);
  }



/* template<typename eT> */
/* arma_inline */
/* subview_cube_each1<eT> */
/* Cube<eT>::each_slice() */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   return subview_cube_each1<eT>(*this); */
/*   } */



/* template<typename eT> */
/* arma_inline */
/* const subview_cube_each1<eT> */
/* Cube<eT>::each_slice() const */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   return subview_cube_each1<eT>(*this); */
/*   } */



/* template<typename eT> */
/* template<typename T1> */
/* inline */
/* subview_cube_each2<eT, T1> */
/* Cube<eT>::each_slice(const Base<uword, T1>& indices) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   return subview_cube_each2<eT, T1>(*this, indices); */
/*   } */



/* template<typename eT> */
/* template<typename T1> */
/* inline */
/* const subview_cube_each2<eT, T1> */
/* Cube<eT>::each_slice(const Base<uword, T1>& indices) const */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   return subview_cube_each2<eT, T1>(*this, indices); */
/*   } */



/* //! apply a lambda function to each slice, where each slice is interpreted as a matrix */
/* template<typename eT> */
/* inline */
/* const Cube<eT>& */
/* Cube<eT>::each_slice(const std::function< void(Mat<eT>&) >& F) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   for(uword slice_id=0; slice_id < n_slices; ++slice_id) */
/*     { */
/*     Mat<eT> tmp('j', slice_memptr(slice_id), n_rows, n_cols); */

/*     F(tmp); */
/*     } */

/*   return *this; */
/*   } */



/* template<typename eT> */
/* inline */
/* const Cube<eT>& */
/* Cube<eT>::each_slice(const std::function< void(const Mat<eT>&) >& F) const */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   for(uword slice_id=0; slice_id < n_slices; ++slice_id) */
/*     { */
/*     const Mat<eT> tmp('j', slice_memptr(slice_id), n_rows, n_cols); */

/*     F(tmp); */
/*     } */

/*   return *this; */
/*   } */



/* template<typename eT> */
/* inline */
/* const Cube<eT>& */
/* Cube<eT>::each_slice(const std::function< void(Mat<eT>&) >& F, const bool use_mp) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   if((use_mp == false) || (arma_config::openmp == false)) */
/*     { */
/*     return (*this).each_slice(F); */
/*     } */

/*   #if defined(ARMA_USE_OPENMP) */
/*     { */
/*     const uword local_n_slices = n_slices; */
/*     const int   n_threads      = mp_thread_limit::get(); */

/*     #pragma omp parallel for schedule(static) num_threads(n_threads) */
/*     for(uword slice_id=0; slice_id < local_n_slices; ++slice_id) */
/*       { */
/*       Mat<eT> tmp('j', slice_memptr(slice_id), n_rows, n_cols); */

/*       F(tmp); */
/*       } */
/*     } */
/*   #endif */

/*   return *this; */
/*   } */



/* template<typename eT> */
/* inline */
/* const Cube<eT>& */
/* Cube<eT>::each_slice(const std::function< void(const Mat<eT>&) >& F, const bool use_mp) const */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   if((use_mp == false) || (arma_config::openmp == false)) */
/*     { */
/*     return (*this).each_slice(F); */
/*     } */

/*   #if defined(ARMA_USE_OPENMP) */
/*     { */
/*     const uword local_n_slices = n_slices; */
/*     const int   n_threads      = mp_thread_limit::get(); */

/*     #pragma omp parallel for schedule(static) num_threads(n_threads) */
/*     for(uword slice_id=0; slice_id < local_n_slices; ++slice_id) */
/*       { */
/*       Mat<eT> tmp('j', slice_memptr(slice_id), n_rows, n_cols); */

/*       F(tmp); */
/*       } */
/*     } */
/*   #endif */

/*   return *this; */
/*   } */



/* template<typename eT> */
/* template<typename T1> */
/* inline */
/* subview_cube_slices<eT, T1> */
/* Cube<eT>::slices(const Base<uword, T1>& indices) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   return subview_cube_slices<eT, T1>(*this, indices); */
/*   } */



/* template<typename eT> */
/* template<typename T1> */
/* inline */
/* const subview_cube_slices<eT, T1> */
/* Cube<eT>::slices(const Base<uword, T1>& indices) const */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   return subview_cube_slices<eT, T1>(*this, indices); */
/*   } */



/* //! remove specified row */
/* template<typename eT> */
/* inline */
/* void */
/* Cube<eT>::shed_row(const uword row_num) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   arma_debug_check_bounds( row_num >= n_rows, "Cube::shed_row(): index out of bounds" ); */

/*   shed_rows(row_num, row_num); */
/*   } */



/* //! remove specified column */
/* template<typename eT> */
/* inline */
/* void */
/* Cube<eT>::shed_col(const uword col_num) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   arma_debug_check_bounds( col_num >= n_cols, "Cube::shed_col(): index out of bounds" ); */

/*   shed_cols(col_num, col_num); */
/*   } */



/* //! remove specified slice */
/* template<typename eT> */
/* inline */
/* void */
/* Cube<eT>::shed_slice(const uword slice_num) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   arma_debug_check_bounds( slice_num >= n_slices, "Cube::shed_slice(): index out of bounds" ); */

/*   shed_slices(slice_num, slice_num); */
/*   } */



/* //! remove specified rows */
/* template<typename eT> */
/* inline */
/* void */
/* Cube<eT>::shed_rows(const uword in_row1, const uword in_row2) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   arma_debug_check_bounds */
/*     ( */
/*     (in_row1 > in_row2) || (in_row2 >= n_rows), */
/*     "Cube::shed_rows(): indices out of bounds or incorrectly used" */
/*     ); */

/*   const uword n_keep_front = in_row1; */
/*   const uword n_keep_back  = n_rows - (in_row2 + 1); */

/*   Cube<eT> X(n_keep_front + n_keep_back, n_cols, n_slices, arma_nozeros_indicator()); */

/*   if(n_keep_front > 0) */
/*     { */
/*     X.rows( 0, (n_keep_front-1) ) = rows( 0, (in_row1-1) ); */
/*     } */

/*   if(n_keep_back > 0) */
/*     { */
/*     X.rows( n_keep_front,  (n_keep_front+n_keep_back-1) ) = rows( (in_row2+1), (n_rows-1) ); */
/*     } */

/*   steal_mem(X); */
/*   } */



/* //! remove specified columns */
/* template<typename eT> */
/* inline */
/* void */
/* Cube<eT>::shed_cols(const uword in_col1, const uword in_col2) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   arma_debug_check_bounds */
/*     ( */
/*     (in_col1 > in_col2) || (in_col2 >= n_cols), */
/*     "Cube::shed_cols(): indices out of bounds or incorrectly used" */
/*     ); */

/*   const uword n_keep_front = in_col1; */
/*   const uword n_keep_back  = n_cols - (in_col2 + 1); */

/*   Cube<eT> X(n_rows, n_keep_front + n_keep_back, n_slices, arma_nozeros_indicator()); */

/*   if(n_keep_front > 0) */
/*     { */
/*     X.cols( 0, (n_keep_front-1) ) = cols( 0, (in_col1-1) ); */
/*     } */

/*   if(n_keep_back > 0) */
/*     { */
/*     X.cols( n_keep_front,  (n_keep_front+n_keep_back-1) ) = cols( (in_col2+1), (n_cols-1) ); */
/*     } */

/*   steal_mem(X); */
/*   } */



/* //! remove specified slices */
/* template<typename eT> */
/* inline */
/* void */
/* Cube<eT>::shed_slices(const uword in_slice1, const uword in_slice2) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   arma_debug_check_bounds */
/*     ( */
/*     (in_slice1 > in_slice2) || (in_slice2 >= n_slices), */
/*     "Cube::shed_slices(): indices out of bounds or incorrectly used" */
/*     ); */

/*   const uword n_keep_front = in_slice1; */
/*   const uword n_keep_back  = n_slices - (in_slice2 + 1); */

/*   Cube<eT> X(n_rows, n_cols, n_keep_front + n_keep_back, arma_nozeros_indicator()); */

/*   if(n_keep_front > 0) */
/*     { */
/*     X.slices( 0, (n_keep_front-1) ) = slices( 0, (in_slice1-1) ); */
/*     } */

/*   if(n_keep_back > 0) */
/*     { */
/*     X.slices( n_keep_front,  (n_keep_front+n_keep_back-1) ) = slices( (in_slice2+1), (n_slices-1) ); */
/*     } */

/*   steal_mem(X); */
/*   } */



/* //! remove specified slices */
/* template<typename eT> */
/* template<typename T1> */
/* inline */
/* void */
/* Cube<eT>::shed_slices(const Base<uword, T1>& indices) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   const quasi_unwrap<T1>   U(indices.get_ref()); */
/*   const Mat<uword>& tmp1 = U.M; */

/*   arma_debug_check( ((tmp1.is_vec() == false) && (tmp1.is_empty() == false)), "Cube::shed_slices(): list of indices must be a vector" ); */

/*   if(tmp1.is_empty()) { return; } */

/*   const Col<uword> tmp2(const_cast<uword*>(tmp1.memptr()), tmp1.n_elem, false, false); */

/*   const Col<uword>& slices_to_shed = (tmp2.is_sorted("strictascend") == false) */
/*                                      ? Col<uword>(unique(tmp2)) */
/*                                      : Col<uword>(const_cast<uword*>(tmp2.memptr()), tmp2.n_elem, false, false); */

/*   const uword* slices_to_shed_mem = slices_to_shed.memptr(); */
/*   const uword  N                  = slices_to_shed.n_elem; */

/*   if(arma_config::debug) */
/*     { */
/*     for(uword i=0; i<N; ++i) */
/*       { */
/*       arma_debug_check_bounds( (slices_to_shed_mem[i] >= n_slices), "Cube::shed_slices(): indices out of bounds" ); */
/*       } */
/*     } */

/*   Col<uword> tmp3(n_slices, arma_nozeros_indicator()); */

/*   uword* tmp3_mem = tmp3.memptr(); */

/*   uword i     = 0; */
/*   uword count = 0; */

/*   for(uword j=0; j < n_slices; ++j) */
/*     { */
/*     if(i < N) */
/*       { */
/*       if( j != slices_to_shed_mem[i] ) */
/*         { */
/*         tmp3_mem[count] = j; */
/*         ++count; */
/*         } */
/*       else */
/*         { */
/*         ++i; */
/*         } */
/*       } */
/*     else */
/*       { */
/*       tmp3_mem[count] = j; */
/*       ++count; */
/*       } */
/*     } */

/*   const Col<uword> slices_to_keep(tmp3.memptr(), count, false, false); */

/*   Cube<eT> X = (*this).slices(slices_to_keep); */

/*   steal_mem(X); */
/*   } */



/* template<typename eT> */
/* inline */
/* void */
/* Cube<eT>::insert_rows(const uword row_num, const uword N, const bool set_to_zero) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   const uword t_n_rows = n_rows; */

/*   const uword A_n_rows = row_num; */
/*   const uword B_n_rows = t_n_rows - row_num; */

/*   // insertion at row_num == n_rows is in effect an append operation */
/*   arma_debug_check_bounds( (row_num > t_n_rows), "Cube::insert_rows(): index out of bounds" ); */

/*   if(N > 0) */
/*     { */
/*     Cube<eT> out(t_n_rows + N, n_cols, n_slices, arma_nozeros_indicator()); */

/*     if(A_n_rows > 0) */
/*       { */
/*       out.rows(0, A_n_rows-1) = rows(0, A_n_rows-1); */
/*       } */

/*     if(B_n_rows > 0) */
/*       { */
/*       out.rows(row_num + N, t_n_rows + N - 1) = rows(row_num, t_n_rows-1); */
/*       } */

/*     if(set_to_zero) */
/*       { */
/*       out.rows(row_num, row_num + N - 1).zeros(); */
/*       } */

/*     steal_mem(out); */
/*     } */
/*   } */



/* template<typename eT> */
/* inline */
/* void */
/* Cube<eT>::insert_cols(const uword col_num, const uword N, const bool set_to_zero) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   const uword t_n_cols = n_cols; */

/*   const uword A_n_cols = col_num; */
/*   const uword B_n_cols = t_n_cols - col_num; */

/*   // insertion at col_num == n_cols is in effect an append operation */
/*   arma_debug_check_bounds( (col_num > t_n_cols), "Cube::insert_cols(): index out of bounds" ); */

/*   if(N > 0) */
/*     { */
/*     Cube<eT> out(n_rows, t_n_cols + N, n_slices, arma_nozeros_indicator()); */

/*     if(A_n_cols > 0) */
/*       { */
/*       out.cols(0, A_n_cols-1) = cols(0, A_n_cols-1); */
/*       } */

/*     if(B_n_cols > 0) */
/*       { */
/*       out.cols(col_num + N, t_n_cols + N - 1) = cols(col_num, t_n_cols-1); */
/*       } */

/*     if(set_to_zero) */
/*       { */
/*       out.cols(col_num, col_num + N - 1).zeros(); */
/*       } */

/*     steal_mem(out); */
/*     } */
/*   } */



/* //! insert N slices at the specified slice position, */
/* //! optionally setting the elements of the inserted slices to zero */
/* template<typename eT> */
/* inline */
/* void */
/* Cube<eT>::insert_slices(const uword slice_num, const uword N, const bool set_to_zero) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   const uword t_n_slices = n_slices; */

/*   const uword A_n_slices = slice_num; */
/*   const uword B_n_slices = t_n_slices - slice_num; */

/*   // insertion at slice_num == n_slices is in effect an append operation */
/*   arma_debug_check_bounds( (slice_num > t_n_slices), "Cube::insert_slices(): index out of bounds" ); */

/*   if(N > 0) */
/*     { */
/*     Cube<eT> out(n_rows, n_cols, t_n_slices + N, arma_nozeros_indicator()); */

/*     if(A_n_slices > 0) */
/*       { */
/*       out.slices(0, A_n_slices-1) = slices(0, A_n_slices-1); */
/*       } */

/*     if(B_n_slices > 0) */
/*       { */
/*       out.slices(slice_num + N, t_n_slices + N - 1) = slices(slice_num, t_n_slices-1); */
/*       } */

/*     if(set_to_zero) */
/*       { */
/*       //out.slices(slice_num, slice_num + N - 1).zeros(); */

/*       for(uword i=slice_num; i < (slice_num + N); ++i) */
/*         { */
/*         arrayops::fill_zeros(out.slice_memptr(i), out.n_elem_slice); */
/*         } */
/*       } */

/*     steal_mem(out); */
/*     } */
/*   } */



/* template<typename eT> */
/* template<typename T1> */
/* inline */
/* void */
/* Cube<eT>::insert_rows(const uword row_num, const BaseCube<eT,T1>& X) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   const unwrap_cube<T1> tmp(X.get_ref()); */
/*   const Cube<eT>& C   = tmp.M; */

/*   const uword N = C.n_rows; */

/*   const uword t_n_rows = n_rows; */

/*   const uword A_n_rows = row_num; */
/*   const uword B_n_rows = t_n_rows - row_num; */

/*   // insertion at row_num == n_rows is in effect an append operation */
/*   arma_debug_check_bounds( (row_num  >  t_n_rows), "Cube::insert_rows(): index out of bounds" ); */

/*   arma_debug_check */
/*     ( */
/*     ( (C.n_cols != n_cols) || (C.n_slices != n_slices) ), */
/*     "Cube::insert_rows(): given object has incompatible dimensions" */
/*     ); */

/*   if(N > 0) */
/*     { */
/*     Cube<eT> out(t_n_rows + N, n_cols, n_slices, arma_nozeros_indicator()); */

/*     if(A_n_rows > 0) */
/*       { */
/*       out.rows(0, A_n_rows-1) = rows(0, A_n_rows-1); */
/*       } */

/*     if(B_n_rows > 0) */
/*       { */
/*       out.rows(row_num + N, t_n_rows + N - 1) = rows(row_num, t_n_rows - 1); */
/*       } */

/*     out.rows(row_num, row_num + N - 1) = C; */

/*     steal_mem(out); */
/*     } */
/*   } */



/* template<typename eT> */
/* template<typename T1> */
/* inline */
/* void */
/* Cube<eT>::insert_cols(const uword col_num, const BaseCube<eT,T1>& X) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   const unwrap_cube<T1> tmp(X.get_ref()); */
/*   const Cube<eT>& C   = tmp.M; */

/*   const uword N = C.n_cols; */

/*   const uword t_n_cols = n_cols; */

/*   const uword A_n_cols = col_num; */
/*   const uword B_n_cols = t_n_cols - col_num; */

/*   // insertion at col_num == n_cols is in effect an append operation */
/*   arma_debug_check_bounds( (col_num  >  t_n_cols), "Cube::insert_cols(): index out of bounds" ); */

/*   arma_debug_check */
/*     ( */
/*     ( (C.n_rows != n_rows) || (C.n_slices != n_slices) ), */
/*     "Cube::insert_cols(): given object has incompatible dimensions" */
/*     ); */

/*   if(N > 0) */
/*     { */
/*     Cube<eT> out(n_rows, t_n_cols + N, n_slices, arma_nozeros_indicator()); */

/*     if(A_n_cols > 0) */
/*       { */
/*       out.cols(0, A_n_cols-1) = cols(0, A_n_cols-1); */
/*       } */

/*     if(B_n_cols > 0) */
/*       { */
/*       out.cols(col_num + N, t_n_cols + N - 1) = cols(col_num, t_n_cols - 1); */
/*       } */

/*     out.cols(col_num, col_num + N - 1) = C; */

/*     steal_mem(out); */
/*     } */
/*   } */



/* //! insert the given object at the specified slice position; */
/* //! the given object must have the same number of rows and columns as the cube */
/* template<typename eT> */
/* template<typename T1> */
/* inline */
/* void */
/* Cube<eT>::insert_slices(const uword slice_num, const BaseCube<eT,T1>& X) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   const unwrap_cube<T1> tmp(X.get_ref()); */
/*   const Cube<eT>& C   = tmp.M; */

/*   const uword N = C.n_slices; */

/*   const uword t_n_slices = n_slices; */

/*   const uword A_n_slices = slice_num; */
/*   const uword B_n_slices = t_n_slices - slice_num; */

/*   // insertion at slice_num == n_slices is in effect an append operation */
/*   arma_debug_check_bounds( (slice_num  >  t_n_slices), "Cube::insert_slices(): index out of bounds" ); */

/*   arma_debug_check */
/*     ( */
/*     ( (C.n_rows != n_rows) || (C.n_cols != n_cols) ), */
/*     "Cube::insert_slices(): given object has incompatible dimensions" */
/*     ); */

/*   if(N > 0) */
/*     { */
/*     Cube<eT> out(n_rows, n_cols, t_n_slices + N, arma_nozeros_indicator()); */

/*     if(A_n_slices > 0) */
/*       { */
/*       out.slices(0, A_n_slices-1) = slices(0, A_n_slices-1); */
/*       } */

/*     if(B_n_slices > 0) */
/*       { */
/*       out.slices(slice_num + N, t_n_slices + N - 1) = slices(slice_num, t_n_slices - 1); */
/*       } */

/*     out.slices(slice_num, slice_num + N - 1) = C; */

/*     steal_mem(out); */
/*     } */
/*   } */



/* //! create a cube from GenCube, ie. run the previously delayed element generation operations */
/* template<typename eT> */
/* template<typename gen_type> */
/* inline */
/* Cube<eT>::Cube(const GenCube<eT, gen_type>& X) */
/*   : n_rows(X.n_rows) */
/*   , n_cols(X.n_cols) */
/*   , n_elem_slice(X.n_rows*X.n_cols) */
/*   , n_slices(X.n_slices) */
/*   , n_elem(X.n_rows*X.n_cols*X.n_slices) */
/*   , mem_state(0) */
/*   , mem() */
/*   , mat_ptrs(nullptr) */
/*   { */
/*   arma_extra_debug_sigprint_this(this); */

/*   init_cold(); */

/*   X.apply(*this); */
/*   } */



/* template<typename eT> */
/* template<typename gen_type> */
/* inline */
/* Cube<eT>& */
/* Cube<eT>::operator=(const GenCube<eT, gen_type>& X) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   init_warm(X.n_rows, X.n_cols, X.n_slices); */

/*   X.apply(*this); */

/*   return *this; */
/*   } */



/* template<typename eT> */
/* template<typename gen_type> */
/* inline */
/* Cube<eT>& */
/* Cube<eT>::operator+=(const GenCube<eT, gen_type>& X) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   X.apply_inplace_plus(*this); */

/*   return *this; */
/*   } */



/* template<typename eT> */
/* template<typename gen_type> */
/* inline */
/* Cube<eT>& */
/* Cube<eT>::operator-=(const GenCube<eT, gen_type>& X) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   X.apply_inplace_minus(*this); */

/*   return *this; */
/*   } */



/* template<typename eT> */
/* template<typename gen_type> */
/* inline */
/* Cube<eT>& */
/* Cube<eT>::operator%=(const GenCube<eT, gen_type>& X) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   X.apply_inplace_schur(*this); */

/*   return *this; */
/*   } */



/* template<typename eT> */
/* template<typename gen_type> */
/* inline */
/* Cube<eT>& */
/* Cube<eT>::operator/=(const GenCube<eT, gen_type>& X) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   X.apply_inplace_div(*this); */

/*   return *this; */
/*   } */



// create a cube from OpCube, ie. run the previously delayed unary operations
template<typename eT>
template<typename T1, typename op_type>
inline
Cube<eT>::Cube(const OpCube<T1, op_type>& X)
  : n_rows(0)
  , n_cols(0)
  , n_elem_slice(0)
  , n_slices(0)
  , n_elem(0)
  , mem_state(0)
  , dev_mem({{ NULL, 0 }})
  , mat_ptrs(nullptr)
  {
  coot_extra_debug_sigprint_this(this);

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));

  op_type::apply(*this, X);
  }



// create a cube from OpCube, ie. run the previously delayed unary operations
template<typename eT>
template<typename T1, typename op_type>
inline
Cube<eT>&
Cube<eT>::operator=(const OpCube<T1, op_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));

  op_type::apply(*this, X);

  return *this;
  }



// in-place cube addition, with the right-hand-side operand having delayed operations
template<typename eT>
template<typename T1, typename op_type>
inline
Cube<eT>&
Cube<eT>::operator+=(const OpCube<T1, op_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));

  const Cube<eT> m(X);

  return (*this).operator+=(m);
  }



// in-place cube subtraction, with the right-hand-side operand having delayed operations
template<typename eT>
template<typename T1, typename op_type>
inline
Cube<eT>&
Cube<eT>::operator-=(const OpCube<T1, op_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));

  const Cube<eT> m(X);

  return (*this).operator-=(m);
  }



// in-place cube element-wise multiplication, with the right-hand-side operand having delayed operations
template<typename eT>
template<typename T1, typename op_type>
inline
Cube<eT>&
Cube<eT>::operator%=(const OpCube<T1, op_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));

  const Cube<eT> m(X);

  return (*this).operator%=(m);
  }



// in-place cube element-wise division, with the right-hand-side operand having delayed operations
template<typename eT>
template<typename T1, typename op_type>
inline
Cube<eT>&
Cube<eT>::operator/=(const OpCube<T1, op_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));

  const Cube<eT> m(X);

  return (*this).operator/=(m);
  }



//! create a cube from eOpCube, ie. run the previously delayed unary operations
template<typename eT>
template<typename T1, typename eop_type>
inline
Cube<eT>::Cube(const eOpCube<T1, eop_type>& X)
  : n_rows(0)
  , n_cols(0)
  , n_elem_slice(0)
  , n_slices(0)
  , n_elem(0)
  , mem_state(0)
  , dev_mem({{ NULL, 0 }})
  , mat_ptrs(nullptr)
  {
  coot_extra_debug_sigprint_this(this);

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));

  set_size(X.get_n_rows(), X.get_n_cols(), X.get_n_slices());

  eop_type::apply(*this, X);
  }



//! create a cube from eOpCube, ie. run the previously delayed unary operations
template<typename eT>
template<typename T1, typename eop_type>
inline
Cube<eT>&
Cube<eT>::operator=(const eOpCube<T1, eop_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));

  set_size(X.get_n_rows(), X.get_n_cols(), X.get_n_slices());

  eop_type::apply(*this, X);

  return *this;
  }



//! in-place cube addition, with the right-hand-side operand having delayed operations
template<typename eT>
template<typename T1, typename eop_type>
inline
Cube<eT>&
Cube<eT>::operator+=(const eOpCube<T1, eop_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));

  eop_type::apply_inplace_plus(*this, X);

  return *this;
  }



//! in-place cube subtraction, with the right-hand-side operand having delayed operations
template<typename eT>
template<typename T1, typename eop_type>
inline
Cube<eT>&
Cube<eT>::operator-=(const eOpCube<T1, eop_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));

  eop_type::apply_inplace_minus(*this, X);

  return *this;
  }



//! in-place cube element-wise multiplication, with the right-hand-side operand having delayed operations
template<typename eT>
template<typename T1, typename eop_type>
inline
Cube<eT>&
Cube<eT>::operator%=(const eOpCube<T1, eop_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));

  eop_type::apply_inplace_schur(*this, X);

  return *this;
  }



//! in-place cube element-wise division, with the right-hand-side operand having delayed operations
template<typename eT>
template<typename T1, typename eop_type>
inline
Cube<eT>&
Cube<eT>::operator/=(const eOpCube<T1, eop_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));

  eop_type::apply_inplace_div(*this, X);

  return *this;
  }



template<typename eT>
template<typename T1, typename mtop_type>
inline
Cube<eT>::Cube(const mtOpCube<eT, T1, mtop_type>& X)
  : n_rows(0)
  , n_cols(0)
  , n_elem_slice(0)
  , n_slices(0)
  , n_elem(0)
  , mem_state(0)
  , dev_mem({{ NULL, 0 }})
  , mat_ptrs(nullptr)
  {
  coot_extra_debug_sigprint_this(this);

  mtop_type::apply(*this, X);
  }



template<typename eT>
template<typename T1, typename mtop_type>
inline
Cube<eT>&
Cube<eT>::operator=(const mtOpCube<eT, T1, mtop_type>& X)
  {
  coot_extra_debug_sigprint();

  mtop_type::apply(*this, X);

  return *this;
  }



template<typename eT>
template<typename T1, typename mtop_type>
inline
Cube<eT>&
Cube<eT>::operator+=(const mtOpCube<eT, T1, mtop_type>& X)
  {
  coot_extra_debug_sigprint();

  mtop_type::apply_inplace_plus(*this, X);

  return *this;
  }



template<typename eT>
template<typename T1, typename mtop_type>
inline
Cube<eT>&
Cube<eT>::operator-=(const mtOpCube<eT, T1, mtop_type>& X)
  {
  coot_extra_debug_sigprint();

  mtop_type::apply_inplace_minus(*this, X);

  return *this;
  }



template<typename eT>
template<typename T1, typename mtop_type>
inline
Cube<eT>&
Cube<eT>::operator%=(const mtOpCube<eT, T1, mtop_type>& X)
  {
  coot_extra_debug_sigprint();

  mtop_type::apply_inplace_schur(*this, X);

  return *this;
  }



template<typename eT>
template<typename T1, typename mtop_type>
inline
Cube<eT>&
Cube<eT>::operator/=(const mtOpCube<eT, T1, mtop_type>& X)
  {
  coot_extra_debug_sigprint();

  mtop_type::apply_inplace_div(*this, X);

  return *this;
  }



// create a cube from GlueCube, ie. run the previously delayed binary operations
template<typename eT>
template<typename T1, typename T2, typename glue_type>
inline
Cube<eT>::Cube(const GlueCube<T1, T2, glue_type>& X)
  : n_rows(0)
  , n_cols(0)
  , n_elem_slice(0)
  , n_slices(0)
  , n_elem(0)
  , mem_state(0)
  , dev_mem({{ NULL, 0 }})
  , mat_ptrs(nullptr)
  {
  coot_extra_debug_sigprint_this(this);

  this->operator=(X);
  }



// create a cube from GlueCube, ie. run the previously delayed binary operations
template<typename eT>
template<typename T1, typename T2, typename glue_type>
inline
Cube<eT>&
Cube<eT>::operator=(const GlueCube<T1, T2, glue_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  coot_type_check(( is_same_type< eT, typename T2::elem_type >::no ));

  glue_type::apply(*this, X);

  return *this;
  }


// in-place cube addition, with the right-hand-side operands having delayed operations
template<typename eT>
template<typename T1, typename T2, typename glue_type>
inline
Cube<eT>&
Cube<eT>::operator+=(const GlueCube<T1, T2, glue_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  coot_type_check(( is_same_type< eT, typename T2::elem_type >::no ));

  const Cube<eT> m(X);

  return (*this).operator+=(m);
  }



// in-place cube subtraction, with the right-hand-side operands having delayed operations
template<typename eT>
template<typename T1, typename T2, typename glue_type>
inline
Cube<eT>&
Cube<eT>::operator-=(const GlueCube<T1, T2, glue_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  coot_type_check(( is_same_type< eT, typename T2::elem_type >::no ));

  const Cube<eT> m(X);

  return (*this).operator-=(m);
  }



// in-place cube element-wise multiplication, with the right-hand-side operands having delayed operations
template<typename eT>
template<typename T1, typename T2, typename glue_type>
inline
Cube<eT>&
Cube<eT>::operator%=(const GlueCube<T1, T2, glue_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  coot_type_check(( is_same_type< eT, typename T2::elem_type >::no ));

  const Cube<eT> m(X);

  return (*this).operator%=(m);
  }



// in-place cube element-wise division, with the right-hand-side operands having delayed operations
template<typename eT>
template<typename T1, typename T2, typename glue_type>
inline
Cube<eT>&
Cube<eT>::operator/=(const GlueCube<T1, T2, glue_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  coot_type_check(( is_same_type< eT, typename T2::elem_type >::no ));

  const Cube<eT> m(X);

  return (*this).operator/=(m);
  }



//! create a cube from eGlueCube, ie. run the previously delayed binary operations
template<typename eT>
template<typename T1, typename T2, typename eglue_type>
inline
Cube<eT>::Cube(const eGlueCube<T1, T2, eglue_type>& X)
  : n_rows(0)
  , n_cols(0)
  , n_elem_slice(0)
  , n_slices(0)
  , n_elem(0)
  , mem_state(0)
  , dev_mem({{ NULL, 0 }})
  , mat_ptrs(nullptr)
  {
  coot_extra_debug_sigprint_this(this);

  (*this).operator=(X);
  }



//! create a cube from eGlueCube, ie. run the previously delayed binary operations
template<typename eT>
template<typename T1, typename T2, typename eglue_type>
inline
Cube<eT>&
Cube<eT>::operator=(const eGlueCube<T1, T2, eglue_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  coot_type_check(( is_same_type< eT, typename T2::elem_type >::no ));

  // eglue_core currently forcefully unwraps subcubes to cubes,
  // so currently there can't be dangerous aliasing with the out matrix

  set_size(X.get_n_rows(), X.get_n_cols(), X.get_n_slices());

  eglue_type::apply(*this, X);

  return *this;
  }



//! in-place cube addition, with the right-hand-side operands having delayed operations */
template<typename eT>
template<typename T1, typename T2, typename eglue_type>
inline
Cube<eT>&
Cube<eT>::operator+=(const eGlueCube<T1, T2, eglue_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  coot_type_check(( is_same_type< eT, typename T2::elem_type >::no ));

  // eglue_core currently forcefully unwraps subcubes to cubes,
  // so currently there can't be dangerous aliasing with the out matrix

  eglue_type::apply_inplace_plus(*this, X);

  return *this;
  }



//! in-place cube subtraction, with the right-hand-side operands having delayed operations
template<typename eT>
template<typename T1, typename T2, typename eglue_type>
inline
Cube<eT>&
Cube<eT>::operator-=(const eGlueCube<T1, T2, eglue_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  coot_type_check(( is_same_type< eT, typename T2::elem_type >::no ));

  eglue_type::apply_inplace_minus(*this, X);

  return *this;
  }



//! in-place cube element-wise multiplication, with the right-hand-side operands having delayed operations
template<typename eT>
template<typename T1, typename T2, typename eglue_type>
inline
Cube<eT>&
Cube<eT>::operator%=(const eGlueCube<T1, T2, eglue_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  coot_type_check(( is_same_type< eT, typename T2::elem_type >::no ));

  eglue_type::apply_inplace_schur(*this, X);

  return *this;
  }



//! in-place cube element-wise division, with the right-hand-side operands having delayed operations
template<typename eT>
template<typename T1, typename T2, typename eglue_type>
inline
Cube<eT>&
Cube<eT>::operator/=(const eGlueCube<T1, T2, eglue_type>& X)
  {
  coot_extra_debug_sigprint();

  coot_type_check(( is_same_type< eT, typename T1::elem_type >::no ));
  coot_type_check(( is_same_type< eT, typename T2::elem_type >::no ));

  eglue_type::apply_inplace_div(*this, X);

  return *this;
  }



/* template<typename eT> */
/* template<typename T1, typename T2, typename glue_type> */
/* inline */
/* Cube<eT>::Cube(const mtGlueCube<eT, T1, T2, glue_type>& X) */
/*   : n_rows(0) */
/*   , n_cols(0) */
/*   , n_elem_slice(0) */
/*   , n_slices(0) */
/*   , n_elem(0) */
/*   , mem_state(0) */
/*   , mem() */
/*   , mat_ptrs(nullptr) */
/*   { */
/*   arma_extra_debug_sigprint_this(this); */

/*   glue_type::apply(*this, X); */
/*   } */



/* template<typename eT> */
/* template<typename T1, typename T2, typename glue_type> */
/* inline */
/* Cube<eT>& */
/* Cube<eT>::operator=(const mtGlueCube<eT, T1, T2, glue_type>& X) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   glue_type::apply(*this, X); */

/*   return *this; */
/*   } */



/* template<typename eT> */
/* template<typename T1, typename T2, typename glue_type> */
/* inline */
/* Cube<eT>& */
/* Cube<eT>::operator+=(const mtGlueCube<eT, T1, T2, glue_type>& X) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   const Cube<eT> m(X); */

/*   return (*this).operator+=(m); */
/*   } */



/* template<typename eT> */
/* template<typename T1, typename T2, typename glue_type> */
/* inline */
/* Cube<eT>& */
/* Cube<eT>::operator-=(const mtGlueCube<eT, T1, T2, glue_type>& X) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   const Cube<eT> m(X); */

/*   return (*this).operator-=(m); */
/*   } */



/* template<typename eT> */
/* template<typename T1, typename T2, typename glue_type> */
/* inline */
/* Cube<eT>& */
/* Cube<eT>::operator%=(const mtGlueCube<eT, T1, T2, glue_type>& X) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   const Cube<eT> m(X); */

/*   return (*this).operator%=(m); */
/*   } */



/* template<typename eT> */
/* template<typename T1, typename T2, typename glue_type> */
/* inline */
/* Cube<eT>& */
/* Cube<eT>::operator/=(const mtGlueCube<eT, T1, T2, glue_type>& X) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   const Cube<eT> m(X); */

/*   return (*this).operator/=(m); */
/*   } */



//! linear element accessor (treats the cube as a vector); bounds checking not done when COOT_NO_DEBUG is defined
template<typename eT>
coot_inline
MatValProxy<eT>
Cube<eT>::operator() (const uword i)
  {
  coot_debug_check_bounds( (i >= n_elem), "Cube::operator(): index out of bounds" );

  return MatValProxy<eT>(*this, i);
  }



//! linear element accessor (treats the cube as a vector); bounds checking not done when COOT_NO_DEBUG is defined
template<typename eT>
coot_inline
eT
Cube<eT>::operator() (const uword i) const
  {
  coot_debug_check_bounds( (i >= n_elem), "Cube::operator(): index out of bounds" );

  return MatValProxy<eT>::get_val(*this, i);
  }


//! linear element accessor (treats the cube as a vector); no bounds check.
template<typename eT>
coot_inline
MatValProxy<eT>
Cube<eT>::operator[] (const uword i)
  {
  return MatValProxy<eT>(*this, i);
  }



//! linear element accessor (treats the cube as a vector); no bounds check
template<typename eT>
coot_inline
eT
Cube<eT>::operator[] (const uword i) const
  {
  return MatValProxy<eT>::get_val(*this, i);
  }



//! linear element accessor (treats the cube as a vector); no bounds check.
template<typename eT>
coot_inline
MatValProxy<eT>
Cube<eT>::at(const uword i)
  {
  return MatValProxy<eT>(*this, i);
  }



//! linear element accessor (treats the cube as a vector); no bounds check
template<typename eT>
coot_inline
eT
Cube<eT>::at(const uword i) const
  {
  return MatValProxy<eT>::get_val(*this, i);
  }



//! element accessor; bounds checking not done when COOT_NO_DEBUG is defined
template<typename eT>
coot_inline
MatValProxy<eT>
Cube<eT>::operator() (const uword in_row, const uword in_col, const uword in_slice)
  {
  coot_debug_check_bounds
    (
    (in_row >= n_rows) ||
    (in_col >= n_cols) ||
    (in_slice >= n_slices)
    ,
    "Cube::operator(): index out of bounds"
    );

  return MatValProxy<eT>(*this, in_slice * n_elem_slice + in_col * n_rows + in_row);
  }



//! element accessor; bounds checking not done when COOT_NO_DEBUG is defined
template<typename eT>
coot_inline
eT
Cube<eT>::operator() (const uword in_row, const uword in_col, const uword in_slice) const
  {
  coot_debug_check_bounds
    (
    (in_row >= n_rows) ||
    (in_col >= n_cols) ||
    (in_slice >= n_slices)
    ,
    "Cube::operator(): index out of bounds"
    );

  return MatValProxy<eT>::get_val(*this, in_slice * n_elem_slice + in_col * n_rows + in_row);
  }



//! element accessor; no bounds check
template<typename eT>
coot_inline
MatValProxy<eT>
Cube<eT>::at(const uword in_row, const uword in_col, const uword in_slice)
  {
  return MatValProxy<eT>(*this, in_slice * n_elem_slice + in_col * n_rows + in_row);
  }



//! element accessor; no bounds check */
template<typename eT>
coot_inline
eT
Cube<eT>::at(const uword in_row, const uword in_col, const uword in_slice) const
  {
  return MatValProxy<eT>::get_val(*this, in_slice * n_elem_slice + in_col * n_rows + in_row);
  }



/* //! prefix ++ */
/* template<typename eT> */
/* arma_inline */
/* const Cube<eT>& */
/* Cube<eT>::operator++() */
/*   { */
/*   Cube_aux::prefix_pp(*this); */

/*   return *this; */
/*   } */



/* //! postfix ++  (must not return the object by reference) */
/* template<typename eT> */
/* arma_inline */
/* void */
/* Cube<eT>::operator++(int) */
/*   { */
/*   Cube_aux::postfix_pp(*this); */
/*   } */



/* //! prefix -- */
/* template<typename eT> */
/* arma_inline */
/* const Cube<eT>& */
/* Cube<eT>::operator--() */
/*   { */
/*   Cube_aux::prefix_mm(*this); */
/*   return *this; */
/*   } */



/* //! postfix --  (must not return the object by reference) */
/* template<typename eT> */
/* arma_inline */
/* void */
/* Cube<eT>::operator--(int) */
/*   { */
/*   Cube_aux::postfix_mm(*this); */
/*   } */



// returns true if all of the elements are finite
template<typename eT>
inline
bool
Cube<eT>::is_finite() const
  {
  coot_extra_debug_sigprint();

  // integral types cannot have non-finite values
  if (is_non_integral<eT>::value)
    {
    return !coot_rt_t::any_vec(dev_mem, n_elem, (eT) 0, oneway_real_kernel_id::rel_any_nonfinite, oneway_real_kernel_id::rel_any_nonfinite_small);
    }
  else
    {
    return true;
    }
  }



template<typename eT>
inline
bool
Cube<eT>::has_inf() const
  {
  coot_extra_debug_sigprint();

  // integral types cannot have non-finite values
  if (is_non_integral<eT>::value)
    {
    return coot_rt_t::any_vec(dev_mem, n_elem, (eT) 0, oneway_real_kernel_id::rel_any_inf, oneway_real_kernel_id::rel_any_inf_small);
    }
  else
    {
    return false;
    }
  }



template<typename eT>
inline
bool
Cube<eT>::has_nan() const
  {
  coot_extra_debug_sigprint();

  // integral types cannot have non-finite values
  if (is_non_integral<eT>::value)
    {
    return coot_rt_t::any_vec(dev_mem, n_elem, (eT) 0, oneway_real_kernel_id::rel_any_nan, oneway_real_kernel_id::rel_any_nan_small);
    }
  else
    {
    return false;
    }
  }



//! returns true if the cube has no elements
template<typename eT>
coot_inline
bool
Cube<eT>::is_empty() const
  {
  return (n_elem == 0);
  }



/* //! returns true if the given index is currently in range */
/* template<typename eT> */
/* arma_inline */
/* arma_warn_unused */
/* bool */
/* Cube<eT>::in_range(const uword i) const */
/*   { */
/*   return (i < n_elem); */
/*   } */



/* //! returns true if the given start and end indices are currently in range */
/* template<typename eT> */
/* arma_inline */
/* arma_warn_unused */
/* bool */
/* Cube<eT>::in_range(const span& x) const */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   if(x.whole) */
/*     { */
/*     return true; */
/*     } */
/*   else */
/*     { */
/*     const uword a = x.a; */
/*     const uword b = x.b; */

/*     return ( (a <= b) && (b < n_elem) ); */
/*     } */
/*   } */



/* //! returns true if the given location is currently in range */
/* template<typename eT> */
/* arma_inline */
/* arma_warn_unused */
/* bool */
/* Cube<eT>::in_range(const uword in_row, const uword in_col, const uword in_slice) const */
/*   { */
/*   return ( (in_row < n_rows) && (in_col < n_cols) && (in_slice < n_slices) ); */
/*   } */



/* template<typename eT> */
/* inline */
/* arma_warn_unused */
/* bool */
/* Cube<eT>::in_range(const span& row_span, const span& col_span, const span& slice_span) const */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   const uword in_row1   = row_span.a; */
/*   const uword in_row2   = row_span.b; */

/*   const uword in_col1   = col_span.a; */
/*   const uword in_col2   = col_span.b; */

/*   const uword in_slice1 = slice_span.a; */
/*   const uword in_slice2 = slice_span.b; */


/*   const bool rows_ok   = row_span.whole   ? true : ( (in_row1   <= in_row2)   && (in_row2   < n_rows)   ); */
/*   const bool cols_ok   = col_span.whole   ? true : ( (in_col1   <= in_col2)   && (in_col2   < n_cols)   ); */
/*   const bool slices_ok = slice_span.whole ? true : ( (in_slice1 <= in_slice2) && (in_slice2 < n_slices) ); */


/*   return ( rows_ok && cols_ok && slices_ok ); */
/*   } */



/* template<typename eT> */
/* inline */
/* arma_warn_unused */
/* bool */
/* Cube<eT>::in_range(const uword in_row, const uword in_col, const uword in_slice, const SizeCube& s) const */
/*   { */
/*   const uword l_n_rows   = n_rows; */
/*   const uword l_n_cols   = n_cols; */
/*   const uword l_n_slices = n_slices; */

/*   if( */
/*        ( in_row             >= l_n_rows) || ( in_col             >= l_n_cols) || ( in_slice               >= l_n_slices) */
/*     || ((in_row + s.n_rows) >  l_n_rows) || ((in_col + s.n_cols) >  l_n_cols) || ((in_slice + s.n_slices) >  l_n_slices) */
/*     ) */
/*     { */
/*     return false; */
/*     } */
/*   else */
/*     { */
/*     return true; */
/*     } */
/*   } */



template<typename eT>
inline
dev_mem_t<eT>
Cube<eT>::get_dev_mem(const bool sync) const
  {
  coot_extra_debug_sigprint();

  if (sync) { get_rt().synchronise(); }

  return dev_mem;
  }



//! returns a pointer to array of eTs used by the specified slice in the cube
template<typename eT>
coot_inline
dev_mem_t<eT>
Cube<eT>::slice_get_dev_mem(const uword uslice, const bool synchronise)
  {
  return get_dev_mem(synchronise) + (uslice * n_elem_slice);
  }



//! returns a pointer to array of eTs used by the specified slice in the cube
template<typename eT>
coot_inline
const dev_mem_t<eT>
Cube<eT>::slice_get_dev_mem(const uword uslice, const bool synchronise) const
  {
  return get_dev_mem(synchronise) + (uslice * n_elem_slice);
  }



template<typename eT>
inline
void
Cube<eT>::copy_from_dev_mem(eT* dest_cpu_memptr, const uword N) const
  {
  coot_extra_debug_sigprint();

  if( (n_elem == 0) || (N == 0) )  { return; }

  const uword n_elem_mod = (std::min)(n_elem, N);

  // Treat our device memory as a column vector.
  coot_rt_t::copy_from_dev_mem(dest_cpu_memptr, dev_mem, n_elem_mod, 1, 0, 0, n_elem_mod);
  }



template<typename eT>
inline
void
Cube<eT>::copy_into_dev_mem(const eT* src_cpu_memptr, const uword N)
  {
  coot_extra_debug_sigprint();

  if( (n_elem == 0) || (N == 0) )  { return; }

  const uword n_elem_mod = (std::min)(n_elem, N);

  coot_rt_t::copy_into_dev_mem(dev_mem, src_cpu_memptr, n_elem_mod);
  }



template<typename eT>
inline
Cube<eT>::Cube(const arma::Cube<eT>& X)
  : n_rows(0)
  , n_cols(0)
  , n_elem_slice(0)
  , n_slices(0)
  , n_elem(0)
  , mem_state(0)
  , dev_mem({{ NULL, 0 }})
  {
  coot_extra_debug_sigprint_this(this);

  (*this).operator=(X);
  }



template<typename eT>
inline
Cube<eT>&
Cube<eT>::operator=(const arma::Cube<eT>& X)
  {
  coot_extra_debug_sigprint();

  #if defined(COOT_HAVE_ARMA)
    {
    (*this).set_size(X.n_rows, X.n_cols, X.n_slices);

    (*this).copy_into_dev_mem(X.memptr(), (*this).n_elem);
    }
  #else
    {
    coot_stop_logic_error("#include <armadillo> must be before #include <bandicoot>");
    }
  #endif

  return *this;
  }



template<typename eT>
inline
Cube<eT>::operator arma::Cube<eT> () const
  {
  coot_extra_debug_sigprint();

  #if defined(COOT_HAVE_ARMA)
    {
    arma::Cube<eT> out(n_rows, n_cols, n_slices);

    (*this).copy_from_dev_mem(out.memptr(), (*this).n_elem);

    return out;
    }
  #else
    {
    coot_stop_logic_error("#include <armadillo> must be before #include <bandicoot>");

    return arma::Cube<eT>();
    }
  #endif
  }






//! change the cube to have user specified dimensions (data is not preserved)
template<typename eT>
inline
void
Cube<eT>::set_size(const uword new_n_rows, const uword new_n_cols, const uword new_n_slices)
  {
  coot_extra_debug_sigprint();

  init(new_n_rows, new_n_cols, new_n_slices);
  }



//! change the cube to have user specified dimensions (data is not preserved)
template<typename eT>
inline
void
Cube<eT>::set_size(const SizeCube& s)
  {
  coot_extra_debug_sigprint();

  init(s.n_rows, s.n_cols, s.n_slices);
  }



/* //! change the cube to have user specified dimensions (data is preserved) */
/* template<typename eT> */
/* inline */
/* void */
/* Cube<eT>::reshape(const uword new_n_rows, const uword new_n_cols, const uword new_n_slices) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   op_reshape::apply_cube_inplace((*this), new_n_rows, new_n_cols, new_n_slices); */
/*   } */



/* //! change the cube to have user specified dimensions (data is preserved) */
/* template<typename eT> */
/* inline */
/* void */
/* Cube<eT>::resize(const uword new_n_rows, const uword new_n_cols, const uword new_n_slices) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   op_resize::apply_cube_inplace((*this), new_n_rows, new_n_cols, new_n_slices); */
/*   } */



/* template<typename eT> */
/* inline */
/* void */
/* Cube<eT>::reshape(const SizeCube& s) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   op_reshape::apply_cube_inplace((*this), s.n_rows, s.n_cols, s.n_slices); */
/*   } */



/* template<typename eT> */
/* inline */
/* void */
/* Cube<eT>::resize(const SizeCube& s) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   op_resize::apply_cube_inplace((*this), s.n_rows, s.n_cols, s.n_slices); */
/*   } */



//! change the cube (without preserving data) to have the same dimensions as the given cube
template<typename eT>
template<typename eT2, typename expr>
inline
Cube<eT>&
Cube<eT>::copy_size(const BaseCube<eT2, expr>& m)
  {
  coot_extra_debug_sigprint();

  SizeProxyCube<expr> S(m.get_ref());

  this->set_size(S.get_n_rows(), S.get_n_cols(), S.get_n_slices());

  return *this;
  }



template<typename eT>
inline
Cube<eT>&
Cube<eT>::replace(const eT old_val, const eT new_val)
  {
  coot_extra_debug_sigprint();

  coot_rt_t::replace(dev_mem, dev_mem,
                     old_val, new_val,
                     n_rows, n_cols, n_slices,
                     0, 0, 0, n_rows, n_cols,
                     0, 0, 0, n_rows, n_cols);

  return *this;
  }



/* template<typename eT> */
/* inline */
/* const Cube<eT>& */
/* Cube<eT>::clean(const typename get_pod_type<eT>::result threshold) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   arrayops::clean(memptr(), n_elem, threshold); */

/*   return *this; */
/*   } */



template<typename eT>
inline
Cube<eT>&
Cube<eT>::clamp(const eT min_val, const eT max_val)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (min_val > max_val), "Cube::clamp(): min_val must be less than max_val" );

  coot_rt_t::clamp(dev_mem, dev_mem, min_val, max_val,
                   n_elem_slice, n_slices,
                   0, 0, n_elem_slice,
                   0, 0, n_elem_slice);

  return *this;
  }



//! fill the cube with the specified value
template<typename eT>
inline
Cube<eT>&
Cube<eT>::fill(const eT val)
  {
  coot_extra_debug_sigprint();

  coot_rt_t::fill(dev_mem, val, n_elem_slice, n_slices, 0, 0, n_elem_slice);

  return *this;
  }



template<typename eT>
inline
Cube<eT>&
Cube<eT>::zeros()
  {
  coot_extra_debug_sigprint();

  (*this).fill(eT(0));

  return *this;
  }



template<typename eT>
inline
Cube<eT>&
Cube<eT>::zeros(const uword new_n_rows, const uword new_n_cols, const uword new_n_slices)
  {
  coot_extra_debug_sigprint();

  (*this).set_size(new_n_rows, new_n_cols, new_n_slices);
  (*this).fill(eT(0));

  return *this;
  }



template<typename eT>
inline
Cube<eT>&
Cube<eT>::zeros(const SizeCube& s)
  {
  coot_extra_debug_sigprint();

  (*this).set_size(s);
  (*this).fill(eT(0));

  return *this;
  }



template<typename eT>
inline
Cube<eT>&
Cube<eT>::ones()
  {
  coot_extra_debug_sigprint();

  (*this).fill(eT(1));

  return *this;
  }



template<typename eT>
inline
Cube<eT>&
Cube<eT>::ones(const uword new_n_rows, const uword new_n_cols, const uword new_n_slices)
  {
  coot_extra_debug_sigprint();

  (*this).set_size(new_n_rows, new_n_cols, new_n_slices);
  (*this).fill(eT(1));

  return *this;
  }



template<typename eT>
inline
Cube<eT>&
Cube<eT>::ones(const SizeCube& s)
  {
  coot_extra_debug_sigprint();

  (*this).set_size(s);
  (*this).fill(eT(1));

  return *this;
  }



template<typename eT>
inline
Cube<eT>&
Cube<eT>::randu()
  {
  coot_extra_debug_sigprint();

  coot_rng::fill_randu(dev_mem, n_elem);

  return *this;
  }



template<typename eT>
inline
Cube<eT>&
Cube<eT>::randu(const uword new_n_rows, const uword new_n_cols, const uword new_n_slices)
  {
  coot_extra_debug_sigprint();

  set_size(new_n_rows, new_n_cols, new_n_slices);

  return (*this).randu();
  }



template<typename eT>
inline
Cube<eT>&
Cube<eT>::randu(const SizeCube& s)
  {
  coot_extra_debug_sigprint();

  set_size(s);

  return (*this).randu();
  }



template<typename eT>
inline
Cube<eT>&
Cube<eT>::randn()
  {
  coot_extra_debug_sigprint();

  coot_rng::fill_randn(dev_mem, n_elem);

  return *this;
  }



template<typename eT>
inline
Cube<eT>&
Cube<eT>::randn(const uword new_n_rows, const uword new_n_cols, const uword new_n_slices)
  {
  coot_extra_debug_sigprint();

  set_size(new_n_rows, new_n_cols, new_n_slices);

  return (*this).randn();
  }



template<typename eT>
inline
Cube<eT>&
Cube<eT>::randn(const SizeCube& s)
  {
  coot_extra_debug_sigprint();

  set_size(s);

  return (*this).randn();
  }



template<typename eT>
inline
void
Cube<eT>::reset()
  {
  coot_extra_debug_sigprint();

  init(0, 0, 0);
  }



template<typename eT>
inline
eT
Cube<eT>::min() const
  {
  coot_extra_debug_sigprint();

  if(n_elem == 0)
    {
    coot_debug_check(true, "Cube::min(): object has no elements");

    return Datum<eT>::nan;
    }

  return op_min::apply_direct(*this);
  }



template<typename eT>
inline
eT
Cube<eT>::max() const
  {
  coot_extra_debug_sigprint();

  if(n_elem == 0)
    {
    coot_debug_check(true, "Cube::max(): object has no elements");

    return Datum<eT>::nan;
    }

  return op_max::apply_direct(*this);
  }



/* //! save the cube to a file */
/* template<typename eT> */
/* inline */
/* arma_cold */
/* bool */
/* Cube<eT>::save(const std::string name, const file_type type) const */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   bool save_okay = false; */

/*   switch(type) */
/*     { */
/*     case raw_ascii: */
/*       save_okay = diskio::save_raw_ascii(*this, name); */
/*       break; */

/*     case arma_ascii: */
/*       save_okay = diskio::save_arma_ascii(*this, name); */
/*       break; */

/*     case raw_binary: */
/*       save_okay = diskio::save_raw_binary(*this, name); */
/*       break; */

/*     case arma_binary: */
/*       save_okay = diskio::save_arma_binary(*this, name); */
/*       break; */

/*     case ppm_binary: */
/*       save_okay = diskio::save_ppm_binary(*this, name); */
/*       break; */

/*     case hdf5_binary: */
/*       return (*this).save(hdf5_name(name)); */
/*       break; */

/*     case hdf5_binary_trans:  // kept for compatibility with earlier versions of Armadillo */
/*       return (*this).save(hdf5_name(name, std::string(), hdf5_opts::trans)); */
/*       break; */

/*     default: */
/*       arma_debug_warn_level(1, "Cube::save(): unsupported file type"); */
/*       save_okay = false; */
/*     } */

/*   if(save_okay == false)  { arma_debug_warn_level(3, "Cube::save(): couldn't write; file: ", name); } */

/*   return save_okay; */
/*   } */



/* template<typename eT> */
/* inline */
/* arma_cold */
/* bool */
/* Cube<eT>::save(const hdf5_name& spec, const file_type type) const */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   // handling of hdf5_binary_trans kept for compatibility with earlier versions of Armadillo */

/*   if( (type != hdf5_binary) && (type != hdf5_binary_trans) ) */
/*     { */
/*     arma_stop_runtime_error("Cube::save(): unsupported file type for hdf5_name()"); */
/*     return false; */
/*     } */

/*   const bool do_trans = bool(spec.opts.flags & hdf5_opts::flag_trans  ) || (type == hdf5_binary_trans); */
/*   const bool append   = bool(spec.opts.flags & hdf5_opts::flag_append ); */
/*   const bool replace  = bool(spec.opts.flags & hdf5_opts::flag_replace); */

/*   if(append && replace) */
/*     { */
/*     arma_stop_runtime_error("Cube::save(): only one of 'append' or 'replace' options can be used"); */
/*     return false; */
/*     } */

/*   bool save_okay = false; */
/*   std::string err_msg; */

/*   if(do_trans) */
/*     { */
/*     Cube<eT> tmp; */

/*     op_strans_cube::apply_noalias(tmp, (*this)); */

/*     save_okay = diskio::save_hdf5_binary(tmp, spec, err_msg); */
/*     } */
/*   else */
/*     { */
/*     save_okay = diskio::save_hdf5_binary(*this, spec, err_msg); */
/*     } */

/*   if(save_okay == false) */
/*     { */
/*     if(err_msg.length() > 0) */
/*       { */
/*       arma_debug_warn_level(3, "Cube::save(): ", err_msg, "; file: ", spec.filename); */
/*       } */
/*     else */
/*       { */
/*       arma_debug_warn_level(3, "Cube::save(): couldn't write; file: ", spec.filename); */
/*       } */
/*     } */

/*   return save_okay; */
/*   } */



/* //! save the cube to a stream */
/* template<typename eT> */
/* inline */
/* arma_cold */
/* bool */
/* Cube<eT>::save(std::ostream& os, const file_type type) const */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   bool save_okay = false; */

/*   switch(type) */
/*     { */
/*     case raw_ascii: */
/*       save_okay = diskio::save_raw_ascii(*this, os); */
/*       break; */

/*     case arma_ascii: */
/*       save_okay = diskio::save_arma_ascii(*this, os); */
/*       break; */

/*     case raw_binary: */
/*       save_okay = diskio::save_raw_binary(*this, os); */
/*       break; */

/*     case arma_binary: */
/*       save_okay = diskio::save_arma_binary(*this, os); */
/*       break; */

/*     case ppm_binary: */
/*       save_okay = diskio::save_ppm_binary(*this, os); */
/*       break; */

/*     default: */
/*       arma_debug_warn_level(1, "Cube::save(): unsupported file type"); */
/*       save_okay = false; */
/*     } */

/*   if(save_okay == false)  { arma_debug_warn_level(3, "Cube::save(): couldn't write to stream"); } */

/*   return save_okay; */
/*   } */



/* //! load a cube from a file */
/* template<typename eT> */
/* inline */
/* arma_cold */
/* bool */
/* Cube<eT>::load(const std::string name, const file_type type) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   bool load_okay = false; */
/*   std::string err_msg; */

/*   switch(type) */
/*     { */
/*     case auto_detect: */
/*       load_okay = diskio::load_auto_detect(*this, name, err_msg); */
/*       break; */

/*     case raw_ascii: */
/*       load_okay = diskio::load_raw_ascii(*this, name, err_msg); */
/*       break; */

/*     case arma_ascii: */
/*       load_okay = diskio::load_arma_ascii(*this, name, err_msg); */
/*       break; */

/*     case raw_binary: */
/*       load_okay = diskio::load_raw_binary(*this, name, err_msg); */
/*       break; */

/*     case arma_binary: */
/*       load_okay = diskio::load_arma_binary(*this, name, err_msg); */
/*       break; */

/*     case ppm_binary: */
/*       load_okay = diskio::load_ppm_binary(*this, name, err_msg); */
/*       break; */

/*     case hdf5_binary: */
/*       return (*this).load(hdf5_name(name)); */
/*       break; */

/*     case hdf5_binary_trans:  // kept for compatibility with earlier versions of Armadillo */
/*       return (*this).load(hdf5_name(name, std::string(), hdf5_opts::trans)); */
/*       break; */

/*     default: */
/*       arma_debug_warn_level(1, "Cube::load(): unsupported file type"); */
/*       load_okay = false; */
/*     } */

/*   if(load_okay == false) */
/*     { */
/*     (*this).soft_reset(); */

/*     if(err_msg.length() > 0) */
/*       { */
/*       arma_debug_warn_level(3, "Cube::load(): ", err_msg, "; file: ", name); */
/*       } */
/*     else */
/*       { */
/*       arma_debug_warn_level(3, "Cube::load(): couldn't read; file: ", name); */
/*       } */
/*     } */

/*   return load_okay; */
/*   } */



/* template<typename eT> */
/* inline */
/* arma_cold */
/* bool */
/* Cube<eT>::load(const hdf5_name& spec, const file_type type) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   if( (type != hdf5_binary) && (type != hdf5_binary_trans) ) */
/*     { */
/*     arma_stop_runtime_error("Cube::load(): unsupported file type for hdf5_name()"); */
/*     return false; */
/*     } */

/*   bool load_okay = false; */
/*   std::string err_msg; */

/*   const bool do_trans = bool(spec.opts.flags & hdf5_opts::flag_trans) || (type == hdf5_binary_trans); */

/*   if(do_trans) */
/*     { */
/*     Cube<eT> tmp; */

/*     load_okay = diskio::load_hdf5_binary(tmp, spec, err_msg); */

/*     if(load_okay)  { op_strans_cube::apply_noalias((*this), tmp); } */
/*     } */
/*   else */
/*     { */
/*     load_okay = diskio::load_hdf5_binary(*this, spec, err_msg); */
/*     } */


/*   if(load_okay == false) */
/*     { */
/*     (*this).soft_reset(); */

/*     if(err_msg.length() > 0) */
/*       { */
/*       arma_debug_warn_level(3, "Cube::load(): ", err_msg, "; file: ", spec.filename); */
/*       } */
/*     else */
/*       { */
/*       arma_debug_warn_level(3, "Cube::load(): couldn't read; file: ", spec.filename); */
/*       } */
/*     } */

/*   return load_okay; */
/*   } */



/* //! load a cube from a stream */
/* template<typename eT> */
/* inline */
/* arma_cold */
/* bool */
/* Cube<eT>::load(std::istream& is, const file_type type) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   bool load_okay = false; */
/*   std::string err_msg; */

/*   switch(type) */
/*     { */
/*     case auto_detect: */
/*       load_okay = diskio::load_auto_detect(*this, is, err_msg); */
/*       break; */

/*     case raw_ascii: */
/*       load_okay = diskio::load_raw_ascii(*this, is, err_msg); */
/*       break; */

/*     case arma_ascii: */
/*       load_okay = diskio::load_arma_ascii(*this, is, err_msg); */
/*       break; */

/*     case raw_binary: */
/*       load_okay = diskio::load_raw_binary(*this, is, err_msg); */
/*       break; */

/*     case arma_binary: */
/*       load_okay = diskio::load_arma_binary(*this, is, err_msg); */
/*       break; */

/*     case ppm_binary: */
/*       load_okay = diskio::load_ppm_binary(*this, is, err_msg); */
/*       break; */

/*     default: */
/*       arma_debug_warn_level(1, "Cube::load(): unsupported file type"); */
/*       load_okay = false; */
/*     } */

/*   if(load_okay == false) */
/*     { */
/*     (*this).soft_reset(); */

/*     if(err_msg.length() > 0) */
/*       { */
/*       arma_debug_warn_level(3, "Cube::load(): ", err_msg); */
/*       } */
/*     else */
/*       { */
/*       arma_debug_warn_level(3, "Cube::load(): couldn't load from stream"); */
/*       } */
/*     } */

/*   return load_okay; */
/*   } */



/* template<typename eT> */
/* inline */
/* arma_cold */
/* bool */
/* Cube<eT>::quiet_save(const std::string name, const file_type type) const */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   return (*this).save(name, type); */
/*   } */



/* template<typename eT> */
/* inline */
/* arma_cold */
/* bool */
/* Cube<eT>::quiet_save(const hdf5_name& spec, const file_type type) const */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   return (*this).save(spec, type); */
/*   } */



/* template<typename eT> */
/* inline */
/* arma_cold */
/* bool */
/* Cube<eT>::quiet_save(std::ostream& os, const file_type type) const */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   return (*this).save(os, type); */
/*   } */



/* template<typename eT> */
/* inline */
/* arma_cold */
/* bool */
/* Cube<eT>::quiet_load(const std::string name, const file_type type) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   return (*this).load(name, type); */
/*   } */



/* template<typename eT> */
/* inline */
/* arma_cold */
/* bool */
/* Cube<eT>::quiet_load(const hdf5_name& spec, const file_type type) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   return (*this).load(spec, type); */
/*   } */



/* template<typename eT> */
/* inline */
/* arma_cold */
/* bool */
/* Cube<eT>::quiet_load(std::istream& is, const file_type type) */
/*   { */
/*   arma_extra_debug_sigprint(); */

/*   return (*this).load(is, type); */
/*   } */



//! resets this cube to an empty matrix
template<typename eT>
inline
void
Cube<eT>::clear()
  {
  reset();
  }



//! returns true if the cube has no elements
template<typename eT>
inline
bool
Cube<eT>::empty() const
  {
  return (n_elem == 0);
  }



//! returns the number of elements in this cube
template<typename eT>
inline
uword
Cube<eT>::size() const
  {
  return n_elem;
  }



template<typename eT>
inline
MatValProxy<eT>
Cube<eT>::front()
  {
  coot_debug_check( (n_elem == 0), "Cube::front(): cube is empty" );

  return MatValProxy<eT>(*this, 0);
  }



template<typename eT>
inline
eT
Cube<eT>::front() const
  {
  coot_debug_check( (n_elem == 0), "Cube::front(): cube is empty" );

  return MatValProxy<eT>::get_val(*this, 0);
  }



template<typename eT>
inline
MatValProxy<eT>
Cube<eT>::back()
  {
  coot_debug_check( (n_elem == 0), "Cube::back(): cube is empty" );

  return MatValProxy<eT>(*this, n_elem - 1);
  }



template<typename eT>
inline
eT
Cube<eT>::back() const
  {
  coot_debug_check( (n_elem == 0), "Cube::back(): cube is empty" );

  return MatValProxy<eT>::get_val(*this, n_elem - 1);
  }



/* template<typename eT> */
/* inline */
/* void */
/* Cube<eT>::swap(Cube<eT>& B) */
/*   { */
/*   Cube<eT>& A = (*this); */

/*   arma_extra_debug_sigprint(arma_str::format("A = %x   B = %x") % &A % &B); */

/*   if( (A.mem_state == 0) && (B.mem_state == 0) && (A.n_elem > Cube_prealloc::mem_n_elem) && (B.n_elem > Cube_prealloc::mem_n_elem) ) */
/*     { */
/*     A.delete_mat(); */
/*     B.delete_mat(); */

/*     std::swap( access::rw(A.n_rows),       access::rw(B.n_rows)       ); */
/*     std::swap( access::rw(A.n_cols),       access::rw(B.n_cols)       ); */
/*     std::swap( access::rw(A.n_elem_slice), access::rw(B.n_elem_slice) ); */
/*     std::swap( access::rw(A.n_slices),     access::rw(B.n_slices)     ); */
/*     std::swap( access::rw(A.n_elem),       access::rw(B.n_elem)       ); */
/*     std::swap( access::rw(A.mem),          access::rw(B.mem)          ); */

/*     A.create_mat(); */
/*     B.create_mat(); */
/*     } */
/*   else */
/*   if( (A.mem_state == 0) && (B.mem_state == 0) && (A.n_elem <= Cube_prealloc::mem_n_elem) && (B.n_elem <= Cube_prealloc::mem_n_elem) ) */
/*     { */
/*     A.delete_mat(); */
/*     B.delete_mat(); */

/*     std::swap( access::rw(A.n_rows),       access::rw(B.n_rows)       ); */
/*     std::swap( access::rw(A.n_cols),       access::rw(B.n_cols)       ); */
/*     std::swap( access::rw(A.n_elem_slice), access::rw(B.n_elem_slice) ); */
/*     std::swap( access::rw(A.n_slices),     access::rw(B.n_slices)     ); */
/*     std::swap( access::rw(A.n_elem),       access::rw(B.n_elem)       ); */

/*     const uword N = (std::max)(A.n_elem, B.n_elem); */

/*     eT* A_mem = A.memptr(); */
/*     eT* B_mem = B.memptr(); */

/*     for(uword i=0; i<N; ++i)  { std::swap( A_mem[i], B_mem[i] ); } */

/*     A.create_mat(); */
/*     B.create_mat(); */
/*     } */
/*   else */
/*     { */
/*     // generic swap */

/*     if(A.n_elem <= B.n_elem) */
/*       { */
/*       Cube<eT> C = A; */

/*       A.steal_mem(B); */
/*       B.steal_mem(C); */
/*       } */
/*     else */
/*       { */
/*       Cube<eT> C = B; */

/*       B.steal_mem(A); */
/*       A.steal_mem(C); */
/*       } */
/*     } */
/*   } */



//! try to steal the memory from a given cube;
//! if memory can't be stolen, copy the given cube
template<typename eT>
inline
void
Cube<eT>::steal_mem(Cube<eT>& x)
  {
  coot_extra_debug_sigprint();

  if(this == &x)  { return; }

  if (mem_state == 0 && x.mem_state == 0)
    {
    reset();

    const uword x_n_slices = x.n_slices;

    access::rw(n_rows)       = x.n_rows;
    access::rw(n_cols)       = x.n_cols;
    access::rw(n_elem_slice) = x.n_elem_slice;
    access::rw(n_slices)     = x_n_slices;
    access::rw(n_elem)       = x.n_elem;
    access::rw(mem_state)    = x.mem_state;
    access::rw(dev_mem)      = x.dev_mem;

    if(x_n_slices > Cube_prealloc::mat_ptrs_size)
      {
      access::rw(  mat_ptrs) = x.mat_ptrs;
      access::rw(x.mat_ptrs) = nullptr;
      }
    else
      {
      access::rw(mat_ptrs) = const_cast< Mat<eT>** >(mat_ptrs_local);

      for(uword i=0; i < x_n_slices; ++i)
        {
          mat_ptrs[i] = x.mat_ptrs[i];
        x.mat_ptrs[i] = nullptr;
        }
      }

    access::rw(x.n_rows)       = 0;
    access::rw(x.n_cols)       = 0;
    access::rw(x.n_elem_slice) = 0;
    access::rw(x.n_slices)     = 0;
    access::rw(x.n_elem)       = 0;
    access::rw(x.mem_state)    = 0;
    access::rw(x.dev_mem)      = {{ NULL, 0 }};
    }
  else
    {
    // Either we are an alias or the other cube is an alias, so we have to copy.
    (*this).operator=(x);
    }
  }



#ifdef COOT_EXTRA_CUBE_MEAT
  #include COOT_INCFILE_WRAP(COOT_EXTRA_CUBE_MEAT)
#endif
