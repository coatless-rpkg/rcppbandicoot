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



struct Cube_prealloc
  {
  static constexpr uword mat_ptrs_size = 4;
  };



// Dense cube class

template<typename eT>
class Cube : public BaseCube< eT, Cube<eT> >
  {
  public:

  typedef eT                                elem_type; //!< the type of elements stored in the cube
  typedef typename get_pod_type<eT>::result  pod_type; //!< if eT is std::complex<T>, pod_type is T; otherwise pod_type is eT

  const uword  n_rows;       //!< number of rows     in each slice (read-only)
  const uword  n_cols;       //!< number of columns  in each slice (read-only)
  const uword  n_elem_slice; //!< number of elements in each slice (read-only)
  const uword  n_slices;     //!< number of slices   in the cube   (read-only)
  const uword  n_elem;       //!< number of elements in the cube   (read-only)
  const uword  mem_state;

  // mem_state = 0: normal cube which manages its own memory
  // mem_state = 1: use external memory

  coot_aligned dev_mem_t<eT> dev_mem;

  static constexpr bool is_col = false;
  static constexpr bool is_row = false;
  static constexpr bool is_xvec = false;


  protected:

  using mat_type = Mat<eT>;

  #if defined(COOT_USE_OPENMP)
    using    raw_mat_ptr_type = mat_type*;
    using atomic_mat_ptr_type = mat_type*;
  #elif defined(COOT_USE_STD_MUTEX)
    using    raw_mat_ptr_type = mat_type*;
    using atomic_mat_ptr_type = std::atomic<mat_type*>;
  #else
    using    raw_mat_ptr_type = mat_type*;
    using atomic_mat_ptr_type = mat_type*;
  #endif

  atomic_mat_ptr_type* mat_ptrs = nullptr;

  #if defined(COOT_USE_STD_MUTEX)
    mutable std::mutex mat_mutex;   // required for slice()
  #endif

  coot_aligned atomic_mat_ptr_type mat_ptrs_local[ Cube_prealloc::mat_ptrs_size ];

  public:

  inline ~Cube();
  inline  Cube();

  inline explicit Cube(const uword in_n_rows, const uword in_n_cols, const uword in_n_slices);
  inline explicit Cube(const SizeCube& s);

  /* template<bool do_zeros> inline explicit Cube(const uword in_n_rows, const uword in_n_cols, const uword in_n_slices, const arma_initmode_indicator<do_zeros>&); */
  /* template<bool do_zeros> inline explicit Cube(const SizeCube& s,                                                     const arma_initmode_indicator<do_zeros>&); */

  template<typename fill_type> inline Cube(const uword in_n_rows, const uword in_n_cols, const uword in_n_slices, const fill::fill_class<fill_type>& f);
  template<typename fill_type> inline Cube(const SizeCube& s,                                                     const fill::fill_class<fill_type>& f);

  /* inline Cube(const uword in_rows, const uword in_cols, const uword in_slices, const fill::scalar_holder<eT> f); */
  /* inline Cube(const SizeCube& s,                                               const fill::scalar_holder<eT> f); */

  inline            Cube(Cube&& m);
  inline Cube& operator=(Cube&& m);

  inline Cube(dev_mem_t<eT>                 aux_dev_mem, const uword in_rows, const uword in_cols, const uword in_slices);
  inline Cube(cl_mem                        aux_dev_mem, const uword in_rows, const uword in_cols, const uword in_slices); // OpenCL alias constructor
  inline Cube(typename cuda_type<eT>::type* aux_dev_mem, const uword in_rows, const uword in_cols, const uword in_slices); // CUDA alias constructor

  inline dev_mem_t<eT> get_dev_mem(const bool sync = true) const;

  inline void copy_from_dev_mem(      eT* dest_cpu_mem, const uword N) const;
  inline void copy_into_dev_mem(const eT*  src_cpu_mem, const uword N);

  inline            Cube(const arma::Cube<eT>& X);
  inline Cube& operator=(const arma::Cube<eT>& X);

  inline explicit operator arma::Cube<eT>() const;

  inline Cube& operator= (const eT val);
  inline Cube& operator+=(const eT val);
  inline Cube& operator-=(const eT val);
  inline Cube& operator*=(const eT val);
  inline Cube& operator/=(const eT val);

  inline             Cube(const Cube& m);
  inline Cube& operator= (const Cube& m);
  inline Cube& operator+=(const Cube& m);
  inline Cube& operator-=(const Cube& m);
  inline Cube& operator%=(const Cube& m);
  inline Cube& operator/=(const Cube& m);

  /* template<typename T1, typename T2> */
  /* inline explicit Cube(const BaseCube<pod_type,T1>& A, const BaseCube<pod_type,T2>& B); */

  inline             Cube(const subview_cube<eT>& X);
  inline Cube& operator= (const subview_cube<eT>& X);
  inline Cube& operator+=(const subview_cube<eT>& X);
  inline Cube& operator-=(const subview_cube<eT>& X);
  inline Cube& operator%=(const subview_cube<eT>& X);
  inline Cube& operator/=(const subview_cube<eT>& X);

  // TODO: support non-contiguous slice views
  //template<typename T1> inline             Cube(const subview_cube_slices<eT,T1>& X);
  //template<typename T1> inline Cube& operator= (const subview_cube_slices<eT,T1>& X);
  //template<typename T1> inline Cube& operator+=(const subview_cube_slices<eT,T1>& X);
  //template<typename T1> inline Cube& operator-=(const subview_cube_slices<eT,T1>& X);
  //template<typename T1> inline Cube& operator%=(const subview_cube_slices<eT,T1>& X);
  //template<typename T1> inline Cube& operator/=(const subview_cube_slices<eT,T1>& X);

  coot_inline       subview_cube<eT> row(const uword in_row);
  coot_inline const subview_cube<eT> row(const uword in_row) const;

  coot_inline       subview_cube<eT> col(const uword in_col);
  coot_inline const subview_cube<eT> col(const uword in_col) const;

  inline       Mat<eT>& slice(const uword in_slice);
  inline const Mat<eT>& slice(const uword in_slice) const;

  coot_inline       subview_cube<eT> rows(const uword in_row1, const uword in_row2);
  coot_inline const subview_cube<eT> rows(const uword in_row1, const uword in_row2) const;

  coot_inline       subview_cube<eT> cols(const uword in_col1, const uword in_col2);
  coot_inline const subview_cube<eT> cols(const uword in_col1, const uword in_col2) const;

  coot_inline       subview_cube<eT> slices(const uword in_slice1, const uword in_slice2);
  coot_inline const subview_cube<eT> slices(const uword in_slice1, const uword in_slice2) const;

  coot_inline       subview_cube<eT> subcube(const uword in_row1, const uword in_col1, const uword in_slice1, const uword in_row2, const uword in_col2, const uword in_slice2);
  coot_inline const subview_cube<eT> subcube(const uword in_row1, const uword in_col1, const uword in_slice1, const uword in_row2, const uword in_col2, const uword in_slice2) const;

  inline            subview_cube<eT> subcube(const uword in_row1, const uword in_col1, const uword in_slice1, const SizeCube& s);
  inline      const subview_cube<eT> subcube(const uword in_row1, const uword in_col1, const uword in_slice1, const SizeCube& s) const;

  inline            subview_cube<eT> subcube(const span& row_span, const span& col_span, const span& slice_span);
  inline      const subview_cube<eT> subcube(const span& row_span, const span& col_span, const span& slice_span) const;

  inline            subview_cube<eT> operator()(const span& row_span, const span& col_span, const span& slice_span);
  inline      const subview_cube<eT> operator()(const span& row_span, const span& col_span, const span& slice_span) const;

  inline            subview_cube<eT> operator()(const uword in_row1, const uword in_col1, const uword in_slice1, const SizeCube& s);
  inline      const subview_cube<eT> operator()(const uword in_row1, const uword in_col1, const uword in_slice1, const SizeCube& s) const;

  coot_inline       subview_cube<eT> tube(const uword in_row1, const uword in_col1);
  coot_inline const subview_cube<eT> tube(const uword in_row1, const uword in_col1) const;

  coot_inline       subview_cube<eT> tube(const uword in_row1, const uword in_col1, const uword in_row2, const uword in_col2);
  coot_inline const subview_cube<eT> tube(const uword in_row1, const uword in_col1, const uword in_row2, const uword in_col2) const;

  coot_inline       subview_cube<eT> tube(const uword in_row1, const uword in_col1, const SizeMat& s);
  coot_inline const subview_cube<eT> tube(const uword in_row1, const uword in_col1, const SizeMat& s) const;

  inline            subview_cube<eT> tube(const span& row_span, const span& col_span);
  inline      const subview_cube<eT> tube(const span& row_span, const span& col_span) const;

  inline            subview_cube<eT> head_slices(const uword N);
  inline      const subview_cube<eT> head_slices(const uword N) const;

  inline            subview_cube<eT> tail_slices(const uword N);
  inline      const subview_cube<eT> tail_slices(const uword N) const;

  template<typename T1> coot_inline       subview_elem1<eT,T1> elem(const Base<uword,T1>& a);
  template<typename T1> coot_inline const subview_elem1<eT,T1> elem(const Base<uword,T1>& a) const;

  template<typename T1> coot_inline       subview_elem1<eT,T1> operator()(const Base<uword,T1>& a);
  template<typename T1> coot_inline const subview_elem1<eT,T1> operator()(const Base<uword,T1>& a) const;


  // TODO: implement subview_cube_each
  //coot_inline       subview_cube_each1<eT> each_slice();
  //coot_inline const subview_cube_each1<eT> each_slice() const;

  //template<typename T1> inline       subview_cube_each2<eT, T1> each_slice(const Base<uword, T1>& indices);
  //template<typename T1> inline const subview_cube_each2<eT, T1> each_slice(const Base<uword, T1>& indices) const;

  //template<typename T1> coot_inline       subview_cube_slices<eT,T1> slices(const Base<uword,T1>& indices);
  //template<typename T1> coot_inline const subview_cube_slices<eT,T1> slices(const Base<uword,T1>& indices) const;


  /* inline void shed_row(const uword row_num); */
  /* inline void shed_col(const uword col_num); */
  /* inline void shed_slice(const uword slice_num); */

  /* inline void shed_rows(const uword in_row1, const uword in_row2); */
  /* inline void shed_cols(const uword in_col1, const uword in_col2); */
  /* inline void shed_slices(const uword in_slice1, const uword in_slice2); */

  /* template<typename T1> inline void shed_slices(const Base<uword, T1>& indices); */

  /* inline void insert_rows(const uword row_num, const uword N, const bool set_to_zero = true); */
  /* inline void insert_cols(const uword row_num, const uword N, const bool set_to_zero = true); */
  /* inline void insert_slices(const uword slice_num, const uword N, const bool set_to_zero = true); */

  /* template<typename T1> inline void insert_rows(const uword row_num, const BaseCube<eT,T1>& X); */
  /* template<typename T1> inline void insert_cols(const uword col_num, const BaseCube<eT,T1>& X); */
  /* template<typename T1> inline void insert_slices(const uword slice_num, const BaseCube<eT,T1>& X); */


  /* template<typename gen_type> inline             Cube(const GenCube<eT, gen_type>& X); */
  /* template<typename gen_type> inline Cube& operator= (const GenCube<eT, gen_type>& X); */
  /* template<typename gen_type> inline Cube& operator+=(const GenCube<eT, gen_type>& X); */
  /* template<typename gen_type> inline Cube& operator-=(const GenCube<eT, gen_type>& X); */
  /* template<typename gen_type> inline Cube& operator%=(const GenCube<eT, gen_type>& X); */
  /* template<typename gen_type> inline Cube& operator/=(const GenCube<eT, gen_type>& X); */

  template<typename T1, typename op_type> inline             Cube(const OpCube<T1, op_type>& X);
  template<typename T1, typename op_type> inline Cube& operator= (const OpCube<T1, op_type>& X);
  template<typename T1, typename op_type> inline Cube& operator+=(const OpCube<T1, op_type>& X);
  template<typename T1, typename op_type> inline Cube& operator-=(const OpCube<T1, op_type>& X);
  template<typename T1, typename op_type> inline Cube& operator%=(const OpCube<T1, op_type>& X);
  template<typename T1, typename op_type> inline Cube& operator/=(const OpCube<T1, op_type>& X);

  template<typename T1, typename eop_type> inline             Cube(const eOpCube<T1, eop_type>& X);
  template<typename T1, typename eop_type> inline Cube& operator= (const eOpCube<T1, eop_type>& X);
  template<typename T1, typename eop_type> inline Cube& operator+=(const eOpCube<T1, eop_type>& X);
  template<typename T1, typename eop_type> inline Cube& operator-=(const eOpCube<T1, eop_type>& X);
  template<typename T1, typename eop_type> inline Cube& operator%=(const eOpCube<T1, eop_type>& X);
  template<typename T1, typename eop_type> inline Cube& operator/=(const eOpCube<T1, eop_type>& X);

  template<typename T1, typename op_type> inline             Cube(const mtOpCube<eT, T1, op_type>& X);
  template<typename T1, typename op_type> inline Cube& operator= (const mtOpCube<eT, T1, op_type>& X);
  template<typename T1, typename op_type> inline Cube& operator+=(const mtOpCube<eT, T1, op_type>& X);
  template<typename T1, typename op_type> inline Cube& operator-=(const mtOpCube<eT, T1, op_type>& X);
  template<typename T1, typename op_type> inline Cube& operator%=(const mtOpCube<eT, T1, op_type>& X);
  template<typename T1, typename op_type> inline Cube& operator/=(const mtOpCube<eT, T1, op_type>& X);

  template<typename T1, typename T2, typename glue_type> inline             Cube(const GlueCube<T1, T2, glue_type>& X);
  template<typename T1, typename T2, typename glue_type> inline Cube& operator= (const GlueCube<T1, T2, glue_type>& X);
  template<typename T1, typename T2, typename glue_type> inline Cube& operator+=(const GlueCube<T1, T2, glue_type>& X);
  template<typename T1, typename T2, typename glue_type> inline Cube& operator-=(const GlueCube<T1, T2, glue_type>& X);
  template<typename T1, typename T2, typename glue_type> inline Cube& operator%=(const GlueCube<T1, T2, glue_type>& X);
  template<typename T1, typename T2, typename glue_type> inline Cube& operator/=(const GlueCube<T1, T2, glue_type>& X);

  template<typename T1, typename T2, typename eglue_type> inline             Cube(const eGlueCube<T1, T2, eglue_type>& X);
  template<typename T1, typename T2, typename eglue_type> inline Cube& operator= (const eGlueCube<T1, T2, eglue_type>& X);
  template<typename T1, typename T2, typename eglue_type> inline Cube& operator+=(const eGlueCube<T1, T2, eglue_type>& X);
  template<typename T1, typename T2, typename eglue_type> inline Cube& operator-=(const eGlueCube<T1, T2, eglue_type>& X);
  template<typename T1, typename T2, typename eglue_type> inline Cube& operator%=(const eGlueCube<T1, T2, eglue_type>& X);
  template<typename T1, typename T2, typename eglue_type> inline Cube& operator/=(const eGlueCube<T1, T2, eglue_type>& X);

  // TODO
  //template<typename T1, typename T2, typename glue_type> inline             Cube(const mtGlueCube<eT, T1, T2, glue_type>& X);
  //template<typename T1, typename T2, typename glue_type> inline Cube& operator= (const mtGlueCube<eT, T1, T2, glue_type>& X);
  //template<typename T1, typename T2, typename glue_type> inline Cube& operator+=(const mtGlueCube<eT, T1, T2, glue_type>& X);
  //template<typename T1, typename T2, typename glue_type> inline Cube& operator-=(const mtGlueCube<eT, T1, T2, glue_type>& X);
  //template<typename T1, typename T2, typename glue_type> inline Cube& operator%=(const mtGlueCube<eT, T1, T2, glue_type>& X);
  //template<typename T1, typename T2, typename glue_type> inline Cube& operator/=(const mtGlueCube<eT, T1, T2, glue_type>& X);


  coot_warn_unused coot_inline MatValProxy<eT> operator[] (const uword i);
  coot_warn_unused coot_inline eT              operator[] (const uword i) const;

  coot_warn_unused coot_inline MatValProxy<eT> at(const uword i);
  coot_warn_unused coot_inline eT              at(const uword i) const;

  coot_warn_unused coot_inline MatValProxy<eT> operator() (const uword i);
  coot_warn_unused coot_inline eT              operator() (const uword i) const;

  coot_warn_unused coot_inline MatValProxy<eT> at         (const uword in_row, const uword in_col, const uword in_slice);
  coot_warn_unused coot_inline eT              at         (const uword in_row, const uword in_col, const uword in_slice) const;

  coot_warn_unused coot_inline MatValProxy<eT> operator() (const uword in_row, const uword in_col, const uword in_slice);
  coot_warn_unused coot_inline eT              operator() (const uword in_row, const uword in_col, const uword in_slice) const;

  /* arma_inline const Cube& operator++(); */
  /* arma_inline void        operator++(int); */

  /* arma_inline const Cube& operator--(); */
  /* arma_inline void        operator--(int); */

  coot_warn_unused coot_inline bool is_finite() const;
  coot_warn_unused coot_inline bool is_empty()  const;

  coot_warn_unused inline bool has_inf() const;
  coot_warn_unused inline bool has_nan() const;

  /* arma_inline arma_warn_unused bool in_range(const uword i) const; */
  /* arma_inline arma_warn_unused bool in_range(const span& x) const; */

  /* arma_inline arma_warn_unused bool in_range(const uword   in_row, const uword   in_col, const uword   in_slice) const; */
  /*      inline arma_warn_unused bool in_range(const span& row_span, const span& col_span, const span& slice_span) const; */

  /*      inline arma_warn_unused bool in_range(const uword   in_row, const uword   in_col, const uword   in_slice, const SizeCube& s) const; */

  coot_warn_unused coot_inline       dev_mem_t<eT> slice_get_dev_mem(const uword slice, const bool sync = true);
  coot_warn_unused coot_inline const dev_mem_t<eT> slice_get_dev_mem(const uword slice, const bool sync = true) const;

  inline void set_size(const uword new_n_rows, const uword new_n_cols, const uword new_n_slices);
  inline void set_size(const SizeCube& s);

  // TODO
  //inline void reshape(const uword new_n_rows, const uword new_n_cols, const uword new_n_slices);
  //inline void reshape(const SizeCube& s);

  //inline void resize(const uword new_n_rows, const uword new_n_cols, const uword new_n_slices);
  //inline void resize(const SizeCube& s);


  template<typename eT2, typename expr>
  inline Cube& copy_size(const BaseCube<eT2, expr>& X);

  inline Cube& replace(const eT old_val, const eT new_val);

  inline Cube& clamp(const eT min_val, const eT max_val);

  inline Cube& fill(const eT val);

  inline Cube& zeros();
  inline Cube& zeros(const uword new_n_rows, const uword new_n_cols, const uword new_n_slices);
  inline Cube& zeros(const SizeCube& s);

  inline Cube& ones();
  inline Cube& ones(const uword new_n_rows, const uword new_n_cols, const uword new_n_slices);
  inline Cube& ones(const SizeCube& s);

  inline Cube& randu();
  inline Cube& randu(const uword new_n_rows, const uword new_n_cols, const uword new_n_slices);
  inline Cube& randu(const SizeCube& s);

  inline Cube& randn();
  inline Cube& randn(const uword new_n_rows, const uword new_n_cols, const uword new_n_slices);
  inline Cube& randn(const SizeCube& s);

  inline void      reset();

  /* template<typename T1> inline void set_real(const BaseCube<pod_type,T1>& X); */
  /* template<typename T1> inline void set_imag(const BaseCube<pod_type,T1>& X); */

  coot_warn_unused inline eT min() const;
  coot_warn_unused inline eT max() const;

  inline void  clear();
  inline bool  empty() const;
  inline uword size()  const;

  coot_warn_unused inline MatValProxy<eT> front();
  coot_warn_unused inline eT              front() const;

  coot_warn_unused inline MatValProxy<eT> back();
  coot_warn_unused inline eT              back() const;

  inline void steal_mem(Cube& X);  // only for writing code internal to Bandicoot


  protected:

  inline void cleanup();
  inline void init(const uword in_n_rows, const uword in_n_cols, const uword in_n_slices);

  inline void delete_mat();
  inline void create_mat();

  friend class subview_cube<eT>;
  friend class MatValProxy<eT>;


  public:

  #ifdef COOT_EXTRA_CUBE_BONES
    #include COOT_INCFILE_WRAP(COOT_EXTRA_CUBE_BONES)
  #endif
  };
