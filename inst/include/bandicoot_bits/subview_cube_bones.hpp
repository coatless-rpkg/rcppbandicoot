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



//! Class for storing data required to construct or apply operations to a subcube
//! (ie. where the subcube starts and ends as well as a reference/pointer to the original cube),
template<typename eT>
class subview_cube : public BaseCube< eT, subview_cube<eT> >
  {
  public:

  typedef eT                                       elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;

  coot_aligned const Cube<eT>& m;

  const uword aux_row1;
  const uword aux_col1;
  const uword aux_slice1;

  const uword n_rows;
  const uword n_cols;
  const uword n_elem_slice;
  const uword n_slices;
  const uword n_elem;


  protected:

  coot_inline subview_cube(const Cube<eT>& in_m, const uword in_row1, const uword in_col1, const uword in_slice1, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices);


  public:

  inline ~subview_cube();
  inline  subview_cube() = delete;

  inline  subview_cube(const subview_cube&  in);
  inline  subview_cube(      subview_cube&& in);

  inline explicit operator arma::Cube<eT>() const;

  inline void operator=  (const eT val);
  inline void operator+= (const eT val);
  inline void operator-= (const eT val);
  inline void operator*= (const eT val);
  inline void operator/= (const eT val);

  // deliberately returning void
  template<typename T1> inline void operator=  (const BaseCube<eT,T1>& x);
  template<typename T1> inline void operator+= (const BaseCube<eT,T1>& x);
  template<typename T1> inline void operator-= (const BaseCube<eT,T1>& x);
  template<typename T1> inline void operator%= (const BaseCube<eT,T1>& x);
  template<typename T1> inline void operator/= (const BaseCube<eT,T1>& x);

  inline void operator=  (const subview_cube& x);
  inline void operator+= (const subview_cube& x);
  inline void operator-= (const subview_cube& x);
  inline void operator%= (const subview_cube& x);
  inline void operator/= (const subview_cube& x);

  template<typename T1> inline void operator=  (const Base<eT,T1>& x);
  template<typename T1> inline void operator+= (const Base<eT,T1>& x);
  template<typename T1> inline void operator-= (const Base<eT,T1>& x);
  template<typename T1> inline void operator%= (const Base<eT,T1>& x);
  template<typename T1> inline void operator/= (const Base<eT,T1>& x);

  // TODO
  //template<typename gen_type> inline void operator=(const GenCube<eT,gen_type>& x);

  inline static void       extract(Cube<eT>& out, const subview_cube& in);
  inline static void  plus_inplace(Cube<eT>& out, const subview_cube& in);
  inline static void minus_inplace(Cube<eT>& out, const subview_cube& in);
  inline static void schur_inplace(Cube<eT>& out, const subview_cube& in);
  inline static void   div_inplace(Cube<eT>& out, const subview_cube& in);

  inline static void       extract(Mat<eT>& out, const subview_cube& in);
  inline static void  plus_inplace(Mat<eT>& out, const subview_cube& in);
  inline static void minus_inplace(Mat<eT>& out, const subview_cube& in);
  inline static void schur_inplace(Mat<eT>& out, const subview_cube& in);
  inline static void   div_inplace(Mat<eT>& out, const subview_cube& in);

  // TODO
  //inline void replace(const eT old_val, const eT new_val);

  // TODO
  //inline void clean(const pod_type threshold);

  inline void clamp(const eT min_val, const eT max_val);

  inline void fill(const eT val);
  inline void zeros();
  inline void ones();
  //inline void randu();
  //inline void randn();

  // TODO
  //coot_warn_unused inline bool is_finite() const;
  //coot_warn_unused inline bool is_zero(const pod_type tol = 0) const;

  // TODO
  //coot_warn_unused inline bool has_inf()       const;
  //coot_warn_unused inline bool has_nan()       const;
  //coot_warn_unused inline bool has_nonfinite() const;

  inline MatValProxy<eT>      operator[](const uword i);
  inline eT                   operator[](const uword i) const;

  inline MatValProxy<eT>      operator()(const uword i);
  inline eT                   operator()(const uword i) const;

  coot_inline MatValProxy<eT> operator()(const uword in_row, const uword in_col, const uword in_slice);
  coot_inline eT              operator()(const uword in_row, const uword in_col, const uword in_slice) const;

  coot_inline MatValProxy<eT> at(const uword in_row, const uword in_col, const uword in_slice);
  coot_inline eT              at(const uword in_row, const uword in_col, const uword in_slice) const;

  coot_warn_unused inline eT front() const;
  coot_warn_unused inline eT back() const;

  coot_inline       dev_mem_t<eT> slice_get_dev_mem(const uword in_slice, const uword in_col);
  coot_inline const dev_mem_t<eT> slice_get_dev_mem(const uword in_slice, const uword in_col) const;

  template<typename eT2>
  inline bool check_overlap(const subview_cube<eT2>& x) const;

  inline bool check_overlap(const Mat<eT>&           x) const;

  friend class  Mat<eT>;
  friend class Cube<eT>;
  };
