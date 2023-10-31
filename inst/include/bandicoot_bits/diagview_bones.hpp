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



//! Class for storing data required to extract and set the diagonals of a matrix
template<typename eT>
class diagview : public Base< eT, diagview<eT> >
  {
  public:

  typedef eT                                elem_type;
  typedef typename get_pod_type<eT>::result pod_type;

  coot_aligned const Mat<eT>& m;
  const uword mem_offset; // offset to first element of diagonal of interest

  // externally-visible values

  static constexpr bool is_row  = false;
  static constexpr bool is_col  = true;
  static constexpr bool is_xvec = false;

  const uword n_rows;     // equal to n_elem
  const uword n_elem;

  static constexpr uword n_cols = 1;


  protected:

  coot_inline diagview(const Mat<eT>& in_m, const uword in_row_offset, const uword in_col_offset, const uword len);


  public:

  inline ~diagview();
  inline  diagview() = delete;

  inline  diagview(const diagview&  in);
  inline  diagview(      diagview&& in);

  inline void operator=(const diagview& x);

  inline void operator+=(const eT val);
  inline void operator-=(const eT val);
  inline void operator*=(const eT val);
  inline void operator/=(const eT val);

  inline void operator= (const Mat<eT>& x);
  inline void operator= (const subview<eT>& x);

  template<typename T1> inline void operator= (const Base<eT,T1>& x);
  template<typename T1> inline void operator+=(const Base<eT,T1>& x);
  template<typename T1> inline void operator-=(const Base<eT,T1>& x);
  template<typename T1> inline void operator%=(const Base<eT,T1>& x);
  template<typename T1> inline void operator/=(const Base<eT,T1>& x);


  coot_warn_unused inline MatValProxy<eT>  operator[](const uword ii);
  coot_warn_unused inline eT               operator[](const uword ii) const;

  coot_warn_unused inline MatValProxy<eT>          at(const uword ii);
  coot_warn_unused inline eT                       at(const uword ii) const;

  coot_warn_unused inline MatValProxy<eT>  operator()(const uword ii);
  coot_warn_unused inline eT               operator()(const uword ii) const;

  coot_warn_unused inline MatValProxy<eT>          at(const uword in_n_row, const uword);
  coot_warn_unused inline eT                       at(const uword in_n_row, const uword) const;

  coot_warn_unused inline MatValProxy<eT>  operator()(const uword in_n_row, const uword in_n_col);
  coot_warn_unused inline eT               operator()(const uword in_n_row, const uword in_n_col) const;


  /* inline void replace(const eT old_val, const eT new_val); */

  /* inline void clean(const pod_type threshold); */

  inline void clamp(const eT min_val, const eT max_val);

  inline void fill(const eT val);
  inline void zeros();
  inline void ones();
  inline void randu();
  inline void randn();

  inline static void extract(Mat<eT>& out, const diagview& in);


  friend class Mat<eT>;
  friend class subview<eT>;
  };
