// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2008-2016 Conrad Sanderson (https://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
// Copyright 2025      Ryan Curtin (http://ratml.org)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------


template<typename eT, typename sve2_type>
class subview_elem2 : public Base< eT, subview_elem2<eT, sve2_type> >
  {
  public:

  typedef eT                                       elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;

  static constexpr bool is_row  = false;
  static constexpr bool is_col  = false;
  static constexpr bool is_xvec = false;

  coot_aligned const Mat<eT>& m;

  coot_aligned sve2_type r; // holds row and column indices (if given)

  protected:

  template<typename T1, typename T2>
  coot_inline subview_elem2(const Mat<eT>& in_m, const Base<uword,T1>& in_ri, const Base<uword,T2>& in_ci);


  public:

  inline ~subview_elem2();
  inline  subview_elem2() = delete;

  inline void replace(const eT old_val, const eT new_val);

  //inline void clean(const pod_type threshold);

  inline void clamp(const eT min_val, const eT max_val);

  inline void fill(const eT val);
  inline void zeros();
  inline void ones();
  inline void randu();
  inline void randn();

  coot_warn_unused inline bool is_empty() const;

  inline void operator+= (const eT val);
  inline void operator-= (const eT val);
  inline void operator*= (const eT val);
  inline void operator/= (const eT val);

  template<typename sve2_type2>                      inline void inplace_eq(const subview_elem2<eT, sve2_type2>& x);
  template<typename expr>                            inline void inplace_eq(const Base<eT, expr>& x);
  template<typename eglue_type, typename sve2_type2> inline void inplace_op(const subview_elem2<eT, sve2_type2>& x, const char* op_name);
  template<typename eglue_type, typename expr>       inline void inplace_op(const Base<eT, expr>& x, const char* op_name);

  // deliberately returning void
  template<typename sve2_type2> inline void operator=   (const subview_elem2<eT,sve2_type2>& x);
                                inline void operator=   (const subview_elem2<eT,sve2_type>& x);
  template<typename sve2_type2> inline void operator+=  (const subview_elem2<eT,sve2_type2>& x);
  template<typename sve2_type2> inline void operator-=  (const subview_elem2<eT,sve2_type2>& x);
  template<typename sve2_type2> inline void operator%=  (const subview_elem2<eT,sve2_type2>& x);
  template<typename sve2_type2> inline void operator/=  (const subview_elem2<eT,sve2_type2>& x);

  template<typename expr> inline void operator=  (const Base<eT,expr>& x);
  template<typename expr> inline void operator+= (const Base<eT,expr>& x);
  template<typename expr> inline void operator-= (const Base<eT,expr>& x);
  template<typename expr> inline void operator%= (const Base<eT,expr>& x);
  template<typename expr> inline void operator/= (const Base<eT,expr>& x);

  inline static void extract(Mat<eT>& out, const subview_elem2& in);


  friend class Mat<eT>;
  };



template<typename eT, typename T1, typename T2>
class subview_elem2_both
  {
  public:

  coot_aligned const Base<uword,T1>& base_ri;
  coot_aligned const Base<uword,T2>& base_ci;

  coot_inline subview_elem2_both(const Base<uword,T1>& in_ri, const Base<uword,T2>& in_ci);

  inline void randu(subview_elem2<eT, subview_elem2_both<eT, T1, T2>>& base);
  inline void randn(subview_elem2<eT, subview_elem2_both<eT, T1, T2>>& base);

  inline bool is_empty(const subview_elem2<eT, subview_elem2_both<eT, T1, T2>>& s) const;
  };

template<typename eT, typename T1>
class subview_elem2_all_cols
  {
  public:

  coot_aligned const Base<uword,T1>& base_ri;

  template<typename T2>
  coot_inline subview_elem2_all_cols(const Base<uword,T1>& in_ri, const Base<uword,T2>& in_ci /* ignored */);

  inline void randu(subview_elem2<eT, subview_elem2_all_cols<eT, T1>>& base);
  inline void randn(subview_elem2<eT, subview_elem2_all_cols<eT, T1>>& base);

  inline bool is_empty(const subview_elem2<eT, subview_elem2_all_cols<eT, T1>>& s) const;
  };

template<typename eT, typename T2>
class subview_elem2_all_rows
  {
  public:

  coot_aligned const Base<uword,T2>& base_ci;

  template<typename T1>
  coot_inline subview_elem2_all_rows(const Base<uword,T1>& in_ri /* ignored */, const Base<uword,T2>& in_ci);

  inline void randu(subview_elem2<eT, subview_elem2_all_rows<eT, T2>>& base);
  inline void randn(subview_elem2<eT, subview_elem2_all_rows<eT, T2>>& base);

  inline bool is_empty(const subview_elem2<eT, subview_elem2_all_rows<eT, T2>>& s) const;
  };
