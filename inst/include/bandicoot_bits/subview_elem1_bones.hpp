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



//
// a subview of an object where we are only accessing a vector of indices,
// and that vector is stored in the T1 (subview_elem1.a)
//
// NOTE: this is a very early implementation and backend kernels do not
// support indirect accesses (yet), so it will often extract the matrix
// into a standalone matrix before performing operations!
//
template<typename eT, typename T1>
class subview_elem1 : public Base< eT, subview_elem1<eT,T1> >
  {
  public:

  typedef eT                                       elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;

  static constexpr bool is_row  = false;
  static constexpr bool is_col  = true;
  static constexpr bool is_xvec = false;

  coot_aligned const Mat<eT>         fake_m;
  coot_aligned const Mat<eT>&        m;
  coot_aligned const Base<uword,T1>& a;


  protected:

  coot_inline subview_elem1(const  Mat<eT>& in_m, const Base<uword,T1>& in_a);
  coot_inline subview_elem1(const Cube<eT>& in_q, const Base<uword,T1>& in_a);


  public:

  inline ~subview_elem1();
  inline  subview_elem1() = delete;

                        inline void inplace_op(const twoway_kernel_id::enum_id kernel, const eT val_pre, const eT val_post);
  template<typename T2> inline void inplace_op(const twoway_kernel_id::enum_id kernel_id, const subview_elem1<eT,T2>& x);
  template<typename T2> inline void inplace_op(const twoway_kernel_id::enum_id kernel_id, const Base<eT,T2>&          x);

  coot_inline const Op<subview_elem1<eT,T1>,op_htrans>  t() const;
  coot_inline const Op<subview_elem1<eT,T1>,op_htrans> ht() const;
  coot_inline const Op<subview_elem1<eT,T1>,op_strans> st() const;

  inline void replace(const eT old_val, const eT new_val);

  //inline void clean(const pod_type threshold);

  inline void clamp(const eT min_val, const eT max_val);

  inline void fill(const eT val);
  inline void zeros();
  inline void ones();
  inline void randu();
  inline void randn();

  inline void operator+= (const eT val);
  inline void operator-= (const eT val);
  inline void operator*= (const eT val);
  inline void operator/= (const eT val);


  // deliberately returning void
  template<typename T2> inline void operator=   (const subview_elem1<eT,T2>& x);
                        inline void operator=   (const subview_elem1<eT,T1>& x);
  template<typename T2> inline void operator+=  (const subview_elem1<eT,T2>& x);
  template<typename T2> inline void operator-=  (const subview_elem1<eT,T2>& x);
  template<typename T2> inline void operator%=  (const subview_elem1<eT,T2>& x);
  template<typename T2> inline void operator/=  (const subview_elem1<eT,T2>& x);

  template<typename T2> inline void operator=  (const Base<eT,T2>& x);
  template<typename T2> inline void operator+= (const Base<eT,T2>& x);
  template<typename T2> inline void operator-= (const Base<eT,T2>& x);
  template<typename T2> inline void operator%= (const Base<eT,T2>& x);
  template<typename T2> inline void operator/= (const Base<eT,T2>& x);

  inline static void extract(Mat<eT>& out, const subview_elem1& in);

  friend class  Mat<eT>;
  friend class Cube<eT>;
  };
