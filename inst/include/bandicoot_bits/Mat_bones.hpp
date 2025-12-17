// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2017-2023 Ryan Curtin (https://www.ratml.org)
// Copyright 2008-2023 Conrad Sanderson (https://conradsanderson.id.au)
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



template<typename eT>
class Mat : public Base< eT, Mat<eT> >
  {
  public:

  typedef eT                                elem_type;  // the type of elements stored in the matrix
  typedef typename get_pod_type<eT>::result  pod_type;  // if eT is std::complex<T>, pod_type is T; otherwise pod_type is eT

  coot_aligned const uword n_rows;    // number of rows     (read-only)
  coot_aligned const uword n_cols;    // number of columns  (read-only)
  coot_aligned const uword n_elem;    // number of elements (read-only)
  coot_aligned const uword vec_state; // 0: matrix layout; 1: column vector layout; 2: row vector layout
  coot_aligned const uword mem_state; // 0: normal; 1: external; 3: fixed size (used by slices)  TODO: should this be expanded to allow re-allocation if size of aux mem is smaller than requested size?

  static constexpr bool is_col = false;
  static constexpr bool is_row = false;
  static constexpr bool is_xvec = false;


  private:

  coot_aligned dev_mem_t<eT> dev_mem;


  public:

  inline ~Mat();
  inline  Mat();

  inline explicit Mat(const uword in_rows, const uword in_cols);
  inline explicit Mat(const SizeMat& s);

  template<typename fill_type> inline Mat(const uword in_n_rows, const uword in_n_cols, const fill::fill_class<fill_type>& f);
  template<typename fill_type> inline Mat(const SizeMat& s,                             const fill::fill_class<fill_type>& f);

  coot_cold inline            Mat(const char*        text);
  coot_cold inline Mat& operator=(const char*        text);

  coot_cold inline            Mat(const std::string& text);
  coot_cold inline Mat& operator=(const std::string& text);

  inline            Mat(const std::vector<eT>& x);
  inline Mat& operator=(const std::vector<eT>& x);

  inline            Mat(const std::initializer_list<eT>& list);
  inline Mat& operator=(const std::initializer_list<eT>& list);

  inline            Mat(const std::initializer_list< std::initializer_list<eT> >& list);
  inline Mat& operator=(const std::initializer_list< std::initializer_list<eT> >& list);

  inline Mat(dev_mem_t<eT>                 aux_dev_mem, const uword in_rows, const uword in_cols);
  inline Mat(cl_mem                        aux_dev_mem, const uword in_rows, const uword in_cols); // OpenCL alias constructor
  inline Mat(typename cuda_type<eT>::type* aux_dev_mem, const uword in_rows, const uword in_cols); // CUDA alias constructor

  inline dev_mem_t<eT> get_dev_mem(const bool sync = true) const;

  inline void  copy_from_dev_mem(      eT* dest_cpu_mem, const uword N) const;  // TODO: rename to copy_into_cpu_mem()
  inline void  copy_into_dev_mem(const eT*  src_cpu_mem, const uword N);

  inline            Mat(const arma::Mat<eT>& X);
  inline Mat& operator=(const arma::Mat<eT>& X);

  inline explicit operator arma::Mat<eT> () const;

  inline Mat& operator= (const eT val);
  inline Mat& operator+=(const eT val);
  inline Mat& operator-=(const eT val);
  inline Mat& operator*=(const eT val);
  inline Mat& operator/=(const eT val);

  inline             Mat(const Mat& X);
  inline Mat& operator= (const Mat& X);
  inline Mat& operator+=(const Mat& X);
  inline Mat& operator-=(const Mat& X);
  inline Mat& operator*=(const Mat& X);
  inline Mat& operator%=(const Mat& X);
  inline Mat& operator/=(const Mat& X);

  inline                  Mat(Mat&& X);
  inline Mat& operator=(Mat&& X);

  inline void steal_mem(Mat& X);  // only for writing code internal to bandicoot

  inline             Mat(const subview<eT>& X);
  inline Mat& operator= (const subview<eT>& X);
  inline Mat& operator+=(const subview<eT>& X);
  inline Mat& operator-=(const subview<eT>& X);
  inline Mat& operator*=(const subview<eT>& X);
  inline Mat& operator%=(const subview<eT>& X);
  inline Mat& operator/=(const subview<eT>& X);

  inline             Mat(const diagview<eT>& X);
  inline Mat& operator= (const diagview<eT>& X);
  inline Mat& operator+=(const diagview<eT>& X);
  inline Mat& operator-=(const diagview<eT>& X);
  inline Mat& operator*=(const diagview<eT>& X);
  inline Mat& operator%=(const diagview<eT>& X);
  inline Mat& operator/=(const diagview<eT>& X);

  template<typename T1> inline             Mat(const subview_elem1<eT,T1>& X);
  template<typename T1> inline Mat& operator= (const subview_elem1<eT,T1>& X);
  template<typename T1> inline Mat& operator+=(const subview_elem1<eT,T1>& X);
  template<typename T1> inline Mat& operator-=(const subview_elem1<eT,T1>& X);
  template<typename T1> inline Mat& operator*=(const subview_elem1<eT,T1>& X);
  template<typename T1> inline Mat& operator%=(const subview_elem1<eT,T1>& X);
  template<typename T1> inline Mat& operator/=(const subview_elem1<eT,T1>& X);

  template<typename T1, typename T2> inline             Mat(const subview_elem2<eT,T1,T2>& X);
  template<typename T1, typename T2> inline Mat& operator= (const subview_elem2<eT,T1,T2>& X);
  template<typename T1, typename T2> inline Mat& operator+=(const subview_elem2<eT,T1,T2>& X);
  template<typename T1, typename T2> inline Mat& operator-=(const subview_elem2<eT,T1,T2>& X);
  template<typename T1, typename T2> inline Mat& operator*=(const subview_elem2<eT,T1,T2>& X);
  template<typename T1, typename T2> inline Mat& operator%=(const subview_elem2<eT,T1,T2>& X);
  template<typename T1, typename T2> inline Mat& operator/=(const subview_elem2<eT,T1,T2>& X);

  template<typename T1, typename eop_type> inline             Mat(const eOp<T1, eop_type>& X);
  template<typename T1, typename eop_type> inline Mat& operator= (const eOp<T1, eop_type>& X);
  template<typename T1, typename eop_type> inline Mat& operator+=(const eOp<T1, eop_type>& X);
  template<typename T1, typename eop_type> inline Mat& operator-=(const eOp<T1, eop_type>& X);
  template<typename T1, typename eop_type> inline Mat& operator*=(const eOp<T1, eop_type>& X);
  template<typename T1, typename eop_type> inline Mat& operator%=(const eOp<T1, eop_type>& X);
  template<typename T1, typename eop_type> inline Mat& operator/=(const eOp<T1, eop_type>& X);

  template<typename T1, typename T2, typename eglue_type> inline             Mat(const eGlue<T1, T2, eglue_type>& X);
  template<typename T1, typename T2, typename eglue_type> inline Mat& operator= (const eGlue<T1, T2, eglue_type>& X);
  template<typename T1, typename T2, typename eglue_type> inline Mat& operator+=(const eGlue<T1, T2, eglue_type>& X);
  template<typename T1, typename T2, typename eglue_type> inline Mat& operator-=(const eGlue<T1, T2, eglue_type>& X);
  template<typename T1, typename T2, typename eglue_type> inline Mat& operator*=(const eGlue<T1, T2, eglue_type>& X);
  template<typename T1, typename T2, typename eglue_type> inline Mat& operator%=(const eGlue<T1, T2, eglue_type>& X);
  template<typename T1, typename T2, typename eglue_type> inline Mat& operator/=(const eGlue<T1, T2, eglue_type>& X);

  template<typename T1, typename mtop_type> inline             Mat(const mtOp<eT, T1, mtop_type>& X);
  template<typename T1, typename mtop_type> inline Mat& operator= (const mtOp<eT, T1, mtop_type>& X);
  template<typename T1, typename mtop_type> inline Mat& operator+=(const mtOp<eT, T1, mtop_type>& X);
  template<typename T1, typename mtop_type> inline Mat& operator-=(const mtOp<eT, T1, mtop_type>& X);
  template<typename T1, typename mtop_type> inline Mat& operator*=(const mtOp<eT, T1, mtop_type>& X);
  template<typename T1, typename mtop_type> inline Mat& operator%=(const mtOp<eT, T1, mtop_type>& X);
  template<typename T1, typename mtop_type> inline Mat& operator/=(const mtOp<eT, T1, mtop_type>& X);

  template<typename T1, typename op_type> inline             Mat(const Op<T1, op_type>& X);
  template<typename T1, typename op_type> inline Mat& operator= (const Op<T1, op_type>& X);
  template<typename T1, typename op_type> inline Mat& operator+=(const Op<T1, op_type>& X);
  template<typename T1, typename op_type> inline Mat& operator-=(const Op<T1, op_type>& X);
  template<typename T1, typename op_type> inline Mat& operator*=(const Op<T1, op_type>& X);
  template<typename T1, typename op_type> inline Mat& operator%=(const Op<T1, op_type>& X);
  template<typename T1, typename op_type> inline Mat& operator/=(const Op<T1, op_type>& X);

  template<typename T1, typename T2, typename glue_type> inline             Mat(const Glue<T1, T2, glue_type>& X);
  template<typename T1, typename T2, typename glue_type> inline Mat& operator= (const Glue<T1, T2, glue_type>& X);
  template<typename T1, typename T2, typename glue_type> inline Mat& operator+=(const Glue<T1, T2, glue_type>& X);
  template<typename T1, typename T2, typename glue_type> inline Mat& operator-=(const Glue<T1, T2, glue_type>& X);
  template<typename T1, typename T2, typename glue_type> inline Mat& operator*=(const Glue<T1, T2, glue_type>& X);
  template<typename T1, typename T2, typename glue_type> inline Mat& operator%=(const Glue<T1, T2, glue_type>& X);
  template<typename T1, typename T2, typename glue_type> inline Mat& operator/=(const Glue<T1, T2, glue_type>& X);

  template<typename T1, typename T2, typename mtglue_type> inline             Mat(const mtGlue<eT, T1, T2, mtglue_type>& X);
  template<typename T1, typename T2, typename mtglue_type> inline Mat& operator= (const mtGlue<eT, T1, T2, mtglue_type>& X);
  template<typename T1, typename T2, typename mtglue_type> inline Mat& operator+=(const mtGlue<eT, T1, T2, mtglue_type>& X);
  template<typename T1, typename T2, typename mtglue_type> inline Mat& operator-=(const mtGlue<eT, T1, T2, mtglue_type>& X);
  template<typename T1, typename T2, typename mtglue_type> inline Mat& operator*=(const mtGlue<eT, T1, T2, mtglue_type>& X);
  template<typename T1, typename T2, typename mtglue_type> inline Mat& operator%=(const mtGlue<eT, T1, T2, mtglue_type>& X);
  template<typename T1, typename T2, typename mtglue_type> inline Mat& operator/=(const mtGlue<eT, T1, T2, mtglue_type>& X);

  template<typename T1, typename op_type> inline             Mat(const CubeToMatOp<T1, op_type>& X);
  template<typename T1, typename op_type> inline Mat& operator= (const CubeToMatOp<T1, op_type>& X);
  template<typename T1, typename op_type> inline Mat& operator+=(const CubeToMatOp<T1, op_type>& X);
  template<typename T1, typename op_type> inline Mat& operator-=(const CubeToMatOp<T1, op_type>& X);
  template<typename T1, typename op_type> inline Mat& operator*=(const CubeToMatOp<T1, op_type>& X);
  template<typename T1, typename op_type> inline Mat& operator%=(const CubeToMatOp<T1, op_type>& X);
  template<typename T1, typename op_type> inline Mat& operator/=(const CubeToMatOp<T1, op_type>& X);

  coot_inline       diagview<eT> diag(const sword in_id = 0);
  coot_inline const diagview<eT> diag(const sword in_id = 0) const;

  inline Mat& replace(const eT old_val, const eT new_val);

  inline Mat& clamp(const eT min_val, const eT max_val);

  inline Mat& fill(const eT val);

  template<typename fill_type>
  inline Mat& fill(const fill::fill_class<fill_type>& f);

  inline Mat& zeros();
  inline Mat& zeros(const uword new_n_elem);
  inline Mat& zeros(const uword new_n_rows, const uword new_n_cols);
  inline Mat& zeros(const SizeMat& s);

  inline Mat& ones();
  inline Mat& ones(const uword new_n_elem);
  inline Mat& ones(const uword new_n_rows, const uword new_n_cols);
  inline Mat& ones(const SizeMat& s);

  inline Mat& randu();
  inline Mat& randu(const uword new_n_elem);
  inline Mat& randu(const uword new_n_elem, const uword new_n_cols);
  inline Mat& randu(const SizeMat& s);

  inline Mat& randn();
  inline Mat& randn(const uword new_n_elem);
  inline Mat& randn(const uword new_n_elem, const uword new_n_cols);
  inline Mat& randn(const SizeMat& s);

  inline Mat& eye();
  inline Mat& eye(const uword new_n_rows, const uword new_n_cols);
  inline Mat& eye(const SizeMat& s);

  template<typename eT2, typename expr>
  inline Mat& copy_size(const Base<eT2, expr>& X);

  inline void reset();
  inline void set_size(const uword new_n_elem);
  inline void set_size(const uword new_n_rows, const uword new_n_cols);
  inline void set_size(const SizeMat& s);

  inline void   resize(const uword new_n_elem);
  inline void   resize(const uword new_n_rows, const uword new_n_cols);
  inline void   resize(const SizeMat& s);

  inline void  reshape(const uword new_n_rows, const uword new_n_cols);
  inline void  reshape(const SizeMat& s);

  coot_warn_unused inline eT min() const;
  coot_warn_unused inline eT max() const;

  coot_warn_unused inline eT min(uword& index_of_min_val) const;
  coot_warn_unused inline eT max(uword& index_of_max_val) const;

  coot_warn_unused inline eT min(uword& row_of_min_val, uword& col_of_min_val) const;
  coot_warn_unused inline eT max(uword& row_of_max_val, uword& col_of_max_val) const;

  coot_warn_unused inline bool is_vec()    const;
  coot_warn_unused inline bool is_colvec() const;
  coot_warn_unused inline bool is_rowvec() const;
  coot_warn_unused inline bool is_square() const;
  coot_warn_unused inline bool is_empty()  const;
  coot_warn_unused inline bool is_finite() const;

  coot_warn_unused inline bool has_inf() const;
  coot_warn_unused inline bool has_nan() const;

  coot_inline uword get_n_rows() const;
  coot_inline uword get_n_cols() const;
  coot_inline uword get_n_elem() const;

  coot_warn_unused inline MatValProxy<eT> operator[] (const uword ii);
  coot_warn_unused inline eT              operator[] (const uword ii) const;
  coot_warn_unused inline MatValProxy<eT> at         (const uword ii);
  coot_warn_unused inline eT              at         (const uword ii) const;
  coot_warn_unused inline MatValProxy<eT> operator() (const uword ii);
  coot_warn_unused inline eT              operator() (const uword ii) const;

  coot_warn_unused inline MatValProxy<eT> at         (const uword in_row, const uword in_col);
  coot_warn_unused inline eT              at         (const uword in_row, const uword in_col) const;
  coot_warn_unused inline MatValProxy<eT> operator() (const uword in_row, const uword in_col);
  coot_warn_unused inline eT              operator() (const uword in_row, const uword in_col) const;

  coot_inline       subview_row<eT> row(const uword row_num);
  coot_inline const subview_row<eT> row(const uword row_num) const;

  inline            subview_row<eT> operator()(const uword row_num, const span& col_span);
  inline      const subview_row<eT> operator()(const uword row_num, const span& col_span) const;


  coot_inline       subview_col<eT> col(const uword col_num);
  coot_inline const subview_col<eT> col(const uword col_num) const;

  inline            subview_col<eT> operator()(const span& row_span, const uword col_num);
  inline      const subview_col<eT> operator()(const span& row_span, const uword col_num) const;


  coot_inline       subview<eT> rows(const uword in_row1, const uword in_row2);
  coot_inline const subview<eT> rows(const uword in_row1, const uword in_row2) const;

  coot_inline       subview<eT> cols(const uword in_col1, const uword in_col2);
  coot_inline const subview<eT> cols(const uword in_col1, const uword in_col2) const;

  inline            subview<eT> rows(const span& row_span);
  inline      const subview<eT> rows(const span& row_span) const;

  coot_inline       subview<eT> cols(const span& col_span);
  coot_inline const subview<eT> cols(const span& col_span) const;

  template<typename T1> coot_inline       subview_elem1<eT,T1> elem(const Base<uword,T1>& a);
  template<typename T1> coot_inline const subview_elem1<eT,T1> elem(const Base<uword,T1>& a) const;

  template<typename T1> coot_inline       subview_elem1<eT,T1> operator()(const Base<uword,T1>& a);
  template<typename T1> coot_inline const subview_elem1<eT,T1> operator()(const Base<uword,T1>& a) const;

  template<typename T1, typename T2> coot_inline       subview_elem2<eT,T1,T2> elem(const Base<uword,T1>& ri, const Base<uword,T2>& ci);
  template<typename T1, typename T2> coot_inline const subview_elem2<eT,T1,T2> elem(const Base<uword,T1>& ri, const Base<uword,T2>& ci) const;

  template<typename T1, typename T2> coot_inline       subview_elem2<eT,T1,T2> submat(const Base<uword,T1>& ri, const Base<uword,T2>& ci);
  template<typename T1, typename T2> coot_inline const subview_elem2<eT,T1,T2> submat(const Base<uword,T1>& ri, const Base<uword,T2>& ci) const;

  template<typename T1, typename T2> coot_inline       subview_elem2<eT,T1,T2> operator()(const Base<uword,T1>& ri, const Base<uword,T2>& ci);
  template<typename T1, typename T2> coot_inline const subview_elem2<eT,T1,T2> operator()(const Base<uword,T1>& ri, const Base<uword,T2>& ci) const;

  template<typename T1> coot_inline       subview_elem2<eT,T1,T1> rows(const Base<uword,T1>& ri);
  template<typename T1> coot_inline const subview_elem2<eT,T1,T1> rows(const Base<uword,T1>& ri) const;

  template<typename T2> coot_inline       subview_elem2<eT,T2,T2> cols(const Base<uword,T2>& ci);
  template<typename T2> coot_inline const subview_elem2<eT,T2,T2> cols(const Base<uword,T2>& ci) const;

  coot_inline       subview_each1<Mat<eT>, 0> each_col();
  coot_inline       subview_each1<Mat<eT>, 1> each_row();

  coot_inline const subview_each1<Mat<eT>, 0> each_col() const;
  coot_inline const subview_each1<Mat<eT>, 1> each_row() const;

  template<typename T1> inline       subview_each2<Mat<eT>, 0, T1> each_col(const Base<uword, T1>& indices);
  template<typename T1> inline       subview_each2<Mat<eT>, 1, T1> each_row(const Base<uword, T1>& indices);

  template<typename T1> inline const subview_each2<Mat<eT>, 0, T1> each_col(const Base<uword, T1>& indices) const;
  template<typename T1> inline const subview_each2<Mat<eT>, 1, T1> each_row(const Base<uword, T1>& indices) const;

  coot_inline       subview<eT> submat(const uword in_row1, const uword in_col1, const uword in_row2, const uword in_col2);
  coot_inline const subview<eT> submat(const uword in_row1, const uword in_col1, const uword in_row2, const uword in_col2) const;

  coot_inline       subview<eT> submat(const uword in_row1, const uword in_col1, const SizeMat& s);
  coot_inline const subview<eT> submat(const uword in_row1, const uword in_col1, const SizeMat& s) const;

  inline            subview<eT> submat    (const span& row_span, const span& col_span);
  inline      const subview<eT> submat    (const span& row_span, const span& col_span) const;

  inline            subview<eT> operator()(const span& row_span, const span& col_span);
  inline      const subview<eT> operator()(const span& row_span, const span& col_span) const;

  inline            subview<eT> operator()(const uword in_row1, const uword in_col1, const SizeMat& s);
  inline      const subview<eT> operator()(const uword in_row1, const uword in_col1, const SizeMat& s) const;

  inline       subview<eT> head_rows(const uword N);
  inline const subview<eT> head_rows(const uword N) const;

  inline       subview<eT> tail_rows(const uword N);
  inline const subview<eT> tail_rows(const uword N) const;

  inline       subview<eT> head_cols(const uword N);
  inline const subview<eT> head_cols(const uword N) const;

  inline       subview<eT> tail_cols(const uword N);
  inline const subview<eT> tail_cols(const uword N) const;

  inline void  clear();
  inline bool  empty() const;
  inline uword size()  const;

  coot_warn_unused inline eT front() const;
  coot_warn_unused inline eT back() const;

  protected:

  // used by Cube slices
  inline Mat(const char junk, dev_mem_t<eT> mem, const uword n_rows, const uword n_cols);

  inline void cleanup();
  inline void init(const uword new_n_rows, const uword new_n_cols);

  coot_cold inline void init(const std::string& text);

  inline void init(const std::initializer_list<eT>& list);
  inline void init(const std::initializer_list<std::initializer_list<eT>>& list);

  friend class subview<eT>;
  friend class MatValProxy<eT>;
  friend class Cube<eT>;


  public:

  #ifdef COOT_EXTRA_MAT_BONES
    #include COOT_INCFILE_WRAP(COOT_EXTRA_MAT_BONES)
  #endif
  };



class Mat_aux
  {
  public:

  template<typename eT> inline static void prefix_pp(Mat<eT>& x);
  template<typename T>  inline static void prefix_pp(Mat< std::complex<T> >& x);

  template<typename eT> inline static void postfix_pp(Mat<eT>& x);
  template<typename T>  inline static void postfix_pp(Mat< std::complex<T> >& x);

  template<typename eT> inline static void prefix_mm(Mat<eT>& x);
  template<typename T>  inline static void prefix_mm(Mat< std::complex<T> >& x);

  template<typename eT> inline static void postfix_mm(Mat<eT>& x);
  template<typename T>  inline static void postfix_mm(Mat< std::complex<T> >& x);

  template<typename eT, typename T1> inline static void set_real(Mat<eT>&                out, const Base<eT,T1>& X);
  template<typename T,  typename T1> inline static void set_real(Mat< std::complex<T> >& out, const Base< T,T1>& X);

  template<typename eT, typename T1> inline static void set_imag(Mat<eT>&                out, const Base<eT,T1>& X);
  template<typename T,  typename T1> inline static void set_imag(Mat< std::complex<T> >& out, const Base< T,T1>& X);
  };
