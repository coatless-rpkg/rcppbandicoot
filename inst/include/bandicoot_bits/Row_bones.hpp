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



// Class for row vectors (matrices with only one row)

template<typename eT>
class Row : public Mat<eT>
  {
  public:

  typedef eT                                elem_type;  // the type of elements stored in the matrix
  typedef typename get_pod_type<eT>::result  pod_type;  // if eT is std::complex<T>, pod_type is T; otherwise pod_type is eT

  static constexpr bool is_col = false;
  static constexpr bool is_row = true;
  static constexpr bool is_xvec = true;

  inline          Row();
  inline explicit Row(const uword N);
  inline explicit Row(const uword in_rows, const uword in_cols);
  inline explicit Row(const SizeMat& s);

  template<typename fill_type> inline Row(const uword N,                            const fill::fill_class<fill_type>& f);
  template<typename fill_type> inline Row(const uword in_rows, const uword in_cols, const fill::fill_class<fill_type>& f);
  template<typename fill_type> inline Row(const SizeMat& s,                         const fill::fill_class<fill_type>& f);

  inline Row(dev_mem_t<eT>                 aux_dev_mem, const uword N);
  inline Row(cl_mem                        aux_dev_mem, const uword N);
  inline Row(typename cuda_type<eT>::type* aux_dev_mem, const uword N);

  inline                  Row(const Row& X);
  inline const Row& operator=(const Row& X);

  inline                  Row(Row&& X);
  inline const Row& operator=(Row&& X);

  inline            Row(const char*        text);
  inline Row& operator=(const char*        text);

  inline            Row(const std::string& text);
  inline Row& operator=(const std::string& text);

  inline            Row(const std::vector<eT>& x);
  inline Row& operator=(const std::vector<eT>& x);

  inline            Row(const std::initializer_list<eT>& list);
  inline Row& operator=(const std::initializer_list<eT>& list);

  template<typename T1> inline            Row(const Base<eT, T1>& X);
  template<typename T1> inline Row& operator=(const Base<eT, T1>& X);

  inline                  Row(const arma::Row<eT>& X);
  inline const Row& operator=(const arma::Row<eT>& X);

  inline explicit operator arma::Row<eT> () const;

  coot_warn_unused inline const Op<Row<eT>, op_htrans>  t() const;
  coot_warn_unused inline const Op<Row<eT>, op_htrans> ht() const;
  coot_warn_unused inline const Op<Row<eT>, op_strans> st() const;

  using Mat<eT>::cols;
  using Mat<eT>::operator();

  coot_inline       subview_row<eT> cols(const uword in_col1, const uword in_col2);
  coot_inline const subview_row<eT> cols(const uword in_col1, const uword in_col2) const;

  coot_inline       subview_row<eT> subvec(const uword in_col1, const uword in_col2);
  coot_inline const subview_row<eT> subvec(const uword in_col1, const uword in_col2) const;

  coot_inline       subview_row<eT> subvec(const uword start_col, const SizeMat& s);
  coot_inline const subview_row<eT> subvec(const uword start_col, const SizeMat& s) const;

  #ifdef COOT_EXTRA_ROW_BONES
    #include COOT_INCFILE_WRAP(COOT_EXTRA_ROW_BONES)
  #endif
  };
