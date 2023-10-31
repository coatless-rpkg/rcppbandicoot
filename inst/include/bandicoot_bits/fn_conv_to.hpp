// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2020 Ryan Curtin (http://www.ratml.org
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



template<typename T2>
struct conv_to
  {
  static_assert( is_Mat<T2>::value || is_arma_Mat<T2>::value,
      "conv_to<T> can only be used with type T in the set { coot::Mat<eT>, coot::Row<eT>, coot::Col<eT>, arma::Mat<eT>, arma::Row<eT>, arma::Col<eT> }");
  typedef typename T2::elem_type out_eT;

  // Perform a conversion from one type to another.  This returns an mtOp,
  // because other operations in the codebase might be able to use a multi-type
  // kernel to reduce the number of kernels needed.
  template<typename T1>
  inline
  static
  typename enable_if2<
      !is_same_type<typename T1::elem_type, out_eT>::value && is_coot_type<T1>::value && is_coot_type<T2>::value,
      mtOp<out_eT, T1, mtop_conv_to>
  >::result
  from(const T1& in)
    {
    coot_extra_debug_sigprint();

    return mtOp<out_eT, T1, mtop_conv_to>(in);
    }



  // Dummy overload for when in_eT == out_eT---in this case, just return the
  // input; it's already the right type.
  template<typename T1>
  inline
  static
  typename enable_if2<
      is_same_type<out_eT, typename T1::elem_type>::value && is_coot_type<T1>::value && is_coot_type<T2>::value,
      T1
  >::result
  from(const T1& in)
    {
    coot_extra_debug_sigprint();

    return in;
    }



  // When we get an Op or a Glue, there's no hope of delaying the conversion.
  // (Exception: when it's an op_htrans or op_htrans2.)
  // Note that we always expect an Op or Glue to be able to output into a different type.
  template<typename T1, typename op_type>
  inline
  static
  typename enable_if2<
      is_same_type<typename T1::elem_type, out_eT>::no && is_coot_type<T2>::value,
      T2
  >::result
  from(const Op<T1, op_type>& in)
    {
    coot_extra_debug_sigprint();

    T2 out; // must be Mat, Row, or Col
    op_type::apply(out, in);

    return out;
    }



  template<typename T1, typename T3, typename glue_type>
  inline
  static
  typename enable_if2<
      is_same_type<typename T1::elem_type, out_eT>::no && is_coot_type<T2>::value,
      T2
  >::result
  from(const Glue<T1, T3, glue_type>& in)
    {
    coot_extra_debug_sigprint();

    T2 out; // must be Mat, Row, or Col
    glue_type::apply(out, in);

    return out;
    }



  // TODO: use traits instead
  template<typename T1>
  inline
  static
  typename enable_if2<
      is_same_type<typename T1::elem_type, out_eT>::no && is_coot_type<T2>::value,
      mtOp<out_eT, Op<T1, op_htrans>, mtop_conv_to>
  >::result
  from(const Op<T1, op_htrans>& in)
    {
    coot_extra_debug_sigprint();

    return mtOp<out_eT, Op<T1, op_htrans>, mtop_conv_to>(in);
    }



  template<typename T1>
  inline
  static
  typename enable_if2<
      is_same_type<typename T1::elem_type, out_eT>::no && is_coot_type<T2>::value,
      mtOp<out_eT, Op<T1, op_htrans2>, mtop_conv_to>
  >::result
  from(const Op<T1, op_htrans2>& in)
    {
    coot_extra_debug_sigprint();

    return mtOp<out_eT, Op<T1, op_htrans2>, mtop_conv_to>(in);
    }



  // Conversions from Armadillo (could include Armadillo-Armadillo conversions).
  template<typename eT, typename T1>
  inline
  static
  T2
  from(const arma::Base<eT, T1>& in)
    {
    coot_extra_debug_sigprint();

    #if defined(COOT_HAVE_ARMA)
      {
      arma::Mat<out_eT> M = arma::conv_to<arma::Mat<out_eT>>::from(in);
      T2 out(M); // must be Mat, Row, or Col (either Bandicoot or Armadillo)

      return out;
      }
    #else
      {
      coot_stop_logic_error("#include <armadillo> must be before #include <bandicoot>");

      return T2();
      }
    #endif
    }



  // Bandicoot to Armadillo conversion.
  template<typename T1>
  inline
  static
  typename enable_if2<
      is_coot_type<T1>::value && is_arma_Mat<T2>::value,
      T2
  >::result
  from(const T1& in)
    {
    coot_extra_debug_sigprint();

    #if defined(COOT_HAVE_ARMA)
      {
      typedef typename T1::elem_type eT;

      unwrap<T1> U(in);
      arma::Mat<eT> M(U.M); // also works with subviews

      T2 out = arma::conv_to<T2>::from(M);

      return out;
      }
    #else
      {
      coot_stop_logic_error("#include <armadillo> must be before #include <bandicoot>");

      return T2();
      }
    #endif
    }
  };
