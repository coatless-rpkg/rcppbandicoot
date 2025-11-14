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


template<typename T1>
struct get_pod_type
  { typedef T1 result; };

template<typename T2>
struct get_pod_type< std::complex<T2> >
  { typedef T2 result; };



template<typename T>
struct is_Mat_only
  { static constexpr bool value = false; };

template<typename eT>
struct is_Mat_only< Mat<eT> >
  { static constexpr bool value = true; };

template<typename eT>
struct is_Mat_only< const Mat<eT> >
  { static constexpr bool value = true; };



template<typename T>
struct is_Mat
  { static constexpr bool value = false; };

template<typename eT>
struct is_Mat< Mat<eT> >
  { static constexpr bool value = true; };

template<typename eT>
struct is_Mat< const Mat<eT> >
  { static constexpr bool value = true; };

template<typename eT>
struct is_Mat< Row<eT> >
  { static constexpr bool value = true; };

template<typename eT>
struct is_Mat< const Row<eT> >
  { static constexpr bool value = true; };

template<typename eT>
struct is_Mat< Col<eT> >
  { static constexpr bool value = true; };

template<typename eT>
struct is_Mat< const Col<eT> >
  { static constexpr bool value = true; };



template<typename T>
struct is_Row
  { static constexpr bool value = false; };

template<typename eT>
struct is_Row< Row<eT> >
  { static constexpr bool value = true; };

template<typename eT>
struct is_Row< const Row<eT> >
  { static constexpr bool value = true; };



template<typename T>
struct is_Col
  { static constexpr bool value = false; };

template<typename eT>
struct is_Col< Col<eT> >
  { static constexpr bool value = true; };

template<typename eT>
struct is_Col< const Col<eT> >
  { static constexpr bool value = true; };



//
//
//



template<typename T>
struct is_arma_Mat
  { static const bool value = false; };

template<typename eT>
struct is_arma_Mat< arma::Mat<eT> >
  { static const bool value = true; };

template<typename eT>
struct is_arma_Mat< const arma::Mat<eT> >
  { static const bool value = true; };

template<typename eT>
struct is_arma_Mat< arma::Col<eT> >
  { static const bool value = true; };

template<typename eT>
struct is_arma_Mat< const arma::Col<eT> >
  { static const bool value = true; };

template<typename eT>
struct is_arma_Mat< arma::Row<eT> >
  { static const bool value = true; };

template<typename eT>
struct is_arma_Mat< const arma::Row<eT> >
  { static const bool value = true; };



//
//
//



template<typename T>
struct is_arma_SpMat
  { static const bool value = false; };

template<typename eT>
struct is_arma_SpMat< arma::SpMat<eT> >
  { static const bool value = true; };

template<typename eT>
struct is_arma_SpMat< const arma::SpMat<eT> >
  { static const bool value = true; };



//
//
//



template<typename T>
struct is_arma_Cube
  { static const bool value = false; };

template<typename eT>
struct is_arma_Cube< arma::Cube<eT> >
  { static const bool value = true; };

template<typename eT>
struct is_arma_Cube< const arma::Cube<eT> >
  { static const bool value = true; };



//
//
//



template<typename T>
struct is_subview
  { static constexpr bool value = false; };

template<typename eT>
struct is_subview< subview<eT> >
  { static constexpr bool value = true; };

template<typename eT>
struct is_subview< const subview<eT> >
  { static constexpr bool value = true; };


template<typename T>
struct is_subview_row
  { static constexpr bool value = false; };

template<typename eT>
struct is_subview_row< subview_row<eT> >
  { static constexpr bool value = true; };

template<typename eT>
struct is_subview_row< const subview_row<eT> >
  { static constexpr bool value = true; };


template<typename T>
struct is_subview_col
  { static constexpr bool value = false; };

template<typename eT>
struct is_subview_col< subview_col<eT> >
  { static constexpr bool value = true; };

template<typename eT>
struct is_subview_col< const subview_col<eT> >
  { static constexpr bool value = true; };


template<typename T>
struct is_diagview
  { static constexpr bool value = false; };

template<typename eT>
struct is_diagview< diagview<eT> >
  { static constexpr bool value = true; };

template<typename eT>
struct is_diagview< const diagview<eT> >
  { static constexpr bool value = true; };



//
//
//


template<typename T>
struct is_Gen
  { static constexpr bool value = false; };

template<typename T1, typename gen_type>
struct is_Gen< Gen<T1,gen_type> >
  { static constexpr bool value = true; };

template<typename T1, typename gen_type>
struct is_Gen< const Gen<T1,gen_type> >
  { static constexpr bool value = true; };


template<typename T>
struct is_Op
  { static constexpr bool value = false; };

template<typename T1, typename op_type>
struct is_Op< Op<T1, op_type> >
  { static constexpr bool value = true; };

template<typename T1, typename op_type>
struct is_Op< const Op<T1, op_type> >
  { static constexpr bool value = true; };


template<typename T>
struct is_CubeToMatOp
  { static constexpr bool value = false; };

template<typename T1, typename op_type>
struct is_CubeToMatOp< CubeToMatOp<T1,op_type> >
  { static constexpr bool value = true; };

template<typename T1, typename op_type>
struct is_CubeToMatOp< const CubeToMatOp<T1,op_type> >
  { static constexpr bool value = true; };


template<typename T>
struct is_eOp
  { static constexpr bool value = false; };

template<typename T1, typename eop_type>
struct is_eOp< eOp<T1, eop_type> >
  { static constexpr bool value = true; };

template<typename T1, typename eop_type>
struct is_eOp< const eOp<T1, eop_type> >
  { static constexpr bool value = true; };


template<typename T>
struct is_mtOp
  { static constexpr bool value = false; };

template<typename eT, typename T1, typename mtop_type>
struct is_mtOp< mtOp<eT, T1, mtop_type> >
  { static constexpr bool value = true; };

template<typename eT, typename T1, typename mtop_type>
struct is_mtOp< const mtOp<eT, T1, mtop_type> >
  { static constexpr bool value = true; };


template<typename T>
struct is_Glue
  { static constexpr bool value = false; };

template<typename T1, typename T2, typename glue_type>
struct is_Glue< Glue<T1, T2, glue_type> >
  { static constexpr bool value = true; };

template<typename T1, typename T2, typename glue_type>
struct is_Glue< const Glue<T1, T2, glue_type> >
  { static constexpr bool value = true; };


template<typename T>
struct is_eGlue
  { static constexpr bool value = false; };

template<typename T1, typename T2, typename eglue_type>
struct is_eGlue< eGlue<T1, T2, eglue_type> >
  { static constexpr bool value = true; };

template<typename T1, typename T2, typename eglue_type>
struct is_eGlue< const eGlue<T1, T2, eglue_type> >
  { static constexpr bool value = true; };


template<typename T>
struct is_mtGlue
  { static constexpr bool value = false; };

template<typename out_eT, typename T1, typename T2, typename mtglue_type>
struct is_mtGlue< mtGlue<out_eT, T1, T2, mtglue_type> >
  { static constexpr bool value = true; };

template<typename out_eT, typename T1, typename T2, typename mtglue_type>
struct is_mtGlue< const mtGlue<out_eT, T1, T2, mtglue_type> >
  { static constexpr bool value = true; };


//
//
//



template<typename T>
struct is_Cube
  { static constexpr bool value = false; };

template<typename eT>
struct is_Cube< Cube<eT> >
  { static constexpr bool value = true; };

template<typename eT>
struct is_Cube< const Cube<eT> >
  { static constexpr bool value = true; };

template<typename T>
struct is_subview_cube
  { static constexpr bool value = false; };

template<typename eT>
struct is_subview_cube< subview_cube<eT> >
  { static constexpr bool value = true; };

template<typename eT>
struct is_subview_cube< const subview_cube<eT> >
  { static constexpr bool value = true; };


//
//
//



template<typename T>
struct is_GenCube
  { static constexpr bool value = false; };

template<typename eT, typename gen_type>
struct is_GenCube< GenCube<eT,gen_type> >
  { static constexpr bool value = true; };

template<typename eT, typename gen_type>
struct is_GenCube< const GenCube<eT,gen_type> >
  { static constexpr bool value = true; };


template<typename T>
struct is_OpCube
  { static constexpr bool value = false; };

template<typename T1, typename op_type>
struct is_OpCube< OpCube<T1,op_type> >
  { static constexpr bool value = true; };

template<typename T1, typename op_type>
struct is_OpCube< const OpCube<T1,op_type> >
  { static constexpr bool value = true; };


template<typename T>
struct is_eOpCube
  { static constexpr bool value = false; };

template<typename T1, typename eop_type>
struct is_eOpCube< eOpCube<T1,eop_type> >
  { static constexpr bool value = true; };

template<typename T1, typename eop_type>
struct is_eOpCube< const eOpCube<T1,eop_type> >
  { static constexpr bool value = true; };


template<typename T>
struct is_mtOpCube
  { static constexpr bool value = false; };

template<typename out_eT, typename T1, typename mtop_type>
struct is_mtOpCube< mtOpCube<out_eT, T1, mtop_type> >
  { static constexpr bool value = true; };

template<typename out_eT, typename T1, typename mtop_type>
struct is_mtOpCube< const mtOpCube<out_eT, T1, mtop_type> >
  { static constexpr bool value = true; };


template<typename T>
struct is_GlueCube
  { static constexpr bool value = false; };

template<typename T1, typename T2, typename glue_type>
struct is_GlueCube< GlueCube<T1,T2,glue_type> >
  { static constexpr bool value = true; };

template<typename T1, typename T2, typename glue_type>
struct is_GlueCube< const GlueCube<T1,T2,glue_type> >
  { static constexpr bool value = true; };


template<typename T>
struct is_eGlueCube
  { static constexpr bool value = false; };

template<typename T1, typename T2, typename eglue_type>
struct is_eGlueCube< eGlueCube<T1,T2,eglue_type> >
  { static constexpr bool value = true; };

template<typename T1, typename T2, typename eglue_type>
struct is_eGlueCube< const eGlueCube<T1,T2,eglue_type> >
  { static constexpr bool value = true; };



//
//
//



template<typename T1>
struct is_coot_type
  {
  static constexpr bool value
  =  is_Mat<T1>::value
  || is_Gen<T1>::value
  || is_Op<T1>::value
  || is_Glue<T1>::value
  || is_eOp<T1>::value
  || is_eGlue<T1>::value
  || is_mtOp<T1>::value
  || is_mtGlue<T1>::value
  || is_subview<T1>::value
  || is_subview_col<T1>::value
  || is_subview_row<T1>::value
  || is_diagview<T1>::value
  || is_CubeToMatOp<T1>::value
  ;
  };



template<typename T1>
struct is_coot_cube_type
  {
  static constexpr bool value
  =  is_Cube<T1>::value
  || is_GenCube<T1>::value
  || is_OpCube<T1>::value
  || is_eOpCube<T1>::value
  || is_mtOpCube<T1>::value
  || is_GlueCube<T1>::value
  || is_eGlueCube<T1>::value
  //|| is_mtGlueCube<T1>::value
  || is_subview_cube<T1>::value
  //|| is_subview_cube_slices<T1>::value
  ;
  };



//
//
//


template<typename T1, typename T2>
struct is_same_type
  {
  static constexpr bool value = false;
  static constexpr bool yes   = false;
  static constexpr bool no    = true;
  };


template<typename T1>
struct is_same_type<T1,T1>
  {
  static constexpr bool value = true;
  static constexpr bool yes   = true;
  static constexpr bool no    = false;
  };



//
//
//


template<typename T1>
struct is_u8
  { static constexpr bool value = false; };

template<>
struct is_u8<u8>
  { static constexpr bool value = true; };



template<typename T1>
struct is_s8
  { static constexpr bool value = false; };

template<>
struct is_s8<s8>
  { static constexpr bool value = true; };



template<typename T1>
struct is_u16
  { static constexpr bool value = false; };

template<>
struct is_u16<u16>
  { static constexpr bool value = true; };



template<typename T1>
struct is_s16
  { static constexpr bool value = false; };

template<>
struct is_s16<s16>
  { static constexpr bool value = true; };



template<typename T1>
struct is_u32
  { static constexpr bool value = false; };

template<>
struct is_u32<u32>
  { static constexpr bool value = true; };



template<typename T1>
struct is_s32
  { static constexpr bool value = false; };

template<>
struct is_s32<s32>
  { static constexpr bool value = true; };



template<typename T1>
struct is_u64
  { static constexpr bool value = false; };

template<>
struct is_u64<u64>
  { static constexpr bool value = true; };


template<typename T1>
struct is_s64
  { static constexpr bool value = false; };

template<>
struct is_s64<s64>
  { static constexpr bool value = true; };



template<typename T1>
struct is_uword
  { static constexpr bool value = false; };

template<>
struct is_uword<uword>
  { static constexpr bool value = true; };



template<typename T1>
struct is_sword
  { static constexpr bool value = false; };

template<>
struct is_sword<sword>
  { static constexpr bool value = true; };



template<typename T1>
struct is_fp16
  { static constexpr bool value = false; };

template<>
struct is_fp16<fp16>
  { static constexpr bool value = true; };



template<typename T1>
struct is_float
  { static constexpr bool value = false; };

template<>
struct is_float<float>
  { static constexpr bool value = true; };



template<typename T1>
struct is_double
  { static constexpr bool value = false; };

template<>
struct is_double<double>
  { static constexpr bool value = true; };



template<typename T1>
struct is_real
  { static constexpr bool value = false; };

template<>
struct is_real<fp16>
  { static constexpr bool value = true; };

template<>
struct is_real<float>
  { static constexpr bool value = true; };

template<>
struct is_real<double>
  { static constexpr bool value = true; };




template<typename T1>
struct is_not_cx
  { static constexpr bool value = true; };

template<typename eT>
struct is_not_cx< std::complex<eT> >
  { static constexpr bool value = false; };



template<typename T1>
struct is_cx_float
  { static constexpr bool value = false; };

template<>
struct is_cx_float< std::complex<float> >
  { static constexpr bool value = true; };



template<typename T1>
struct is_cx_double
  { static constexpr bool value = false; };

template<>
struct is_cx_double< std::complex<double> >
  { static constexpr bool value = true; };



template<typename T1>
struct is_cx_strict
  { static constexpr bool value = false; };

template<>
struct is_cx_strict< std::complex<float> >
  { static constexpr bool value = true; };

template<>
struct is_cx_strict< std::complex<double> >
  { static constexpr bool value = true; };



template<typename T1>
struct is_cx
  {
  static constexpr bool value = false;
  static constexpr bool yes   = false;
  static constexpr bool no    = true;
  };

// template<>
template<typename T>
struct is_cx< std::complex<T> >
  {
  static constexpr bool value = true;
  static constexpr bool yes   = true;
  static constexpr bool no    = false;
  };



// check for a weird implementation of the std::complex class
template<typename T1>
struct is_supported_complex
  { static constexpr bool value = false; };

//template<>
template<typename eT>
struct is_supported_complex< std::complex<eT> >
  { static constexpr bool value = ( sizeof(std::complex<eT>) == 2*sizeof(eT) ); };



template<typename T1>
struct is_supported_complex_float
  { static constexpr bool value = false; };

template<>
struct is_supported_complex_float< std::complex<float> >
  { static constexpr bool value = ( sizeof(std::complex<float>) == 2*sizeof(float) ); };



template<typename T1>
struct is_supported_complex_double
  { static constexpr bool value = false; };

template<>
struct is_supported_complex_double< std::complex<double> >
  { static constexpr bool value = ( sizeof(std::complex<double>) == 2*sizeof(double) ); };



template<typename T1>
struct is_supported_elem_type
  {
  static constexpr bool value = \
    is_u8<T1>::value ||
    is_s8<T1>::value ||
    is_u16<T1>::value ||
    is_s16<T1>::value ||
    is_u32<T1>::value ||
    is_s32<T1>::value ||
    is_u64<T1>::value ||
    is_s64<T1>::value ||
    is_uword<T1>::value ||
    is_sword<T1>::value ||
    is_fp16<T1>::value ||
    is_float<T1>::value ||
    is_double<T1>::value ||
    is_supported_complex_float<T1>::value ||
    is_supported_complex_double<T1>::value;
  };



template<typename T1>
struct is_supported_blas_type
  {
  static constexpr bool value = \
    is_float<T1>::value ||
    is_double<T1>::value ||
    is_supported_complex_float<T1>::value ||
    is_supported_complex_double<T1>::value;
  };



template<typename T1>
struct is_supported_real_blas_type
  {
  static constexpr bool value = \
    is_float<T1>::value ||
    is_double<T1>::value;
  };



template<typename T, bool is_integral>
struct is_signed_helper
  {
  static constexpr bool value = true; // should only be used by float/double/similar
  };

template<typename T>
struct is_signed_helper<T, true>
  {
  static constexpr bool value = std::is_signed<T>::value;
  };

template<typename T>
struct is_signed
  {
  static constexpr bool value = is_signed_helper<T, !is_real<T>::value>::value;
  };



template<typename T>
struct is_non_integral
  {
  static constexpr bool value = false;
  };


template<> struct is_non_integral<              fp16    > { static constexpr bool value = true; };
template<> struct is_non_integral<              float   > { static constexpr bool value = true; };
template<> struct is_non_integral<              double  > { static constexpr bool value = true; };
template<> struct is_non_integral< std::complex<float>  > { static constexpr bool value = true; };
template<> struct is_non_integral< std::complex<double> > { static constexpr bool value = true; };




//

class coot_junk_class;

template<typename T1, typename T2>
struct force_different_type
  {
  typedef T1 T1_result;
  typedef T2 T2_result;
  };


template<typename T1>
struct force_different_type<T1,T1>
  {
  typedef T1              T1_result;
  typedef coot_junk_class T2_result;
  };



//


template<typename T1>
struct resolves_to_vector_default { static constexpr bool value = false;                    };

template<typename T1>
struct resolves_to_vector_test    { static constexpr bool value = T1::is_col || T1::is_row || T1::is_xvec; };


template<typename T1, bool condition>
struct resolves_to_vector_redirect {};

template<typename T1>
struct resolves_to_vector_redirect<T1, false> { typedef resolves_to_vector_default<T1> result; };

template<typename T1>
struct resolves_to_vector_redirect<T1, true>  { typedef resolves_to_vector_test<T1>    result; };


template<typename T1>
struct resolves_to_vector : public resolves_to_vector_redirect<T1, is_coot_type<T1>::value>::result {};


//

template<typename T1>
struct resolves_to_rowvector_default { static constexpr bool value = false;      };

template<typename T1>
struct resolves_to_rowvector_test    { static constexpr bool value = T1::is_row; };


template<typename T1, bool condition>
struct resolves_to_rowvector_redirect {};

template<typename T1>
struct resolves_to_rowvector_redirect<T1, false> { typedef resolves_to_rowvector_default<T1> result; };

template<typename T1>
struct resolves_to_rowvector_redirect<T1, true>  { typedef resolves_to_rowvector_test<T1>    result; };


template<typename T1>
struct resolves_to_rowvector : public resolves_to_rowvector_redirect<T1, is_coot_type<T1>::value>::result {};

//

template<typename T1>
struct resolves_to_colvector_default { static constexpr bool value = false;      };

template<typename T1>
struct resolves_to_colvector_test    { static constexpr bool value = T1::is_col; };


template<typename T1, bool condition>
struct resolves_to_colvector_redirect {};

template<typename T1>
struct resolves_to_colvector_redirect<T1, false> { typedef resolves_to_colvector_default<T1> result; };

template<typename T1>
struct resolves_to_colvector_redirect<T1, true>  { typedef resolves_to_colvector_test<T1>    result; };


template<typename T1>
struct resolves_to_colvector : public resolves_to_colvector_redirect<T1, is_coot_type<T1>::value>::result {};

//

template<typename T1>
struct resolves_to_diagmat
  {
  static constexpr bool value = false;
  };

template<typename T1>
struct resolves_to_diagmat< Op<T1, op_diagmat> >
  {
  static constexpr bool value = true;
  };

template<typename T1, typename eop_type>
struct resolves_to_diagmat< eOp<Op<T1, op_diagmat>, eop_type> >
  {
  static constexpr bool value = true;
  };

template<typename T1>
struct resolves_to_diagmat< Op<Op<T1, op_diagmat>, op_htrans> >
  {
  static constexpr bool value = true;
  };

template<typename T1>
struct resolves_to_diagmat< Op<Op<T1, op_diagmat>, op_htrans2> >
  {
  static constexpr bool value = true;
  };



//

template<typename T1>
struct resolves_to_symmat
  {
  static constexpr bool value = false;
  };

template<typename T1>
struct resolves_to_symmat< Op<T1, op_symmat> >
  {
  static constexpr bool value = true;
  };

template<typename T1, typename eop_type>
struct resolves_to_symmat< eOp<Op<T1, op_symmat>, eop_type> >
  {
  static constexpr bool value = true;
  };

template<typename T1>
struct resolves_to_symmat< Op<Op<T1, op_symmat>, op_htrans> >
  {
  static constexpr bool value = true;
  };

template<typename T1>
struct resolves_to_symmat< Op<Op<T1, op_symmat>, op_htrans2> >
  {
  static constexpr bool value = true;
  };



//

template<typename T>
struct has_nested_op_traits
  {
  typedef char yes[1];
  typedef char  no[2];

  template<typename X> static yes& check(typename X::template traits<void>*);
  template<typename>   static  no& check(...);

  static constexpr bool value = ( sizeof(check<T>(0)) == sizeof(yes) );
  };
