// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2017-2023 Ryan Curtin (https://www.ratml.org)
// Copyright 2008-2017 Conrad Sanderson (https://conradsanderson.id.au)
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



template<typename eglue_type>
struct eglue_core
  {

  // matrices

  template<typename eT3, typename T1, typename T2> inline static void apply              (Mat<eT3>& out, const eGlue<T1, T2, eglue_type>& x);
  template<typename eT3, typename T1, typename T2> inline static void apply_inplace_plus (Mat<eT3>& out, const eGlue<T1, T2, eglue_type>& x);
  template<typename eT3, typename T1, typename T2> inline static void apply_inplace_minus(Mat<eT3>& out, const eGlue<T1, T2, eglue_type>& x);
  template<typename eT3, typename T1, typename T2> inline static void apply_inplace_schur(Mat<eT3>& out, const eGlue<T1, T2, eglue_type>& x);
  template<typename eT3, typename T1, typename T2> inline static void apply_inplace_div  (Mat<eT3>& out, const eGlue<T1, T2, eglue_type>& x);

  // cubes

  template<typename eT3, typename T1, typename T2> inline static void apply              (Cube<eT3>& out, const eGlueCube<T1, T2, eglue_type>& x);
  template<typename eT3, typename T1, typename T2> inline static void apply_inplace_plus (Cube<eT3>& out, const eGlueCube<T1, T2, eglue_type>& x);
  template<typename eT3, typename T1, typename T2> inline static void apply_inplace_minus(Cube<eT3>& out, const eGlueCube<T1, T2, eglue_type>& x);
  template<typename eT3, typename T1, typename T2> inline static void apply_inplace_schur(Cube<eT3>& out, const eGlueCube<T1, T2, eglue_type>& x);
  template<typename eT3, typename T1, typename T2> inline static void apply_inplace_div  (Cube<eT3>& out, const eGlueCube<T1, T2, eglue_type>& x);
  };



class eglue_plus : public eglue_core<eglue_plus>
  {
  public:

  inline static const char* text() { return "addition"; }
  };



class eglue_minus : public eglue_core<eglue_minus>
  {
  public:

  inline static const char* text() { return "subtraction"; }
  };



class eglue_div : public eglue_core<eglue_div>
  {
  public:

  inline static const char* text() { return "element-wise division"; }
  };



class eglue_schur : public eglue_core<eglue_schur>
  {
  public:

  inline static const char* text() { return "element-wise multiplication"; }
  };



class eglue_atan2 : public eglue_core<eglue_atan2>
  {
  public:

  inline static const char* text() { return "element-wise atan2"; }
  };



class eglue_hypot : public eglue_core<eglue_hypot>
  {
  public:

  inline static const char* text() { return "element-wise hypot"; }
  };
