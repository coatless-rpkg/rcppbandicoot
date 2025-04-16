// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
// Copyright 2023      Marcus Edel (http://www.kurg.org)
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



template<typename elem_type, typename derived>
struct BaseCube_eval_Cube
  {
  coot_warn_unused coot_inline const derived& eval() const;
  };


template<typename elem_type, typename derived>
struct BaseCube_eval_expr
  {
  coot_warn_unused inline Cube<elem_type> eval() const;   //!< force the immediate evaluation of a delayed expression
  };


template<typename elem_type, typename derived, bool condition>
struct BaseCube_eval {};

template<typename elem_type, typename derived>
struct BaseCube_eval<elem_type, derived, true>  { typedef BaseCube_eval_Cube<elem_type, derived>  result; };

template<typename elem_type, typename derived>
struct BaseCube_eval<elem_type, derived, false> { typedef BaseCube_eval_expr<elem_type, derived> result; };



//! Analog of the Base class, intended for cubes
template<typename elem_type, typename derived>
struct BaseCube
  : public BaseCube_eval<elem_type, derived, is_Cube<derived>::value>::result
  {
  coot_inline const derived& get_ref() const;

  coot_cold inline void print(                           const std::string extra_text = "") const;
  coot_cold inline void print(std::ostream& user_stream, const std::string extra_text = "") const;

  coot_cold inline void raw_print(                           const std::string extra_text = "") const;
  coot_cold inline void raw_print(std::ostream& user_stream, const std::string extra_text = "") const;

  //coot_cold inline void brief_print(                           const std::string extra_text = "") const;
  //coot_cold inline void brief_print(std::ostream& user_stream, const std::string extra_text = "") const;

  coot_warn_unused inline elem_type min() const;
  coot_warn_unused inline elem_type max() const;

  coot_warn_unused inline uword index_min() const;
  coot_warn_unused inline uword index_max() const;

  coot_warn_unused inline const CubeToMatOp<derived, op_row_as_mat> row_as_mat(const uword in_row) const;
  coot_warn_unused inline const CubeToMatOp<derived, op_col_as_mat> col_as_mat(const uword in_col) const;
  };
