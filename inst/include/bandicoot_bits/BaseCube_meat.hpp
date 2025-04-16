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
coot_inline
const derived&
BaseCube<elem_type, derived>::get_ref() const
  {
  return static_cast<const derived&>(*this);
  }



template<typename elem_type, typename derived>
coot_cold
inline
void
BaseCube<elem_type,derived>::print(const std::string extra_text) const
  {
  coot_extra_debug_sigprint();

  const unwrap_cube<derived> tmp( (*this).get_ref() );

  if(extra_text.length() != 0)
    {
    const std::streamsize orig_width = get_cout_stream().width();

    get_cout_stream() << extra_text << '\n';

    get_cout_stream().width(orig_width);
    }

  coot_ostream::print(get_cout_stream(), tmp.M, true);
  }



template<typename elem_type, typename derived>
coot_cold
inline
void
BaseCube<elem_type,derived>::print(std::ostream& user_stream, const std::string extra_text) const
  {
  coot_extra_debug_sigprint();

  const unwrap_cube<derived> tmp( (*this).get_ref() );

  if(extra_text.length() != 0)
    {
    const std::streamsize orig_width = user_stream.width();

    user_stream << extra_text << '\n';

    user_stream.width(orig_width);
    }

  coot_ostream::print(user_stream, tmp.M, true);
  }



template<typename elem_type, typename derived>
coot_cold
inline
void
BaseCube<elem_type,derived>::raw_print(const std::string extra_text) const
  {
  coot_extra_debug_sigprint();

  const unwrap_cube<derived> tmp( (*this).get_ref() );

  if(extra_text.length() != 0)
    {
    const std::streamsize orig_width = get_cout_stream().width();

    get_cout_stream() << extra_text << '\n';

    get_cout_stream().width(orig_width);
    }

  coot_ostream::print(get_cout_stream(), tmp.M, false);
  }



template<typename elem_type, typename derived>
coot_cold
inline
void
BaseCube<elem_type,derived>::raw_print(std::ostream& user_stream, const std::string extra_text) const
  {
  coot_extra_debug_sigprint();

  const unwrap_cube<derived> tmp( (*this).get_ref() );

  if(extra_text.length() != 0)
    {
    const std::streamsize orig_width = user_stream.width();

    user_stream << extra_text << '\n';

    user_stream.width(orig_width);
    }

  coot_ostream::print(user_stream, tmp.M, false);
  }



//template<typename elem_type, typename derived>
//coot_cold
//inline
//void
//BaseCube<elem_type,derived>::brief_print(const std::string extra_text) const
//  {
//  coot_extra_debug_sigprint();
//
//  const unwrap_cube<derived> tmp( (*this).get_ref() );
//
//  if(extra_text.length() != 0)
//    {
//    const std::streamsize orig_width = get_cout_stream().width();
//
//    get_cout_stream() << extra_text << '\n';
//
//    get_cout_stream().width(orig_width);
//    }
//
//  coot_ostream::brief_print(get_cout_stream(), tmp.M);
//  }



//template<typename elem_type, typename derived>
//coot_cold
//inline
//void
//BaseCube<elem_type,derived>::brief_print(std::ostream& user_stream, const std::string extra_text) const
//  {
//  coot_extra_debug_sigprint();
//
//  const unwrap_cube<derived> tmp( (*this).get_ref() );
//
//  if(extra_text.length() != 0)
//    {
//    const std::streamsize orig_width = user_stream.width();
//
//    user_stream << extra_text << '\n';
//
//    user_stream.width(orig_width);
//    }
//
//  coot_ostream::brief_print(user_stream, tmp.M);
//  }



template<typename elem_type, typename derived>
inline
elem_type
BaseCube<elem_type,derived>::min() const
  {
  return op_min::apply_direct( (*this).get_ref() );
  }



template<typename elem_type, typename derived>
inline
elem_type
BaseCube<elem_type,derived>::max() const
  {
  return op_max::apply_direct( (*this).get_ref() );
  }



template<typename elem_type, typename derived>
inline
uword
BaseCube<elem_type,derived>::index_min() const
  {
  return mtop_index_min::apply_direct( (*this).get_ref() );
  }



template<typename elem_type, typename derived>
inline
uword
BaseCube<elem_type,derived>::index_max() const
  {
  return mtop_index_max::apply_direct( (*this).get_ref() );
  }



template<typename elem_type, typename derived>
inline
const CubeToMatOp<derived, op_row_as_mat>
BaseCube<elem_type, derived>::row_as_mat(const uword in_row) const
  {
  return CubeToMatOp<derived, op_row_as_mat>( (*this).get_ref(), in_row );
  }



template<typename elem_type, typename derived>
inline
const CubeToMatOp<derived, op_col_as_mat>
BaseCube<elem_type, derived>::col_as_mat(const uword in_col) const
  {
  return CubeToMatOp<derived, op_col_as_mat>( (*this).get_ref(), in_col );
  }



//
// extra functions defined in BaseCube_eval_Cube

template<typename elem_type, typename derived>
coot_inline
const derived&
BaseCube_eval_Cube<elem_type, derived>::eval() const
  {
  coot_extra_debug_sigprint();

  return static_cast<const derived&>(*this);
  }



//
// extra functions defined in BaseCube_eval_expr

template<typename elem_type, typename derived>
inline
Cube<elem_type>
BaseCube_eval_expr<elem_type, derived>::eval() const
  {
  coot_extra_debug_sigprint();

  return Cube<elem_type>( static_cast<const derived&>(*this) );
  }
