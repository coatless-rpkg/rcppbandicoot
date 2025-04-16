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



template<typename elem_type, typename derived>
coot_inline
const derived&
Base<elem_type,derived>::get_ref() const
  {
  return static_cast<const derived&>(*this);
  }



template<typename elem_type, typename derived>
inline
void
Base<elem_type,derived>::print(const std::string extra_text) const
  {
  coot_extra_debug_sigprint();

  const unwrap<derived> tmp( (*this).get_ref() );

  if(extra_text.length() != 0)
    {
    const std::streamsize orig_width = get_cout_stream().width();

    get_cout_stream() << extra_text << '\n';

    get_cout_stream().width(orig_width);
    }

  coot_ostream::print(get_cout_stream(), tmp.M, true);
  }




template<typename elem_type, typename derived>
inline
void
Base<elem_type,derived>::print(std::ostream& user_stream, const std::string extra_text) const
  {
  coot_extra_debug_sigprint();

  const unwrap<derived> tmp( (*this).get_ref() );

  if(extra_text.length() != 0)
    {
    const std::streamsize orig_width = user_stream.width();

    user_stream << extra_text << '\n';

    user_stream.width(orig_width);
    }

  coot_ostream::print(user_stream, tmp.M, true);
  }



template<typename elem_type, typename derived>
inline
void
Base<elem_type,derived>::raw_print(const std::string extra_text) const
  {
  coot_extra_debug_sigprint();

  const unwrap<derived> tmp( (*this).get_ref() );

  if(extra_text.length() != 0)
    {
    const std::streamsize orig_width = get_cout_stream().width();

    get_cout_stream() << extra_text << '\n';

    get_cout_stream().width(orig_width);
    }

  coot_ostream::print(get_cout_stream(), tmp.M, false);
  }



template<typename elem_type, typename derived>
inline
void
Base<elem_type,derived>::raw_print(std::ostream& user_stream, const std::string extra_text) const
  {
  coot_extra_debug_sigprint();

  const unwrap<derived> tmp( (*this).get_ref() );

  if(extra_text.length() != 0)
    {
    const std::streamsize orig_width = user_stream.width();

    user_stream << extra_text << '\n';

    user_stream.width(orig_width);
    }

  coot_ostream::print(user_stream, tmp.M, false);
  }



//
// extra functions defined in Base_inv_yes

template<typename derived>
inline
const Op<derived, op_inv>
Base_inv_yes<derived>::i() const
  {
  return Op<derived, op_inv>(static_cast<const derived&>(*this));
  }



//
// extra functions defined in Base_eval_Mat

template<typename elem_type, typename derived>
inline
const derived&
Base_eval_Mat<elem_type, derived>::eval() const
  {
  coot_extra_debug_sigprint();

  return static_cast<const derived&>(*this);
  }



//
// extra functions defined in Base_eval_expr

template<typename elem_type, typename derived>
inline
Mat<elem_type>
Base_eval_expr<elem_type, derived>::eval() const
  {
  coot_extra_debug_sigprint();

  return Mat<elem_type>( static_cast<const derived&>(*this) );
  }



//
// extra functions defined in Base_trans_cx

template<typename derived>
inline
const Op<derived, op_htrans>
Base_trans_cx<derived>::t() const
  {
  return Op<derived, op_htrans>( static_cast<const derived&>(*this) );
  }



template<typename derived>
inline
const Op<derived, op_htrans>
Base_trans_cx<derived>::ht() const
  {
  return Op<derived, op_htrans>( static_cast<const derived&>(*this) );
  }



template<typename derived>
inline
const Op<derived, op_strans>
Base_trans_cx<derived>::st() const
  {
  return Op<derived, op_strans>( static_cast<const derived&>(*this) );
  }



//
// extra functions defined in Base_trans_default

template<typename derived>
inline
const Op<derived, op_htrans>
Base_trans_default<derived>::t() const
  {
  return Op<derived, op_htrans>( static_cast<const derived&>(*this) );
  }



template<typename derived>
inline
const Op<derived, op_htrans>
Base_trans_default<derived>::ht() const
  {
  return Op<derived, op_htrans>( static_cast<const derived&>(*this) );
  }



template<typename derived>
inline
const Op<derived, op_htrans>
Base_trans_default<derived>::st() const
  {
  return Op<derived, op_htrans>( static_cast<const derived&>(*this) );
  }



template<typename elem_type, typename derived>
inline
elem_type
Base<elem_type,derived>::min() const
  {
  return op_min::apply_direct( (*this).get_ref() );
  }



template<typename elem_type, typename derived>
inline
elem_type
Base<elem_type,derived>::max() const
  {
  return op_max::apply_direct( (*this).get_ref() );
  }



template<typename elem_type, typename derived>
inline
elem_type
Base<elem_type,derived>::min(uword& index_of_min_val) const
  {
  // We have to actually unwrap and evaluate.
  const unwrap<derived> U( (*this).get_ref() );

  index_of_min_val = mtop_index_min::apply_direct(U.M);

  return U.M.at(index_of_min_val);
  }



template<typename elem_type, typename derived>
inline
elem_type
Base<elem_type,derived>::max(uword& index_of_max_val) const
  {
  // We have to actually unwrap and evaluate.
  const unwrap<derived> U( (*this).get_ref() );

  index_of_max_val = mtop_index_max::apply_direct(U.M);

  return U.M.at(index_of_max_val);
  }



template<typename elem_type, typename derived>
inline
elem_type
Base<elem_type,derived>::min(uword& row_of_min_val, uword& col_of_min_val) const
  {
  const unwrap<derived> U( (*this).get_ref() );

  uword index = mtop_index_min::apply_direct(U.M);

  const uword local_n_rows = U.M.n_rows;

  row_of_min_val = index % local_n_rows;
  col_of_min_val = index / local_n_rows;

  return U.M.at(index);
  }



template<typename elem_type, typename derived>
inline
elem_type
Base<elem_type,derived>::max(uword& row_of_max_val, uword& col_of_max_val) const
  {

  const unwrap<derived> U( (*this).get_ref() );

  uword index = mtop_index_max::apply_direct(U.M);

  const uword local_n_rows = U.M.n_rows;

  row_of_max_val = index % local_n_rows;
  col_of_max_val = index / local_n_rows;

  return U.M.at(index);
  }



template<typename elem_type, typename derived>
inline
uword
Base<elem_type,derived>::index_min() const
  {
  return mtop_index_min::apply_direct( (*this).get_ref() );
  }



template<typename elem_type, typename derived>
inline
uword
Base<elem_type,derived>::index_max() const
  {
  return mtop_index_max::apply_direct( (*this).get_ref() );
  }
