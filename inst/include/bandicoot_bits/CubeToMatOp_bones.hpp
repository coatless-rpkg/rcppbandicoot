// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
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



template<typename T1, typename op_type>
class CubeToMatOp : public Base< typename T1::elem_type, CubeToMatOp<T1, op_type> >
  {
  public:

  typedef typename T1::elem_type                   elem_type;
  typedef typename get_pod_type<elem_type>::result pod_type;

  inline explicit CubeToMatOp(const T1& in_m);
  inline          CubeToMatOp(const T1& in_m, const uword in_aux_uword);
  inline         ~CubeToMatOp();

  const T1&   m;            //!< the operand; must be derived from BaseCube
        uword aux_uword;    //!< auxiliary data, uword format

  template<typename T2>
  constexpr bool is_alias(const T2&) const { return false; }

  static constexpr bool is_row  = op_type::template traits<T1>::is_row;
  static constexpr bool is_col  = op_type::template traits<T1>::is_col;
  static constexpr bool is_xvec = op_type::template traits<T1>::is_xvec;
  };
