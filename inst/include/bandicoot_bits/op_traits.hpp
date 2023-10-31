// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2023      Ryan Curtin (https://www.ratml.org)
// Copyright 2008-2016 Conrad Sanderson (https://conradsanderson.id.au)
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



struct traits_op_default
  {
  template<typename T1>
  struct traits
    {
    static constexpr bool is_row  = false;
    static constexpr bool is_col  = false;
    static constexpr bool is_xvec = false;
    };
  };


struct traits_op_xvec
  {
  template<typename T1>
  struct traits
    {
    static constexpr bool is_row  = false;
    static constexpr bool is_col  = false;
    static constexpr bool is_xvec = true;
    };
  };


struct traits_op_col
  {
  template<typename T1>
  struct traits
    {
    static constexpr bool is_row  = false;
    static constexpr bool is_col  = true;
    static constexpr bool is_xvec = false;
    };
  };


struct traits_op_row
  {
  template<typename T1>
  struct traits
    {
    static constexpr bool is_row  = true;
    static constexpr bool is_col  = false;
    static constexpr bool is_xvec = false;
    };
  };


struct traits_op_passthru
  {
  template<typename T1>
  struct traits
    {
    static constexpr bool is_row  = T1::is_row;
    static constexpr bool is_col  = T1::is_col;
    static constexpr bool is_xvec = T1::is_xvec;
    };
  };



struct traits_op_passthru_trans
  {
  template<typename T1>
  struct traits
    {
    static constexpr bool is_row  = T1::is_col; // deliberately swapped
    static constexpr bool is_col  = T1::is_row;
    static constexpr bool is_xvec = T1::is_xvec;
    };
  };
