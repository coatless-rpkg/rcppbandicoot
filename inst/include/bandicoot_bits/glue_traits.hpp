// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2023 Ryan Curtin (https://www.ratml.org)
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



struct traits_glue_default
  {
  template<typename T1, typename T2>
  struct traits
    {
    static constexpr bool is_row  = false;
    static constexpr bool is_col  = false;
    static constexpr bool is_xvec = false;
    };
  };



struct traits_glue_or
  {
  template<typename T1, typename T2>
  struct traits
    {
    static constexpr bool is_row  = (T1::is_row  || T2::is_row);
    static constexpr bool is_col  = (T1::is_col  || T2::is_col);
    static constexpr bool is_xvec = (T1::is_xvec || T2::is_xvec);
    };
  };
