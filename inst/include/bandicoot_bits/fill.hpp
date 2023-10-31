// SPDX-License-Identifier: Apache-2.0
// 
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


namespace fill
  {
  struct fill_none  {};
  struct fill_zeros {};
  struct fill_ones  {};
  struct fill_eye   {};
  struct fill_randu {};
  struct fill_randn {};
  
  template<typename fill_type> 
  struct fill_class { inline constexpr fill_class() {} };
  
  static constexpr fill_class<fill_none > none;
  static constexpr fill_class<fill_zeros> zeros;
  static constexpr fill_class<fill_ones > ones;
  static constexpr fill_class<fill_eye  > eye;
  static constexpr fill_class<fill_randu> randu;
  static constexpr fill_class<fill_randn> randn;
  
  // TODO: adapt fill::value from Armadillo
  }
