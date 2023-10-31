// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2023 Ryan Curtin (http://www.ratml.org)
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


// Definitions of Armadillo public classes that may be used by Bandicoot.
// Note that Armadillo is not a requirement for Bandicoot, but Bandicoot provides some convenience interoperability functions.
// In order to provide those, we at least need forward definitions of the relevant classes.

namespace arma
  {

  template<typename base, typename derived>
  class Base;

  template<typename eT>
  class Mat;

  template<typename eT>
  class Col;

  template<typename eT>
  class Row;

  template<typename T1>
  struct conv_to;

  }
