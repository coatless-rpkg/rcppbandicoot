// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2017-2023 Ryan Curtin (https://www.ratml.org)
// Copyright 2017      Conrad Sanderson (https://conradsanderson.id.au)
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



struct coot_config
  {
  #if defined(COOT_NO_DEBUG)
    static constexpr bool debug = false;
  #else
    static constexpr bool debug = true;
  #endif


  #if defined(COOT_EXTRA_DEBUG)
    static constexpr bool extra_debug = true;
  #else
    static constexpr bool extra_debug = false;
  #endif


  #if defined(COOT_GOOD_COMPILER)
    static constexpr bool good_comp = true;
  #else
    static constexpr bool good_comp = false;
  #endif


  #if (  \
         defined(COOT_EXTRA_MAT_BONES)   || defined(COOT_EXTRA_MAT_MEAT)   \
      || defined(COOT_EXTRA_COL_BONES)   || defined(COOT_EXTRA_COL_MEAT)   \
      || defined(COOT_EXTRA_ROW_BONES)   || defined(COOT_EXTRA_ROW_MEAT)   \
      )
    static constexpr bool extra_code = true;
  #else
    static constexpr bool extra_code = false;
  #endif


  // TODO: may need to link with -lbandicoot anyway, to provide the runtime library
  #if defined(COOT_USE_WRAPPER)
    static constexpr bool wrapper = true;
  #else
    static constexpr bool wrapper = false;
  #endif
  };
