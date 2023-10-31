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


#undef COOT_HAVE_CXX11
#undef COOT_HAVE_CXX14
#undef COOT_HAVE_CXX17
#undef COOT_HAVE_CXX20

#if (__cplusplus >= 201103L)
  #define COOT_HAVE_CXX11
#endif

#if (__cplusplus >= 201402L)
  #define COOT_HAVE_CXX14
#endif

#if (__cplusplus >= 201703L)
  #define COOT_HAVE_CXX17
#endif

#if (__cplusplus >= 202002L)
  #define COOT_HAVE_CXX20
#endif


// MS really can't get its proverbial shit together
#if defined(_MSVC_LANG)
  
  #if (_MSVC_LANG >= 201402L)
    #undef  COOT_HAVE_CXX11
    #define COOT_HAVE_CXX11
    
    #undef  COOT_HAVE_CXX14
    #define COOT_HAVE_CXX14
  #endif
  
  #if (_MSVC_LANG >= 201703L)
    #undef  COOT_HAVE_CXX17
    #define COOT_HAVE_CXX17
  #endif
  
  #if (_MSVC_LANG >= 202002L)
    #undef  COOT_HAVE_CXX20
    #define COOT_HAVE_CXX20
  #endif
  
#endif


#if !defined(COOT_HAVE_CXX11)
  #error "*** C++11 compiler required; enable C++11 mode in your compiler"
#endif


#undef COOT_HAVE_ARMA

#if defined(ARMA_VERSION_MAJOR)
  
  #define COOT_HAVE_ARMA
  
  #if (ARMA_VERSION_MAJOR < 9)
    #pragma message ("WARNING: detected unsupported version of Armadillo")
  #endif
  
#endif
