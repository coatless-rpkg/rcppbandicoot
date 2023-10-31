// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2023 Ryan Curtin (http://www.ratml.org)
// Copyright 2023 Conrad Sanderson (https://conradsanderson.id.au)
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



template<typename T1>
coot_warn_unused
inline
typename
enable_if2<
  is_real<typename T1::elem_type>::value,
  const Op<T1, op_pinv>
>::result
pinv
  (
  const Base<typename T1::elem_type, T1>& X
  )
  {
  coot_extra_debug_sigprint();

  return Op<T1, op_pinv>(X.get_ref(), typename T1::elem_type(0));
  }


template<typename T1>
coot_warn_unused
inline
typename
enable_if2<
  is_real<typename T1::elem_type>::value,
  const Op<T1, op_pinv>
>::result
pinv
  (
  const Base<typename T1::elem_type, T1>& X,
  const typename T1::elem_type            tol,
  const char*                             method = nullptr
  )
  {
  coot_extra_debug_sigprint();

  typedef typename T1::elem_type eT;

  // We only support the standard method for now, but provide the option for compatibility with Armadillo.
  if (method != nullptr)
    {
    const char sig = method[0];

    coot_debug_check( (sig == 'd'), "pinv(): \"dc\" (divide and conquer) method not supported by Bandicoot" );

    coot_debug_check( ((sig != 's') && (sig != 'd')), "pinv(): unknown method specified" );
    }

  return Op<T1, op_pinv>(X.get_ref(), eT(tol));
  }



template<typename T1>
inline
typename
enable_if2<
  is_real<typename T1::elem_type>::value,
  bool
>::result
pinv
  (
         Mat<typename T1::elem_type>&     out,
  const Base<typename T1::elem_type, T1>& X,
  const typename T1::elem_type            tol = 0.0,
  const char*                             method = nullptr
  )
  {
  coot_extra_debug_sigprint();

  if (method != nullptr)
    {
    const char sig = method[0];

    coot_debug_check( (sig == 'd'), "pinv(): \"dc\" (divide and conquer) method not supported by Bandicoot" );

    coot_debug_check( ((sig != 's') && (sig != 'd')), "pinv(): unknown method specified" );
    }

  const std::tuple<bool, std::string> result = op_pinv::apply_direct(out, X.get_ref(), tol);

  if (std::get<0>(result) == false)
    {
    out.reset();
    coot_debug_warn_level(3, "pinv(): " + std::get<1>(result));
    }

  return std::get<0>(result);
  }
