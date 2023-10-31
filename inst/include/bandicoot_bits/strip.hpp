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



template<typename T1>
struct strip_diagmat
  {
  typedef T1 stored_type;

  inline
  strip_diagmat(const T1& X)
    : M(X)
    {
    coot_extra_debug_sigprint();
    }

  static constexpr bool do_diagmat = false;

  const T1& M;
  };



template<typename T1>
struct strip_diagmat< Op<T1, op_diagmat> >
  {
  typedef T1 stored_type;

  inline
  strip_diagmat(const Op<T1, op_diagmat>& X)
    : M(X.m)
    {
    coot_extra_debug_sigprint();
    }

  static constexpr bool do_diagmat = true;

  const T1& M;
  };



// Turn an op_htrans2 (scalar * X.t()) into just (scalar * X)
template<typename T1>
struct strip_diagmat< Op<Op<T1, op_htrans2>, op_diagmat> >
  {
  typedef eOp<T1, eop_scalar_times> stored_type;

  inline
  strip_diagmat(const Op<Op<T1, op_htrans2>, op_diagmat>& X)
    : M(X.m.m, X.m.aux)
    {
    coot_extra_debug_sigprint();
    }

  static constexpr bool do_diagmat = true;

  const eOp<T1, eop_scalar_times> M;
  };



// push eOps through the diagmat
template<typename T1, typename eop_type>
struct strip_diagmat< eOp<Op<T1, op_diagmat>, eop_type> >
  {
  typedef eOp<T1, eop_type> stored_type;

  inline
  strip_diagmat(const eOp<Op<T1, op_diagmat>, eop_type>& X)
    : M(X.m.Q.m, X.aux, X.aux_uword_a, X.aux_uword_b)
    {
    coot_extra_debug_sigprint();
    }

  static constexpr bool do_diagmat = true;

  const eOp<T1, eop_type> M;
  };



// transposes are still diagonal
// NOTE: this could have problems with complex elements
template<typename T1>
struct strip_diagmat< Op<Op<T1, op_diagmat>, op_htrans> >
  {
  typedef T1 stored_type;

  inline
  strip_diagmat(const Op<Op<T1, op_diagmat>, op_htrans>& X)
    : M(X.m.m)
    {
    coot_extra_debug_sigprint();
    }

  static constexpr bool do_diagmat = true;

  const T1& M;
  };



// transposes are still diagonal
template<typename T1>
struct strip_diagmat< Op<Op<T1, op_diagmat>, op_strans> >
  {
  typedef T1 stored_type;

  inline
  strip_diagmat(const Op<Op<T1, op_diagmat>, op_strans>& X)
    : M(X.m.m)
    {
    coot_extra_debug_sigprint();
    }

  static constexpr bool do_diagmat = true;

  const T1& M;
  };



// rewrite scalar * X.t() as scalar * X
template<typename T1>
struct strip_diagmat< Op<Op<T1, op_diagmat>, op_htrans2> >
  {
  typedef eOp<T1, eop_scalar_times> stored_type;

  inline
  strip_diagmat(const Op<Op<T1, op_diagmat>, op_htrans2>& X)
    : M(X.m.m, X.aux)
    {
    coot_extra_debug_sigprint();
    }

  static constexpr bool do_diagmat = true;

  const eOp<T1, eop_scalar_times> M;
  };
