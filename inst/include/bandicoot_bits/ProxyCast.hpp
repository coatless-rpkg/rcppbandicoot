// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2026 Ryan Curtin (https://www.ratml.org)
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



//
// Compile-time shims to reinterpret a T1 as a one-dimensional vector,
// two-dimensional matrix, or three-dimensional cube.
//
// Note this will not actually materialize anything beyong what Proxy<T1>
// will!  Instead, these shims are used at compile-time to generate more
// complex accessor and bounds check macros that implicitly reinterpret
// the T1.
//
// For example, suppose we want to access a submatrix as a vector; the
// access pattern will be similar to this:
//
//    at(i) = mem[(i % n_rows) + (i / n_rows) * M_n_rows]
//
// where n_rows and M_n_rows are the number of rows in the submatrix and
// its parent matrix, respectively.
//
// Access patterns for other casts on other types may be more complex;
// for example, 2D-to-3D reinterpretation first requires linearizing the
// 2D object as a 1D object, and then reinterpreting that as a 3D object.
//
// For the guts of how those are assembled, see the corresponding overloads
// for each macro type in kernel_gen/.
//
// The `proxy_uses_ref` template parameter controls whether a Proxy< Proxy*Cast<...> >
// will hold a Proxy<T1> as a reference or an object---it is not directly used
// in the class definitions here.
//



template<typename T1, size_t src_dims = Proxy<T1>::num_dims, bool proxy_uses_ref = false>
struct ProxyColCast
  {
  const T1& Q;

  typedef typename T1::elem_type elem_type;

  inline ProxyColCast(const T1& in_Q) : Q(in_Q) { }
  };



template<typename T1, size_t src_dims = Proxy<T1>::num_dims, bool proxy_uses_ref = false>
struct ProxyMatCast
  {
  const T1& Q;
  const uword n_rows;
  const uword n_cols;

  typedef typename T1::elem_type elem_type;

  inline ProxyMatCast(const T1& in_Q, const uword in_n_rows, const uword in_n_cols) : Q(in_Q), n_rows(in_n_rows), n_cols(in_n_cols) { }
  };



template<typename T1, size_t src_dims = Proxy<T1>::num_dims, bool proxy_uses_ref = false>
struct ProxyCubeCast
  {
  const T1& Q;
  const uword n_rows;
  const uword n_cols;
  const uword n_slices;

  typedef typename T1::elem_type elem_type;

  inline ProxyCubeCast(const T1& in_Q, const uword in_n_rows, const uword in_n_cols, const uword in_n_slices) : Q(in_Q), n_rows(in_n_rows), n_cols(in_n_cols), n_slices(in_n_slices) { }
  };
