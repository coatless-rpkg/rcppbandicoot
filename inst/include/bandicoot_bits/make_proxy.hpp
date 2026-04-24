// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2025 Ryan Curtin (https://www.ratml.org)
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
// Convenience functions to create a Proxy without needing to type the typenames:
//
//  * make_proxy(t):      creates a Proxy<decltype(t)>
//  * make_proxy_col(t):  creates a Proxy and uses a ProxyColCast to make it appear 1-dimensional (if needed)
//  * make_proxy_mat(t):  creates a Proxy and uses a ProxyMatCast to make it appear 2-dimensional (if needed)
//  * make_proxy_cube(t): creates a Proxy and uses a ProxyCubeCast to make it appear 3-dimensional (if needed)
//

template<typename T1>
inline
Proxy<T1>
make_proxy(const T1& t)
  {
  return Proxy<T1>(t);
  }



// When we use a ProxyColCast, the returned Proxy will be a Proxy<Col<eT>> for Mats and Cubes too.
template<typename T1> struct proxy_col_inner_type           { typedef T1 type;      };
template<typename eT> struct proxy_col_inner_type<Mat<eT>>  { typedef Col<eT> type; };
template<typename eT> struct proxy_col_inner_type<Cube<eT>> { typedef Col<eT> type; };



template<typename T1, bool X> struct proxy_col_type_helper           { typedef ProxyColCast<T1> type;                        };
template<typename T1>         struct proxy_col_type_helper<T1, true> { typedef typename proxy_col_inner_type<T1>::type type; };

template<typename T1> struct proxy_col_type
  {
  typedef Proxy<typename proxy_col_type_helper<T1, Proxy<T1>::num_dims == 1 || is_Mat<T1>::value == true || is_Cube<T1>::value == true>::type> type;
  };



// Mat and Cube are contiguous in memory so we can just treat them as columns
template<typename T1>
inline
typename
enable_if2
  <
  (Proxy<T1>::num_dims == 1 || is_Mat<T1>::value == true || is_Cube<T1>::value == true),
  Proxy<typename proxy_col_inner_type<T1>::type>
  >::result
make_proxy_col(const T1& t)
  {
  return Proxy<typename proxy_col_inner_type<T1>::type>(t);
  }



template<typename T1>
inline
typename
enable_if2
  <
  (Proxy<T1>::num_dims != 1 && is_Mat<T1>::value == false && is_Cube<T1>::value == false),
  Proxy<ProxyColCast<T1>>
  >::result
make_proxy_col(const T1& t)
  {
  return Proxy<ProxyColCast<T1>>(t);
  }



template<typename T1>
inline
typename
enable_if2
  <
  Proxy<T1>::num_dims == 2,
  Proxy<T1>
  >::result
make_proxy_mat(const T1& t, const uword new_n_rows = 0, const uword new_n_cols = 0)
  {
  Proxy<T1> r(t);
  if (new_n_rows != 0 || new_n_cols != 0)
    {
    coot_check_runtime_error( new_n_rows != r.get_n_rows() || new_n_cols != r.get_n_cols(), "make_proxy_mat(): size cannot be changed when a 2D object is given as input" );
    }
  return r;
  }



template<typename T1>
inline
typename
enable_if2
  <
  Proxy<T1>::num_dims != 2,
  Proxy<ProxyMatCast<T1>>
  >::result
make_proxy_mat(const T1& t, const uword new_n_rows, const uword new_n_cols)
  {
  return Proxy<ProxyMatCast<T1>>(t, new_n_rows, new_n_cols);
  }



template<typename T1>
inline
typename
enable_if2
  <
  Proxy<T1>::num_dims == 2,
  const Proxy<T1>&
  >::result
make_proxy_mat(const Proxy<T1>& t)
  {
  return t;
  }



template<typename T1>
inline
typename
enable_if2
  <
  Proxy<T1>::num_dims != 2,
  Proxy<ProxyMatCast<T1>>
  >::result
make_proxy_mat(const Proxy<T1>& t)
  {
  return Proxy<ProxyMatCast<T1>>(t, t.get_n_rows(), t.get_n_cols());
  }



template<typename T1>
inline
typename
enable_if2
  <
  Proxy<T1>::num_dims == 3,
  Proxy<T1>
  >::result
make_proxy_cube(const T1& t)
  {
  return Proxy<T1>(t);
  }



template<typename T1>
inline
typename
enable_if2
  <
  Proxy<T1>::num_dims != 3,
  Proxy<ProxyCubeCast<T1>>
  >::result
make_proxy_cube(const T1& t, const uword new_n_rows, const uword new_n_cols, const uword new_n_slices)
  {
  return Proxy<ProxyCubeCast<T1>>(t, new_n_rows, new_n_cols, new_n_slices);
  }



template<typename T1, typename T2>
inline
typename
enable_if2
  <
  Proxy<T2>::num_dims == 1 && Proxy<T1>::num_dims != 1,
  const Proxy<ProxyColCast<T1, Proxy<T1>::num_dims, true>>
  >::result
make_proxy_same_dim(const Proxy<T1>& in, const Proxy<T2>& target)
  {
  coot_ignore(target);
  return Proxy<ProxyColCast<T1, Proxy<T1>::num_dims, true>>(in);
  }



template<typename T1, typename T2>
inline
typename
enable_if2
  <
  Proxy<T2>::num_dims == 1 && Proxy<T1>::num_dims == 1,
  const Proxy<T1>&
  >::result
make_proxy_same_dim(const Proxy<T1>& in, const Proxy<T2>& target)
  {
  coot_ignore(target);
  return in;
  }



template<typename T1, typename T2>
inline
typename
enable_if2
  <
  Proxy<T2>::num_dims == 2 && Proxy<T1>::num_dims != 2,
  const Proxy<ProxyMatCast<T1, Proxy<T1>::num_dims, true>>
  >::result
make_proxy_same_dim(const Proxy<T1>& in, const Proxy<T2>& target)
  {
  return Proxy<ProxyMatCast<T1, Proxy<T1>::num_dims, true>>(in, target.get_n_rows(), target.get_n_cols());
  }



template<typename T1, typename T2>
inline
typename
enable_if2
  <
  Proxy<T2>::num_dims == 2 && Proxy<T1>::num_dims == 2,
  const Proxy<T1>&
  >::result
make_proxy_same_dim(const Proxy<T1>& in, const Proxy<T2>& target)
  {
  coot_ignore(target);
  return in;
  }



template<typename T1, typename T2>
inline
typename
enable_if2
  <
  Proxy<T2>::num_dims == 3 && Proxy<T1>::num_dims != 3,
  const Proxy<ProxyCubeCast<T1, Proxy<T1>::num_dims, true>>
  >::result
make_proxy_same_dim(const Proxy<T1>& in, const Proxy<T2>& target)
  {
  return Proxy<ProxyCubeCast<T1, Proxy<T1>::num_dims, true>>(in, target.get_n_rows(), target.get_n_cols(), target.get_n_slices());
  }



template<typename T1, typename T2>
inline
typename
enable_if2
  <
  Proxy<T2>::num_dims == 3 && Proxy<T1>::num_dims == 3,
  const Proxy<T1>&
  >::result
make_proxy_same_dim(const Proxy<T1>& in, const Proxy<T2>& target)
  {
  coot_ignore(target);
  return in;
  }



