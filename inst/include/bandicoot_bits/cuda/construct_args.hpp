// Copyright 2025 Ryan Curtin (http://www.ratml.org)
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



/**
 * Given a set of arguments to a kernel, some of which are matrix-like objects
 * and some of which are not, assemble them into a const void*[] so they can be
 * passed into a CUDA kernel.
 */



template<typename T>
inline
void*
to_arg(const T& t)
  {
  return (void*) &t;
  }



template<typename T, size_t offset>
inline
void*
to_proxy_arg(const Proxy<T>& t)
  {
  return (void*) &std::get<offset>(t.args());
  }



template<size_t offset, size_t len>
inline
void
fill_args(std::array<void*, len>& result)
  {
  coot_ignore(result);
  }



template<size_t offset, size_t len, typename T, typename... Ts>
inline
typename
enable_if2
  <
  is_Proxy<T>::value == false,
  void
  >::result
fill_args(std::array<void*, len>& result, const T& t, const Ts&... ts)
  {
  result[offset] = to_arg<T>(t);
  fill_args<offset + 1>(result, ts...);
  }



template<size_t offset, size_t proxy_offset, size_t len, typename T1>
inline
typename
enable_if2
  <
  (proxy_offset >= Proxy<T1>::num_args),
  void
  >::result
fill_proxy_arg(std::array<void*, len>& result, const Proxy<T1>& t)
  {
  coot_ignore(result);
  coot_ignore(t);

  return; // nothing to do
  }



template<size_t offset, size_t proxy_offset, size_t len, typename T1>
inline
typename
enable_if2
  <
  (proxy_offset < Proxy<T1>::num_args),
  void
  >::result
fill_proxy_arg(std::array<void*, len>& result, const Proxy<T1>& t)
  {
  result[offset] = to_proxy_arg<T1, proxy_offset>(t);
  fill_proxy_arg<offset + 1, proxy_offset + 1, len, T1>(result, t);
  }



template<size_t offset, size_t len, typename T1, typename... Ts>
inline
void
fill_args(std::array<void*, len>& result, const Proxy<T1>& t, const Ts&... ts)
  {
  fill_proxy_arg<offset, 0>(result, t);
  fill_args<offset + Proxy<T1>::num_args>(result, ts...);
  }



template<typename... Ts>
struct num_args { };

template<typename T, typename... Ts>
struct num_args< T, Ts... >
  {
  static constexpr const size_t result = 1 + num_args<Ts...>::result;
  };

template<typename T, typename... Ts>
struct num_args< Proxy<T>, Ts... >
  {
  static constexpr const size_t result = Proxy<T>::num_args + num_args<Ts...>::result;
  };

template<>
struct num_args<>
  {
  static constexpr const size_t result = 0;
  };


template<typename T, typename... Ts>
inline
typename
std::array<void*, num_args<T, Ts...>::result>
construct_args(const T& t, const Ts&... ts)
  {
  typename std::array<void*, num_args<T, Ts...>::result> result;
  fill_args<0>(result, t, ts...);
  return result;
  }
