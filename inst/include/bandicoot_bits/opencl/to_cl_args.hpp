// Copyright 2025-2026 Ryan Curtin (http://www.ratml.org/)
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



template<typename... Ts>
struct cl_args;



template<typename T>
struct to_cl_types { typedef std::tuple<typename cl_type<T>::type> result; };

template<typename T>
struct to_cl_types<const T&> { typedef std::tuple<const typename cl_type<T>::type&> result; };

template<>
struct to_cl_types< uword > { typedef std::tuple< runtime_t::adapt_uword > result; };

template<>
struct to_cl_types< const uword& > { typedef std::tuple< runtime_t::adapt_uword > result; };

template<typename eT>
struct to_cl_types< dev_mem_t<eT> > { typedef std::tuple< cl_mem, runtime_t::adapt_uword > result; };

template<typename eT>
struct to_cl_types< dev_mem_t<eT>& > { typedef std::tuple< cl_mem, runtime_t::adapt_uword > result; };

template<typename eT>
struct to_cl_types< const dev_mem_t<eT>& > { typedef std::tuple< cl_mem, runtime_t::adapt_uword > result; };

// utility to unpack a std::tuple of arguments into processing by to_cl_types<>
template<typename T>
struct cl_type_applier { };

template<>
struct cl_type_applier< std::tuple<> > { typedef std::tuple<> result; };

template<typename... Ts>
struct cl_type_applier< std::tuple<Ts...> > { typedef typename merge_tuple< typename to_cl_types<Ts>::result... >::result result; };

template<typename T>
struct to_cl_types< Proxy<T> > { typedef typename cl_type_applier< typename Proxy<T>::arg_types >::result result; };



template<typename... Ts>
struct cl_args
  {
  typedef typename merge_tuple< typename to_cl_types<Ts>::result... >::result result;
  };



template<typename T>
inline
std::tuple< const T& >
to_cl_arg(const T& t)
  {
  return std::tie<const T&>(t);
  }



inline
std::tuple< runtime_t::adapt_uword >
to_cl_arg(const uword& t)
  {
  return std::make_tuple(runtime_t::adapt_uword(t));
  }



template<typename eT>
inline
std::tuple< cl_mem, runtime_t::adapt_uword >
to_cl_arg(dev_mem_t<eT>& t)
  {
  return std::make_tuple( t.cl_mem_ptr.ptr, runtime_t::adapt_uword(t.cl_mem_ptr.offset) );
  }



template<typename eT>
inline
std::tuple< cl_mem, runtime_t::adapt_uword >
to_cl_arg(const dev_mem_t<eT>& t)
  {
  return std::make_tuple( t.cl_mem_ptr.ptr, runtime_t::adapt_uword(t.cl_mem_ptr.offset) );
  }



template<typename T>
inline
typename cl_args<const T&>::result
to_cl_args(const T& t)
  {
  return to_cl_arg(t);
  }



inline
std::tuple<>
to_cl_arg_applier(const std::tuple<>& t)
  {
  return t;
  }



template<typename T, size_t... Is>
inline
typename to_cl_types< Proxy<T> >::result
to_cl_args_inner(const Proxy<T>& t, const std::integer_sequence<size_t, Is...>& junk)
  {
  coot_ignore(junk);
  return std::tuple_cat( to_cl_args(std::get<Is>(t.args()))... );
  }


template<typename T>
inline
typename to_cl_types< Proxy<T> >::result
to_cl_args(const Proxy<T>& t)
  {
  return to_cl_args_inner(t, std::make_index_sequence<Proxy<T>::num_args>{});
  }



template<typename T, typename... Ts>
inline
typename cl_args<T, Ts...>::result
to_cl_args(const T& t, const Ts&... args)
  {
  return std::tuple_cat(
      to_cl_args(t),
      to_cl_args(args...));
  }
