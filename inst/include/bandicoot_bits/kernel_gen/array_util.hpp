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


//
// Utilities for dealing with compile-time constexpr strings.
//

template<size_t N>
struct char_array
  {
  char data[N];

  constexpr const char* begin() const noexcept { return (const char*) data;       }
  constexpr const char* end()   const noexcept { return (const char*) data + N;   }

  constexpr char* begin() noexcept { return (char*) data;     }
  constexpr char* end()   noexcept { return (char*) data + N; }

  constexpr       char& operator[](size_t i)       noexcept { return data[i]; }
  constexpr const char& operator[](size_t i) const noexcept { return data[i]; }

  template<size_t N2>
  constexpr bool operator==(const char_array<N2>&) const
    {
    return false;
    }

  constexpr bool operator==(const char_array<N>& other) const
    {
    size_t i = 0;
    while (i < N)
      {
      if (data[i] != other.data[i])
        return false;
      ++i;
      }

    return true;
    }

  template<size_t N2>
  constexpr bool operator!=(const char_array<N2>& other) const
    {
    return !(*this == other);
    }
  };



//
// Input types (P) are expected to be structs containing:
//  * ::len() (returns a constexpr size_t)
//  * ::str() (returns a char_array<len + 1>) -- includes a null terminator
//

template<size_t N>   constexpr inline size_t arg_len_impl(const char(&)[N])     { return N - 1; /* skip null terminator */ }
template<size_t N>   constexpr inline size_t arg_len_impl(const char_array<N>&) { return N - 1; /* skip null terminator */ }

template<typename T, typename = void>
struct has_len_member
  {
  static const bool value = false;
  };

template<typename T>
struct has_len_member<T, decltype(T::len, void())>
  {
  static const bool value = true;
  };

template<bool has_len, typename T>
struct arg_len_helper
  {
  static inline constexpr size_t len() { return arg_len_impl(T::str()); }
  };

template<typename T>
struct arg_len_helper<true, T>
  {
  static inline constexpr size_t len() { return T::len(); }
  };

template<typename T>
struct arg_len : public arg_len_helper< has_len_member<T>::value, T > { };



template<typename... Ts>
struct concat_str
  {
  static inline constexpr size_t len() { return 0; }
  static inline constexpr char_array<1> str() { return char_array<1>{ '\0' }; }
  };



template<typename T1>
struct concat_str<T1>
  {
  static inline constexpr size_t len() { return arg_len<T1>::len(); }
  static inline constexpr char_array<len() + 1> str()
    {
    char_array<len() + 1> r {};
    constexpr const auto s1 = T1::str();
    constexpr const size_t l1 = arg_len<T1>::len();

    size_t p = 0;
    for (size_t i = 0; i < l1; ++i, ++p)
      r[p] = s1[i];
    r[len()] = '\0';

    return r;
    }
  };



template<typename T1, typename T2>
struct concat_str<T1, T2>
  {
  static inline constexpr size_t len() { return arg_len<T1>::len() + arg_len<T2>::len(); }
  static inline constexpr char_array<len() + 1> str()
    {
    char_array<len() + 1> r {};
    constexpr const size_t l1 = arg_len<T1>::len();
    constexpr const size_t l2 = arg_len<T2>::len();
    constexpr const auto s1 = T1::str();
    constexpr const auto s2 = T2::str();

    size_t p = 0;
    for (size_t i = 0; i < l1; ++i, ++p)
      r[p] = s1[i];
    for (size_t i = 0; i < l2; ++i, ++p)
      r[p] = s2[i];
    r[len()] = '\0';

    return r;
    }
  };



template<typename T1, typename T2, typename... Ts>
struct concat_str<T1, T2, Ts...> : public concat_str< concat_str<T1, T2>, Ts... > { };



// these will be assembled at runtime, but their components will be assembled at compile-time if possible
// *all* components must be concat_strs or nested_concat_strs!
template<typename... Ts>
struct nested_concat_str
  {
  static inline constexpr size_t len() { return 0; }
  static inline constexpr char_array<1> str() { return char_array<1>{ '\0' }; }
  };



// really, hopefully, this should never be called!
template<typename T1>
struct nested_concat_str<T1>
  {
  static inline constexpr size_t len() { return T1::len(); }
  static inline char_array<len() + 1> str() { return T1::str(); }
  };



template<typename T1, typename T2>
struct nested_concat_str<T1, T2>
  {
  static inline constexpr size_t len() { return arg_len<T1>::len() + arg_len<T2>::len(); }
  static inline char_array<len() + 1> str()
    {
    char_array<len() + 1> r {};
    const auto s1 = T1::str();
    const auto s2 = T2::str();
    constexpr const size_t l1 = arg_len<T1>::len();
    constexpr const size_t l2 = arg_len<T2>::len();

    size_t p = 0;
    for (size_t i = 0; i < l1; ++i, ++p)
      r[p] = s1[i];
    for (size_t i = 0; i < l2; ++i, ++p)
      r[p] = s2[i];
    r[len()] = '\0';

    return r;
    }
  };



template<typename T1, typename T2, typename... Ts>
struct nested_concat_str<T1, T2, Ts...> : public nested_concat_str< nested_concat_str<T1, T2>, Ts... > { };



//
// helper utility to convert a const char[] into a char_array
//

struct to_array
  {
  template<size_t N>
  static inline constexpr char_array<N> array(const char (&s)[N])
    {
    char_array<N> r {};
    size_t i = 0;
    while (i < N)
      {
      r[i] = s[i];
      ++i;
      }

    return r;
    }
  };
