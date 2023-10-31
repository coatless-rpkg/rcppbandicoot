// Copyright 2020 Ryan Curtin (http://www.ratml.org)
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



template<typename T>
inline
std::string
perform_type_mapping()
  {
  // probably will cause a later compilation failure...
  return "unknown";
  }



template<>
inline
std::string
perform_type_mapping<u32>()
  {
  return "uint";
  }



template<>
inline
std::string
perform_type_mapping<s32>()
  {
  return "int";
  }



template<>
inline
std::string
perform_type_mapping<u64>()
  {
  return "ulong";
  }



template<>
inline
std::string
perform_type_mapping<s64>()
  {
  return "long";
  }



template<>
inline
std::string
perform_type_mapping<float>()
  {
  return "float";
  }



template<>
inline
std::string
perform_type_mapping<double>()
  {
  return "double";
  }



struct type_to_dev_string
  {
  template<typename T>
  static
  inline
  std::string map()
    {
    return perform_type_mapping<T>();
    }



  // Return the name of the abs() function to use for a floating point type.
  template<typename T>
  static
  inline
  std::string
  abs_func(const typename enable_if2<std::is_floating_point<T>::value, void>::result* = 0)
    {
    return "fabs";
    }



  // Return the name of the abs() function to use for a signed integral type.
  template<typename T>
  static
  inline
  std::string
  abs_func(const typename enable_if2<std::is_signed<T>::value && !std::is_floating_point<T>::value, void>::result* = 0)
    {
    return "abs";
    }



  // Return the name of the abs() function to use for an unsigned type.  (Here, we use nothing...)
  template<typename T>
  static
  inline
  std::string
  abs_func(const typename enable_if2<!std::is_signed<T>::value, void>::result* = 0)
    {
    return "";
    }
  };
