// Copyright 2024 Ryan Curtin (http://www.ratml.org/)
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


// Utility function to get the next power of 2.
// Make sure eT is unsigned, and sizeof(eT) is a power of 2!

template<typename T>
inline
T
next_pow2(T in, const typename std::enable_if<sizeof(T) == 1>::type* = 0)
  {
  --in;
  in |= (in >> 1);
  in |= (in >> 2);
  in |= (in >> 4);
  ++in;

  return in;
  }



template<typename T>
inline
T
next_pow2(T in, const typename std::enable_if<sizeof(T) == 2>::type* = 0)
  {
  --in;
  in |= (in >> 1);
  in |= (in >> 2);
  in |= (in >> 4);
  in |= (in >> 8);
  ++in;

  return in;
  }



template<typename T>
inline
T
next_pow2(T in, const typename std::enable_if<sizeof(T) == 4>::type* = 0)
  {
  --in;
  in |= (in >> 1);
  in |= (in >> 2);
  in |= (in >> 4);
  in |= (in >> 8);
  in |= (in >> 16);
  ++in;

  return in;
  }


template<typename T>
inline
T
next_pow2(T in, const typename std::enable_if<sizeof(T) == 8>::type* = 0)
  {
  --in;
  in |= (in >> 1);
  in |= (in >> 2);
  in |= (in >> 4);
  in |= (in >> 8);
  in |= (in >> 16);
  in |= (in >> 32);
  ++in;

  return in;
  }
