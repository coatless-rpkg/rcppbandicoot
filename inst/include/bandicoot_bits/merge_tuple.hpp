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
struct merge_tuple { };

template<typename... T1>
struct merge_tuple<std::tuple<T1...>>
  {
  typedef std::tuple<T1...> result;
  };

template<typename... T1, typename... T2>
struct merge_tuple<std::tuple<T1...>, std::tuple<T2...>>
  {
  typedef typename std::tuple<T1..., T2...> result;
  };

template<typename... T1, typename... T2, typename... Ts>
struct merge_tuple<std::tuple<T1...>, std::tuple<T2...>, Ts...>
  {
  typedef typename merge_tuple< std::tuple<T1..., T2...>, Ts... >::result result;
  };
