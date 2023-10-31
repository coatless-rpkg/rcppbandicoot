// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2019 Ryan Curtin <ryan@ratml.org>
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


struct coot_rng
  {
  template<typename eT>
  static inline void fill_randu(dev_mem_t<eT> dest, const uword n);

  template<typename eT>
  static inline void fill_randn(dev_mem_t<eT> dest, const uword n, const distr_param& param = distr_param());

  template<typename eT>
  static inline void fill_randi(dev_mem_t<eT> dest, const uword n, const distr_param& param = distr_param());

  // seed handling

  static inline void set_seed(const u64 seed);

  static inline void set_seed_random();
  };
