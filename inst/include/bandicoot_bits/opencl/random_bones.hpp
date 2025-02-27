// Copyright 2019 Ryan Curtin (http://www.ratml.org/)
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

// Utility functions for generating random numbers via OpenCL.

template<typename eT>
inline void init_xorwow_state(coot_cl_mem state, const size_t num_rng_threads, const u64 seed = 0);

inline void init_philox_state(coot_cl_mem state, const size_t num_rng_threads, const u64 seed = 0);

template<typename eT>
inline void fill_randu(dev_mem_t<eT> dest, const uword n);

template<typename eT>
inline void fill_randn(dev_mem_t<eT> dest, const uword n, const double mu, const double sd);

template<typename eT>
inline void fill_randi(dev_mem_t<eT> dest, const uword n, const int lo, const int hi);
