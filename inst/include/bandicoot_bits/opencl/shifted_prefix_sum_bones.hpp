// Copyright 2024 Ryan Curtin (http://www.ratml.org)
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



// NOTE: this is not exactly prefix-sum as typically taught, but it's what's needed for radix search.  It returns a result that is shifted by one element.
// An input of [1, 3, 2, 4] returns an output of [0, 1, 4, 6]---*not* the "typical" output of [1, 4, 6, 10].



template<typename eT>
inline
void
shifted_prefix_sum_small(dev_mem_t<eT> mem, const uword n_elem, const size_t total_num_threads, const size_t local_group_size);



template<typename eT>
inline
void
shifted_prefix_sum_large(dev_mem_t<eT> mem, const uword n_elem, const size_t total_num_threads, const size_t local_group_size);



template<typename eT>
inline
void
shifted_prefix_sum(dev_mem_t<eT> mem, const uword n_elem);
