// Copyright 2021 Marcus Edel (http://kurg.org)
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



template<typename eT>
inline
void
shuffle(dev_mem_t<eT> out, const uword out_row_offset, const uword out_col_offset, const uword out_M_n_rows,
        const dev_mem_t<eT> in, const uword in_row_offset, const uword in_col_offset, const uword in_M_n_rows,
        const uword n_rows, const uword n_cols, const uword dim);



template<typename eT>
inline
void
shuffle_small(      dev_mem_t<eT> out, const uword out_offset, const uword out_incr, const uword out_elem_stride,
              const dev_mem_t<eT> in,  const uword in_offset,  const uword in_incr,  const uword in_elem_stride,
              const uword n_elem, const uword elems_per_elem,
              const uword n_elem_pow2, const uword num_bits, const dev_mem_t<uword> philox_keys);



template<typename eT>
inline
void
shuffle_large(      dev_mem_t<eT> out, const uword out_offset, const uword out_incr, const uword out_elem_stride,
              const dev_mem_t<eT> in,  const uword in_offset,  const uword in_incr,  const uword in_elem_stride,
              const uword n_elem, const uword elems_per_elem,
              const uword n_elem_pow2, const uword num_bits, const size_t local_group_size, const dev_mem_t<uword> philox_keys);
