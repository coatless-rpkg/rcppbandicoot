// Copyright 2026 Marcus Edel (http://www.kurg.org)
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
eT
vk_abs_val(const eT x)
  {
  if constexpr (std::is_unsigned<eT>::value)
    return x;
  else
    return (x < eT(0)) ? -x : x;
  }



template<typename eT>
inline
eT
vk_abs_diff(const eT a, const eT b)
  {
  return (a > b) ? (a - b) : (b - a);
  }



template<typename eT>
inline
bool
approx_equal(const dev_mem_t<eT> A,
             const uword A_row_offset,
             const uword A_col_offset,
             const uword A_M_n_rows,
             const dev_mem_t<eT> B,
             const uword B_row_offset,
             const uword B_col_offset,
             const uword B_M_n_rows,
             const uword n_rows,
             const uword n_cols,
             const char sig,
             const eT abs_tol,
             const eT rel_tol)
  {
  coot_debug_sigprint();

  const uword n_elem = n_rows * n_cols;
  if (n_elem == 0) { return true; }

  const bool do_abs = (sig == 'a' || sig == 'b');
  const bool do_rel = (sig == 'r' || sig == 'b');

  runtime_t& rt = get_rt().vk_rt;

  const eT* A_ptr = reinterpret_cast<const eT*>(rt.get_pool_mapped()) + A.vk_mem_ptr.offset;
  const eT* B_ptr = reinterpret_cast<const eT*>(rt.get_pool_mapped()) + B.vk_mem_ptr.offset;

  bool equal = true;

  for (uword col = 0; col < n_cols && equal; ++col)
    {
    for (uword row = 0; row < n_rows && equal; ++row)
      {
      const uword A_idx = (A_row_offset + row) + (A_col_offset + col) * A_M_n_rows;
      const uword B_idx = (B_row_offset + row) + (B_col_offset + col) * B_M_n_rows;

      const eT A_val = A_ptr[A_idx];
      const eT B_val = B_ptr[B_idx];

      if (coot_isnan(A_val) || coot_isnan(B_val))
        {
        equal = false;
        break;
        }

      const eT absdiff = vk_abs_diff(A_val, B_val);

      if (do_abs)
        {
        if (absdiff > abs_tol) { equal = false; break; }
        }

      if (do_rel)
        {
        const eT max_val = std::max(vk_abs_val(A_val), vk_abs_val(B_val));
        if (max_val >= eT(1))
          {
          if (absdiff > rel_tol * max_val) { equal = false; break; }
          }
        else if (max_val > eT(0))
          {
          if (absdiff / max_val > rel_tol) { equal = false; break; }
          }
        }
      }
    }

  return equal;
  }



template<typename eT>
inline
bool
approx_equal_cube(const dev_mem_t<eT> A,
                  const uword A_row_offset,
                  const uword A_col_offset,
                  const uword A_slice_offset,
                  const uword A_M_n_rows,
                  const uword A_M_n_cols,
                  const dev_mem_t<eT> B,
                  const uword B_row_offset,
                  const uword B_col_offset,
                  const uword B_slice_offset,
                  const uword B_M_n_rows,
                  const uword B_M_n_cols,
                  const uword n_rows,
                  const uword n_cols,
                  const uword n_slices,
                  const char sig,
                  const eT abs_tol,
                  const eT rel_tol)
  {
  coot_debug_sigprint();

  const uword n_elem = n_rows * n_cols * n_slices;
  if (n_elem == 0) { return true; }

  const bool do_abs = (sig == 'a' || sig == 'b');
  const bool do_rel = (sig == 'r' || sig == 'b');

  runtime_t& rt = get_rt().vk_rt;

  const eT* A_ptr = reinterpret_cast<const eT*>(rt.get_pool_mapped()) + A.vk_mem_ptr.offset;
  const eT* B_ptr = reinterpret_cast<const eT*>(rt.get_pool_mapped()) + B.vk_mem_ptr.offset;

  const uword A_slice_stride = A_M_n_rows * A_M_n_cols;
  const uword B_slice_stride = B_M_n_rows * B_M_n_cols;

  bool equal = true;

  for (uword s = 0; s < n_slices && equal; ++s)
    {
    for (uword col = 0; col < n_cols && equal; ++col)
      {
      for (uword row = 0; row < n_rows && equal; ++row)
        {
        const uword A_idx = (A_row_offset + row)
                          + (A_col_offset + col) * A_M_n_rows
                          + (A_slice_offset + s) * A_slice_stride;
        const uword B_idx = (B_row_offset + row)
                          + (B_col_offset + col) * B_M_n_rows
                          + (B_slice_offset + s) * B_slice_stride;

        const eT A_val = A_ptr[A_idx];
        const eT B_val = B_ptr[B_idx];

        if (coot_isnan(A_val) || coot_isnan(B_val))
          {
          equal = false;
          break;
          }

        const eT absdiff = vk_abs_diff(A_val, B_val);

        if (do_abs)
          {
          if (absdiff > abs_tol) { equal = false; break; }
          }

        if (do_rel)
          {
          const eT max_val = std::max(vk_abs_val(A_val), vk_abs_val(B_val));
          if (max_val >= eT(1))
            {
            if (absdiff > rel_tol * max_val) { equal = false; break; }
            }
          else if (max_val > eT(0))
            {
            if (absdiff / max_val > rel_tol) { equal = false; break; }
            }
          }
        }
      }
    }

  return equal;
  }
