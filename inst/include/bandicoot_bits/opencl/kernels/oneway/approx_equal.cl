// Copyright 2023 Ryan Curtin (http://www.ratml.org/)
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

__kernel
void
COOT_FN(PREFIX,approx_equal)(__global uint* out_mem,
                             __global const eT1* A_mem,
                             const UWORD A_offset,
                             const UWORD A_M_n_rows,
                             __global const eT1* B_mem,
                             const UWORD B_offset,
                             const UWORD B_M_n_rows,
                             const UWORD n_rows,
                             const UWORD n_elem,
                             __local volatile uint* aux_mem,
                             const UWORD mode,
                             const eT1 abs_tol,
                             const eT1 rel_tol)
  {
  const UWORD tid = get_local_id(0);
  UWORD i = get_group_id(0) * (get_local_size(0) * 2) + tid;
  const UWORD grid_size = get_local_size(0) * 2 * get_num_groups(0);

  aux_mem[tid] = 1;

  while (i + get_local_size(0) < n_elem)
    {
    // A bit painful...
    const UWORD row1 = i % n_rows;
    const UWORD col1 = i / n_rows;
    const UWORD row2 = (i + get_local_size(0)) % n_rows;
    const UWORD col2 = (i + get_local_size(0)) / n_rows;

    const UWORD A_loc1 = A_offset + row1 + col1 * A_M_n_rows;
    const UWORD A_loc2 = A_offset + row2 + col2 * A_M_n_rows;
    const UWORD B_loc1 = B_offset + row1 + col1 * B_M_n_rows;
    const UWORD B_loc2 = B_offset + row2 + col2 * B_M_n_rows;

    const eT1 A_val1 = A_mem[A_loc1];
    const eT1 B_val1 = B_mem[B_loc1];
    const eT1 A_val2 = A_mem[A_loc2];
    const eT1 B_val2 = B_mem[B_loc2];

    if (COOT_FN(coot_isnan_,eT1)(A_val1) || COOT_FN(coot_isnan_,eT1)(B_val1) || COOT_FN(coot_isnan_,eT1)(A_val2) || COOT_FN(coot_isnan_,eT1)(B_val2))
      {
      // Not approximately equal.
      aux_mem[tid] &= 0;
      }

    const eT1 absdiff1 = COOT_FN(coot_absdiff_,eT1)(A_val1, B_val1);
    const eT1 absdiff2 = COOT_FN(coot_absdiff_,eT1)(A_val2, B_val2);

    if ((mode & 1) == 1) // absolute
      {
      aux_mem[tid] &= (absdiff1 <= abs_tol);
      aux_mem[tid] &= (absdiff2 <= abs_tol);
      }

    if ((mode & 2) == 2) // relative
      {
      const eT1 max_val1 = max(ET1_ABS(A_val1), ET1_ABS(B_val1));
      const eT1 max_val2 = max(ET1_ABS(A_val2), ET1_ABS(B_val2));

      if (max_val1 >= (eT1) 1)
        {
        aux_mem[tid] &= (absdiff1 <= rel_tol * max_val1);
        aux_mem[tid] &= (absdiff2 <= rel_tol * max_val2);
        }
      else
        {
        aux_mem[tid] &= (absdiff1 / max_val1 <= rel_tol);
        aux_mem[tid] &= (absdiff2 / max_val2 <= rel_tol);
        }
      }

    i += grid_size;
    }
  if (i < n_elem)
    {
    const UWORD row = i % n_rows;
    const UWORD col = i / n_rows;

    const UWORD A_loc = A_offset + row + col * A_M_n_rows;
    const UWORD B_loc = B_offset + row + col * B_M_n_rows;

    const eT1 A_val = A_mem[A_loc];
    const eT1 B_val = B_mem[B_loc];

    if (COOT_FN(coot_isnan_,eT1)(A_val) || COOT_FN(coot_isnan_,eT1)(B_val))
      {
      // Not approximately equal.
      aux_mem[tid] &= 0;
      }

    const eT1 absdiff = COOT_FN(coot_absdiff_,eT1)(A_val, B_val);

    if ((mode & 1) == 1) // absolute
      {
      aux_mem[tid] &= (absdiff <= abs_tol);
      }

    if ((mode & 2) == 2) // relative
      {
      const eT1 max_val = max(ET1_ABS(A_val), ET1_ABS(B_val));

      if (max_val >= (eT1) 1)
        {
        aux_mem[tid] &= (absdiff <= rel_tol * max_val);
        }
      else
        {
        aux_mem[tid] &= (absdiff / max_val <= rel_tol);
        }
      }
    }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (UWORD s = get_local_size(0) / 2; s > SUBGROUP_SIZE; s >>= 1)
    {
    if (tid < s)
      {
      aux_mem[tid] &= aux_mem[tid + s];
      }
    barrier(CLK_LOCAL_MEM_FENCE);
    }

  if (tid < SUBGROUP_SIZE)
    {
    COOT_FN(u32_and_subgroup_reduce_,SUBGROUP_SIZE_NAME)(aux_mem, tid);
    }

  if (tid == 0)
    {
    out_mem[get_group_id(0)] = aux_mem[0];
    }
  }
