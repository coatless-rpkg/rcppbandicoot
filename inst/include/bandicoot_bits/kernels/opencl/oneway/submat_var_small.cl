// Copyright 2023 Ryan Curtin (http://www.ratml.org)
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
COOT_FN(PREFIX,submat_var_small)(__global const eT1* in_mem,
                                 const UWORD in_mem_offset,
                                 const UWORD n_elem,
                                 __global eT1* out_mem,
                                 const UWORD out_mem_offset,
                                 __local volatile eT1* aux_mem,
                                 const eT1 mean_val,
                                 const UWORD in_n_rows,
                                 const UWORD start_row,
                                 const UWORD start_col,
                                 const UWORD sub_n_rows,
                                 const UWORD sub_n_cols)
  {
  const UWORD tid = get_local_id(0);
  UWORD i = get_group_id(0) * (get_local_size(0) * 2) + tid;
  const UWORD grid_size = get_local_size(0) * 2 * get_num_groups(0);

  aux_mem[tid] = 0;

  while (i + get_local_size(0) < n_elem)
    {
    const UWORD col1 = (i                    ) / sub_n_rows;
    const UWORD col2 = (i + get_local_size(0)) / sub_n_rows;
    const UWORD row1 = (i                    ) % sub_n_rows;
    const UWORD row2 = (i + get_local_size(0)) % sub_n_rows;
    const UWORD index1 = (col1 + start_col) * in_n_rows + (row1 + start_row);
    const UWORD index2 = (col2 + start_col) * in_n_rows + (row2 + start_row);

    const eT1 val1 = (in_mem[in_mem_offset + index1] - mean_val);
    const eT1 val2 = (in_mem[in_mem_offset + index2] - mean_val);
    aux_mem[tid] += (val1 * val1) + (val2 * val2);
    i += grid_size;
    }
  if (i < n_elem)
    {
    const UWORD col = i / sub_n_rows;
    const UWORD row = i % sub_n_rows;
    const UWORD index = (col + start_col) * in_n_rows + (row + start_row);

    const eT1 val = (in_mem[in_mem_offset + index] - mean_val);
    aux_mem[tid] += (val * val);
    }

  for (UWORD s = get_local_size(0) / 2; s > 0; s >>= 1)
    {
    SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);

    if (tid < s)
      {
      aux_mem[tid] += aux_mem[tid + s];
      }
    }

  if (tid == 0)
    {
    out_mem[out_mem_offset + get_group_id(0)] = aux_mem[0];
    }
  }
