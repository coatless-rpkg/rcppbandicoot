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

__global__
void
COOT_FN(PREFIX,submat_var_small)(const eT1* in_mem,
                                 const UWORD n_elem,
                                 eT1* out_mem,
                                 const eT1 mean_val,
                                 const UWORD in_n_rows,
                                 const UWORD start_row,
                                 const UWORD start_col,
                                 const UWORD sub_n_rows,
                                 const UWORD sub_n_cols)
  {
  eT1* aux_mem = (eT1*) aux_shared_mem;

  const UWORD tid = threadIdx.x;
  UWORD i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  const UWORD grid_size = blockDim.x * 2 * gridDim.x;

  aux_mem[tid] = 0;

  while (i + blockDim.x < n_elem)
    {
    // copy to local shared memory
    const UWORD col1 = (i             ) / sub_n_rows;
    const UWORD col2 = (i + blockDim.x) / sub_n_rows;
    const UWORD row1 = (i             ) % sub_n_rows;
    const UWORD row2 = (i + blockDim.x) % sub_n_rows;
    const UWORD index1 = (col1 + start_col) * in_n_rows + (row1 + start_row);
    const UWORD index2 = (col2 + start_col) * in_n_rows + (row2 + start_row);

    const eT1 val1 = (in_mem[index1] - mean_val);
    const eT1 val2 = (in_mem[index2] - mean_val);
    aux_mem[tid] += (val1 * val1) + (val2 * val2);
    i += grid_size;
    }
  if (i < n_elem)
    {
    const UWORD col = i / sub_n_rows;
    const UWORD row = i % sub_n_rows;
    const UWORD index = (col + start_col) * in_n_rows + (row + start_row);

    const eT1 val = (in_mem[index] - mean_val);
    aux_mem[tid] += (val * val);
    }

  for (UWORD s = blockDim.x / 2; s > 0; s >>= 1)
    {
    if (tid < s)
      {
      aux_mem[tid] += aux_mem[tid + s];
      }
    }

  if (tid == 0)
    {
    out_mem[blockIdx.x] = aux_mem[0];
    }
  }
