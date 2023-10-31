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



// Forward declarations we may need.
void COOT_FN(PREFIX,prod_subgroup_reduce_other)(__local volatile eT1* data, UWORD tid);
void COOT_FN(PREFIX,prod_subgroup_reduce_8)(__local volatile eT1* data, UWORD tid);
void COOT_FN(PREFIX,prod_subgroup_reduce_16)(__local volatile eT1* data, UWORD tid);
void COOT_FN(PREFIX,prod_subgroup_reduce_32)(__local volatile eT1* data, UWORD tid);
void COOT_FN(PREFIX,prod_subgroup_reduce_64)(__local volatile eT1* data, UWORD tid);
void COOT_FN(PREFIX,prod_subgroup_reduce_128)(__local volatile eT1* data, UWORD tid);



// Compute the product of the elements on the diagonal of a matrix.
__kernel
void
COOT_FN(PREFIX,diag_prod)(__global const eT1* in_mem,
                           const UWORD n_rows,
                           __global eT1* out_mem,
                           __local volatile eT1* aux_mem)
  {
  const UWORD tid = get_local_id(0);
  UWORD i = get_group_id(0) * (get_local_size(0) * 2) + tid;
  const UWORD grid_size = get_local_size(0) * 2 * get_num_groups(0);

  aux_mem[tid] = 1;

  while (i + get_local_size(0) < n_rows)
    {
    const UWORD index1 = i * n_rows + i;
    const eT1 v1 = in_mem[index1];
    const UWORD index2 = (i + get_local_size(0)) * n_rows + (i + get_local_size(0));
    const eT1 v2 = in_mem[index2];
    aux_mem[tid] *= v1 * v2;
    i += grid_size;
    }
  if (i < n_rows)
    {
    const UWORD index = i * n_rows + i;
    const eT1 v = in_mem[index];
    aux_mem[tid] *= v;
    }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (UWORD s = get_local_size(0) / 2; s > SUBGROUP_SIZE; s >>= 1)
    {
    if (tid < s)
      {
      aux_mem[tid] *= aux_mem[tid + s];
      }
    barrier(CLK_LOCAL_MEM_FENCE);
    }

  if (tid < SUBGROUP_SIZE)
    {
    COOT_FN_3(PREFIX,prod_subgroup_reduce_,SUBGROUP_SIZE_NAME)(aux_mem, tid);
    }

  if (tid == 0)
    {
    out_mem[get_group_id(0)] = aux_mem[0];
    }
  }
