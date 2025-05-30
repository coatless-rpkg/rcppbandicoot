// Copyright 2024 Ryan Curtin (https://www.ratml.org/)
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

void
COOT_FN(PREFIX,index_max_subgroup_reduce_other)(__local volatile eT1* data, __local volatile UWORD* data_uword, UWORD tid)
  {
  for(UWORD i = SUBGROUP_SIZE; i > 0; i >>= 1)
    {
    if (tid < i)
      {
      if (data[tid + i] > data[tid])
        {
        data[tid] = data[tid + i];
        data_uword[tid] = data_uword[tid + i];
        }
      }
    SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
    }
  }



void
COOT_FN(PREFIX,index_max_subgroup_reduce_8)(__local volatile eT1* data, __local volatile UWORD* data_uword, UWORD tid)
  {
  if (data[tid + 8] > data[tid])
    {
    data[tid] = data[tid + 8];
    data_uword[tid] = data_uword[tid + 8];
    }
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  if (data[tid + 4] > data[tid])
    {
    data[tid] = data[tid + 4];
    data_uword[tid] = data_uword[tid + 4];
    }
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  if (data[tid + 2] > data[tid])
    {
    data[tid] = data[tid + 2];
    data_uword[tid] = data_uword[tid + 2];
    }
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  if (data[tid + 1] > data[tid])
    {
    data[tid] = data[tid + 1];
    data_uword[tid] = data_uword[tid + 1];
    }
  }



void
COOT_FN(PREFIX,index_max_subgroup_reduce_16)(__local volatile eT1* data, __local volatile UWORD* data_uword, UWORD tid)
  {
  if (data[tid + 16] > data[tid])
    {
    data[tid] = data[tid + 16];
    data_uword[tid] = data_uword[tid + 16];
    }
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  if (data[tid + 8] > data[tid])
    {
    data[tid] = data[tid + 8];
    data_uword[tid] = data_uword[tid + 8];
    }
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  if (data[tid + 4] > data[tid])
    {
    data[tid] = data[tid + 4];
    data_uword[tid] = data_uword[tid + 4];
    }
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  if (data[tid + 2] > data[tid])
    {
    data[tid] = data[tid + 2];
    data_uword[tid] = data_uword[tid + 2];
    }
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  if (data[tid + 1] > data[tid])
    {
    data[tid] = data[tid + 1];
    data_uword[tid] = data_uword[tid + 1];
    }
  }



void
COOT_FN(PREFIX,index_max_subgroup_reduce_32)(__local volatile eT1* data, __local volatile UWORD* data_uword, UWORD tid)
  {
  if (data[tid + 32] > data[tid])
    {
    data[tid] = data[tid + 32];
    data_uword[tid] = data_uword[tid + 32];
    }
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  if (data[tid + 16] > data[tid])
    {
    data[tid] = data[tid + 16];
    data_uword[tid] = data_uword[tid + 16];
    }
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  if (data[tid + 8] > data[tid])
    {
    data[tid] = data[tid + 8];
    data_uword[tid] = data_uword[tid + 8];
    }
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  if (data[tid + 4] > data[tid])
    {
    data[tid] = data[tid + 4];
    data_uword[tid] = data_uword[tid + 4];
    }
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  if (data[tid + 2] > data[tid])
    {
    data[tid] = data[tid + 2];
    data_uword[tid] = data_uword[tid + 2];
    }
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  if (data[tid + 1] > data[tid])
    {
    data[tid] = data[tid + 1];
    data_uword[tid] = data_uword[tid + 1];
    }
  }



void
COOT_FN(PREFIX,index_max_subgroup_reduce_64)(__local volatile eT1* data, __local volatile UWORD* data_uword, UWORD tid)
  {
  if (data[tid + 64] > data[tid])
    {
    data[tid] = data[tid + 64];
    data_uword[tid] = data_uword[tid + 64];
    }
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  if (data[tid + 32] > data[tid])
    {
    data[tid] = data[tid + 32];
    data_uword[tid] = data_uword[tid + 32];
    }
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  if (data[tid + 16] > data[tid])
    {
    data[tid] = data[tid + 16];
    data_uword[tid] = data_uword[tid + 16];
    }
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  if (data[tid + 8] > data[tid])
    {
    data[tid] = data[tid + 8];
    data_uword[tid] = data_uword[tid + 8];
    }
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  if (data[tid + 4] > data[tid])
    {
    data[tid] = data[tid + 4];
    data_uword[tid] = data_uword[tid + 4];
    }
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  if (data[tid + 2] > data[tid])
    {
    data[tid] = data[tid + 2];
    data_uword[tid] = data_uword[tid + 2];
    }
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  if (data[tid + 1] > data[tid])
    {
    data[tid] = data[tid + 1];
    data_uword[tid] = data_uword[tid + 1];
    }
  }



void COOT_FN(PREFIX,index_max_subgroup_reduce_128)(__local volatile eT1* data, __local volatile UWORD* data_uword, UWORD tid)
  {
  if (data[tid + 128] > data[tid])
    {
    data[tid] = data[tid + 128];
    data_uword[tid] = data_uword[tid + 128];
    }
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  if (data[tid + 64] > data[tid])
    {
    data[tid] = data[tid + 64];
    data_uword[tid] = data_uword[tid + 64];
    }
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  if (data[tid + 32] > data[tid])
    {
    data[tid] = data[tid + 32];
    data_uword[tid] = data_uword[tid + 32];
    }
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  if (data[tid + 16] > data[tid])
    {
    data[tid] = data[tid + 16];
    data_uword[tid] = data_uword[tid + 16];
    }
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  if (data[tid + 8] > data[tid])
    {
    data[tid] = data[tid + 8];
    data_uword[tid] = data_uword[tid + 8];
    }
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  if (data[tid + 4] > data[tid])
    {
    data[tid] = data[tid + 4];
    data_uword[tid] = data_uword[tid + 4];
    }
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  if (data[tid + 2] > data[tid])
    {
    data[tid] = data[tid + 2];
    data_uword[tid] = data_uword[tid + 2];
    }
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  if (data[tid + 1] > data[tid])
    {
    data[tid] = data[tid + 1];
    data_uword[tid] = data_uword[tid + 1];
    }
  }
