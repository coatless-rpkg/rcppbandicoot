// Copyright 2023 Ryan Curtin (https://www.ratml.org/)
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
or_subgroup_reduce_other_u32(__local volatile uint* data, UWORD tid)
  {
  for(UWORD i = SUBGROUP_SIZE; i > 0; i >>= 1)
    {
    if (tid < i)
      data[tid] |= data[tid + i];
    SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
    }
  }



void
or_subgroup_reduce_8_u32(__local volatile uint* data, UWORD tid)
  {
  data[tid] |= data[tid + 8];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 4];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 2];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 1];
  }



void
or_subgroup_reduce_16_u32(__local volatile uint* data, UWORD tid)
  {
  data[tid] |= data[tid + 16];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 8];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 4];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 2];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 1];
  }



void
or_subgroup_reduce_32_u32(__local volatile uint* data, UWORD tid)
  {
  data[tid] |= data[tid + 32];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 16];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 8];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 4];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 2];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 1];
  }



void
or_subgroup_reduce_64_u32(__local volatile uint* data, UWORD tid)
  {
  data[tid] |= data[tid + 64];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 32];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 16];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 8];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 4];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 2];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 1];
  }



void
or_subgroup_reduce_128_u32(__local volatile uint* data, UWORD tid)
  {
  data[tid] |= data[tid + 128];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 64];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 32];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 16];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 8];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 4];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 2];
  SUBGROUP_BARRIER(CLK_LOCAL_MEM_FENCE);
  data[tid] |= data[tid + 1];
  }
