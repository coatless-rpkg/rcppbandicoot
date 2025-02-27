// Copyright 2024 Ryan Curtin (http://www.ratml.org/)
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

__device__
void
COOT_FN(PREFIX,index_min_warp_reduce)(volatile eT1* data, volatile UWORD* uword_data, int tid)
  {
  if (data[tid + 32] < data[tid])
    {
    data[tid] = data[tid + 32];
    uword_data[tid] = uword_data[tid + 32];
    }

  if (data[tid + 16] < data[tid])
    {
    data[tid] = data[tid + 16];
    uword_data[tid] = uword_data[tid + 16];
    }

  if (data[tid + 8] < data[tid])
    {
    data[tid] = data[tid + 8];
    uword_data[tid] = uword_data[tid + 8];
    }

  if (data[tid + 4] < data[tid])
    {
    data[tid] = data[tid + 4];
    uword_data[tid] = uword_data[tid + 4];
    }

  if (data[tid + 2] < data[tid])
    {
    data[tid] = data[tid + 2];
    uword_data[tid] = uword_data[tid + 2];
    }

  if (data[tid + 1] < data[tid])
    {
    data[tid] = data[tid + 1];
    uword_data[tid] = uword_data[tid + 1];
    }
  }
