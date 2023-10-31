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
COOT_FN(PREFIX,count_nonzeros)(__global const eT1* A,
                               __global UWORD* thread_counts,
                               const UWORD n_elem,
                               __local volatile uint_eT1* aux_mem)
  {
  // We want to pass over the memory in A and count the number of nonzero elements.
  // This will give us a count for each individual thread; we then want to prefix-sum this.
  // This kernel is meant to be used as the first part of find().

  const UWORD tid = get_global_id(0);

  const UWORD num_threads = get_global_size(0);
  const UWORD elems_per_thread = (n_elem + num_threads - 1) / num_threads; // this is ceil(n_elem / num_threads)
  const UWORD start_elem = tid * elems_per_thread;
  UWORD end_elem = min((tid + 1) * elems_per_thread, n_elem);

  UWORD local_count = 0;

  UWORD i = start_elem;
  while (i + 1 < end_elem)
    {
    if (A[i] != (eT1) 0)
      {
      ++local_count;
      }
    if (A[i + 1] != (eT1) 0)
      {
      ++local_count;
      }

    i += 2;
    }
  if (i < end_elem)
    {
    if (A[i] != (eT1) 0)
      {
      ++local_count;
      }
    }

  // Aggregate the counts for all threads.
  aux_mem[tid] = local_count;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Up-sweep total sum into final element.
  UWORD offset = 1;

  for (UWORD s = num_threads / 2; s > 0; s >>= 1)
    {
    if (tid < s)
      {
      const UWORD ai = offset * (2 * tid + 1) - 1;
      const UWORD bi = offset * (2 * tid + 2) - 1;
      aux_mem[bi] += aux_mem[ai];
      }
    offset *= 2;
    barrier(CLK_LOCAL_MEM_FENCE);
    }

  if (tid == 0)
    {
    // Set the last element correctly.
    thread_counts[num_threads] = aux_mem[num_threads - 1];
    aux_mem[num_threads - 1] = 0;
    }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Down-sweep to build prefix sum.
  for (UWORD s = 1; s <= num_threads / 2; s *= 2)
    {
    offset >>= 1;
    if (tid < s)
      {
      const UWORD ai = offset * (2 * tid + 1) - 1;
      const UWORD bi = offset * (2 * tid + 2) - 1;
      uint_eT1 tmp = aux_mem[ai];
      aux_mem[ai] = aux_mem[bi];
      aux_mem[bi] += tmp;
      }
    barrier(CLK_LOCAL_MEM_FENCE);
    }

  thread_counts[tid] = aux_mem[tid];
  }
