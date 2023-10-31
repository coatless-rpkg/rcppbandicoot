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
COOT_FN(PREFIX,stable_radix_sort_index_ascending)(eT1* A,
                                                  UWORD* A_index,
                                                  eT1* tmp_mem,
                                                  UWORD* tmp_mem_index,
                                                  const UWORD n_elem)
  {
  // The stable sort differs from the rest of our radix sorts in that we must avoid ever "reversing" point orders.
  // We do this by adapting the regular radix sort to also consider the highest bit (the sign bit for signed types).
  // This alleviates the need to ever unpack points in a reverse order, and so the sort is stable.

  uint_eT1* aux_mem = (uint_eT1*) aux_shared_mem;

  const UWORD tid = threadIdx.x;

  const UWORD num_threads = blockDim.x;
  const UWORD elems_per_thread = (n_elem + num_threads - 1) / num_threads; // this is ceil(n_elem / num_threads)
  const UWORD start_elem = tid * elems_per_thread;
  UWORD end_elem = min((tid + 1) * elems_per_thread, n_elem);

  // Fill tmp_mem_index with [0, 1, ..., n_elem - 1].
  UWORD i = start_elem;
  while (i + 1 < end_elem)
    {
    tmp_mem_index[i] = i;
    tmp_mem_index[i + 1] = i + 1;
    i += 2;
    }
  if (i < end_elem)
    {
    tmp_mem_index[i] = i;
    }

  __syncthreads();

  // This is 4 instead of two because we need to account for the sign bit.
  UWORD local_counts[4];

  // We are doing an odd number of iterations, so set things up such that A_index will be holding the final results.
  eT1* unsorted_memptr = A;
  UWORD* unsorted_index_memptr = tmp_mem_index;
  eT1* sorted_memptr = tmp_mem;
  UWORD* sorted_index_memptr = A_index;

  const UWORD last_bit = 8 * sizeof(eT1) - 1;
  uint_eT1 sign_mask = (((uint_eT1) 1) << last_bit);

  for (UWORD b = 0; b < 8 * sizeof(eT1) - 1; ++b)
    {
    // Step 1: count the number of elements with each bit value that belong to this thread.
    uint_eT1* memptr = reinterpret_cast<uint_eT1*>(unsorted_memptr);

    local_counts[0] = 0; // holds the count of elements with sign 0 and bit value 0
    local_counts[1] = 0; // holds the count of elements with sign 0 and bit value 1
    local_counts[2] = 0; // holds the count of elements with sign 1 and bit value 0
    local_counts[3] = 0; // holds the count of elements with sign 1 and bit value 1

    uint_eT1 mask = (((uint_eT1) 1) << b);

    i = start_elem;
    while (i + 1 < end_elem)
      {
      ++local_counts[((memptr[i    ] & mask) >> b) + ((memptr[i    ] & sign_mask) >> (last_bit - 1))];
      ++local_counts[((memptr[i + 1] & mask) >> b) + ((memptr[i + 1] & sign_mask) >> (last_bit - 1))];
      i += 2;
      }
    if (i < end_elem)
      {
      ++local_counts[((memptr[i] & mask) >> b) + ((memptr[i] & sign_mask) >> (last_bit - 1))];
      }

    // Step 2: aggregate the counts for all threads.
    // There are a couple cases here to get things in an ascending order:
    //  * Floating point number: [11, 10, 00, 01]
    //  * Unsigned integer:      [00, 01, 10, 11]
    //  * Signed integer:        [10, 11, 00, 01]
    // Note that the notation "11" indicates, e.g., a point whose sign is 1 and bit value in bit b is 1.
    // For unsigned integers, we treat the top bit as a "sign" bit even though it's not---but we choose an ordering that's still correct.

    if (!coot_is_signed((eT1) 0))
      {
      // Unsigned integer (00, 01, 10, 11)
      aux_mem[tid                  ] = local_counts[0];
      aux_mem[tid +     num_threads] = local_counts[1];
      aux_mem[tid + 2 * num_threads] = local_counts[2];
      aux_mem[tid + 3 * num_threads] = local_counts[3];
      }
    else if (coot_is_fp((eT1) 0))
      {
      // Floating point (11, 10, 00, 01)
      aux_mem[tid                  ] = local_counts[3];
      aux_mem[tid +     num_threads] = local_counts[2];
      aux_mem[tid + 2 * num_threads] = local_counts[0];
      aux_mem[tid + 3 * num_threads] = local_counts[1];
      }
    else
      {
      // Signed integer (10, 11, 00, 01)
      aux_mem[tid                  ] = local_counts[2];
      aux_mem[tid +     num_threads] = local_counts[3];
      aux_mem[tid + 2 * num_threads] = local_counts[0];
      aux_mem[tid + 3 * num_threads] = local_counts[1];
      }
    __syncthreads();

    // Now, we must assign four sections of memory for `tid` to put its points in.
    // We do this by a prefix-sum operation across all threads.
    // At the end of this operation (at the beginning of Step 3):
    //
    //    local_counts[0] indicates the first place to put a sign-0 bit-value-0 point
    //    local_counts[1] indicates the first place to put a sign-0 bit-value-1 point
    //    local_counts[2] indicates the first place to put a sign-1 bit-value-0 point
    //    local_counts[3] indicates the first place to put a sign-1 bit-value-1 point

    // Step 2a: up-sweep total sum into final element.
    UWORD offset = 1;

    // Since we have auxiliary memory size of 4x the number of threads, we need to add an extra iteration where each thread handles two values.
    const UWORD ai1 = offset * (2 * tid + 1) - 1;
    const UWORD bi1 = offset * (2 * tid + 2) - 1;
    aux_mem[bi1] += aux_mem[ai1];
    const UWORD ai2 = offset * (2 * (tid + num_threads) + 1) - 1;
    const UWORD bi2 = offset * (2 * (tid + num_threads) + 2) - 1;
    aux_mem[bi2] += aux_mem[ai2];
    offset *= 2;
    __syncthreads();

    for (UWORD s = num_threads; s > 0; s >>= 1)
      {
      if (tid < s)
        {
        const UWORD ai = offset * (2 * tid + 1) - 1;
        const UWORD bi = offset * (2 * tid + 2) - 1;
        aux_mem[bi] += aux_mem[ai];
        }
      offset *= 2;
      __syncthreads();
      }

    if (tid == 0)
      {
      aux_mem[4 * num_threads - 1] = 0;
      }
    __syncthreads();

    // Step 2b: down-sweep to build prefix sum.
    for (UWORD s = 1; s <= num_threads; s *= 2)
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
      __syncthreads();
      }

    // Since we have auxiliary memory size of 4x the number of threads, we need to add an extra iteration where each thread handles two values.
    offset >>= 1;
    const UWORD ai3 = offset * (2 * tid + 1) - 1;
    const UWORD bi3 = offset * (2 * tid + 2) - 1;
    uint_eT1 tmp3 = aux_mem[ai3];
    aux_mem[ai3] = aux_mem[bi3];
    aux_mem[bi3] += tmp3;

    const UWORD ai4 = offset * (2 * (tid + num_threads) + 1) - 1;
    const UWORD bi4 = offset * (2 * (tid + num_threads) + 2) - 1;
    uint_eT1 tmp4 = aux_mem[ai4];
    aux_mem[ai4] = aux_mem[bi4];
    aux_mem[bi4] += tmp4;
    __syncthreads();

    // Step 3: move points into the correct place.
    // There are a couple cases here to get things in an ascending order:
    //  * Floating point number: [11, 10, 00, 01]
    //  * Unsigned integer:      [00, 01, 10, 11]
    //  * Signed integer:        [10, 11, 00, 01]

    if (!coot_is_signed((eT1) 0))
      {
      // Unsigned integer (00, 01, 10, 11)
      local_counts[0] = aux_mem[tid                  ];
      local_counts[1] = aux_mem[tid +     num_threads];
      local_counts[2] = aux_mem[tid + 2 * num_threads];
      local_counts[3] = aux_mem[tid + 3 * num_threads];
      }
    else if (coot_is_fp((eT1) 0))
      {
      // Floating point (11, 10, 00, 01)
      local_counts[0] = aux_mem[tid + 2 * num_threads];
      local_counts[1] = aux_mem[tid + 3 * num_threads];
      local_counts[2] = aux_mem[tid +     num_threads];
      local_counts[3] = aux_mem[tid                  ];
      }
    else
      {
      // Signed integer (10, 11, 00, 01)
      local_counts[0] = aux_mem[tid + 2 * num_threads];
      local_counts[1] = aux_mem[tid + 3 * num_threads];
      local_counts[2] = aux_mem[tid                  ];
      local_counts[3] = aux_mem[tid +     num_threads];
      }

    i = start_elem;
    while (i + 1 < end_elem)
      {
      const eT1 val1 = unsorted_memptr[i];
      const UWORD index1 = unsorted_index_memptr[i];
      const UWORD out_index1 = local_counts[((memptr[i] & mask) >> b) + ((memptr[i] & sign_mask) >> (last_bit - 1))]++;
      sorted_memptr[out_index1] = val1;
      sorted_index_memptr[out_index1] = index1;

      const eT1 val2 = unsorted_memptr[i + 1];
      const UWORD index2 = unsorted_index_memptr[i + 1];
      const UWORD out_index2 = local_counts[((memptr[i + 1] & mask) >> b) + ((memptr[i + 1] & sign_mask) >> (last_bit - 1))]++;
      sorted_memptr[out_index2] = val2;
      sorted_index_memptr[out_index2] = index2;

      i += 2;
      }
    if (i < end_elem)
      {
      const eT1 val = unsorted_memptr[i];
      const UWORD index = unsorted_index_memptr[i];
      const UWORD out_index = local_counts[((memptr[i] & mask) >> b) + ((memptr[i] & sign_mask) >> (last_bit - 1))]++;
      sorted_memptr[out_index] = val;
      sorted_index_memptr[out_index] = index;
      }

    // Now swap pointers.
    eT1* tmp = unsorted_memptr;
    unsorted_memptr = sorted_memptr;
    sorted_memptr = tmp;
    UWORD* tmp_index = unsorted_index_memptr;
    unsorted_index_memptr = sorted_index_memptr;
    sorted_index_memptr = tmp_index;

    __syncthreads();
    }

  // Since we did an odd number of iterations, the result is stored in A_index.
  }
