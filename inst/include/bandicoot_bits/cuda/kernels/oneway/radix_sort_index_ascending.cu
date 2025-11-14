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
COOT_FN(PREFIX,radix_sort_index_ascending)(eT1* A,
                                           UWORD* A_index,
                                           eT1* tmp_mem,
                                           UWORD* tmp_mem_index,
                                           const UWORD n_elem)
  {
  UWORD* aux_mem = (UWORD*) aux_shared_mem;

  const UWORD tid = threadIdx.x;

  const UWORD num_threads = blockDim.x;
  const UWORD elems_per_thread = (n_elem + num_threads - 1) / num_threads; // this is ceil(n_elem / num_threads)
  const UWORD start_elem = tid * elems_per_thread;
  UWORD end_elem = min((tid + 1) * elems_per_thread, n_elem);

  // Fill A_index with [0, 1, ..., n_elem - 1].
  UWORD i = start_elem;
  while (i + 1 < end_elem)
    {
    A_index[i] = i;
    A_index[i + 1] = i + 1;
    i += 2;
    }
  if (i < end_elem)
    {
    A_index[i] = i;
    }

  __syncthreads();

  UWORD local_counts[2];

  eT1* unsorted_memptr = A;
  UWORD* unsorted_index_memptr = A_index;
  eT1* sorted_memptr = tmp_mem;
  UWORD* sorted_index_memptr = tmp_mem_index;

  // If the type is unsigned, all the work will be done the same way.
  const UWORD max_bit = coot_is_signed(TO_ET1(0)) ? 8 * sizeof(eT1) - 1 : 8 * sizeof(eT1);

  for (UWORD b = 0; b < max_bit; ++b)
    {
    // Step 1: count the number of elements with each bit value that belong to this thread.
    uint_eT1* memptr = reinterpret_cast<uint_eT1*>(unsorted_memptr);

    local_counts[0] = 0; // holds the count of elements with bit value 0
    local_counts[1] = 0; // holds the count of elements with bit value 1

    uint_eT1 mask = (((uint_eT1) 1) << b);

    i = start_elem;
    while (i + 1 < end_elem)
      {
      ++local_counts[(memptr[i    ] & mask) >> b];
      ++local_counts[(memptr[i + 1] & mask) >> b];
      i += 2;
      }
    if (i < end_elem)
      {
      ++local_counts[(memptr[i] & mask) >> b];
      }

    // Step 2: aggregate the counts for all threads.
    aux_mem[tid              ] = local_counts[0];
    aux_mem[tid + num_threads] = local_counts[1];
    __syncthreads();

    // At the end of this, `tid` should be assigned two sections of memory to put its 0 and 1 points in.
    //    aux_mem[tid] should hold the first place to put a 0 point
    //    aux_mem[tid + num_threads] should hold the first place to put a 1 point
    // More specifically:
    //    aux_mem[tid]               := sum_{i = 0}^{tid - 1} aux_mem[i]
    //    aux_mem[tid + num_threads] := sum_{i = 0}^{tid + num_threads - 1} aux_mem[i]
    // which means that this is just a prefix-sum operation on the full length of aux_mem.

    // Step 2a: up-sweep total sum into final element.
    UWORD offset = 1;
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
      aux_mem[2 * num_threads - 1] = 0;
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
        UWORD tmp = aux_mem[ai];
        aux_mem[ai] = aux_mem[bi];
        aux_mem[bi] += tmp;
        }
      __syncthreads();
      }

    // Step 3: move points into the correct place.
    local_counts[0] = aux_mem[tid              ]; // contains the first place we should put a 0 point
    local_counts[1] = aux_mem[tid + num_threads]; // contains the first place we should put a 1 point
    i = start_elem;
    while (i + 1 < end_elem)
      {
      const eT1 val1 = unsorted_memptr[i];
      const UWORD index1 = unsorted_index_memptr[i];
      const UWORD out_index1 = local_counts[((memptr[i] & mask) >> b)]++;
      sorted_memptr[out_index1] = val1;
      sorted_index_memptr[out_index1] = index1;

      const eT1 val2 = unsorted_memptr[i + 1];
      const UWORD index2 = unsorted_index_memptr[i + 1];
      const UWORD out_index2 = local_counts[((memptr[i + 1] & mask) >> b)]++;
      sorted_memptr[out_index2] = val2;
      sorted_index_memptr[out_index2] = index2;

      i += 2;
      }
    if (i < end_elem)
      {
      const eT1 val = unsorted_memptr[i];
      const UWORD index = unsorted_index_memptr[i];
      const UWORD out_index = local_counts[((memptr[i] & mask) >> b)]++;
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

  // If the type is integral, we're now done---we don't have to handle a sign bit differently.
  if (!coot_is_signed(TO_ET1(0)))
    {
    return;
    }

  // Only signed types get here.
  // In both cases, we have to put the 1-bit values before the 0-bit values.
  // But, for floating point signed types, we need to reverse the order of the 1-bit points.
  // So, we need a slightly different implementation for both cases.
  uint_eT1* memptr = reinterpret_cast<uint_eT1*>(unsorted_memptr);
  local_counts[0] = 0;
  local_counts[1] = 0;

  const UWORD last_bit = 8 * sizeof(eT1) - 1;
  uint_eT1 mask = (((uint_eT1) 1) << last_bit);

  i = start_elem;
  while (i + 1 < end_elem)
    {
    ++local_counts[(memptr[i    ] & mask) >> last_bit];
    ++local_counts[(memptr[i + 1] & mask) >> last_bit];
    i += 2;
    }
  if (i < end_elem)
    {
    ++local_counts[(memptr[i] & mask) >> last_bit];
    }

  // local_counts[0] now holds the number of positive points; local_counts[1] holds the number of negative points
  // swap these and perform a prefix sum, as with the rest of the bits
  aux_mem[tid              ] = local_counts[1];
  aux_mem[tid + num_threads] = local_counts[0];
  __syncthreads();

  // Up-sweep total sum into final element.
  UWORD offset = 1;
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
    aux_mem[2 * num_threads - 1] = 0;
    }
  __syncthreads();

  // Down-sweep to build prefix sum.
  for (UWORD s = 1; s <= num_threads; s *= 2)
    {
    offset >>= 1;
    if (tid < s)
      {
      const UWORD ai = offset * (2 * tid + 1) - 1;
      const UWORD bi = offset * (2 * tid + 2) - 1;
      UWORD tmp = aux_mem[ai];
      aux_mem[ai] = aux_mem[bi];
      aux_mem[bi] += tmp;
      }
    __syncthreads();
    }

  // Step 3: move points into the correct place.
  // This is different for integral and floating point types.
  if (coot_is_fp(TO_ET1(0)))
    {
    // Floating point implementation:
    // For negative values, we have things sorted in reverse order, so we need to reverse that in our final swap pass.
    // That means that thread 0's negative points go into the last slots, and the last thread's negative points go into the first slots.
    local_counts[0] = aux_mem[tid + num_threads]; // contains the first place we should put a 0 point (we will move upwards)
    local_counts[1] = aux_mem[num_threads] - aux_mem[tid]; // contains the first place we should put a 1 point (we will move downwards)
    local_counts[1] = (local_counts[1] == 0) ? 0 : local_counts[1] - 1; // avoid underflow
    i = start_elem;
    while (i + 1 < end_elem)
      {
      const eT1 val1 = unsorted_memptr[i];
      const UWORD index1 = unsorted_index_memptr[i];
      const UWORD bit_val1 = ((memptr[i] & mask) >> last_bit);
      const UWORD out_index1 = local_counts[bit_val1];
      const int offset1 = (bit_val1 == 1) ? -1 : 1;
      local_counts[bit_val1] += offset1; // decrements for negative values, increments for positive values
      sorted_memptr[out_index1] = val1;
      sorted_index_memptr[out_index1] = index1;

      const eT1 val2 = unsorted_memptr[i + 1];
      const UWORD index2 = unsorted_index_memptr[i + 1];
      const UWORD bit_val2 = ((memptr[i + 1] & mask) >> last_bit);
      const UWORD out_index2 = local_counts[bit_val2];
      const int offset2 = (bit_val2 == 1) ? -1 : 1;
      local_counts[bit_val2] += offset2; // decrements for negative values, increments for positive values
      sorted_memptr[out_index2] = val2;
      sorted_index_memptr[out_index2] = index2;

      i += 2;
      }
    if (i < end_elem)
      {
      const eT1 val = unsorted_memptr[i];
      const UWORD index = unsorted_index_memptr[i];
      const UWORD bit_val = ((memptr[i] & mask) >> last_bit);
      const UWORD out_index = local_counts[bit_val];
      const int offset = (bit_val == 1) ? -1 : 1;
      local_counts[bit_val] += offset; // decrements for negative values, increments for positive values
      sorted_memptr[out_index] = val;
      sorted_index_memptr[out_index] = index;
      }
    }
  else
    {
    // Signed integral implementation:
    // Here, we have values in the right order, we just need to put the negative values ahead of the positive values.
    local_counts[0] = aux_mem[tid + num_threads];
    local_counts[1] = aux_mem[tid];
    i = start_elem;
    while (i + 1 < end_elem)
      {
      const eT1 val1 = unsorted_memptr[i];
      const UWORD index1 = unsorted_index_memptr[i];
      const UWORD bit_val1 = ((memptr[i] & mask) >> last_bit);
      const UWORD out_index1 = local_counts[bit_val1]++;
      sorted_memptr[out_index1] = val1;
      sorted_index_memptr[out_index1] = index1;

      const eT1 val2 = unsorted_memptr[i + 1];
      const UWORD index2 = unsorted_index_memptr[i + 1];
      const UWORD bit_val2 = ((memptr[i + 1] & mask) >> last_bit);
      const UWORD out_index2 = local_counts[bit_val2]++;
      sorted_memptr[out_index2] = val2;
      sorted_index_memptr[out_index2] = index2;

      i += 2;
      }
    if (i < end_elem)
      {
      const eT1 val = unsorted_memptr[i];
      const UWORD index = unsorted_index_memptr[i];
      const UWORD bit_val = ((memptr[i] & mask) >> last_bit);
      const UWORD out_index = local_counts[bit_val]++;
      sorted_memptr[out_index] = val;
      sorted_index_memptr[out_index] = index;
      }
    }

  // Since there are an even number of bits in every data type (or... well... I am going to assume that!), the sorted result is now in A.
  }
