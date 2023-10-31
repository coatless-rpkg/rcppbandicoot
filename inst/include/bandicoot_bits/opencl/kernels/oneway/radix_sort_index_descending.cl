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
COOT_FN(PREFIX,radix_sort_index_descending)(__global eT1* A,
                                            __global UWORD* A_index,
                                            __global eT1* tmp_mem,
                                            __global UWORD* tmp_mem_index,
                                            const UWORD n_elem,
                                            __local volatile uint_eT1* aux_mem)
  {
  const UWORD tid = get_global_id(0);

  const UWORD num_threads = get_global_size(0);
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

  barrier(CLK_LOCAL_MEM_FENCE);

  UWORD local_counts[2];

  __global eT1* unsorted_memptr = A;
  __global UWORD* unsorted_index_memptr = A_index;
  __global eT1* sorted_memptr = tmp_mem;
  __global UWORD* sorted_index_memptr = tmp_mem_index;

  // If the type is unsigned, all the work will be done the same way.
  const UWORD max_bit = COOT_FN(coot_is_signed_,eT1)() ? 8 * sizeof(eT1) - 1 : 8 * sizeof(eT1);

  for (UWORD b = 0; b < max_bit; ++b)
    {
    // Step 1: count the number of elements with each bit value that belong to this thread.
    __global uint_eT1* memptr = (__global uint_eT1*) unsorted_memptr;

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
    aux_mem[tid              ] = local_counts[1];
    aux_mem[tid + num_threads] = local_counts[0];
    barrier(CLK_LOCAL_MEM_FENCE);

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
      barrier(CLK_LOCAL_MEM_FENCE);
      }

    if (tid == 0)
      {
      aux_mem[2 * num_threads - 1] = 0;
      }
    barrier(CLK_LOCAL_MEM_FENCE);

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
      barrier(CLK_LOCAL_MEM_FENCE);
      }

    // Step 3: move points into the correct place.
    local_counts[0] = aux_mem[tid + num_threads]; // contains the first place we should put a 0 point
    local_counts[1] = aux_mem[tid              ]; // contains the first place we should put a 1 point
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
    __global eT1* tmp = unsorted_memptr;
    __global UWORD* tmp_index = unsorted_index_memptr;
    unsorted_memptr = sorted_memptr;
    unsorted_index_memptr = sorted_index_memptr;
    sorted_memptr = tmp;
    sorted_index_memptr = tmp_index;

    barrier(CLK_GLOBAL_MEM_FENCE);
    }

  // If the type is integral, we're now done---we don't have to handle a sign bit differently.
  if (!COOT_FN(coot_is_signed_,eT1)())
    {
    return;
    }

  // Only signed types get here.
  // In both cases, we have to put the 1-bit values before the 0-bit values.
  // But, for floating point signed types, we need to reverse the order of the 1-bit points.
  // So, we need a slightly different implementation for both cases.
  __global uint_eT1* memptr = (__global uint_eT1*) unsorted_memptr;
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
  // perform a prefix sum, as with the rest of the bits
  aux_mem[tid              ] = local_counts[0];
  aux_mem[tid + num_threads] = local_counts[1];
  barrier(CLK_LOCAL_MEM_FENCE);

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
    barrier(CLK_LOCAL_MEM_FENCE);
    }

  if (tid == 0)
    {
    aux_mem[2 * num_threads - 1] = 0;
    }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Down-sweep to build prefix sum.
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
    barrier(CLK_LOCAL_MEM_FENCE);
    }

  // Step 3: move points into the correct place.
  // This is different for integral and floating point types.
  if (COOT_FN(coot_is_fp_,eT1)())
    {
    // Floating point implementation:
    // For negative values, we have things sorted in reverse order, so we need to reverse that in our final swap pass.
    // That means that thread 0's negative points go into the last slots, and the last thread's negative points go into the first slots.
    local_counts[0] = aux_mem[tid]; // contains the first place we should put a 0 point (we will move upwards)
    local_counts[1] = n_elem - 1 - (aux_mem[num_threads + tid] - aux_mem[num_threads]); // contains the first place we should put a 1 point (we will move downwards)
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
    local_counts[0] = aux_mem[tid];
    local_counts[1] = aux_mem[tid + num_threads];
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
