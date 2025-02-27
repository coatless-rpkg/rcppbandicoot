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
COOT_FN(PREFIX,radix_sort_multi_wg_bit_count)(__global eT1* A,
                                              const UWORD A_offset,
                                              __global uint_eT1* counts,
                                              const UWORD counts_offset,
                                              const UWORD n_elem,
                                              const UWORD sort_type,
                                              const UWORD start_bit)
  {
  const UWORD tid = get_global_id(0);

  const UWORD num_threads = get_global_size(0);
  const UWORD elems_per_thread = (n_elem + num_threads - 1) / num_threads; // this is ceil(n_elem / num_threads)
  const UWORD start_elem = tid * elems_per_thread;
  const UWORD end_elem = min((tid + 1) * elems_per_thread, n_elem);

  const uint_eT1 mask = (((uint_eT1) 3) << start_bit);

  __global uint_eT1* uA = (__global uint_eT1*) A; // so that we can mask elements of A bitwise

  uint_eT1 local_counts[4] = { 0, 0, 0, 0 };

  // Count the number of elements with each bit value (00/01/10/11) that belong
  // to this thread.

  UWORD i = start_elem;
  while (i + 1 < end_elem)
    {
    ++local_counts[(uA[A_offset + i    ] & mask) >> start_bit];
    ++local_counts[(uA[A_offset + i + 1] & mask) >> start_bit];
    i += 2;
    }
  if (i < end_elem)
    {
    ++local_counts[(uA[A_offset + i] & mask) >> start_bit];
    }

  // Save results to the right place for later processing.
  if (sort_type == 0)
    {
    counts[counts_offset + tid                  ] = local_counts[0];
    counts[counts_offset + tid +     num_threads] = local_counts[1];
    counts[counts_offset + tid + 2 * num_threads] = local_counts[2];
    counts[counts_offset + tid + 3 * num_threads] = local_counts[3];
    }
  else if (sort_type == 1)
    {
    // If sort_type == 1 (descending), we want to store the results in the bit
    // order 11/10/01/00, instead of the order of local_counts (00/01/10/11).
    counts[counts_offset + tid                  ] = local_counts[3];
    counts[counts_offset + tid +     num_threads] = local_counts[2];
    counts[counts_offset + tid + 2 * num_threads] = local_counts[1];
    counts[counts_offset + tid + 3 * num_threads] = local_counts[0];
    }
  else if (sort_type == 2)
    {
    // If sort_type == 2 (highest two bits of a signed integer, ascending), we
    // want to store the results in the bit order 10/11/00/01
    counts[counts_offset + tid                  ] = local_counts[2];
    counts[counts_offset + tid +     num_threads] = local_counts[3];
    counts[counts_offset + tid + 2 * num_threads] = local_counts[0];
    counts[counts_offset + tid + 3 * num_threads] = local_counts[1];
    }
  else if (sort_type == 3)
    {
    // If sort_type == 3 (highest two bits of a signed integer, descending), we
    // want to store the results in the bit order 01/00/11/10
    counts[counts_offset + tid                  ] = local_counts[1];
    counts[counts_offset + tid +     num_threads] = local_counts[0];
    counts[counts_offset + tid + 2 * num_threads] = local_counts[3];
    counts[counts_offset + tid + 3 * num_threads] = local_counts[2];
    }
  else if (sort_type == 4 || sort_type == 6)
    {
    // If sort_type == 4 or 6 (highest two bits of floating-point number, ascending),
    // we want to store the results in the bit order 11/10/00/01
    counts[counts_offset + tid                  ] = local_counts[3];
    counts[counts_offset + tid +     num_threads] = local_counts[2];
    counts[counts_offset + tid + 2 * num_threads] = local_counts[0];
    counts[counts_offset + tid + 3 * num_threads] = local_counts[1];
    }
  else if (sort_type == 5 || sort_type == 7)
    {
    // If sort_type == 5 or 7 (highest two bits of floating-point number,
    // descending), we want to store the results in the bit order 01/00/10/11
    counts[counts_offset + tid                  ] = local_counts[1];
    counts[counts_offset + tid +     num_threads] = local_counts[0];
    counts[counts_offset + tid + 2 * num_threads] = local_counts[2];
    counts[counts_offset + tid + 3 * num_threads] = local_counts[3];
    }
  }
