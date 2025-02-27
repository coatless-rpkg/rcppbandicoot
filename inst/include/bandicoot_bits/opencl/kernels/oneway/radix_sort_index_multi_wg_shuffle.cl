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
COOT_FN(PREFIX,radix_sort_index_multi_wg_shuffle)(__global eT1* A,
                                                  const UWORD A_offset,
                                                  __global UWORD* A_index,
                                                  const UWORD A_index_offset,
                                                  __global eT1* out,
                                                  const UWORD out_offset,
                                                  __global UWORD* out_index,
                                                  const UWORD out_index_offset,
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

  __global uint_eT1* uA = (__global uint_eT1*) A;
  const uint_eT1 mask = (((uint_eT1) 3) << start_bit);

  int upper_bit_shift = 1;
  uint_eT1 local_offsets[4];
  if (sort_type == 0)
    {
    // for an ascending sort, the offsets are ordered for bit values 00/01/10/11
    local_offsets[0] = counts[counts_offset + tid                  ]; // first place we should put a 00 point
    local_offsets[1] = counts[counts_offset + tid +     num_threads]; // first place we should put a 01 point
    local_offsets[2] = counts[counts_offset + tid + 2 * num_threads]; // first place we should put a 10 point
    local_offsets[3] = counts[counts_offset + tid + 3 * num_threads]; // first place we should put a 11 point
    }
  else if (sort_type == 1)
    {
    // for a descending sort, the offsets are ordered for bit values 11/10/01/00
    local_offsets[0] = counts[counts_offset + tid + 3 * num_threads]; // first place we should put a 00 point
    local_offsets[1] = counts[counts_offset + tid + 2 * num_threads]; // first place we should put a 01 point
    local_offsets[2] = counts[counts_offset + tid +     num_threads]; // first place we should put a 10 point
    local_offsets[3] = counts[counts_offset + tid                  ]; // first place we should put a 11 point
    }
  else if (sort_type == 2)
    {
    // for the last bits of a signed integer in an ascending sort, the offsets are ordered for bit values (10/11/00/01)
    local_offsets[0] = counts[counts_offset + tid + 2 * num_threads]; // first place we should put a 00 point
    local_offsets[1] = counts[counts_offset + tid + 3 * num_threads]; // first place we should put a 01 point
    local_offsets[2] = counts[counts_offset + tid                  ]; // first place we should put a 10 point
    local_offsets[3] = counts[counts_offset + tid     + num_threads]; // first place we should put a 11 point
    }
  else if (sort_type == 3)
    {
    // for the last bits of a signed integer in a descending sort, the offsets are ordered for bit values (01/00/11/10)
    local_offsets[0] = counts[counts_offset + tid     + num_threads]; // first place we should put a 00 point
    local_offsets[1] = counts[counts_offset + tid                  ]; // first place we should put a 01 point
    local_offsets[2] = counts[counts_offset + tid + 3 * num_threads]; // first place we should put a 10 point
    local_offsets[3] = counts[counts_offset + tid + 2 * num_threads]; // first place we should put a 11 point
    }
  else if (sort_type == 4)
    {
    // for the last bits of a floating-point number in an ascending sort, the offsets are ordered for bit values (11/10/00/01)
    // and, the negative values are ordered in a descending order, so we have to reverse them
    local_offsets[0] = counts[counts_offset + tid + 2 * num_threads]; // first place we should put a 00 point
    local_offsets[1] = counts[counts_offset + tid + 3 * num_threads]; // first place we should put a 01 point
    local_offsets[2] = counts[counts_offset + 2 * num_threads] - counts[counts_offset + tid     + num_threads]; // first place we should put a 10 point
    local_offsets[3] = counts[counts_offset +     num_threads] - counts[counts_offset + tid                  ]; // first place we should put a 11 point

    // avoid underflow
    local_offsets[2] = (local_offsets[2] == 0) ? 0 : local_offsets[2] - 1;
    local_offsets[3] = (local_offsets[3] == 0) ? 0 : local_offsets[3] - 1;

    upper_bit_shift = -1; // sort negative values backwards
    }
  else if (sort_type == 5)
    {
    // for the last bits of a floating-point number in a descending sort, the offsets are ordered for bit values (01/00/10/11)
    // and, the negative values are ordered in a descending order, so we have to reverse them
    local_offsets[0] = counts[counts_offset + tid +     num_threads]; // first place we should put a 00 point
    local_offsets[1] = counts[counts_offset + tid                  ]; // first place we should put a 01 point
    local_offsets[2] = counts[counts_offset + 3 * num_threads] - (counts[counts_offset + tid + 2 * num_threads] - counts[counts_offset + 2 * num_threads]); // first place we should put a 10 point
    local_offsets[3] = n_elem                                  - (counts[counts_offset + tid + 3 * num_threads] - counts[counts_offset + 3 * num_threads]); // first place we should put a 11 point

    // avoid underflow
    local_offsets[2] = (local_offsets[2] == 0) ? 0 : local_offsets[2] - 1;
    local_offsets[3] = (local_offsets[3] == 0) ? 0 : local_offsets[3] - 1;

    upper_bit_shift = -1; // sort negative values backwards
    }
  else if (sort_type == 6)
    {
    // for the last bits of a floating-point number in a stable ascending sort,
    // the offsets are ordered for bit values (11/10/00/01)
    // but, we do not need to reverse any values
    local_offsets[0] = counts[tid + 2 * num_threads]; // first place we should put a 00 point
    local_offsets[1] = counts[tid + 3 * num_threads]; // first place we should put a 01 point
    local_offsets[2] = counts[tid +     num_threads]; // first place we should put a 10 point
    local_offsets[3] = counts[tid                  ]; // first place we should put a 11 point
    }
  else if (sort_type == 7)
    {
    // for the last bits of a floating-point number in a stable descending sort,
    // the offsets are ordered for bit values (01/00/10/11)
    // but, we do not need to reverse any values
    local_offsets[0] = counts[tid +     num_threads]; // first place we should put a 00 point
    local_offsets[1] = counts[tid                  ]; // first place we should put a 01 point
    local_offsets[2] = counts[tid + 2 * num_threads]; // first place we should put a 10 point
    local_offsets[3] = counts[tid + 3 * num_threads]; // first place we should put a 11 point
    }

  // Move all points that this thread is responsible for into the correct place.
  UWORD i = start_elem;
  while (i + 1 < end_elem)
    {
    const uint_eT1 val1 = uA[A_offset + i    ];
    const uint_eT1 val2 = uA[A_offset + i + 1];

    const uint_eT1 loc1 = ((val1 & mask) >> start_bit);
    const uint_eT1 loc2 = ((val2 & mask) >> start_bit);

    const uint_eT1 out_index1 = local_offsets[loc1];
    local_offsets[loc1] += ((loc1 >= 2) ? upper_bit_shift : 1);
    const uint_eT1 out_index2 = local_offsets[loc2];
    local_offsets[loc2] += ((loc2 >= 2) ? upper_bit_shift : 1);

    out[out_offset + out_index1] = A[A_offset + i];
    out_index[out_index_offset + out_index1] = A_index[A_index_offset + i];

    out[out_offset + out_index2] = A[A_offset + i + 1];
    out_index[out_offset + out_index2] = A_index[A_index_offset + i + 1];

    i += 2;
    }
  if (i < end_elem)
    {
    const uint_eT1 val = uA[A_offset + i];
    const uint_eT1 loc = ((val & mask) >> start_bit);
    const uint_eT1 out_index1 = local_offsets[loc];
    local_offsets[loc] += ((loc >= 2) ? upper_bit_shift : 1);
    out[out_offset + out_index1] = A[A_offset + i];
    out_index[out_index_offset + out_index1] = A_index[A_index_offset + i];
    }
  }
