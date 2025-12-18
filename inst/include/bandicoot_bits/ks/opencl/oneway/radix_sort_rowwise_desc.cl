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
COOT_FN(PREFIX,radix_sort_rowwise_desc)(__global eT1* A,
                                        const UWORD A_offset,
                                        __global eT1* tmp_mem,
                                        const UWORD A_n_rows,
                                        const UWORD A_n_cols,
                                        const UWORD A_M_n_rows)
  {
  const UWORD row = get_global_id(0);
  if(row < A_n_rows)
    {
    __global eT1* unsorted_rowptr =       &A[A_offset + row];
    __global eT1* sorted_rowptr =   &tmp_mem[           row];

    UWORD unsorted_n_rows = A_M_n_rows;
    UWORD sorted_n_rows   = A_n_rows;

    UWORD counts[2];

    // If the type is unsigned, all the work will be done the same way.
    const UWORD max_bit = COOT_FN(coot_is_signed_,eT1)() ? (8 * sizeof(eT1) - 1) : (8 * sizeof(eT1));

    for (UWORD b = 0; b < max_bit; ++b)
      {
      // Since we are sorting bitwise, we should treat the data as unsigned integers to make bitwise operations easy.
      __global uint_eT1* rowptr = (__global uint_eT1*) unsorted_rowptr;

      counts[0] = 0; // holds the count of points with bit value 0
      counts[1] = 0; // holds the count of points with bit value 1

      uint_eT1 mask = (((uint_eT1) 1) << b);

      for (UWORD i = 0; i < A_n_cols; ++i)
        {
        ++counts[(rowptr[i * unsorted_n_rows] & mask) >> b];
        }

      counts[0] = counts[1]; // now holds the offset to put the next value at
      counts[1] = 0;

      for (UWORD i = 0; i < A_n_cols; ++i)
        {
        const UWORD in_index = i * unsorted_n_rows;
        const eT1 val = unsorted_rowptr[in_index];
        const UWORD out_index = (counts[((rowptr[in_index] & mask) >> b)]++) * sorted_n_rows;
        sorted_rowptr[out_index] = val;
        }

      // swap pointers (unsorted is now sorted)
      __global eT1* tmp = unsorted_rowptr;
      unsorted_rowptr = sorted_rowptr;
      sorted_rowptr = tmp;

      UWORD tmp2 = unsorted_n_rows;
      unsorted_n_rows = sorted_n_rows;
      sorted_n_rows = tmp2;
      }

    // If the type is unsigned, we're now done---we don't have to handle a sign bit differently.
    if (!COOT_FN(coot_is_signed_,eT1)())
      {
      return;
      }

    // Only signed types get here.
    // In both cases, we have to put the 1-bit values before the 0-bit values.
    // But, for floating point signed types, we need to reverse the order of the 1-bit points.
    // So, we need a slightly different implementation for both cases.
    __global uint_eT1* rowptr = (__global uint_eT1*) unsorted_rowptr;

    const UWORD last_bit = 8 * sizeof(eT1) - 1;
    uint_eT1 mask = (((uint_eT1) 1) << last_bit);

    if (COOT_FN(coot_is_fp_,eT1)())
      {
      counts[0] = 0;            // now holds the offset to put the next positive value at
      counts[1] = A_n_cols - 1; // now holds the offset to put the next negative value at (we move backwards)

      for (UWORD i = 0; i < A_n_cols; ++i)
        {
        const UWORD in_index = i * unsorted_n_rows;
        const eT1 val = unsorted_rowptr[in_index];
        const UWORD bit_val = ((rowptr[in_index] & mask) >> last_bit);
        const UWORD out_index = counts[bit_val] * sorted_n_rows;
        const int offset = (bit_val == 1) ? -1 : 1;
        counts[bit_val] += offset; // decrements for negative values, increments for positive values
        sorted_rowptr[out_index] = val;
        }
      }
    else
      {
      counts[0] = 0;
      counts[1] = 0;

      for (UWORD i = 0; i < A_n_cols; ++i)
        {
        ++counts[(rowptr[i * unsorted_n_rows] & mask) >> last_bit];
        }
      // counts[0] now holds the number of positive points; counts[1] holds the number of negative points

      counts[1] = counts[0]; // now holds the offset to put the next positive value at
      counts[0] = 0;         // now holds the offset to put the next negative value at

      for (UWORD i = 0; i < A_n_cols; ++i)
        {
        const UWORD in_index = i * unsorted_n_rows;
        const eT1 val = unsorted_rowptr[in_index];
        const UWORD bit_val = ((rowptr[in_index] & mask) >> last_bit);
        const UWORD out_index = (counts[bit_val]++) * sorted_n_rows;
        sorted_rowptr[out_index] = val;
        }
      }
    }

  // Since there are an even number of bits in every data type (or... well... I am going to assume that!), the sorted result is now in A.
  }
