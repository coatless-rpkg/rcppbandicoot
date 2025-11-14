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
COOT_FN(PREFIX,radix_sort_colwise_ascending)(eT1* A,
                                             eT1* tmp_mem,
                                             const UWORD A_n_rows,
                                             const UWORD A_n_cols,
                                             const UWORD A_M_n_rows)
  {
  const UWORD col = blockIdx.x * blockDim.x + threadIdx.x;
  if(col < A_n_cols)
    {
    eT1* unsorted_colptr =       &A[col * A_M_n_rows];
    eT1* sorted_colptr =   &tmp_mem[col * A_n_rows];

    UWORD counts[2];

    // If the type is unsigned, all the work will be done the same way.
    const UWORD max_bit = coot_is_signed(TO_ET1(0)) ? (8 * sizeof(eT1) - 1) : (8 * sizeof(eT1));

    for (UWORD b = 0; b < max_bit; ++b)
      {
      // Since we are sorting bitwise, we should treat the data as unsigned integers to make bitwise operations easy.
      uint_eT1* colptr = reinterpret_cast<uint_eT1*>(unsorted_colptr);

      counts[0] = 0; // holds the count of points with bit value 0
      counts[1] = 0; // holds the count of points with bit value 1

      uint_eT1 mask = (((uint_eT1) 1) << b);

      for (UWORD i = 0; i < A_n_rows; ++i)
        {
        ++counts[((colptr[i] & mask) >> b)];
        }

      counts[1] = counts[0]; // now holds the offset to put the next value at
      counts[0] = 0;

      for (UWORD i = 0; i < A_n_rows; ++i)
        {
        const eT1 val = unsorted_colptr[i];
        const UWORD out_index = counts[((colptr[i] & mask) >> b)]++;
        sorted_colptr[out_index] = val;
        }

      // swap pointers (unsorted is now sorted)
      eT1* tmp = unsorted_colptr;
      unsorted_colptr = sorted_colptr;
      sorted_colptr = tmp;
      }

    // If the type is unsigned, we're now done---we don't have to handle a sign bit differently.
    if (!coot_is_signed(TO_ET1(0)))
      {
      return;
      }

    // Only signed types get here.
    // In both cases, we have to put the 1-bit values before the 0-bit values.
    // But, for floating point signed types, we need to reverse the order of the 1-bit points.
    // So, we need a slightly different implementation for both cases.
    uint_eT1* colptr = reinterpret_cast<uint_eT1*>(unsorted_colptr);
    counts[0] = 0;
    counts[1] = 0;

    const UWORD last_bit = 8 * sizeof(eT1) - 1;
    uint_eT1 mask = (((uint_eT1) 1) << last_bit);

    for (UWORD i = 0; i < A_n_rows; ++i)
      {
      ++counts[((colptr[i] & mask) >> last_bit)];
      }
    // counts[0] now holds the number of positive points; counts[1] holds the number of negative points

    // This is different for integral and floating point types.
    if (coot_is_fp(TO_ET1(0)))
      {
      // Floating point implementation:
      // For negative values, we have things sorted in reverse order, so we need to reverse that in our final swap pass.
      // That means that thread 0's negative points go into the last slots, and the last thread's negative points go into the first slots.
      counts[0] = counts[1];     // now holds the offset to put the next positive value at
      counts[1] = counts[0] - 1; // now holds the offset to put the next negative value at (we move backwards)

      for (UWORD i = 0; i < A_n_rows; ++i)
        {
        const eT1 val = unsorted_colptr[i];
        const UWORD bit_val = ((colptr[i] & mask) >> last_bit);
        const UWORD out_index = counts[bit_val];
        const int offset = (bit_val == 1) ? -1 : 1;
        counts[bit_val] += offset; // decrements for negative values, increments for positive values
        sorted_colptr[out_index] = val;
        }
      }
    else
      {
      // Signed integral implementation:
      // Here, we have values in the right order, we just need to put the negative values ahead of the positive values.
      counts[0] = counts[1]; // now holds the offset to put the next positive value at
      counts[1] = 0;         // now holds the offset to put the next negative value at

      for (UWORD i = 0; i < A_n_rows; ++i)
        {
        const eT1 val = unsorted_colptr[i];
        const UWORD bit_val = ((colptr[i] & mask) >> last_bit);
        const UWORD out_index = counts[bit_val]++;
        sorted_colptr[out_index] = val;
        }
      }
    }

    // Since there are an even number of bits in every data type (or... well... I am going to assume that!), the sorted result is now in A.
  }
