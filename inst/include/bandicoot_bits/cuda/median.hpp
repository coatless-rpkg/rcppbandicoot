// Copyright 2023 Ryan Curtin (http://www.ratml.org)
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



/**
 * Compute the row-wise or column-wise mean of the input matrix, storing the result in the output matrix.
 */
template<typename eT2, typename eT1>
inline
void
median(dev_mem_t<eT2> dest,
       dev_mem_t<eT1> src,
       const uword n_rows,
       const uword n_cols,
       const uword dim,
       // subview arguments
       const uword dest_offset,
       const uword dest_mem_incr,
       const uword src_row_offset,
       const uword src_col_offset,
       const uword src_M_n_rows)
  {
  coot_debug_sigprint();

  coot_check_runtime_error( (get_rt().cuda_rt.is_valid() == false), "coot::cuda::median(): CUDA runtime not valid" );

  if (dim == 0)
    {
    // Sort the data in each column.
    sort(src, n_rows, n_cols, 0, 0, src_row_offset, src_col_offset, src_M_n_rows);
    const uword middle_element = (n_rows / 2);

    if (n_rows % 2 == 0)
      {
      // Even number of elements; we have to do a little extra processing.
      sum(dest, src,
          2, n_cols,
          0, true,
          dest_offset, dest_mem_incr,
          src_row_offset + (middle_element - 1), src_col_offset, src_M_n_rows);

      // Divide by 2.
      Mat<eT2> dest_alias(dest + dest_offset, dest_mem_incr, n_cols);
      dest_alias.row(0) /= 2;
      }
    else
      {
      // Odd number of elements: the middle element is the result.
      // Now extract that row into the output.
      const Proxy<subview<eT2>> PD(dest, dest_offset, 0, 1, n_cols, dest_mem_incr);
      const Proxy<subview<eT1>> PS(src, src_row_offset + middle_element, src_col_offset, 1, n_cols, src_M_n_rows);
      copy(PD, PS);
      }
    }
  else
    {
    // Sort the data in each row.
    sort(src, n_rows, n_cols, 0, 1, src_row_offset, src_col_offset, src_M_n_rows);
    const uword middle_element = (n_cols / 2);

    if (n_cols % 2 == 0)
      {
      // Even number of elements; we have to do a little extra processing.
      sum(dest, src,
          n_rows, 2,
          1, true,
          dest_offset, dest_mem_incr,
          src_row_offset, src_col_offset + (middle_element - 1), src_M_n_rows);

      // Divide by 2.
      Mat<eT2> dest_alias(dest + dest_offset, dest_mem_incr, n_rows);
      dest_alias.row(0) /= 2;
      }
    else
      {
      // Odd number of elements: the middle element is the result.
      // Now extract that column into the output.
      const Proxy<subview<eT2>> PD(dest, dest_offset, 0, 1, n_rows, dest_mem_incr);
      const Proxy<subview<eT1>> PS(src, src_row_offset + (src_col_offset + middle_element) * src_M_n_rows, 0, 1, n_rows, 1);
      copy(PD, PS);
      }
    }
  }



template<typename eT>
inline
eT
median_vec(dev_mem_t<eT> in, const uword n_elem)
  {
  coot_debug_sigprint();

  coot_check_runtime_error( (get_rt().cuda_rt.is_valid() == false), "coot::cuda::median(): CUDA runtime not valid" );

  // Sort the data.
  sort_vec(in, n_elem, 0);
  // Now get the median element.
  const uword middle_element = n_elem / 2;
  if (n_elem % 2 == 0)
    {
    // Even number of elements: average the two middle elements.
    eT val1 = get_val(in, middle_element - 1);
    eT val2 = get_val(in, middle_element);
    return (val1 + val2) / eT(2);
    }
  else
    {
    // Odd number of elements: the easy case.
    return get_val(in, middle_element);
    }
  }
