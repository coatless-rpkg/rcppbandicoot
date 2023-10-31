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
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot::opencl::median(): OpenCL runtime not valid" );

  if (dim == 0)
    {
    // Sort the data in each column.
    sort(src, n_rows, n_cols, 0, 0, src_row_offset, src_col_offset, src_M_n_rows);
    const uword middle_element = (n_rows / 2);

    if (n_rows % 2 == 0)
      {
      // Even number of elements: we have to do a little extra processing.
      sum(dest, src,
          2, n_cols,
          0, true,
          dest_offset, dest_mem_incr,
          src_row_offset + (middle_element - 1), src_col_offset, src_M_n_rows);

      eop_scalar(twoway_kernel_id::equ_array_div_scalar_post,
                 dest, dest,
                 eT2(2), eT2(1),
                 1, n_cols,
                 dest_offset, 0, dest_mem_incr,
                 dest_offset, 0, dest_mem_incr);
      }
    else
      {
      // Odd number of elements: the middle element is the result.
      // Now extract that row into the output.
      copy_mat(dest, src,
               1, n_cols,
               dest_offset, 0, dest_mem_incr,
               src_row_offset + middle_element, src_col_offset, src_M_n_rows);
      }
    }
  else
    {
    // Sort the data in each row.
    sort(src, n_rows, n_cols, 0, 1, src_row_offset, src_col_offset, src_M_n_rows);
    const uword middle_element = (n_cols / 2);

    if (n_cols % 2 == 0)
      {
      // Even number of elements: we have to do a little extra processing.
      sum(dest, src,
          n_rows, 2,
          1, true,
          dest_offset, dest_mem_incr,
          src_row_offset, src_col_offset + (middle_element - 1), src_M_n_rows);

      eop_scalar(twoway_kernel_id::equ_array_div_scalar_post,
                 dest, dest,
                 eT2(2), eT2(1),
                 // logically treat as column vector so dest_mem_incr can be used
                 1, n_rows,
                 dest_offset, 0, dest_mem_incr,
                 dest_offset, 0, dest_mem_incr);
      }
    else
      {
      // Odd number of elements: the middle element is the result.
      // Now extract the column into the output.
      copy_mat(dest, src,
               // logically treat as column vector so dest_mem_incr can be used
               1, n_rows,
               dest_offset, 0, dest_mem_incr,
               src_row_offset + (src_col_offset + middle_element) * n_rows, 0, 1);
      }
    }
  }



template<typename eT>
inline
eT
median_vec(dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot::opencl::median_vec(): OpenCL runtime not valid" );

  // Sort the data.
  sort_vec(mem, n_elem, 0);
  // Now get the median element.
  const uword middle_element = n_elem / 2;
  if (n_elem % 2 == 0)
    {
    // Even number of elements: average the two middle elements.
    eT val1 = get_val(mem, middle_element - 1);
    eT val2 = get_val(mem, middle_element);
    return (val1 + val2) / 2;
    }
  else
    {
    // Odd number of elements: the easy case.
    return get_val(mem, middle_element);
    }
  }
