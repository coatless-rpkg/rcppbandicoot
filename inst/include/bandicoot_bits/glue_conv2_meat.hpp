// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2023 Ryan Curtin (https://www.ratml.org)
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



template<typename out_eT, typename T1, typename T2>
inline
void
glue_conv2::apply(Mat<out_eT>& out, const Glue<T1, T2, glue_conv2>& in)
  {
  coot_extra_debug_sigprint();

  // This implementation is a pretty simple im2col-based implementation.

  // TODO: handle transposed inputs and other delayed inputs or optimizations
  unwrap<T1> UA(in.A);
  unwrap<T2> UB(in.B);

  const uword mode = in.aux_uword;

  apply_direct(out, UA.M, UB.M, mode);
  }



template<typename out_eT, typename eT>
inline
void
glue_conv2::apply_direct(Mat<out_eT>& out, const Mat<eT>& A_in, const Mat<eT>& B_in, const uword mode, const typename enable_if<is_same_type<out_eT, eT>::no>::result* junk)
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  Mat<eT> tmp;
  apply_direct(tmp, A_in, B_in, mode);
  out.set_size(tmp.n_rows, tmp.n_cols);
  coot_rt_t::copy_mat(out.get_dev_mem(false), tmp.get_dev_mem(false),
                      tmp.n_rows, tmp.n_cols,
                      0, 0, out.n_rows,
                      0, 0, tmp.n_rows);
  }



template<typename eT>
inline
void
glue_conv2::apply_direct(Mat<eT>& out, const Mat<eT>& A_in, const Mat<eT>& B_in, const uword mode)
  {
  coot_extra_debug_sigprint();

  // We compute with A, the "constant" matrix, and K, the "kernel" that we rotate.
  // Armadillo selects the "kernel" based on maximum number of elements.
  // However, here we use the number of rows; our code that repacks A into a format where we can use gemvs to compute the result requires it.
  const Mat<eT>& A = (A_in.n_rows >= B_in.n_rows) ? A_in : B_in;
  const Mat<eT>& K = (A_in.n_rows >= B_in.n_rows) ? B_in : A_in;

  // If `A` is the same as `out`, we need to use a temporary matrix as output.
  Mat<eT> out_tmp;
  Mat<eT>& out_ref = (&A == &out) ? out_tmp : out;

  // First let's start with the trivial implementation.
  // Here we treat the smaller matrix (the kernel, call it B) as a row vector.
  // Then, we have to create a "buffer" matrix with size A.n_rows x (B.n_cols * A.n_cols).
  // We will extract copies of the windows being used by each output element with a custom kernel.
  // These copies will correspond only to interleaved columns, not interleaved rows.
  // Thus we will need to loop over rows.
  // In any case, this gives a gemv call (not a gemm call because we only have one kernel).
  // The output of the gemv call is a row of the output matrix.
  // Then we will iterate over rows (B.n_rows) and do the same thing to get each row of the output matrix.

  Mat<eT> buffer;

  // We want to restrict the size of our temporary buffer so that it's not larger than A and K combined.
  // But, we also need to make sure it's big enough to hold a single column of patches...
  // (We also set a lower maximum size equivalent to an 1024x1024 matrix.)
  uword max_buffer_size = (std::max)((std::max)(A.n_elem + K.n_elem, K.n_elem * (A.n_cols + K.n_cols - 1)), (uword) 1048576);
  uword buffer_n_rows = 0;
  uword buffer_n_cols = 0;
  uword out_n_rows = 0;
  uword out_n_cols = 0;
  uword buffer_top_padding = 0;
  uword buffer_bottom_padding = 0;
  uword buffer_row_offset = 0;
  uword buffer_col_offset = 0;

  if (mode == 0)
    {
    // "full"
    get_gemv_full_sizes(A, K, buffer_n_rows, buffer_n_cols, out_n_rows, out_n_cols, buffer_top_padding, buffer_bottom_padding, buffer_row_offset, buffer_col_offset);
    }
  else
    {
    // "same"
    // Note that we have to create the buffer differently depending on what the output size is.
    if (&A == &A_in)
      {
      get_gemv_same_sizes(A, K, buffer_n_rows, buffer_n_cols, out_n_rows, out_n_cols, buffer_top_padding, buffer_bottom_padding, buffer_row_offset, buffer_col_offset);
      }
    else
      {
      get_gemv_same_sizes_small(A, K, buffer_n_rows, buffer_n_cols, out_n_rows, out_n_cols, buffer_top_padding, buffer_bottom_padding, buffer_row_offset, buffer_col_offset);
      }
    }

  if (A.n_rows == 0 || A.n_cols == 0 || K.n_rows == 0 || K.n_cols == 0)
    {
    out.reset();
    return;
    }

  // The kernel "K" needs to be repacked into a vector in a specific way:
  // Specifically, we need to flip K and store it in a column-major form.
  Col<eT> K_mod(K.n_elem);
  coot_rt_t::rotate_180(K_mod.get_dev_mem(false), K.get_dev_mem(false), K.n_rows, K.n_cols);

  out_ref.set_size(out_n_rows, out_n_cols);

  const uword cols_per_batch = std::min(buffer_n_cols, max_buffer_size / buffer_n_rows); // rounds down
  buffer.set_size(buffer_n_rows, cols_per_batch);

  // Pad the top and bottom of the buffer with zeros to correspond to regions where K's columns do not fully overlap with A's columns.
  fill_gemv_buffer_top_bottom(buffer, (K.n_cols - 1), (K.n_cols - 1), K.n_rows);

  // Loop over chunks of the output matrix so that we don't exceed the temporary memory limit.
  // Note that the terminology here is confusing:
  //   `cols_per_batch` refers to the number of columns in `buffer`,
  //   but each column of `buffer` corresponds to a *row* in `out`;
  //   this is why we iterate from c = 0 to c = out_n_rows.
  for (uword c = 0; c < out_n_rows; c += cols_per_batch)
    {
    const uword num_cols = (c + cols_per_batch > out_n_rows) ? (out_n_rows - c) : cols_per_batch;

    // Now fill the buffer.
    for (size_t i = c; i < c + num_cols; ++i)
      {
      fill_gemv_buffer_col(buffer, i + buffer_row_offset, i - c, A, K, buffer_top_padding, buffer_col_offset);
      }

    // Multiply the flattened kernel with the patches of each row of the buffer to get the results.
    for (uword i = 0; i < out_ref.n_cols; ++i)
      {
      coot_rt_t::gemv<eT, true>(out_ref.get_dev_mem(false),
                                buffer.get_dev_mem(false),
                                K.n_elem, // this is the number of rows in each column
                                num_cols,
                                K_mod.get_dev_mem(false),
                                1.0,
                                0.0,
                                i * out_ref.n_rows + c,
                                1,
                                i * K.n_rows,
                                0,
                                buffer.n_rows, // this is the actual number of rows in `buffer`
                                0,
                                1);
      }
    }

  // Reset alias if needed.
  if (&out_ref != &out)
    {
    out.steal_mem(out_ref);
    }
  }



template<typename eT>
inline
void
glue_conv2::get_gemv_full_sizes(const Mat<eT>& A, const Mat<eT>& K, uword& buffer_n_rows, uword& buffer_n_cols, uword& out_n_rows, uword& out_n_cols, uword& buffer_top_padding, uword& buffer_bottom_padding, uword& buffer_row_offset, uword& buffer_col_offset)
  {
  coot_extra_debug_sigprint();

  buffer_n_rows = K.n_rows * (A.n_cols + 2 * (K.n_cols - 1));
  buffer_n_cols = A.n_rows + K.n_rows - 1;

  out_n_rows = A.n_rows + K.n_rows - 1;
  out_n_cols = A.n_cols + K.n_cols - 1;

  buffer_top_padding = K.n_cols - 1;
  buffer_bottom_padding = K.n_cols - 1;

  buffer_row_offset = 0;
  buffer_col_offset = 0;
  }



template<typename eT>
inline
void
glue_conv2::get_gemv_same_sizes(const Mat<eT>& A, const Mat<eT>& K, uword& buffer_n_rows, uword& buffer_n_cols, uword& out_n_rows, uword& out_n_cols, uword& buffer_top_padding, uword& buffer_bottom_padding, uword& buffer_row_offset, uword& buffer_col_offset)
  {
  coot_extra_debug_sigprint();

  // This is the same as create_gemv_full_buffer()---but here the output matrix is the size of A.
  // Thus, there is some padding, but not as much as the "full" strategy.
  //
  // We use the same logic as Armadillo/Octave for determining how much zero padding to apply on each side.

  buffer_n_rows = K.n_rows * (A.n_cols + K.n_cols - 1);
  buffer_n_cols = A.n_rows;

  out_n_rows = A.n_rows;
  out_n_cols = A.n_cols;

  const uword start_row = uword( K.n_rows / 2 );
  const uword start_col = uword( K.n_cols / 2 );

  buffer_top_padding = K.n_cols - start_col - 1;
  buffer_bottom_padding = K.n_cols - buffer_top_padding - 1;

  buffer_row_offset = start_row;
  buffer_col_offset = 0;
  }



template<typename eT>
inline
void
glue_conv2::get_gemv_same_sizes_small(const Mat<eT>& A, const Mat<eT>& K, uword& buffer_n_rows, uword& buffer_n_cols, uword& out_n_rows, uword& out_n_cols, uword& buffer_top_padding, uword& buffer_bottom_padding, uword& buffer_row_offset, uword& buffer_col_offset)
  {
  coot_extra_debug_sigprint();

  // This is the same as create_gemv_same_buffer()---but here the output matrix is the size of K, not the size of A.
  // Note that K.n_rows < A.n_rows.
  //
  // We use the same logic as Armadillo/Octave for determining how much zero padding to apply on each side.

  buffer_n_rows = K.n_rows * (K.n_cols + K.n_cols - 1);
  buffer_n_cols = K.n_rows;

  out_n_rows = K.n_rows;
  out_n_cols = K.n_cols;

  const uword start_row = uword( A.n_rows / 2 );
  const uword start_col = uword( A.n_cols / 2 );

  // The buffer top padding for first full overlap is 0.
  // We are looking for the place where `start_col + 1` rows overlap.
  buffer_top_padding = (start_col < K.n_cols) ? K.n_cols - start_col - 1 : 0;
  buffer_bottom_padding = (start_col < K.n_cols) ? (2 * K.n_cols - 1 - A.n_cols - buffer_top_padding) : 0;

  buffer_row_offset = start_row;
  buffer_col_offset = (start_col < (K.n_cols - 1)) ? 0 : (start_col - (K.n_cols - 1));
  }



template<typename eT>
inline
void
glue_conv2::fill_gemv_buffer_top_bottom(Mat<eT>& buffer, const uword buffer_top_padding, const uword buffer_bottom_padding, const uword kernel_rows)
  {
  // The top and bottom rows of the buffer correspond to sections where K's columns do not fully overlap with A's columns.
  // We zero these out with operations equivalent to the following:
  //    buffer.rows(0, kernel_rows * buffer_top_padding - 1) = 0
  //    buffer.rows(buffer.n_rows - kernel_rows * buffer_bottom_padding, buffer.n_rows - 1) = 0
  if (buffer_top_padding > 0)
    {
    coot_rt_t::fill(buffer.get_dev_mem(false),
                    (eT) 0,
                    kernel_rows * buffer_top_padding,
                    buffer.n_cols,
                    0,
                    0,
                    buffer.n_rows);
    }

  if (buffer_bottom_padding > 0)
    {
    coot_rt_t::fill(buffer.get_dev_mem(false),
                    (eT) 0,
                    kernel_rows * buffer_bottom_padding,
                    buffer.n_cols,
                    buffer.n_rows - (kernel_rows * buffer_bottom_padding),
                    0,
                    buffer.n_rows);
    }
  }



// `i` is the index into the "full" buffer
// `j` is the index of the column in `buffer` that we will use
template<typename eT>
inline
void
glue_conv2::fill_gemv_buffer_col(Mat<eT>& buffer, const uword i, const uword j, const Mat<eT>& A, const Mat<eT>& K, const uword buffer_top_padding, const uword A_col_offset)
  {
  const uword cols_to_copy = (std::min)(A.n_cols - A_col_offset, buffer.n_rows / K.n_rows);

  if (i < K.n_rows - 1)
    {
    // This column corresponds to where K does not yet fully overlap A.
    //
    // Rows in the range [0, K.n_rows - i - 2] are filled with zeros.
    // The way that we do this is a little bit clever, but treat buffer.col(j) as a matrix of size K.n_rows x A.n_cols (call it bufmat_j).
    // Note that for buffer.col(j) we ignore the top and bottom row zero padding.
    // Then, we can say:
    //    bufmat_j.submat(0, 0, K.n_rows - i - 2, A.n_cols - 1) = 0
    //    bufmat_j.submat(K.n_rows - i - 1, 0, K.n_rows, A.n_cols - 1) = A.submat(0, 0, i, A.n_cols - 1)
    coot_rt_t::fill(buffer.get_dev_mem(false),
                    (eT) 0,
                    K.n_rows - i - 1,
                    cols_to_copy,
                    j * buffer.n_rows + K.n_rows * buffer_top_padding, // manual offset
                    0,
                    K.n_rows);

    coot_rt_t::copy_mat(buffer.get_dev_mem(false), A.get_dev_mem(false),
                        i + 1, cols_to_copy,
                        K.n_rows - i - 1 + ((j * buffer.n_rows) + K.n_rows * buffer_top_padding), 0, K.n_rows,
                        0, A_col_offset, A.n_rows);
    }
  else if (i < A.n_rows)
    {
    // This column corresponds to the region where K fully overlaps A.
    const uword A_row = i - (K.n_rows - 1);

    // Copy each individual block.
    // Equivalent to:
    //    buffer.col(j) = vectorise(A.submat(A_row, 0, A_row + K.n_rows - 1, A.n_cols - 1))
    // Note that for buffer.col(j) we ignore the top and bottom row zero padding.
    coot_rt_t::copy_mat(buffer.get_dev_mem(false), A.get_dev_mem(false),
                        K.n_rows, cols_to_copy,
                        j * buffer.n_rows + K.n_rows * buffer_top_padding, 0, K.n_rows,
                        A_row, A_col_offset, A.n_rows);
    }
  else if (i < A.n_rows + 2 * (K.n_rows - 1))
    {
    // Each individual patch from A has its last (i - (kernel_rows + A.n_rows - 1) + 1) rows filled with zeros.
    // (That's rows [(i - (kernel_rows + A.n_rows - 1) + 1), kernel_rows - 1].)
    const uword num_zero_rows = i - A.n_rows + 1;

    // The way that we do this is a little bit clever, but treat buffer.col(j) as a matrix of size kernel_rows x A.n_cols (call it bufmat_j).
    // Then, we can say:
    //    bufmat_j.submat(0, 0, K.n_rows - num_zero_rows - 1, A.n_cols - 1) = A.submat(i - K.n_rows - 1, 0, A.n_rows - 1, A.n_cols - 1)
    //    bufmat_j.submat(K.n_rows - num_zero_rows, 0, K.n_rows - 1, A.n_cols - 1) = 0
    // Note that for buffer.col(j) (or bufmat_j) we ignore the top and bottom zero padding.
    coot_rt_t::copy_mat(buffer.get_dev_mem(false), A.get_dev_mem(false),
                        K.n_rows - num_zero_rows, cols_to_copy,
                        j * buffer.n_rows + K.n_rows * buffer_top_padding, 0, K.n_rows,
                        i - (K.n_rows - 1), A_col_offset, A.n_rows);
    coot_rt_t::fill(buffer.get_dev_mem(false),
                    (eT) 0,
                    num_zero_rows,
                    cols_to_copy,
                    /* manual offset */ (j * buffer.n_rows + K.n_rows * buffer_top_padding) + /* row offset */ (K.n_rows - num_zero_rows),
                    0,
                    K.n_rows);
    }
  }



template<typename T1, typename T2>
inline
uword
glue_conv2::compute_n_rows(const Glue<T1, T2, glue_conv2>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(A_n_cols);
  coot_ignore(B_n_cols);

  if (glue.aux_uword == 0)
    {
    // full
    return A_n_rows + B_n_rows - 1;
    }
  else
    {
    // same
    return A_n_rows;
    }
  }



template<typename T1, typename T2>
inline
uword
glue_conv2::compute_n_cols(const Glue<T1, T2, glue_conv2>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols)
  {
  coot_ignore(A_n_rows);
  coot_ignore(B_n_rows);

  if (glue.aux_uword == 0)
    {
    // full
    return A_n_cols + B_n_cols - 1;
    }
  else
    {
    // same
    return A_n_cols;
    }
  }
