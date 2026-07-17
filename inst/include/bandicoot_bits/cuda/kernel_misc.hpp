// Copyright 2019 Ryan Curtin (http://www.ratml.org)
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



inline kernel_dims create_kernel_dims()
  {
  kernel_dims k = {{1, 1, 1, 1, 1, 1}};
  return k;
  }



/**
 * Compute one-dimensional grid and block dimensions.
 *
 * This is primarily useful for elementwise kernels where we just need a thread to do an operation over a large, contiguous array.
 */
inline kernel_dims one_dimensional_grid_dims(const uword n_elem)
  {
  const size_t mtpb = (size_t) get_rt().cuda_rt.dev_prop.maxThreadsPerBlock;
  const size_t max_rows = (size_t) get_rt().cuda_rt.dev_prop.maxThreadsDim[0];

  const size_t max_val = (std::min)(mtpb, max_rows);

  kernel_dims result = create_kernel_dims();
  result.d[3] = (int) (std::min)(max_val, n_elem);
  result.d[0] = (int) ((n_elem + max_val - 1) / max_val);

  return result;
  }



/**
 * Compute two-dimensional grid and block dimensions.
 *
 * This is primarily useful for kernels that operate in a 2-dimensional fashion on a matrix.
 */
inline kernel_dims two_dimensional_grid_dims(const uword n_rows, const uword n_cols)
  {
  const size_t mtpb = (size_t) get_rt().cuda_rt.dev_prop.maxThreadsPerBlock;
  const size_t max_rows = (size_t) get_rt().cuda_rt.dev_prop.maxThreadsDim[0];
  const size_t max_cols = (size_t) get_rt().cuda_rt.dev_prop.maxThreadsDim[1];

  const size_t rows = (size_t) n_rows;
  const size_t cols = (size_t) n_cols;

  kernel_dims result = create_kernel_dims();

  // Ideally, we'd like to fit everything into one block, but that may not be possible.
  //
  // Our restrictions are that the block dimensions cannot exceed `max_rows` in width,
  // cannot exceed `max_cols` in its height, and the total number of elements (rows * cols)
  // must be less than `mtpb` (max threads per block).
  //
  // Because Bandicoot matrices are generally such that threads in a row are going to
  // be accessing adjacent elements of a matrix, we want to maximize the width of the grid as
  // much as possible.
  result.d[3] = rows;
  result.d[4] = cols;

  const size_t rows_bound = (std::min)(mtpb, max_rows);
  const size_t block_rows = (std::min)(rows, rows_bound);

  const size_t cols_bound = (std::min)(max_cols, mtpb / block_rows);
  const size_t block_cols = (std::min)(cols, cols_bound);

  // Now compute how many grid blocks we need.
  const size_t grid_rows = (rows + block_rows - 1) / block_rows;
  const size_t grid_cols = (cols + block_cols - 1) / block_cols;

  result.d[0] = (int) grid_rows;
  result.d[1] = (int) grid_cols;
  result.d[3] = (int) block_rows;
  result.d[4] = (int) block_cols;

  return result;
  }



/**
 * Compute three-dimensional grid and block dimensions.
 *
 * This is primarily useful for kernels that operate in a 3-dimensional fashion on a cube.
 */
inline kernel_dims three_dimensional_grid_dims(const uword n_rows, const uword n_cols, const uword n_slices)
  {
  const size_t mtpb = (size_t) get_rt().cuda_rt.dev_prop.maxThreadsPerBlock;
  const size_t max_rows = (size_t) get_rt().cuda_rt.dev_prop.maxThreadsDim[0];
  const size_t max_cols = (size_t) get_rt().cuda_rt.dev_prop.maxThreadsDim[1];
  const size_t max_slices = (size_t) get_rt().cuda_rt.dev_prop.maxThreadsDim[2];

  const size_t rows = (size_t) n_rows;
  const size_t cols = (size_t) n_cols;
  const size_t slices = (size_t) n_slices;

  kernel_dims result = create_kernel_dims();

  // This is a packing problem: how can we fit the work into as few blocks as possible given that:
  //
  // * no block may exceed `mtpb` total elements
  // * the X dimension cannot exceed `max_rows` elements
  // * the Y dimension cannot exceed `max_cols` elements
  // * the Z dimension cannot exceed `max_slices` elements
  //
  // Because Bandicoot matrices are generally such that threads in a row are going to
  // be accessing adjacent elements of a matrix, we want to maximize the width of the grid as
  // much as possible.

  const size_t rows_bound = (std::min)(max_rows, mtpb);
  const size_t block_rows = (std::min)(rows, rows_bound);

  const size_t cols_bound = (std::min)(max_cols, mtpb / block_rows);
  const size_t block_cols = (std::min)(cols, cols_bound);

  const size_t slices_bound = (std::min)(max_slices, mtpb / (block_rows * block_cols));
  const size_t block_slices = (std::min)(slices, slices_bound);

  // Now compute how many grid blocks we need.
  const size_t grid_rows = (rows + block_rows - 1) / block_rows;
  const size_t grid_cols = (cols + block_cols - 1) / block_cols;
  const size_t grid_slices = (slices + block_slices - 1) / block_slices;

  result.d[0] = (int) grid_rows;
  result.d[1] = (int) grid_cols;
  result.d[2] = (int) grid_slices;
  result.d[3] = (int) block_rows;
  result.d[4] = (int) block_cols;
  result.d[5] = (int) block_slices;

  return result;
  }



template<typename T1>
inline
typename
enable_if2
  <
  (Proxy<T1>::num_dims == 1),
  kernel_dims
  >::result
grid_dims(const Proxy<T1>& x)
  {
  return one_dimensional_grid_dims(x.get_n_elem());
  }



template<typename T1>
inline
typename
enable_if2
  <
  (Proxy<T1>::num_dims == 2),
  kernel_dims
  >::result
grid_dims(const Proxy<T1>& x)
  {
  return two_dimensional_grid_dims(x.get_n_rows(), x.get_n_cols());
  }



template<typename T1>
inline
typename
enable_if2
  <
  (Proxy<T1>::num_dims == 3),
  kernel_dims
  >::result
grid_dims(const Proxy<T1>& x)
  {
  return three_dimensional_grid_dims(x.get_n_rows(), x.get_n_cols(), x.get_n_slices());
  }
