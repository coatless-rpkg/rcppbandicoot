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

  kernel_dims result = create_kernel_dims();
  result.d[3] = (std::min)(mtpb, n_elem);
  result.d[0] = (n_elem + mtpb - 1) / mtpb;

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

  const size_t rows = (size_t) n_rows;
  const size_t cols = (size_t) n_cols;
  const size_t elem = rows * cols;

  kernel_dims result = create_kernel_dims();

  // Ideally, we'd like to fit everything into one block, but that may not be possible.
  result.d[3] = rows;
  result.d[4] = cols;

  if (rows > mtpb)
    {
    // If the number of rows is greater than the maximum threads per block, we can handle one column at a time in each block.
    result.d[3] = mtpb; // blockSize[0]
    result.d[4] = 1;    // blockSize[1]
    result.d[0] = (rows + mtpb - 1) / mtpb; // gridSize[0]
    result.d[1] = cols; // gridSize[1]

    // TODO: what if this is greater than the maximum grid size?  (seems very unlikely)
    }
  else if (elem > mtpb)
    {
    // We can't fit everything in a single block, so we'll process multiple columns in each block.
    result.d[3] = rows;           // blockSize[0]
    result.d[4] = mtpb / rows;    // blockSize[1] ; fit as many columns as we can
    result.d[1] = ((cols + result.d[4] - 1) / result.d[4]); // gridSize[1]
    }

  return result;
  }
