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



template<typename eT>
inline
void
join_cols(dev_mem_t<eT> out,
          const dev_mem_t<eT> A,
          const uword A_n_rows,
          const uword A_n_cols,
          const dev_mem_t<eT> B,
          const uword B_n_rows,
          const uword B_n_cols,
          const dev_mem_t<eT> C,
          const uword C_n_rows,
          const uword C_n_cols,
          const dev_mem_t<eT> D,
          const uword D_n_rows,
          const uword D_n_cols,
          // subview arguments
          const uword out_row_offset,
          const uword out_col_offset,
          const uword out_M_n_rows,
          const uword A_row_offset,
          const uword A_col_offset,
          const uword A_M_n_rows,
          const uword B_row_offset,
          const uword B_col_offset,
          const uword B_M_n_rows,
          const uword C_row_offset,
          const uword C_col_offset,
          const uword C_M_n_rows,
          const uword D_row_offset,
          const uword D_col_offset,
          const uword D_M_n_rows)
  {
  coot_extra_debug_sigprint();

  // When the types are all the same, we can use cudaMemcpy2D().

  const uword A_n_elem = A_n_rows * A_n_cols;
  const uword B_n_elem = B_n_rows * B_n_cols;
  const uword C_n_elem = C_n_rows * C_n_cols;
  const uword D_n_elem = D_n_rows * D_n_cols;

  cudaError_t result;
  if (A_n_elem > 0)
    {
    result = coot_wrapper(cudaMemcpy2D)((void*) (out.cuda_mem_ptr + out_row_offset + out_col_offset * out_M_n_rows),
                                        out_M_n_rows * sizeof(eT),
                                        (void*) (A.cuda_mem_ptr + A_row_offset + A_col_offset * A_M_n_rows),
                                        A_M_n_rows * sizeof(eT),
                                        A_n_rows * sizeof(eT),
                                        A_n_cols,
                                        cudaMemcpyDeviceToDevice);
    coot_check_cuda_error(result, "coot::cuda::join_cols(): could not copy first argument");
    }

  if (B_n_elem > 0)
    {
    result = coot_wrapper(cudaMemcpy2D)((void*) (out.cuda_mem_ptr + A_n_rows + out_row_offset + out_col_offset * out_M_n_rows),
                                        out_M_n_rows * sizeof(eT),
                                        (void*) (B.cuda_mem_ptr + B_row_offset + B_col_offset * B_M_n_rows),
                                        B_M_n_rows * sizeof(eT),
                                        B_n_rows * sizeof(eT),
                                        B_n_cols,
                                        cudaMemcpyDeviceToDevice);
    coot_check_cuda_error(result, "coot::cuda::join_cols(): could not copy second argument");
    }

  if (C_n_elem > 0)
    {
    result = coot_wrapper(cudaMemcpy2D)((void*) (out.cuda_mem_ptr + A_n_rows + B_n_rows + out_row_offset + out_col_offset * out_M_n_rows),
                                        out_M_n_rows * sizeof(eT),
                                        (void*) (C.cuda_mem_ptr + C_row_offset + C_col_offset * C_M_n_rows),
                                        C_M_n_rows * sizeof(eT),
                                        C_n_rows * sizeof(eT),
                                        C_n_cols,
                                        cudaMemcpyDeviceToDevice);
    coot_check_cuda_error(result, "coot::cuda::join_cols(): could not copy third argument");
    }

  if (D_n_elem > 0)
    {
    result = coot_wrapper(cudaMemcpy2D)((void*) (out.cuda_mem_ptr + A_n_rows + B_n_rows + C_n_rows + out_row_offset + out_col_offset * out_M_n_rows),
                                        out_M_n_rows * sizeof(eT),
                                        (void*) (D.cuda_mem_ptr + D_row_offset + D_col_offset * D_M_n_rows),
                                        D_M_n_rows * sizeof(eT),
                                        D_n_rows * sizeof(eT),
                                        D_n_cols,
                                        cudaMemcpyDeviceToDevice);
    coot_check_cuda_error(result, "coot::cuda::join_cols(): could not copy fourth argument");
    }
  }



template<typename eT1, typename eT2, typename eT3, typename eT4, typename eT5>
inline
void
join_cols(dev_mem_t<eT5> out,
          const dev_mem_t<eT1> A,
          const uword A_n_rows,
          const uword A_n_cols,
          const dev_mem_t<eT2> B,
          const uword B_n_rows,
          const uword B_n_cols,
          const dev_mem_t<eT3> C,
          const uword C_n_rows,
          const uword C_n_cols,
          const dev_mem_t<eT4> D,
          const uword D_n_rows,
          const uword D_n_cols,
          // subview arguments
          const uword out_row_offset,
          const uword out_col_offset,
          const uword out_M_n_rows,
          const uword A_row_offset,
          const uword A_col_offset,
          const uword A_M_n_rows,
          const uword B_row_offset,
          const uword B_col_offset,
          const uword B_M_n_rows,
          const uword C_row_offset,
          const uword C_col_offset,
          const uword C_M_n_rows,
          const uword D_row_offset,
          const uword D_col_offset,
          const uword D_M_n_rows)
  {
  coot_extra_debug_sigprint();

  const uword A_n_elem = A_n_rows * A_n_cols;
  const uword B_n_elem = B_n_rows * B_n_cols;
  const uword C_n_elem = C_n_rows * C_n_cols;
  const uword D_n_elem = D_n_rows * D_n_cols;

  // If the types are different, we need to perform a cast during the copy.
  if (A_n_elem > 0)
    {
    copy_mat(out, A,
             A_n_rows, A_n_cols,
             out_row_offset, out_col_offset, out_M_n_rows,
             A_row_offset, A_col_offset, A_M_n_rows);
    }

  if (B_n_elem > 0)
    {
    copy_mat(out, B,
             B_n_rows, B_n_cols,
             A_n_rows + out_row_offset, out_col_offset, out_M_n_rows,
             B_row_offset, B_col_offset, B_M_n_rows);
    }

  if (C_n_elem > 0)
    {
    copy_mat(out, C,
             C_n_rows, C_n_cols,
             A_n_rows + B_n_rows + out_row_offset, out_col_offset, out_M_n_rows,
             C_row_offset, C_col_offset, C_M_n_rows);
    }

  if (D_n_elem > 0)
    {
    copy_mat(out, D,
             D_n_rows, D_n_cols,
             A_n_rows + B_n_rows + C_n_rows + out_row_offset, out_col_offset, out_M_n_rows,
             D_row_offset, D_col_offset, D_M_n_rows);
    }
  }
