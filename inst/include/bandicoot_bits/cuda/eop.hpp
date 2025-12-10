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



/**
 * Run a CUDA elementwise kernel.
 */
template<typename eT1, typename eT2>
inline
void
eop_scalar(const twoway_kernel_id::enum_id num,
           dev_mem_t<eT2> dest,
           const dev_mem_t<eT1> src,
           const eT1 aux_val_pre,
           const eT2 aux_val_post,
           // logical size of source and destination
           const uword n_rows,
           const uword n_cols,
           const uword n_slices,
           // submatrix destination offsets (set to 0, 0, and n_rows if not a subview)
           const uword dest_row_offset,
           const uword dest_col_offset,
           const uword dest_slice_offset,
           const uword dest_M_n_rows,
           const uword dest_M_n_cols,
           // submatrix source offsets (set to 0, 0, and n_rows if not a subview)
           const uword src_row_offset,
           const uword src_col_offset,
           const uword src_slice_offset,
           const uword src_M_n_rows,
           const uword src_M_n_cols)
  {
  coot_extra_debug_sigprint();

  if (n_rows == 0 || n_cols == 0 || n_slices == 0)
    {
    return;
    }

  // Get kernel.
  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT2, eT1>(num);

  const uword dest_offset = dest_row_offset + dest_col_offset * dest_M_n_rows + dest_slice_offset * dest_M_n_rows * dest_M_n_cols;
  const uword src_offset  =  src_row_offset +  src_col_offset * src_M_n_rows  +  src_slice_offset * src_M_n_rows * src_M_n_cols;

  typedef typename cuda_type<eT1>::type ceT1;
  typedef typename cuda_type<eT2>::type ceT2;

  const ceT1* src_ptr  =  src.cuda_mem_ptr + src_offset;
  const ceT2* dest_ptr = dest.cuda_mem_ptr + dest_offset;

  const ceT1 cuda_aux_val_pre = to_cuda_type(aux_val_pre);
  const ceT2 cuda_aux_val_post = to_cuda_type(aux_val_post);

  const void* args[] = {
      &dest_ptr,
      &src_ptr,
      &cuda_aux_val_pre,
      &cuda_aux_val_post,
      (uword*) &n_rows,
      (uword*) &n_cols,
      (uword*) &n_slices,
      (uword*) &dest_M_n_rows,
      (uword*) &dest_M_n_cols,
      (uword*) &src_M_n_rows,
      (uword*) &src_M_n_cols };

  const kernel_dims dims = three_dimensional_grid_dims(n_rows, n_cols, n_slices);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL, // shared mem and stream
      (void**) args, // arguments
      0);

  coot_check_cuda_error(result, "coot::cuda::eop_scalar(): cuLaunchKernel() failed");
  }



template<typename eT1, typename eT2>
inline
void
eop_scalar_subview_elem1(const twoway_kernel_id::enum_id num,
                         dev_mem_t<eT2> dest,
                         const dev_mem_t<uword> dest_locs,
                         const dev_mem_t<eT1> src,
                         const dev_mem_t<uword> src_locs,
                         const eT1 aux_val_pre,
                         const eT2 aux_val_post,
                         const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (n_elem == 0)
    {
    return;
    }

  // Get kernel.
  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT2, eT1>(num);

  typedef typename cuda_type<eT1>::type ceT1;
  typedef typename cuda_type<eT2>::type ceT2;

  const ceT1 cuda_aux_val_pre = to_cuda_type(aux_val_pre);
  const ceT2 cuda_aux_val_post = to_cuda_type(aux_val_post);

  const void* args[] = {
      &dest.cuda_mem_ptr,
      &dest_locs.cuda_mem_ptr,
      &src.cuda_mem_ptr,
      &src_locs.cuda_mem_ptr,
      &cuda_aux_val_pre,
      &cuda_aux_val_post,
      (uword*) &n_elem };

  const kernel_dims dims = one_dimensional_grid_dims(n_elem);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL, // shared mem and stream
      (void**) args, // arguments
      0);

  coot_check_cuda_error(result, "coot::cuda::eop_scalar_subview_elem1(): cuLaunchKernel() failed");
  }



template<typename eT1, typename eT2>
inline
void
eop_scalar_subview_elem2(const twoway_kernel_id::enum_id num,
                         dev_mem_t<eT2> dest,
                         const dev_mem_t<uword> dest_row_locs,
                         const dev_mem_t<uword> dest_col_locs,
                         const dev_mem_t<eT1> src,
                         const dev_mem_t<uword> src_row_locs,
                         const dev_mem_t<uword> src_col_locs,
                         const eT1 aux_val_pre,
                         const eT2 aux_val_post,
                         const uword n_row_locs,
                         const uword n_col_locs,
                         const uword dest_n_rows,
                         const uword src_n_rows)
  {
  coot_extra_debug_sigprint();

  if (n_row_locs == 0 || n_col_locs == 0)
    {
    return;
    }

  // Get kernel.
  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT2, eT1>(num);

  typedef typename cuda_type<eT1>::type ceT1;
  typedef typename cuda_type<eT2>::type ceT2;

  const ceT1 cuda_aux_val_pre = to_cuda_type(aux_val_pre);
  const ceT2 cuda_aux_val_post = to_cuda_type(aux_val_post);

  const void* args[] = {
      &dest.cuda_mem_ptr,
      &dest_row_locs.cuda_mem_ptr,
      &dest_col_locs.cuda_mem_ptr,
      &src.cuda_mem_ptr,
      &src_row_locs.cuda_mem_ptr,
      &src_col_locs.cuda_mem_ptr,
      &cuda_aux_val_pre,
      &cuda_aux_val_post,
      (uword* )&n_row_locs,
      (uword* )&n_col_locs,
      (uword* )&dest_n_rows,
      (uword* )&src_n_rows };

  const kernel_dims dims = two_dimensional_grid_dims(n_row_locs, n_col_locs);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL, // shared mem and stream
      (void**) args, // arguments
      0);

  coot_check_cuda_error(result, "coot::cuda::eop_scalar_subview_elem2(): cuLaunchKernel() failed");
  }



/**
 * Run a CUDA elementwise kernel that performs an operation on two matrices.
 */
template<typename eT1, typename eT2, typename eT3>
inline
void
eop_mat(const threeway_kernel_id::enum_id num,
        dev_mem_t<eT3> dest,
        const dev_mem_t<eT1> src_A,
        const dev_mem_t<eT2> src_B,
        // logical size of source and destination
        const uword n_rows,
        const uword n_cols,
        // submatrix destination offsets (set to 0, 0, and n_rows if not a subview)
        const uword dest_row_offset,
        const uword dest_col_offset,
        const uword dest_M_n_rows,
        // submatrix source offsets (set to 0, 0, and n_rows if not a subview)
        const uword src_A_row_offset,
        const uword src_A_col_offset,
        const uword src_A_M_n_rows,
        const uword src_B_row_offset,
        const uword src_B_col_offset,
        const uword src_B_M_n_rows)
  {
  coot_extra_debug_sigprint();

  if (n_rows == 0 || n_cols == 0)
    {
    return;
    }

  // Get kernel.
  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT3, eT2, eT1>(num);

  const uword src_A_offset = src_A_row_offset + src_A_col_offset * src_A_M_n_rows;
  const uword src_B_offset = src_B_row_offset + src_B_col_offset * src_B_M_n_rows;
  const uword dest_offset  =  dest_row_offset +  dest_col_offset * dest_M_n_rows;

  typedef typename cuda_type<eT1>::type ceT1;
  typedef typename cuda_type<eT2>::type ceT2;
  typedef typename cuda_type<eT3>::type ceT3;

  const ceT1* src_A_ptr = src_A.cuda_mem_ptr + src_A_offset;
  const ceT2* src_B_ptr = src_B.cuda_mem_ptr + src_B_offset;
  const ceT3* dest_ptr  =  dest.cuda_mem_ptr + dest_offset;

  const void* args[] = {
      &dest_ptr,
      &src_A_ptr,
      &src_B_ptr,
      (uword*) &n_rows,
      (uword*) &n_cols,
      (uword*) &dest_M_n_rows,
      (uword*) &src_A_M_n_rows,
      (uword*) &src_B_M_n_rows };

  const kernel_dims dims = two_dimensional_grid_dims(n_rows, n_cols);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL, // shared mem and stream
      (void**) args, // arguments
      0);

  coot_check_cuda_error( result, "coot::cuda::eop_mat(): cuLaunchKernel() failed" );
  }



/**
 * Run a CUDA elementwise kernel that performs an operation on two subview_elem1s.
 */
template<typename eT1, typename eT2>
inline
void
eop_subview_elem1(const twoway_kernel_id::enum_id num,
                  dev_mem_t<eT2> dest,
                  const dev_mem_t<uword> dest_locs,
                  const dev_mem_t<eT1> src,
                  const dev_mem_t<uword> src_locs,
                  const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (n_elem == 0)
    {
    return;
    }

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT2, eT1>(num);

  const void* args[] = {
      &(dest.cuda_mem_ptr),
      &(dest_locs.cuda_mem_ptr),
      &(src.cuda_mem_ptr),
      &(src_locs.cuda_mem_ptr),
      (uword*) &n_elem };

  const kernel_dims dims = one_dimensional_grid_dims(n_elem);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL, // shared mem and stream
      (void**) args, // arguments
      0);

  coot_check_cuda_error( result, "coot::cuda::eop_subview_elem1(): cuLaunchKernel() failed" );
  }



/**
 * Run a CUDA elementwise kernel on a matrix into a subview_elem1.
 */
template<typename eT1, typename eT2>
inline
void
eop_subview_elem1_array(const twoway_kernel_id::enum_id num,
                        dev_mem_t<eT2> dest,
                        const dev_mem_t<uword> dest_locs,
                        const dev_mem_t<eT1> src,
                        const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (n_elem == 0)
    {
    return;
    }

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT2, eT1>(num);

  const void* args[] = {
      &(dest.cuda_mem_ptr),
      &(dest_locs.cuda_mem_ptr),
      &(src.cuda_mem_ptr),
      (uword*) &n_elem };

  const kernel_dims dims = one_dimensional_grid_dims(n_elem);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL, // shared mem and stream
      (void**) args, // arguments
      0);

  coot_check_cuda_error( result, "coot::cuda::eop_subview_elem1_array(): cuLaunchKernel() failed" );
  }



/**
 * Run a CUDA elementwise kernel that performs an operation on two subview_elem1s.
 */
template<typename eT1, typename eT2>
inline
void
eop_subview_elem2(const twoway_kernel_id::enum_id num,
                  dev_mem_t<eT2> dest,
                  const dev_mem_t<uword> dest_row_locs,
                  const dev_mem_t<uword> dest_col_locs,
                  const dev_mem_t<eT1> src,
                  const dev_mem_t<uword> src_row_locs,
                  const dev_mem_t<uword> src_col_locs,
                  const uword n_rows,
                  const uword n_cols,
                  const uword dest_n_rows,
                  const uword src_n_rows)
  {
  coot_extra_debug_sigprint();

  if (n_rows == 0 || n_cols == 0)
    {
    return;
    }

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT2, eT1>(num);

  const void* args[] = {
      &(dest.cuda_mem_ptr),
      &(dest_row_locs.cuda_mem_ptr),
      &(dest_col_locs.cuda_mem_ptr),
      &(src.cuda_mem_ptr),
      &(src_row_locs.cuda_mem_ptr),
      &(src_col_locs.cuda_mem_ptr),
      (uword*) &n_rows,
      (uword*) &n_cols,
      (uword*) &dest_n_rows,
      (uword*) &src_n_rows };

  const kernel_dims dims = two_dimensional_grid_dims(n_rows, n_cols);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL, // shared mem and stream
      (void**) args, // arguments
      0);

  coot_check_cuda_error( result, "coot::cuda::eop_subview_elem2(): cuLaunchKernel() failed" );
  }



/**
 * Run a CUDA elementwise kernel on a matrix into a subview_elem1.
 */
template<typename eT1, typename eT2>
inline
void
eop_subview_elem2_array(const twoway_kernel_id::enum_id num,
                        dev_mem_t<eT2> dest,
                        const dev_mem_t<uword> dest_row_locs,
                        const dev_mem_t<uword> dest_col_locs,
                        const dev_mem_t<eT1> src,
                        const uword n_rows,
                        const uword n_cols,
                        const uword dest_n_rows)
  {
  coot_extra_debug_sigprint();

  if (n_rows == 0 || n_cols == 0)
    {
    return;
    }

  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT2, eT1>(num);

  const void* args[] = {
      &(dest.cuda_mem_ptr),
      &(dest_row_locs.cuda_mem_ptr),
      &(dest_col_locs.cuda_mem_ptr),
      &(src.cuda_mem_ptr),
      (uword*) &n_rows,
      (uword*) &n_cols,
      (uword*) &dest_n_rows };

  const kernel_dims dims = two_dimensional_grid_dims(n_rows, n_cols);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL, // shared mem and stream
      (void**) args, // arguments
      0);

  coot_check_cuda_error( result, "coot::cuda::eop_subview_elem2_array(): cuLaunchKernel() failed" );
  }



/**
 * Run a CUDA elementwise kernel that performs an operation on two cubes.
 */
template<typename eT1, typename eT2>
inline
void
eop_cube(const twoway_kernel_id::enum_id num,
         dev_mem_t<eT2> dest,
         const dev_mem_t<eT2> src_A,
         const dev_mem_t<eT1> src_B,
         // logical size of source and destination
         const uword n_rows,
         const uword n_cols,
         const uword n_slices,
         // subcube destination offsets (set to 0, 0, 0, n_rows, and n_cols if not a subview)
         const uword dest_row_offset,
         const uword dest_col_offset,
         const uword dest_slice_offset,
         const uword dest_M_n_rows,
         const uword dest_M_n_cols,
         // subcube source offsets (set to 0, 0, 0, n_rows, and n_cols if not a subview)
         const uword src_A_row_offset,
         const uword src_A_col_offset,
         const uword src_A_slice_offset,
         const uword src_A_M_n_rows,
         const uword src_A_M_n_cols,
         const uword src_B_row_offset,
         const uword src_B_col_offset,
         const uword src_B_slice_offset,
         const uword src_B_M_n_rows,
         const uword src_B_M_n_cols)
  {
  coot_extra_debug_sigprint();

  if (n_rows == 0 || n_cols == 0 || n_slices == 0)
    {
    return;
    }

  // Get kernel.
  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT2, eT1>(num);

  const uword src_A_offset = src_A_row_offset + src_A_col_offset * src_A_M_n_rows + src_A_slice_offset * src_A_M_n_rows * src_A_M_n_cols;
  const uword src_B_offset = src_B_row_offset + src_B_col_offset * src_B_M_n_rows + src_B_slice_offset * src_B_M_n_rows * src_B_M_n_cols;
  const uword dest_offset  =  dest_row_offset +  dest_col_offset * dest_M_n_rows  +  dest_slice_offset * dest_M_n_rows * dest_M_n_cols;

  typedef typename cuda_type<eT1>::type ceT1;
  typedef typename cuda_type<eT2>::type ceT2;

  const ceT2* src_A_ptr = src_A.cuda_mem_ptr + src_A_offset;
  const ceT1* src_B_ptr = src_B.cuda_mem_ptr + src_B_offset;
  const ceT2* dest_ptr  =  dest.cuda_mem_ptr + dest_offset;

  const void* args[] = {
      &dest_ptr,
      &src_A_ptr,
      &src_B_ptr,
      (uword*) &n_rows,
      (uword*) &n_cols,
      (uword*) &n_slices,
      (uword*) &dest_M_n_rows,
      (uword*) &dest_M_n_cols,
      (uword*) &src_A_M_n_rows,
      (uword*) &src_A_M_n_cols,
      (uword*) &src_B_M_n_rows,
      (uword*) &src_B_M_n_cols };

  const kernel_dims dims = three_dimensional_grid_dims(n_rows, n_cols, n_slices);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL, // shared mem and stream
      (void**) args, // arguments
      0);

  coot_check_cuda_error( result, "coot::cuda::eop_cube(): cuLaunchKernel() failed" );
  }



/**
 * Run a CUDA elementwise kernel that performs an operation on two cubes.
 */
template<typename eT1, typename eT2, typename eT3>
inline
void
eop_cube(const threeway_kernel_id::enum_id num,
         dev_mem_t<eT3> dest,
         const dev_mem_t<eT1> src_A,
         const dev_mem_t<eT2> src_B,
         // logical size of source and destination
         const uword n_rows,
         const uword n_cols,
         const uword n_slices,
         // subcube destination offsets (set to 0, 0, 0, n_rows, and n_cols if not a subview)
         const uword dest_row_offset,
         const uword dest_col_offset,
         const uword dest_slice_offset,
         const uword dest_M_n_rows,
         const uword dest_M_n_cols,
         // subcube source offsets (set to 0, 0, 0, n_rows, and n_cols if not a subview)
         const uword src_A_row_offset,
         const uword src_A_col_offset,
         const uword src_A_slice_offset,
         const uword src_A_M_n_rows,
         const uword src_A_M_n_cols,
         const uword src_B_row_offset,
         const uword src_B_col_offset,
         const uword src_B_slice_offset,
         const uword src_B_M_n_rows,
         const uword src_B_M_n_cols)
  {
  coot_extra_debug_sigprint();

  // Get kernel.
  CUfunction kernel = get_rt().cuda_rt.get_kernel<eT3, eT2, eT1>(num);

  const uword src_A_offset = src_A_row_offset + src_A_col_offset * src_A_M_n_rows + src_A_slice_offset * src_A_M_n_rows * src_A_M_n_cols;
  const uword src_B_offset = src_B_row_offset + src_B_col_offset * src_B_M_n_rows + src_B_slice_offset * src_B_M_n_rows * src_B_M_n_cols;
  const uword dest_offset  =  dest_row_offset +  dest_col_offset * dest_M_n_rows  +  dest_slice_offset * dest_M_n_rows * dest_M_n_cols;

  typedef typename cuda_type<eT1>::type ceT1;
  typedef typename cuda_type<eT2>::type ceT2;
  typedef typename cuda_type<eT3>::type ceT3;

  const ceT1* src_A_ptr = src_A.cuda_mem_ptr + src_A_offset;
  const ceT2* src_B_ptr = src_B.cuda_mem_ptr + src_B_offset;
  const ceT3* dest_ptr  =  dest.cuda_mem_ptr + dest_offset;

  const void* args[] = {
      &dest_ptr,
      &src_A_ptr,
      &src_B_ptr,
      (uword*) &n_rows,
      (uword*) &n_cols,
      (uword*) &n_slices,
      (uword*) &dest_M_n_rows,
      (uword*) &dest_M_n_cols,
      (uword*) &src_A_M_n_rows,
      (uword*) &src_A_M_n_cols,
      (uword*) &src_B_M_n_rows,
      (uword*) &src_B_M_n_cols };

  const kernel_dims dims = three_dimensional_grid_dims(n_rows, n_cols, n_slices);

  CUresult result = coot_wrapper(cuLaunchKernel)(
      kernel,
      dims.d[0], dims.d[1], dims.d[2],
      dims.d[3], dims.d[4], dims.d[5],
      0, NULL, // shared mem and stream
      (void**) args, // arguments
      0);

  coot_check_cuda_error( result, "coot::cuda::eop_cube(): cuLaunchKernel() failed" );
  }
