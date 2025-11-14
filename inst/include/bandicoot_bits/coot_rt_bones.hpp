// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2017-2023 Ryan Curtin (https://www.ratml.org)
// Copyright 2017      Conrad Sanderson (https://conradsanderson.id.au)
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



enum coot_backend_t
  {
  CL_BACKEND = 0,
  CUDA_BACKEND
  };

// TODO: if this is placed into a run-time library and executed there, what happens when two programs use the run-time library at the same time?
class coot_rt_t
  {
  public:

  coot_backend_t backend;
  bool initialised;

  #if defined(COOT_USE_OPENCL)
  opencl::runtime_t cl_rt;
  #endif

  #if defined(COOT_USE_CUDA)
  cuda::runtime_t cuda_rt;
  #endif

  inline ~coot_rt_t();
  inline  coot_rt_t();

  inline bool init(const bool print_info = false);
  inline bool init(const char*       filename, const bool print_info = false);
  inline bool init(const std::string filename, const bool print_info = false);
  inline bool init(const uword wanted_platform, const uword wanted_device, const bool print_info = false);

                   coot_rt_t(const coot_rt_t&) = delete;
  coot_rt_t&       operator=(const coot_rt_t&) = delete;

  /**
   * all of the functions below here are redirected to the current backend that is in use
   */

  template<typename eT>
  static inline dev_mem_t<eT> acquire_memory(const uword n_elem);

  template<typename eT>
  static inline void release_memory(dev_mem_t<eT> dev_mem);

  template<typename eT>
  static inline bool is_supported_type();

  static inline void set_rng_seed(const u64 seed);

  /**
   * Copy one matrix to another matrix.
   * The offsets and M_n_rows are meant to allow the destination to be a subview of a larger matrix.
   */
  template<typename eT2, typename eT1>
  static inline void copy_mat(dev_mem_t<eT2> dest,
                              const dev_mem_t<eT1> src,
                              // logical size of matrix
                              const uword n_rows,
                              const uword n_cols,
                              // offsets for subviews
                              const uword dest_row_offset,
                              const uword dest_col_offset,
                              const uword dest_M_n_rows,
                              const uword src_row_offset,
                              const uword src_col_offset,
                              const uword src_M_n_rows);

  /**
   * Copy one cube to another cube.
   */
  template<typename eT2, typename eT1>
  static inline void copy_cube(dev_mem_t<eT2> dest,
                               const dev_mem_t<eT1> src,
                               // logical size of cube
                               const uword n_rows,
                               const uword n_cols,
                               const uword n_slices,
                               // offsets for subviews
                               const uword dest_row_offset,
                               const uword dest_col_offset,
                               const uword dest_slice_offset,
                               const uword dest_M_n_rows,
                               const uword dest_M_n_cols,
                               const uword src_row_offset,
                               const uword src_col_offset,
                               const uword src_slice_offset,
                               const uword src_M_n_rows,
                               const uword src_M_n_cols);

  template<typename eT>
  static inline void reorder_cols(dev_mem_t<eT> out, const dev_mem_t<eT> mem, const uword n_rows, const dev_mem_t<uword> order, const uword out_n_cols);

  /**
   * Fill a matrix or subview with a scalar value.
   * The offsets and M_n_rows are meant to allow the destination to be a subview of a larger matrix.
   */
  template<typename eT>
  static inline void fill(dev_mem_t<eT> dest,
                          const eT val,
                          const uword n_rows,
                          const uword n_cols,
                          const uword row_offset,
                          const uword col_offset,
                          const uword M_n_rows);

  template<typename eT1, typename eT2>
  static inline void replace(dev_mem_t<eT2> mem_out,
                             const dev_mem_t<eT1> mem_in,
                             const eT1 val_find,
                             const eT1 val_replace,
                             // logical size of matrix
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
                             const uword src_M_n_cols);

  template<typename eT1, typename eT2>
  static inline void htrans(dev_mem_t<eT2> dest, const dev_mem_t<eT1> src, const uword n_rows, const uword n_cols);

  template<typename eT1, typename eT2>
  static inline void strans(dev_mem_t<eT2> dest, const dev_mem_t<eT1> src, const uword n_rows, const uword n_cols);

  template<typename eT>
  static inline void fill_randu(dev_mem_t<eT> dest, const uword n);

  template<typename eT>
  static inline void fill_randn(dev_mem_t<eT> dest, const uword n, const double mu, const double sd);

  template<typename eT>
  static inline void fill_randi(dev_mem_t<eT> dest, const uword n, const int lo, const int hi);

  /**
   * Perform an elementwise scalar operation on a matrix or cube of size `n_rows x n_cols x n_slices`, storing the result in `dest`.
   * The offsets and M_n_rows are meant to allow the source (and destination) to be subviews of a larger matrix or cube.
   */
  template<typename eT1, typename eT2>
  static inline void eop_scalar(const twoway_kernel_id::enum_id num,
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
                                const uword src_M_n_cols);

  /**
   * Perform an elementwise matrix operation on two matrices of size `n_rows` x `n_cols`.
   */
  template<typename eT1, typename eT2, typename eT3>
  static inline void eop_mat(const threeway_kernel_id::enum_id num,
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
                             const uword src_B_M_n_rows);

  /**
   * Perform an elementwise operation on two cubes of size `n_rows` x `n_cols` x `n_slices`.
   */
  template<typename eT1, typename eT2>
  static inline void eop_cube(const twoway_kernel_id::enum_id num,
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
                              const uword src_B_M_n_cols);

  template<typename eT>
  static inline eT prod(const dev_mem_t<eT> mem, const uword n_elem);

  template<typename eT>
  static inline eT max_abs(const dev_mem_t<eT> mem, const uword n_elem);

  template<typename eT1, typename eT2>
  static inline bool all_vec(const dev_mem_t<eT1> mem, const uword n_elem, const eT2 val, const twoway_kernel_id::enum_id num, const twoway_kernel_id::enum_id num_small);

  template<typename eT1, typename eT2>
  static inline void all(dev_mem_t<uword> out_mem, const dev_mem_t<eT1> in_mem, const uword n_rows, const uword n_cols, const eT2 val, const twoway_kernel_id::enum_id num, const bool colwise);

  template<typename eT1, typename eT2>
  static inline bool any_vec(const dev_mem_t<eT1> mem, const uword n_elem, const eT2 val, const twoway_kernel_id::enum_id num, const twoway_kernel_id::enum_id num_small);

  template<typename eT>
  static inline bool any_vec(const dev_mem_t<eT> mem, const uword n_elem, const eT val, const oneway_real_kernel_id::enum_id num, const oneway_real_kernel_id::enum_id num_small);

  template<typename eT1, typename eT2>
  static inline void any(dev_mem_t<uword> out_mem, const dev_mem_t<eT1> in_mem, const uword n_rows, const uword n_cols, const eT2 val, const twoway_kernel_id::enum_id num, const bool colwise);

  template<typename eT1, typename eT2>
  static inline void relational_scalar_op(dev_mem_t<uword> out_mem, const dev_mem_t<eT1> in_mem, const uword n_elem, const eT2 val, const twoway_kernel_id::enum_id num, const std::string& name);

  template<typename eT1>
  static inline void relational_unary_array_op(dev_mem_t<uword> out_mem, const dev_mem_t<eT1> in_mem, const uword n_elem, const oneway_real_kernel_id::enum_id num, const std::string& name);

  template<typename eT1, typename eT2>
  static inline void relational_array_op(dev_mem_t<uword> out_mem, const dev_mem_t<eT1> X_mem, const dev_mem_t<eT2> Y_mem, const uword n_elem, const twoway_kernel_id::enum_id num, const std::string& name);

  template<typename eT>
  static inline std::tuple<bool, std::string> chol(dev_mem_t<eT> out, const uword n_rows);

  template<typename eT>
  static inline std::tuple<bool, std::string> lu(dev_mem_t<eT> L, dev_mem_t<eT> U, dev_mem_t<eT> in, const bool pivoting, dev_mem_t<eT> P, const uword n_rows, const uword n_cols);

  template<typename eT>
  static inline std::tuple<bool, std::string> det(dev_mem_t<eT> A, const uword n_rows, eT& out_val);

  template<typename eT>
  static inline std::tuple<bool, std::string> svd(dev_mem_t<eT> U, dev_mem_t<eT> S, dev_mem_t<eT> V, dev_mem_t<eT> A, const uword n_rows, const uword n_cols, const bool compute_u_vt);

  template<typename eT>
  static inline std::tuple<bool, std::string> eig_sym(dev_mem_t<eT> mem, const uword n_rows, const bool eigenvectors, dev_mem_t<eT> eigenvalues);

  template<typename eT>
  static inline std::tuple<bool, std::string> solve_square_fast(dev_mem_t<eT> A, const bool trans_A, dev_mem_t<eT> B, const uword n_rows, const uword n_cols);

  template<typename eT>
  static inline void copy_from_dev_mem(eT* dest,
                                       const dev_mem_t<eT> src,
                                       const uword n_rows,
                                       const uword n_cols,
                                       const uword src_row_offset,
                                       const uword src_col_offset,
                                       const uword src_M_n_rows);

  template<typename eT>
  static inline void copy_into_dev_mem(dev_mem_t<eT> dest, const eT* src, const uword N);

  template<typename eT1, typename eT2>
  static inline void extract_subview(dev_mem_t<eT2> out, const dev_mem_t<eT1> in, const uword M_n_rows, const uword M_n_cols, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols);

  template<typename eT>
  static inline void eye(dev_mem_t<eT> out, const uword n_rows, const uword n_cols);

  template<typename eT>
  static inline eT get_val(const dev_mem_t<eT> mem, const uword index);

  template<typename eT>
  static inline void set_val(dev_mem_t<eT> mem, const uword index, const eT val);

  template<typename eT> static inline void   val_add_inplace(dev_mem_t<eT> mem, const uword index, const eT val);
  template<typename eT> static inline void val_minus_inplace(dev_mem_t<eT> mem, const uword index, const eT val);
  template<typename eT> static inline void   val_mul_inplace(dev_mem_t<eT> mem, const uword index, const eT val);
  template<typename eT> static inline void   val_div_inplace(dev_mem_t<eT> mem, const uword index, const eT val);

  template<typename eT, const bool do_trans_A, const bool do_trans_B>
  static inline void gemm(dev_mem_t<eT> C_mem,
                          const uword C_n_rows,
                          const uword C_n_cols,
                          const dev_mem_t<eT> A_mem,
                          const uword A_n_rows,
                          const uword A_n_cols,
                          const dev_mem_t<eT> B_mem,
                          const eT alpha,
                          const eT beta,
                          // subview arguments
                          const uword C_row_offset,
                          const uword C_col_offset,
                          const uword C_M_n_rows,
                          const uword A_row_offset,
                          const uword A_col_offset,
                          const uword A_M_n_rows,
                          const uword B_row_offset,
                          const uword B_col_offset,
                          const uword B_M_n_rows);

  template<typename eT, const bool do_trans_A>
  static inline void gemv(dev_mem_t<eT> y_mem,
                          const dev_mem_t<eT> A_mem,
                          const uword A_n_rows,
                          const uword A_n_cols,
                          const dev_mem_t<eT> x_mem,
                          const eT alpha,
                          const eT beta,
                          // subview arguments
                          const uword y_offset,
                          const uword y_mem_incr,
                          const uword A_row_offset,
                          const uword A_col_offset,
                          const uword A_M_n_rows,
                          const uword x_offset,
                          const uword x_mem_incr);

  template<typename eT>
  static inline void mul_diag(dev_mem_t<eT> C_mem,
                              const uword C_n_rows,
                              const uword C_n_cols,
                              const eT alpha,
                              const dev_mem_t<eT> A_mem,
                              const bool A_is_diag,
                              const bool A_trans,
                              const uword A_row_offset,
                              const uword A_col_offset,
                              const uword A_M_n_rows,
                              const dev_mem_t<eT> B_mem,
                              const bool B_is_diag,
                              const bool B_trans,
                              const uword B_row_offset,
                              const uword B_col_offset,
                              const uword B_M_n_rows);

  template<typename eT1, typename eT2>
  static inline void sum(dev_mem_t<eT2> dest,
                         const dev_mem_t<eT1> src,
                         const uword n_rows,
                         const uword n_cols,
                         const uword dim,
                         const bool post_conv_apply,
                         // subview arguments
                         const uword dest_offset,
                         const uword dest_mem_incr,
                         const uword src_row_offset,
                         const uword src_col_offset,
                         const uword src_M_n_rows);

  template<typename eT>
  static inline eT accu(const dev_mem_t<eT> mem, const uword n_elem);

  template<typename eT>
  static inline eT accu_subview(const dev_mem_t<eT> mem, const uword M_n_rows, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols);

  template<typename eT1, typename eT2>
  static inline void min(dev_mem_t<eT2> dest,
                         const dev_mem_t<eT1> src,
                         const uword n_rows,
                         const uword n_cols,
                         const uword dim,
                         const bool post_conv_apply,
                         // subview arguments
                         const uword dest_offset,
                         const uword dest_mem_incr,
                         const uword src_row_offset,
                         const uword src_col_offset,
                         const uword src_M_n_rows);

  template<typename eT1, typename eT2>
  static inline void min_cube_col(dev_mem_t<eT2> dest,
                                  const dev_mem_t<eT1> src,
                                  const uword n_rows,
                                  const uword n_cols,
                                  const uword n_slices,
                                  const bool post_conv_apply);

  template<typename eT>
  static inline eT min_vec(const dev_mem_t<eT> mem, const uword n_elem);

  template<typename eT1, typename eT2>
  static inline void max(dev_mem_t<eT2> dest,
                         const dev_mem_t<eT1> src,
                         const uword n_rows,
                         const uword n_cols,
                         const uword dim,
                         const bool post_conv_apply,
                         // subview arguments
                         const uword dest_offset,
                         const uword dest_mem_incr,
                         const uword src_row_offset,
                         const uword src_col_offset,
                         const uword src_M_n_rows);

  template<typename eT>
  static inline eT max_vec(const dev_mem_t<eT> mem, const uword n_elem);

  template<typename eT1, typename eT2>
  static inline void max_cube_col(dev_mem_t<eT2> dest,
                                  const dev_mem_t<eT1> src,
                                  const uword n_rows,
                                  const uword n_cols,
                                  const uword n_slices,
                                  const bool post_conv_apply);

  template<typename eT>
  static inline void index_min(dev_mem_t<uword> dest,
                               const dev_mem_t<eT> src,
                               const uword n_rows,
                               const uword n_cols,
                               const uword dim,
                               // subview arguments
                               const uword dest_offset,
                               const uword dest_mem_incr,
                               const uword src_row_offset,
                               const uword src_col_offset,
                               const uword src_M_n_rows);

  template<typename eT>
  static inline void index_min_cube_col(dev_mem_t<uword> dest,
                                        const dev_mem_t<eT> src,
                                        const uword n_rows,
                                        const uword n_cols,
                                        const uword n_slices);

  template<typename eT>
  static inline uword index_min_vec(const dev_mem_t<eT> mem, const uword n_elem, eT* min_val = nullptr);

  template<typename eT>
  static inline void index_max(dev_mem_t<uword> dest,
                               const dev_mem_t<eT> src,
                               const uword n_rows,
                               const uword n_cols,
                               const uword dim,
                               // subview arguments
                               const uword dest_offset,
                               const uword dest_mem_incr,
                               const uword src_row_offset,
                               const uword src_col_offset,
                               const uword src_M_n_rows);

  template<typename eT>
  static inline void index_max_cube_col(dev_mem_t<uword> dest,
                                        const dev_mem_t<eT> src,
                                        const uword n_rows,
                                        const uword n_cols,
                                        const uword n_slices);

  template<typename eT>
  static inline uword index_max_vec(const dev_mem_t<eT> mem, const uword n_elem, eT* max_val = nullptr);

  template<typename eT>
  static inline eT trace(const dev_mem_t<eT> mem, const uword n_rows, const uword n_cols);

  template<typename eT1, typename eT2>
  static inline typename promote_type<eT1, eT2>::result dot(const dev_mem_t<eT1> mem1, const dev_mem_t<eT2> mem2, const uword n_elem);

  template<typename eT1, typename eT2>
  static inline void broadcast_op(const twoway_kernel_id::enum_id num,
                                  dev_mem_t<eT2> dest,
                                  const dev_mem_t<eT2> dest_in,
                                  const dev_mem_t<eT1> src,
                                  const uword src_n_rows,
                                  const uword src_n_cols,
                                  const uword copies_per_row,
                                  const uword copies_per_col,
                                  // subview arguments
                                  const uword dest_row_offset,
                                  const uword dest_col_offset,
                                  const uword dest_M_n_rows,
                                  const uword dest_in_row_offset,
                                  const uword dest_in_col_offset,
                                  const uword dest_in_M_n_rows,
                                  const uword src_row_offset,
                                  const uword src_col_offset,
                                  const uword src_M_n_rows);

  template<typename eT1, typename eT2>
  static inline void broadcast_subset_op(const twoway_kernel_id::enum_id num,
                                         dev_mem_t<eT2> dest,
                                         const dev_mem_t<eT2> dest_in,
                                         const dev_mem_t<eT1> src,
                                         const dev_mem_t<uword> indices,
                                         const uword mode, // 0 => src_n_rows == indices.n_elem
                                         const uword src_n_rows,
                                         const uword src_n_cols,
                                         const uword copies_per_row,
                                         const uword copies_per_col,
                                         // subview arguments
                                         const uword dest_row_offset,
                                         const uword dest_col_offset,
                                         const uword dest_M_n_rows,
                                         const uword dest_in_row_offset,
                                         const uword dest_in_col_offset,
                                         const uword dest_in_M_n_rows,
                                         const uword src_row_offset,
                                         const uword src_col_offset,
                                         const uword src_M_n_rows,
                                         const uword indices_offset,
                                         const uword indices_incr);

  template<typename eT>
  static inline void linspace(dev_mem_t<eT> mem,
                              const uword mem_incr,
                              const eT start,
                              const eT end,
                              const uword num);

  template<typename eT>
  static inline void logspace(dev_mem_t<eT> mem,
                              const uword mem_incr,
                              const eT start,
                              const eT end,
                              const uword num);

  template<typename eT>
  static inline void regspace(dev_mem_t<eT> mem,
                              const uword mem_incr,
                              const eT start,
                              const eT delta,
                              const eT end,
                              const uword num,
                              const bool desc);

  template<typename eT1, typename eT2>
  static inline void clamp(dev_mem_t<eT2> dest,
                           const dev_mem_t<eT1> src,
                           const eT1 min_val,
                           const eT1 max_val,
                           const uword n_rows,
                           const uword n_cols,
                           const uword dest_row_offset,
                           const uword dest_col_offset,
                           const uword dest_M_n_rows,
                           const uword src_row_offset,
                           const uword src_col_offset,
                           const uword src_M_n_rows);

  template<typename eT>
  static inline eT vec_norm_1(dev_mem_t<eT> mem, const uword n_elem);

  template<typename eT>
  static inline eT vec_norm_2(dev_mem_t<eT> mem, const uword n_elem);

  template<typename eT>
  static inline eT vec_norm_k(dev_mem_t<eT> mem, const uword n_elem, const uword k);

  template<typename eT>
  static inline eT vec_norm_min(dev_mem_t<eT> mem, const uword n_elem);

  template<typename eT1, typename eT2>
  static inline void mean(dev_mem_t<eT2> dest,
                          const dev_mem_t<eT1> src,
                          const uword n_rows,
                          const uword n_cols,
                          const uword dim,
                          const bool post_conv_apply,
                          // subview arguments
                          const uword dest_offset,
                          const uword dest_mem_incr,
                          const uword src_row_offset,
                          const uword src_col_offset,
                          const uword src_M_n_rows);

  template<typename eT1, typename eT2>
  static inline void median(dev_mem_t<eT2> dest,
                            dev_mem_t<eT1> src,
                            const uword n_rows,
                            const uword n_cols,
                            const uword dim,
                            // subview arguments
                            const uword dest_offset,
                            const uword dest_mem_incr,
                            const uword src_row_offset,
                            const uword src_col_offset,
                            const uword src_M_n_rows);

  template<typename eT>
  static inline eT median_vec(dev_mem_t<eT> mem, const uword n_elem);

  template<typename eT>
  static inline void var(dev_mem_t<eT> dest,
                         const dev_mem_t<eT> src,
                         const dev_mem_t<eT> src_means,
                         const uword n_rows,
                         const uword n_cols,
                         const uword dim,
                         const uword norm_type,
                         // subview arguments
                         const uword dest_offset,
                         const uword dest_mem_incr,
                         const uword src_row_offset,
                         const uword src_col_offset,
                         const uword src_M_n_rows,
                         const uword src_means_offset,
                         const uword src_means_mem_incr);

  template<typename eT>
  static inline eT var_vec(const dev_mem_t<eT> mem, const eT mean, const uword n_elem, const uword norm_type);

  template<typename eT>
  static inline eT var_vec_subview(const dev_mem_t<eT> mem, const eT mean, const uword M_n_rows, const uword M_n_cols, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const uword norm_type);

  template<typename eT1, typename eT2, typename eT3, typename eT4, typename eT5>
  static inline void join_cols(dev_mem_t<eT5> out,
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
                               const uword D_M_n_rows);

  template<typename eT1, typename eT2, typename eT3, typename eT4, typename eT5>
  static inline void join_rows(dev_mem_t<eT5> out,
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
                               const uword D_M_n_rows);

  template<typename eT>
  static inline void sort(dev_mem_t<eT> mem,
                          const uword n_rows,
                          const uword n_cols,
                          const uword sort_type,
                          const uword dim,
                          // subview arguments
                          const uword row_offset,
                          const uword col_offset,
                          const uword M_n_rows);

  template<typename eT>
  static inline void sort_vec(dev_mem_t<eT> mem, const uword n_elem, const uword sort_type);

  template<typename eT>
  static inline void sort_index_vec(dev_mem_t<uword> out, dev_mem_t<eT> mem, const uword n_elem, const uword sort_type, const uword stable_sort);

  template<typename eT>
  static inline void find(dev_mem_t<uword>& out, uword& out_len, const dev_mem_t<eT> A, const uword n_elem, const uword k, const uword find_type);

  template<typename eT1, typename eT2>
  static inline void symmat(dev_mem_t<eT2> out, const dev_mem_t<eT1> in, const uword size, const uword lower);

  template<typename eT1, typename eT2>
  static inline void cross(dev_mem_t<eT2> out, const dev_mem_t<eT1> A, const dev_mem_t<eT1> B);

  template<typename eT>
  static inline void rotate_180(dev_mem_t<eT> out, const dev_mem_t<eT> in, const uword n_rows, const uword n_cols);

  template<typename eT>
  static inline bool approx_equal(const dev_mem_t<eT> A,
                                  const uword A_row_offset,
                                  const uword A_col_offset,
                                  const uword A_M_n_rows,
                                  const dev_mem_t<eT> B,
                                  const uword B_row_offset,
                                  const uword B_col_offset,
                                  const uword B_M_n_rows,
                                  const uword n_rows,
                                  const uword n_cols,
                                  const char sig,
                                  const eT abs_tol,
                                  const eT rel_tol);

  template<typename eT>
  static inline bool approx_equal_cube(const dev_mem_t<eT> A,
                                       const uword A_row_offset,
                                       const uword A_col_offset,
                                       const uword A_slice_offset,
                                       const uword A_M_n_rows,
                                       const uword A_M_n_cols,
                                       const dev_mem_t<eT> B,
                                       const uword B_row_offset,
                                       const uword B_col_offset,
                                       const uword B_slice_offset,
                                       const uword B_M_n_rows,
                                       const uword B_M_n_cols,
                                       const uword n_rows,
                                       const uword n_cols,
                                       const uword n_slices,
                                       const char sig,
                                       const eT abs_tol,
                                       const eT rel_tol);

  template<typename eT>
  static inline void shuffle(dev_mem_t<eT> out,
                             const uword out_row_offset,
                             const uword out_col_offset,
                             const uword out_M_n_rows,
                             const dev_mem_t<eT> in,
                             const uword in_row_offset,
                             const uword in_col_offset,
                             const uword in_M_n_rows,
                             const uword n_rows,
                             const uword n_cols,
                             const uword dim);

  template<typename eT1, typename eT2>
  static inline void extract_cx(dev_mem_t<eT1> out_mem,
                                const uword out_row_offset,
                                const uword out_col_offset,
                                const uword out_M_n_rows,
                                const dev_mem_t<eT2> in_mem,
                                const uword in_row_offset,
                                const uword in_col_offset,
                                const uword in_M_n_rows,
                                const uword n_rows,
                                const uword n_cols,
                                const bool imag);

  static inline void synchronise();

  // RC-TODO: unified interface for some other operations?
  };



// Store coot_rt_t as a singleton.
inline coot_rt_t& get_rt_internal()
  {
  static coot_rt_t rt;
  return rt;
  }



inline coot_rt_t& get_rt()
  {
  coot_rt_t& rt = get_rt_internal();
  if (!rt.initialised)
    {
    rt.init();
    }

  return rt;
  }



inline bool init_rt(const bool print_info)
  {
  coot_rt_t& rt = get_rt_internal();
  return rt.init(print_info);
  }



inline bool init_rt(const coot_backend_t backend, const bool print_info, const uword platform_id, const uword device_id)
  {
  coot_rt_t& rt = get_rt_internal();
  rt.backend = backend;
  return rt.init(print_info, platform_id, device_id);
  }
