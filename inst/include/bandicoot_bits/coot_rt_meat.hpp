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



inline
coot_rt_t::~coot_rt_t()
  {
  coot_extra_debug_sigprint_this(this);
  }



inline
coot_rt_t::coot_rt_t()
  {
  coot_extra_debug_sigprint_this(this);
  backend = COOT_DEFAULT_BACKEND;
  initialised = false;
  }



inline
bool
coot_rt_t::init(const bool print_info)
  {
  coot_extra_debug_sigprint();

  // TODO: investigate reading a config file by default; if a config file exist, use the specifed platform and device within the config file
  // TODO: config file may exist in several places: (1) globally accessible, such as /etc/bandicoot_config, or locally, such as ~/.config/bandicoot_config
  // TODO: use case: user puts code on a server which has a different configuration than the user's workstation

  // prevent recursive initialisation
  initialised = true;

  if (backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    initialised = cl_rt.init(false, 0, 0, print_info);
    return initialised;
    #else
    initialised = false;
    coot_stop_runtime_error("coot_rt::init(): OpenCL backend not enabled");
    #endif
    }
  else if (backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    initialised = cuda_rt.init(false, 0, 0, print_info);
    return initialised;
    #else
    initialised = false;
    coot_stop_runtime_error("coot_rt::init(): CUDA backend not enabled");
    #endif
    }
  else
    {
    initialised = false;
    coot_stop_runtime_error("coot_rt::init(): unknown backend");
    }

  initialised = false;
  return false;
  }



inline
bool
coot_rt_t::init(const char* filename, const bool print_info)
  {
  coot_extra_debug_sigprint();

  return coot_rt_t::init(std::string(filename), print_info);
  }



inline
bool
coot_rt_t::init(const std::string filename, const bool print_info)
  {
  coot_extra_debug_sigprint();

  // TODO: handling of config files is currently rudimentary

  if(print_info)  {std::cout << "coot::opencl::runtime_t::init(): reading " << filename << std::endl; }

  uword wanted_platform = 0;
  uword wanted_device   = 0;

  std::ifstream f;
  f.open(filename.c_str(), std::fstream::binary);

  if(f.is_open() == false)
    {
    std::cout << "coot::opencl::runtime_t::init(): couldn't read " << filename << std::endl;
    return false;
    }

  f >> wanted_platform;
  f >> wanted_device;

  if(f.good() == false)
    {
    wanted_platform = 0;
    wanted_device   = 0;

    std::cout << "coot::opencl::runtime_t::init(): couldn't read " << filename << std::endl;
    return false;
    }
  else
    {
    if(print_info)  { std::cout << "coot::opencl::runtime::init(): wanted_platform = " << wanted_platform << "   wanted_device = " << wanted_device << std::endl; }
    }

  // prevent recursive initialisation
  initialised = true;

  if (backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    initialised = cl_rt.init(true, wanted_platform, wanted_device, print_info);
    return initialised;
    #else
    initialised = false;
    coot_stop_runtime_error("coot_rt::init(): OpenCL backend not enabled");
    #endif
    }
  else if (backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    initialised = cuda_rt.init(true, wanted_platform, wanted_device, print_info);
    return initialised;
    #else
    initialised = false;
    coot_stop_runtime_error("coot_rt::init(): CUDA backend not enabled");
    #endif
    }
  else
    {
    initialised = false;
    coot_stop_runtime_error("coot_rt::init(): unknown backend");
    }

  initialised = false;
  return false;
  }



inline
bool
coot_rt_t::init(const uword wanted_platform, const uword wanted_device, const bool print_info)
  {
  // prevent recursive initialisation
  initialised = true;

  if (backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    initialised = cl_rt.init(true, wanted_platform, wanted_device, print_info);
    return initialised;
    #else
    initialised = false;
    coot_stop_runtime_error("coot_rt::init(): OpenCL backend not enabled");

    #endif
    }
  else if (backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    initialised = cuda_rt.init(true, wanted_platform, wanted_device, print_info);
    return initialised;
    #else
    initialised = false;
    coot_stop_runtime_error("coot_rt::init(): CUDA backend not enabled");
    #endif
    }
  else
    {
    initialised = false;
    coot_stop_runtime_error("coot_rt::init(): unknown backend");
    }

  initialised = false;
  return false;
  }



template<typename eT>
inline
dev_mem_t<eT>
coot_rt_t::acquire_memory(const uword n_elem)
  {
  coot_extra_debug_sigprint();

//  coot_check_runtime_error( (valid == false), "coot_rt::acquire_memory(): runtime not valid" );

  if(n_elem == 0)  { return dev_mem_t<eT>({ NULL }); }

  coot_debug_check
   (
   ( size_t(n_elem) > (std::numeric_limits<size_t>::max() / sizeof(eT)) ),
   "coot_rt::acquire_memory(): requested size is too large"
   );

  dev_mem_t<eT> result;

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    result.cl_mem_ptr = get_rt().cl_rt.acquire_memory<eT>(n_elem);
    #else
    coot_stop_runtime_error("coot_rt::acquire_memory(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    result.cuda_mem_ptr = get_rt().cuda_rt.acquire_memory<eT>(n_elem);
    #else
    coot_stop_runtime_error("coot_rt::acquire_memory(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::acquire_memory(): unknown backend");
    }

  return result;
  }



template<typename eT>
inline
void
coot_rt_t::release_memory(dev_mem_t<eT> dev_mem)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    get_rt().cl_rt.release_memory(dev_mem.cl_mem_ptr);
    #else
    coot_stop_runtime_error("coot_rt::release_memory(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    get_rt().cuda_rt.release_memory(dev_mem.cuda_mem_ptr);
    #else
    coot_stop_runtime_error("coot_rt::release_memory(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::release_memory(): unknown backend");
    }
  }



template<typename eT>
inline
bool
coot_rt_t::is_supported_type()
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #ifdef COOT_USE_OPENCL
    return get_rt().cl_rt.is_supported_type<eT>();
    #else
    coot_stop_runtime_error("coot_rt::is_supported_type(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #ifdef COOT_USE_CUDA
    return get_rt().cuda_rt.is_supported_type<eT>();
    #else
    coot_stop_runtime_error("coot_rt::is_supported_type(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::is_supported_type(): unknown backend");
    }

  return false;
  }



inline
void
coot_rt_t::set_rng_seed(const u64 seed)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #ifdef COOT_USE_OPENCL
    get_rt().cl_rt.set_rng_seed(seed);
    #else
    coot_stop_runtime_error("coot_rt::set_rng_seed(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #ifdef COOT_USE_CUDA
    get_rt().cuda_rt.set_rng_seed(seed);
    #else
    coot_stop_runtime_error("coot_rt::set_rng_seed(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::set_rng_seed(): unknown backend");
    }
  }



inline
void
coot_rt_t::synchronise()
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    get_rt().cl_rt.synchronise();
    #else
    coot_stop_runtime_error("coot_rt::synchronise(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    get_rt().cuda_rt.synchronise();
    #else
    coot_stop_runtime_error("coot_rt::synchronise(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::synchronise(): unknown backend");
    }
  }



template<typename eT2, typename eT1>
inline
void
coot_rt_t::copy_mat(dev_mem_t<eT2> dest,
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
                    const uword src_M_n_rows)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::copy_mat(dest, src,
                     n_rows, n_cols,
                     dest_row_offset, dest_col_offset, dest_M_n_rows,
                     src_row_offset, src_col_offset, src_M_n_rows);
    #else
    coot_stop_runtime_error("coot_rt::copy_mat(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::copy_mat(dest, src,
                   n_rows, n_cols,
                   dest_row_offset, dest_col_offset, dest_M_n_rows,
                   src_row_offset, src_col_offset, src_M_n_rows);
    #else
    coot_stop_runtime_error("coot_rt::copy_mat(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::copy_mat(): unknown backend");
    }
  }



template<typename eT2, typename eT1>
inline
void
coot_rt_t::copy_cube(dev_mem_t<eT2> dest,
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
                     const uword src_M_n_cols)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::copy_cube(dest, src,
                      n_rows, n_cols, n_slices,
                      dest_row_offset, dest_col_offset, dest_slice_offset, dest_M_n_rows, dest_M_n_cols,
                      src_row_offset, src_col_offset, src_slice_offset, src_M_n_rows, src_M_n_cols);
    #else
    coot_stop_runtime_error("coot_rt::copy_cube(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::copy_cube(dest, src,
                    n_rows, n_cols, n_slices,
                    dest_row_offset, dest_col_offset, dest_slice_offset, dest_M_n_rows, dest_M_n_cols,
                    src_row_offset, src_col_offset, src_slice_offset, src_M_n_rows, src_M_n_cols);
    #else
    coot_stop_runtime_error("coot_rt::copy_cube(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::copy_cube(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::reorder_cols(dev_mem_t<eT> out, const dev_mem_t<eT> mem, const uword n_rows, const dev_mem_t<uword> order, const uword out_n_cols)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::reorder_cols(out, mem, n_rows, order, out_n_cols);
    #else
    coot_stop_runtime_error("coot_rt::reorder_cols(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::reorder_cols(out, mem, n_rows, order, out_n_cols);
    #else
    coot_stop_runtime_error("coot_rt::reorder_cols(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::reorder_cols(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::fill(dev_mem_t<eT> dest,
                const eT val,
                const uword n_rows,
                const uword n_cols,
                const uword row_offset,
                const uword col_offset,
                const uword M_n_rows)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::fill(dest, val, n_rows, n_cols, row_offset, col_offset, M_n_rows);
    #else
    coot_stop_runtime_error("coot_rt::fill(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::fill(dest, val, n_rows, n_cols, row_offset, col_offset, M_n_rows);
    #else
    coot_stop_runtime_error("coot_rt::fill(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::fill(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::replace(dev_mem_t<eT> mem, const uword n_elem, const eT val_find, const eT val_replace)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::replace(mem, n_elem, val_find, val_replace);
    #else
    coot_stop_runtime_error("coot_rt::replace(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::replace(mem, n_elem, val_find, val_replace);
    #else
    coot_stop_runtime_error("coot_rt::replace(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::replace(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::htrans(dev_mem_t<eT2> dest, const dev_mem_t<eT1> src, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::htrans(dest, src, n_rows, n_cols);
    #else
    coot_stop_runtime_error("coot_rt::htrans(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::htrans(dest, src, n_rows, n_cols);
    #else
    coot_stop_runtime_error("coot_rt::htrans(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::htrans(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::strans(dev_mem_t<eT2> dest, const dev_mem_t<eT1> src, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::strans(dest, src, n_rows, n_cols);
    #else
    coot_stop_runtime_error("coot_rt::strans(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::strans(dest, src, n_rows, n_cols);
    #else
    coot_stop_runtime_error("coot_rt::strans(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::strans(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::fill_randu(dev_mem_t<eT> dest, const uword n)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::fill_randu(dest, n);
    #else
    coot_stop_runtime_error("coot_rt::fill_randu(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::fill_randu(dest, n);
    #else
    coot_stop_runtime_error("coot_rt::fill_randu(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::fill_randu(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::fill_randn(dev_mem_t<eT> dest, const uword n, const double mu, const double sd)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::fill_randn(dest, n, mu, sd);
    #else
    coot_stop_runtime_error("coot_rt::fill_randn(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::fill_randn(dest, n, mu, sd);
    #else
    coot_stop_runtime_error("coot_rt::fill_randn(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::fill_randn(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::fill_randi(dev_mem_t<eT> dest, const uword n, const int lo, const int hi)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::fill_randi(dest, n, lo, hi);
    #else
    coot_stop_runtime_error("coot_rt::fill_randi(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::fill_randi(dest, n, lo, hi);
    #else
    coot_stop_runtime_error("coot_rt::fill_randi(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::fill_randi(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::eop_scalar(const twoway_kernel_id::enum_id num,
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

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::eop_scalar(num, dest, src,
                       aux_val_pre, aux_val_post,
                       n_rows, n_cols, n_slices,
                       dest_row_offset, dest_col_offset, dest_slice_offset, dest_M_n_rows, dest_M_n_cols,
                       src_row_offset, src_col_offset, src_slice_offset, src_M_n_rows, src_M_n_cols);
    #else
    coot_stop_runtime_error("coot_rt::eop_scalar(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::eop_scalar(num, dest, src,
                     aux_val_pre, aux_val_post,
                     n_rows, n_cols, n_slices,
                     dest_row_offset, dest_col_offset, dest_slice_offset, dest_M_n_rows, dest_M_n_cols,
                     src_row_offset, src_col_offset, src_slice_offset, src_M_n_rows, src_M_n_cols);
    #else
    coot_stop_runtime_error("coot_rt::eop_scalar(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::eop_scalar(): unknown backend");
    }
  }



template<typename eT1, typename eT2, typename eT3>
inline
void
coot_rt_t::eop_mat(const threeway_kernel_id::enum_id num,
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

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::eop_mat(num, dest, src_A, src_B,
                    n_rows, n_cols,
                    dest_row_offset, dest_col_offset, dest_M_n_rows,
                    src_A_row_offset, src_A_col_offset, src_A_M_n_rows,
                    src_B_row_offset, src_B_col_offset, src_B_M_n_rows);
    #else
    coot_stop_runtime_error("coot_rt::eop_mat(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::eop_mat(num, dest, src_A, src_B,
                  n_rows, n_cols,
                  dest_row_offset, dest_col_offset, dest_M_n_rows,
                  src_A_row_offset, src_A_col_offset, src_A_M_n_rows,
                  src_B_row_offset, src_B_col_offset, src_B_M_n_rows);
    #else
    coot_stop_runtime_error("coot_rt::eop_mat(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::eop_mat(): unknown backend");
    }
  }



/**
 * Perform an elementwise operation on two cubes of size `n_rows` x `n_cols` x `n_slices`.
 */
template<typename eT1, typename eT2>
inline
void
coot_rt_t::eop_cube(const twoway_kernel_id::enum_id num,
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

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::eop_cube(num, dest, src_A, src_B,
                     n_rows, n_cols, n_slices,
                     dest_row_offset, dest_col_offset, dest_slice_offset, dest_M_n_rows, dest_M_n_cols,
                     src_A_row_offset, src_A_col_offset, src_A_slice_offset, src_A_M_n_rows, src_A_M_n_cols,
                     src_B_row_offset, src_B_col_offset, src_B_slice_offset, src_B_M_n_rows, src_B_M_n_cols);
    #else
    coot_stop_runtime_error("coot_rt::eop_cube(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::eop_cube(num, dest, src_A, src_B,
                   n_rows, n_cols, n_slices,
                   dest_row_offset, dest_col_offset, dest_slice_offset, dest_M_n_rows, dest_M_n_cols,
                   src_A_row_offset, src_A_col_offset, src_A_slice_offset, src_A_M_n_rows, src_A_M_n_cols,
                   src_B_row_offset, src_B_col_offset, src_B_slice_offset, src_B_M_n_rows, src_B_M_n_cols);
    #else
    coot_stop_runtime_error("coot_rt::eop_cube(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::eop_cube(): unknown backend");
    }
  }



template<typename eT>
inline
eT
coot_rt_t::prod(const dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::prod(mem, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::prod(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::prod(mem, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::prod(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::prod(): unknown backend");
    }

  return eT(0); // fix warnings
  }



template<typename eT>
inline
eT
coot_rt_t::max_abs(const dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::max_abs(mem, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::max_abs(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::max_abs(mem, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::max_abs(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::max_abs(): unknown backend");
    }

  return eT(0); // fix warnings
  }



template<typename eT1, typename eT2>
inline
bool
coot_rt_t::all_vec(const dev_mem_t<eT1> mem, const uword n_elem, const eT2 val, const twoway_kernel_id::enum_id num, const twoway_kernel_id::enum_id num_small)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::all_vec(mem, n_elem, val, num, num_small);
    #else
    coot_stop_runtime_error("coot_rt::all_vec(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::all_vec(mem, n_elem, val, num, num_small);
    #else
    coot_stop_runtime_error("coot_rt::all_vec(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::all_vec(): unknown backend");
    }

  return false; // stop compilation warnings
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::all(dev_mem_t<uword> out_mem, const dev_mem_t<eT1> in_mem, const uword n_rows, const uword n_cols, const eT2 val, const twoway_kernel_id::enum_id num, const bool colwise)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::all(out_mem, in_mem, n_rows, n_cols, val, num, colwise);
    #else
    coot_stop_runtime_error("coot_rt::all(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::all(out_mem, in_mem, n_rows, n_cols, val, num, colwise);
    #else
    coot_stop_runtime_error("coot_rt::all(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::all(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
bool
coot_rt_t::any_vec(const dev_mem_t<eT1> mem, const uword n_elem, const eT2 val, const twoway_kernel_id::enum_id num, const twoway_kernel_id::enum_id num_small)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::any_vec(mem, n_elem, val, num, num_small);
    #else
    coot_stop_runtime_error("coot_rt::any_vec(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::any_vec(mem, n_elem, val, num, num_small);
    #else
    coot_stop_runtime_error("coot_rt::any_vec(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::any_vec(): unknown backend");
    }

  return false; // stop compilation warnings
  }



template<typename eT>
inline
bool
coot_rt_t::any_vec(const dev_mem_t<eT> mem, const uword n_elem, const eT val, const oneway_real_kernel_id::enum_id num, const oneway_real_kernel_id::enum_id num_small)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::any_vec(mem, n_elem, val, num, num_small);
    #else
    coot_stop_runtime_error("coot_rt::any_vec(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::any_vec(mem, n_elem, val, num, num_small);
    #else
    coot_stop_runtime_error("coot_rt::any_vec(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::any_vec(): unknown backend");
    }

  return false; // stop compilation warnings
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::any(dev_mem_t<uword> out_mem, const dev_mem_t<eT1> in_mem, const uword n_rows, const uword n_cols, const eT2 val, const twoway_kernel_id::enum_id num, const bool colwise)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::any(out_mem, in_mem, n_rows, n_cols, val, num, colwise);
    #else
    coot_stop_runtime_error("coot_rt::any(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::any(out_mem, in_mem, n_rows, n_cols, val, num, colwise);
    #else
    coot_stop_runtime_error("coot_rt::any(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::any(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::relational_scalar_op(dev_mem_t<uword> out_mem, const dev_mem_t<eT1> in_mem, const uword n_elem, const eT2 val, const twoway_kernel_id::enum_id num, const std::string& name)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::relational_scalar_op(out_mem, in_mem, n_elem, val, num, name);
    #else
    coot_stop_runtime_error("coot_rt::relational_scalar_op(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::relational_scalar_op(out_mem, in_mem, n_elem, val, num, name);
    #else
    coot_stop_runtime_error("coot_rt::relational_scalar_op(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::relational_scalar_op(): unknown backend");
    }
  }



template<typename eT1>
inline
void
coot_rt_t::relational_unary_array_op(dev_mem_t<uword> out_mem, const dev_mem_t<eT1> in_mem, const uword n_elem, const oneway_real_kernel_id::enum_id num, const std::string& name)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::relational_unary_array_op(out_mem, in_mem, n_elem, num, name);
    #else
    coot_stop_runtime_error("coot_rt::relational_unary_array_op(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::relational_unary_array_op(out_mem, in_mem, n_elem, num, name);
    #else
    coot_stop_runtime_error("coot_rt::relational_unary_array_op(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::relational_unary_array_op(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::relational_array_op(dev_mem_t<uword> out_mem, const dev_mem_t<eT1> X_mem, const dev_mem_t<eT2> Y_mem, const uword n_elem, const twoway_kernel_id::enum_id num, const std::string& name)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::relational_array_op(out_mem, X_mem, Y_mem, n_elem, num, name);
    #else
    coot_stop_runtime_error("coot_rt::relational_array_op(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::relational_array_op(out_mem, X_mem, Y_mem, n_elem, num, name);
    #else
    coot_stop_runtime_error("coot_rt::relational_array_op(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::relational_array_op(): unknown backend");
    }
  }



template<typename eT>
inline
std::tuple<bool, std::string>
coot_rt_t::chol(dev_mem_t<eT> out, const uword n_rows)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::chol(out, n_rows);
    #else
    coot_stop_runtime_error("coot_rt::chol(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::chol(out, n_rows);
    #else
    coot_stop_runtime_error("coot_rt::chol(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::chol(): unknown backend");
    }

  return std::make_tuple(false, ""); // fix warnings
  }



template<typename eT>
inline
std::tuple<bool, std::string>
coot_rt_t::lu(dev_mem_t<eT> L, dev_mem_t<eT> U, dev_mem_t<eT> in, const bool pivoting, dev_mem_t<eT> P, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::lu(L, U, in, pivoting, P, n_rows, n_cols);
    #else
    coot_stop_runtime_error("coot_rt::lu(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::lu(L, U, in, pivoting, P, n_rows, n_cols);
    #else
    coot_stop_runtime_error("coot_rt::lu(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::lu(): unknown backend");
    }

  return std::make_tuple(false, ""); // fix warnings
  }



template<typename eT>
inline
std::tuple<bool, std::string>
coot_rt_t::det(dev_mem_t<eT> A, const uword n_rows, eT& out_val)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::det(A, n_rows, out_val);
    #else
    coot_stop_runtime_error("coot_rt::det(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::det(A, n_rows, out_val);
    #else
    coot_stop_runtime_error("coot_rt::det(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::det(): unknown backend");
    }

  return std::make_tuple(false, ""); // fix warnings

  }



template<typename eT>
inline
std::tuple<bool, std::string>
coot_rt_t::svd(dev_mem_t<eT> U, dev_mem_t<eT> S, dev_mem_t<eT> V, dev_mem_t<eT> A, const uword n_rows, const uword n_cols, const bool compute_u_vt)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::svd(U, S, V, A, n_rows, n_cols, compute_u_vt);
    #else
    coot_stop_runtime_error("coot_rt::svd(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::svd(U, S, V, A, n_rows, n_cols, compute_u_vt);
    #else
    coot_stop_runtime_error("coot_rt::svd(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::svd(): unknown backend");
    }

  return std::make_tuple(false, ""); // fix warnings
  }



template<typename eT>
inline
std::tuple<bool, std::string>
coot_rt_t::eig_sym(dev_mem_t<eT> mem, const uword n_rows, const bool eigenvectors, dev_mem_t<eT> eigenvalues)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::eig_sym(mem, n_rows, eigenvectors, eigenvalues);
    #else
    coot_stop_runtime_error("coot_rt::eig_sym(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::eig_sym(mem, n_rows, eigenvectors, eigenvalues);
    #else
    coot_stop_runtime_error("coot_rt::eig_sym(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::eig_sym(): unknown backend");
    }

  return std::make_tuple(false, ""); // fix warnings
  }



template<typename eT>
inline
std::tuple<bool, std::string>
coot_rt_t::solve_square_fast(dev_mem_t<eT> A, const bool trans_A, dev_mem_t<eT> B, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::solve_square_fast(A, trans_A, B, n_rows, n_cols);
    #else
    coot_stop_runtime_error("coot_rt::solve_square_fast(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::solve_square_fast(A, trans_A, B, n_rows, n_cols);
    #else
    coot_stop_runtime_error("coot_rt::solve_square_fast(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::solve_square_fast(): unknown backend");
    }

  return std::make_tuple(false, "");
  }



template<typename eT>
inline
void
coot_rt_t::copy_from_dev_mem(eT* dest,
                             const dev_mem_t<eT> src,
                             const uword n_rows,
                             const uword n_cols,
                             const uword src_row_offset,
                             const uword src_col_offset,
                             const uword src_M_n_rows)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::copy_from_dev_mem(dest, src,
                              n_rows, n_cols,
                              src_row_offset, src_col_offset, src_M_n_rows);
    #else
    coot_stop_runtime_error("coot_rt::copy_from_dev_mem(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::copy_from_dev_mem(dest, src,
                            n_rows, n_cols,
                            src_row_offset, src_col_offset, src_M_n_rows);
    #else
    coot_stop_runtime_error("coot_rt::copy_from_dev_mem(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::copy_from_dev_mem(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::copy_into_dev_mem(dev_mem_t<eT> dest, const eT* src, const uword N)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::copy_into_dev_mem(dest, src, N);
    #else
    coot_stop_runtime_error("coot_rt::copy_into_dev_mem(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::copy_into_dev_mem(dest, src, N);
    #else
    coot_stop_runtime_error("coot_rt::copy_into_dev_mem(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::copy_into_dev_mem(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::eye(dev_mem_t<eT> mem, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::eye(mem, n_rows, n_cols);
    #else
    coot_stop_runtime_error("coot_rt::eye(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::eye(mem, n_rows, n_cols);
    #else
    coot_stop_runtime_error("coot_rt::eye(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::eye(): unknown backend");
    }
  }



template<typename eT>
inline
eT
coot_rt_t::get_val(const dev_mem_t<eT> mem, const uword index)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::get_val(mem, index);
    #else
    coot_stop_runtime_error("coot_rt::get_val(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::get_val(mem, index);
    #else
    coot_stop_runtime_error("coot_rt::get_val(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::get_val(): unknown backend");
    }

  return eT(0); // fix warnings
  }



template<typename eT>
inline
void
coot_rt_t::set_val(dev_mem_t<eT> mem, const uword index, const eT val)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::set_val(mem, index, val);
    #else
    coot_stop_runtime_error("coot_rt::set_val(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::set_val(mem, index, val);
    #else
    coot_stop_runtime_error("coot_rt::set_val(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::set_val(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::val_add_inplace(dev_mem_t<eT> mem, const uword index, const eT val)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::val_add_inplace(mem, index, val);
    #else
    coot_stop_runtime_error("coot_rt::val_add_inplace(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::val_add_inplace(mem, index, val);
    #else
    coot_stop_runtime_error("coot_rt::val_add_inplace(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::val_add_inplace(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::val_minus_inplace(dev_mem_t<eT> mem, const uword index, const eT val)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::val_minus_inplace(mem, index, val);
    #else
    coot_stop_runtime_error("coot_rt::val_minus_inplace(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::val_minus_inplace(mem, index, val);
    #else
    coot_stop_runtime_error("coot_rt::val_minus_inplace(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::val_minus_inplace(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::val_mul_inplace(dev_mem_t<eT> mem, const uword index, const eT val)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::val_mul_inplace(mem, index, val);
    #else
    coot_stop_runtime_error("coot_rt::val_mul_inplace(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::val_mul_inplace(mem, index, val);
    #else
    coot_stop_runtime_error("coot_rt::val_mul_inplace(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::val_mul_inplace(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::val_div_inplace(dev_mem_t<eT> mem, const uword index, const eT val)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::val_div_inplace(mem, index, val);
    #else
    coot_stop_runtime_error("coot_rt::val_div_inplace(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::val_div_inplace(mem, index, val);
    #else
    coot_stop_runtime_error("coot_rt::val_div_inplace(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::val_div_inplace(): unknown backend");
    }
  }



template<typename eT, const bool do_trans_A, const bool do_trans_B>
inline
void
coot_rt_t::gemm(dev_mem_t<eT> C_mem,
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
                const uword B_M_n_rows)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::gemm<do_trans_A, do_trans_B>::apply(C_mem,
                                                C_n_rows, C_n_cols,
                                                A_mem,
                                                A_n_rows, A_n_cols,
                                                B_mem,
                                                alpha, beta,
                                                C_row_offset, C_col_offset, C_M_n_rows,
                                                A_row_offset, A_col_offset, A_M_n_rows,
                                                B_row_offset, B_col_offset, B_M_n_rows);
    #else
    coot_stop_runtime_error("coot_rt::gemm(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::gemm<do_trans_A, do_trans_B>::apply(C_mem,
                                              C_n_rows, C_n_cols,
                                              A_mem,
                                              A_n_rows, A_n_cols,
                                              B_mem,
                                              alpha, beta,
                                              C_row_offset, C_col_offset, C_M_n_rows,
                                              A_row_offset, A_col_offset, A_M_n_rows,
                                              B_row_offset, B_col_offset, B_M_n_rows);
    #else
    coot_stop_runtime_error("coot_rt::gemm(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::gemm(): unknown backend");
    }
  }



template<typename eT, const bool do_trans_A>
inline
void
coot_rt_t::gemv(dev_mem_t<eT> y_mem,
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
                const uword x_mem_incr)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::gemv<do_trans_A>::apply(y_mem,
                                    A_mem,
                                    A_n_rows, A_n_cols,
                                    x_mem,
                                    alpha, beta,
                                    y_offset, y_mem_incr,
                                    A_row_offset, A_col_offset, A_M_n_rows,
                                    x_offset, x_mem_incr);
    #else
    coot_stop_runtime_error("coot_rt::gemv(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::gemv<do_trans_A>::apply(y_mem,
                                  A_mem,
                                  A_n_rows, A_n_cols,
                                  x_mem,
                                  alpha, beta,
                                  y_offset, y_mem_incr,
                                  A_row_offset, A_col_offset, A_M_n_rows,
                                  x_offset, x_mem_incr);
    #else
    coot_stop_runtime_error("coot_rt::gemv(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::gemv(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::mul_diag(dev_mem_t<eT> C_mem, const uword C_n_rows, const uword C_n_cols, const eT alpha, const dev_mem_t<eT> A_mem, const bool A_is_diag, const bool A_trans, const dev_mem_t<eT> B_mem, const bool B_is_diag, const bool B_trans)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::mul_diag(C_mem, C_n_rows, C_n_cols, alpha, A_mem, A_is_diag, A_trans, B_mem, B_is_diag, B_trans);
    #else
    coot_stop_runtime_error("coot_rt::mul_diag(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::mul_diag(C_mem, C_n_rows, C_n_cols, alpha, A_mem, A_is_diag, A_trans, B_mem, B_is_diag, B_trans);
    #else
    coot_stop_runtime_error("coot_rt::mul_diag(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::mul_diag(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::sum(dev_mem_t<eT2> dest,
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
               const uword src_M_n_rows)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::sum(dest, src,
                n_rows, n_cols,
                dim, post_conv_apply,
                dest_offset, dest_mem_incr,
                src_row_offset, src_col_offset, src_M_n_rows);
    #else
    coot_stop_runtime_error("coot_rt::sum(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::sum(dest, src,
              n_rows, n_cols,
              dim, post_conv_apply,
              dest_offset, dest_mem_incr,
              src_row_offset, src_col_offset, src_M_n_rows);
    #else
    coot_stop_runtime_error("coot_rt::sum(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::sum(): unknown backend");
    }
  }



template<typename eT>
inline
eT
coot_rt_t::accu(const dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::accu(mem, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::accu(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::accu(mem, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::accu(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::accu(): unknown backend");
    }

  return eT(0);
  }



template<typename eT>
inline
eT
coot_rt_t::accu_subview(const dev_mem_t<eT> mem, const uword M_n_rows, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::accu_subview(mem, M_n_rows, aux_row1, aux_col1, n_rows, n_cols);
    #else
    coot_stop_runtime_error("coot_rt::accu_subview(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::accu_subview(mem, M_n_rows, aux_row1, aux_col1, n_rows, n_cols);
    #else
    coot_stop_runtime_error("coot_rt::accu_subview(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::accu_subview(): unknown backend");
    }

  return eT(0); // fix warnings
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::min(dev_mem_t<eT2> dest,
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
               const uword src_M_n_rows)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::min(dest, src,
                n_rows, n_cols,
                dim, post_conv_apply,
                dest_offset, dest_mem_incr,
                src_row_offset, src_col_offset, src_M_n_rows);
    #else
    coot_stop_runtime_error("coot_rt::min(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::min(dest, src,
              n_rows, n_cols,
              dim, post_conv_apply,
              dest_offset, dest_mem_incr,
              src_row_offset, src_col_offset, src_M_n_rows);
    #else
    coot_stop_runtime_error("coot_rt::min(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::min(): unknown backend");
    }
  }



template<typename eT>
inline
eT
coot_rt_t::min_vec(const dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::min_vec(mem, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::min_vec(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::min_vec(mem, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::min_vec(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::min_vec(): unknown backend");
    }

  return eT(0); // fix warnings
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::max(dev_mem_t<eT2> dest,
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
               const uword src_M_n_rows)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::max(dest, src,
                n_rows, n_cols,
                dim, post_conv_apply,
                dest_offset, dest_mem_incr,
                src_row_offset, src_col_offset, src_M_n_rows);
    #else
    coot_stop_runtime_error("coot_rt::max(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::max(dest, src,
              n_rows, n_cols,
              dim, post_conv_apply,
              dest_offset, dest_mem_incr,
              src_row_offset, src_col_offset, src_M_n_rows);
    #else
    coot_stop_runtime_error("coot_rt::max(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::max(): unknown backend");
    }
  }



template<typename eT>
inline
eT
coot_rt_t::max_vec(const dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::max_vec(mem, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::max_vec(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::max_vec(mem, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::max_vec(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::max_vec(): unknown backend");
    }

  return eT(0); // fix warnings
  }



template<typename eT>
inline
void
coot_rt_t::index_min(dev_mem_t<uword> dest,
                     const dev_mem_t<eT> src,
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

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::index_min(dest, src, n_rows, n_cols, dim,
                             dest_offset, dest_mem_incr,
                             src_row_offset, src_col_offset, src_M_n_rows);
    #else
    coot_stop_runtime_error("coot_rt::index_min(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::index_min(dest, src, n_rows, n_cols, dim,
                           dest_offset, dest_mem_incr,
                           src_row_offset, src_col_offset, src_M_n_rows);
    #else
    coot_stop_runtime_error("coot_rt::index_min(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::index_min(): unknown backend");
    }
  }



template<typename eT>
inline
uword
coot_rt_t::index_min_vec(const dev_mem_t<eT> mem, const uword n_elem, eT* min_val)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::index_min_vec(mem, n_elem, min_val);
    #else
    coot_stop_runtime_error("coot_rt::index_min_vec(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::index_min_vec(mem, n_elem, min_val);
    #else
    coot_stop_runtime_error("coot_rt::index_min_vec(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::index_min_vec(): unknown backend");
    }

  return uword(0); // fix warnings
  }



template<typename eT>
inline
void
coot_rt_t::index_max(dev_mem_t<uword> dest,
                     const dev_mem_t<eT> src,
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

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::index_max(dest, src, n_rows, n_cols, dim,
                             dest_offset, dest_mem_incr,
                             src_row_offset, src_col_offset, src_M_n_rows);
    #else
    coot_stop_runtime_error("coot_rt::index_max(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::index_max(dest, src, n_rows, n_cols, dim,
                           dest_offset, dest_mem_incr,
                           src_row_offset, src_col_offset, src_M_n_rows);
    #else
    coot_stop_runtime_error("coot_rt::index_max(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::index_max(): unknown backend");
    }
  }



template<typename eT>
inline
uword
coot_rt_t::index_max_vec(const dev_mem_t<eT> mem, const uword n_elem, eT* max_val)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::index_max_vec(mem, n_elem, max_val);
    #else
    coot_stop_runtime_error("coot_rt::index_max_vec(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::index_max_vec(mem, n_elem, max_val);
    #else
    coot_stop_runtime_error("coot_rt::index_max_vec(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::index_max_vec(): unknown backend");
    }

  return uword(0); // fix warnings
  }



template<typename eT>
inline
eT
coot_rt_t::trace(const dev_mem_t<eT> mem, const uword n_rows, const uword n_cols)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::trace(mem, n_rows, n_cols);
    #else
    coot_stop_runtime_error("coot_rt::trace(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::trace(mem, n_rows, n_cols);
    #else
    coot_stop_runtime_error("coot_rt::trace(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::trace(): unknown backend");
    }

  return eT(0); // fix warning
  }



template<typename eT1, typename eT2>
inline
typename promote_type<eT1, eT2>::result
coot_rt_t::dot(const dev_mem_t<eT1> mem1, const dev_mem_t<eT2> mem2, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::dot(mem1, mem2, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::dot(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::dot(mem1, mem2, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::dot(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::dot(): unknown backend");
    }

  return typename promote_type<eT1, eT2>::result(0); // fix warning
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::broadcast_op(const twoway_kernel_id::enum_id num,
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
                        const uword src_M_n_rows)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::broadcast_op(num, dest, dest_in, src,
                         src_n_rows, src_n_cols,
                         copies_per_row, copies_per_col,
                         dest_row_offset, dest_col_offset, dest_M_n_rows,
                         dest_in_row_offset, dest_in_col_offset, dest_in_M_n_rows,
                         src_row_offset, src_col_offset, src_M_n_rows);
    #else
    coot_stop_runtime_error("coot_rt::broadcast_op(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::broadcast_op(num, dest, dest_in, src,
                       src_n_rows, src_n_cols,
                       copies_per_row, copies_per_col,
                       dest_row_offset, dest_col_offset, dest_M_n_rows,
                       dest_in_row_offset, dest_in_col_offset, dest_in_M_n_rows,
                       src_row_offset, src_col_offset, src_M_n_rows);
    #else
    coot_stop_runtime_error("coot_rt::broadcast_op(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::broadcast_op(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::broadcast_subset_op(const twoway_kernel_id::enum_id num,
                               dev_mem_t<eT2> dest,
                               const dev_mem_t<eT2> dest_in,
                               const dev_mem_t<eT1> src,
                               const dev_mem_t<uword> indices,
                               const uword mode,
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
                               const uword indices_incr)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::broadcast_subset_op(num, dest, dest_in, src, indices, mode,
                                src_n_rows, src_n_cols,
                                copies_per_row, copies_per_col,
                                dest_row_offset, dest_col_offset, dest_M_n_rows,
                                dest_in_row_offset, dest_in_col_offset, dest_in_M_n_rows,
                                src_row_offset, src_col_offset, src_M_n_rows,
                                indices_offset, indices_incr);
    #else
    coot_stop_runtime_error("coot_rt::broadcast_subset_op(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::broadcast_subset_op(num, dest, dest_in, src, indices, mode,
                              src_n_rows, src_n_cols,
                              copies_per_row, copies_per_col,
                              dest_row_offset, dest_col_offset, dest_M_n_rows,
                              dest_in_row_offset, dest_in_col_offset, dest_in_M_n_rows,
                              src_row_offset, src_col_offset, src_M_n_rows,
                              indices_offset, indices_incr);
    #else
    coot_stop_runtime_error("coot_rt::broadcast_subset_op(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::broadcast_subset_op(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::linspace(dev_mem_t<eT> mem, const uword mem_incr, const eT start, const eT end, const uword num)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::linspace(mem, mem_incr, start, end, num);
    #else
    coot_stop_runtime_error("coot_rt::linspace(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::linspace(mem, mem_incr, start, end, num);
    #else
    coot_stop_runtime_error("coot_rt::linspace(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::linspace(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::logspace(dev_mem_t<eT> mem, const uword mem_incr, const eT start, const eT end, const uword num)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::logspace(mem, mem_incr, start, end, num);
    #else
    coot_stop_runtime_error("coot_rt::logspace(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::logspace(mem, mem_incr, start, end, num);
    #else
    coot_stop_runtime_error("coot_rt::logspace(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::logspace(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::regspace(dev_mem_t<eT> mem, const uword mem_incr, const eT start, const eT delta, const eT end, const uword num, const bool desc)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::regspace(mem, mem_incr, start, delta, end, num, desc);
    #else
    coot_stop_runtime_error("coot_rt::regspace(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::regspace(mem, mem_incr, start, delta, end, num, desc);
    #else
    coot_stop_runtime_error("coot_rt::regspace(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::regspace(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::clamp(dev_mem_t<eT2> dest,
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
                 const uword src_M_n_rows)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::clamp(dest, src,
                  min_val, max_val,
                  n_rows, n_cols,
                  dest_row_offset, dest_col_offset, dest_M_n_rows,
                  src_row_offset, src_col_offset, src_M_n_rows);
    #else
    coot_stop_runtime_error("coot_rt::clamp(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::clamp(dest, src,
                min_val, max_val,
                n_rows, n_cols,
                dest_row_offset, dest_col_offset, dest_M_n_rows,
                src_row_offset, src_col_offset, src_M_n_rows);
    #else
    coot_stop_runtime_error("coot_rt::clamp(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::clamp(): unknown backend");
    }
  }



template<typename eT>
inline
eT
coot_rt_t::vec_norm_1(const dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::vec_norm_1(mem, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::vec_norm_1(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::vec_norm_1(mem, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::vec_norm_1(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::vec_norm_1(): unknown backend");
    }

  return eT(0); // fix warning
  }



template<typename eT>
inline
eT
coot_rt_t::vec_norm_2(const dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::vec_norm_2(mem, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::vec_norm_2(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::vec_norm_2(mem, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::vec_norm_2(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::vec_norm_2(): unknown backend");
    }

  return eT(0); // fix warning
  }



template<typename eT>
inline
eT
coot_rt_t::vec_norm_k(const dev_mem_t<eT> mem, const uword n_elem, const uword k)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::vec_norm_k(mem, n_elem, k);
    #else
    coot_stop_runtime_error("coot_rt::vec_norm_k(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::vec_norm_k(mem, n_elem, k);
    #else
    coot_stop_runtime_error("coot_rt::vec_norm_k(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::vec_norm_k(): unknown backend");
    }

  return eT(0); // fix warning
  }



template<typename eT>
inline
eT
coot_rt_t::vec_norm_min(const dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::vec_norm_min(mem, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::vec_norm_min(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::vec_norm_min(mem, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::vec_norm_min(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::vec_norm_min(): unknown backend");
    }

  return eT(0); // fix warning
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::mean(dev_mem_t<eT2> dest,
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
                const uword src_M_n_rows)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::mean(dest, src,
                 n_rows, n_cols, dim, post_conv_apply,
                 dest_offset, dest_mem_incr,
                 src_row_offset, src_col_offset, src_M_n_rows);
    #else
    coot_stop_runtime_error("coot_rt::mean(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::mean(dest, src,
               n_rows, n_cols, dim, post_conv_apply,
               dest_offset, dest_mem_incr,
               src_row_offset, src_col_offset, src_M_n_rows);
    #else
    coot_stop_runtime_error("coot_rt::mean(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::mean(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::median(dev_mem_t<eT2> dest,
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

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::median(dest, src,
                   n_rows, n_cols,
                   dim,
                   dest_offset, dest_mem_incr,
                   src_row_offset, src_col_offset, src_M_n_rows);
    #else
    coot_stop_runtime_error("coot_rt::median(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::median(dest, src,
                 n_rows, n_cols,
                 dim,
                 dest_offset, dest_mem_incr,
                 src_row_offset, src_col_offset, src_M_n_rows);
    #else
    coot_stop_runtime_error("coot_rt::median(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::median(): unknown backend");
    }
  }



template<typename eT>
inline
eT
coot_rt_t::median_vec(dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::median_vec(mem, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::median_vec(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::median_vec(mem, n_elem);
    #else
    coot_stop_runtime_error("coot_rt::median_vec(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::median_vec(): unknown backend");
    }

  return eT(0); // fix warning
  }



template<typename eT>
inline
void
coot_rt_t::var(dev_mem_t<eT> dest,
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
               const uword src_means_mem_incr)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::var(dest, src, src_means,
                n_rows, n_cols,
                dim, norm_type,
                dest_offset, dest_mem_incr,
                src_row_offset, src_col_offset, src_M_n_rows,
                src_means_offset, src_means_mem_incr);
    #else
    coot_stop_runtime_error("coot_rt::var(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::var(dest, src, src_means,
              n_rows, n_cols,
              dim, norm_type,
              dest_offset, dest_mem_incr,
              src_row_offset, src_col_offset, src_M_n_rows,
              src_means_offset, src_means_mem_incr);
    #else
    coot_stop_runtime_error("coot_rt::var(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::var(): unknown backend");
    }
  }



template<typename eT>
inline
eT
coot_rt_t::var_vec(const dev_mem_t<eT> mem, const eT mean, const uword n_elem, const uword norm_type)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::var_vec(mem, mean, n_elem, norm_type);
    #else
    coot_stop_runtime_error("coot_rt::var_vec(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::var_vec(mem, mean, n_elem, norm_type);
    #else
    coot_stop_runtime_error("coot_rt::var_vec(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::var_vec(): unknown backend");
    }

  return eT(0); // fix warning
  }



template<typename eT>
inline
eT
coot_rt_t::var_vec_subview(const dev_mem_t<eT> mem, const eT mean, const uword M_n_rows, const uword M_n_cols, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols, const uword norm_type)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::var_vec_subview(mem, mean, M_n_rows, M_n_cols, aux_row1, aux_col1, n_rows, n_cols, norm_type);
    #else
    coot_stop_runtime_error("coot_rt::var_vec_subview(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::var_vec_subview(mem, mean, M_n_rows, M_n_cols, aux_row1, aux_col1, n_rows, n_cols, norm_type);
    #else
    coot_stop_runtime_error("coot_rt::var_vec_subview(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::var_vec_subview(): unknown backend");
    }

  return eT(0); // fix warning
  }



template<typename eT1, typename eT2, typename eT3, typename eT4, typename eT5>
inline
void
coot_rt_t::join_cols(dev_mem_t<eT5> out,
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

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::join_cols(out,
                      A, A_n_rows, A_n_cols,
                      B, B_n_rows, B_n_cols,
                      C, C_n_rows, C_n_cols,
                      D, D_n_rows, D_n_cols,
                      out_row_offset, out_col_offset, out_M_n_rows,
                      A_row_offset, A_col_offset, A_M_n_rows,
                      B_row_offset, B_col_offset, B_M_n_rows,
                      C_row_offset, C_col_offset, C_M_n_rows,
                      D_row_offset, D_col_offset, D_M_n_rows);
    #else
    coot_stop_runtime_error("coot_rt::join_cols(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::join_cols(out,
                    A, A_n_rows, A_n_cols,
                    B, B_n_rows, B_n_cols,
                    C, C_n_rows, C_n_cols,
                    D, D_n_rows, D_n_cols,
                    out_row_offset, out_col_offset, out_M_n_rows,
                    A_row_offset, A_col_offset, A_M_n_rows,
                    B_row_offset, B_col_offset, B_M_n_rows,
                    C_row_offset, C_col_offset, C_M_n_rows,
                    D_row_offset, D_col_offset, D_M_n_rows);
    #else
    coot_stop_runtime_error("coot_rt::join_cols(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::join_cols(): unknown backend");
    }
  }



template<typename eT1, typename eT2, typename eT3, typename eT4, typename eT5>
inline
void
coot_rt_t::join_rows(dev_mem_t<eT5> out,
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

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::join_rows(out,
                      A, A_n_rows, A_n_cols,
                      B, B_n_rows, B_n_cols,
                      C, C_n_rows, C_n_cols,
                      D, D_n_rows, D_n_cols,
                      out_row_offset, out_col_offset, out_M_n_rows,
                      A_row_offset, A_col_offset, A_M_n_rows,
                      B_row_offset, B_col_offset, B_M_n_rows,
                      C_row_offset, C_col_offset, C_M_n_rows,
                      D_row_offset, D_col_offset, D_M_n_rows);
    #else
    coot_stop_runtime_error("coot_rt::join_rows(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::join_rows(out,
                    A, A_n_rows, A_n_cols,
                    B, B_n_rows, B_n_cols,
                    C, C_n_rows, C_n_cols,
                    D, D_n_rows, D_n_cols,
                    out_row_offset, out_col_offset, out_M_n_rows,
                    A_row_offset, A_col_offset, A_M_n_rows,
                    B_row_offset, B_col_offset, B_M_n_rows,
                    C_row_offset, C_col_offset, C_M_n_rows,
                    D_row_offset, D_col_offset, D_M_n_rows);
    #else
    coot_stop_runtime_error("coot_rt::join_rows(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::join_rows(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::sort(dev_mem_t<eT> mem,
                const uword n_rows,
                const uword n_cols,
                const uword sort_type,
                const uword dim,
                // subview arguments
                const uword row_offset,
                const uword col_offset,
                const uword M_n_rows)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::sort(mem, n_rows, n_cols, sort_type, dim, row_offset, col_offset, M_n_rows);
    #else
    coot_stop_runtime_error("coot_rt::sort(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::sort(mem, n_rows, n_cols, sort_type, dim, row_offset, col_offset, M_n_rows);
    #else
    coot_stop_runtime_error("coot_rt::sort(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::sort(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::sort_vec(dev_mem_t<eT> mem, const uword n_elem, const uword sort_type)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::sort_vec(mem, n_elem, sort_type);
    #else
    coot_stop_runtime_error("coot_rt::sort_vec(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::sort_vec(mem, n_elem, sort_type);
    #else
    coot_stop_runtime_error("coot_rt::sort_vec(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::sort_vec(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::sort_index_vec(dev_mem_t<uword> out, dev_mem_t<eT> A, const uword n_elem, const uword sort_type, const uword stable_sort)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::sort_index_vec(out, A, n_elem, sort_type, stable_sort);
    #else
    coot_stop_runtime_error("coot_rt::sort_index_vec(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::sort_index_vec(out, A, n_elem, sort_type, stable_sort);
    #else
    coot_stop_runtime_error("coot_rt::sort_index_vec(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::sort_index_vec(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::find(dev_mem_t<uword>& out, uword& out_len, const dev_mem_t<eT> A, const uword n_elem, const uword k, const uword find_type)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::find(out, out_len, A, n_elem, k, find_type);
    #else
    coot_stop_runtime_error("coot_rt::find(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::find(out, out_len, A, n_elem, k, find_type);
    #else
    coot_stop_runtime_error("coot_rt::find(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::find(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::symmat(dev_mem_t<eT2> out, const dev_mem_t<eT1> in, const uword size, const uword lower)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::symmat(out, in, size, lower);
    #else
    coot_stop_runtime_error("coot_rt::symmat(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::symmat(out, in, size, lower);
    #else
    coot_stop_runtime_error("coot_rt::symmat(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::symmat(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::cross(dev_mem_t<eT2> out, const dev_mem_t<eT1> A, const dev_mem_t<eT1> B)
  {
  coot_extra_debug_sigprint();

  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::cross(out, A, B);
    #else
    coot_stop_runtime_error("coot_rt::cross(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::cross(out, A, B);
    #else
    coot_stop_runtime_error("coot_rt::cross(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::cross(): unknown backend");
    }
  }



template<typename eT>
inline
void
coot_rt_t::rotate_180(dev_mem_t<eT> out, const dev_mem_t<eT> in, const uword n_rows, const uword n_cols)
  {
  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::rotate_180(out, in, n_rows, n_cols);
    #else
    coot_stop_runtime_error("coot_rt::rotate_180(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::rotate_180(out, in, n_rows, n_cols);
    #else
    coot_stop_runtime_error("coot_rt::rotate_180(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::rotate_180(): unknown backend");
    }
  }



template<typename eT>
inline
bool
coot_rt_t::approx_equal(const dev_mem_t<eT> A,
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
                        const eT rel_tol)
  {
  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    return opencl::approx_equal(A, A_row_offset, A_col_offset, A_M_n_rows,
                                B, B_row_offset, B_col_offset, B_M_n_rows,
                                n_rows, n_cols,
                                sig, abs_tol, rel_tol);
    #else
    coot_stop_runtime_error("coot_rt::approx_equal(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    return cuda::approx_equal(A, A_row_offset, A_col_offset, A_M_n_rows,
                              B, B_row_offset, B_col_offset, B_M_n_rows,
                              n_rows, n_cols,
                              sig, abs_tol, rel_tol);
    #else
    coot_stop_runtime_error("coot_rt::approx_equal(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::approx_equal(): unknown backend");
    }

  return false; // fix warning
  }



template<typename eT>
inline
void
coot_rt_t::shuffle(dev_mem_t<eT> out,
                   const uword out_row_offset,
                   const uword out_col_offset,
                   const uword out_M_n_rows,
                   const dev_mem_t<eT> in,
                   const uword in_row_offset,
                   const uword in_col_offset,
                   const uword in_M_n_rows,
                   const uword n_rows,
                   const uword n_cols,
                   const uword dim)
  {
  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::shuffle(out, out_row_offset, out_col_offset, out_M_n_rows,
                    in, in_row_offset, in_col_offset, in_M_n_rows,
                    n_rows, n_cols,
                    dim);
    #else
    coot_stop_runtime_error("coot_rt::shuffle(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::shuffle(out, out_row_offset, out_col_offset, out_M_n_rows,
                  in, in_row_offset, in_col_offset, in_M_n_rows,
                  n_rows, n_cols,
                  dim);
    #else
    coot_stop_runtime_error("coot_rt::shuffle(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::shuffle(): unknown backend");
    }
  }



template<typename eT1, typename eT2>
inline
void
coot_rt_t::extract_cx(dev_mem_t<eT1> out_mem,
                      const uword out_row_offset,
                      const uword out_col_offset,
                      const uword out_M_n_rows,
                      const dev_mem_t<eT2> in_mem,
                      const uword in_row_offset,
                      const uword in_col_offset,
                      const uword in_M_n_rows,
                      const uword n_rows,
                      const uword n_cols,
                      const bool imag)
  {
  if (get_rt().backend == CL_BACKEND)
    {
    #if defined(COOT_USE_OPENCL)
    opencl::extract_cx(out_mem, out_row_offset, out_col_offset, out_M_n_rows,
                       in_mem, in_row_offset, in_col_offset, in_M_n_rows,
                       n_rows, n_cols,
                       imag);
    #else
    coot_stop_runtime_error("coot_rt::extract_cx(): OpenCL backend not enabled");
    #endif
    }
  else if (get_rt().backend == CUDA_BACKEND)
    {
    #if defined(COOT_USE_CUDA)
    cuda::extract_cx(out_mem, out_row_offset, out_col_offset, out_M_n_rows,
                     in_mem, in_row_offset, in_col_offset, in_M_n_rows,
                     n_rows, n_cols,
                     imag);
    #else
    coot_stop_runtime_error("coot_rt::extract_cx(): CUDA backend not enabled");
    #endif
    }
  else
    {
    coot_stop_runtime_error("coot_rt::extract_cx(): unknown backend");
    }
  }
