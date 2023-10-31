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

// This file contains source code adapted from
// clMAGMA 1.3 (2014-11-14) and/or MAGMA 2.7 (2022-11-09).
// clMAGMA 1.3 and MAGMA 2.7 are distributed under a
// 3-clause BSD license as follows:
//
//  -- Innovative Computing Laboratory
//  -- Electrical Engineering and Computer Science Department
//  -- University of Tennessee
//  -- (C) Copyright 2009-2015
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions
//  are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of the University of Tennessee, Knoxville nor the
//    names of its contributors may be used to endorse or promote products
//    derived from this software without specific prior written permission.
//
//  This software is provided by the copyright holders and contributors
//  ``as is'' and any express or implied warranties, including, but not
//  limited to, the implied warranties of merchantability and fitness for
//  a particular purpose are disclaimed. In no event shall the copyright
//  holders or contributors be liable for any direct, indirect, incidental,
//  special, exemplary, or consequential damages (including, but not
//  limited to, procurement of substitute goods or services; loss of use,
//  data, or profits; or business interruption) however caused and on any
//  theory of liability, whether in contract, strict liability, or tort
//  (including negligence or otherwise) arising in any way out of the use
//  of this software, even if advised of the possibility of such damage.



// Adaptations of magmablas_* functions to use existing bandicoot backend functionality.

inline
void
magmablas_slaset
  (
  magma_uplo_t uplo,
  magma_int_t m,
  magma_int_t n,
  float offdiag,
  float diag,
  magmaFloat_ptr dA,
  size_t dA_offset,
  magma_int_t ldda,
  magma_queue_t queue
  )
  {
  magma_int_t info = 0;
  if (uplo != MagmaLower && uplo != MagmaUpper && uplo != MagmaFull)
    info = -1;
  else if ( m < 0 )
    info = -2;
  else if ( n < 0 )
    info = -3;
  else if ( ldda < std::max(1,m) )
    info = -7;

  if (info != 0)
    {
    // magma_xerbla( __func__, -(info) );
    return;  //info;
    }

  if (m == 0 || n == 0)
    {
    return;
    }

  opencl::magma_real_kernel_id::enum_id num;
  if (uplo == MagmaLower)
    {
    num = opencl::magma_real_kernel_id::laset_lower;
    }
  else if (uplo == MagmaUpper)
    {
    num = opencl::magma_real_kernel_id::laset_upper;
    }
  else
    {
    num = opencl::magma_real_kernel_id::laset_full;
    }

  magmablas_run_laset_kernel(num, uplo, m, n, offdiag, diag, dA, dA_offset, ldda, queue);
  }



inline
void
magmablas_slaswp
  (
  magma_int_t n,
  magmaFloat_ptr dAT,
  size_t dAT_offset,
  magma_int_t ldda,
  magma_int_t k1,
  magma_int_t k2,
  const magma_int_t* ipiv,
  magma_int_t inci,
  magma_queue_t queue
  )
  {
  return magmablas_laswp<float>(n, (cl_mem) dAT, dAT_offset, ldda, k1, k2, ipiv, inci, queue);
  }



inline
void
magmablas_dlaswp
  (
  magma_int_t n,
  magmaDouble_ptr dAT,
  size_t dAT_offset,
  magma_int_t ldda,
  magma_int_t k1,
  magma_int_t k2,
  const magma_int_t* ipiv,
  magma_int_t inci,
  magma_queue_t queue
  )
  {
  return magmablas_laswp<double>(n, (cl_mem) dAT, dAT_offset, ldda, k1, k2, ipiv, inci, queue);
  }



template<typename eT>
inline
void
magmablas_laswp
  (
  magma_int_t n,
  cl_mem dAT,
  size_t dAT_offset,
  magma_int_t ldda,
  magma_int_t k1,
  magma_int_t k2,
  const magma_int_t* ipiv,
  magma_int_t inci,
  magma_queue_t queue
  )
  {
  cl_kernel kernel;
  cl_int err;
  int i;

  magma_int_t info = 0;
  if ( n < 0 )
    {
    info = -1;
    }
  else if ( k1 < 1 || k1 > n )
    {
    info = -4;
    }
  else if ( k2 < 1 || k2 > n )
    {
    info = -5;
    }
  else if ( inci <= 0 )
    {
    info = -7;
    }

  if (info != 0)
    {
    //magma_xerbla( __func__, -(info) );
    return;  //info;
    }

  size_t grid[1] = { ((size_t) (n + MAGMABLAS_LASWP_NTHREADS - 1)) / 64 };
  size_t threads[1] = { MAGMABLAS_LASWP_NTHREADS };
  grid[0] *= threads[0];
  magmablas_laswp_params_t params;

  kernel = get_rt().cl_rt.get_kernel<eT>(opencl::magma_real_kernel_id::laswp);

  for( int k = k1-1; k < k2; k += MAGMABLAS_LASWP_MAX_PIVOTS )
    {
    int npivots = std::min( MAGMABLAS_LASWP_MAX_PIVOTS, k2-k );
    params.npivots = npivots;
    for( int j = 0; j < npivots; ++j )
      {
      params.ipiv[j] = ipiv[(k+j)*inci] - k - 1;
      }

    if ( kernel != NULL )
      {
      err = 0;
      i   = 0;
      size_t k_offset = dAT_offset + k*ldda;
      err |= coot_wrapper(clSetKernelArg)( kernel, i++, sizeof(n       ), &n        );
      err |= coot_wrapper(clSetKernelArg)( kernel, i++, sizeof(dAT     ), &dAT      );
      err |= coot_wrapper(clSetKernelArg)( kernel, i++, sizeof(k_offset), &k_offset );
      err |= coot_wrapper(clSetKernelArg)( kernel, i++, sizeof(ldda    ), &ldda     );
      err |= coot_wrapper(clSetKernelArg)( kernel, i++, sizeof(params  ), &params   );
      coot_check_runtime_error( err, "coot::opencl::magmablas_laswp(): couldn't set laswp kernel arguments" );

      err = coot_wrapper(clEnqueueNDRangeKernel)( queue, kernel, 1, NULL, grid, threads, 0, NULL, NULL );
      coot_check_runtime_error( err, "coot::opencl::magmablas_laswp(): couldn't run laswp kernel" );
      }
    }
  }



inline
void
magmablas_dlaset
  (
  magma_uplo_t uplo,
  magma_int_t m,
  magma_int_t n,
  double offdiag,
  double diag,
  magmaDouble_ptr dA,
  size_t dA_offset,
  magma_int_t ldda,
  magma_queue_t queue
  )
  {
  magma_int_t info = 0;
  if (uplo != MagmaLower && uplo != MagmaUpper && uplo != MagmaFull)
    info = -1;
  else if ( m < 0 )
    info = -2;
  else if ( n < 0 )
    info = -3;
  else if ( ldda < std::max(1,m) )
    info = -7;

  if (info != 0)
    {
    // magma_xerbla( __func__, -(info) );
    return;  //info;
    }

  if (m == 0 || n == 0)
    {
    return;
    }

  opencl::magma_real_kernel_id::enum_id num;
  if (uplo == MagmaLower)
    {
    num = opencl::magma_real_kernel_id::laset_lower;
    }
  else if (uplo == MagmaUpper)
    {
    num = opencl::magma_real_kernel_id::laset_upper;
    }
  else
    {
    num = opencl::magma_real_kernel_id::laset_full;
    }

  magmablas_run_laset_kernel(num, uplo, m, n, offdiag, diag, dA, dA_offset, ldda, queue);
  }



template<typename eT>
inline
void
magmablas_run_laset_kernel
  (
  const opencl::magma_real_kernel_id::enum_id num,
  magma_uplo_t uplo,
  magma_int_t m,
  magma_int_t n,
  eT offdiag,
  eT diag,
  cl_mem dA,
  size_t dA_offset,
  magma_int_t ldda,
  magma_queue_t queue
  )
  {
  cl_int status;

  opencl::runtime_t::adapt_uword local_m(m);
  opencl::runtime_t::adapt_uword local_n(n);
  opencl::runtime_t::adapt_uword local_dA_offset(dA_offset);
  opencl::runtime_t::adapt_uword local_ldda(ldda);

  cl_kernel k = get_rt().cl_rt.get_kernel<eT>(num);

  status  = coot_wrapper(clSetKernelArg)(k, 0, local_m.size,         local_m.addr);
  status |= coot_wrapper(clSetKernelArg)(k, 1, local_n.size,         local_n.addr);
  status |= coot_wrapper(clSetKernelArg)(k, 2, sizeof(eT),           &offdiag);
  status |= coot_wrapper(clSetKernelArg)(k, 3, sizeof(eT),           &diag);
  status |= coot_wrapper(clSetKernelArg)(k, 4, sizeof(cl_mem),       &dA);
  status |= coot_wrapper(clSetKernelArg)(k, 5, local_dA_offset.size, local_dA_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(k, 6, local_ldda.size,      local_ldda.addr);
  coot_check_runtime_error(status, "coot::opencl::magmablas_run_laset_kernel(): couldn't set kernel arguments");

  size_t threads[2] = { MAGMABLAS_BLK_X,                     1                                   };
  size_t grid[2]    = { size_t(m - 1) / MAGMABLAS_BLK_X + 1, size_t(n - 1) / MAGMABLAS_BLK_Y + 1 };
  grid[0] *= threads[0];
  grid[1] *= threads[1];

  status |= coot_wrapper(clEnqueueNDRangeKernel)(queue, k, 2, NULL, grid, threads, 0, NULL, NULL);
  coot_check_runtime_error(status, "coot::opencl::magmablas_run_laset_kernel(): couldn't execute kernel");
  }



inline
void
magmablas_slaset_band(magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t k, float offdiag, float diag, magmaFloat_ptr dA, size_t dA_offset, magma_int_t ldda, magma_queue_t queue)
  {
  magmablas_laset_band<float>(uplo, m, n, k, offdiag, diag, (cl_mem) dA, dA_offset, ldda, queue);
  }



inline
void
magmablas_dlaset_band(magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t k, double offdiag, double diag, magmaDouble_ptr dA, size_t dA_offset, magma_int_t ldda, magma_queue_t queue)
  {
  magmablas_laset_band<double>(uplo, m, n, k, offdiag, diag, (cl_mem) dA, dA_offset, ldda, queue);
  }



template<typename eT>
inline
void
magmablas_laset_band(magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t k, eT offdiag, eT diag, cl_mem dA, size_t dA_offset, magma_int_t ldda, magma_queue_t queue)
  {
  cl_int status;

  opencl::runtime_t::adapt_uword local_m(m);
  opencl::runtime_t::adapt_uword local_n(n);
  opencl::runtime_t::adapt_uword local_k(k);
  opencl::runtime_t::adapt_uword local_dA_offset(dA_offset);
  opencl::runtime_t::adapt_uword local_ldda(ldda);

  opencl::magma_real_kernel_id::enum_id num;
  if (uplo == MagmaLower)
    {
    num = opencl::magma_real_kernel_id::laset_band_lower;
    }
  else
    {
    num = opencl::magma_real_kernel_id::laset_band_upper;
    }

  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT>(num);

  status  = coot_wrapper(clSetKernelArg)(kernel, 0, local_m.size,         local_m.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, local_n.size,         local_n.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, sizeof(eT),           &offdiag);
  status |= coot_wrapper(clSetKernelArg)(kernel, 3, sizeof(eT),           &diag);
  status |= coot_wrapper(clSetKernelArg)(kernel, 4, sizeof(cl_mem),       &dA);
  status |= coot_wrapper(clSetKernelArg)(kernel, 5, local_dA_offset.size, local_dA_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(kernel, 6, local_ldda.size,      local_ldda.addr);
  coot_check_runtime_error(status, "coot::opencl::magmablas_laset_band(): couldn't set kernel arguments");

  size_t threads;
  size_t grid;
  if (uplo == MagmaUpper)
    {
    threads = size_t( std::min(k, n) );
    grid = size_t( magma_ceildiv( std::min(m + k - 1, n), MAGMABLAS_LASET_BAND_NB ) );
    }
  else
    {
    threads = size_t( std::min(k, m) );
    grid = size_t( magma_ceildiv( std::min(m, n), MAGMABLAS_LASET_BAND_NB ) );
    }
  grid *= threads;

  status |= coot_wrapper(clEnqueueNDRangeKernel)(queue, kernel, 1, NULL, &grid, &threads, 0, NULL, NULL);
  coot_check_runtime_error(status, "coot::opencl::magmablas_laset_band(): couldn't execute kernel");
  }



inline
void
magmablas_stranspose
  (
  magma_int_t m,
  magma_int_t n,
  magmaFloat_const_ptr dA,
  size_t dA_offset,
  magma_int_t ldda,
  magmaFloat_ptr dAT,
  size_t dAT_offset,
  magma_int_t lddat,
  magma_queue_t queue
  )
  {
  magmablas_transpose<float>(m, n, dA, dA_offset, ldda, dAT, dAT_offset, lddat, queue);
  }



inline
void
magmablas_dtranspose
  (
  magma_int_t m,
  magma_int_t n,
  magmaDouble_const_ptr dA,
  size_t dA_offset,
  magma_int_t ldda,
  magmaDouble_ptr dAT,
  size_t dAT_offset,
  magma_int_t lddat,
  magma_queue_t queue
  )
  {
  magmablas_transpose<double>(m, n, dA, dA_offset, ldda, dAT, dAT_offset, lddat, queue);
  }



template<typename eT>
inline
void
magmablas_transpose
  (
  magma_int_t m,
  magma_int_t n,
  cl_mem dA,
  size_t dA_offset,
  magma_int_t ldda,
  cl_mem dAT,
  size_t dAT_offset,
  magma_int_t lddat,
  magma_queue_t queue
  )
  {
  magma_int_t info = 0;
  if ( m < 0 )
    info = -1;
  else if ( n < 0 )
    info = -2;
  else if ( ldda < m )
    info = -4;
  else if ( lddat < n )
    info = -6;

  if ( info != 0 )
    {
    //magma_xerbla( __func__, -(info) );
    return;  //info;
    }

  /* Quick return */
  if ( (m == 0) || (n == 0) )
    return;

  size_t threads[2] = { MAGMABLAS_TRANS_NX,                                      MAGMABLAS_TRANS_NY                                      };
  size_t grid[2] =    { size_t(m + MAGMABLAS_TRANS_NB - 1) / MAGMABLAS_TRANS_NB, size_t(n + MAGMABLAS_TRANS_NB - 1) / MAGMABLAS_TRANS_NB };
  grid[0] *= threads[0];
  grid[1] *= threads[1];

  cl_kernel k = get_rt().cl_rt.get_kernel<eT>(opencl::magma_real_kernel_id::transpose_magma);

  opencl::runtime_t::adapt_uword local_m(m);
  opencl::runtime_t::adapt_uword local_n(n);
  opencl::runtime_t::adapt_uword local_dA_offset(dA_offset);
  opencl::runtime_t::adapt_uword local_ldda(ldda);
  opencl::runtime_t::adapt_uword local_dAT_offset(dAT_offset);
  opencl::runtime_t::adapt_uword local_lddat(lddat);

  cl_int status;
  status  = coot_wrapper(clSetKernelArg)(k, 0, local_m.size,          local_m.addr);
  status |= coot_wrapper(clSetKernelArg)(k, 1, local_n.size,          local_n.addr);
  status |= coot_wrapper(clSetKernelArg)(k, 2, sizeof(cl_mem),        &dA);
  status |= coot_wrapper(clSetKernelArg)(k, 3, local_dA_offset.size,  local_dA_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(k, 4, local_ldda.size,       local_ldda.addr);
  status |= coot_wrapper(clSetKernelArg)(k, 5, sizeof(cl_mem),        &dAT);
  status |= coot_wrapper(clSetKernelArg)(k, 6, local_dAT_offset.size, local_dAT_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(k, 7, local_lddat.size,      local_lddat.addr);
  coot_check_runtime_error(status, "coot::opencl::magmablas_transpose(): couldn't set kernel arguments");

  status = coot_wrapper(clEnqueueNDRangeKernel)(queue, k, 2, NULL, grid, threads, 0, NULL, NULL);
  coot_check_runtime_error(status, "coot::opencl::magmablas_transpose(): couldn't execute kernel");
  }



inline
void
magmablas_stranspose_inplace
  (
  magma_int_t n,
  magmaFloat_ptr dA,
  size_t dA_offset,
  magma_int_t ldda,
  magma_queue_t queue
  )
  {
  magmablas_transpose_inplace<float>(n, dA, dA_offset, ldda, queue);
  }



inline
void
magmablas_dtranspose_inplace
  (
  magma_int_t n,
  magmaDouble_ptr dA,
  size_t dA_offset,
  magma_int_t ldda,
  magma_queue_t queue
  )
  {
  magmablas_transpose_inplace<double>(n, dA, dA_offset, ldda, queue);
  }



template<typename eT>
inline
void
magmablas_transpose_inplace
  (
  magma_int_t n,
  cl_mem dA,
  size_t dA_offset,
  magma_int_t ldda,
  magma_queue_t queue
  )
  {
  magma_int_t info = 0;
  if (n < 0)
    info = -1;
  else if (ldda < n)
    info = -3;

  if (info != 0)
    {
    //magma_xerbla( __func__, -(info) );
    return;  //info;
    }

  size_t threads[2] = { MAGMABLAS_TRANS_INPLACE_NB, MAGMABLAS_TRANS_INPLACE_NB };
  int nblock = (n + MAGMABLAS_TRANS_INPLACE_NB - 1) / MAGMABLAS_TRANS_INPLACE_NB;

  // need 1/2 * (nblock+1) * nblock to cover lower triangle and diagonal of matrix.
  // block assignment differs depending on whether nblock is odd or even.
  cl_kernel k;
  size_t grid[2];
  if (nblock % 2 == 1)
    {
    grid[0] = nblock             * threads[0];
    grid[1] = ((nblock + 1) / 2) * threads[1];
    k = get_rt().cl_rt.get_kernel<eT>(opencl::magma_real_kernel_id::transpose_inplace_odd_magma);
    }
  else
    {
    grid[0] = (nblock + 1)       * threads[0];
    grid[1] = (nblock / 2)       * threads[1];
    k = get_rt().cl_rt.get_kernel<eT>(opencl::magma_real_kernel_id::transpose_inplace_even_magma);
    }

  opencl::runtime_t::adapt_uword local_n(n);
  opencl::runtime_t::adapt_uword local_dA_offset(dA_offset);
  opencl::runtime_t::adapt_uword local_ldda(ldda);

  cl_int status;
  status  = coot_wrapper(clSetKernelArg)(k, 0, local_n.size,         local_n.addr);
  status |= coot_wrapper(clSetKernelArg)(k, 1, sizeof(cl_mem),       &dA);
  status |= coot_wrapper(clSetKernelArg)(k, 2, local_dA_offset.size, local_dA_offset.addr);
  status |= coot_wrapper(clSetKernelArg)(k, 3, local_ldda.size,      local_ldda.addr);
  coot_check_runtime_error(status, "coot::opencl::magmablas_transpose_inplace(): couldn't set kernel arguments");

  status = coot_wrapper(clEnqueueNDRangeKernel)(queue, k, 2, NULL, grid, threads, 0, NULL, NULL);
  coot_check_runtime_error(status, "coot::opencl::magmablas_transpose_inplace(): couldn't execute kernel");
  }



inline
void
magmablas_slascl
  (
  magma_type_t type,
  magma_int_t kl,
  magma_int_t ku,
  float cfrom,
  float cto,
  magma_int_t m,
  magma_int_t n,
  magmaFloat_ptr dA,
  size_t dA_offset,
  magma_int_t ldda,
  magma_queue_t queue,
  magma_int_t *info
  )
  {
  magmablas_lascl<float>(type, kl, ku, cfrom, cto, m, n, (cl_mem) dA, dA_offset, ldda, queue, info);
  }



inline
void
magmablas_dlascl
  (
  magma_type_t type,
  magma_int_t kl,
  magma_int_t ku,
  double cfrom,
  double cto,
  magma_int_t m,
  magma_int_t n,
  magmaDouble_ptr dA,
  size_t dA_offset,
  magma_int_t ldda,
  magma_queue_t queue,
  magma_int_t *info
  )
  {
  magmablas_lascl<double>(type, kl, ku, cfrom, cto, m, n, (cl_mem) dA, dA_offset, ldda, queue, info);
  }



template<typename eT>
inline
void
magmablas_lascl
  (
  magma_type_t type,
  magma_int_t kl,
  magma_int_t ku,
  eT cfrom,
  eT cto,
  magma_int_t m,
  magma_int_t n,
  cl_mem dA,
  size_t dA_offset,
  magma_int_t ldda,
  magma_queue_t queue,
  magma_int_t *info
  )
  {
  *info = 0;
  if ( type != MagmaLower && type != MagmaUpper && type != MagmaFull )
    {
    *info = -1;
    }
  else if ( cfrom == 0 || std::isnan(cfrom) )
    {
    *info = -4;
    }
  else if ( std::isnan(cto) )
    {
    *info = -5;
    }
  else if ( m < 0 )
    {
    *info = -6;
    }
  else if ( n < 0 )
    {
    *info = -3;
    }
  else if ( ldda < std::max(1,m) )
    {
    *info = -7;
    }

  if (*info != 0)
    {
    //magma_xerbla( __func__, -(*info) );
    return;  //info;
    }

  size_t threads = size_t( MAGMABLAS_LASCL_NB );
  size_t grid = size_t( magma_ceildiv( m, MAGMABLAS_LASCL_NB ) );
  grid *= threads;

  eT smlnum = 0, bignum = 0, cfromc = 0, ctoc = 0, cto1 = 0, cfrom1 = 0, mul = 0;
  magma_int_t done = false;

  cfromc = cfrom;
  ctoc   = cto;
  while( ! done )
    {
    cfrom1 = cfromc*smlnum;
    if ( cfrom1 == cfromc )
      {
      // cfromc is an inf.  Multiply by a correctly signed zero for
      // finite ctoc, or a nan if ctoc is infinite.
      mul  = ctoc / cfromc;
      done = true;
      cto1 = ctoc;
      }
    else
      {
      cto1 = ctoc / bignum;
      if ( cto1 == ctoc )
        {
        // ctoc is either 0 or an inf.  In both cases, ctoc itself
        // serves as the correct multiplication factor.
        mul  = ctoc;
        done = true;
        cfromc = 1;
        }
      else if ( std::abs(cfrom1) > std::abs(ctoc) && ctoc != 0 )
        {
        mul  = smlnum;
        done = false;
        cfromc = cfrom1;
        }
      else if ( std::abs(cto1) > std::abs(cfromc) )
        {
        mul  = bignum;
        done = false;
        ctoc = cto1;
        }
      else
        {
        mul  = ctoc / cfromc;
        done = true;
        }
      }

    cl_kernel k;
    if (type == MagmaLower)
      {
      k = get_rt().cl_rt.get_kernel<eT>(opencl::magma_real_kernel_id::lascl_lower);
      }
    else if (type == MagmaUpper)
      {
      k = get_rt().cl_rt.get_kernel<eT>(opencl::magma_real_kernel_id::lascl_upper);
      }
    else
      {
      k = get_rt().cl_rt.get_kernel<eT>(opencl::magma_real_kernel_id::lascl_full);
      }

    opencl::runtime_t::adapt_uword local_m(m);
    opencl::runtime_t::adapt_uword local_n(n);
    opencl::runtime_t::adapt_uword local_dA_offset(dA_offset);
    opencl::runtime_t::adapt_uword local_ldda(ldda);

    cl_int status;
    status  = coot_wrapper(clSetKernelArg)(k, 0, local_m.size,          local_m.addr);
    status |= coot_wrapper(clSetKernelArg)(k, 1, local_n.size,          local_n.addr);
    status |= coot_wrapper(clSetKernelArg)(k, 2, sizeof(eT),            &mul);
    status |= coot_wrapper(clSetKernelArg)(k, 3, sizeof(cl_mem),        &dA);
    status |= coot_wrapper(clSetKernelArg)(k, 4, local_dA_offset.size,  local_dA_offset.addr);
    status |= coot_wrapper(clSetKernelArg)(k, 5, local_ldda.size,       local_ldda.addr);
    coot_check_runtime_error(status, "coot::opencl::magmablas_lascl(): couldn't set kernel arguments");

    status = coot_wrapper(clEnqueueNDRangeKernel)( queue, k, 1, NULL, &grid, &threads, 0, NULL, NULL );
    coot_check_runtime_error(status, "coot::opencl::magmablas_lascl(): couldn't run kernel");
    }
  }




inline
float
magmablas_slansy(magma_norm_t norm, magma_uplo_t uplo, magma_int_t n, magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t ldda, magmaFloat_ptr dwork, size_t dwork_offset, magma_int_t lddwork, magma_queue_t queue)
  {
  coot_ignore(lddwork);
  const int i = magmablas_lansy<float>(norm, uplo, n, (cl_mem) dA, dA_offset, ldda, (cl_mem) dwork, dwork_offset, queue);
  float res;
  magma_sgetvector(1, dwork, dwork_offset + i, 1, &res, 1, queue);
  return res;
  }



inline
double
magmablas_dlansy(magma_norm_t norm, magma_uplo_t uplo, magma_int_t n, magmaDouble_const_ptr dA, size_t dA_offset, magma_int_t ldda, magmaDouble_ptr dwork, size_t dwork_offset, magma_int_t lddwork, magma_queue_t queue)
  {
  coot_ignore(lddwork);
  const int i = magmablas_lansy<double>(norm, uplo, n, (cl_mem) dA, dA_offset, ldda, (cl_mem) dwork, dwork_offset, queue);
  double res;
  magma_dgetvector(1, dwork, dwork_offset + i, 1, &res, 1, queue);
  return res;
  }



template<typename eT>
inline
int
magmablas_lansy(magma_norm_t norm, magma_uplo_t uplo, magma_int_t n, cl_mem dA, size_t dA_offset, magma_int_t ldda, cl_mem dwork, size_t dwork_offset, magma_queue_t queue)
  {
  magma_int_t info = 0;
  // 1-norm == inf-norm since A is symmetric
  bool inf_norm = (norm == MagmaInfNorm || norm == MagmaOneNorm);
  bool max_norm = (norm == MagmaMaxNorm);

  // inf_norm Double-Complex requires > 16 KB shared data (arch >= 200)
  const bool inf_implemented = true;

  if ( ! (max_norm || (inf_norm && inf_implemented)) )
    {
    info = -1;
    }
  else if ( uplo != MagmaUpper && uplo != MagmaLower )
    {
    info = -2;
    }
  else if ( n < 0 )
    {
    info = -3;
    }
  else if ( ldda < n )
    {
    info = -5;
    }

  if ( info != 0 )
    {
    //magma_xerbla( __func__, -(info) );
    return info;
    }

  /* Quick return */
  if ( n == 0 )
    {
    return 0;
    }

  if ( inf_norm )
    {
    magmablas_lansy_inf<eT>( uplo, n, dA, dA_offset, ldda, dwork, dwork_offset, queue );
    }
  else
    {
    magmablas_lansy_max<eT>( uplo, n, dA, dA_offset, ldda, dwork, dwork_offset, queue );
    }

  return magma_isamax( n, dwork, dwork_offset, 1, queue ) - 1;
  }



template<typename eT>
inline
void
magmablas_lansy_inf
  (
  magma_uplo_t uplo, int n,
  magmaFloat_const_ptr A, size_t A_offset, int lda,
  magmaFloat_ptr dwork, size_t dwork_offset,
  magma_queue_t queue
  )
  {
  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT>((uplo == MagmaLower) ? opencl::magma_real_kernel_id::lansy_inf_lower : opencl::magma_real_kernel_id::lansy_inf_upper);

  cl_int err = 0;

  int blocks = (n - 1) / MAGMABLAS_LANSY_INF_BS + 1;
  size_t grid[3] = { size_t(blocks), 1, 1 };
  size_t threads[3] = { MAGMABLAS_LANSY_INF_BS, 4, 1 };
  grid[0] *= threads[0];
  grid[1] *= threads[1];
  grid[2] *= threads[2];

  int n_full_block = (n - n % MAGMABLAS_LANSY_INF_BS) / MAGMABLAS_LANSY_INF_BS;
  int n_mod_bs = n % MAGMABLAS_LANSY_INF_BS;

  opencl::runtime_t::adapt_uword local_n(n);
  opencl::runtime_t::adapt_uword local_A_offset(A_offset);
  opencl::runtime_t::adapt_uword local_lda(lda);
  opencl::runtime_t::adapt_uword local_dwork_offset(dwork_offset);
  opencl::runtime_t::adapt_uword local_n_full_block(n_full_block);
  opencl::runtime_t::adapt_uword local_n_mod_bs(n_mod_bs);

  err  = coot_wrapper(clSetKernelArg)(kernel, 0, local_n.size,            local_n.addr);
  err |= coot_wrapper(clSetKernelArg)(kernel, 1, sizeof(cl_mem),          &A);
  err |= coot_wrapper(clSetKernelArg)(kernel, 2, local_A_offset.size,     local_A_offset.addr);
  err |= coot_wrapper(clSetKernelArg)(kernel, 3, local_lda.size,          local_lda.addr);
  err |= coot_wrapper(clSetKernelArg)(kernel, 4, sizeof(cl_mem),          &dwork);
  err |= coot_wrapper(clSetKernelArg)(kernel, 5, local_dwork_offset.size, local_dwork_offset.addr);
  err |= coot_wrapper(clSetKernelArg)(kernel, 6, local_n_full_block.size, local_n_full_block.addr);
  err |= coot_wrapper(clSetKernelArg)(kernel, 7, local_n_mod_bs.size,     local_n_mod_bs.addr);
  coot_check_runtime_error(err, "coot::opencl::magmablas_lansy_inf(): couldn't set kernel arguments");

  err = coot_wrapper(clEnqueueNDRangeKernel)( queue, kernel, 3, NULL, grid, threads, 0, NULL, NULL );
  coot_check_runtime_error(err, "coot::opencl::magmablas_lansy_inf(): couldn't run kernel");
  }



template<typename eT>
inline
void
magmablas_lansy_max
  (
  magma_uplo_t uplo, int n,
  magmaFloat_const_ptr A, size_t A_offset, int lda,
  magmaFloat_ptr dwork, size_t dwork_offset,
  magma_queue_t queue
  )
  {
  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT>((uplo == MagmaLower) ? opencl::magma_real_kernel_id::lansy_max_lower : opencl::magma_real_kernel_id::lansy_max_upper);

  cl_int err = 0;

  int blocks = (n - 1) / MAGMABLAS_LANSY_MAX_BS + 1;
  size_t grid[3] = { size_t(blocks), 1, 1 };
  size_t threads[3] = { MAGMABLAS_LANSY_MAX_BS, 1, 1 };
  grid[0] *= threads[0];
  grid[1] *= threads[1];
  grid[2] *= threads[2];

  opencl::runtime_t::adapt_uword local_n(n);
  opencl::runtime_t::adapt_uword local_A_offset(A_offset);
  opencl::runtime_t::adapt_uword local_lda(lda);
  opencl::runtime_t::adapt_uword local_dwork_offset(dwork_offset);

  err  = coot_wrapper(clSetKernelArg)(kernel, 0, local_n.size,            local_n.addr);
  err |= coot_wrapper(clSetKernelArg)(kernel, 1, sizeof(cl_mem),          &A);
  err |= coot_wrapper(clSetKernelArg)(kernel, 2, local_A_offset.size,     local_A_offset.addr);
  err |= coot_wrapper(clSetKernelArg)(kernel, 3, local_lda.size,          local_lda.addr);
  err |= coot_wrapper(clSetKernelArg)(kernel, 4, sizeof(cl_mem),          &dwork);
  err |= coot_wrapper(clSetKernelArg)(kernel, 5, local_dwork_offset.size, local_dwork_offset.addr);
  coot_check_runtime_error(err, "coot::opencl::magmablas_lansy_max(): couldn't set kernel arguments");

  err = coot_wrapper(clEnqueueNDRangeKernel)( queue, kernel, 3, NULL, grid, threads, 0, NULL, NULL );
  coot_check_runtime_error(err, "coot::opencl::magmablas_lansy_max(): couldn't run kernel");
  }
