// Copyright 2017 Conrad Sanderson (http://conradsanderson.id.au)
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
// clMAGMA 1.3 (2014-11-14) and/or MAGMA 2.2 (2016-11-20).
// clMAGMA 1.3 and MAGMA 2.2 are distributed under a
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



// Forward declaration required for some functionality.
namespace opencl
  {
  template<typename eT>
  inline
  void
  fill(dev_mem_t<eT> dest,
       const eT val,
       const uword n_rows,
       const uword n_cols,
       const uword row_offset,
       const uword col_offset,
       const uword M_n_rows);
  }



inline
void
check_error(const cl_int error_code)
  {
  opencl::coot_check_cl_error(error_code, "magma_function");
  }



#if defined(COOT_USE_CLBLAST)
inline
void
check_clblast_error(const CLBlastStatusCode error_code)
  {
  opencl::coot_check_clblast_error(error_code, "magma_function");
  }
#endif



#if defined(COOT_USE_CLBLAS)
inline
void
check_clblas_error(const cl_int error_code)
  {
  opencl::coot_check_clblas_error(error_code, "magma_function");
  }
#endif



// TODO: refactor code to avoid using the following global variable
// TODO: the code below seems to only write to get_g_event(); it's never read, meaning it's not used for waiting
// This is stuffed into a singleton for now to avoid linking issues.
inline magma_event_t* get_g_event()
  {
  static magma_event_t* g_event;
  return g_event;
  }



/////////////////////
// CS: adaptations for OpenCL

inline
magma_int_t
magma_malloc_cpu( void** ptrPtr, size_t size )
{
    if ( size == 0 )  { size = 16; }

    *ptrPtr = malloc( size );
    if ( *ptrPtr == NULL ) {
        return MAGMA_ERR_HOST_ALLOC;
    }
    return MAGMA_SUCCESS;
}


//inline magma_int_t magma_dmalloc_pinned( double **ptr_ptr, size_t n ) { return magma_malloc_pinned( (void**) ptr_ptr, n*sizeof(double)             ); }
inline magma_int_t magma_dmalloc_pinned( double **ptr_ptr, size_t n ) { return magma_malloc_cpu( (void**) ptr_ptr, n*sizeof(double) ); }
inline magma_int_t magma_smalloc_pinned( float  **ptr_ptr, size_t n ) { return magma_malloc_cpu( (void**) ptr_ptr, n*sizeof(float)  ); }
inline magma_int_t magma_imalloc_pinned( int    **ptr_ptr, size_t n ) { return magma_malloc_cpu( (void**) ptr_ptr, n*sizeof(int)  ); }

inline magma_int_t magma_free_pinned( void* ptr )  { free( ptr ); return MAGMA_SUCCESS; }

inline magma_int_t magma_dmalloc_cpu( double** ptr_ptr, size_t n ) { return magma_malloc_cpu( (void**) ptr_ptr, n*sizeof(double) ); }
inline magma_int_t magma_smalloc_cpu( float**  ptr_ptr, size_t n ) { return magma_malloc_cpu( (void**) ptr_ptr, n*sizeof(float)  ); }
inline magma_int_t magma_imalloc_cpu( int**    ptr_ptr, size_t n ) { return magma_malloc_cpu( (void**) ptr_ptr, n*sizeof(int)    ); }

inline magma_int_t magma_free_cpu( void* ptr ) { free( ptr ); return MAGMA_SUCCESS; }



inline
magma_int_t
magma_malloc( magma_ptr* ptr_ptr, size_t size )
  {
  // malloc and free sometimes don't work for size=0, so allocate some minimal size
  if ( size == 0 ) { size = 16; }

  cl_int err;
  *ptr_ptr = coot_wrapper(clCreateBuffer)( get_rt().cl_rt.get_context(), CL_MEM_READ_WRITE, size, NULL, &err );

  if ( err != CL_SUCCESS )
    {
    return MAGMA_ERR_DEVICE_ALLOC;
    }
  return MAGMA_SUCCESS;
  }



inline
magma_int_t
magma_free( magma_ptr ptr )
  {
  cl_int err = coot_wrapper(clReleaseMemObject)( ptr );
  if ( err != CL_SUCCESS )
    {
    return MAGMA_ERR_INVALID_PTR;
    }
  return MAGMA_SUCCESS;
  }



inline magma_int_t magma_dmalloc( magmaDouble_ptr* ptr_ptr, size_t n ) { return magma_malloc( (magma_ptr*) ptr_ptr, n*sizeof(double) ); }
inline magma_int_t magma_smalloc( magmaFloat_ptr*  ptr_ptr, size_t n ) { return magma_malloc( (magma_ptr*) ptr_ptr, n*sizeof(float)  ); }



inline
magma_queue_t
magma_queue_create()
  {
  magma_queue_t result;
  get_rt().cl_rt.create_extra_cq(result);
  return result;
  }



inline
void
magma_queue_destroy(magma_queue_t queue)
  {
  get_rt().cl_rt.delete_extra_cq(queue);
  }



//
// double

inline
void
magma_dgetmatrix(magma_int_t m, magma_int_t n, magmaDouble_const_ptr dA_src, size_t dA_offset, magma_int_t ldda, double* hB_dst, magma_int_t ldhb, magma_queue_t queue)
  {
  if (m <= 0 || n <= 0)  { return; }

  size_t buffer_origin[3] = { dA_offset*sizeof(double), 0, 0 };
  size_t host_orig[3]     = { 0, 0, 0 };
  size_t region[3]        = { m*sizeof(double), size_t(n), 1 };

  cl_int err = coot_wrapper(clEnqueueReadBufferRect)(
      queue, dA_src, CL_TRUE,  // blocking
      buffer_origin, host_orig, region,
      ldda*sizeof(double), 0,
      ldhb*sizeof(double), 0,
      hB_dst, 0, NULL, get_g_event() );

  // TODO: get_g_event() can probably be replaced with NULL; check if any other function required by magma_dpotrf_gpu accesses get_g_event()
  // TODO: as the call is blocking, why use get_g_event() in the first place?

  // OpenCL 1.2 docs for clEnqueueReadBufferRect()
  // event
  // Returns an event object that identifies this particular read command and can be used to query or queue a wait for this particular command to complete.
  // event can be NULL in which case it will not be possible for the application to query the status of this command or queue a wait for this command to complete.
  // If the event_wait_list and the event arguments are not NULL, the event argument should not refer to an element of the event_wait_list array.

  check_error( err );  // TODO: replace check_error() with corresponding bandicoot function
  }



inline
void
magma_dsetmatrix(magma_int_t m, magma_int_t n, double* hA_src, magma_int_t ldha, magmaDouble_ptr dB_dst, size_t dB_offset, magma_int_t lddb, magma_queue_t queue)
  {
  if (m <= 0 || n <= 0)  { return; }

  size_t buffer_origin[3] = { dB_offset*sizeof(double), 0, 0 };
  size_t host_orig[3]     = { 0, 0, 0 };
  size_t region[3]        = { m*sizeof(double), size_t(n), 1 };

  cl_int err = coot_wrapper(clEnqueueWriteBufferRect)(
      queue, dB_dst, CL_TRUE,  // blocking
      buffer_origin, host_orig, region,
      lddb*sizeof(double), 0,
      ldha*sizeof(double), 0,
      hA_src, 0, NULL, get_g_event() );

  // TODO: get_g_event() can probably be replaced with NULL; see note above
  // TODO: as the call is blocking, why use get_g_event() in the first place?

  // OpenCL 1.2 docs for clEnqueueWriteBufferRect()
  // event
  //
  // Returns an event object that identifies this particular write command and can be used to query or queue a wait for this particular command to complete.
  // event can be NULL in which case it will not be possible for the application to query the status of this command or queue a wait for this command to complete.

  check_error( err );
  }



inline
void
magma_dgetvector(magma_int_t n,
                 magmaDouble_const_ptr dx_src, size_t dx_offset, magma_int_t incx,
                 double* hy_dst,               magma_int_t incy,
                 magma_queue_t queue)
  {
  magma_dgetmatrix(1, n, dx_src, dx_offset, incx, hy_dst, incy, queue);
  }



inline
void
magma_dsetvector(magma_int_t m,
                 double* hx_src,                           magma_int_t incx,
                 magmaDouble_ptr dy_dst, size_t dy_offset, magma_int_t incy,
                 magma_queue_t queue)
  {
  magma_dsetmatrix(1, m, hx_src, incx, dy_dst, dy_offset, incy, queue);
  }



//
// float

inline
void
magma_sgetmatrix(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA_src, size_t dA_offset, magma_int_t ldda,
    float*          hB_dst,                   magma_int_t ldhb,
    magma_queue_t queue )
{
    if (m <= 0 || n <= 0)
       return;

    size_t buffer_origin[3] = { dA_offset*sizeof(float), 0, 0 };
    size_t host_orig[3]     = { 0, 0, 0 };
    size_t region[3]        = { m*sizeof(float), size_t(n), 1 };
    cl_int err = coot_wrapper(clEnqueueReadBufferRect)(
        queue, dA_src, CL_TRUE,  // blocking
        buffer_origin, host_orig, region,
        ldda*sizeof(float), 0,
        ldhb*sizeof(float), 0,
        hB_dst, 0, NULL, get_g_event() );
    check_error( err );
}



inline
void
magma_ssetmatrix(
    magma_int_t m, magma_int_t n,
    float* hA_src,                              magma_int_t ldha,
    magmaFloat_ptr    dB_dst, size_t dB_offset, magma_int_t lddb,
    magma_queue_t queue )
{
    if (m <= 0 || n <= 0)
        return;

    size_t buffer_origin[3] = { dB_offset*sizeof(float), 0, 0 };
    size_t host_orig[3]     = { 0, 0, 0 };
    size_t region[3]        = { m*sizeof(float), size_t(n), 1 };
    cl_int err = coot_wrapper(clEnqueueWriteBufferRect)(
        queue, dB_dst, CL_TRUE,  // blocking
        buffer_origin, host_orig, region,
        lddb*sizeof(float), 0,
        ldha*sizeof(float), 0,
        hA_src, 0, NULL, get_g_event() );
    check_error( err );
}



inline
void
magma_sgetvector(magma_int_t n,
                 magmaFloat_const_ptr dx_src, size_t dx_offset, magma_int_t incx,
                 float* hy_dst,               magma_int_t incy,
                 magma_queue_t queue)
  {
  magma_sgetmatrix(1, n, dx_src, dx_offset, incx, hy_dst, incy, queue);
  }



inline
void
magma_ssetvector(magma_int_t m,
                 float* hx_src,                           magma_int_t incx,
                 magmaFloat_ptr dy_dst, size_t dy_offset, magma_int_t incy,
                 magma_queue_t queue)
  {
  magma_ssetmatrix(1, m, hx_src, incx, dy_dst, dy_offset, incy, queue);
  }



//
// double
//



inline
void
magma_dgetmatrix_async(magma_int_t m, magma_int_t n, magmaDouble_const_ptr dA_src, size_t dA_offset, magma_int_t ldda, double* hB_dst, magma_int_t ldhb, magma_queue_t queue, magma_event_t *event)
  {
  if (m <= 0 || n <= 0)  { return; }

  size_t buffer_origin[3] = { dA_offset*sizeof(double), 0, 0 };
  size_t host_orig[3]     = { 0, 0, 0 };
  size_t region[3]        = { m*sizeof(double), size_t(n), 1 };

  cl_int err = coot_wrapper(clEnqueueReadBufferRect)(
      queue, dA_src, CL_FALSE,  // non-blocking
      buffer_origin, host_orig, region,
      ldda*sizeof(double), 0,
      ldhb*sizeof(double), 0,
      hB_dst, 0, NULL, event );

  coot_wrapper(clFlush)(queue);
  check_error( err );
  }



inline
void
magma_dgetmatrix_async(magma_int_t m, magma_int_t n, magmaDouble_const_ptr dA_src, size_t dA_offset, magma_int_t ldda, double* hB_dst, magma_int_t ldhb, magma_queue_t queue)
  {
  if (m <= 0 || n <= 0)  { return; }

  size_t buffer_origin[3] = { dA_offset*sizeof(double), 0, 0 };
  size_t host_orig[3]     = { 0, 0, 0 };
  size_t region[3]        = { m*sizeof(double), size_t(n), 1 };

  cl_int err = coot_wrapper(clEnqueueReadBufferRect)(
      queue, dA_src, CL_FALSE,  // non-blocking
      buffer_origin, host_orig, region,
      ldda*sizeof(double), 0,
      ldhb*sizeof(double), 0,
      hB_dst, 0, NULL, NULL );

  coot_wrapper(clFlush)(queue);
  check_error( err );
  }



inline
void
magma_dsetmatrix_async(magma_int_t m, magma_int_t n, double* hA_src, magma_int_t ldha, magmaDouble_ptr dB_dst, size_t dB_offset, magma_int_t lddb, magma_queue_t queue, magma_event_t *event)
  {
  if (m <= 0 || n <= 0)  { return; }

  size_t buffer_origin[3] = { dB_offset*sizeof(double), 0, 0 };
  size_t host_orig[3]     = { 0, 0, 0 };
  size_t region[3]        = { m*sizeof(double), size_t(n), 1 };

  cl_int err = coot_wrapper(clEnqueueWriteBufferRect)(
      queue, dB_dst, CL_FALSE,  // non-blocking
      buffer_origin, host_orig, region,
      lddb*sizeof(double), 0,
      ldha*sizeof(double), 0,
      hA_src, 0, NULL, event );

  coot_wrapper(clFlush)(queue);
  check_error( err );
  }



inline
void
magma_dsetmatrix_async(magma_int_t m, magma_int_t n, double* hA_src, magma_int_t ldha, magmaDouble_ptr dB_dst, size_t dB_offset, magma_int_t lddb, magma_queue_t queue)
  {
  if (m <= 0 || n <= 0)  { return; }

  size_t buffer_origin[3] = { dB_offset*sizeof(double), 0, 0 };
  size_t host_orig[3]     = { 0, 0, 0 };
  size_t region[3]        = { m*sizeof(double), size_t(n), 1 };

  cl_int err = coot_wrapper(clEnqueueWriteBufferRect)(
      queue, dB_dst, CL_FALSE,  // non-blocking
      buffer_origin, host_orig, region,
      lddb*sizeof(double), 0,
      ldha*sizeof(double), 0,
      hA_src, 0, NULL, NULL );

  coot_wrapper(clFlush)(queue);
  check_error( err );
  }



inline
void
magma_dsetvector_async
  (
  magma_int_t n,
  double          *hx_src,                   magma_int_t incx,
  magmaDouble_ptr  dy_dst, size_t dy_offset, magma_int_t incy,
  magma_queue_t queue
  )
  {
  magma_dsetmatrix(1, n, hx_src, incx, dy_dst, dy_offset, incy, queue);
  }



inline
void
magma_dgetvector_async
  (
  magma_int_t n,
  magmaDouble_const_ptr  dx_src, size_t dx_offset, magma_int_t incx,
  double                *hy_dst,                   magma_int_t incy,
  magma_queue_t queue,
  const char* /* func */, const char* /* file */, int /* line */
  )
  {
  magma_dgetmatrix_async(1, n, dx_src, dx_offset, incx, hy_dst, incy, queue);
  }



inline
void
magma_dcopymatrix
  (
  magma_int_t m, magma_int_t n,
  magmaDouble_const_ptr dA_src, size_t dA_offset, magma_int_t ldda,
  magmaDouble_ptr       dB_dst, size_t dB_offset, magma_int_t lddb,
  magma_queue_t queue
  )
  {
  if (m <= 0 || n <= 0)
    {
    return;
    }

  size_t src_origin[3] = { dA_offset*sizeof(double), 0, 0 };
  size_t dst_orig[3]   = { dB_offset*sizeof(double), 0, 0 };
  size_t region[3]     = { m*sizeof(double), size_t(n), 1 };
  cl_int err = coot_wrapper(clEnqueueCopyBufferRect)(
      queue, dA_src, dB_dst,
      src_origin, dst_orig, region,
      ldda*sizeof(double), 0,
      lddb*sizeof(double), 0,
      0, NULL, NULL );
  check_error( err );
  }



//
// float
//



inline
void
magma_sgetmatrix_async(magma_int_t m, magma_int_t n, magmaFloat_const_ptr dA_src, size_t dA_offset, magma_int_t ldda, float* hB_dst, magma_int_t ldhb, magma_queue_t queue, magma_event_t *event)
  {
  if (m <= 0 || n <= 0)  { return; }

  size_t buffer_origin[3] = { dA_offset*sizeof(float), 0, 0 };
  size_t host_orig[3]     = { 0, 0, 0 };
  size_t region[3]        = { m*sizeof(float), size_t(n), 1 };

  cl_int err = coot_wrapper(clEnqueueReadBufferRect)(
      queue, dA_src, CL_FALSE,  // non-blocking
      buffer_origin, host_orig, region,
      ldda*sizeof(float), 0,
      ldhb*sizeof(float), 0,
      hB_dst, 0, NULL, event );

  coot_wrapper(clFlush)(queue);
  check_error( err );
  }



inline
void
magma_sgetmatrix_async(magma_int_t m, magma_int_t n, magmaFloat_const_ptr dA_src, size_t dA_offset, magma_int_t ldda, float* hB_dst, magma_int_t ldhb, magma_queue_t queue)
  {
  if (m <= 0 || n <= 0)  { return; }

  size_t buffer_origin[3] = { dA_offset*sizeof(float), 0, 0 };
  size_t host_orig[3]     = { 0, 0, 0 };
  size_t region[3]        = { m*sizeof(float), size_t(n), 1 };

  cl_int err = coot_wrapper(clEnqueueReadBufferRect)(
      queue, dA_src, CL_FALSE,  // non-blocking
      buffer_origin, host_orig, region,
      ldda*sizeof(float), 0,
      ldhb*sizeof(float), 0,
      hB_dst, 0, NULL, NULL );

  coot_wrapper(clFlush)(queue);
  check_error( err );
  }



inline
void
magma_ssetmatrix_async(magma_int_t m, magma_int_t n, float* hA_src, magma_int_t ldha, magmaFloat_ptr dB_dst, size_t dB_offset, magma_int_t lddb, magma_queue_t queue, magma_event_t *event)
  {
  if (m <= 0 || n <= 0)  { return; }

  size_t buffer_origin[3] = { dB_offset*sizeof(float), 0, 0 };
  size_t host_orig[3]     = { 0, 0, 0 };
  size_t region[3]        = { m*sizeof(float), size_t(n), 1 };

  cl_int err = coot_wrapper(clEnqueueWriteBufferRect)(
      queue, dB_dst, CL_FALSE,  // non-blocking
      buffer_origin, host_orig, region,
      lddb*sizeof(float), 0,
      ldha*sizeof(float), 0,
      hA_src, 0, NULL, event );

  coot_wrapper(clFlush)(queue);
  check_error( err );
  }


inline
void
magma_ssetmatrix_async(magma_int_t m, magma_int_t n, float* hA_src, magma_int_t ldha, magmaFloat_ptr dB_dst, size_t dB_offset, magma_int_t lddb, magma_queue_t queue)
  {
  if (m <= 0 || n <= 0)  { return; }

  size_t buffer_origin[3] = { dB_offset*sizeof(float), 0, 0 };
  size_t host_orig[3]     = { 0, 0, 0 };
  size_t region[3]        = { m*sizeof(float), size_t(n), 1 };

  cl_int err = coot_wrapper(clEnqueueWriteBufferRect)(
      queue, dB_dst, CL_FALSE,  // non-blocking
      buffer_origin, host_orig, region,
      lddb*sizeof(float), 0,
      ldha*sizeof(float), 0,
      hA_src, 0, NULL, NULL );

  coot_wrapper(clFlush)(queue);
  check_error( err );
  }



inline
void
magma_ssetvector_async
  (
  magma_int_t n,
  float          *hx_src,                   magma_int_t incx,
  magmaFloat_ptr  dy_dst, size_t dy_offset, magma_int_t incy,
  magma_queue_t queue
  )
  {
  magma_ssetmatrix(1, n, hx_src, incx, dy_dst, dy_offset, incy, queue);
  }



inline
void
magma_sgetvector_async
  (
  magma_int_t n,
  magmaFloat_const_ptr  dx_src, size_t dx_offset, magma_int_t incx,
  float                *hy_dst,                   magma_int_t incy,
  magma_queue_t queue,
  const char* /* func */, const char* /* file */, int /* line */
  )
  {
  magma_sgetmatrix_async(1, n, dx_src, dx_offset, incx, hy_dst, incy, queue);
  }



inline
void
magma_scopymatrix
  (
  magma_int_t m, magma_int_t n,
  magmaFloat_const_ptr dA_src, size_t dA_offset, magma_int_t ldda,
  magmaFloat_ptr       dB_dst, size_t dB_offset, magma_int_t lddb,
  magma_queue_t queue
  )
  {
  if (m <= 0 || n <= 0)
    {
    return;
    }

  size_t src_origin[3] = { dA_offset*sizeof(float), 0, 0 };
  size_t dst_orig[3]   = { dB_offset*sizeof(float), 0, 0 };
  size_t region[3]     = { m*sizeof(float), size_t(n), 1 };
  cl_int err = coot_wrapper(clEnqueueCopyBufferRect)(
      queue, dA_src, dB_dst,
      src_origin, dst_orig, region,
      ldda*sizeof(float), 0,
      lddb*sizeof(float), 0,
      0, NULL, NULL );
  check_error( err );
  }



// This deals with a subtle bug with returning lwork as a float.
// If lwork > 2**24, then it will get rounded as a float;
// we need to ensure it is rounded up instead of down,
// by multiplying by 1.+eps in Double precision:
inline
double
magma_dmake_lwork( magma_int_t lwork )
  {
  return double(lwork);
  }



inline
float
magma_smake_lwork( magma_int_t lwork )
  {
  double one_eps = 1. + Datum<double>::eps;
  return float(lwork * one_eps);
  }



////////////////////////////////////////////////////////////////////


inline
magma_int_t
magma_queue_sync( magma_queue_t queue )
  {
  cl_int err = coot_wrapper(clFinish)( queue );
  coot_wrapper(clFlush)( queue );
  check_error( err );
  return err;
  }


///////////////////////////
// LAPACK interface related

// TODO: what a horror
inline const char** get_magma2lapack_constants()
  {
  static const char *magma2lapack_constants[] =
    {
    "No",                                    //  0: MagmaFalse
    "Yes",                                   //  1: MagmaTrue (zlatrs)
    "", "", "", "", "", "", "", "", "",      //  2-10
    "", "", "", "", "", "", "", "", "", "",  // 11-20
    "", "", "", "", "", "", "", "", "", "",  // 21-30
    "", "", "", "", "", "", "", "", "", "",  // 31-40
    "", "", "", "", "", "", "", "", "", "",  // 41-50
    "", "", "", "", "", "", "", "", "", "",  // 51-60
    "", "", "", "", "", "", "", "", "", "",  // 61-70
    "", "", "", "", "", "", "", "", "", "",  // 71-80
    "", "", "", "", "", "", "", "", "", "",  // 81-90
    "", "", "", "", "", "", "", "", "", "",  // 91-100
    "Row",                                   // 101: MagmaRowMajor
    "Column",                                // 102: MagmaColMajor
    "", "", "", "", "", "", "", "",          // 103-110
    "No transpose",                          // 111: MagmaNoTrans
    "Transpose",                             // 112: MagmaTrans
    "Conjugate transpose",                   // 113: MagmaConjTrans
    "", "", "", "", "", "", "",              // 114-120
    "Upper",                                 // 121: MagmaUpper
    "Lower",                                 // 122: MagmaLower
    "General",                               // 123: MagmaFull; see lascl for "G"
    "", "", "", "", "", "", "",              // 124-130
    "Non-unit",                              // 131: MagmaNonUnit
    "Unit",                                  // 132: MagmaUnit
    "", "", "", "", "", "", "", "",          // 133-140
    "Left",                                  // 141: MagmaLeft
    "Right",                                 // 142: MagmaRight
    "Both",                                  // 143: MagmaBothSides (dtrevc)
    "", "", "", "", "", "", "",              // 144-150
    "", "", "", "", "", "", "", "", "", "",  // 151-160
    "", "", "", "", "", "", "", "", "", "",  // 161-170
    "1 norm",                                // 171: MagmaOneNorm
    "",                                      // 172: MagmaRealOneNorm
    "2 norm",                                // 173: MagmaTwoNorm
    "Frobenius norm",                        // 174: MagmaFrobeniusNorm
    "Infinity norm",                         // 175: MagmaInfNorm
    "",                                      // 176: MagmaRealInfNorm
    "Maximum norm",                          // 177: MagmaMaxNorm
    "",                                      // 178: MagmaRealMaxNorm
    "", "",                                  // 179-180
    "", "", "", "", "", "", "", "", "", "",  // 181-190
    "", "", "", "", "", "", "", "", "", "",  // 191-200
    "Uniform",                               // 201: MagmaDistUniform
    "Symmetric",                             // 202: MagmaDistSymmetric
    "Normal",                                // 203: MagmaDistNormal
    "", "", "", "", "", "", "",              // 204-210
    "", "", "", "", "", "", "", "", "", "",  // 211-220
    "", "", "", "", "", "", "", "", "", "",  // 221-230
    "", "", "", "", "", "", "", "", "", "",  // 231-240
    "Hermitian",                             // 241 MagmaHermGeev
    "Positive ev Hermitian",                 // 242 MagmaHermPoev
    "NonSymmetric pos sv",                   // 243 MagmaNonsymPosv
    "Symmetric pos sv",                      // 244 MagmaSymPosv
    "", "", "", "", "", "",                  // 245-250
    "", "", "", "", "", "", "", "", "", "",  // 251-260
    "", "", "", "", "", "", "", "", "", "",  // 261-270
    "", "", "", "", "", "", "", "", "", "",  // 271-280
    "", "", "", "", "", "", "", "", "", "",  // 281-290
    "No Packing",                            // 291 MagmaNoPacking
    "U zero out subdiag",                    // 292 MagmaPackSubdiag
    "L zero out superdiag",                  // 293 MagmaPackSupdiag
    "C",                                     // 294 MagmaPackColumn
    "R",                                     // 295 MagmaPackRow
    "B",                                     // 296 MagmaPackLowerBand
    "Q",                                     // 297 MagmaPackUpeprBand
    "Z",                                     // 298 MagmaPackAll
    "", "",                                  // 299-300
    "No vectors",                            // 301 MagmaNoVec
    "Vectors needed",                        // 302 MagmaVec
    "I",                                     // 303 MagmaIVec
    "All",                                   // 304 MagmaAllVec
    "Some",                                  // 305 MagmaSomeVec
    "Overwrite",                             // 306 MagmaOverwriteVec
    "", "", "", "",                          // 307-310
    "All",                                   // 311 MagmaRangeAll
    "V",                                     // 312 MagmaRangeV
    "I",                                     // 313 MagmaRangeI
    "", "", "", "", "", "", "",              // 314-320
    "",                                      // 321
    "Q",                                     // 322
    "P",                                     // 323
    "", "", "", "", "", "", "",              // 324-330
    "", "", "", "", "", "", "", "", "", "",  // 331-340
    "", "", "", "", "", "", "", "", "", "",  // 341-350
    "", "", "", "", "", "", "", "", "", "",  // 351-360
    "", "", "", "", "", "", "", "", "", "",  // 361-370
    "", "", "", "", "", "", "", "", "", "",  // 371-380
    "", "", "", "", "", "", "", "", "", "",  // 381-390
    "Forward",                               // 391: MagmaForward
    "Backward",                              // 392: MagmaBackward
    "", "", "", "", "", "", "", "",          // 393-400
    "Columnwise",                            // 401: MagmaColumnwise
    "Rowwise",                               // 402: MagmaRowwise
    "", "", "", "", "", "", "", ""           // 403-410
    // Remember to add a comma!
    };

  return magma2lapack_constants;
  }



inline
const char* lapack_trans_const( magma_trans_t magma_const )
  {
  assert( magma_const >= MagmaNoTrans   );
  assert( magma_const <= MagmaConjTrans );
  return get_magma2lapack_constants()[ magma_const ];
  }



inline
const char* lapack_uplo_const ( magma_uplo_t magma_const )
  {
  assert( magma_const >= MagmaUpper );
  assert( magma_const <= MagmaFull  );
  return get_magma2lapack_constants()[ magma_const ];
  }



inline
const char* lapack_diag_const ( magma_diag_t magma_const )
  {
  assert( magma_const >= MagmaNonUnit );
  assert( magma_const <= MagmaUnit    );
  return get_magma2lapack_constants()[ magma_const ];
  }



inline
const char* lapack_side_const ( magma_side_t magma_const )
  {
  assert( magma_const >= MagmaLeft  );
  assert( magma_const <= MagmaBothSides );
  return get_magma2lapack_constants()[ magma_const ];
  }



inline
const char* lapack_vec_const   ( magma_vec_t    magma_const )
  {
  assert( magma_const >= MagmaNoVec );
  assert( magma_const <= MagmaOverwriteVec );
  return get_magma2lapack_constants()[ magma_const ];
  }


//////////////////////////////
// clBLAS and CLBlast wrappers
//////////////////////////////



#if defined(COOT_USE_CLBLAST)
inline
CLBlastTriangle
clblast_uplo_const( magma_uplo_t magma_const )
  {
  assert( magma_const >= MagmaUpper );
  assert( magma_const <= MagmaLower );

  switch (magma_const)
    {
    case MagmaUpper:  return CLBlastTriangleUpper;
    case MagmaLower:  return CLBlastTriangleLower;
    default:          return CLBlastTriangleUpper;
    }
  }


inline
CLBlastTranspose
clblast_trans_const( magma_trans_t magma_const )
  {
  assert( magma_const >= MagmaNoTrans   );
  assert( magma_const <= MagmaConjTrans );

  switch (magma_const)
    {
    case MagmaNoTrans:    return CLBlastTransposeNo;
    case MagmaTrans:      return CLBlastTransposeYes;
    case MagmaConjTrans:  return CLBlastTransposeConjugate;
    default:              return CLBlastTransposeNo;
    }
  }



inline
CLBlastSide
clblast_side_const( magma_side_t magma_const )
  {
  assert( magma_const >= MagmaLeft  );
  assert( magma_const <= MagmaRight );

  switch (magma_const)
    {
    case MagmaLeft:   return CLBlastSideLeft;
    case MagmaRight:  return CLBlastSideRight;
    default:          return CLBlastSideLeft;
    }
  }



inline
CLBlastDiagonal
clblast_diag_const( magma_diag_t magma_const )
  {
  assert( magma_const >= MagmaNonUnit );
  assert( magma_const <= MagmaUnit    );

  switch (magma_const)
    {
    case MagmaNonUnit:  return CLBlastDiagonalNonUnit;
    case MagmaUnit:     return CLBlastDiagonalUnit;
    default:            return CLBlastDiagonalNonUnit;
    }
  }
#endif



#if defined(COOT_USE_CLBLAS)
inline
clblasUplo
clblas_uplo_const ( magma_uplo_t magma_const )
  {
  assert( magma_const >= MagmaUpper );
  assert( magma_const <= MagmaLower );

  switch (magma_const)
    {
    case MagmaUpper:  return clblasUpper;
    case MagmaLower:  return clblasLower;
    default:          return clblasUpper;
    }
  }



inline
clblasTranspose
clblas_trans_const( magma_trans_t magma_const )
  {
  assert( magma_const >= MagmaNoTrans   );
  assert( magma_const <= MagmaConjTrans );

  switch (magma_const)
    {
    case MagmaNoTrans:    return clblasNoTrans;
    case MagmaTrans:      return clblasTrans;
    case MagmaConjTrans:  return clblasConjTrans;
    default:              return clblasNoTrans;
    }
  }



inline
clblasSide
clblas_side_const ( magma_side_t magma_const )
  {
  assert( magma_const >= MagmaLeft  );
  assert( magma_const <= MagmaRight );

  switch (magma_const)
    {
    case MagmaLeft:   return clblasLeft;
    case MagmaRight:  return clblasRight;
    default:          return clblasLeft;
    }
  }



inline
clblasDiag
clblas_diag_const ( magma_diag_t magma_const )
  {
  assert( magma_const >= MagmaNonUnit );
  assert( magma_const <= MagmaUnit    );

  switch (magma_const)
    {
    case MagmaNonUnit:  return clblasNonUnit;
    case MagmaUnit:     return clblasUnit;
    default:            return clblasNonUnit;
    }
  }
#endif



//
// gemm

inline
void
magma_dgemm
  (
  magma_trans_t transA, magma_trans_t transB,
  magma_int_t m, magma_int_t n, magma_int_t k,
  double alpha,
  magmaDouble_const_ptr dA, size_t dA_offset, magma_int_t ldda,
  magmaDouble_const_ptr dB, size_t dB_offset, magma_int_t lddb,
  double beta,
  magmaDouble_ptr       dC, size_t dC_offset, magma_int_t lddc,
  magma_queue_t queue
  )
  {
  if ( m <= 0 || n <= 0 || k <= 0 )  { return; }

  #if defined(COOT_USE_CLBLAST)
    {
    if (beta == 0.0)
      {
      // paranoia: CLBlast doesn't seem to like it when C is not initialized to 0
      dev_mem_t<double> dC_mem;
      dC_mem.cl_mem_ptr = coot_cl_mem({ dC, 0 });
      opencl::fill(dC_mem, 0.0, m, n, dC_offset, 0, lddc);
      }

    CLBlastStatusCode err = coot_wrapper(CLBlastDgemm)(
        CLBlastLayoutColMajor,
        clblast_trans_const( transA ),
        clblast_trans_const( transB ),
        m, n, k,
        alpha, dA, dA_offset, ldda,
               dB, dB_offset, lddb,
        beta,  dC, dC_offset, lddc,
        &queue, get_g_event() );

    coot_wrapper(clFlush)(queue);
    opencl::coot_check_clblast_error(err, "magma_dgemm()");
    }
  #elif defined(COOT_USE_CLBLAS)
    {
    cl_int err = coot_wrapper(clblasDgemm)(
        clblasColumnMajor,
        clblas_trans_const( transA ),
        clblas_trans_const( transB ),
        m, n, k,
        alpha, dA, dA_offset, ldda,
               dB, dB_offset, lddb,
        beta,  dC, dC_offset, lddc,
        1, &queue, 0, NULL, get_g_event() );

    coot_wrapper(clFlush)(queue);
    opencl::coot_check_clblas_error(err, "magma_dgemm()");
    }
  #else
    {
    coot_stop_runtime_error("OpenCL MAGMA decompositions require CLBlast or clBLAS support to be enabled");
    }
  #endif
  }


inline
void
magma_sgemm(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaFloat_const_ptr dB, size_t dB_offset, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, size_t dC_offset, magma_int_t lddc,
    magma_queue_t queue )
  {
  if ( m <= 0 || n <= 0 || k <= 0 )
    {
    return;
    }

  #if defined(COOT_USE_CLBLAST)
    {
    if (beta == 0.0f)
      {
      // paranoia: CLBlast doesn't seem to like it when C is not initialized to 0
      dev_mem_t<float> dC_mem;
      dC_mem.cl_mem_ptr = coot_cl_mem({ dC, 0 });
      opencl::fill(dC_mem, 0.0f, m, n, dC_offset, 0, lddc);
      }

    CLBlastStatusCode err = coot_wrapper(CLBlastSgemm)(
        CLBlastLayoutColMajor,
        clblast_trans_const( transA ),
        clblast_trans_const( transB ),
        m, n, k,
        alpha, dA, dA_offset, ldda,
               dB, dB_offset, lddb,
        beta,  dC, dC_offset, lddc,
        &queue, get_g_event() );
    coot_wrapper(clFlush)(queue);
    opencl::coot_check_clblast_error(err, "magma_sgemm()");
    }
  #elif defined(COOT_USE_CLBLAS)
    {
    cl_int err = coot_wrapper(clblasSgemm)(
        clblasColumnMajor,
        clblas_trans_const( transA ),
        clblas_trans_const( transB ),
        m, n, k,
        alpha, dA, dA_offset, ldda,
               dB, dB_offset, lddb,
        beta,  dC, dC_offset, lddc,
        1, &queue, 0, NULL, get_g_event() );
    coot_wrapper(clFlush)(queue);
    opencl::coot_check_clblas_error(err, "magma_sgemm()");
    }
  #else
    {
    coot_stop_runtime_error("OpenCL MAGMA decompositions require CLBlast or clBLAS support to be enabled");
    }
  #endif
  }



//
// gemv

inline
void
magma_dgemv
  (
  magma_trans_t trans, magma_int_t m, magma_int_t n,
  double alpha,
  magmaDouble_const_ptr dA, size_t dA_offset, magma_int_t ldda,
  magmaDouble_const_ptr dx, size_t dx_offset, magma_int_t incx,
  double beta,
  magmaDouble_ptr dy, size_t dy_offset, magma_int_t incy,
  magma_queue_t queue)
  {
  if (m <= 0 || n <= 0)
    {
    return;
    }

  #if defined(COOT_USE_CLBLAST)
    {
    if (beta == 0.0)
      {
      // Paranoia: CLBlast doesn't seem to like it when y is not initialized to 0
      dev_mem_t<double> dy_mem;
      dy_mem.cl_mem_ptr = coot_cl_mem({ dy, 0 });
      opencl::fill(dy_mem, 0.0, 1, (trans == MagmaNoTrans) ? m : n, dy_offset, 0, incy);
      }

    CLBlastStatusCode err = coot_wrapper(CLBlastDgemv)(
        CLBlastLayoutColMajor,
        clblast_trans_const( trans ),
        size_t(m), size_t(n),
        alpha, dA, dA_offset, ldda,
               dx, dx_offset, incx,
        beta,  dy, dy_offset, incy,
        &queue, get_g_event() );
    coot_wrapper(clFlush)(queue);
    opencl::coot_check_clblast_error(err, "magma_dgemv()");
    }
  #elif defined(COOT_USE_CLBLAS)
    {
    cl_int err = coot_wrapper(clblasDgemv)(
        clblasColumnMajor,
        clblas_trans_const( trans ),
        size_t(m), size_t(n),
        alpha, dA, dA_offset, ldda,
               dx, dx_offset, incx,
        beta,  dy, dy_offset, incy,
        1, &queue, 0, NULL, get_g_event() );
    coot_wrapper(clFlush)(queue);
    opencl::coot_check_clblas_error(err, "magma_dgemv()");
    }
  #else
    {
    coot_stop_runtime_error("OpenCL MAGMA decompositions require CLBlast or clBLAS support to be enabled");
    }
  #endif
  }



inline
void
magma_sgemv
  (
  magma_trans_t trans, magma_int_t m, magma_int_t n,
  float alpha,
  magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t ldda,
  magmaFloat_const_ptr dx, size_t dx_offset, magma_int_t incx,
  float beta,
  magmaFloat_ptr dy, size_t dy_offset, magma_int_t incy,
  magma_queue_t queue)
  {
  if (m <= 0 || n <= 0)
    return;

  #if defined(COOT_USE_CLBLAST)
    {
    // Paranoia: CLBlast doesn't seem to like it when y is not initialized to 0
    if (beta == 0.0)
      {
      dev_mem_t<float> dy_mem;
      dy_mem.cl_mem_ptr = coot_cl_mem({ dy, 0 });
      opencl::fill(dy_mem, 0.0f, 1, (trans == MagmaNoTrans) ? m : n, dy_offset, 0, incy);
      }

    CLBlastStatusCode err = coot_wrapper(CLBlastSgemv)(
        CLBlastLayoutColMajor,
        clblast_trans_const( trans ),
        size_t(m), size_t(n),
        alpha, dA, dA_offset, ldda,
               dx, dx_offset, incx,
        beta,  dy, dy_offset, incy,
        &queue, get_g_event() );
    coot_wrapper(clFlush)(queue);
    opencl::coot_check_clblast_error(err, "magma_sgemv()");
    }
  #elif defined(COOT_USE_CLBLAS)
    {
    cl_int err = coot_wrapper(clblasSgemv)(
        clblasColumnMajor,
        clblas_trans_const( trans ),
        size_t(m), size_t(n),
        alpha, dA, dA_offset, ldda,
               dx, dx_offset, incx,
        beta,  dy, dy_offset, incy,
        1, &queue, 0, NULL, get_g_event() );
    coot_wrapper(clFlush)(queue);
    opencl::coot_check_clblas_error(err, "magma_sgemv()");
    }
  #else
    {
    coot_stop_runtime_error("OpenCL MAGMA decompositions require CLBlast or clBLAS support to be enabled");
    }
  #endif
  }



//
// syrk

inline
void
magma_dsyrk
  (
  magma_uplo_t uplo, magma_trans_t trans,
  magma_int_t n, magma_int_t k,
  double alpha,
  magmaDouble_const_ptr dA, size_t dA_offset, magma_int_t ldda,
  double beta,
  magmaDouble_ptr       dC, size_t dC_offset, magma_int_t lddc,
  magma_queue_t queue
  )
  {
  #if defined(COOT_USE_CLBLAST)
    {
    CLBlastStatusCode err = coot_wrapper(CLBlastDsyrk)(
        CLBlastLayoutColMajor,
        clblast_uplo_const( uplo ),
        clblast_trans_const( trans ),
        n, k,
        alpha,
        dA, dA_offset, ldda,
        beta,
        dC, dC_offset, lddc,
        &queue, get_g_event());
    opencl::coot_check_clblast_error(err, "magma_dsyrk()");
    }
  #elif defined(COOT_USE_CLBLAS)
    {
    cl_int err = coot_wrapper(clblasDsyrk)(
        clblasColumnMajor,
        clblas_uplo_const( uplo ),
        clblas_trans_const( trans ),
        n, k,
        alpha,
        dA, dA_offset, ldda,
        beta,
        dC, dC_offset, lddc,
        1, &queue, 0, NULL, get_g_event() );
    opencl::coot_check_clblas_error(err, "magma_dsyrk()");
    }
  #else
    {
    coot_stop_runtime_error("OpenCL MAGMA decompositions require CLBlast or clBLAS support to be enabled");
    }
  #endif
  }


inline
void
magma_ssyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t ldda,
    float beta,
    magmaFloat_ptr       dC, size_t dC_offset, magma_int_t lddc,
    magma_queue_t queue )
  {
  #if defined(COOT_USE_CLBLAST)
    {
    CLBlastStatusCode err = coot_wrapper(CLBlastSsyrk)(
        CLBlastLayoutColMajor,
        clblast_uplo_const( uplo ),
        clblast_trans_const( trans ),
        n, k,
        alpha, dA, dA_offset, ldda,
        beta,  dC, dC_offset, lddc,
        &queue, get_g_event());
    opencl::coot_check_clblast_error(err, "magma_ssyrk()");
    }
  #elif defined(COOT_USE_CLBLAS)
    {
    cl_int err = coot_wrapper(clblasSsyrk)(
        clblasColumnMajor,
        clblas_uplo_const( uplo ),
        clblas_trans_const( trans ),
        n, k,
        alpha, dA, dA_offset, ldda,
        beta,  dC, dC_offset, lddc,
        1, &queue, 0, NULL, get_g_event() );
    opencl::coot_check_clblas_error(err, "magma_ssyrk()");
    }
  #else
    {
    coot_stop_runtime_error("OpenCL MAGMA decompositions require CLBlast or clBLAS support to be enabled");
    }
  #endif
  }



//
// trsm


inline
void
magma_dtrsm
  (
  magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
  magma_int_t m, magma_int_t n,
  double alpha,
  magmaDouble_const_ptr dA, size_t dA_offset, magma_int_t ldda,
  magmaDouble_ptr       dB, size_t dB_offset, magma_int_t lddb,
  magma_queue_t queue
  )
  {
  if (m <= 0 || n <= 0)  { return; }

  #if defined(COOT_USE_CLBLAST)
    {
    CLBlastStatusCode err = coot_wrapper(CLBlastDtrsm)(
        CLBlastLayoutColMajor,
        clblast_side_const( side ),
        clblast_uplo_const( uplo ),
        clblast_trans_const( trans ),
        clblast_diag_const( diag ),
        m, n,
        alpha, dA, dA_offset, ldda,
               dB, dB_offset, lddb,
        &queue, get_g_event());
    coot_wrapper(clFlush)(queue);
    opencl::coot_check_clblast_error(err, "magma_dtrsm()");
    }
  #elif defined(COOT_USE_CLBLAS)
    {
    cl_int err = coot_wrapper(clblasDtrsm)(
        clblasColumnMajor,
        clblas_side_const( side ),
        clblas_uplo_const( uplo ),
        clblas_trans_const( trans ),
        clblas_diag_const( diag ),
        m, n,
        alpha, dA, dA_offset, ldda,
               dB, dB_offset, lddb,
        1, &queue, 0, NULL, get_g_event() );
    coot_wrapper(clFlush)(queue);
    opencl::coot_check_clblas_error(err, "magma_dtrsm()");
    }
  #else
    {
    coot_stop_runtime_error("OpenCL MAGMA decompositions require CLBlast or clBLAS support to be enabled");
    }
  #endif
  }


inline
void
magma_strsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaFloat_ptr       dB, size_t dB_offset, magma_int_t lddb,
    magma_queue_t queue )
  {
  if (m <= 0 || n <= 0)
    {
    return;
    }

  #if defined(COOT_USE_CLBLAST)
    {
    CLBlastStatusCode err = coot_wrapper(CLBlastStrsm)(
        CLBlastLayoutColMajor,
        clblast_side_const( side ),
        clblast_uplo_const( uplo ),
        clblast_trans_const( trans ),
        clblast_diag_const( diag ),
        m, n,
        alpha, dA, dA_offset, ldda,
               dB, dB_offset, lddb,
        &queue, get_g_event());
    coot_wrapper(clFlush)(queue);
    opencl::coot_check_clblast_error(err, "magma_strsm()");
    }
  #elif defined(COOT_USE_CLBLAS)
    {
    cl_int err = coot_wrapper(clblasStrsm)(
        clblasColumnMajor,
        clblas_side_const( side ),
        clblas_uplo_const( uplo ),
        clblas_trans_const( trans ),
        clblas_diag_const( diag ),
        m, n,
        alpha, dA, dA_offset, ldda,
               dB, dB_offset, lddb,
        1, &queue, 0, NULL, get_g_event() );
    coot_wrapper(clFlush)(queue);
    opencl::coot_check_cl_error(err, "magma_strsm()");
    }
  #else
    {
    coot_stop_runtime_error("OpenCL MAGMA decompositions require CLBlast or clBLAS support to be enabled");
    }
  #endif
  }



//
// trmm

inline
void
magma_dtrmm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaDouble_ptr       dB, size_t dB_offset, magma_int_t lddb,
    magma_queue_t queue )
  {
  if (m <= 0 || n <= 0)
    {
    return;
    }

  #if defined(COOT_USE_CLBLAST)
    {
    CLBlastStatusCode err = coot_wrapper(CLBlastDtrmm)(
        CLBlastLayoutColMajor,
        clblast_side_const( side ),
        clblast_uplo_const( uplo ),
        clblast_trans_const( trans ),
        clblast_diag_const( diag ),
        m, n,
        alpha, dA, dA_offset, ldda,
               dB, dB_offset, lddb,
        &queue, get_g_event());
    coot_wrapper(clFlush)(queue);
    opencl::coot_check_clblast_error(err, "magma_dtrmm()");
    }
  #elif defined(COOT_USE_CLBLAS)
    {
    cl_int err = coot_wrapper(clblasDtrmm)(
        clblasColumnMajor,
        clblas_side_const( side ),
        clblas_uplo_const( uplo ),
        clblas_trans_const( trans ),
        clblas_diag_const( diag ),
        m, n,
        alpha, dA, dA_offset, ldda,
               dB, dB_offset, lddb,
        1, &queue, 0, NULL, get_g_event() );
    coot_wrapper(clFlush)(queue);
    opencl::coot_check_clblas_error(err, "magma_dtrmm()");
    }
  #else
    {
    coot_stop_runtime_error("OpenCL MAGMA decompositions require CLBlast or clBLAS support to be enabled");
    }
  #endif
  }



inline
void
magma_strmm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t ldda,
    magmaFloat_ptr       dB, size_t dB_offset, magma_int_t lddb,
    magma_queue_t queue )
  {
  if (m <= 0 || n <= 0)
    {
    return;
    }

  #if defined(COOT_USE_CLBLAST)
    {
    CLBlastStatusCode err = coot_wrapper(CLBlastStrmm)(
        CLBlastLayoutColMajor,
        clblast_side_const( side ),
        clblast_uplo_const( uplo ),
        clblast_trans_const( trans ),
        clblast_diag_const( diag ),
        m, n,
        alpha, dA, dA_offset, ldda,
               dB, dB_offset, lddb,
        &queue, get_g_event());
    coot_wrapper(clFlush)(queue);
    opencl::coot_check_clblast_error(err, "magma_strmm()");
    }
  #elif defined(COOT_USE_CLBLAS)
    {
    cl_int err = coot_wrapper(clblasStrmm)(
        clblasColumnMajor,
        clblas_side_const( side ),
        clblas_uplo_const( uplo ),
        clblas_trans_const( trans ),
        clblas_diag_const( diag ),
        m, n,
        alpha, dA, dA_offset, ldda,
               dB, dB_offset, lddb,
        1, &queue, 0, NULL, get_g_event() );
    coot_wrapper(clFlush)(queue);
    opencl::coot_check_clblas_error(err, "magma_strmm()");
    }
  #else
    {
    coot_stop_runtime_error("OpenCL MAGMA decompositions require CLBlast or clBLAS support to be enabled");
    }
  #endif
  }



// trsv
// triangular matrix vector solve



inline
void
magma_dtrsv
  (
  magma_uplo_t uplo,
  magma_trans_t transA,
  magma_diag_t diag,
  magma_int_t n,
  magmaDouble_const_ptr dA,
  size_t dA_offset,
  magma_int_t ldda,
  magmaDouble_ptr dx,
  size_t dx_offset,
  magma_int_t incx,
  magma_queue_t queue
  )
  {
  if (n <= 0)
    return;

  #if defined(COOT_USE_CLBLAST)
    {
    CLBlastStatusCode err = coot_wrapper(CLBlastDtrsv)(
        CLBlastLayoutColMajor,
        clblast_uplo_const( uplo ),
        clblast_trans_const( transA ),
        clblast_diag_const( diag ),
        n,
        dA, dA_offset, ldda,
        dx, dx_offset, incx,
        &queue, get_g_event());
    coot_wrapper(clFlush)(queue);
    opencl::coot_check_clblast_error(err, "magma_dtrsv()");
    }
  #elif defined(COOT_USE_CLBLAS)
    {
    if (incx != 1)
      {
      throw std::runtime_error("magma_dtrsv() with clBLAS cannot accept incx != 1");
      }

    // clBLAS's TRSV implementation generates invalid kernels that violate the OpenCL standard!
    // See https://github.com/clMathLibraries/clBLAS/issues/341
    // So, instead, we use TRSM...
    cl_int err = coot_wrapper(clblasDtrsm)(
        clblasColumnMajor,
        clblasLeft,
        clblas_uplo_const( uplo ),
        clblas_trans_const( transA ),
        clblas_diag_const( diag ),
        n, 1, (double) 1.0,
        dA, dA_offset, ldda,
        dx, dx_offset, n,
        1,
        &queue, 0, NULL, get_g_event());
    coot_wrapper(clFlush)(queue);
    opencl::coot_check_clblas_error(err, "magma_dtrsv()");
    }
  #else
    {
    coot_stop_runtime_error("OpenCL MAGMA decompositions require CLBlast or clBLAS support to be enabled");
    }
  #endif
  }



inline
void
magma_strsv
  (
  magma_uplo_t uplo,
  magma_trans_t transA,
  magma_diag_t diag,
  magma_int_t n,
  magmaFloat_const_ptr dA,
  size_t dA_offset,
  magma_int_t ldda,
  magmaFloat_ptr dx,
  size_t dx_offset,
  magma_int_t incx,
  magma_queue_t queue
  )
  {
  if (n <= 0)
    return;

  #if defined(COOT_USE_CLBLAST)
    {
    CLBlastStatusCode err = coot_wrapper(CLBlastStrsv)(
        CLBlastLayoutColMajor,
        clblast_uplo_const( uplo ),
        clblast_trans_const( transA ),
        clblast_diag_const( diag ),
        n,
        dA, dA_offset, ldda,
        dx, dx_offset, incx,
        &queue, get_g_event());
    coot_wrapper(clFlush)(queue);
    opencl::coot_check_clblast_error(err, "magma_strsv()");
    }
  #elif defined(COOT_USE_CLBLAS)
    {
    if (incx != 1)
      {
      throw std::runtime_error("magma_strsv() cannot accept incx != 1");
      }

    // clBLAS's TRSV implementation generates invalid kernels that violate the OpenCL standard!
    // See https://github.com/clMathLibraries/clBLAS/issues/341
    // So, instead, we use TRSM...
    cl_int err = coot_wrapper(clblasStrsm)(
        clblasColumnMajor,
        clblasLeft,
        clblas_uplo_const( uplo ),
        clblas_trans_const( transA ),
        clblas_diag_const( diag ),
        n, 1, (float) 1.0,
        dA, dA_offset, ldda,
        dx, dx_offset, n, 1,
        &queue, 0, NULL, get_g_event());
    coot_wrapper(clFlush)(queue);
    opencl::coot_check_clblas_error(err, "magma_strsv()");
    }
  #else
    {
    coot_stop_runtime_error("OpenCL MAGMA decompositions require CLBlast or clBLAS support to be enabled");
    }
  #endif
  }



// symv



inline
void
magmablas_ssymv_work
  (
  magma_uplo_t uplo, magma_int_t n,
  float alpha,
  magmaFloat_const_ptr dA, size_t dA_offset, magma_int_t ldda,
  magmaFloat_const_ptr dx, size_t dx_offset, magma_int_t incx,
  float beta,
  magmaFloat_ptr dy, size_t dy_offset, magma_int_t incy,
  magmaFloat_ptr dwork, size_t dwork_offset, magma_int_t lwork, // unused!
  magma_queue_t queue
  )
  {
  coot_ignore(dwork);
  coot_ignore(dwork_offset);
  coot_ignore(lwork);

  if (n <= 0)
    {
    return;
    }

  #if defined(COOT_USE_CLBLAST)
    {
    CLBlastStatusCode err = coot_wrapper(CLBlastSsymv)(
        CLBlastLayoutColMajor,
        clblast_uplo_const( uplo ),
        n,
        alpha, dA, dA_offset, ldda,
               dx, dx_offset, incx,
        beta,  dy, dy_offset, incy,
        &queue, get_g_event());
    coot_wrapper(clFlush)(queue);
    opencl::coot_check_clblast_error(err, "magmablas_ssymv_work()");
    }
  #elif defined(COOT_USE_CLBLAS)
    {
    cl_int err = coot_wrapper(clblasSsymv)(
        clblasColumnMajor,
        clblas_uplo_const( uplo ),
        n,
        alpha, dA, dA_offset, ldda,
               dx, dx_offset, incx,
        beta,  dy, dy_offset, incy,
        1, &queue, 0, NULL, get_g_event() );
    coot_wrapper(clFlush)(queue);
    opencl::coot_check_clblas_error(err, "magmablas_ssymv_work()");
    }
  #else
    {
    coot_stop_runtime_error("OpenCL MAGMA decompositions require CLBlast or clBLAS support to be enabled");
    }
  #endif
  }



inline
void
magmablas_dsymv_work
  (
  magma_uplo_t uplo, magma_int_t n,
  double alpha,
  magmaDouble_const_ptr dA, size_t dA_offset, magma_int_t ldda,
  magmaDouble_const_ptr dx, size_t dx_offset, magma_int_t incx,
  double beta,
  magmaDouble_ptr dy, size_t dy_offset, magma_int_t incy,
  magmaDouble_ptr dwork, size_t dwork_offset, magma_int_t lwork,
  magma_queue_t queue
  )
  {
  coot_ignore(dwork);
  coot_ignore(dwork_offset);
  coot_ignore(lwork);

  if (n <= 0)
    {
    return;
    }

  #if defined(COOT_USE_CLBLAST)
    {
    CLBlastStatusCode err = coot_wrapper(CLBlastDsymv)(
        CLBlastLayoutColMajor,
        clblast_uplo_const( uplo ),
        n,
        alpha, dA, dA_offset, ldda,
               dx, dx_offset, incx,
        beta,  dy, dy_offset, incy,
        &queue, get_g_event());
    coot_wrapper(clFlush)(queue);
    opencl::coot_check_clblast_error(err, "magmablas_dsymv_work()");
    }
  #elif defined(COOT_USE_CLBLAS)
    {
    cl_int err = coot_wrapper(clblasDsymv)(
        clblasColumnMajor,
        clblas_uplo_const( uplo ),
        n,
        alpha, dA, dA_offset, ldda,
               dx, dx_offset, incx,
        beta,  dy, dy_offset, incy,
        1, &queue, 0, NULL, get_g_event() );
    coot_wrapper(clFlush)(queue);
    opencl::coot_check_clblas_error(err, "magmablas_dsymv_work()");
    }
  #else
    {
    coot_stop_runtime_error("OpenCL MAGMA decompositions require CLBlast or clBLAS support to be enabled");
    }
  #endif
  }



// symmetric rank-2 update (syr2k)



inline
void
magmablas_ssyr2k
  (
  magma_uplo_t uplo, magma_trans_t trans,
  magma_int_t n, magma_int_t k,
  float alpha,
  magmaFloat_ptr dA, size_t dA_offset, magma_int_t ldda,
  magmaFloat_ptr dB, size_t dB_offset, magma_int_t lddb,
  float beta,
  magmaFloat_ptr dC, size_t dC_offset, magma_int_t lddc,
  magma_queue_t queue
  )
  {
  if (n <= 0 || k <= 0)
    {
    return;
    }

  #if defined(COOT_USE_CLBLAST)
    {
    CLBlastStatusCode err = coot_wrapper(CLBlastSsyr2k)(
        CLBlastLayoutColMajor,
        clblast_uplo_const( uplo ),
        clblast_trans_const( trans ),
        n, k,
        alpha, dA, dA_offset, ldda,
               dB, dB_offset, lddb,
        beta,  dC, dC_offset, lddc,
        &queue, get_g_event());
    coot_wrapper(clFlush)(queue);
    opencl::coot_check_clblast_error(err, "magmablas_ssyr2k()");
    }
  #elif defined(COOT_USE_CLBLAS)
    {
    cl_int err = coot_wrapper(clblasSsyr2k)(
        clblasColumnMajor,
        clblas_uplo_const( uplo ),
        clblas_trans_const( trans ),
        n, k,
        alpha, dA, dA_offset, ldda,
               dB, dB_offset, lddb,
        beta,  dC, dC_offset, lddc,
        1, &queue, 0, NULL, get_g_event() );
    coot_wrapper(clFlush)(queue);
    opencl::coot_check_clblas_error(err, "magmablas_ssyr2k()");
    }
  #else
    {
    coot_stop_runtime_error("OpenCL MAGMA decompositions require CLBlast or clBLAS support to be enabled");
    }
  #endif
  }



inline
void
magmablas_dsyr2k
  (
  magma_uplo_t uplo, magma_trans_t trans,
  magma_int_t n, magma_int_t k,
  double alpha,
  magmaDouble_ptr dA, size_t dA_offset, magma_int_t ldda,
  magmaDouble_ptr dB, size_t dB_offset, magma_int_t lddb,
  double beta,
  magmaDouble_ptr dC, size_t dC_offset, magma_int_t lddc,
  magma_queue_t queue
  )
  {
  if (n <= 0 || k <= 0)
    {
    return;
    }

  #if defined(COOT_USE_CLBLAST)
    {
    CLBlastStatusCode err = coot_wrapper(CLBlastDsyr2k)(
        CLBlastLayoutColMajor,
        clblast_uplo_const( uplo ),
        clblast_trans_const( trans ),
        n, k,
        alpha, dA, dA_offset, ldda,
               dB, dB_offset, lddb,
        beta,  dC, dC_offset, lddc,
        &queue, get_g_event());
    coot_wrapper(clFlush)(queue);
    opencl::coot_check_clblast_error(err, "magmablas_dsyr2k()");
    }
  #elif defined(COOT_USE_CLBLAS)
    {
    cl_int err = coot_wrapper(clblasDsyr2k)(
        clblasColumnMajor,
        clblas_uplo_const( uplo ),
        clblas_trans_const( trans ),
        n, k,
        alpha, dA, dA_offset, ldda,
               dB, dB_offset, lddb,
        beta,  dC, dC_offset, lddc,
        1, &queue, 0, NULL, get_g_event() );
    coot_wrapper(clFlush)(queue);
    opencl::coot_check_clblas_error(err, "magmablas_ssyr2k()");
    }
  #else
    {
    coot_stop_runtime_error("OpenCL MAGMA decompositions require CLBlast or clBLAS support to be enabled");
    }
  #endif
  }



// dot products



inline
float
magma_cblas_sdot
  (
  magma_int_t n,
  const float *x, magma_int_t incx,
  const float *y, magma_int_t incy
  )
  {
  // after too many issues with MKL and other BLAS, just write our own dot product!
  float value = 0.0;
  magma_int_t i;
  if ( incx == 1 && incy == 1 )
    {
    for( i=0; i < n; ++i )
      {
      value = value + x[i] * y[i];
      }
    }
  else
    {
    magma_int_t ix=0, iy=0;
    if ( incx < 0 ) { ix = (-n + 1)*incx; }
    if ( incy < 0 ) { iy = (-n + 1)*incy; }
    for( i=0; i < n; ++i )
      {
      value = value + x[ix] * y[iy];
      ix += incx;
      iy += incy;
      }
    }

  return value;
  }



inline
double
magma_cblas_ddot
  (
  magma_int_t n,
  const double *x, magma_int_t incx,
  const double *y, magma_int_t incy
  )
  {
  // after too many issues with MKL and other BLAS, just write our own dot product!
  double value = 0.0;
  magma_int_t i;
  if ( incx == 1 && incy == 1 )
    {
    for( i=0; i < n; ++i )
      {
      value = value + x[i] * y[i];
      }
    }
  else
    {
    magma_int_t ix=0, iy=0;
    if ( incx < 0 ) { ix = (-n + 1)*incx; }
    if ( incy < 0 ) { iy = (-n + 1)*incy; }
    for( i=0; i < n; ++i )
      {
      value = value + x[ix] * y[iy];
      ix += incx;
      iy += incy;
      }
    }

  return value;
  }



// nrm2



inline
float
magma_cblas_snrm2
  (
  magma_int_t n,
  const float *x,
  magma_int_t incx
  )
  {
  if (n <= 0 || incx <= 0)
    {
    return 0;
    }
  else
    {
    float scale = 0;
    float ssq   = 1;
    // the following loop is equivalent to this call to the lapack
    // auxiliary routine:
    // call zlassq( n, x, incx, scale, ssq )
    for( magma_int_t ix=0; ix < 1 + (n-1)*incx; ix += incx )
      {
      if ( x[ix] != 0 )
        {
        float temp = fabs( x[ix] );
        if (scale < temp)
          {
          ssq = 1 + ssq * (scale/temp) * (scale/temp);
          scale = temp;
          }
        else
          {
          ssq += (temp/scale) * (temp/scale);
          }
        }
      }

    return scale * std::sqrt(ssq);
    }
  }



inline
double
magma_cblas_dnrm2
  (
  magma_int_t n,
  const double *x,
  magma_int_t incx
  )
  {
  if (n <= 0 || incx <= 0)
    {
    return 0;
    }
  else
    {
    double scale = 0;
    double ssq   = 1;
    // the following loop is equivalent to this call to the lapack
    // auxiliary routine:
    // call zlassq( n, x, incx, scale, ssq )
    for( magma_int_t ix=0; ix < 1 + (n-1)*incx; ix += incx )
      {
      if ( x[ix] != 0 )
        {
        double temp = fabs( x[ix] );
        if (scale < temp)
          {
          ssq = 1 + ssq * (scale/temp) * (scale/temp);
          scale = temp;
          }
        else
          {
          ssq += (temp/scale) * (temp/scale);
          }
        }
      }

    return scale * std::sqrt(ssq);
    }
  }



// amax



inline
magma_int_t
magma_isamax(magma_int_t n, magmaFloat_const_ptr dx, size_t dx_offset, magma_int_t incx, magma_queue_t queue)
  {
  if (n <= 0)
    {
    return 0;
    }

  // need to initialize one GPU unsigned int to store the result...
  coot_cl_mem out = get_rt().cl_rt.acquire_memory<unsigned int>(1);

  #if defined(COOT_USE_CLBLAST)
    {
    CLBlastStatusCode status = coot_wrapper(CLBlastiSamax)(n, out.ptr, 0, dx, dx_offset, incx, &queue, get_g_event() );
    opencl::coot_check_clblast_error(status, "coot::opencl::magma_isamax(): call to CLBlastiSamax() failed");
    }
  #elif defined(COOT_USE_CLBLAS)
    {
    coot_cl_mem dwork = get_rt().cl_rt.acquire_memory<float>(2 * n);
    cl_int status = coot_wrapper(clblasiSamax)(n, out.ptr, 0, dx, dx_offset, incx, dwork.ptr, 1, &queue, 0, NULL, get_g_event() );
    opencl::coot_check_clblas_error(status, "coot::opencl::magma_isamax(): call to clblasiSamax() failed");
    get_rt().cl_rt.release_memory(dwork);
    }
  #else
    {
    coot_stop_runtime_error("OpenCL MAGMA decompositions require CLBlast or clBLAS support to be enabled");
    }
  #endif

  int result = 0;
  cl_int status = coot_wrapper(clEnqueueReadBuffer)(queue, out.ptr, CL_TRUE, 0, sizeof(unsigned int), &result, 0, NULL, NULL);
  opencl::coot_check_cl_error(status, "coot::opencl::magma_isamax(): getting result from device memory failed");

  get_rt().cl_rt.release_memory(out);

  return result;
  }



inline
magma_int_t
magma_idamax(magma_int_t n, magmaDouble_const_ptr dx, size_t dx_offset, magma_int_t incx, magma_queue_t queue)
  {
  if (n <= 0)
    {
    return 0;
    }

  // need to initialize one GPU unsigned int to store the result...
  coot_cl_mem out = get_rt().cl_rt.acquire_memory<unsigned int>(1);

  #if defined(COOT_USE_CLBLAST)
    {
    CLBlastStatusCode status = coot_wrapper(CLBlastiDamax)(n, out.ptr, 0, dx, dx_offset, incx, &queue, get_g_event() );
    opencl::coot_check_clblast_error(status, "coot::opencl::magma_idamax(): call to CLBlastiDamax() failed");
    }
  #elif defined(COOT_USE_CLBLAS)
    {
    coot_cl_mem dwork = get_rt().cl_rt.acquire_memory<double>(2 * n);
    cl_int status = coot_wrapper(clblasiDamax)(n, out.ptr, 0, dx, dx_offset, incx, dwork.ptr, 1, &queue, 0, NULL, get_g_event() );
    opencl::coot_check_clblas_error(status, "coot::opencl::magma_isamax(): call to clblasiSamax() failed");
    get_rt().cl_rt.release_memory(dwork);
    }
  #else
    {
    coot_stop_runtime_error("OpenCL MAGMA decompositions require CLBlast or clBLAS support to be enabled");
    }
  #endif

  int result = 0;
  cl_int status = coot_wrapper(clEnqueueReadBuffer)(queue, out.ptr, CL_TRUE, 0, sizeof(unsigned int), &result, 0, NULL, NULL);
  opencl::coot_check_cl_error(status, "coot::opencl::magma_isamax(): getting result from device memory failed");

  get_rt().cl_rt.release_memory(out);

  return result;
  }



inline
magma_int_t
magma_get_smlsize_divideconquer()
  {
  return 128;
  }



////////////////////////////////////////


#define MAGMA_D_ZERO              ( 0.0)
#define MAGMA_D_ONE               ( 1.0)
#define MAGMA_D_HALF              ( 0.5)
#define MAGMA_D_NEG_ONE           (-1.0)
#define MAGMA_D_NEG_HALF          (-0.5)

#define MAGMA_S_ZERO              ( 0.0)
#define MAGMA_S_ONE               ( 1.0)
#define MAGMA_S_HALF              ( 0.5)
#define MAGMA_S_NEG_ONE           (-1.0)
#define MAGMA_S_NEG_HALF          (-0.5)
