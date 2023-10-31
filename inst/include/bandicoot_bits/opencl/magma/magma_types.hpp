// Copyright 2017 Conrad Sanderson (http://conradsanderson.id.au)
//~
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//~
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
//~
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


// -----------------------------------------------------------------------------
// Return codes
// LAPACK argument errors are < 0 but > MAGMA_ERR.
// MAGMA errors are < MAGMA_ERR.

#define MAGMA_SUCCESS               0       ///< operation was successful
#define MAGMA_ERR                  -100     ///< unspecified error
#define MAGMA_ERR_NOT_INITIALIZED  -101     ///< magma_init() was not called
#define MAGMA_ERR_REINITIALIZED    -102     // unused
#define MAGMA_ERR_NOT_SUPPORTED    -103     ///< not supported on this GPU
#define MAGMA_ERR_ILLEGAL_VALUE    -104     // unused
#define MAGMA_ERR_NOT_FOUND        -105     ///< file not found
#define MAGMA_ERR_ALLOCATION       -106     // unused
#define MAGMA_ERR_INTERNAL_LIMIT   -107     // unused
#define MAGMA_ERR_UNALLOCATED      -108     // unused
#define MAGMA_ERR_FILESYSTEM       -109     // unused
#define MAGMA_ERR_UNEXPECTED       -110     // unused
#define MAGMA_ERR_SEQUENCE_FLUSHED -111     // unused
#define MAGMA_ERR_HOST_ALLOC       -112     ///< could not malloc CPU host memory
#define MAGMA_ERR_DEVICE_ALLOC     -113     ///< could not malloc GPU device memory
#define MAGMA_ERR_CUDASTREAM       -114     // unused
#define MAGMA_ERR_INVALID_PTR      -115     ///< can't free invalid pointer
#define MAGMA_ERR_UNKNOWN          -116     ///< unspecified error
#define MAGMA_ERR_NOT_IMPLEMENTED  -117     ///< not implemented yet
#define MAGMA_ERR_NAN              -118     ///< NaN (not-a-number) detected


#define MagmaUpperStr         "Upper"
#define MagmaLowerStr         "Lower"
#define MagmaFullStr          "Full"

#define MagmaNonUnitStr       "NonUnit"
#define MagmaUnitStr          "Unit"

#define MagmaForwardStr       "Forward"
#define MagmaBackwardStr      "Backward"

#define MagmaColumnwiseStr    "Columnwise"
#define MagmaRowwiseStr       "Rowwise"

typedef enum {
    MagmaUpper         = 121,
    MagmaLower         = 122,
    MagmaFull          = 123,  /* lascl, laset */
    MagmaHessenberg    = 124   /* lascl */
} magma_uplo_t;

typedef magma_uplo_t magma_type_t;  /* lascl */

typedef enum {
    MagmaLeft          = 141,
    MagmaRight         = 142,
    MagmaBothSides     = 143   /* trevc */
} magma_side_t;

typedef enum {
    MagmaNoTrans       = 111,
    MagmaTrans         = 112,
    MagmaConjTrans     = 113,
    Magma_ConjTrans    = MagmaConjTrans
} magma_trans_t;

typedef enum {
    MagmaNonUnit       = 131,
    MagmaUnit          = 132
} magma_diag_t;

typedef enum {
    MagmaOneNorm       = 171,  /* lange, lanhe */
    MagmaRealOneNorm   = 172,
    MagmaTwoNorm       = 173,
    MagmaFrobeniusNorm = 174,
    MagmaInfNorm       = 175,
    MagmaRealInfNorm   = 176,
    MagmaMaxNorm       = 177,
    MagmaRealMaxNorm   = 178
} magma_norm_t;

typedef enum {
    MagmaNoVec         = 301,  /* geev, syev, gesvd */
    MagmaVec           = 302,  /* geev, syev */
    MagmaIVec          = 303,  /* stedc */
    MagmaAllVec        = 304,  /* gesvd, trevc */
    MagmaSomeVec       = 305,  /* gesvd, trevc */
    MagmaOverwriteVec  = 306,  /* gesvd */
    MagmaBacktransVec  = 307   /* trevc */
} magma_vec_t;

typedef enum {
    MagmaForward       = 391,  /* larfb */
    MagmaBackward      = 392
} magma_direct_t;

typedef enum {
    MagmaRangeAll      = 311,  /* syevx, etc. */
    MagmaRangeV        = 312,
    MagmaRangeI        = 313
} magma_range_t;

typedef enum {
    MagmaQ             = 322,  /* unmbr, ungbr */
    MagmaP             = 323
} magma_vect_t;

typedef enum {
    MagmaColumnwise    = 401,  /* larfb */
    MagmaRowwise       = 402
} magma_storev_t;

// NOTE: this is not the same as how MAGMA defines things!
// However, this makes it a million times easier to support BLAS libraries with integers of different widths.
typedef blas_int magma_int_t;

typedef cl_mem magma_ptr;
typedef cl_mem magmaInt_ptr;
typedef cl_mem magmaIndex_ptr;
typedef cl_mem magmaFloat_ptr;
typedef cl_mem magmaDouble_ptr;
typedef cl_mem magmaFloatComplex_ptr;
typedef cl_mem magmaDoubleComplex_ptr;

typedef cl_mem magma_const_ptr;
typedef cl_mem magmaInt_const_ptr;
typedef cl_mem magmaIndex_const_ptr;
typedef cl_mem magmaFloat_const_ptr;
typedef cl_mem magmaDouble_const_ptr;
typedef cl_mem magmaFloatComplex_const_ptr;
typedef cl_mem magmaDoubleComplex_const_ptr;

typedef cl_command_queue  magma_queue_t;
typedef cl_event          magma_event_t;
typedef cl_device_id      magma_device_t;



/// For integers x >= 0, y > 0, returns ceil( x/y ).
/// For x == 0, this is 0.
inline magma_int_t magma_ceildiv(magma_int_t x, magma_int_t y)
  {
  return (x + y - 1) / y;
  }



/// For integers x >= 0, y > 0, returns x rounded up to multiple of y.
/// That is, ceil(x/y)*y.
/// For x == 0, this is 0.
/// This implementation does not assume y is a power of 2.
inline magma_int_t magma_roundup(magma_int_t x, magma_int_t y)
  {
  return magma_ceildiv(x, y) * y;
  }
