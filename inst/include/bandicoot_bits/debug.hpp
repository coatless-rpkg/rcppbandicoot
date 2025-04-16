// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2017-2023 Ryan Curtin (https://www.ratml.org)
// Copyright 2008-2023 Conrad Sanderson (https://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
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
std::ostream&
get_cout_stream()
  {
  return (COOT_COUT_STREAM);
  }



inline
std::ostream&
get_cerr_stream()
  {
  return (COOT_CERR_STREAM);
  }



// print a message to get_cerr_stream() and throw logic_error exception
template<typename T1>
coot_cold
coot_noinline
static
void
coot_stop_logic_error(const T1& x)
  {
  #if defined(COOT_PRINT_EXCEPTIONS)
    {
    get_cerr_stream() << "\nerror: " << x << std::endl;
    }
  #endif

  throw std::logic_error( std::string(x) );
  }



//! print a message to get_cerr_stream() and throw out_of_range exception
template<typename T1>
coot_cold
coot_noinline
static
void
coot_stop_bounds_error(const T1& x)
  {
  #if defined(COOT_PRINT_EXCEPTIONS)
    {
    get_cerr_stream() << "\nerror: " << x << std::endl;
    }
  #endif

  throw std::out_of_range( std::string(x) );
  }



// print a message to get_cerr_stream() and throw bad_alloc exception
template<typename T1>
coot_cold
coot_noinline
static
void
coot_stop_bad_alloc(const T1& x)
  {
  #if defined(COOT_PRINT_EXCEPTIONS)
    {
    get_cerr_stream() << "\nerror: " << x << std::endl;
    }
  #else
    {
    coot_ignore(x);
    }
  #endif

  throw std::bad_alloc();
  }



// print a message to get_cerr_stream() and throw runtime_error exception
template<typename T1>
coot_cold
coot_noinline
static
void
coot_stop_runtime_error(const T1& x)
  {
  #if defined(COOT_PRINT_EXCEPTIONS)
    {
    get_cerr_stream() << "\nerror: " << x << std::endl;
    }
  #endif

  throw std::runtime_error( std::string(x) );
  }



// print a message to get_cerr_stream() and throw runtime_error exception
template<typename T1, typename T2>
coot_cold
coot_noinline
static
void
coot_stop_runtime_error(const T1& x, const T2& y)
  {
  #if defined(COOT_PRINT_EXCEPTIONS)
    {
    get_cerr_stream() << "\nerror: " << x << ": " << y << std::endl;
    }
  #endif

  throw std::runtime_error( std::string(x) + std::string(": ") + std::string(y) );
  }



//
// coot_print


coot_cold
inline
void
coot_print()
  {
  get_cerr_stream() << std::endl;
  }


template<typename T1>
coot_cold
coot_noinline
static
void
coot_print(const T1& x)
  {
  get_cerr_stream() << x << std::endl;
  }



template<typename T1, typename T2>
coot_cold
coot_noinline
static
void
coot_print(const T1& x, const T2& y)
  {
  get_cerr_stream() << x << y << std::endl;
  }



template<typename T1, typename T2, typename T3>
coot_cold
coot_noinline
static
void
coot_print(const T1& x, const T2& y, const T3& z)
  {
  get_cerr_stream() << x << y << z << std::endl;
  }






//
// coot_sigprint

// print a message the the log stream with a preceding @ character.
// by default the log stream is cout.
// used for printing the signature of a function
// (see the coot_extra_debug_sigprint macro)
inline
void
coot_sigprint(const char* x)
  {
  get_cerr_stream() << "@ " << x;
  }



//
// coot_bktprint


inline
void
coot_bktprint()
  {
  get_cerr_stream() << std::endl;
  }


template<typename T1>
inline
void
coot_bktprint(const T1& x)
  {
  get_cerr_stream() << " [" << x << ']' << std::endl;
  }



template<typename T1, typename T2>
inline
void
coot_bktprint(const T1& x, const T2& y)
  {
  get_cerr_stream() << " [" << x << y << ']' << std::endl;
  }






//
// coot_thisprint

inline
void
coot_thisprint(const void* this_ptr)
  {
  get_cerr_stream() << " [this = " << this_ptr << ']' << std::endl;
  }



//
// coot_warn


// print a message to the warn stream
template<typename T1>
coot_cold
coot_noinline
static
void
coot_warn(const T1& x)
  {
  get_cerr_stream() << "\nwarning: " << x << '\n';
  }


template<typename T1, typename T2>
coot_cold
coot_noinline
static
void
coot_warn(const T1& x, const T2& y)
  {
  get_cerr_stream() << "\nwarning: " << x << y << '\n';
  }


template<typename T1, typename T2, typename T3>
coot_cold
coot_noinline
static
void
coot_warn(const T1& x, const T2& y, const T3& z)
  {
  get_cerr_stream() << "\nwarning: " << x << y << z << '\n';
  }



//
// coot_warn_level


template<typename T1>
inline
void
coot_warn_level(const uword level, const T1& arg1)
  {
  constexpr uword config_level = (sword(COOT_WARN_LEVEL) > 0) ? uword(COOT_WARN_LEVEL) : uword(0);
  
  if((config_level > 0) && (level <= config_level))  { coot_warn(arg1); }
  }


template<typename T1, typename T2>
inline
void
coot_warn_level(const uword level, const T1& arg1, const T2& arg2)
  {
  constexpr uword config_level = (sword(COOT_WARN_LEVEL) > 0) ? uword(COOT_WARN_LEVEL) : uword(0);
  
  if((config_level > 0) && (level <= config_level))  { coot_warn(arg1,arg2); }
  }


template<typename T1, typename T2, typename T3>
inline
void
coot_warn_level(const uword level, const T1& arg1, const T2& arg2, const T3& arg3)
  {
  constexpr uword config_level = (sword(COOT_WARN_LEVEL) > 0) ? uword(COOT_WARN_LEVEL) : uword(0);
  
  if((config_level > 0) && (level <= config_level))  { coot_warn(arg1,arg2,arg3); }
  }


template<typename T1, typename T2, typename T3, typename T4>
inline
void
coot_warn_level(const uword level, const T1& arg1, const T2& arg2, const T3& arg3, const T4& arg4)
  {
  constexpr uword config_level = (sword(COOT_WARN_LEVEL) > 0) ? uword(COOT_WARN_LEVEL) : uword(0);
  
  if((config_level > 0) && (level <= config_level))  { coot_warn(arg1,arg2,arg3,arg4); }
  }



//
// coot_check

// if state is true, abort program
template<typename T1>
coot_hot
inline
void
coot_check(const bool state, const T1& x)
  {
  if(state)  { coot_stop_logic_error(coot_str::str_wrapper(x)); }
  }


template<typename T1, typename T2>
coot_hot
inline
void
coot_check(const bool state, const T1& x, const T2& y)
  {
  if(state)  { coot_stop_logic_error( std::string(x) + std::string(y) ); }
  }


template<typename T1>
coot_hot
inline
void
coot_check_bad_alloc(const bool state, const T1& x)
  {
  if(state)  { coot_stop_bad_alloc(x); }
  }


template<typename T1>
coot_hot
inline
void
coot_check_runtime_error(const bool state, const T1& x)
  {
  if(state)  { coot_stop_runtime_error(x); }
  }

template<typename T1>
coot_hot
inline
void
coot_check_bounds(const bool state, const T1& x)
  {
  if(state)  { coot_stop_bounds_error( std::string(x) ); }
  }



coot_hot
coot_inline
void
coot_set_error(bool& err_state, char*& err_msg, const bool expression, const char* message)
  {
  if(expression)
    {
    err_state = true;
    err_msg   = const_cast<char*>(message);
    }
  }



//
// functions for generating strings indicating size errors

coot_cold
coot_noinline
static
std::string
coot_incompat_size_string(const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols, const char* x)
  {
  std::ostringstream tmp;

  tmp << x << ": incompatible matrix dimensions: " << A_n_rows << 'x' << A_n_cols << " and " << B_n_rows << 'x' << B_n_cols;

  return tmp.str();
  }



coot_cold
coot_noinline
static
std::string
coot_incompat_size_string(const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices, const char* x)
  {
  std::ostringstream tmp;

  tmp << x << ": incompatible cube dimensions: " << A_n_rows << 'x' << A_n_cols << 'x' << A_n_slices << " and " << B_n_rows << 'x' << B_n_cols << 'x' << B_n_slices;

  return tmp.str();
  }



//
// functions for checking whether two dense matrices have the same dimensions



coot_inline
coot_hot
void
coot_assert_same_size(const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols, const char* x)
  {
  if( (A_n_rows != B_n_rows) || (A_n_cols != B_n_cols) )
    {
    coot_stop_logic_error( coot_incompat_size_string(A_n_rows, A_n_cols, B_n_rows, B_n_cols, x) );
    }
  }



// stop if given matrices have different sizes
template<typename eT1, typename eT2>
coot_hot
inline
void
coot_assert_same_size(const Mat<eT1>& A, const Mat<eT2>& B, const char* x)
  {
  const uword A_n_rows = A.n_rows;
  const uword A_n_cols = A.n_cols;

  const uword B_n_rows = B.n_rows;
  const uword B_n_cols = B.n_cols;

  if( (A_n_rows != B_n_rows) || (A_n_cols != B_n_cols) )
    {
    coot_stop_logic_error( coot_incompat_size_string(A_n_rows, A_n_cols, B_n_rows, B_n_cols, x) );
    }
  }



coot_inline
coot_hot
void
coot_assert_same_size(const uword A_n_rows, const uword A_n_cols, const uword A_n_slices, const uword B_n_rows, const uword B_n_cols, const uword B_n_slices, const char* x)
  {
  if( (A_n_rows != B_n_rows) || (A_n_cols != B_n_cols) || (A_n_slices != B_n_slices) )
    {
    coot_stop_logic_error( coot_incompat_size_string(A_n_rows, A_n_cols, A_n_slices, B_n_rows, B_n_cols, B_n_slices, x) );
    }
  }



// stop if given cubes have different sizes
template<typename T1, typename T2>
coot_hot
inline
void
coot_assert_same_size(const T1& A, const T2& B, const char* x,
                      const typename enable_if2<is_coot_cube_type<T1>::value, void*>::result* junk1 = 0,
                      const typename enable_if2<is_coot_cube_type<T2>::value, void*>::result* junk2 = 0)
  {
  coot_ignore(junk1);
  coot_ignore(junk2);

  const uword A_n_rows = A.n_rows;
  const uword A_n_cols = A.n_cols;
  const uword A_n_slices = A.n_slices;

  const uword B_n_rows = B.n_rows;
  const uword B_n_cols = B.n_cols;
  const uword B_n_slices = B.n_slices;

  if( (A_n_rows != B_n_rows) || (A_n_cols != B_n_cols) || (A_n_slices != B_n_slices) )
    {
    coot_stop_logic_error( coot_incompat_size_string(A_n_rows, A_n_cols, A_n_slices, B_n_rows, B_n_cols, B_n_slices, x) );
    }
  }



//
// functions for checking whether two matrices have dimensions that are compatible with the matrix multiply operation



coot_hot
inline
void
coot_assert_mul_size(const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols, const char* x)
  {
  if(A_n_cols != B_n_rows)
    {
    coot_stop_logic_error( coot_incompat_size_string(A_n_rows, A_n_cols, B_n_rows, B_n_cols, x) );
    }
  }



// stop if given matrices are incompatible for multiplication
template<typename eT1, typename eT2>
coot_hot
inline
void
coot_assert_mul_size(const Mat<eT1>& A, const Mat<eT2>& B, const char* x)
  {
  const uword A_n_cols = A.n_cols;
  const uword B_n_rows = B.n_rows;

  if(A_n_cols != B_n_rows)
    {
    coot_stop_logic_error( coot_incompat_size_string(A.n_rows, A_n_cols, B_n_rows, B.n_cols, x) );
    }
  }



// stop if given matrices are incompatible for multiplication
template<typename eT1, typename eT2>
coot_hot
inline
void
coot_assert_mul_size(const Mat<eT1>& A, const Mat<eT2>& B, const bool do_trans_A, const bool do_trans_B, const char* x)
  {
  const uword final_A_n_cols = (do_trans_A == false) ? A.n_cols : A.n_rows;
  const uword final_B_n_rows = (do_trans_B == false) ? B.n_rows : B.n_cols;

  if(final_A_n_cols != final_B_n_rows)
    {
    const uword final_A_n_rows = (do_trans_A == false) ? A.n_rows : A.n_cols;
    const uword final_B_n_cols = (do_trans_B == false) ? B.n_cols : B.n_rows;

    coot_stop_logic_error( coot_incompat_size_string(final_A_n_rows, final_A_n_cols, final_B_n_rows, final_B_n_cols, x) );
    }
  }



template<const bool do_trans_A, const bool do_trans_B>
coot_hot
inline
void
coot_assert_trans_mul_size(const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols, const char* x)
  {
  const uword final_A_n_cols = (do_trans_A == false) ? A_n_cols : A_n_rows;
  const uword final_B_n_rows = (do_trans_B == false) ? B_n_rows : B_n_cols;

  if(final_A_n_cols != final_B_n_rows)
    {
    const uword final_A_n_rows = (do_trans_A == false) ? A_n_rows : A_n_cols;
    const uword final_B_n_cols = (do_trans_B == false) ? B_n_cols : B_n_rows;

    coot_stop_logic_error( coot_incompat_size_string(final_A_n_rows, final_A_n_cols, final_B_n_rows, final_B_n_cols, x) );
    }
  }



template<typename T1>
coot_hot
inline
void
coot_assert_blas_size(const T1& A)
  {
  if(sizeof(uword) >= sizeof(blas_int))
    {
    bool overflow;

    overflow = (A.n_rows > COOT_MAX_BLAS_INT);
    overflow = (A.n_cols > COOT_MAX_BLAS_INT) || overflow;

    if(overflow)
      {
      coot_stop_runtime_error("integer overflow: matrix dimensions are too large for integer type used by BLAS and LAPACK");
      }
    }
  }



template<typename T1, typename T2>
coot_hot
inline
void
coot_assert_blas_size(const T1& A, const T2& B)
  {
  if(sizeof(uword) >= sizeof(blas_int))
    {
    bool overflow;

    overflow = (A.n_rows > COOT_MAX_BLAS_INT);
    overflow = (A.n_cols > COOT_MAX_BLAS_INT) || overflow;
    overflow = (B.n_rows > COOT_MAX_BLAS_INT) || overflow;
    overflow = (B.n_cols > COOT_MAX_BLAS_INT) || overflow;

    if(overflow)
      {
      coot_stop_runtime_error("integer overflow: matrix dimensions are too large for integer type used by BLAS and LAPACK");
      }
    }
  }



//
// macros


// #define COOT_STRING1(x) #x
// #define COOT_STRING2(x) COOT_STRING1(x)
// #define COOT_FILELINE  __FILE__ ": " COOT_STRING2(__LINE__)


#if defined(COOT_NO_DEBUG)

  #undef COOT_EXTRA_DEBUG

  #define coot_debug_print                   true ? (void)0 : coot_print
  #define coot_debug_warn                    true ? (void)0 : coot_warn
  #define coot_debug_warn_level              true ? (void)0 : coot_warn_level
  #define coot_debug_check                   true ? (void)0 : coot_check
  #define coot_debug_check_bounds            true ? (void)0 : coot_check_bounds
  #define coot_debug_set_error               true ? (void)0 : coot_set_error
  #define coot_debug_assert_same_size        true ? (void)0 : coot_assert_same_size
  #define coot_debug_assert_mul_size         true ? (void)0 : coot_assert_mul_size
  #define coot_debug_assert_trans_mul_size   true ? (void)0 : coot_assert_trans_mul_size
  #define coot_debug_assert_blas_size        true ? (void)0 : coot_assert_blas_size

#else

  #define coot_debug_print                 coot_print
  #define coot_debug_warn                  coot_warn
  #define coot_debug_warn_level            coot_warn_level
  #define coot_debug_check                 coot_check
  #define coot_debug_check_bounds          coot_check_bounds
  #define coot_debug_set_error             coot_set_error
  #define coot_debug_assert_same_size      coot_assert_same_size
  #define coot_debug_assert_mul_size       coot_assert_mul_size
  #define coot_debug_assert_trans_mul_size coot_assert_trans_mul_size
  #define coot_debug_assert_blas_size      coot_assert_blas_size

#endif



#if defined(COOT_EXTRA_DEBUG)

  #define coot_extra_debug_sigprint       coot_sigprint(COOT_FNSIG); coot_bktprint
  #define coot_extra_debug_sigprint_this  coot_sigprint(COOT_FNSIG); coot_thisprint
  #define coot_extra_debug_print          coot_print
  #define coot_extra_debug_warn           coot_warn

#else

  #define coot_extra_debug_sigprint        true ? (void)0 : coot_bktprint
  #define coot_extra_debug_sigprint_this   true ? (void)0 : coot_thisprint
  #define coot_extra_debug_print           true ? (void)0 : coot_print
  #define coot_extra_debug_warn            true ? (void)0 : coot_warn

#endif




#if defined(COOT_EXTRA_DEBUG)

  namespace junk
    {
    class coot_first_extra_debug_message
      {
      public:

      inline
      coot_first_extra_debug_message()
        {
        union
          {
          unsigned short a;
          unsigned char  b[sizeof(unsigned short)];
          } endian_test;

        endian_test.a = 1;

        const bool        little_endian = (endian_test.b[0] == 1);
        const std::string note          = COOT_VERSION_NOTE;

        std::ostream& out = get_cerr_stream();

        out << "@ ---" << '\n';
        out << "@ Bandicoot " << coot_version::major << '.' << coot_version::minor << '.' << coot_version::patch;
        if(note.length() > 0)  { out << " (" << note << ')'; }
        out << '\n';
        out << "@ coot_config::wrapper    = " << coot_config::wrapper  << '\n';
        out << "@ coot_config::extra_code = " << coot_config::extra_code   << '\n';
        out << "@ sizeof(void*)    = " << sizeof(void*)    << '\n';
        out << "@ sizeof(int)      = " << sizeof(int)      << '\n';
        out << "@ sizeof(long)     = " << sizeof(long)     << '\n';
        out << "@ sizeof(uword)    = " << sizeof(uword)    << '\n';
        out << "@ sizeof(blas_int) = " << sizeof(blas_int) << '\n';
        out << "@ little_endian    = " << little_endian    << '\n';
        out << "@ ---" << std::endl;
        }

      };

    static coot_first_extra_debug_message coot_first_extra_debug_message_run;
    }

#endif
