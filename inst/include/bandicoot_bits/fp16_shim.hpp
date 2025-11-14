// Copyright 2025 Ryan Curtin (http://www.ratml.org/)
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


// Utility struct to represent an FP16 GPU type.
// This internally holds a float (fp32), so that host-side arithmetic is
// possible.

struct fp16_shim
  {
  float x;



  inline fp16_shim() : x(0.0f) { }



  template<typename T>
  inline fp16_shim(const T& in_x, const typename enable_if<std::is_arithmetic<T>::value>::result* = 0) : x(float(in_x)) { }



  inline operator float() const { return x; }



  //
  // These implementations are adapted directly from the Khronos OpenCL headers
  // (also Apache licensed) because we don't have a guarantee that the OpenCL
  // distribution contains cl_half.h (or these utility functions), since they do
  // not appear to be a part of the official specification.
  //
  #ifdef COOT_USE_OPENCL
  #ifndef CL_HALF_EXP_MASK
    #define COOT_UNDEF_CL_HALF_EXP_MASK
    #define CL_HALF_EXP_MASK 0x7C00
  #endif
  #ifndef CL_HALF_MAX_FINITE_MAG
    #define COOT_UNDEF_CL_HALF_MAX_FINITE_MAG
    #define CL_HALF_MAX_FINITE_MAG 0x7BFF
  #endif
  inline fp16_shim(const cl_half& h)
    {
    // Type-punning to get direct access to underlying bits
    union f32
      {
      float f;
      uint32_t i;
      } f32;

    // Extract sign bit
    uint16_t sign = h >> 15;

    // Extract FP16 exponent and mantissa
    uint16_t h_exp = (h >> (CL_HALF_MANT_DIG - 1)) & 0x1F;
    uint16_t h_mant = h & 0x3FF;

    // Remove FP16 exponent bias
    int32_t exp = h_exp - CL_HALF_MAX_EXP + 1;

    // Add FP32 exponent bias
    uint32_t f_exp = exp + CL_FLT_MAX_EXP - 1;

    // Check for NaN / infinity
    if (h_exp == 0x1F)
      {
      if (h_mant)
        {
        // NaN -> propagate mantissa and silence it
        uint32_t f_mant = h_mant << (CL_FLT_MANT_DIG - CL_HALF_MANT_DIG);
        f_mant |= 0x400000;
        f32.i = (sign << 31) | 0x7F800000 | f_mant;
        x = f32.f;
        return;
        }
      else
        {
        // Infinity -> zero mantissa
        f32.i = (sign << 31) | 0x7F800000;
        x = f32.f;
        return;
        }
      }

    // Check for zero / denormal
    if (h_exp == 0)
      {
      if (h_mant == 0)
        {
        // Zero -> zero exponent
        f_exp = 0;
        }
      else
        {
        // Denormal -> normalize it
        // - Shift mantissa to make most-significant 1 implicit
        // - Adjust exponent accordingly
        uint32_t shift = 0;
        while ((h_mant & 0x400) == 0)
          {
          h_mant <<= 1;
          shift++;
          }
        h_mant &= 0x3FF;
        f_exp -= shift - 1;
        }
      }

    f32.i = (sign << 31) | (f_exp << 23) | (h_mant << 13);
    x = f32.f;
    }



  inline explicit operator cl_half() const
    {
    const int rounding_mode = fegetround();

    // Type-punning to get direct access to underlying bits
    union
      {
      float f;
      uint32_t i;
      } f32;
    f32.f = x;

    // Extract sign bit
    uint16_t sign = f32.i >> 31;

    // Extract FP32 exponent and mantissa
    uint32_t f_exp = (f32.i >> (FLT_MANT_DIG - 1)) & 0xFF;
    uint32_t f_mant = f32.i & ((1 << (FLT_MANT_DIG - 1)) - 1);

    // Remove FP32 exponent bias
    int32_t exp = f_exp - CL_FLT_MAX_EXP + 1;

    // Add FP16 exponent bias
    uint16_t h_exp = (uint16_t)(exp + CL_HALF_MAX_EXP - 1);

    // Position of the bit that will become the FP16 mantissa LSB
    uint32_t lsb_pos = FLT_MANT_DIG - CL_HALF_MANT_DIG;

    // Check for NaN / infinity
    if (f_exp == 0xFF)
      {
      if (f_mant)
        {
        // NaN -> propagate mantissa and silence it
        uint16_t h_mant = (uint16_t)(f_mant >> lsb_pos);
        h_mant |= 0x200;
        return (sign << 15) | CL_HALF_EXP_MASK | h_mant;
        }
      else
        {
        // Infinity -> zero mantissa
        return (sign << 15) | CL_HALF_EXP_MASK;
        }
      }

    // Check for zero
    if (!f_exp && !f_mant)
      {
      return (sign << 15);
      }

    // Check for overflow
    if (exp >= CL_HALF_MAX_EXP)
      {
      //
      // inlined cl_half_handle_overflow():
      //
      if (rounding_mode == FE_TOWARDZERO)
        {
        // Round overflow towards zero -> largest finite number (preserving sign)
        return (sign << 15) | CL_HALF_MAX_FINITE_MAG;
        }
      else if (rounding_mode == FE_UPWARD && sign)
        {
        // Round negative overflow towards positive infinity -> most negative finite number
        return (1 << 15) | CL_HALF_MAX_FINITE_MAG;
        }
      else if (rounding_mode == FE_DOWNWARD && !sign)
        {
        // Round positive overflow towards negative infinity -> largest finite number
        return CL_HALF_MAX_FINITE_MAG;
        }

      // Overflow to infinity
      return (sign << 15) | CL_HALF_EXP_MASK;
      }

    // Check for underflow
    if (exp < (CL_HALF_MIN_EXP - CL_HALF_MANT_DIG - 1))
      {
      //
      // inlined cl_half_handle_underflow()
      //
      if (rounding_mode == CL_HALF_RTP && !sign)
        {
        // Round underflow towards positive infinity -> smallest positive value
        return (sign << 15) | 1;
        }
      else if (rounding_mode == CL_HALF_RTN && sign)
        {
        // Round underflow towards negative infinity -> largest negative value
        return (sign << 15) | 1;
        }

      // Flush to zero
      return (sign << 15);
      }

    // Check for value that will become denormal
    if (exp < -14)
      {
      // Denormal -> include the implicit 1 from the FP32 mantissa
      h_exp = 0;
      f_mant |= 1 << (FLT_MANT_DIG - 1);

      // Mantissa shift amount depends on exponent
      lsb_pos = -exp + (FLT_MANT_DIG - 25);
      }

    // Generate FP16 mantissa by shifting FP32 mantissa
    uint16_t h_mant = (uint16_t)(f_mant >> lsb_pos);

    // Check whether we need to round
    uint32_t halfway = 1 << (lsb_pos - 1);
    uint32_t mask = (halfway << 1) - 1;
    switch (rounding_mode)
      {
      case FE_TONEAREST:
        if ((f_mant & mask) > halfway)
          {
          // More than halfway -> round up
          h_mant += 1;
          }
        else if ((f_mant & mask) == halfway)
          {
          // Exactly halfway -> round to nearest even
          if (h_mant & 0x1)
          h_mant += 1;
          }
        break;
      case FE_TOWARDZERO:
        // Mantissa has already been truncated -> do nothing
        break;
      case FE_UPWARD:
        if ((f_mant & mask) && !sign)
          {
          // Round positive numbers up
          h_mant += 1;
          }
        break;
      case FE_DOWNWARD:
        if ((f_mant & mask) && sign)
          {
          // Round negative numbers down
          h_mant += 1;
          }
        break;
      }

    // Check for mantissa overflow
    if (h_mant & 0x400)
      {
      h_exp += 1;
      h_mant = 0;
      }

    return (sign << 15) | (h_exp << 10) | h_mant;
    }
  #ifdef COOT_UNDEF_CL_HALF_EXP_MASK
    #undef COOT_UNDEF_CL_HALF_EXP_MASK
    #undef CL_HALF_EXP_MASK
  #endif
  #ifdef COOT_UNDEF_CL_HALF_MAX_FINITE_MAG
    #undef COOT_UNDEF_CL_HALF_MAX_FINITE_MAG
    #undef CL_HALF_MAX_FINITE_MAG
  #endif
  #endif



  //
  // We only need to provide in-place operators; for all other operators,
  // the shim will be implicitly cast to a float and handled from there.
  //

  template<typename T>
  typename enable_if2<std::is_arithmetic<T>::value, fp16_shim&>::result
  operator+=(const T& in_x)
    {
    x += in_x;
    return *this;
    }



  fp16_shim&
  operator+=(const fp16_shim& in_x)
    {
    x += in_x.x;
    return *this;
    }



  template<typename T>
  typename enable_if2<std::is_arithmetic<T>::value, fp16_shim&>::result
  operator-=(const T& in_x)
    {
    x -= in_x;
    return *this;
    }



  fp16_shim&
  operator-=(const fp16_shim& in_x)
    {
    x -= in_x.x;
    return *this;
    }



  template<typename T>
  typename enable_if2<std::is_arithmetic<T>::value, fp16_shim&>::result
  operator*=(const T& in_x)
    {
    x *= in_x;
    return *this;
    }



  fp16_shim&
  operator*=(const fp16_shim& in_x)
    {
    x *= in_x.x;
    return *this;
    }



  template<typename T>
  typename enable_if2<std::is_arithmetic<T>::value, fp16_shim&>::result
  operator/=(const T& in_x)
    {
    x /= in_x;
    return *this;
    }



  fp16_shim&
  operator/=(const fp16_shim& in_x)
    {
    x /= in_x.x;
    return *this;
    }



  fp16_shim&
  operator++()
    {
    ++x;
    return *this;
    }



  fp16_shim&
  operator++(int)
    {
    x++;
    return *this;
    }



  fp16_shim&
  operator--()
    {
    --x;
    return *this;
    }



  fp16_shim&
  operator--(int)
    {
    x--;
    return *this;
    }

  };



inline
bool
operator&&(const fp16_shim& a, const fp16_shim& b)
  {
  return a.x && b.x;
  }



inline
bool
operator||(const fp16_shim& a, const fp16_shim& b)
  {
  return a.x || b.x;
  }



inline
bool
operator==(const fp16_shim& a, const fp16_shim& b)
  {
  return a.x == b.x;
  }



inline
bool
operator!=(const fp16_shim& a, const fp16_shim& b)
  {
  return a.x != b.x;
  }



inline
bool
operator>(const fp16_shim& a, const fp16_shim& b)
  {
  return a.x > b.x;
  }



inline
bool
operator>=(const fp16_shim& a, const fp16_shim& b)
  {
  return a.x >= b.x;
  }



inline
bool
operator<(const fp16_shim& a, const fp16_shim& b)
  {
  return a.x < b.x;
  }



inline
bool
operator<=(const fp16_shim& a, const fp16_shim& b)
  {
  return a.x <= b.x;
  }



#if defined(COOT_USE_CUDA)
//
// CUDA provides implicit conversion to/from any integer type from __half,
// so the compiler will have all kinds of ambiguous overload problems for
// mixed-type arithmetic operations.  We provide these overloads because
// they will preferentially be chosen, fixing the ambiguous overload issue.
//
template<typename eT>
inline
typename
enable_if2
  <
  std::is_arithmetic<eT>::value,
  __half
  >::result
operator+(const eT& a, const __half& b)
  {
  return __half(a) + b;
  }



template<typename eT>
inline
typename
enable_if2
  <
  std::is_arithmetic<eT>::value,
  __half
  >::result
operator+(const __half& a, const eT& b)
  {
  return a + __half(b);
  }



template<typename eT>
inline
typename
enable_if2
  <
  std::is_arithmetic<eT>::value,
  __half
  >::result
operator-(const eT& a, const __half& b)
  {
  return __half(a) - b;
  }



template<typename eT>
inline
typename
enable_if2
  <
  std::is_arithmetic<eT>::value,
  __half
  >::result
operator-(const __half& a, const eT& b)
  {
  return a - __half(b);
  }



template<typename eT>
inline
typename
enable_if2
  <
  std::is_arithmetic<eT>::value,
  __half
  >::result
operator*(const eT& a, const __half& b)
  {
  return __half(a) * b;
  }



template<typename eT>
inline
typename
enable_if2
  <
  std::is_arithmetic<eT>::value,
  __half
  >::result
operator*(const __half& a, const eT& b)
  {
  return a * __half(b);
  }



template<typename eT>
inline
typename
enable_if2
  <
  std::is_arithmetic<eT>::value,
  __half
  >::result
operator/(const eT& a, const __half& b)
  {
  return __half(a) / b;
  }



template<typename eT>
inline
typename
enable_if2
  <
  std::is_arithmetic<eT>::value,
  __half
  >::result
operator/(const __half& a, const eT& b)
  {
  return a / __half(b);
  }



//
// We also need logical and/or operations for __half since CUDA does not provide them.
//



inline
bool
operator&&(const __half& a, const __half& b)
  {
  return float(a) && float(b);
  }



inline
bool
operator||(const __half& a, const __half& b)
  {
  return float(a) || float(b);
  }
#endif
