// Copyright 2019 Ryan Curtin (http://www.ratml.org/)
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

// Utility functions for generating random numbers via CUDA (cuRAND).

// TODO: allow setting the seed!


template<typename eT>
inline
void
fill_randu(dev_mem_t<eT> dest, const uword n)
  {
  coot_extra_debug_sigprint();

  if (n == 0) { return; }

  // For integral types, truncate to [0, 1] just like Armadillo.
  // We'll generate numbers using a floating-point type of the same width, then pass over it to truncate and cast back to the right type.
  if (std::is_same<eT, u32>::value || std::is_same<eT, s32>::value)
    {
    dev_mem_t<float> reinterpreted_mem;
    reinterpreted_mem.cuda_mem_ptr = (float*) dest.cuda_mem_ptr;
    fill_randu(reinterpreted_mem, n);
    copy_mat(dest, reinterpreted_mem, n, 1, 0, 0, n, 0, 0, n);
    }
  else if (std::is_same<eT, u64>::value || std::is_same<eT, s64>::value)
    {
    dev_mem_t<double> reinterpreted_mem;
    reinterpreted_mem.cuda_mem_ptr = (double*) dest.cuda_mem_ptr;
    fill_randu(reinterpreted_mem, n);
    copy_mat(dest, reinterpreted_mem, n, 1, 0, 0, n, 0, 0, n);
    }
  else
    {
    std::ostringstream oss;
    oss << "coot::cuda::fill_randu(): not implemented for type " << typeid(eT).name();
    coot_stop_runtime_error(oss.str());
    }
  }



template<>
inline
void
fill_randu(dev_mem_t<float> dest, const uword n)
  {
  coot_extra_debug_sigprint();

  if (n == 0) { return; }

  curandStatus_t result = coot_wrapper(curandGenerateUniform)(get_rt().cuda_rt.xorwow_rand, dest.cuda_mem_ptr, n);
  coot_check_curand_error(result, "coot::cuda::fill_randu(): curandGenerateUniform() failed");
  }



template<>
inline
void
fill_randu(dev_mem_t<double> dest, const uword n)
  {
  coot_extra_debug_sigprint();

  if (n == 0) { return; }

  curandStatus_t result = coot_wrapper(curandGenerateUniformDouble)(get_rt().cuda_rt.xorwow_rand, dest.cuda_mem_ptr, n);
  coot_check_curand_error(result, "coot::cuda::fill_randu(): curandGenerateUniform() failed");
  }



template<typename eT>
inline
void
fill_randn(dev_mem_t<eT> dest, const uword n, const double mu, const double sd)
  {
  coot_extra_debug_sigprint();

  if (n == 0) { return; }

  // For integral types, truncate just like Armadillo.
  // We'll generate numbers using a floating-point type of the same width, then pass over it to truncate and cast back to the right type.
  if (std::is_same<eT, u32>::value || std::is_same<eT, s32>::value)
    {
    dev_mem_t<float> reinterpreted_mem;
    reinterpreted_mem.cuda_mem_ptr = (float*) dest.cuda_mem_ptr;
    fill_randn(reinterpreted_mem, n, mu, sd);
    copy_mat(dest, reinterpreted_mem, n, 1, 0, 0, n, 0, 0, n);
    }
  else if (std::is_same<eT, u64>::value || std::is_same<eT, s64>::value)
    {
    dev_mem_t<double> reinterpreted_mem;
    reinterpreted_mem.cuda_mem_ptr = (double*) dest.cuda_mem_ptr;
    fill_randn(reinterpreted_mem, n, mu, sd);
    copy_mat(dest, reinterpreted_mem, n, 1, 0, 0, n, 0, 0, n);
    }
  else
    {
    std::ostringstream oss;
    oss << "coot::cuda::fill_randn(): not implemented for type " << typeid(eT).name();
    coot_stop_runtime_error(oss.str());
    }
  }



template<>
inline
void
fill_randn(dev_mem_t<float> dest, const uword n, const double mu, const double sd)
  {
  coot_extra_debug_sigprint();

  if (n == 0) { return; }

  curandStatus_t result = coot_wrapper(curandGenerateNormal)(get_rt().cuda_rt.philox_rand, dest.cuda_mem_ptr, n, mu, sd);
  coot_check_curand_error(result, "coot::cuda::fill_randn(): curandGenerateNormal() failed");
  }



template<>
inline
void
fill_randn(dev_mem_t<double> dest, const uword n, const double mu, const double sd)
  {
  coot_extra_debug_sigprint();

  if (n == 0) { return; }

  curandStatus_t result = coot_wrapper(curandGenerateNormalDouble)(get_rt().cuda_rt.philox_rand, dest.cuda_mem_ptr, n, mu, sd);
  coot_check_curand_error(result, "coot::cuda::fill_randn(): curandGenerateNormalDouble() failed");
  }



// 32-bit version
template<typename eT>
inline
void
fill_randi(dev_mem_t<eT> dest, const uword n, const int lo, const int hi, const typename enable_if<std::is_same<typename uint_type<eT>::result, u32>::value>::result* junk = nullptr)
  {
  coot_extra_debug_sigprint();

  if (n == 0) { return; }

  // 32-bit types may have a smaller effective range.  (But not if they are floating point.)
  const u32 bounded_hi = (std::is_floating_point<eT>::value) ? hi : std::min((u32) hi, (u32) std::numeric_limits<eT>::max());

  // Strategy: generate completely random bits in [0, u32_MAX]; modulo down to [0, range]; add lo to finally get [lo, hi];
  // then make sure the return type is correct.
  //
  // This overload uses curandGenerate(), which makes 32-bit unsigned integers.

  dev_mem_t<u32> u32_dest;
  u32_dest.cuda_mem_ptr = (u32*) dest.cuda_mem_ptr;

  const u32 range = (bounded_hi - lo);
  curandStatus_t result = coot_wrapper(curandGenerate)(get_rt().cuda_rt.xorwow_rand, u32_dest.cuda_mem_ptr, n);
  coot_check_curand_error(result, "coot::cuda::fill_randi(): curandGenerate() failed");

  // [0, u32_MAX] --> [0, range] (only needed if range != u32_MAX)
  if (range != std::numeric_limits<u32>::max())
    {
    eop_scalar(twoway_kernel_id::equ_array_mod_scalar,
               u32_dest, u32_dest,
               (u32) range + 1, (u32) range + 1,
               n, 1, 1,
               0, 0, 0, n, 1,
               0, 0, 0, n, 1);
    }

  // Now cast it to the correct type, if needed.
  if (!std::is_same<eT, u32>::value)
    {
    copy_mat(dest, u32_dest, n, 1, 0, 0, n, 0, 0, n);
    }

  // [0, range] --> [lo, hi]
  // We do this after the casting, in case eT is a signed type and lo < 0.
  if (lo != 0)
    {
    eop_scalar(twoway_kernel_id::equ_array_plus_scalar,
               dest, dest,
               (eT) lo, (eT) 0,
               n, 1, 1,
               0, 0, 0, n, 1,
               0, 0, 0, n, 1);
    }
  }



// 64-bit version
template<typename eT>
inline
void
fill_randi(dev_mem_t<eT> dest, const uword n, const int lo, const int hi, const typename enable_if<std::is_same<typename uint_type<eT>::result, u64>::value>::result* junk = nullptr)
  {
  coot_extra_debug_sigprint();

  if (n == 0) { return; }

  // Strategy: generate completely random bits in [0, u64_MAX]; modulo down to [0, range]; add lo to finally get [lo, hi];
  // then make sure the return type is correct.

  dev_mem_t<u64> u64_dest;
  u64_dest.cuda_mem_ptr = (u64*) dest.cuda_mem_ptr;

  const u64 range = (hi - lo);
  // We use the 32-bit XORWOW generator, but just generate a sequence of twice the usual length (since we are generating for u64s not u32s).
  curandStatus_t result = coot_wrapper(curandGenerate)(get_rt().cuda_rt.xorwow_rand, (u32*) dest.cuda_mem_ptr, 2 * n);
  coot_check_curand_error(result, "coot::cuda::fill_randi(): curandGenerate() failed");

  // [0, u64_MAX] --> [0, range] (only needed if range != u64_MAX)
  if (range != std::numeric_limits<u64>::max())
    {
    eop_scalar(twoway_kernel_id::equ_array_mod_scalar,
               u64_dest, u64_dest,
               (u64) range + 1, (u64) range + 1,
               n, 1, 1,
               0, 0, 0, n, 1,
               0, 0, 0, n, 1);
    }

  // Now cast it to the correct type, if needed.
  if (!std::is_same<eT, u64>::value)
    {
    copy_mat(dest, u64_dest, n, 1, 0, 0, n, 0, 0, n);
    }

  // [0, range] --> [lo, hi]
  // We do this after the casting, in case eT is a signed type and lo < 0.
  if (lo != 0)
    {
    eop_scalar(twoway_kernel_id::equ_array_plus_scalar,
               dest, dest,
               (eT) lo, (eT) 0,
               n, 1, 1,
               0, 0, 0, n, 1,
               0, 0, 0, n, 1);
    }
  }
