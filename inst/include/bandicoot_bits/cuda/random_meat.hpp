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
  coot_debug_sigprint();

  if (n == 0) { return; }

  // For integral types, truncate to [0, 1] just like Armadillo.
  // We'll generate numbers using a floating-point type of the same width, then pass over it to truncate and cast back to the right type.
  if (is_same_type<eT, u32>::yes || is_same_type<eT, s32>::yes)
    {
    dev_mem_t<float> reinterpreted_mem({{ NULL, 0 }});
    reinterpreted_mem.cuda_mem_ptr = (float*) dest.cuda_mem_ptr;
    fill_randu(reinterpreted_mem, n);

    // TODO: when randu is replaced with a generated Proxy kernel this can be fixed
    const Col<eT> dest_alias(dest, n);
    const Col<float> src_alias(reinterpreted_mem, n);
    copy(make_proxy(dest_alias), make_proxy(src_alias));
    }
  else if (is_same_type<eT, u64>::yes || is_same_type<eT, s64>::yes)
    {
    dev_mem_t<double> reinterpreted_mem({{ NULL, 0 }});
    reinterpreted_mem.cuda_mem_ptr = (double*) dest.cuda_mem_ptr;
    fill_randu(reinterpreted_mem, n);

    // TODO: when randu is replaced with a generated Proxy kernel this can be fixed
    const Col<eT> dest_alias(dest, n);
    const Col<double> src_alias(reinterpreted_mem, n);
    copy(make_proxy(dest_alias), make_proxy(src_alias));
    }
  else if (!is_real<eT>::value || is_same_type<eT, fp16>::yes)
    {
    // Unfortunately allocating a temporary matrix is our best strategy here.
    dev_mem_t<float>            tmp_mem({{ NULL, 0 }});
    runtime_t::mem_array<float> tmp_mem_array(n);

    tmp_mem.cuda_mem_ptr = tmp_mem_array.memptr();
    fill_randu(tmp_mem, n);

    // TODO: when randu is replaced with a generated Proxy kernel this can be fixed
    const Col<eT> dest_alias(dest, n);
    const Col<float> src_alias(tmp_mem, n);
    copy(make_proxy(dest_alias), make_proxy(src_alias));
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
  coot_debug_sigprint();

  if (n == 0) { return; }

  curandStatus_t result = coot_wrapper(curandGenerateUniform)(get_rt().cuda_rt.xorwow_rand, dest.cuda_mem_ptr, n);
  coot_check_curand_error(result, "coot::cuda::fill_randu(): curandGenerateUniform() failed");
  }



template<>
inline
void
fill_randu(dev_mem_t<double> dest, const uword n)
  {
  coot_debug_sigprint();

  if (n == 0) { return; }

  curandStatus_t result = coot_wrapper(curandGenerateUniformDouble)(get_rt().cuda_rt.xorwow_rand, dest.cuda_mem_ptr, n);
  coot_check_curand_error(result, "coot::cuda::fill_randu(): curandGenerateUniform() failed");
  }



template<typename eT>
inline
void
fill_randn(dev_mem_t<eT> dest, const uword n, const double mu, const double sd)
  {
  coot_debug_sigprint();

  if (n == 0) { return; }

  // For integral types, truncate just like Armadillo.
  // We'll generate numbers using a floating-point type of the same width, then pass over it to truncate and cast back to the right type.
  if (is_same_type<eT, u32>::yes || is_same_type<eT, s32>::yes)
    {
    dev_mem_t<float> reinterpreted_mem({{ NULL, 0 }});
    reinterpreted_mem.cuda_mem_ptr = (float*) dest.cuda_mem_ptr;
    fill_randn(reinterpreted_mem, n, mu, sd);

    // TODO: when randn is replaced with a generated Proxy kernel this can be fixed
    const Col<eT> dest_alias(dest, n);
    const Col<float> src_alias(reinterpreted_mem, n);
    copy(make_proxy(dest_alias), make_proxy(src_alias));
    }
  else if (is_same_type<eT, u64>::yes || is_same_type<eT, s64>::yes)
    {
    dev_mem_t<double> reinterpreted_mem({{ NULL, 0 }});
    reinterpreted_mem.cuda_mem_ptr = (double*) dest.cuda_mem_ptr;
    fill_randn(reinterpreted_mem, n, mu, sd);

    // TODO: when randn is replaced with a generated Proxy kernel this can be fixed
    const Col<eT> dest_alias(dest, n);
    const Col<double> src_alias(reinterpreted_mem, n);
    copy(make_proxy(dest_alias), make_proxy(src_alias));
    }
  else if (!is_real<eT>::value || is_same_type<eT, fp16>::yes)
    {
    // Unfortunately allocating a temporary matrix is our best strategy here.
    dev_mem_t<float>            tmp_mem({{ NULL, 0 }});
    runtime_t::mem_array<float> tmp_mem_array(n);

    tmp_mem.cuda_mem_ptr = tmp_mem_array.memptr();
    fill_randn(tmp_mem, n, mu, sd);

    // TODO: when randn is replaced with a generated Proxy kernel this can be fixed
    const Col<eT> dest_alias(dest, n);
    const Col<float> src_alias(tmp_mem, n);
    copy(make_proxy(dest_alias), make_proxy(src_alias));
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
  coot_debug_sigprint();

  if (n == 0) { return; }

  curandStatus_t result = coot_wrapper(curandGenerateNormal)(get_rt().cuda_rt.philox_rand, dest.cuda_mem_ptr, n, mu, sd);
  coot_check_curand_error(result, "coot::cuda::fill_randn(): curandGenerateNormal() failed");
  }



template<>
inline
void
fill_randn(dev_mem_t<double> dest, const uword n, const double mu, const double sd)
  {
  coot_debug_sigprint();

  if (n == 0) { return; }

  curandStatus_t result = coot_wrapper(curandGenerateNormalDouble)(get_rt().cuda_rt.philox_rand, dest.cuda_mem_ptr, n, mu, sd);
  coot_check_curand_error(result, "coot::cuda::fill_randn(): curandGenerateNormalDouble() failed");
  }



// 32-bit version
template<typename eT>
inline
void
fill_randi(dev_mem_t<eT> dest, const uword n, const int lo, const int hi, const typename enable_if<is_same_type<typename promote_type<typename uint_type<eT>::result, u32>::result, u32>::yes>::result* junk = nullptr)
  {
  coot_debug_sigprint();
  coot_ignore(junk);

  if (n == 0) { return; }

  // Strategy: generate completely random bits in [0, u32_MAX]; modulo down to [0, range]; add lo to finally get [lo, hi];
  // then make sure the return type is correct.
  //
  // This overload uses curandGenerate(), which makes 32-bit unsigned integers.

  dev_mem_t<u32> u32_dest({{ NULL, 0 }});
  u32_dest.cuda_mem_ptr = (u32*) dest.cuda_mem_ptr;

  curandStatus_t result = coot_wrapper(curandGenerate)(get_rt().cuda_rt.xorwow_rand, u32_dest.cuda_mem_ptr, n / (sizeof(u32) / sizeof(eT)));
  coot_check_curand_error(result, "coot::cuda::fill_randi(): curandGenerate() failed");

  typedef typename uint_type<eT>::result ueT;
  dev_mem_t<ueT> ueT_dest({{ NULL, 0 }});
  ueT_dest.cuda_mem_ptr = (ueT*) dest.cuda_mem_ptr;

  // 32-bit types may have a smaller effective range.  (But not if they are floating point.)
  const ueT bounded_hi = (is_real<eT>::value) ? hi : std::min((ueT) hi, (ueT) Datum<eT>::max);
  const ueT range = (bounded_hi - lo);

  // [0, ueT_MAX] --> [0, range] (only needed if range != ueT_MAX)
  if (range != Datum<ueT>::max)
    {
    Mat<ueT> ueT_dest_alias(ueT_dest, n, 1);
    eop_modulo::apply(ueT_dest_alias, eOp<Mat<ueT>, eop_modulo>(ueT_dest_alias, (ueT) (range + 1)));
    }

  // Now cast it to the correct type, if needed.
  if (is_same_type<eT, ueT>::no)
    {
    // TODO: when randi uses Proxy arguments, this can be cleaned up
    const Col<eT> dest_alias(dest, n);
    const Col<ueT> src_alias(ueT_dest, n);
    copy(make_proxy(dest_alias), make_proxy(src_alias));
    }

  // [0, range] --> [lo, hi]
  // We do this after the casting, in case eT is a signed type and lo < 0.
  if (lo != 0)
    {
    Mat<eT> dest_alias(dest, n, 1);
    dest_alias += (eT) lo;
    }
  }



// 64-bit version
template<typename eT>
inline
void
fill_randi(dev_mem_t<eT> dest, const uword n, const int lo, const int hi, const typename enable_if<is_same_type<typename uint_type<eT>::result, u64>::yes>::result* junk = nullptr)
  {
  coot_debug_sigprint();
  coot_ignore(junk);

  if (n == 0) { return; }

  // Strategy: generate completely random bits in [0, u64_MAX]; modulo down to [0, range]; add lo to finally get [lo, hi];
  // then make sure the return type is correct.

  dev_mem_t<u64> u64_dest({{ NULL, 0 }});
  u64_dest.cuda_mem_ptr = (u64*) dest.cuda_mem_ptr;

  const u64 range = (hi - lo);
  // We use the 32-bit XORWOW generator, but just generate a sequence of twice the usual length (since we are generating for u64s not u32s).
  curandStatus_t result = coot_wrapper(curandGenerate)(get_rt().cuda_rt.xorwow_rand, (u32*) dest.cuda_mem_ptr, 2 * n);
  coot_check_curand_error(result, "coot::cuda::fill_randi(): curandGenerate() failed");

  // [0, u64_MAX] --> [0, range] (only needed if range != u64_MAX)
  if (range != Datum<u64>::max)
    {
    Mat<u64> u64_dest_alias(u64_dest, n, 1);
    eop_modulo::apply(u64_dest_alias, eOp<Mat<u64>, eop_modulo>(u64_dest_alias, (u64) (range + 1)));
    }

  // Now cast it to the correct type, if needed.
  if (is_same_type<eT, u64>::no)
    {
    // TODO: when randi uses Proxy arguments, this can be cleaned up
    const Col<eT> dest_alias(dest, n);
    const Col<u64> src_alias(u64_dest, n);
    copy(make_proxy(dest_alias), make_proxy(src_alias));
    }

  // [0, range] --> [lo, hi]
  // We do this after the casting, in case eT is a signed type and lo < 0.
  if (lo != 0)
    {
    Mat<eT> dest_alias(dest, n, 1);
    dest_alias += (eT) lo;
    }
  }
