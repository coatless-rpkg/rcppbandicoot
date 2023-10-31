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

// Utility functions for generating random numbers via OpenCL.

template<typename eT> struct preferred_rng { };
template<> struct preferred_rng<u32> { typedef std::mt19937 result; };
template<> struct preferred_rng<u64> { typedef std::mt19937_64 result; };

// Should be called with an unsigned int as the eT type.
template<typename eT>
inline
void
init_xorwow_state(cl_mem xorwow_state, const size_t num_rng_threads, const u64 seed)
  {
  coot_extra_debug_sigprint();

  // Since the states are relatively small, and we only do the seeding once, we'll initialize the values on the CPU, then copy them over.
  // We ensure that all values are odd.
  eT* cpu_state = new eT[6 * num_rng_threads];
  const eT trunc_seed = eT(seed);
  typename preferred_rng<eT>::result rng(trunc_seed);
  for (size_t i = 0; i < 6 * num_rng_threads; ++i)
    {
    eT val = rng();
    if (val % 2 == 0)
      val += 1;
    cpu_state[i] = val;
    }

  // Copy the state to the GPU memory.
  dev_mem_t<eT> m;
  m.cl_mem_ptr = xorwow_state;
  copy_into_dev_mem(m, cpu_state, 6 * num_rng_threads);
  delete[] cpu_state;
  }



inline
void
init_philox_state(cl_mem philox_state, const size_t num_rng_threads, const u64 seed)
  {
  coot_extra_debug_sigprint();

  // Since the states are small, we seed on the CPU, and then transfer the memory.
  // For now we always initialize the counters to 0.  (TODO: should this be an option?)
  u32* cpu_state = new u32[6 * num_rng_threads];
  memset(cpu_state, 0, sizeof(u32) * 6 * num_rng_threads);
  const u32 trunc_seed = u32(seed);
  preferred_rng<u32>::result rng(trunc_seed);
  for (size_t i = 0; i < num_rng_threads; ++i)
    {
    cpu_state[6 * i + 4] = rng();
    cpu_state[6 * i + 5] = rng();
    }

  // Copy the state to the GPU.
  dev_mem_t<u32> m;
  m.cl_mem_ptr = philox_state;
  copy_into_dev_mem(m, cpu_state, 6 * num_rng_threads);
  delete[] cpu_state;
  }



template<typename eT>
inline
void
fill_randu(dev_mem_t<eT> dest, const uword n)
  {
  coot_extra_debug_sigprint();

  if (n == 0) { return; }

  // For integral types, truncate to [0, 1] just like Armadillo.

  // Get the kernel and set up to run it.
  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::inplace_xorwow_randu);

  runtime_t::cq_guard guard;
  runtime_t::adapt_uword n_cl(n);

  cl_int status = 0;

  cl_mem xorwow_state = get_rt().cl_rt.get_xorwow_state<eT>();

  status |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem), &(dest.cl_mem_ptr) );
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, sizeof(cl_mem), &(xorwow_state) );
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, n_cl.size,      n_cl.addr );

  // Each thread will do as many elements as it can.
  // This avoids memory synchronization issues, since each RNG state will be local to only a single run of the kernel.
  const size_t num_rng_threads = get_rt().cl_rt.get_num_rng_threads();
  const size_t num_threads = std::min(num_rng_threads, n);

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 1, NULL, &num_threads, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "randu()");
  }



template<typename eT>
inline
void
fill_randn(dev_mem_t<eT> dest, const uword n, const double mu, const double sd)
  {
  coot_extra_debug_sigprint();

  if (n == 0) { return; }

  // For integral types, we truncate just like Armadillo.

  // Get the kernel and set up to run it.
  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::inplace_philox_randn);

  runtime_t::cq_guard guard;
  runtime_t::adapt_uword n_cl(n);

  cl_int status = 0;

  cl_mem philox_state = get_rt().cl_rt.get_philox_state();

  typedef typename promote_type<eT, float>::result fp_eT1;
  fp_eT1 cl_mu(mu);
  fp_eT1 cl_sd(sd);

  status |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem), &(dest.cl_mem_ptr) );
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, sizeof(cl_mem), &(philox_state) );
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, n_cl.size,      n_cl.addr );
  status |= coot_wrapper(clSetKernelArg)(kernel, 3, sizeof(cl_mu),  &(cl_mu) );
  status |= coot_wrapper(clSetKernelArg)(kernel, 4, sizeof(cl_sd),  &(cl_sd) );

  // Each thread will do as many elements as it can.
  // This avoids memory synchronization issues, since each RNG state will be local to only a single run of the kernel.
  const size_t num_rng_threads = get_rt().cl_rt.get_num_rng_threads();
  const size_t num_threads = std::min(num_rng_threads, n);

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 1, NULL, &num_threads, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "randn()");
  }



template<typename eT>
inline
void
fill_randi(dev_mem_t<eT> dest, const uword n, const int lo, const int hi)
  {
  coot_extra_debug_sigprint();

  if (n == 0) { return; }

  // Get the kernel and set up to run it.
  cl_kernel kernel = get_rt().cl_rt.get_kernel<eT>(oneway_kernel_id::inplace_xorwow_randi);

  runtime_t::cq_guard guard;
  runtime_t::adapt_uword n_cl(n);

  cl_int status = 0;

  cl_mem xorwow_state = get_rt().cl_rt.get_xorwow_state<eT>();

  typedef typename uint_type<eT>::result uint_eT;
  uint_eT range;
  if (std::is_same<uint_eT, u32>::value)
    {
    uint_eT bounded_hi = (std::is_floating_point<eT>::value) ? hi : std::min((u32) hi, (u32) std::numeric_limits<eT>::max());
    range = (bounded_hi - lo);
    }
  else
    {
    range = (hi - lo);
    }
  // OpenCL kernels cannot use `bool` arguments.
  char needs_modulo = (range != std::numeric_limits<uint_eT>::max());
  eT cl_lo = eT(lo);

  status |= coot_wrapper(clSetKernelArg)(kernel, 0, sizeof(cl_mem),  &(dest.cl_mem_ptr) );
  status |= coot_wrapper(clSetKernelArg)(kernel, 1, sizeof(cl_mem),  &(xorwow_state) );
  status |= coot_wrapper(clSetKernelArg)(kernel, 2, n_cl.size,       n_cl.addr );
  status |= coot_wrapper(clSetKernelArg)(kernel, 3, sizeof(eT),      &cl_lo );
  status |= coot_wrapper(clSetKernelArg)(kernel, 4, sizeof(uint_eT), &range );
  status |= coot_wrapper(clSetKernelArg)(kernel, 5, sizeof(char),    &needs_modulo );

  // Each thread will do as many elements as it can.
  // This avoids memory synchronization issues, since each RNG state will be local to only a single run of the kernel.
  const size_t num_rng_threads = get_rt().cl_rt.get_num_rng_threads();
  const size_t num_threads = std::min(num_rng_threads, n);

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), kernel, 1, NULL, &num_threads, NULL, 0, NULL, NULL);

  coot_check_cl_error(status, "randi()");
  }
