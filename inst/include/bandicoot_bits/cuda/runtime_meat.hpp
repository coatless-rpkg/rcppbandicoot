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

// utility functions for compiled-on-the-fly CUDA kernels



#define STR2(A) STR(A)
#define STR(A) #A



inline
bool
runtime_t::init(const bool /* manual_selection */, const uword wanted_platform, const uword wanted_device, const bool /* print_info */)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (wanted_platform != 0), "coot::cuda_rt.init(): wanted_platform must be 0 for the CUDA backend" );

  valid = false;

  CUresult result = coot_wrapper(cuInit)(0);
  coot_check_cuda_error(result, "coot::cuda_rt.init(): cuInit() failed");

  int device_count = 0;
  result = coot_wrapper(cuDeviceGetCount)(&device_count);
  coot_check_cuda_error(result, "coot::cuda_rt.init(): cuDeviceGetCount() failed");

  // Ensure that the desired device is within the range of devices we have.
  // TODO: better error message?
  coot_debug_check( ((int) wanted_device >= device_count), "coot::cuda_rt.init(): invalid wanted_device" );

  result = coot_wrapper(cuDeviceGet)(&cuDevice, wanted_device);
  coot_check_cuda_error(result, "coot::cuda_rt.init(): cuDeviceGet() failed");

  #if CUDA_VERSION >= 13000
  result = coot_wrapper(cuCtxCreate)(&context, NULL, 0, cuDevice);
  #else
  result = coot_wrapper(cuCtxCreate)(&context, 0, cuDevice);
  #endif
  coot_check_cuda_error(result, "coot::cuda_rt.init(): cuCtxCreate() failed");

  // NOTE: it seems size_t will have the same size on the device and host;
  // given the definition of uword, we will assume uword on the host is equivalent
  // to size_t on the device.
  //
  // NOTE: float will also have the same size as the host (generally 32 bits)
  cudaError_t result2 = coot_wrapper(cudaGetDeviceProperties)(&dev_prop, wanted_device);
  coot_check_cuda_error(result2, "coot::cuda_rt.init(): couldn't get device properties");

  generate_unique_host_device_id();

  // Initialize RNG struct.
  curandStatus_t result3;
  result3 = coot_wrapper(curandCreateGenerator)(&xorwow_rand, CURAND_RNG_PSEUDO_XORWOW);
  coot_check_curand_error(result3, "coot::cuda_rt.init(): curandCreateGenerator() failed");
  result3 = coot_wrapper(curandCreateGenerator)(&philox_rand, CURAND_RNG_PSEUDO_PHILOX4_32_10);
  coot_check_curand_error(result3, "coot::cuda_rt.init(): curandCreateGenerator() failed");

  // Initialize cuBLAS.
  coot_wrapper(cublasCreate)(&cublas_handle);

  cusolverStatus_t status = coot_wrapper(cusolverDnCreate)(&cusolver_handle);
  coot_check_cusolver_error(status, "coot::cuda::chol(): cusolverDnCreate() failed");

  // Set up options for compilation.
  nvrtc_opts = { std::string("--fmad=false") };

  // Get compute capabilities.
  int major, minor = 0;
  result = coot_wrapper(cuDeviceGetAttribute)(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice);
  coot_check_cuda_error(result, "coot::cuda_rt.init(): cuDeviceGetAttribute() failed");
  result = coot_wrapper(cuDeviceGetAttribute)(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice);
  coot_check_cuda_error(result, "coot::cuda_rt.init(): cuDeviceGetAttribute() failed");
  int card_arch = 10 * major + minor; // hopefully this does not change in future versions of the CUDA toolkit...

  // Get the supported architectures.
  int num_archs = 0;
  nvrtcResult result4 = coot_wrapper(nvrtcGetNumSupportedArchs)(&num_archs);
  coot_check_nvrtc_error(result4, "coot::cuda_rt.init(): nvrtcGetNumSupportedArchs() failed");
  int* archs = new int[num_archs];
  result4 = coot_wrapper(nvrtcGetSupportedArchs)(archs);
  coot_check_nvrtc_error(result4, "coot::cuda_rt.init(): nvrtcGetSupportedArchs() failed");

  // We will use the maximum architecture supported by both NVRTC and the card.
  // This is based on the assumption that all architectures are backwards-compatible;
  // I can't seem to find this in writing in the NVIDIA documentation but it appears to be true.
  int use_arch = archs[0];
  for (int i = 0; i < num_archs; ++i)
    {
    if (archs[i] > use_arch && archs[i] <= card_arch)
      {
      use_arch = archs[i];
      }
    }

  // Compute capability 5.3 is necessary for FP16 support; see
  // https://developer.nvidia.com/blog/mixed-precision-programming-cuda-8/ for
  // reference.
  has_fp16 = (use_arch >= 53);

  delete[] archs;

  std::stringstream gpu_arch_opt;
  gpu_arch_opt << "--gpu-architecture=sm_" << use_arch;
  gpu_arch_str = gpu_arch_opt.str();
  nvrtc_opts.push_back(gpu_arch_str);

  // Now add all the necessary include directories.
  std::string incl_dirs(STR2(COOT_CUDA_INCLUDE_PATH));
  std::istringstream incl_str(incl_dirs);
  std::string dir;
  while (std::getline(incl_str, dir, ';'))
    {
    nvrtc_opts.push_back(std::string("-I") + dir);
    }

  src_preamble = kernel_src::init_src_preamble(has_fp16);

  valid = true;

  return true;

  // TODO: destroy context in destructor
  }



inline
void
runtime_t::generate_unique_host_device_id()
  {
  // Generate a string that corresponds to this specific device and CUDA version.
  // We'll use the UUID of the device, and the version of the runtime.
  std::ostringstream oss;
  int runtime_version;
  cudaError_t result = coot_wrapper(cudaRuntimeGetVersion)(&runtime_version);
  coot_check_cuda_error(result, "coot::cuda_rt.unique_host_device_id(): cudaRuntimeGetVersion() failed");
  // Print each half-byte in hex.
  for (size_t i = 0; i < 16; i++)
    {
    oss << std::setw(2) << std::setfill('0') << std::hex << ((unsigned int) dev_prop.uuid.bytes[i] & 0xFF);
    }
  oss << "_" << std::dec << runtime_version;
  unique_host_device_id = oss.str();
  }



inline
bool
runtime_t::load_cached_kernel(const std::string& kernel_name, CUfunction& function)
  {
  coot_extra_debug_sigprint();

  // First check the cache to see if we've already built this kernel before.
  const size_t kernel_size = cache::has_cached_kernel(unique_host_device_id, kernel_name);
  if (kernel_size == 0)
    {
    // We don't have the kernel.
    return false;
    }

  // Allocate a buffer large enough to store the program.
  char* kernel_buffer = new char[kernel_size];
  bool status = cache::read_cached_kernel(unique_host_device_id, kernel_name, (unsigned char*) kernel_buffer);
  if (status == false)
    {
    coot_debug_warn("coot::cuda::load_cached_kernel(): could not load kernel '" + kernel_name + "' for unique host device id '" + unique_host_device_id + "'");
    delete[] kernel_buffer;
    return false;
    }

  CUresult result = coot_wrapper(cuInit)(0);
  CUmodule module;
  result = coot_wrapper(cuModuleLoadDataEx)(&module, kernel_buffer, 0, 0, 0);
  coot_check_cuda_error(result, "coot::cuda::load_cached_kernel(): cuModuleLoadDataEx() failed");

  result = coot_wrapper(cuModuleGetFunction)(&function, module, kernel_name.c_str());
  coot_check_cuda_error(result, "coot::cuda::load_cached_kernel(): cuModuleGetFunction() failed");

  delete[] kernel_buffer;
  return true;
  }



inline
runtime_t::~runtime_t()
  {
  if (valid)
    {
    // Clean up RNGs.
    curandStatus_t status = coot_wrapper(curandDestroyGenerator)(xorwow_rand);
    coot_check_curand_error(status, "coot::cuda_rt.cleanup(): curandDestroyGenerator() failed");
    status = coot_wrapper(curandDestroyGenerator)(philox_rand);
    coot_check_curand_error(status, "coot::cuda_rt.cleanup(): curandDestroyGenerator() failed");

    // Clean up cuBLAS handle.
    coot_wrapper(cublasDestroy)(cublas_handle);
    // Clean up cuSolver handle.
    coot_wrapper(cusolverDnDestroy)(cusolver_handle);
    }
  }



inline
std::string
runtime_t::generate_kernel(const zeroway_kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  std::string source =
      src_preamble +
      // There are no #defines for types.
      kernel_src::get_zeroway_source(num);

  return source;
  }



template<typename eT>
inline
std::string
runtime_t::generate_kernel(const oneway_kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  std::string source =
      src_preamble +
      kernel_src::get_oneway_defines<eT>() +
      kernel_src::get_oneway_source(num);

  return source;
  }



template<typename eT>
inline
std::string
runtime_t::generate_kernel(const oneway_real_kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  std::string source =
      src_preamble +
      kernel_src::get_oneway_defines<eT>() +
      kernel_src::get_oneway_real_source(num);

  return source;
  }



template<typename eT>
inline
std::string
runtime_t::generate_kernel(const oneway_integral_kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  std::string source =
      src_preamble +
      kernel_src::get_oneway_defines<eT>() +
      kernel_src::get_oneway_integral_source(num);

  return source;
  }



template<typename eT1, typename eT2>
inline
std::string
runtime_t::generate_kernel(const twoway_kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  std::string source =
      src_preamble +
      kernel_src::get_twoway_defines<eT2, eT1>() +
      kernel_src::get_twoway_source(num);

  return source;
  }



template<typename eT1, typename eT2, typename eT3>
inline
std::string
runtime_t::generate_kernel(const threeway_kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  std::string source =
      src_preamble +
      kernel_src::get_threeway_defines<eT3, eT2, eT1>() +
      kernel_src::get_threeway_source(num);

  return source;
  }



inline
void
runtime_t::compile_kernel(const std::string& kernel_name,
                          const std::string& source,
                          CUfunction& function)
  {
  // The kernel's not in the cache, so we'll use NVRTC to compile it.
  nvrtcProgram prog;
  nvrtcResult result = coot_wrapper(nvrtcCreateProgram)(
      &prog,               // CUDA runtime compilation program
      source.c_str(),      // CUDA program source
      kernel_name.c_str(), // CUDA program name
      0,                   // number of headers used
      NULL,                // sources of the headers
      NULL);               // name of each header
  coot_check_nvrtc_error(result, "coot::cuda::compile_kernel(): nvrtcCreateProgram() failed while compiling " + kernel_name);

  std::vector<const char*> nvrtc_opts_use(nvrtc_opts.size(), NULL);
  for (size_t i = 0; i < nvrtc_opts.size(); ++i)
    {
    nvrtc_opts_use[i] = nvrtc_opts[i].c_str();
    }

  result = coot_wrapper(nvrtcCompileProgram)(prog,                   // CUDA runtime compilation program
                                             nvrtc_opts_use.size(),  // number of compile options
                                             nvrtc_opts_use.data()); // compile options

  // If compilation failed, display what went wrong.  The NVRTC outputs aren't
  // always very helpful though...
  if (result != NVRTC_SUCCESS)
    {
    size_t logSize;
    result = coot_wrapper(nvrtcGetProgramLogSize)(prog, &logSize);
    coot_check_nvrtc_error(result, "coot::cuda::compile_kernel(): nvrtcGetProgramLogSize() failed while compiling " + kernel_name);

    char *log = new char[logSize];
    result = coot_wrapper(nvrtcGetProgramLog)(prog, log);
    coot_check_nvrtc_error(result, "coot::cuda::compile_kernel(): nvrtcGetProgramLog() failed while compiling " + kernel_name);

    coot_stop_runtime_error("coot::cuda::compile_kernel(): compilation of " + kernel_name + " kernel failed", std::string(log));
    }

  // Obtain CUBIN from the program.
  size_t cubin_size;
  result = coot_wrapper(nvrtcGetCUBINSize)(prog, &cubin_size);
  coot_check_nvrtc_error(result, "coot::cuda::compile_kernel(): nvrtcGetCUBINSize() failed while compiling " + kernel_name);

  char *cubin = new char[cubin_size];
  result = coot_wrapper(nvrtcGetCUBIN)(prog, cubin);
  coot_check_nvrtc_error(result, "coot::cuda::compile_kernel(): nvrtcGetCUBIN() failed while compiling " + kernel_name);

  CUmodule module;
  CUresult result2 = coot_wrapper(cuModuleLoadDataEx)(&module, cubin, 0, 0, 0);
  coot_check_cuda_error(result2, "coot::cuda::compile_kernel(): cuModuleLoadDataEx() failed while compiling " + kernel_name);

  // Now that everything is compiled, unpack the results into individual kernels
  // that we can access.
  result2 = coot_wrapper(cuModuleGetFunction)(&function, module, kernel_name.c_str());
  coot_check_cuda_error(result2, "coot::cuda::compile_kernel(): cuModuleGetFunction() failed while compiling " + kernel_name);

  // Save the kernel to the cache.
  const bool cache_result = cache::cache_kernel(unique_host_device_id, kernel_name, (unsigned char*) cubin, cubin_size);
  if (cache_result == false)
    {
    coot_debug_warn("coot::cuda::compile_kernel(): could not cache compiled CUDA kernel " + kernel_name);
    // This is not fatal, so we can proceed.
    }

  delete[] cubin;
  }



inline
const CUfunction&
runtime_t::get_kernel(const zeroway_kernel_id::enum_id num)
  {
  const std::tuple<bool, CUfunction&> t = get_kernel(zeroway_kernels, num);
  if (std::get<0>(t) == true)
    {
    const std::string name = rt_common::get_kernel_name(num);

    if (!load_cached_kernel(name, std::get<1>(t)))
      {
      // We will have to compile the kernel on the spot.
      const std::string source = generate_kernel(num);
      compile_kernel(name, source, std::get<1>(t));
      }
    }

  return std::get<1>(t);
  }



template<typename eT>
inline
const CUfunction&
runtime_t::get_kernel(const oneway_kernel_id::enum_id num)
  {
  const std::tuple<bool, CUfunction&> t = get_kernel<eT>(oneway_kernels, num);
  if (std::get<0>(t) == true)
    {
    const std::string name = rt_common::get_kernel_name<eT>(num);

    if (!load_cached_kernel(name, std::get<1>(t)))
      {
      // We will have to compile the kernel on the spot.
      const std::string source = generate_kernel<eT>(num);
      compile_kernel(name, source, std::get<1>(t));
      }
    }

  return std::get<1>(t);
  }



template<typename eT>
inline
const CUfunction&
runtime_t::get_kernel(const oneway_real_kernel_id::enum_id num)
  {
  const std::tuple<bool, CUfunction&> t = get_kernel<eT>(oneway_real_kernels, num);
  if (std::get<0>(t) == true)
    {
    const std::string name = rt_common::get_kernel_name<eT>(num);

    if (!load_cached_kernel(name, std::get<1>(t)))
      {
      // We will have to compile the kernel on the spot.
      const std::string source = generate_kernel<eT>(num);
      compile_kernel(name, source, std::get<1>(t));
      }
    }

  return std::get<1>(t);
  }



template<typename eT>
inline
const CUfunction&
runtime_t::get_kernel(const oneway_integral_kernel_id::enum_id num)
  {
  const std::tuple<bool, CUfunction&> t = get_kernel<eT>(oneway_integral_kernels, num);
  if (std::get<0>(t) == true)
    {
    const std::string name = rt_common::get_kernel_name<eT>(num);

    if (!load_cached_kernel(name, std::get<1>(t)))
      {
      // We will have to compile the kernel on the spot.
      const std::string source = generate_kernel<eT>(num);
      compile_kernel(name, source, std::get<1>(t));
      }
    }

  return std::get<1>(t);
  }



template<typename eT1, typename eT2>
inline
const CUfunction&
runtime_t::get_kernel(const twoway_kernel_id::enum_id num)
  {
  const std::tuple<bool, CUfunction&> t = get_kernel<eT1, eT2>(twoway_kernels, num);
  if (std::get<0>(t) == true)
    {
    const std::string name = rt_common::get_kernel_name<eT1, eT2>(num);

    if (!load_cached_kernel(name, std::get<1>(t)))
      {
      // We will have to compile the kernel on the spot.
      const std::string source = generate_kernel<eT1, eT2>(num);
      compile_kernel(name, source, std::get<1>(t));
      }
    }

  return std::get<1>(t);
  }



template<typename eT1, typename eT2, typename eT3>
inline
const CUfunction&
runtime_t::get_kernel(const threeway_kernel_id::enum_id num)
  {
  const std::tuple<bool, CUfunction&> t = get_kernel<eT1, eT2, eT3>(threeway_kernels, num);
  if (std::get<0>(t) == true)
    {
    const std::string name = rt_common::get_kernel_name<eT1, eT2, eT3>(num);

    if (!load_cached_kernel(name, std::get<1>(t)))
      {
      // We will have to compile the kernel on the spot.
      const std::string source = generate_kernel<eT1, eT2, eT3>(num);
      compile_kernel(name, source, std::get<1>(t));
      }
    }

  return std::get<1>(t);
  }



template<typename eT1, typename... eTs, typename HeldType, typename EnumType>
inline
std::tuple<bool, CUfunction&>
runtime_t::get_kernel(rt_common::kernels_t<HeldType>& k, const EnumType num)
  {
  coot_extra_debug_sigprint();

       if(is_same_type<eT1,u8    >::yes)  { return get_kernel<eTs...>(  k.u8_kernels, num); }
  else if(is_same_type<eT1,s8    >::yes)  { return get_kernel<eTs...>(  k.s8_kernels, num); }
  else if(is_same_type<eT1,u16   >::yes)  { return get_kernel<eTs...>( k.u16_kernels, num); }
  else if(is_same_type<eT1,s16   >::yes)  { return get_kernel<eTs...>( k.s16_kernels, num); }
  else if(is_same_type<eT1,u32   >::yes)  { return get_kernel<eTs...>( k.u32_kernels, num); }
  else if(is_same_type<eT1,s32   >::yes)  { return get_kernel<eTs...>( k.s32_kernels, num); }
  else if(is_same_type<eT1,u64   >::yes)  { return get_kernel<eTs...>( k.u64_kernels, num); }
  else if(is_same_type<eT1,s64   >::yes)  { return get_kernel<eTs...>( k.s64_kernels, num); }
  else if(is_same_type<eT1,fp16  >::yes)  { return get_kernel<eTs...>(   k.h_kernels, num); }
  else if(is_same_type<eT1,float >::yes)  { return get_kernel<eTs...>(   k.f_kernels, num); }
  else if(is_same_type<eT1,double>::yes)  { return get_kernel<eTs...>(   k.d_kernels, num); }
  else if(is_same_type<eT1,uword >::yes)
    {
    // this can happen if uword != u32 or u64
    if (sizeof(uword) == sizeof(u32))
      {
      return get_kernel<eTs...>(k.u32_kernels, num);
      }
    else if (sizeof(uword) == sizeof(u64))
      {
      return get_kernel<eTs...>(k.u64_kernels, num);
      }
    else
      {
      // hopefully nobody ever, ever, ever sees this
      throw std::invalid_argument("coot::cuda_rt.get_kernel(): unknown size for uword");
      }
    }
  else if(is_same_type<eT1,sword >::yes)
    {
    if (sizeof(sword) == sizeof(s32))
      {
      return get_kernel<eTs...>(k.s32_kernels, num);
      }
    else if (sizeof(sword) == sizeof(s64))
      {
      return get_kernel<eTs...>(k.s64_kernels, num);
      }
    else
      {
      // hopefully nobody ever, ever, ever sees this
      throw std::invalid_argument("coot::cuda_rt.get_kernel(): unknown size for sword");
      }
    }
  else
    {
    coot_debug_check(true, "unsupported element type" );
    }
  }



template<typename EnumType>
inline
std::tuple<bool, CUfunction&>
runtime_t::get_kernel(std::unordered_map<EnumType, CUfunction>& kernels,
                      const EnumType num)
  {
  if (kernels.count(num) == 0)
    {
    kernels[num] = CUfunction();
    return std::forward_as_tuple(true, kernels[num]);
    }
  else
    {
    return std::forward_as_tuple(false, kernels[num]);
    }
  }



template<typename eT>
inline
typename cuda_type<eT>::type*
runtime_t::acquire_memory(const uword n_elem)
  {
  void* result;
  typedef typename cuda_type<eT>::type ceT;
  cudaError_t error = coot_wrapper(cudaMalloc)(&result, sizeof(ceT) * n_elem);

  coot_check_cuda_error(error, "coot::cuda_rt.acquire_memory(): couldn't allocate memory");

  return (ceT*) result;
  }



template<typename eT>
inline
void
runtime_t::release_memory(eT* cuda_mem)
  {
  if(cuda_mem)
    {
    cudaError_t error = coot_wrapper(cudaFree)(cuda_mem);

    coot_check_cuda_error(error, "coot::cuda_rt.release_memory(): couldn't free memory");
    }
  }



inline
void
runtime_t::synchronise()
  {
  coot_wrapper(cuCtxSynchronize)();
  }



inline
void
runtime_t::set_rng_seed(const u64 seed)
  {
  coot_extra_debug_sigprint();

  curandStatus_t status = coot_wrapper(curandSetPseudoRandomGeneratorSeed)(xorwow_rand, seed);
  coot_check_curand_error(status, "cuda::set_rng_seed(): curandSetPseudoRandomGeneratorSeed() failed");

  status = coot_wrapper(curandSetPseudoRandomGeneratorSeed)(philox_rand, seed);
  coot_check_curand_error(status, "cuda::set_rng_seed(): curandSetPseudoRandomGeneratorSeed() failed");
  }



template<typename eT>
inline
bool
runtime_t::is_supported_type() const
  {
  if (is_fp16<eT>::value)
    {
    return has_fp16;
    }
  else
    {
    return true;
    }
  }



#undef STR2
#undef STR
