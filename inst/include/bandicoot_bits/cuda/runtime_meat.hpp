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

inline
bool
runtime_t::init(const bool manual_selection, const uword wanted_platform, const uword wanted_device, const bool print_info)
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

  result = coot_wrapper(cuCtxCreate)(&context, 0, cuDevice);
  coot_check_cuda_error(result, "coot::cuda_rt.init(): cuCtxCreate() failed");

  // NOTE: it seems size_t will have the same size on the device and host;
  // given the definition of uword, we will assume uword on the host is equivalent
  // to size_t on the device.
  //
  // NOTE: float will also have the same size as the host (generally 32 bits)
  cudaError_t result2 = coot_wrapper(cudaGetDeviceProperties)(&dev_prop, wanted_device);
  coot_check_cuda_error(result2, "coot::cuda_rt.init(): couldn't get device properties");

  // Attempt to load cached kernels, if available.
  const std::string unique_host_id = unique_host_device_id();
  size_t cached_kernel_size = cache::has_cached_kernels(unique_host_id);
  bool load_success = false;
  if (cached_kernel_size > 0)
    {
    load_success = load_cached_kernels(unique_host_id, cached_kernel_size);
    if (!load_success)
      {
      coot_debug_warn("coot::cuda_rt.init(): couldn't load cached kernels for unique host id '" + unique_host_id + "'");
      }
    }

  if (cached_kernel_size == 0 || !load_success)
    {
    // compile_kernels() will also attempt to perform caching internally
    bool status = compile_kernels(unique_host_id);
    if (status == false)
      {
      coot_debug_warn("coot::cuda_rt.init(): couldn't set up CUDA kernels");
      return false;
      }
    }

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

  valid = true;

  return true;

  // TODO: destroy context in destructor
  }



inline
std::string
runtime_t::unique_host_device_id() const
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
  return oss.str();
  }



inline
bool
runtime_t::load_cached_kernels(const std::string& unique_host_device_id, const size_t kernel_size)
  {
  coot_extra_debug_sigprint();

  // Allocate a buffer large enough to store the program.
  char* kernel_buffer = new char[kernel_size];
  bool status = cache::read_cached_kernels(unique_host_device_id, (unsigned char*) kernel_buffer);
  if (status == false)
    {
    coot_debug_warn("coot::cuda_rt.init(): could not load kernels for unique host device id '" + unique_host_device_id + "'");
    delete[] kernel_buffer;
    return false;
    }

  // Create the map of kernel names.
  std::vector<std::pair<std::string, CUfunction*>> name_map;
  rt_common::init_zero_elem_kernel_map(zeroway_kernels, name_map, zeroway_kernel_id::get_names());
  rt_common::init_one_elem_real_kernel_map(oneway_real_kernels, name_map, oneway_real_kernel_id::get_names(), "", true);
  rt_common::init_one_elem_integral_kernel_map(oneway_integral_kernels, name_map, oneway_integral_kernel_id::get_names(), "");
  rt_common::init_one_elem_kernel_map(oneway_kernels, name_map, oneway_kernel_id::get_names(), "", true);
  rt_common::init_two_elem_kernel_map(twoway_kernels, name_map, twoway_kernel_id::get_names(), "", true);
  rt_common::init_three_elem_kernel_map(threeway_kernels, name_map, threeway_kernel_id::get_names(), "", true);

  status = create_kernels(name_map, kernel_buffer);
  delete[] kernel_buffer;
  return status;
  }



inline
bool
runtime_t::compile_kernels(const std::string& unique_host_device_id)
  {
  std::vector<std::pair<std::string, CUfunction*>> name_map;
  type_to_dev_string type_map;
  std::string source =
      get_cuda_src_preamble() +
      rt_common::get_zero_elem_kernel_src(zeroway_kernels, get_cuda_zeroway_kernel_src(), zeroway_kernel_id::get_names(), name_map, type_map) +
      rt_common::get_one_elem_real_kernel_src(oneway_real_kernels, get_cuda_oneway_real_kernel_src(), oneway_real_kernel_id::get_names(), "", name_map, type_map, true) +
      rt_common::get_one_elem_integral_kernel_src(oneway_integral_kernels, get_cuda_oneway_integral_kernel_src(), oneway_integral_kernel_id::get_names(), "", name_map, type_map) +
      rt_common::get_one_elem_kernel_src(oneway_kernels, get_cuda_oneway_kernel_src(), oneway_kernel_id::get_names(), "", name_map, type_map, true) +
      rt_common::get_two_elem_kernel_src(twoway_kernels, get_cuda_twoway_kernel_src(), twoway_kernel_id::get_names(), "", name_map, type_map, true) +
      rt_common::get_three_elem_kernel_src(threeway_kernels, get_cuda_threeway_kernel_src(), threeway_kernel_id::get_names(), name_map, type_map, true) +
      get_cuda_src_epilogue();

  // We'll use NVRTC to compile each of the kernels we need on the fly.
  nvrtcProgram prog;
  nvrtcResult result = coot_wrapper(nvrtcCreateProgram)(
      &prog,          // CUDA runtime compilation program
      source.c_str(), // CUDA program source
      "coot_kernels", // CUDA program name
      0,              // number of headers used
      NULL,           // sources of the headers
      NULL);          // name of each header
  coot_check_nvrtc_error(result, "coot::cuda_rt.init(): nvrtcCreateProgram() failed");

  std::vector<const char*> opts =
    {
    "--fmad=false",
    "-D UWORD=size_t",
    };

  // Get compute capabilities.
  int major, minor = 0;
  CUresult result2 = coot_wrapper(cuDeviceGetAttribute)(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice);
  coot_check_cuda_error(result2, "coot::cuda_rt.init(): cuDeviceGetAttribute() failed");
  result2 = coot_wrapper(cuDeviceGetAttribute)(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice);
  coot_check_cuda_error(result2, "coot::cuda_rt.init(): cuDeviceGetAttribute() failed");
  int card_arch = 10 * major + minor; // hopefully this does not change in future versions of the CUDA toolkit...

  // Get the supported architectures.
  int num_archs = 0;
  result = coot_wrapper(nvrtcGetNumSupportedArchs)(&num_archs);
  coot_check_nvrtc_error(result, "coot::cuda_rt.init(): nvrtcGetNumSupportedArchs() failed");
  int* archs = new int[num_archs];
  result = coot_wrapper(nvrtcGetSupportedArchs)(archs);
  coot_check_nvrtc_error(result, "coot::cuda_rt.init(): nvrtcGetSupportedArchs() failed");

  // We will use the maximum architecture supported by both NVRTC and the card.
  // This is based on the assumption that all architectures are backwards-compatible;
  // I can't seem to find this in writing in the NVIDIA documentation but it appears to be true.
  int use_arch = archs[0];
  for (size_t i = 0; i < num_archs; ++i)
    {
    if (archs[i] > use_arch && archs[i] <= card_arch)
      {
      use_arch = archs[i];
      }
    }

  std::stringstream gpu_arch_opt;
  gpu_arch_opt << "--gpu-architecture=sm_" << use_arch;
  const std::string& gpu_arch_opt_tmp = gpu_arch_opt.str();
  opts.push_back(gpu_arch_opt_tmp.c_str());

  result = coot_wrapper(nvrtcCompileProgram)(prog,         // CUDA runtime compilation program
                                             opts.size(),  // number of compile options
                                             opts.data()); // compile options

  // If compilation failed, display what went wrong.  The NVRTC outputs aren't
  // always very helpful though...
  if (result != NVRTC_SUCCESS)
    {
    size_t logSize;
    result = coot_wrapper(nvrtcGetProgramLogSize)(prog, &logSize);
    coot_check_nvrtc_error(result, "coot::cuda_rt.init(): nvrtcGetProgramLogSize() failed");

    char *log = new char[logSize];
    result = coot_wrapper(nvrtcGetProgramLog)(prog, log);
    coot_check_nvrtc_error(result, "coot::cuda_rt.init(): nvrtcGetProgramLog() failed");

    coot_stop_runtime_error("coot::cuda_rt.init(): compilation failed", std::string(log));
    }

  // Obtain CUBIN from the program.
  size_t cubin_size;
  result = coot_wrapper(nvrtcGetCUBINSize)(prog, &cubin_size);
  coot_check_nvrtc_error(result, "coot::cuda_rt.init(): nvrtcGetCUBINSize() failed");

  char *cubin = new char[cubin_size];
  result = coot_wrapper(nvrtcGetCUBIN)(prog, cubin);
  coot_check_nvrtc_error(result, "coot::cuda_rt.init(): nvrtcGetCUBIN() failed");

  bool create_kernel_result = create_kernels(name_map, cubin);

  if (create_kernel_result)
    {
    // Try to cache the kernels we compiled.
    const bool cache_result = cache::cache_kernels(unique_host_device_id, (unsigned char*) cubin, cubin_size);
    if (cache_result == false)
      {
      coot_debug_warn("coot::cuda_rt.init(): could not cache compiled CUDA kernels");
      // This is not fatal, so we can proceed.
      }
    }

  delete[] cubin;

  return create_kernel_result;
  }



inline
bool
runtime_t::create_kernels(const std::vector<std::pair<std::string, CUfunction*>>& name_map,
                          char* cubin)
  {
  CUresult result = coot_wrapper(cuInit)(0);
  CUmodule module;
  result = coot_wrapper(cuModuleLoadDataEx)(&module, cubin, 0, 0, 0);
  coot_check_cuda_error(result, "coot::cuda_rt.init(): cuModuleLoadDataEx() failed");

  // Now that everything is compiled, unpack the results into individual kernels
  // that we can access.
  for (uword i = 0; i < name_map.size(); ++i)
    {
    result = coot_wrapper(cuModuleGetFunction)(name_map.at(i).second, module, name_map.at(i).first.c_str());
    coot_check_cuda_error(result, "coot::cuda_rt.init(): cuModuleGetFunction() failed for function " + name_map.at(i).first);
    }

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
const CUfunction&
runtime_t::get_kernel(const zeroway_kernel_id::enum_id num)
  {
  return zeroway_kernels.at(num);
  }



template<typename eT>
inline
const CUfunction&
runtime_t::get_kernel(const oneway_kernel_id::enum_id num)
  {
  return get_kernel<eT>(oneway_kernels, num);
  }



template<typename eT>
inline
const CUfunction&
runtime_t::get_kernel(const oneway_real_kernel_id::enum_id num)
  {
  return get_kernel<eT>(oneway_real_kernels, num);
  }



template<typename eT>
inline
const CUfunction&
runtime_t::get_kernel(const oneway_integral_kernel_id::enum_id num)
  {
  return get_kernel<eT>(oneway_integral_kernels, num);
  }



template<typename eT1, typename eT2>
inline
const CUfunction&
runtime_t::get_kernel(const twoway_kernel_id::enum_id num)
  {
  return get_kernel<eT1, eT2>(twoway_kernels, num);
  }



template<typename eT1, typename eT2, typename eT3>
inline
const CUfunction&
runtime_t::get_kernel(const threeway_kernel_id::enum_id num)
  {
  return get_kernel<eT1, eT2, eT3>(threeway_kernels, num);
  }



template<typename eT1, typename... eTs, typename HeldType, typename EnumType>
inline
const CUfunction&
runtime_t::get_kernel(const rt_common::kernels_t<HeldType>& k, const EnumType num)
  {
  coot_extra_debug_sigprint();

       if(is_same_type<eT1,u32   >::yes)  { return get_kernel<eTs...>(k.u32_kernels, num); }
  else if(is_same_type<eT1,s32   >::yes)  { return get_kernel<eTs...>(k.s32_kernels, num); }
  else if(is_same_type<eT1,u64   >::yes)  { return get_kernel<eTs...>(k.u64_kernels, num); }
  else if(is_same_type<eT1,s64   >::yes)  { return get_kernel<eTs...>(k.s64_kernels, num); }
  else if(is_same_type<eT1,float >::yes)  { return get_kernel<eTs...>(  k.f_kernels, num); }
  else if(is_same_type<eT1,double>::yes)  { return get_kernel<eTs...>(  k.d_kernels, num); }
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



template<typename eT, typename EnumType>
inline
const CUfunction&
runtime_t::get_kernel(const rt_common::kernels_t<std::vector<CUfunction>>& k, const EnumType num)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (valid == false), "coot::cuda_rt not valid" );

       if(is_same_type<eT,u32   >::yes)  { return k.u32_kernels.at(num); }
  else if(is_same_type<eT,s32   >::yes)  { return k.s32_kernels.at(num); }
  else if(is_same_type<eT,u64   >::yes)  { return k.u64_kernels.at(num); }
  else if(is_same_type<eT,s64   >::yes)  { return k.s64_kernels.at(num); }
  else if(is_same_type<eT,float >::yes)  { return   k.f_kernels.at(num); }
  else if(is_same_type<eT,double>::yes)  { return   k.d_kernels.at(num); }
  else if(is_same_type<eT,uword >::yes)
    {
    // this can happen if uword != u32 or u64
    if (sizeof(uword) == sizeof(u32))
      {
      return k.u32_kernels.at(num);
      }
    else if (sizeof(uword) == sizeof(u64))
      {
      return k.u64_kernels.at(num);
      }
    else
      {
      // hopefully nobody ever, ever, ever sees this
      throw std::invalid_argument("coot::cuda_rt.get_kernel(): unknown size for uword");
      }
    }
  else if(is_same_type<eT,sword >::yes)
    {
    if (sizeof(sword) == sizeof(s32))
      {
      return k.s32_kernels.at(num);
      }
    else if (sizeof(sword) == sizeof(s64))
      {
      return k.s64_kernels.at(num);
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



template<typename eT>
inline
eT*
runtime_t::acquire_memory(const uword n_elem)
  {
  void* result;
  cudaError_t error = coot_wrapper(cudaMalloc)(&result, sizeof(eT) * n_elem);

  coot_check_cuda_error(error, "coot::cuda_rt.acquire_memory(): couldn't allocate memory");

  return (eT*) result;
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
