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

struct runtime_t
  {
  public:

  inline bool init(const bool manual_selection, const uword wanted_platform, const uword wanted_device, const bool print_info);

  inline ~runtime_t();

  inline std::string unique_host_device_id() const;

  inline bool load_cached_kernels(const std::string& unique_host_device_id, const size_t kernel_size);

  inline bool compile_kernels(const std::string& unique_host_device_id);

  inline bool create_kernels(const std::vector<std::pair<std::string, CUfunction*>>& name_map, char* cubin);

  inline const CUfunction& get_kernel(const zeroway_kernel_id::enum_id num);

  template<typename eT1>
  inline const CUfunction& get_kernel(const oneway_kernel_id::enum_id num);

  template<typename eT1>
  inline const CUfunction& get_kernel(const oneway_real_kernel_id::enum_id num);

  template<typename eT1>
  inline const CUfunction& get_kernel(const oneway_integral_kernel_id::enum_id num);

  template<typename eT2, typename eT1>
  inline const CUfunction& get_kernel(const twoway_kernel_id::enum_id num);

  template<typename eT3, typename eT2, typename eT1>
  inline const CUfunction& get_kernel(const threeway_kernel_id::enum_id num);

  template<typename eT>
  inline eT* acquire_memory(const uword n_elem);

  template<typename eT>
  inline void release_memory(eT* cuda_mem);

  inline void synchronise();

  inline bool is_valid() const { return valid; }

  inline void set_rng_seed(const u64 seed);

  // all types are currently supported by CUDA
  template<typename eT>
  inline constexpr bool is_supported_type() { return true; }

  // use CURAND_ORDERING_PSEUDO_SEEDED with XORWOW / CURAND_ORDERING_PSEUDO_BEST
  // We use XORWOW for uniform distributions, and Philox for normal distributions.
  coot_aligned curandGenerator_t  xorwow_rand;
  coot_aligned curandGenerator_t  philox_rand;

  coot_aligned cudaDeviceProp     dev_prop;

  coot_aligned cublasHandle_t     cublas_handle;
  coot_aligned cusolverDnHandle_t cusolver_handle;

  // TODO: is it necessary to have a lock() and unlock()?
  // since all CUdevice and CUcontext are are pointers, I don't think we need to specifically lock them

  private:

  // internal functions to actually get the right kernel

  template<typename eT1, typename... eTs, typename HeldType, typename EnumType>
  inline
  const CUfunction&
  get_kernel(const rt_common::kernels_t<HeldType>& k, const EnumType num);

  template<typename eT, typename EnumType>
  inline
  const CUfunction&
  get_kernel(const rt_common::kernels_t<std::vector<CUfunction>>& k, const EnumType num);

  coot_aligned bool                     valid;

  coot_aligned std::vector<CUfunction>                                                                   zeroway_kernels;
  coot_aligned rt_common::kernels_t<std::vector<CUfunction>>                                             oneway_kernels;
  coot_aligned rt_common::kernels_t<std::vector<CUfunction>>                                             oneway_real_kernels;
  coot_aligned rt_common::kernels_t<std::vector<CUfunction>>                                             oneway_integral_kernels;
  coot_aligned rt_common::kernels_t<rt_common::kernels_t<std::vector<CUfunction>>>                       twoway_kernels;
  coot_aligned rt_common::kernels_t<rt_common::kernels_t<rt_common::kernels_t<std::vector<CUfunction>>>> threeway_kernels;

  coot_aligned CUdevice cuDevice;
  coot_aligned CUcontext context;
  };
