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



inline
runtime_t::~runtime_t()
  {
  coot_extra_debug_sigprint_this(this);

  internal_cleanup();

  valid = false;
  }



inline
runtime_t::runtime_t()
  {
  coot_extra_debug_sigprint_this(this);

  valid   = false;
  plt_id  = NULL;
  dev_id  = NULL;
  ctxt    = NULL;
  cq      = NULL;
  }



inline
bool
runtime_t::init(const bool manual_selection, const uword wanted_platform, const uword wanted_device, const bool print_info)
  {
  coot_extra_debug_sigprint();

  internal_cleanup();
  valid = false;

  bool status = false;

  status = search_devices(plt_id, dev_id, manual_selection, wanted_platform, wanted_device, print_info);
  if(status == false)  { coot_debug_warn("coot::cl_rt.init(): couldn't find a suitable device"); return false; }

  interrogate_device(dev_info, plt_id, dev_id, print_info);

  if(dev_info.opencl_ver < 120)  { coot_debug_warn("coot::cl_rt.init(): selected device has OpenCL version < 1.2"); return false; }

  status = setup_queue(ctxt, cq, plt_id, dev_id);
  if(status == false)  { coot_debug_warn("coot::cl_rt.init(): couldn't setup queue"); return false; }

  // setup clBLAS
  #if defined(COOT_USE_CLBLAS)
  coot_extra_debug_warn("coot::cl_rt.init(): begin clBLAS setup");
  cl_int clblas_status = coot_wrapper(clblasSetup)();
  coot_extra_debug_warn("coot::cl_rt.init(): finished clBLAS setup");

  if(clblas_status != CL_SUCCESS)  { coot_debug_warn("coot::cl_rt.init(): couldn't setup clBLAS"); return false; }
  #endif

  if(status == false)
    {
    internal_cleanup();
    valid = false;

    return false;
    }

  // Get the name we will use to cache kernels.
  generate_unique_host_device_id();

  // Get the build options we will use to compile each kernel.
  build_options = "";
  if (dev_info.opencl_ver >= 300)
    build_options += "-cl-std=CL3.0 ";
  else if (dev_info.opencl_ver >= 200)
    build_options += "-cl-std=CL2.0 ";
  else if (dev_info.opencl_ver >= 120)
    build_options += "-cl-std=CL1.2 ";
  else if (dev_info.opencl_ver >= 110)
    build_options += "-cl-std=CL1.1 ";
  build_options += ((sizeof(uword) >= 8) && dev_info.has_sizet64) ? std::string("-D UWORD=ulong") : std::string("-D UWORD=uint");

  valid = true;

  // Initialize the preamble we will use when compiling kernels.
  const bool need_subgroup_extension = (dev_info.opencl_ver < 210) && has_subgroups();
  src_preamble = kernel_src::init_src_preamble(has_float64(), has_float16(), has_sizet64(), has_subgroups(), get_subgroup_size(), must_synchronise_subgroups(), need_subgroup_extension);

  // Now set up the XORWOW RNGs for float and double.
  // For type eT, we must store 6 * sizeof(eT) * num_rng_threads for each RNG,
  // where num_rng_threads is the maximum kernel work group size for the randu kernel.
  // This means that we will effectively have one RNG per thread.
  cl_kernel rng_kernel = get_kernel<float>(oneway_kernel_id::inplace_xorwow32_randu);
  status = coot_wrapper(clGetKernelWorkGroupInfo)(rng_kernel, dev_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &num_rng_threads, NULL);
  coot_check_cl_error(status, "coot::cl_rt.init()");
  size_t preferred_work_group_size_multiple;
  status = coot_wrapper(clGetKernelWorkGroupInfo)(rng_kernel, dev_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &preferred_work_group_size_multiple, NULL);
  coot_check_cl_error(status, "coot::cl_rt.init()");
  num_rng_threads *= preferred_work_group_size_multiple;

  xorwow32_state = acquire_memory<u32>(6 * num_rng_threads);
  init_xorwow_state<u32>(xorwow32_state, num_rng_threads);

  if (has_sizet64())
    {
    xorwow64_state = acquire_memory<u64>(6 * num_rng_threads);
    init_xorwow_state<u64>(xorwow64_state, num_rng_threads);
    }

  philox_state = acquire_memory<u32>(6 * num_rng_threads);
  init_philox_state(philox_state, num_rng_threads);
  valid = true;

  return true;
  }



inline
void
runtime_t::lock()
  {
  coot_extra_debug_sigprint();

  coot_extra_debug_print("coot::cl_rt: calling mutex.lock()");
  mutex.lock();
  }




inline
void
runtime_t::unlock()
  {
  coot_extra_debug_sigprint();

  coot_extra_debug_print("coot::cl_rt: calling mutex.unlock()");
  mutex.unlock();
  }



inline
void
runtime_t::internal_cleanup()
  {
  coot_extra_debug_sigprint();

  if(cq != NULL)  { coot_wrapper(clFinish)(cq); }

  #if defined(COOT_USE_CLBLAS)
  coot_wrapper(clblasTeardown)();
  #endif

  // TODO: clean up RNGs

  // TODO: go through each kernel vector

  //const uword f_kernels_size = f_kernels.size();

  //for(uword i=0; i<f_kernels_size; ++i)  { if(f_kernels.at(i) != NULL)  { clReleaseKernel(f_kernels.at(i)); } }

  if(cq   != NULL)  { coot_wrapper(clReleaseCommandQueue)(cq); cq   = NULL; }
  if(ctxt != NULL)  { coot_wrapper(clReleaseContext)(ctxt);    ctxt = NULL; }
  }



inline
bool
runtime_t::search_devices(cl_platform_id& out_plt_id, cl_device_id& out_dev_id, const bool manual_selection, const uword wanted_platform, const uword wanted_device, const bool print_info) const
  {
  coot_extra_debug_sigprint();

  // first, get a list of platforms and the devices on each platform

  cl_int  status      = 0;
  cl_uint n_platforms = 0;

  status = coot_wrapper(clGetPlatformIDs)(0, NULL, &n_platforms);

  if((status != CL_SUCCESS) || (n_platforms == 0))
    {
    coot_debug_warn("coot::cl_rt.init(): no OpenCL platforms available");
    return false;
    }

  std::vector<cl_platform_id> platform_ids(n_platforms);

  status = coot_wrapper(clGetPlatformIDs)(n_platforms, &(platform_ids[0]), NULL);

  if(status != CL_SUCCESS)
    {
    coot_debug_warn("coot::cl_rt.init(): couldn't get info on OpenCL platforms");
    return false;
    }


  // go through each platform

  std::vector< std::vector<cl_device_id> > device_ids(n_platforms);
  std::vector< std::vector<int         > > device_pri(n_platforms);  // device priorities

  for(size_t platform_count = 0; platform_count < n_platforms; ++platform_count)
    {
    cl_platform_id tmp_platform_id = platform_ids.at(platform_count);

    cl_uint local_n_devices = 0;

    status = coot_wrapper(clGetDeviceIDs)(tmp_platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &local_n_devices);

    if((status != CL_SUCCESS) || (local_n_devices == 0))
      {
      continue;  // go to the next platform
      }

    std::vector<cl_device_id>& local_device_ids = device_ids.at(platform_count);
    std::vector<int>&          local_device_pri = device_pri.at(platform_count);

    local_device_ids.resize(local_n_devices);
    local_device_pri.resize(local_n_devices);

    status = coot_wrapper(clGetDeviceIDs)(tmp_platform_id, CL_DEVICE_TYPE_ALL, local_n_devices, &(local_device_ids[0]), NULL);

    // go through each device on this platform
    for(size_t local_device_count = 0; local_device_count < local_n_devices; ++local_device_count)
      {
      cl_device_id local_device_id = local_device_ids.at(local_device_count);

      if(print_info)
        {
        get_cerr_stream().flush();
        get_cerr_stream() << "platform: " << platform_count << " / device: " << local_device_count << std::endl;
        }

      runtime_dev_info tmp_info;

      const bool ok = interrogate_device(tmp_info, tmp_platform_id, local_device_id, print_info);

      if(print_info)
        {
        if(ok == false)
          {
          get_cerr_stream().flush();
          get_cerr_stream() << "problem with getting info about device" << std::endl;
          }

        get_cerr_stream() << std::endl;
        }

      local_device_pri.at(local_device_count) = 0;

      if(tmp_info.is_gpu)           { local_device_pri.at(local_device_count) +=  2; }
      if(tmp_info.has_float16)      { local_device_pri.at(local_device_count) +=  1; }
      if(tmp_info.has_float64)      { local_device_pri.at(local_device_count) +=  1; }
      if(tmp_info.opencl_ver < 120) { local_device_pri.at(local_device_count)  = -1; }
      }
    }


  if(manual_selection)
    {
    if(wanted_platform >= platform_ids.size())
      {
      coot_debug_warn("coot::cl_rt.init(): invalid platform number");
      return false;
      }

    std::vector<cl_device_id>& local_device_ids = device_ids.at(wanted_platform);

    if(wanted_device >= local_device_ids.size())
      {
      coot_debug_warn("coot::cl_rt.init(): invalid device number");
      return false;
      }

    if(print_info)
      {
      get_cerr_stream() << "selected: platform: " << wanted_platform << " / device: " << wanted_device << std::endl;
      }

    out_plt_id = platform_ids.at(wanted_platform);
    out_dev_id = local_device_ids.at(wanted_device);

    return true;
    }


  // select the device with highest priority

  bool found_device = false;

  int    best_val          = -1;
  size_t best_platform_num =  0;
  size_t best_device_num   =  0;

  for(size_t platform_count = 0; platform_count < n_platforms; ++platform_count)
    {
    std::vector<cl_device_id>& local_device_ids = device_ids.at(platform_count);
    std::vector<int>&          local_device_pri = device_pri.at(platform_count);

    size_t local_n_devices = local_device_ids.size();

    for(size_t local_device_count = 0; local_device_count < local_n_devices; ++local_device_count)
      {
      const int tmp_val = local_device_pri.at(local_device_count);
      if(best_val < tmp_val)
        {
        best_val          = tmp_val;
        best_platform_num = platform_count;
        best_device_num   = local_device_count;

        found_device = true;
        }
      }
    }

  if(found_device)
    {
    if(print_info)
      {
      get_cerr_stream() << "selected: platform: " << best_platform_num << " / device: " << best_device_num << std::endl;
      }

    std::vector<cl_device_id>& local_device_ids = device_ids.at(best_platform_num);

    out_plt_id = platform_ids.at(best_platform_num);
    out_dev_id = local_device_ids.at(best_device_num);
    }

  return found_device;
  }



inline
bool
runtime_t::interrogate_device(runtime_dev_info& out_info, cl_platform_id in_plt_id, cl_device_id in_dev_id, const bool print_info) const
  {
  coot_extra_debug_sigprint();

  cl_char dev_name1[1024]; // TODO: use dynamic memory allocation (podarray or std::vector)
  cl_char dev_name2[1024];
  cl_char dev_name3[1024];

  dev_name1[0] = cl_char(0);
  dev_name2[0] = cl_char(0);
  dev_name3[0] = cl_char(0);

  cl_device_type      dev_type = 0;
  cl_device_fp_config dev_fp64 = 0;
  cl_device_fp_config dev_fp16 = 0;

  cl_uint dev_n_units     = 0;
  cl_uint dev_sizet_width = 0;
  cl_uint dev_ptr_width   = 0;
  cl_uint dev_opencl_ver  = 0;
  cl_uint dev_align       = 0;

  size_t dev_max_wg         = 0;
  size_t dev_subgroup_size  = 0;
  size_t dev_max_wg_ndims   = 0;
  size_t* dev_max_wg_dims   = nullptr;

  bool has_subgroup_extension = false;
  bool has_intel_subgroup_extension = false;
  bool has_nv_device_attribute_query_extension = false;
  bool dev_has_subgroups = false;
  bool dev_must_synchronise_subgroups = true;

  coot_wrapper(clGetDeviceInfo)(in_dev_id, CL_DEVICE_VENDOR,                   sizeof(dev_name1),           &dev_name1,        NULL);
  coot_wrapper(clGetDeviceInfo)(in_dev_id, CL_DEVICE_NAME,                     sizeof(dev_name2),           &dev_name2,        NULL);
  coot_wrapper(clGetDeviceInfo)(in_dev_id, CL_DEVICE_VERSION,                  sizeof(dev_name3),           &dev_name3,        NULL);
  coot_wrapper(clGetDeviceInfo)(in_dev_id, CL_DEVICE_TYPE,                     sizeof(cl_device_type),      &dev_type,         NULL);
  #if defined(CL_DEVICE_DOUBLE_FP_CONFIG)
  coot_wrapper(clGetDeviceInfo)(in_dev_id, CL_DEVICE_DOUBLE_FP_CONFIG,         sizeof(cl_device_fp_config), &dev_fp64,         NULL);
  #endif
  #if defined(CL_DEVICE_HALF_FP_CONFIG)
  coot_wrapper(clGetDeviceInfo)(in_dev_id, CL_DEVICE_HALF_FP_CONFIG,           sizeof(cl_device_fp_config), &dev_fp16,         NULL);
  #endif
  coot_wrapper(clGetDeviceInfo)(in_dev_id, CL_DEVICE_MAX_COMPUTE_UNITS,        sizeof(cl_uint),             &dev_n_units,      NULL);
  coot_wrapper(clGetDeviceInfo)(in_dev_id, CL_DEVICE_MEM_BASE_ADDR_ALIGN,      sizeof(cl_uint),             &dev_align,        NULL);
  coot_wrapper(clGetDeviceInfo)(in_dev_id, CL_DEVICE_MAX_WORK_GROUP_SIZE,      sizeof(size_t),              &dev_max_wg,       NULL);
  coot_wrapper(clGetDeviceInfo)(in_dev_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(size_t),              &dev_max_wg_ndims, NULL);
  if (dev_max_wg_ndims != 0)
    {
    dev_max_wg_dims = new size_t[dev_max_wg_ndims];
    coot_wrapper(clGetDeviceInfo)(in_dev_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, dev_max_wg_ndims * sizeof(size_t), dev_max_wg_dims, NULL);
    }

  // search for extensions we care about
  size_t dev_extension_size;
  coot_wrapper(clGetDeviceInfo)(in_dev_id, CL_DEVICE_EXTENSIONS, 0, NULL, &dev_extension_size);
  char* dev_extension_buffer = new char[dev_extension_size];
  coot_wrapper(clGetDeviceInfo)(in_dev_id, CL_DEVICE_EXTENSIONS, dev_extension_size, dev_extension_buffer, NULL);
  std::string dev_extension_str(dev_extension_buffer);
  delete[] dev_extension_buffer;
  size_t last_space = 0;
  size_t next_space = dev_extension_str.find(' ');
  do
    {
    const size_t actual_last_space = (last_space == 0) ? 0 : (last_space + 1);
    std::string ext = dev_extension_str.substr(actual_last_space, (next_space - actual_last_space));

    // This extension, if present, allows us to get the subgroup size when on
    // OpenCL < 2.0 devices.
    if (ext == "cl_khr_subgroups")
      {
      has_subgroup_extension = true;
      }
    else if (ext == "cl_intel_subgroups")
      {
      has_intel_subgroup_extension = true;
      }
    else if (ext == "cl_nv_device_attribute_query")
      {
      has_nv_device_attribute_query_extension = true;
      }

    last_space = next_space;
    }
  while ((next_space = dev_extension_str.find(' ', next_space + 1)) != std::string::npos);

  // contrary to the official OpenCL specification (OpenCL 1.2, sec 4.2 and sec 6.1.1).
  // certain OpenCL implementations use internal size_t which doesn't correspond to CL_DEVICE_ADDRESS_BITS
  // example: Clover from Mesa 13.0.4, running as AMD OLAND (DRM 2.48.0 / 4.9.14-200.fc25.x86_64, LLVM 3.9.1)


  const char* tmp_program_src = \
    "__kernel void coot_interrogate(__global uint* out) \n"
    "  {                                                \n"
    "  const size_t i = get_global_id(0);               \n"
    "  if(i == 0)                                       \n"
    "    {                                              \n"
    "    out[0] = (uint)sizeof(size_t);                 \n"
    "    out[1] = (uint)sizeof(void*);                  \n"
    "    out[2] = (uint)(__OPENCL_VERSION__);           \n"
    "    }                                              \n"
    "  }                                                \n";

  cl_context       tmp_context    = NULL;
  cl_command_queue tmp_queue      = NULL;
  cl_program       tmp_program    = NULL;
  cl_kernel        tmp_kernel     = NULL;
  cl_mem           tmp_dev_mem    = NULL;
  cl_uint          tmp_cpu_mem[4] = { 0, 0, 0, 0 };


  cl_int status = 0;

  if(setup_queue(tmp_context, tmp_queue, in_plt_id, in_dev_id))
    {
    tmp_program = coot_wrapper(clCreateProgramWithSource)(tmp_context, 1, (const char **)&(tmp_program_src), NULL, &status);

    if(status == CL_SUCCESS)
      {
      status = coot_wrapper(clBuildProgram)(tmp_program, 0, NULL, NULL, NULL, NULL);

      // COOT_COUT_STREAM << "status: " << coot_cl_error::as_string(status) << std::endl;

      // size_t len = 0;
      // char buffer[10240];

      // clGetProgramBuildInfo(tmp_program, in_dev_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
      // COOT_COUT_STREAM << "output from clGetProgramBuildInfo():" << std::endl;
      // COOT_COUT_STREAM << buffer << std::endl;

      if(status == CL_SUCCESS)
        {
        tmp_kernel = coot_wrapper(clCreateKernel)(tmp_program, "coot_interrogate", &status);

        if(status == CL_SUCCESS)
          {
          tmp_dev_mem = coot_wrapper(clCreateBuffer)(tmp_context, CL_MEM_READ_WRITE, sizeof(cl_uint)*4, NULL, &status);

          coot_wrapper(clSetKernelArg)(tmp_kernel, 0, sizeof(cl_mem),  &tmp_dev_mem);
          status = coot_wrapper(clEnqueueTask)(tmp_queue, tmp_kernel, 0, NULL, NULL);  // TODO: replace with clEnqueueNDRangeKernel to avoid deprecation warnings

          if(status == CL_SUCCESS)
            {
            coot_wrapper(clFinish)(cq);

            status = coot_wrapper(clEnqueueReadBuffer)(tmp_queue, tmp_dev_mem, CL_TRUE, 0, sizeof(cl_uint)*4, tmp_cpu_mem, 0, NULL, NULL);

            if(status == CL_SUCCESS)
              {
              coot_wrapper(clFinish)(cq);

              dev_sizet_width = tmp_cpu_mem[0];
              dev_ptr_width   = tmp_cpu_mem[1];
              dev_opencl_ver  = tmp_cpu_mem[2];

              // Extract the subgroup size, if available.  Before OpenCL 2.0,
              // subgroups were an extension.
              if (tmp_cpu_mem[2] /* opencl_ver */ >= 210)
                {
                dev_has_subgroups = true;
                }
              else
                {
                dev_has_subgroups = has_subgroup_extension || has_intel_subgroup_extension;
                }

              if (dev_has_subgroups)
                {
                // It seems possible this could be different per kernel, but we'll hope not.
                // We'll choose an input size that is larger than any reasonable subgroup size.
                status = coot_sub_group_size(tmp_kernel, in_dev_id, 32768, dev_subgroup_size);

                // It's not pointed out in the standards, but sometimes CL_INVALID_OPERATION will be returned if for some reason subgroups are not supported despite it being part of the standard.
                // (I am looking at you, nvidia OpenCL driver.)
                if (status == CL_INVALID_OPERATION || dev_subgroup_size == 0)
                  {
                  dev_has_subgroups = false;
                  dev_subgroup_size = 0;
                  status = CL_SUCCESS;

                  // Now, on nvidia devices, there is still the concept of "warp", and the nvidia OpenCL Programming Guide,
                  // in section 3.4.3, points out that warp-synchronous programming without barriers is okay so long as the
                  // memory is marked volatile.
                  //
                  // So, if we are on an nvidia card, we will enable subgroups with subgroup size equal to the warp size.
                  if (has_nv_device_attribute_query_extension == true)
                    {
                    status = coot_nv_warp_size(in_dev_id, dev_subgroup_size);
                    if (status == CL_SUCCESS && dev_subgroup_size > 0)
                      {
                      dev_has_subgroups = true;
                      dev_must_synchronise_subgroups = false;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

  if(status != CL_SUCCESS)
    {
    coot_debug_warn(coot_cl_error::as_string(status));
    }

  if(tmp_dev_mem != NULL)  { coot_wrapper(clReleaseMemObject   )(tmp_dev_mem); }
  if(tmp_kernel  != NULL)  { coot_wrapper(clReleaseKernel      )(tmp_kernel ); }
  if(tmp_program != NULL)  { coot_wrapper(clReleaseProgram     )(tmp_program); }
  if(tmp_queue   != NULL)  { coot_wrapper(clReleaseCommandQueue)(tmp_queue);   }
  if(tmp_context != NULL)  { coot_wrapper(clReleaseContext     )(tmp_context); }

  if(print_info)
    {
    get_cerr_stream().flush();
    get_cerr_stream() << "name1:                      " << dev_name1 << std::endl;
    get_cerr_stream() << "name2:                      " << dev_name2 << std::endl;
    get_cerr_stream() << "name3:                      " << dev_name3 << std::endl;
    get_cerr_stream() << "is_gpu:                     " << (dev_type == CL_DEVICE_TYPE_GPU)  << std::endl;
    get_cerr_stream() << "fp64:                       " << dev_fp64 << std::endl;
    get_cerr_stream() << "fp16:                       " << dev_fp16 << std::endl;
    get_cerr_stream() << "sizet_width:                " << dev_sizet_width  << std::endl;
    get_cerr_stream() << "ptr_width:                  " << dev_ptr_width << std::endl;
    get_cerr_stream() << "n_units:                    " << dev_n_units << std::endl;
    get_cerr_stream() << "opencl_ver:                 " << dev_opencl_ver << std::endl;
  //get_cerr_stream() << "align:                      " << dev_align  << std::endl;
    get_cerr_stream() << "max_wg:                     " << dev_max_wg << std::endl;
    get_cerr_stream() << "max_wg_ndims:               " << dev_max_wg_ndims << std::endl;
    get_cerr_stream() << "max_wg_dims:                (";
    for (uword i = 0; i + 1 < dev_max_wg_ndims; ++i)
      get_cerr_stream() << dev_max_wg_dims[i] << ", ";
    get_cerr_stream() << dev_max_wg_dims[dev_max_wg_ndims - 1] << ")" << std::endl;
    get_cerr_stream() << "subgroup_size:              " << dev_subgroup_size << std::endl;
    get_cerr_stream() << "must_synchronise_subgroups: " << dev_must_synchronise_subgroups << std::endl;
    }

  out_info.is_gpu                     = (dev_type == CL_DEVICE_TYPE_GPU);
  out_info.has_float16                = (dev_fp16 != 0);
  out_info.has_float64                = (dev_fp64 != 0);
  out_info.has_sizet64                = (dev_sizet_width >= 8);
  out_info.has_subgroups              = dev_has_subgroups;
  out_info.must_synchronise_subgroups = dev_must_synchronise_subgroups;
  out_info.ptr_width                  = uword(dev_ptr_width);
  out_info.n_units                    = uword(dev_n_units);
  out_info.opencl_ver                 = uword(dev_opencl_ver);
  out_info.max_wg                     = uword(dev_max_wg);
  out_info.subgroup_size              = uword(dev_subgroup_size);
  out_info.max_wg_ndims               = uword(dev_max_wg_ndims);
  out_info.max_wg_dims                = new uword[dev_max_wg_ndims];
  for (size_t i = 0; i < dev_max_wg_ndims; ++i)
    out_info.max_wg_dims[i] = (uword) dev_max_wg_dims[i];

  delete[] dev_max_wg_dims;

  return (status == CL_SUCCESS);
  }




inline
void
runtime_t::generate_unique_host_device_id()
  {
  // Generate a string that corresponds to this specific device and OpenCL version.

  std::ostringstream oss;

  // Use the reported name and vendor ID of the device.  (This preserves a little bit of human readability.)
  char buffer[1025]; // hopefully way larger than necessary
  memset(buffer, 0, 1025);
  cl_int status = coot_wrapper(clGetDeviceInfo)(dev_id, CL_DEVICE_NAME, 1024, buffer, NULL);
  if (status != CL_SUCCESS)
    {
    coot_stop_runtime_error("coot::cl_rt.init(): unable to get device name: " + coot_cl_error::as_string(status));
    }

  // Remove non-alphanumeric characters from the device name.
  char buffer2[1024];
  size_t buf_len = strnlen(buffer, 1024);
  for (size_t i = 0; i < buf_len; ++i)
    {
    const char c = buffer[i];
    buffer2[i] = std::isalnum(c) ? std::tolower(c) : '_';
    }
  buffer2[buf_len] = buffer[buf_len];
  oss << buffer2 << "_";

  cl_uint vendor_id;
  status = coot_wrapper(clGetDeviceInfo)(dev_id, CL_DEVICE_VENDOR_ID, sizeof(cl_uint), &vendor_id, NULL);
  if (status != CL_SUCCESS)
    {
    coot_stop_runtime_error("coot::cl_rt.init(): unable to get device vendor ID: " + coot_cl_error::as_string(status));
    }
  oss << vendor_id << "_";

  // Get the OpenCL driver version that the device supports.
  status = coot_wrapper(clGetDeviceInfo)(dev_id, CL_DEVICE_VERSION, 1024, buffer, NULL);
  if (status != CL_SUCCESS)
    {
    coot_stop_runtime_error("coot::cl_rt.init(): unable to get OpenCL version of device: " + coot_cl_error::as_string(status));
    }
  buf_len = strnlen(buffer, 1024);
  for (size_t i = 0; i < buf_len; ++i)
    {
    const char c = buffer[i];
    buffer2[i] = std::isalnum(c) ? std::tolower(c) : '_';
    }
  buffer2[buf_len] = buffer[buf_len];

  oss << buffer2;
  unique_host_device_id = oss.str();
  }



inline
bool
runtime_t::setup_queue(cl_context& out_context, cl_command_queue& out_queue, cl_platform_id in_plat_id, cl_device_id in_dev_id) const
  {
  coot_extra_debug_sigprint();

  cl_context_properties prop[3] = { CL_CONTEXT_PLATFORM, cl_context_properties(in_plat_id), 0 };

  cl_int status = 0;

  out_context = coot_wrapper(clCreateContext)(prop, 1, &in_dev_id, NULL, NULL, &status);

  if((status != CL_SUCCESS) || (out_context == NULL))
    {
    coot_debug_warn(coot_cl_error::as_string(status));
    return false;
    }

  // NOTE: clCreateCommandQueue is deprecated as of OpenCL 2.0, but it will be supported for the "foreseeable future"
  // NOTE: clCreateCommandQueue is replaced with clCreateCommandQueueWithProperties in OpenCL 2.0
  // NOTE: http://stackoverflow.com/questions/28500496/opencl-function-found-deprecated-by-visual-studio

  out_queue = coot_wrapper(clCreateCommandQueue)(out_context, in_dev_id, 0, &status);

  if((status != CL_SUCCESS) || (out_queue == NULL))
    {
    coot_debug_warn(coot_cl_error::as_string(status));
    return false;
    }

  return true;
  }



inline
bool
runtime_t::load_cached_kernel(const std::string& kernel_name, cl_kernel& kernel)
  {
  coot_extra_debug_sigprint();

  const size_t kernel_size = cache::has_cached_kernel(unique_host_device_id, kernel_name);
  if (kernel_size == 0)
    {
    // We don't have the kernel.
    return false;
    }

  // Allocate a buffer large enough to store the program.
  unsigned char* kernel_buffer = new unsigned char[kernel_size];
  bool status = cache::read_cached_kernel(unique_host_device_id, kernel_name, kernel_buffer);
  if (status == false)
    {
    coot_debug_warn("coot::opencl::load_cached_kernel(): could not load kernel '" + kernel_name + "' for unique host device id '" + unique_host_device_id + "'");
    delete[] kernel_buffer;
    return false;
    }

  cl_int binary_status, errcode_ret;
  runtime_t::program_wrapper prog_holder;  // program_wrapper will automatically call clReleaseProgram() when it goes out of scope
  prog_holder.prog = coot_wrapper(clCreateProgramWithBinary)(ctxt, 1, &dev_id, &kernel_size, (const unsigned char**) &kernel_buffer, &binary_status, &errcode_ret);
  if (errcode_ret != CL_SUCCESS)
    {
    coot_debug_warn(coot_cl_error::as_string(errcode_ret));
    delete[] kernel_buffer;
    return false;
    }
  else if (binary_status != CL_SUCCESS)
    {
    coot_debug_warn(coot_cl_error::as_string(binary_status));
    delete[] kernel_buffer;
    return false;
    }

  delete[] kernel_buffer;

  // We actually have to build the program, even when the kernel is cached.  Ideally, depending on the driver, this doesn't do much.
  binary_status = coot_wrapper(clBuildProgram)(prog_holder.prog, 0, NULL, build_options.c_str(), NULL, NULL);
  coot_check_cl_error(binary_status, "coot::opencl::load_cached_kernel(): clBuildProgram() for kernel '" + kernel_name + "' failed");

  // If we got to here, we succeeded at creating the program.
  // So, load the compiled kernel, after initializing the name map.
  kernel = coot_wrapper(clCreateKernel)(prog_holder.prog, kernel_name.c_str(), &errcode_ret);
  if(kernel == NULL)
    {
    coot_debug_warn("coot::opencl::load_cached_kernel(): couldn't create kernel '" + kernel_name + "': " + coot_cl_error::as_string(errcode_ret));
    return false;
    }

  return true;
  }



inline
void
runtime_t::compile_kernel(const std::string& kernel_name,
                          const std::string& source,
                          cl_kernel& kernel)
  {
  coot_extra_debug_sigprint();

  // Gather the sources we need to compile.
  runtime_t::program_wrapper prog_holder;  // program_wrapper will automatically call clReleaseProgram() when it goes out of scope


  // cl_program clCreateProgramWithSource(cl_context context, cl_uint count, const char **strings, const size_t *lengths, cl_int *errcode_ret);
  // strings = An array of N pointers (where N = count) to optionally null-terminated character strings that make up the source code.
  // lengths = An array with the number of chars in each string (the string length). If an element in lengths is zero, its accompanying string is null-terminated.
  //           If lengths is NULL, all strings in the strings argument are considered null-terminated.
  //           Any length value passed in that is greater than zero excludes the null terminator in its count.

  cl_int status = 0;

  const char* source_c_str = source.c_str();

  prog_holder.prog = coot_wrapper(clCreateProgramWithSource)(ctxt, 1, &source_c_str, NULL, &status);
  coot_check_cl_error(status, "coot::opencl::compile_kernel(): couldn't create program for kernel '" + kernel_name + "'");
  if(prog_holder.prog == NULL)
    {
    coot_stop_runtime_error("coot::opencl::compile_kernel(): returned program from clCreateProgramWithSource() for kernel '" + kernel_name + "' was NULL");
    }

  // Now actually have to build the program first.
  status = coot_wrapper(clBuildProgram)(prog_holder.prog, 0, NULL, build_options.c_str(), NULL, NULL);

  if(status != CL_SUCCESS)
    {
    size_t len = 0;

    // Get the length of the error log and then allocate enough space for it.
    coot_wrapper(clGetProgramBuildInfo)(prog_holder.prog, dev_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
    char* buffer = new char[len];

    coot_wrapper(clGetProgramBuildInfo)(prog_holder.prog, dev_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
    coot_stop_runtime_error("coot::cl_rt.compile_kernel(): couldn't compile kernel '" + kernel_name + "'; output from clGetProgramBuildInfo():", buffer);
    }

  kernel = coot_wrapper(clCreateKernel)(prog_holder.prog, kernel_name.c_str(), &status);
  coot_check_cl_error(status, "coot::opencl::compile_kernel(): couldn't create kernel '" + kernel_name + "'; source:\n\n" + source + "\n\n");

  // Now that we have built the kernel, cache it.
  size_t binary_size;
  status = coot_wrapper(clGetProgramInfo)(prog_holder.prog, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binary_size, NULL);
  if (status != CL_SUCCESS)
    {
    coot_debug_warn("coot::opencl::compile_kernel(): clGetProgramInfo() call to get binary size of kernel '" + kernel_name + "'failed: " + coot_cl_error::as_string(status));
    return; // Don't keep trying to cache... but we did build the kernel.
    }
  else if (binary_size == 0)
    {
    coot_debug_warn("coot::opencl::compile_kernel(): reported binary size for kernel '" + kernel_name + "' is 0; not caching");
    return;
    }

  // Now allocate something to hold the program.
  unsigned char* buffer = new unsigned char[binary_size];
  status = coot_wrapper(clGetProgramInfo)(prog_holder.prog, CL_PROGRAM_BINARIES, sizeof(size_t), &buffer, NULL);
  if (status != CL_SUCCESS)
    {
    coot_debug_warn("coot::opencl::compile_kernel(): clGetProgramInfo() call to get binaries for kernel '" + kernel_name + "' failed with " + coot_cl_error::as_string(status));
    return;
    }

  bool success = cache::cache_kernel(unique_host_device_id, kernel_name, buffer, binary_size);
  if (success == false)
    {
    coot_debug_warn("coot::opencl::compile_kernel(): could not cache compiled OpenCL kernel " + kernel_name);
    }

  delete[] buffer;
  }



inline
uword
runtime_t::get_n_units() const
  {
  return (valid) ? dev_info.n_units : uword(0);
  }



inline
uword
runtime_t::get_max_wg() const
  {
  return (valid) ? dev_info.max_wg : uword(0);
  }



inline
uword
runtime_t::get_subgroup_size() const
  {
  return dev_info.subgroup_size;
  }



inline
uword
runtime_t::get_max_wg_ndims() const
  {
  return dev_info.max_wg_ndims;
  }



inline
uword
runtime_t::get_max_wg_dim(const uword i) const
  {
  return dev_info.max_wg_dims[i];
  }



inline
bool
runtime_t::is_valid() const
  {
  return valid;
  }



inline
bool
runtime_t::has_sizet64() const
  {
  return dev_info.has_sizet64;
  }



inline
bool
runtime_t::has_float16() const
  {
  return dev_info.has_float16;
  }



inline
bool
runtime_t::has_float64() const
  {
  return dev_info.has_float64;
  }



inline
bool
runtime_t::has_subgroups() const
  {
  return dev_info.has_subgroups;
  }



inline
bool
runtime_t::must_synchronise_subgroups() const
  {
  return dev_info.must_synchronise_subgroups;
  }



template<typename eT>
inline
coot_cl_mem
runtime_t::acquire_memory(const uword n_elem)
  {
  coot_extra_debug_sigprint();

  coot_check_runtime_error( (valid == false), "coot::cl_rt.acquire_memory(): runtime not valid" );

  if(n_elem == 0)  { return coot_cl_mem{ NULL, 0 }; }

  coot_debug_check
   (
   ( size_t(n_elem) > (Datum<size_t>::max / sizeof(eT)) ),
   "coot::cl_rt.acquire_memory(): requested size is too large"
   );

  typedef typename cl_type<eT>::type ceT;

  cl_int status = 0;
  cl_mem result = coot_wrapper(clCreateBuffer)(ctxt, CL_MEM_READ_WRITE, sizeof(ceT)*(std::max)(uword(1), n_elem), NULL, &status);

  coot_check_bad_alloc( ((status != CL_SUCCESS) || (result == NULL)), "coot::cl_rt.acquire_memory(): not enough memory on device" );

  return coot_cl_mem{ result, 0 };
  }



inline
void
runtime_t::release_memory(coot_cl_mem dev_mem)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (valid == false), "coot::cl_rt not valid" );

  if(dev_mem.ptr)  { coot_wrapper(clReleaseMemObject)(dev_mem.ptr); }
  }



inline
void
runtime_t::synchronise()
  {
  coot_wrapper(clFinish)(get_cq());
  }



inline
cl_device_id
runtime_t::get_device()
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (valid == false), "coot::cl_rt not valid" );

  return dev_id;
  }



inline
cl_context
runtime_t::get_context()
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (valid == false), "coot::cl_rt not valid" );

  return ctxt;
  }



inline
cl_command_queue
runtime_t::get_cq()
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (valid == false), "coot::cl_rt not valid" );

  return cq;
  }



inline
bool
runtime_t::create_extra_cq(cl_command_queue& out_queue)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (valid == false), "coot::cl_rt not valid" );

  cl_int status = 0;

  out_queue = coot_wrapper(clCreateCommandQueue)((*this).ctxt, (*this).dev_id, 0, &status);

  if((status != CL_SUCCESS) || (out_queue == NULL))
    {
    coot_debug_warn(coot_cl_error::as_string(status));
    return false;
    }

  return true;
  }



inline
void
runtime_t::delete_extra_cq(cl_command_queue& in_queue)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (valid == false), "coot::cl_rt not valid" );

  if(in_queue != NULL)
    {
    coot_wrapper(clFinish)(in_queue); // force all queued operations to finish
    coot_wrapper(clReleaseCommandQueue)(in_queue);
    in_queue = NULL;
    }
  }



inline
std::string
runtime_t::generate_kernel(const zeroway_kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  std::string source =
      src_preamble +
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



template<typename eT>
inline
std::string
runtime_t::generate_kernel(const magma_real_kernel_id::enum_id num)
  {
  coot_extra_debug_sigprint();

  std::string source =
      src_preamble +
      kernel_src::get_magma_defines() +
      kernel_src::get_oneway_defines<eT>() +
      kernel_src::get_magma_real_source(num);

  return source;
  }



inline
const cl_kernel&
runtime_t::get_kernel(const zeroway_kernel_id::enum_id num)
  {
  const std::tuple<bool, cl_kernel&> t = get_kernel(zeroway_kernels, num);
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
const cl_kernel&
runtime_t::get_kernel(const oneway_kernel_id::enum_id num)
  {
  const std::tuple<bool, cl_kernel&> t = get_kernel<eT>(oneway_kernels, num);
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
const cl_kernel&
runtime_t::get_kernel(const oneway_real_kernel_id::enum_id num)
  {
  const std::tuple<bool, cl_kernel&> t = get_kernel<eT>(oneway_real_kernels, num);
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
const cl_kernel&
runtime_t::get_kernel(const oneway_integral_kernel_id::enum_id num)
  {
  const std::tuple<bool, cl_kernel&> t = get_kernel<eT>(oneway_integral_kernels, num);
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
const cl_kernel&
runtime_t::get_kernel(const twoway_kernel_id::enum_id num)
  {
  const std::tuple<bool, cl_kernel&> t = get_kernel<eT1, eT2>(twoway_kernels, num);
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
const cl_kernel&
runtime_t::get_kernel(const threeway_kernel_id::enum_id num)
  {
  const std::tuple<bool, cl_kernel&> t = get_kernel<eT1, eT2, eT3>(threeway_kernels, num);
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



template<typename eT>
inline
const cl_kernel&
runtime_t::get_kernel(const magma_real_kernel_id::enum_id num)
  {
  const std::tuple<bool, cl_kernel&> t = get_kernel<eT>(magma_real_kernels, num);
  if (std::get<0>(t) == true)
    {
    const std::string name = type_prefix<eT>() + "_" + magma_real_kernel_id::get_names()[num];

    if (!load_cached_kernel(name, std::get<1>(t)))
      {
      // We will have to compile the kernel on the spot.
      const std::string source = generate_kernel<eT>(num);
      compile_kernel(name, source, std::get<1>(t));
      }
    }

  return std::get<1>(t);
  }



template<typename eT1, typename... eTs, typename HeldType, typename EnumType>
inline
std::tuple<bool, cl_kernel&>
runtime_t::get_kernel(rt_common::kernels_t<HeldType>& k, const EnumType num)
  {
  coot_extra_debug_sigprint();

  if(is_same_type<eT1, u8>::yes)         { return get_kernel<eTs...>(k.u8_kernels, num);  }
  else if(is_same_type<eT1,  s8>::yes)   { return get_kernel<eTs...>(k.s8_kernels, num);  }
  else if(is_same_type<eT1, u16>::yes)   { return get_kernel<eTs...>(k.u16_kernels, num); }
  else if(is_same_type<eT1, s16>::yes)   { return get_kernel<eTs...>(k.s16_kernels, num); }
  else if(is_same_type<eT1, u32>::yes)   { return get_kernel<eTs...>(k.u32_kernels, num); }
  else if(is_same_type<eT1, s32>::yes)   { return get_kernel<eTs...>(k.s32_kernels, num); }
  else if(is_same_type<eT1, u64>::yes)   { return get_kernel<eTs...>(k.u64_kernels, num); }
  else if(is_same_type<eT1, s64>::yes)   { return get_kernel<eTs...>(k.s64_kernels, num); }
  else if(is_same_type<eT1, fp16>::yes)
    {
    coot_debug_check( has_float16() == false, "coot::cl_rt.get_kernel(): device does not support float16 (fp16) kernels; use a different element type (such as float)" );

    return get_kernel<eTs...>(k.h_kernels, num);
    }
  else if(is_same_type<eT1, float>::yes) { return get_kernel<eTs...>(k.f_kernels, num);   }
  else if(is_same_type<eT1, double>::yes)
    {
    coot_debug_check( has_float64() == false, "coot::cl_rt.get_kernel(): device does not support float64 (double) kernels; use a different element type (such as float)" );

    return get_kernel<eTs...>(k.d_kernels, num);
    }
  else if(is_same_type<eT1, uword>::yes)
    {
    // only encountered if uword != u32 or u64; but we need to figure out how large a uword is
    // (this can happen if, e.g., u32 == unsigned int and u64 == unsigned long long but uword == unsigned long)
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
      throw std::invalid_argument("coot::cl_rt.get_kernel(): unknown size for uword");
      }
    }
  else if(is_same_type<eT1, sword>::yes)
    {
    // only encountered if sword != s32 or s64
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
      throw std::invalid_argument("coot::cl_rt.get_kernel(): unknown size for sword");
      }
    }

  throw std::invalid_argument("coot::cl_rt.get_kernel(): unsupported element type");
  }



template<typename EnumType>
inline
std::tuple<bool, cl_kernel&>
runtime_t::get_kernel(std::unordered_map<EnumType, cl_kernel>& k, const EnumType num)
  {
  coot_extra_debug_sigprint();

  if(k.count(num) == 0)
    {
    k[num] = cl_kernel();
    return std::forward_as_tuple(true, k[num]);
    }
  else
    {
    return std::forward_as_tuple(false, k[num]);
    }
  }



template<typename eT>
inline
coot_cl_mem
runtime_t::get_xorwow_state() const
  {
  // It's possible that uword and sword may be different types altogether than
  // u32/s32/u64/s64.  In this case, we need to find the width of them and call
  // the appropriate overload.
  if (is_same_type<eT, uword>::yes)
    {
    if (sizeof(uword) == sizeof(u32))
      {
      return get_xorwow_state<u32>();
      }
    else if (sizeof(uword) == sizeof(u64))
      {
      return get_xorwow_state<u64>();
      }
    else
      {
      // hopefully nobody ever, ever, ever sees this error
      coot_stop_runtime_error("coot::cl_rt.get_xorwow_state(): unknown size for uword");
      }
    }
  else if (is_same_type<eT, sword>::yes)
    {
    if (sizeof(sword) == sizeof(s32))
      {
      return get_xorwow_state<s32>();
      }
    else if (sizeof(sword) == sizeof(s64))
      {
      return get_xorwow_state<s64>();
      }
    else
      {
      // hopefully nobody ever, ever, ever sees this error
      coot_stop_runtime_error("coot::cl_rt.get_xorwow_state(): unknown size for sword");
      }
    }

  std::ostringstream oss;
  oss << "coot::cl_rt.get_xorwow_state(): no RNG available for type " << typeid(eT).name();
  coot_stop_runtime_error(oss.str());
  return coot_cl_mem{ NULL, 0 };
  }



template<> inline coot_cl_mem runtime_t::get_xorwow_state<u8    >() const { return xorwow32_state; }
template<> inline coot_cl_mem runtime_t::get_xorwow_state<s8    >() const { return xorwow32_state; }
template<> inline coot_cl_mem runtime_t::get_xorwow_state<u16   >() const { return xorwow32_state; }
template<> inline coot_cl_mem runtime_t::get_xorwow_state<s16   >() const { return xorwow32_state; }
template<> inline coot_cl_mem runtime_t::get_xorwow_state<fp16  >() const { return xorwow32_state; }
template<> inline coot_cl_mem runtime_t::get_xorwow_state<float >() const { return xorwow32_state; }
template<> inline coot_cl_mem runtime_t::get_xorwow_state<u32   >() const { return xorwow32_state; }
template<> inline coot_cl_mem runtime_t::get_xorwow_state<s32   >() const { return xorwow32_state; }
template<> inline coot_cl_mem runtime_t::get_xorwow_state<double>() const { return xorwow64_state; }
template<> inline coot_cl_mem runtime_t::get_xorwow_state<u64   >() const { return xorwow64_state; }
template<> inline coot_cl_mem runtime_t::get_xorwow_state<s64   >() const { return xorwow64_state; }



inline
coot_cl_mem
runtime_t::get_philox_state() const
  {
  return philox_state;
  }



inline
size_t
runtime_t::get_num_rng_threads() const
  {
  return num_rng_threads;
  }



inline
void
runtime_t::set_rng_seed(const u64 seed)
  {
  coot_extra_debug_sigprint();

  // reset RNG memory with correct seed
  init_xorwow_state<u32>(xorwow32_state, num_rng_threads, seed);
  init_xorwow_state<u64>(xorwow64_state, num_rng_threads, seed);
  init_philox_state(philox_state, num_rng_threads, seed);
  }



//
// program_wrapper

inline
runtime_t::program_wrapper::program_wrapper()
  {
  coot_extra_debug_sigprint();

  prog = NULL;
  }



inline
runtime_t::program_wrapper::~program_wrapper()
  {
  coot_extra_debug_sigprint();

  if(prog != NULL)  { coot_wrapper(clReleaseProgram)(prog); }
  }






//
// cq_guard

inline
runtime_t::cq_guard::cq_guard()
  {
  coot_extra_debug_sigprint();

  get_rt().cl_rt.lock();

  if(get_rt().cl_rt.is_valid())
    {
    coot_extra_debug_print("coot::cl_rt: calling clFinish()");
    coot_wrapper(clFinish)(get_rt().cl_rt.get_cq());  // force synchronisation

    //coot_extra_debug_print("calling clFlush()");
    //clFlush(get_rt().cl_rt.get_cq());  // submit all enqueued commands
    }
  }



inline
runtime_t::cq_guard::~cq_guard()
  {
  coot_extra_debug_sigprint();

  if(get_rt().cl_rt.is_valid())
    {
    coot_extra_debug_print("coot::cl_rt: calling clFlush()");
    coot_wrapper(clFlush)(get_rt().cl_rt.get_cq());  // submit all enqueued commands
    }

  get_rt().cl_rt.unlock();
  }




//
// adapt_uword

inline
runtime_t::adapt_uword::adapt_uword(const uword val)
  {
  if((sizeof(uword) >= 8) && get_rt().cl_rt.has_sizet64())
    {
    size  = sizeof(u64);
    addr  = (void*)(&val64);
    val64 = u64(val);
    }
  else
    {
    size   = sizeof(u32);
    addr   = (void*)(&val32);
    val32  = u32(val);

    coot_check_runtime_error( ((sizeof(uword) >= 8) && (val > 0xffffffffU)), "adapt_uword: given value doesn't fit into unsigned 32 bit integer" );
    }
  }
