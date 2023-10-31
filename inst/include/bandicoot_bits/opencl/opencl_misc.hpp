// Copyright 2023 Ryan Curtin (http://www.ratml.org)
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

// This file contains safe implementations of OpenCL functions across different
// versions.  Use these instead of the direct OpenCL functions, since those
// OpenCL functions may not be present depending on what version of OpenCL is
// being used.

// coot_sub_group_size(k): get the subgroup size of a given kernel, if subgroups
// are supported; otherwise return 0.
#if defined(CL_VERSION_2_1) || defined(CL_VERSION_3_0)

inline
cl_int
coot_sub_group_size(cl_kernel kernel, cl_device_id dev_id, const size_t input_size, size_t& subgroup_size)
  {
  // OpenCL 2.0 moved this into the base specification, so the KHR suffix is
  // deprecated.
  return coot_wrapper(clGetKernelSubGroupInfo)(kernel,
                                               dev_id,
                                               CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE,
                                               sizeof(size_t),
                                               (const void*) &input_size,
                                               sizeof(size_t),
                                               (void*) &subgroup_size,
                                               NULL);
  }

#elif defined(cl_khr_subgroups) || defined(cl_intel_subgroups)

inline
cl_int
coot_sub_group_size(cl_kernel kernel, cl_device_id dev_id, const size_t input_size, size_t& subgroup_size)
  {
  // This is a version of OpenCL where subgroups are available, but only as an
  // extension.
  return coot_wrapper(clGetKernelSubGroupInfoKHR)(kernel,
                                                  dev_id,
                                                  CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE,
                                                  sizeof(size_t),
                                                  (const void*) &input_size,
                                                  sizeof(size_t),
                                                  (void*) &subgroup_size,
                                                  NULL);
  }

#else

inline
cl_int
coot_sub_group_size(cl_kernel kernel, cl_device_id dev_id, const size_t input_size, size_t& subgroup_size)
  {
  coot_ignore(kernel);
  coot_ignore(dev_id);
  coot_ignore(input_size);

  // Subgroups are not available, so assume a subgroup size of 0.
  // (This is bad because it does not allow us to do subgroup-level
  // synchronization operations, which are way less overhead than regular
  // barriers!)
  subgroup_size = 0;
  return CL_SUCCESS;
  }

#endif



// Return the warp size, if the device is an nvidia device.

inline
cl_int
coot_nv_warp_size(const cl_device_id dev_id, size_t& warp_size)
  {
#if defined(CL_DEVICE_WARP_SIZE_NV)
  cl_uint warp_size_out;
  cl_int status = coot_wrapper(clGetDeviceInfo)(dev_id, CL_DEVICE_WARP_SIZE_NV, sizeof(cl_uint), (void*) &warp_size_out, NULL);
  warp_size = (size_t) warp_size_out;
  return status;
#else
  warp_size = 0;
  return CL_SUCCESS;
#endif
  }
