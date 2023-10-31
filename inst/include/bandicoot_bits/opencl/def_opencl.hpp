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



extern "C"
  {



  //
  // setup functions
  //



  extern cl_context coot_wrapper(clCreateContext)(const cl_context_properties* properties,
                                                  cl_uint num_devices,
                                                  const cl_device_id* devices,
                                                  void (CL_CALLBACK* pfn_notify)(const char* errinfo,
                                                                                 const void* private_info,
                                                                                 size_t cb,
                                                                                 void* user_data),
                                                  void* user_data,
                                                  cl_int* errcode_ret);



  extern cl_command_queue coot_wrapper(clCreateCommandQueue)(cl_context context,
                                                             cl_device_id device,
                                                             cl_command_queue_properties properties,
                                                             cl_int* errcode_ret);



  extern cl_int coot_wrapper(clGetPlatformIDs)(cl_uint num_entries,
                                               cl_platform_id* platforms,
                                               cl_uint* num_platforms);



  extern cl_int coot_wrapper(clGetDeviceIDs)(cl_platform_id platform,
                                             cl_device_type device_type,
                                             cl_uint num_entries,
                                             cl_device_id* devices,
                                             cl_uint* num_devices);



  extern cl_int coot_wrapper(clGetDeviceInfo)(cl_device_id device,
                                              cl_device_info param_name,
                                              size_t param_value_size,
                                              void* param_value,
                                              size_t* param_value_size_ret);



  //
  // kernel compilation
  //



  extern cl_program coot_wrapper(clCreateProgramWithSource)(cl_context context,
                                                            cl_uint count,
                                                            const char** strings,
                                                            const size_t* lengths,
                                                            cl_int* errcode_ret);



  extern cl_program coot_wrapper(clCreateProgramWithBinary)(cl_context context,
                                                            cl_uint num_devices,
                                                            const cl_device_id* device_list,
                                                            const size_t* lengths,
                                                            const unsigned char** binaries,
                                                            cl_int* binary_status,
                                                            cl_int* errcode_ret);



  extern cl_int coot_wrapper(clBuildProgram)(cl_program program,
                                             cl_uint num_devices,
                                             const cl_device_id* device_list,
                                             const char* options,
                                             void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
                                             void* user_data);



  extern cl_int coot_wrapper(clGetProgramBuildInfo)(cl_program program,
                                                    cl_device_id device,
                                                    cl_program_build_info param_name,
                                                    size_t param_value_size,
                                                    void* param_value,
                                                    size_t* param_value_size_ret);



  extern cl_int coot_wrapper(clGetProgramInfo)(cl_program program,
                                               cl_program_info param_name,
                                               size_t param_value_size,
                                               void* param_value,
                                               size_t* param_value_size_ret);



  extern cl_kernel coot_wrapper(clCreateKernel)(cl_program program,
                                                const char* kernel_name,
                                                cl_int* errcode_ret);



  #if defined(CL_VERSION_2_1)
  extern cl_int coot_wrapper(clGetKernelSubGroupInfo)(cl_kernel kernel,
                                                      cl_device_id device,
                                                      cl_kernel_sub_group_info param_name,
                                                      size_t input_value_size,
                                                      const void* input_value,
                                                      size_t param_value_size,
                                                      void* param_value,
                                                      size_t* param_value_size_ret);
  #elif defined(cl_khr_subgroups) || defined(cl_intel_subgroups)
  extern cl_int coot_wrapper(clGetKernelSubGroupInfoKHR)(cl_kernel kernel,
                                                         cl_device_id device,
                                                         cl_kernel_sub_group_info param_name,
                                                         size_t input_value_size,
                                                         const void* input_value,
                                                         size_t param_value_size,
                                                         void* param_value,
                                                         size_t* param_value_size_ret);
  #endif



  //
  // memory handling functions
  //



  extern cl_mem coot_wrapper(clCreateBuffer)(cl_context context,
                                             cl_mem_flags flags,
                                             size_t size,
                                             void* host_ptr,
                                             cl_int* errcode_ret);



  extern cl_int coot_wrapper(clEnqueueReadBuffer)(cl_command_queue command_queue,
                                                  cl_mem buffer,
                                                  cl_bool blocking_read,
                                                  size_t offset,
                                                  size_t size,
                                                  void* ptr,
                                                  cl_uint num_events_in_wait_list,
                                                  const cl_event* event_wait_list,
                                                  cl_event* event);



  extern cl_int coot_wrapper(clEnqueueWriteBuffer)(cl_command_queue command_queue,
                                                   cl_mem buffer,
                                                   cl_bool blocking_write,
                                                   size_t offset,
                                                   size_t size,
                                                   const void* ptr,
                                                   cl_uint num_events_in_wait_list,
                                                   const cl_event* event_wait_list,
                                                   cl_event* event);



  extern cl_int coot_wrapper(clEnqueueReadBufferRect)(cl_command_queue command_queue,
                                                      cl_mem buffer,
                                                      cl_bool blocking_read,
                                                      const size_t* buffer_origin,
                                                      const size_t* host_origin,
                                                      const size_t* region,
                                                      size_t buffer_row_pitch,
                                                      size_t buffer_slice_pitch,
                                                      size_t host_row_pitch,
                                                      size_t host_slice_pitch,
                                                      void* ptr,
                                                      cl_uint num_events_in_wait_list,
                                                      const cl_event* event_wait_list,
                                                      cl_event* event);



  extern cl_int coot_wrapper(clEnqueueWriteBufferRect)(cl_command_queue command_queue,
                                                       cl_mem buffer,
                                                       cl_bool blocking_write,
                                                       const size_t* buffer_origin,
                                                       const size_t* host_origin,
                                                       const size_t* region,
                                                       size_t buffer_row_pitch,
                                                       size_t buffer_slice_pitch,
                                                       size_t host_row_pitch,
                                                       size_t host_slice_pitch,
                                                       const void* ptr,
                                                       cl_uint num_events_in_wait_list,
                                                       const cl_event* event_wait_list,
                                                       cl_event* event);



  extern void* coot_wrapper(clEnqueueMapBuffer)(cl_command_queue command_queue,
                                                cl_mem buffer,
                                                cl_bool blocking_map,
                                                cl_map_flags map_flags,
                                                size_t offset,
                                                size_t size,
                                                cl_uint num_events_in_wait_list,
                                                const cl_event* event_wait_list,
                                                cl_event* event,
                                                cl_int* errcode_ret);



  extern cl_int coot_wrapper(clEnqueueUnmapMemObject)(cl_command_queue command_queue,
                                                      cl_mem memobj,
                                                      void* mapped_ptr,
                                                      cl_uint num_events_in_wait_list,
                                                      const cl_event* event_wait_list,
                                                      cl_event* event);



  extern cl_int coot_wrapper(clEnqueueCopyBuffer)(cl_command_queue command_queue,
                                                  cl_mem src_buffer,
                                                  cl_mem dst_buffer,
                                                  size_t src_offset,
                                                  size_t dst_offset,
                                                  size_t size,
                                                  cl_uint num_events_in_wait_list,
                                                  const cl_event* event_wait_list,
                                                  cl_event* event);



  extern cl_int coot_wrapper(clEnqueueCopyBufferRect)(cl_command_queue command_queue,
                                                      cl_mem src_buffer,
                                                      cl_mem dst_buffer,
                                                      const size_t* src_origin,
                                                      const size_t* dst_origin,
                                                      const size_t* region,
                                                      size_t src_row_pitch,
                                                      size_t src_slice_pitch,
                                                      size_t dst_row_pitch,
                                                      size_t dst_slice_pitch,
                                                      cl_uint num_events_in_wait_list,
                                                      const cl_event* event_wait_list,
                                                      cl_event* event);



  //
  // running kernels
  //



  extern cl_int coot_wrapper(clSetKernelArg)(cl_kernel kernel,
                                             cl_uint arg_index,
                                             size_t arg_size,
                                             const void* arg_value);



  extern cl_int coot_wrapper(clGetKernelWorkGroupInfo)(cl_kernel kernel,
                                                       cl_device_id device,
                                                       cl_kernel_work_group_info param_name,
                                                       size_t param_value_size,
                                                       void* param_value,
                                                       size_t* param_value_size_ret);



  extern cl_int coot_wrapper(clEnqueueNDRangeKernel)(cl_command_queue command_queue,
                                                     cl_kernel kernel,
                                                     cl_uint work_dim,
                                                     const size_t* global_work_offset,
                                                     const size_t* global_work_size,
                                                     const size_t* local_work_size,
                                                     cl_uint num_events_in_wait_list,
                                                     const cl_event* event_wait_list,
                                                     cl_event* event);



  extern cl_int coot_wrapper(clEnqueueTask)(cl_command_queue command_queue,
                                            cl_kernel kernel,
                                            cl_uint num_events_in_wait_list,
                                            const cl_event* event_wait_list,
                                            cl_event* event);



  //
  // synchronisation
  //



  extern cl_int coot_wrapper(clFinish)(cl_command_queue command_queue);
  extern cl_int coot_wrapper(clFlush)(cl_command_queue command_queue);



  //
  // cleanup
  //



  extern cl_int coot_wrapper(clReleaseMemObject)(cl_mem memobj);
  extern cl_int coot_wrapper(clReleaseKernel)(cl_kernel kernel);
  extern cl_int coot_wrapper(clReleaseProgram)(cl_program program);
  extern cl_int coot_wrapper(clReleaseCommandQueue)(cl_command_queue command_queue);
  extern cl_int coot_wrapper(clReleaseContext)(cl_context context);



  }
