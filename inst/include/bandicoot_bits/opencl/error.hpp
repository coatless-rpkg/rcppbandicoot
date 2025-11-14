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



struct coot_cl_error
  {
  coot_cold
  static
  inline
  std::string
  as_string(const cl_int error_code)
    {
    switch(error_code)
      {
      case   0: return "cl_success";
      case  -1: return "cl_device_not_found";
      case  -2: return "cl_device_not_available";
      case  -3: return "cl_compiler_not_available";
      case  -4: return "cl_mem_object_allocation_failure";
      case  -5: return "cl_out_of_resources";
      case  -6: return "cl_out_of_host_memory";
      case  -7: return "cl_profiling_info_not_available";
      case  -8: return "cl_mem_copy_overlap";
      case  -9: return "cl_image_format_mismatch";
      case -10: return "cl_image_format_not_supported";
      case -11: return "cl_build_program_failure";
      case -12: return "cl_map_failure";
      case -13: return "cl_misaligned_sub_buffer_offset";
      case -14: return "cl_exec_status_error_for_events_in_wait_list";
      case -15: return "cl_compile_program_failure";
      case -16: return "cl_linker_not_available";
      case -17: return "cl_link_program_failure";
      case -18: return "cl_device_partition_failed";
      case -19: return "cl_kernel_arg_info_not_available";
      case -30: return "cl_invalid_value";
      case -31: return "cl_invalid_device_type";
      case -32: return "cl_invalid_platform";
      case -33: return "cl_invalid_device";
      case -34: return "cl_invalid_context";
      case -35: return "cl_invalid_queue_properties";
      case -36: return "cl_invalid_command_queue";
      case -37: return "cl_invalid_host_ptr";
      case -38: return "cl_invalid_mem_object";
      case -39: return "cl_invalid_image_format_descriptor";
      case -40: return "cl_invalid_image_size";
      case -41: return "cl_invalid_sampler";
      case -42: return "cl_invalid_binary";
      case -43: return "cl_invalid_build_options";
      case -44: return "cl_invalid_program";
      case -45: return "cl_invalid_program_executable";
      case -46: return "cl_invalid_kernel_name";
      case -47: return "cl_invalid_kernel_definition";
      case -48: return "cl_invalid_kernel";
      case -49: return "cl_invalid_arg_index";
      case -50: return "cl_invalid_arg_value";
      case -51: return "cl_invalid_arg_size";
      case -52: return "cl_invalid_kernel_args";
      case -53: return "cl_invalid_work_dimension";
      case -54: return "cl_invalid_work_group_size";
      case -55: return "cl_invalid_work_item_size";
      case -56: return "cl_invalid_global_offset";
      case -57: return "cl_invalid_event_wait_list";
      case -58: return "cl_invalid_event";
      case -59: return "cl_invalid_operation";
      case -60: return "cl_invalid_gl_object";
      case -61: return "cl_invalid_buffer_size";
      case -62: return "cl_invalid_mip_level";
      case -63: return "cl_invalid_global_work_size";
      case -64: return "cl_invalid_property";
      case -65: return "cl_invalid_image_descriptor";
      case -66: return "cl_invalid_compiler_options";
      case -67: return "cl_invalid_linker_options";
      case -68: return "cl_invalid_device_partition_count";
      case -69: return "cl_invalid_pipe_size";
      case -70: return "cl_invalid_device_queue";
      default:  return "unknown error code";
      }
    }
  };



#if defined(COOT_USE_CLBLAS)
struct coot_clblas_error
  {
  coot_cold
  static
  inline
  std::string
  as_string(const cl_int error_code)
    {
    switch(error_code)
      {
      case clblasSuccess             : return "cl_success";
      case clblasInvalidValue        : return "cl_invalid_value";
      case clblasInvalidCommandQueue : return "cl_invalid_command_queue";
      case clblasInvalidContext      : return "cl_invalid_context";
      case clblasInvalidMemObject    : return "cl_invalid_mem_object";
      case clblasInvalidDevice       : return "cl_invalid_device";
      case clblasInvalidEventWaitList: return "cl_invalid_event_wait_list";
      case clblasOutOfResources      : return "cl_out_of_resources";
      case clblasOutOfHostMemory     : return "cl_out_of_host_memory";
      case clblasInvalidOperation    : return "cl_invalid_operation";
      case clblasCompilerNotAvailable: return "cl_compiler_not_available";
      case clblasBuildProgramFailure : return "cl_build_program_failure";
      // extended codes onwards
      case clblasNotImplemented      : return "functionality is not implemented";
      case clblasNotInitialized      : return "clblas library is not initialized yet";
      case clblasInvalidMatA         : return "matrix A is not a valid memory object";
      case clblasInvalidMatB         : return "matrix B is not a valid memory object";
      case clblasInvalidMatC         : return "matrix C is not a valid memory object";
      case clblasInvalidVecX         : return "vector X is not a valid memory object";
      case clblasInvalidVecY         : return "vector Y is not a valid memory object";
      case clblasInvalidDim          : return "an input dimension (M,N,K) is invalid";
      case clblasInvalidLeadDimA     : return "leading dimension A must not be less than the size of the first dimension";
      case clblasInvalidLeadDimB     : return "leading dimension B must not be less than the size of the second dimension";
      case clblasInvalidLeadDimC     : return "leading dimension C must not be less than the size of the third dimension";
      case clblasInvalidIncX         : return "the increment for a vector X must not be 0";
      case clblasInvalidIncY         : return "the increment for a vector Y must not be 0";
      case clblasInsufficientMemMatA : return "the memory object for Matrix A is too small";
      case clblasInsufficientMemMatB : return "the memory object for Matrix B is too small";
      case clblasInsufficientMemMatC : return "the memory object for Matrix C is too small";
      case clblasInsufficientMemVecX : return "the memory object for Vector X is too small";
      case clblasInsufficientMemVecY : return "the memory object for Vector Y is too small";
      default:                         return "unknown clBLAS error code";
      }
    }
  };
#endif



#if defined(COOT_USE_CLBLAST)
struct coot_clblast_error
  {
  coot_cold
  static
  inline
  std::string
  as_string(const CLBlastStatusCode error_code)
    {
    switch(error_code)
      {
      // Status codes in common with the OpenCL standard
      case CLBlastSuccess:                    return coot_cl_error::as_string(CL_SUCCESS);
      case CLBlastOpenCLCompilerNotAvailable: return coot_cl_error::as_string(CL_COMPILER_NOT_AVAILABLE);
      case CLBlastTempBufferAllocFailure:     return coot_cl_error::as_string(CL_MEM_OBJECT_ALLOCATION_FAILURE);
      case CLBlastOpenCLOutOfResources:       return coot_cl_error::as_string(CL_OUT_OF_RESOURCES);
      case CLBlastOpenCLOutOfHostMemory:      return coot_cl_error::as_string(CL_OUT_OF_HOST_MEMORY);
      case CLBlastOpenCLBuildProgramFailure:  return coot_cl_error::as_string(CL_BUILD_PROGRAM_FAILURE);
      case CLBlastInvalidValue:               return coot_cl_error::as_string(CL_INVALID_VALUE);
      case CLBlastInvalidCommandQueue:        return coot_cl_error::as_string(CL_INVALID_COMMAND_QUEUE);
      case CLBlastInvalidMemObject:           return coot_cl_error::as_string(CL_INVALID_MEM_OBJECT);
      case CLBlastInvalidBinary:              return coot_cl_error::as_string(CL_INVALID_BINARY);
      case CLBlastInvalidBuildOptions:        return coot_cl_error::as_string(CL_INVALID_BUILD_OPTIONS);
      case CLBlastInvalidProgram:             return coot_cl_error::as_string(CL_INVALID_PROGRAM);
      case CLBlastInvalidProgramExecutable:   return coot_cl_error::as_string(CL_INVALID_PROGRAM_EXECUTABLE);
      case CLBlastInvalidKernelName:          return coot_cl_error::as_string(CL_INVALID_KERNEL_NAME);
      case CLBlastInvalidKernelDefinition:    return coot_cl_error::as_string(CL_INVALID_KERNEL_DEFINITION);
      case CLBlastInvalidKernel:              return coot_cl_error::as_string(CL_INVALID_KERNEL);
      case CLBlastInvalidArgIndex:            return coot_cl_error::as_string(CL_INVALID_ARG_INDEX);
      case CLBlastInvalidArgValue:            return coot_cl_error::as_string(CL_INVALID_ARG_VALUE);
      case CLBlastInvalidArgSize:             return coot_cl_error::as_string(CL_INVALID_ARG_SIZE);
      case CLBlastInvalidKernelArgs:          return coot_cl_error::as_string(CL_INVALID_KERNEL_ARGS);
      case CLBlastInvalidLocalNumDimensions:  return coot_cl_error::as_string(CL_INVALID_WORK_DIMENSION);
      case CLBlastInvalidLocalThreadsTotal:   return coot_cl_error::as_string(CL_INVALID_WORK_GROUP_SIZE);
      case CLBlastInvalidLocalThreadsDim:     return coot_cl_error::as_string(CL_INVALID_WORK_ITEM_SIZE);
      case CLBlastInvalidGlobalOffset:        return coot_cl_error::as_string(CL_INVALID_GLOBAL_OFFSET);
      case CLBlastInvalidEventWaitList:       return coot_cl_error::as_string(CL_INVALID_EVENT_WAIT_LIST);
      case CLBlastInvalidEvent:               return coot_cl_error::as_string(CL_INVALID_EVENT);
      case CLBlastInvalidOperation:           return coot_cl_error::as_string(CL_INVALID_OPERATION);
      case CLBlastInvalidBufferSize:          return coot_cl_error::as_string(CL_INVALID_BUFFER_SIZE);
      case CLBlastInvalidGlobalWorkSize:      return coot_cl_error::as_string(CL_INVALID_GLOBAL_WORK_SIZE);

      // Status codes in common with the clBLAS library
      case CLBlastNotImplemented:             return "routine or functionality not implemented yet";
      case CLBlastInvalidMatrixA:             return "matrix A is not a valid OpenCL buffer";
      case CLBlastInvalidMatrixB:             return "matrix B is not a valid OpenCL buffer";
      case CLBlastInvalidMatrixC:             return "matrix C is not a valid OpenCL buffer";
      case CLBlastInvalidVectorX:             return "vector X is not a valid OpenCL buffer";
      case CLBlastInvalidVectorY:             return "vector Y is not a valid OpenCL buffer";
      case CLBlastInvalidDimension:           return "dimensions M, N, and K have to be larger than zero";
      case CLBlastInvalidLeadDimA:            return "leading dimension of A is smaller than the matrix's first dimension";
      case CLBlastInvalidLeadDimB:            return "leading dimension of B is smaller than the matrix's first dimension";
      case CLBlastInvalidLeadDimC:            return "leading dimension of C is smaller than the matrix's first dimension";
      case CLBlastInvalidIncrementX:          return "increment of vector X cannot be zero";
      case CLBlastInvalidIncrementY:          return "increment of vector Y cannot be zero";
      case CLBlastInsufficientMemoryA:        return "matrix A's OpenCL buffer is too small";
      case CLBlastInsufficientMemoryB:        return "matrix B's OpenCL buffer is too small";
      case CLBlastInsufficientMemoryC:        return "matrix C's OpenCL buffer is too small";
      case CLBlastInsufficientMemoryX:        return "vector X's OpenCL buffer is too small";
      case CLBlastInsufficientMemoryY:        return "vector Y's OpenCL buffer is too small";

      // Custom additional status codes for CLBlast
      case CLBlastInsufficientMemoryTemp:     return "temporary buffer provided to GEMM routine is too small";
      case CLBlastInvalidBatchCount:          return "the batch count needs to be positive";
      case CLBlastInvalidOverrideKernel:      return "trying to override parameters for an invalid kernel";
      case CLBlastMissingOverrideParameter:   return "missing override parameter(s) for the target kernel";
      case CLBlastInvalidLocalMemUsage:       return "not enough local memory available on this device";
      case CLBlastNoHalfPrecision:            return "half precision (16-bits) not supported by the device";
      case CLBlastNoDoublePrecision:          return "double precision (64-bits) not supported by the device";
      case CLBlastInvalidVectorScalar:        return "the unit-sized vector is not a valid OpenCL buffer";
      case CLBlastInsufficientMemoryScalar:   return "the unit-sized vector's OpenCL buffer is too small";
      case CLBlastDatabaseError:              return "entry for the device was not found in the database";
      case CLBlastUnknownError:               return "unspecified error";
      case CLBlastUnexpectedError:            return "unexpected exception";

      default:                                return "unknown CLBlast error code";
      }
    }
  };
#endif
