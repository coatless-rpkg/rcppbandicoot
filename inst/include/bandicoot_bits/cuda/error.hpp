// Copyright 2019 Ryan Curtin (http://ratml.org)
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



inline std::string error_as_string(const cudaError_t error_code)
  {
  switch (error_code)
    {
    case cudaSuccess:                                   return "cudaSuccess";
    case cudaErrorInvalidValue:                         return "cudaErrorInvalidValue";
    case cudaErrorMemoryAllocation:                     return "cudaErrorMemoryAllocation";
    case cudaErrorInitializationError:                  return "cudaErrorInitializationError";
    case cudaErrorCudartUnloading:                      return "cudaErrorCudartUnloading";
    case cudaErrorProfilerDisabled:                     return "cudaErrorProfilerDisabled";
    case cudaErrorProfilerNotInitialized:               return "cudaErrorProfilerNotInitialized";
    case cudaErrorProfilerAlreadyStarted:               return "cudaErrorProfilerAlreadyStarted";
    case cudaErrorProfilerAlreadyStopped:               return "cudaErrorProfilerAlreadyStopped";
    case cudaErrorInvalidConfiguration:                 return "cudaErrorInvalidConfiguration";
    case cudaErrorInvalidPitchValue:                    return "cudaErrorInvalidPitchValue";
    case cudaErrorInvalidSymbol:                        return "cudaErrorInvalidSymbol";
    case cudaErrorInvalidHostPointer:                   return "cudaErrorInvalidHostPointer";
    case cudaErrorInvalidDevicePointer:                 return "cudaErrorInvalidDevicePointer";
    case cudaErrorInvalidTexture:                       return "cudaErrorInvalidTexture";
    case cudaErrorInvalidTextureBinding:                return "cudaErrorInvalidTextureBinding";
    case cudaErrorInvalidChannelDescriptor:             return "cudaErrorInvalidChannelDescriptor";
    case cudaErrorInvalidMemcpyDirection:               return "cudaErrorInvalidMemcpyDirection";
    case cudaErrorAddressOfConstant:                    return "cudaErrorAddressOfConstant";
    case cudaErrorTextureFetchFailed:                   return "cudaErrorTextureFetchFailed";
    case cudaErrorTextureNotBound:                      return "cudaErrorTextureNotBound";
    case cudaErrorSynchronizationError:                 return "cudaErrorSynchronizationError";
    case cudaErrorInvalidFilterSetting:                 return "cudaErrorInvalidFilterSetting";
    case cudaErrorInvalidNormSetting:                   return "cudaErrorInvalidNormSetting";
    case cudaErrorMixedDeviceExecution:                 return "cudaErrorMixedDeviceExecution";
    case cudaErrorNotYetImplemented:                    return "cudaErrorNotYetImplemented";
    case cudaErrorMemoryValueTooLarge:                  return "cudaErrorMemoryValueTooLarge";
    case cudaErrorInsufficientDriver:                   return "cudaErrorInsufficientDriver";
    case cudaErrorInvalidSurface:                       return "cudaErrorInvalidSurface";
    case cudaErrorDuplicateVariableName:                return "cudaErrorDuplicateVariableName";
    case cudaErrorDuplicateTextureName:                 return "cudaErrorDuplicateTextureName";
    case cudaErrorDuplicateSurfaceName:                 return "cudaErrorDuplicateSurfaceName";
    case cudaErrorDevicesUnavailable:                   return "cudaErrorDevicesUnavailable";
    case cudaErrorIncompatibleDriverContext:            return "cudaErrorIncompatibleDriverContext";
    case cudaErrorMissingConfiguration:                 return "cudaErrorMissingConfiguration";
    case cudaErrorPriorLaunchFailure:                   return "cudaErrorPriorLaunchFailure";
    case cudaErrorLaunchMaxDepthExceeded:               return "cudaErrorLaunchMaxDepthExceeded";
    case cudaErrorLaunchFileScopedTex:                  return "cudaErrorLaunchFileScopedTex";
    case cudaErrorLaunchFileScopedSurf:                 return "cudaErrorLaunchFileScopedSurf";
    case cudaErrorSyncDepthExceeded:                    return "cudaErrorSyncDepthExceeded";
    case cudaErrorLaunchPendingCountExceeded:           return "cudaErrorLaunchPendingCountExceeded";
    case cudaErrorInvalidDeviceFunction:                return "cudaErrorInvalidDeviceFunction";
    case cudaErrorNoDevice:                             return "cudaErrorNoDevice";
    case cudaErrorInvalidDevice:                        return "cudaErrorInvalidDevice";
    case cudaErrorStartupFailure:                       return "cudaErrorStartupFailure";
    case cudaErrorInvalidKernelImage:                   return "cudaErrorInvalidKernelImage";
    // amusing typo... try to read it out loud
//    case cudaErrorDeviceUninitilialized:                  return "cudaErrorDeviceUninitilialized";
    case cudaErrorMapBufferObjectFailed:                return "cudaErrorMapBufferObjectFailed";
    case cudaErrorUnmapBufferObjectFailed:              return "cudaErrorUnmapBufferObjectFailed";
//    case cudaErrorArrayIsMapped:                        return "cudaErrorArrayIsMapped";
//    case cudaErrorAlreadyMapped:                        return "cudaErrorAlreadyMapped";
    case cudaErrorNoKernelImageForDevice:               return "cudaErrorNoKernelImageForDevice";
//    case cudaErrorAlreadyAcquired:                      return "cudaErrorAlreadyAcquired";
//    case cudaErrorNotMapped:                            return "cudaErrorNotMapped";
//    case cudaErrorNotMappedAsArray:                     return "cudaErrorNotMappedAsArray";
//    case cudaErrorNotMappedAsPointer:                   return "cudaErrorNotMappedAsPointer";
    case cudaErrorECCUncorrectable:                     return "cudaErrorECCUncorrectable";
    case cudaErrorUnsupportedLimit:                     return "cudaErrorUnsupportedLimit";
    case cudaErrorDeviceAlreadyInUse:                   return "cudaErrorDeviceAlreadyInUse";
    case cudaErrorPeerAccessUnsupported:                return "cudaErrorPeerAccessUnsupported";
    case cudaErrorInvalidPtx:                           return "cudaErrorInvalidPtx";
    case cudaErrorInvalidGraphicsContext:               return "cudaErrorInvalidGraphicsContext";
    case cudaErrorNvlinkUncorrectable:                  return "cudaErrorNvlinkUncorrectable";
    case cudaErrorJitCompilerNotFound:                  return "cudaErrorJitCompilerNotFound";
//    case cudaErrorInvalidSource:                        return "cudaErrorInvalidSource";
//    case cudaErrorFileNotFound:                         return "cudaErrorFileNotFound";
    case cudaErrorSharedObjectSymbolNotFound:           return "cudaErrorSharedObjectSymbolNotFound";
    case cudaErrorSharedObjectInitFailed:               return "cudaErrorSharedObjectInitFailed";
    case cudaErrorOperatingSystem:                      return "cudaErrorOperatingSystem";
    case cudaErrorInvalidResourceHandle:                return "cudaErrorInvalidResourceHandle";
//    case cudaErrorIllegalState:                         return "cudaErrorIllegalState";
//    case cudaErrorSymbolNotFound:                       return "cudaErrorSymbolNotFound";
    case cudaErrorNotReady:                             return "cudaErrorNotReady";
    case cudaErrorIllegalAddress:                       return "cudaErrorIllegalAddress";
    case cudaErrorLaunchOutOfResources:                 return "cudaErrorLaunchOutOfResources";
    case cudaErrorLaunchTimeout:                        return "cudaErrorLaunchTimeout";
//    case cudaErrorLaunchIncompatibleTexturing:          return "cudaErrorLaunchIncompatibleTexturing";
    case cudaErrorPeerAccessAlreadyEnabled:             return "cudaErrorPeerAccessAlreadyEnabled";
    case cudaErrorPeerAccessNotEnabled:                 return "cudaErrorPeerAccessNotEnabled";
    case cudaErrorSetOnActiveProcess:                   return "cudaErrorSetOnActiveProcess";
//    case cudaErrorContextIsDestroyed:                   return "cudaErrorContextIsDestroyed";
    case cudaErrorAssert:                               return "cudaErrorAssert";
    case cudaErrorTooManyPeers:                         return "cudaErrorTooManyPeers";
    case cudaErrorHostMemoryAlreadyRegistered:          return "cudaErrorHostMemoryAlreadyRegistered";
    case cudaErrorHostMemoryNotRegistered:              return "cudaErrorHostMemoryNotRegistered";
    case cudaErrorHardwareStackError:                   return "cudaErrorHardwareStackError";
    case cudaErrorIllegalInstruction:                   return "cudaErrorIllegalInstruction";
    case cudaErrorMisalignedAddress:                    return "cudaErrorMisalignedAddress";
    case cudaErrorInvalidAddressSpace:                  return "cudaErrorInvalidAddressSpace";
    case cudaErrorInvalidPc:                            return "cudaErrorInvalidPc";
    case cudaErrorLaunchFailure:                        return "cudaErrorLaunchFailure";
    case cudaErrorCooperativeLaunchTooLarge:            return "cudaErrorCooperativeLaunchTooLarge";
    case cudaErrorNotPermitted:                         return "cudaErrorNotPermitted";
    case cudaErrorNotSupported:                         return "cudaErrorNotSupported";
//    case cudaErrorSystemNotReady:                       return "cudaErrorSystemNotReady";
//    case cudaErrorSystemDriverMismatch:                 return "cudaErrorSystemDriverMismatch";
//    case cudaErrorCompatNotSupportedOnDevice:           return "cudaErrorCompatNotSupportedOnDevice";
//    case cudaErrorStreamCaptureUnsupported:             return "cudaErrorStreamCaptureUnsupported";
//    case cudaErrorStreamCaptureInvalidated:             return "cudaErrorStreamCaptureInvalidated";
//    case cudaErrorStreamCaptureMerge:                   return "cudaErrorStreamCaptureMerge";
//    case cudaErrorStreamCaptureUnmatched:               return "cudaErrorStreamCaptureUnmatched";
//    case cudaErrorStreamCaptureUnjoined:                return "cudaErrorStreamCaptureUnjoined";
//    case cudaErrorStreamCaptureIsolation:               return "cudaErrorStreamCaptureIsolation";
//    case cudaErrorStreamCaptureImplicit:                return "cudaErrorStreamCaptureImplicit";
//    case cudaErrorCapturedEvent:                        return "cudaErrorCapturedEvent";
//    case cudaErrorStreamCaptureWrongThread:             return "cudaErrorStreamCaptureWrongThread";
    case cudaErrorUnknown:                              return "cudaErrorUnknown";
    case cudaErrorApiFailureBase:                       return "cudaErrorApiFailureBase";
    default:                                            return "unknown error code";
    }
  }



inline std::string error_as_string(const CUresult error_code)
  {
  switch (error_code)
    {
    case CUDA_SUCCESS:                                  return "CUDA_SUCCESS";
    case CUDA_ERROR_INVALID_VALUE:                      return "CUDA_ERROR_INVALID_VALUE";
    case CUDA_ERROR_OUT_OF_MEMORY:                      return "CUDA_ERROR_OUT_OF_MEMORY";
    case CUDA_ERROR_NOT_INITIALIZED:                    return "CUDA_ERROR_NOT_INITIALIZED";
    case CUDA_ERROR_DEINITIALIZED:                      return "CUDA_ERROR_DEINITIALIZED";
    case CUDA_ERROR_PROFILER_DISABLED:                  return "CUDA_ERROR_PROFILER_DISABLED";
    case CUDA_ERROR_PROFILER_NOT_INITIALIZED:           return "CUDA_ERROR_PROFILER_NOT_INITIALIZED";
    case CUDA_ERROR_PROFILER_ALREADY_STARTED:           return "CUDA_ERROR_PROFILER_ALREADY_STARTED";
    case CUDA_ERROR_PROFILER_ALREADY_STOPPED:           return "CUDA_ERROR_PROFILER_ALREADY_STOPPED";
    case CUDA_ERROR_NO_DEVICE:                          return "CUDA_ERROR_NO_DEVICE";
    case CUDA_ERROR_INVALID_DEVICE:                     return "CUDA_ERROR_INVALID_DEVICE";
    case CUDA_ERROR_INVALID_IMAGE:                      return "CUDA_ERROR_INVALID_IMAGE";
    case CUDA_ERROR_INVALID_CONTEXT:                    return "CUDA_ERROR_INVALID_CONTEXT";
    case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:            return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";
    case CUDA_ERROR_MAP_FAILED:                         return "CUDA_ERROR_MAP_FAILED";
    case CUDA_ERROR_UNMAP_FAILED:                       return "CUDA_ERROR_UNMAP_FAILED";
    case CUDA_ERROR_ARRAY_IS_MAPPED:                    return "CUDA_ERROR_ARRAY_IS_MAPPED";
    case CUDA_ERROR_ALREADY_MAPPED:                     return "CUDA_ERROR_ALREADY_MAPPED";
    case CUDA_ERROR_NO_BINARY_FOR_GPU:                  return "CUDA_ERROR_NO_BINARY_FOR_GPU";
    case CUDA_ERROR_ALREADY_ACQUIRED:                   return "CUDA_ERROR_ALREADY_ACQUIRED";
    case CUDA_ERROR_NOT_MAPPED:                         return "CUDA_ERROR_NOT_MAPPED";
    case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:                return "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";
    case CUDA_ERROR_NOT_MAPPED_AS_POINTER:              return "CUDA_ERROR_NOT_MAPPED_AS_POINTER";
    case CUDA_ERROR_ECC_UNCORRECTABLE:                  return "CUDA_ERROR_ECC_UNCORRECTABLE";
    case CUDA_ERROR_UNSUPPORTED_LIMIT:                  return "CUDA_ERROR_UNSUPPORTED_LIMIT";
    case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:             return "CUDA_ERROR_CONTEXT_ALREADY_IN_USE";
    case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:            return "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED";
    case CUDA_ERROR_INVALID_PTX:                        return "CUDA_ERROR_INVALID_PTX";
    case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT:           return "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT";
    case CUDA_ERROR_NVLINK_UNCORRECTABLE:               return "CUDA_ERROR_NVLINK_UNCORRECTABLE";
    case CUDA_ERROR_JIT_COMPILER_NOT_FOUND:             return "CUDA_ERROR_JIT_COMPILER_NOT_FOUND";
    case CUDA_ERROR_INVALID_SOURCE:                     return "CUDA_ERROR_INVALID_SOURCE";
    case CUDA_ERROR_FILE_NOT_FOUND:                     return "CUDA_ERROR_FILE_NOT_FOUND";
    case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:     return "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";
    case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:          return "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";
    case CUDA_ERROR_OPERATING_SYSTEM:                   return "CUDA_ERROR_OPERATING_SYSTEM";
    case CUDA_ERROR_INVALID_HANDLE:                     return "CUDA_ERROR_INVALID_HANDLE";
//    case CUDA_ERROR_ILLEGAL_STATE:                      return "CUDA_ERROR_ILLEGAL_STATE";
    case CUDA_ERROR_NOT_FOUND:                          return "CUDA_ERROR_NOT_FOUND";
    case CUDA_ERROR_NOT_READY:                          return "CUDA_ERROR_NOT_READY";
    case CUDA_ERROR_ILLEGAL_ADDRESS:                    return "CUDA_ERROR_ILLEGAL_ADDRESS";
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:            return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";
    case CUDA_ERROR_LAUNCH_TIMEOUT:                     return "CUDA_ERROR_LAUNCH_TIMEOUT";
    case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:      return "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";
    case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:        return "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";
    case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:            return "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";
    case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:             return "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE";
    case CUDA_ERROR_CONTEXT_IS_DESTROYED:               return "CUDA_ERROR_CONTEXT_IS_DESTROYED";
    case CUDA_ERROR_ASSERT:                             return "CUDA_ERROR_ASSERT";
    case CUDA_ERROR_TOO_MANY_PEERS:                     return "CUDA_ERROR_TOO_MANY_PEERS";
    case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:     return "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED";
    case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:         return "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED";
    case CUDA_ERROR_HARDWARE_STACK_ERROR:               return "CUDA_ERROR_HARDWARE_STACK_ERROR";
    case CUDA_ERROR_ILLEGAL_INSTRUCTION:                return "CUDA_ERROR_ILLEGAL_INSTRUCTION";
    case CUDA_ERROR_MISALIGNED_ADDRESS:                 return "CUDA_ERROR_MISALIGNED_ADDRESS";
    case CUDA_ERROR_INVALID_ADDRESS_SPACE:              return "CUDA_ERROR_INVALID_ADDRESS_SPACE";
    case CUDA_ERROR_INVALID_PC:                         return "CUDA_ERROR_INVALID_PC";
    case CUDA_ERROR_LAUNCH_FAILED:                      return "CUDA_ERROR_LAUNCH_FAILED";
    case CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE:       return "CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE";
    case CUDA_ERROR_NOT_PERMITTED:                      return "CUDA_ERROR_NOT_PERMITTED";
    case CUDA_ERROR_NOT_SUPPORTED:                      return "CUDA_ERROR_NOT_SUPPORTED";
//    case CUDA_ERROR_SYSTEM_NOT_READY:                   return "CUDA_ERROR_SYSTEM_NOT_READY";
//    case CUDA_ERROR_SYSTEM_DRIVER_MISMATCH:             return "CUDA_ERROR_SYSTEM_DRIVER_MISMATCH";
//    case CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE:     return "CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE";
//    case CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED:         return "CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED";
//    case CUDA_ERROR_STREAM_CAPTURE_INVALIDATED:         return "CUDA_ERROR_STREAM_CAPTURE_INVALIDATED";
//    case CUDA_ERROR_STREAM_CAPTURE_MERGE:               return "CUDA_ERROR_STREAM_CAPTURE_MERGE";
//    case CUDA_ERROR_STREAM_CAPTURE_UNMATCHED:           return "CUDA_ERROR_STREAM_CAPTURE_UNMATCHED";
//    case CUDA_ERROR_STREAM_CAPTURE_UNJOINED:            return "CUDA_ERROR_STREAM_CAPTURE_UNJOINED";
//    case CUDA_ERROR_STREAM_CAPTURE_ISOLATION:           return "CUDA_ERROR_STREAM_CAPTURE_ISOLATION";
//    case CUDA_ERROR_STREAM_CAPTURE_IMPLICIT:            return "CUDA_ERROR_STREAM_CAPTURE_IMPLICIT";
//    case CUDA_ERROR_CAPTURED_EVENT:                     return "CUDA_ERROR_CAPTURED_EVENT";
//    case CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD:        return "CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD";
    case CUDA_ERROR_UNKNOWN:                            return "CUDA_ERROR_UNKNOWN";
    default:                                            return "unknown error code";
    }
  }



inline std::string error_as_string(const nvrtcResult error_code)
  {
  switch(error_code)
    {
    case NVRTC_SUCCESS:                                       return "NVRTC_SUCCESS";
    case NVRTC_ERROR_OUT_OF_MEMORY:                           return "NVRTC_ERROR_OUT_OF_MEMORY";
    case NVRTC_ERROR_PROGRAM_CREATION_FAILURE:                return "NVRTC_ERROR_PROGRAM_CREATION_FAILURE";
    case NVRTC_ERROR_INVALID_INPUT:                           return "NVRTC_ERROR_INVALID_INPUT";
    case NVRTC_ERROR_INVALID_PROGRAM:                         return "NVRTC_ERROR_INVALID_PROGRAM";
    case NVRTC_ERROR_INVALID_OPTION:                          return "NVRTC_ERROR_INVALID_OPTION";
    case NVRTC_ERROR_COMPILATION:                             return "NVRTC_ERROR_COMPILATION";
    case NVRTC_ERROR_BUILTIN_OPERATION_FAILURE:               return "NVRTC_ERROR_BUILTIN_OPERATION_FAILURE";
    case NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION:   return "NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION";
    case NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION:     return "NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION";
    case NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID:               return "NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID";
    case NVRTC_ERROR_INTERNAL_ERROR:                          return "NVRTC_ERROR_INTERNAL_ERROR";
    default:                                                  return "unknown error code";
    }
  }



inline std::string error_as_string(const cusolverStatus_t error_code)
  {
  switch(error_code)
    {
    case CUSOLVER_STATUS_SUCCESS:                     return "CUSOLVER_STATUS_SUCCESS";
    case CUSOLVER_STATUS_NOT_INITIALIZED:             return "CUSOLVER_STATUS_NOT_INITIALIZED";
    case CUSOLVER_STATUS_ALLOC_FAILED:                return "CUSOLVER_STATUS_ALLOC_FAILED";
    case CUSOLVER_STATUS_INVALID_VALUE:               return "CUSOLVER_STATUS_INVALID_VALUE";
    case CUSOLVER_STATUS_ARCH_MISMATCH:               return "CUSOLVER_STATUS_ARCH_MISMATCH";
    case CUSOLVER_STATUS_EXECUTION_FAILED:            return "CUSOLVER_STATUS_EXECUTION_FAILED";
    case CUSOLVER_STATUS_INTERNAL_ERROR:              return "CUSOLVER_STATUS_INTERNAL_ERROR";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:   return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    default:                                          return "unknown cuSolver error code";
    }
  }



inline std::string error_as_string(const cublasStatus_t error_code)
  {
  switch(error_code)
    {
    case CUBLAS_STATUS_SUCCESS:               return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:       return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:          return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:         return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:         return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:         return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:        return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:         return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:         return "CUBLAS_STATUS_LICENSE_ERROR";
    default:                                  return "unknown cuBLAS error code";
    }
  }



inline std::string error_as_string(const curandStatus_t error_code)
  {
  switch(error_code)
    {
    case CURAND_STATUS_SUCCESS:               return "CURAND_STATUS_SUCCESS";
    case CURAND_STATUS_VERSION_MISMATCH:      return "CURAND_STATUS_VERSION_MISMATCH";
    case CURAND_STATUS_NOT_INITIALIZED:       return "CURAND_STATUS_NOT_INITIALIZED";
    case CURAND_STATUS_ALLOCATION_FAILED:     return "CURAND_STATUS_ALLOCATION_FAILED";
    case CURAND_STATUS_TYPE_ERROR:            return "CURAND_STATUS_TYPE_ERROR";
    case CURAND_STATUS_OUT_OF_RANGE:          return "CURAND_STATUS_OUT_OF_RANGE";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:   return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    case CURAND_STATUS_LAUNCH_FAILURE:        return "CURAND_STATUS_LAUNCH_FAILURE";
    case CURAND_STATUS_PREEXISTING_FAILURE:   return "CURAND_STATUS_PREEXISTING_FAILURE";
    case CURAND_STATUS_INITIALIZATION_FAILED: return "CURAND_STATUS_INITIALIZATION_FAILED";
    case CURAND_STATUS_ARCH_MISMATCH:         return "CURAND_STATUS_ARCH_MISMATCH";
    case CURAND_STATUS_INTERNAL_ERROR:        return "CURAND_STATUS_INTERNAL_ERROR";
    default:                                  return "unknown cuRand error code";
    }
  }
