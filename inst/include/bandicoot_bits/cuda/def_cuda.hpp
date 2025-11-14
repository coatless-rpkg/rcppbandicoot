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



  extern CUresult coot_wrapper(cuInit)(unsigned int flags);
  extern CUresult coot_wrapper(cuDeviceGetCount)(int* count);
  extern CUresult coot_wrapper(cuDeviceGet)(CUdevice* device, int ordinal);
  extern CUresult coot_wrapper(cuDeviceGetAttribute)(int* pi, CUdevice_attribute attrib, CUdevice dev);
  // The number of parameters changed in CUDA 13.
  #if CUDA_VERSION >= 13000
  extern CUresult coot_wrapper(cuCtxCreate)(CUcontext* pctx, CUctxCreateParams* ctxCreateParams, unsigned int flags, CUdevice dev);
  #else
  extern CUresult coot_wrapper(cuCtxCreate)(CUcontext* pctx, unsigned int flags, CUdevice dev);
  #endif
  extern CUresult coot_wrapper(cuModuleLoadDataEx)(CUmodule* module, const void* image, unsigned int numOptions, CUjit_option* options, void** optionValues);
  extern CUresult coot_wrapper(cuModuleGetFunction)(CUfunction* hfunc, CUmodule hmod, const char* name);

  extern cudaError_t coot_wrapper(cudaGetDeviceProperties)(cudaDeviceProp* prop, int device);
  extern cudaError_t coot_wrapper(cudaRuntimeGetVersion)(int* runtimeVersion);



  //
  // memory handling
  //


  extern cudaError_t coot_wrapper(cudaMalloc)(void** devPtr, size_t size);
  extern cudaError_t coot_wrapper(cudaFree)(void* devPtr);
  extern cudaError_t coot_wrapper(cudaMemcpy)(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
  extern cudaError_t coot_wrapper(cudaMemcpy2D)(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind);



  //
  // running kernels
  //



  extern CUresult coot_wrapper(cuLaunchKernel)(CUfunction f,
                                               unsigned int gridDimX,
                                               unsigned int gridDimY,
                                               unsigned int gridDimZ,
                                               unsigned int blockDimX,
                                               unsigned int blockDimY,
                                               unsigned int blockDimZ,
                                               unsigned int sharedMemBytes,
                                               CUstream hStream,
                                               void** kernelParams,
                                               void** extra);



  //
  // synchronisation
  //


  extern CUresult coot_wrapper(cuCtxSynchronize)();



  }
