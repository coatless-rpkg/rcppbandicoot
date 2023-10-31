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



  extern nvrtcResult coot_wrapper(nvrtcCreateProgram)(nvrtcProgram* prog,
                                                      const char* src,
                                                      const char* name,
                                                      int numHeaders,
                                                      const char* const* headers,
                                                      const char* const* includeNames);



  extern nvrtcResult coot_wrapper(nvrtcCompileProgram)(nvrtcProgram prog,
                                                       int numOptions,
                                                       const char* const* options);



  extern nvrtcResult coot_wrapper(nvrtcGetProgramLogSize)(nvrtcProgram prog,
                                                          size_t* logSizeRet);



  extern nvrtcResult coot_wrapper(nvrtcGetProgramLog)(nvrtcProgram prog,
                                                      char* log);



  extern nvrtcResult coot_wrapper(nvrtcGetCUBINSize)(nvrtcProgram prog,
                                                     size_t* cubinSizeRet);



  extern nvrtcResult coot_wrapper(nvrtcGetCUBIN)(nvrtcProgram prog,
                                                 char* cubin);



  }
