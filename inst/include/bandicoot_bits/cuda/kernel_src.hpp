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

#define STR2(A) STR(A)
#define STR(A) #A

// utility functions for compiled-on-the-fly CUDA kernels

inline
std::string
get_cuda_src_preamble()
  {
  char u32_max[32];
  char u64_max[32];
  snprintf(u32_max, 32, "%llu", (unsigned long long) std::numeric_limits<u32>::max());
  snprintf(u64_max, 32, "%llu", (unsigned long long) std::numeric_limits<u64>::max());

  char s32_min[32];
  char s64_min[32];
  snprintf(s32_min, 32, "%llu", (unsigned long long) std::numeric_limits<u32>::min());
  snprintf(s64_min, 32, "%llu", (unsigned long long) std::numeric_limits<u64>::min());

  char s32_max[32];
  char s64_max[32];
  snprintf(s32_max, 32, "%llu", (unsigned long long) std::numeric_limits<u32>::max());
  snprintf(s64_max, 32, "%llu", (unsigned long long) std::numeric_limits<u64>::max());

  std::string source = \

  "#define uint unsigned int\n"
  "#define COOT_FN2(ARG1, ARG2)  ARG1 ## ARG2 \n"
  "#define COOT_FN(ARG1,ARG2) COOT_FN2(ARG1,ARG2)\n"
  "\n"
  "#define COOT_PI " STR2(M_PI) "\n"
  // Properties for specific types.
  "__device__ inline bool coot_is_fp(const uint) { return false; } \n"
  "__device__ inline bool coot_is_fp(const int) { return false; } \n"
  "__device__ inline bool coot_is_fp(const size_t) { return false; } \n"
  "__device__ inline bool coot_is_fp(const long) { return false; } \n"
  "__device__ inline bool coot_is_fp(const float) { return true; } \n"
  "__device__ inline bool coot_is_fp(const double) { return true; } \n"
  "\n"
  "__device__ inline bool coot_is_signed(const uint) { return false; } \n"
  "__device__ inline bool coot_is_signed(const int) { return true; } \n"
  "__device__ inline bool coot_is_signed(const size_t) { return false; } \n"
  "__device__ inline bool coot_is_signed(const long) { return true; } \n"
  "__device__ inline bool coot_is_signed(const float) { return true; } \n"
  "__device__ inline bool coot_is_signed(const double) { return true; } \n"
  "\n"
  // Utility functions to return the correct min/max value for a given type.
  // These constants are not defined in the CUDA compilation environment so we use the host's version.
  "__device__ inline uint   coot_type_min(const uint)   { return 0; } \n"
  "__device__ inline int    coot_type_min(const int)    { return " + std::string(s32_min) + "; } \n"
  "__device__ inline size_t coot_type_min(const size_t) { return 0; } \n"
  "__device__ inline long   coot_type_min(const long)   { return " + std::string(s64_min) + "; } \n"
  "__device__ inline float  coot_type_min(const float)  { return " STR2(FLT_MIN) "; } \n"
  "__device__ inline double coot_type_min(const double) { return " STR2(DBL_MIN) "; } \n"
  "\n"
  "__device__ inline uint   coot_type_max(const uint)   { return " + std::string(u32_max) + "; } \n"
  "__device__ inline int    coot_type_max(const int)    { return " + std::string(s32_min) + "; } \n"
  "__device__ inline size_t coot_type_max(const size_t) { return " + std::string(u64_max) + "; } \n"
  "__device__ inline long   coot_type_max(const long)   { return " + std::string(s64_max) + "; } \n"
  "__device__ inline float  coot_type_max(const float)  { return " STR2(FLT_MAX) "; } \n"
  "__device__ inline double coot_type_max(const double) { return " STR2(DBL_MAX) "; } \n"
  "\n"
  "__device__ inline bool   coot_isnan(const uint)     { return false;    } \n"
  "__device__ inline bool   coot_isnan(const int)      { return false;    } \n"
  "__device__ inline bool   coot_isnan(const size_t)   { return false;    } \n"
  "__device__ inline bool   coot_isnan(const long)     { return false;    } \n"
  "__device__ inline bool   coot_isnan(const float x)  { return isnan(x); } \n"
  "__device__ inline bool   coot_isnan(const double x) { return isnan(x); } \n"
  "\n"
  "extern \"C\" {\n"
  "\n"
  "extern __shared__ char aux_shared_mem[]; \n" // this may be used in some kernels
  "\n"
  // u32 maps to "unsigned int", so we have to avoid ever using that name.
  "__device__ inline int  coot_type_max_u_float()  { return " + std::string(u32_max) + "; } \n"
  "__device__ inline long coot_type_max_u_double() { return " + std::string(u64_max) + "; } \n"
  "\n"
  // Forward declaration used by some oneway_real kernels.
  "__device__ void u32_or_warp_reduce(volatile unsigned int* data, int tid);"
  "\n"
  ;

  return source;
  }



inline
std::string
get_cuda_src_epilogue()
  {
  return "}\n";
  }



inline
std::string
read_file(const std::string& filename)
  {
  // This is super hacky!  We eventually need a configuration system to track this.
  const std::string this_file = __FILE__;

  // We need to strip the '_src.hpp' from __FILE__.
  const std::string full_filename = this_file.substr(0, this_file.size() - 8) + "s/" + filename;
  std::ifstream f(full_filename);
  std::string file_contents = "";
  if (!f.is_open())
    {
    std::cout << "Failed to open " << full_filename << " (kernel source)!\n";
    throw std::runtime_error("Cannot open required kernel source.");
    }

  // Allocate memory for file contents.
  f.seekg(0, std::ios::end);
  file_contents.reserve(f.tellg());
  f.seekg(0, std::ios::beg);

  file_contents.assign(std::istreambuf_iterator<char>(f),
                       std::istreambuf_iterator<char>());

  return file_contents;
  }



inline
std::string
get_cuda_zeroway_kernel_src()
  {
  // NOTE: kernel names must match the list in the kernel_id struct
  //
  std::vector<std::string> aux_function_filenames = {
      "absdiff.cu",
      "philox.cu"
  };

  std::string result = "";

  // First, load any auxiliary functions (e.g. device-specific functions).
  for (const std::string& filename : aux_function_filenames)
    {
    std::string full_filename = "zeroway/" + filename;
    result += read_file(full_filename);
    }

  // Now, load each file for each kernel.
  for (const std::string& kernel_name : zeroway_kernel_id::get_names())
    {
    std::string filename = "zeroway/" + kernel_name + ".cu";
    result += read_file(filename);
    }

  return result;
  }



inline
std::string
get_cuda_oneway_kernel_src()
  {
  // NOTE: kernel names must match the list in the kernel_id struct

  std::vector<std::string> aux_function_filenames = {
      "accu_warp_reduce.cu",
      "min_warp_reduce.cu",
      "max_warp_reduce.cu",
      "index_min_warp_reduce.cu",
      "index_max_warp_reduce.cu",
      "prod_warp_reduce.cu"
  };

  std::string result = "";

  // First, load any auxiliary functions (e.g. device-specific functions).
  for (const std::string& filename : aux_function_filenames)
    {
    std::string full_filename = "oneway/" + filename;
    result += read_file(full_filename);
    }

  // Now, load each file for each kernel.
  for (const std::string& kernel_name : oneway_kernel_id::get_names())
    {
    std::string filename = "oneway/" + kernel_name + ".cu";
    result += read_file(filename);
    }

  return result;
  }




inline
std::string
get_cuda_oneway_real_kernel_src()
  {
  std::string result = "";

  // Load each file for each kernel.
  for (const std::string& kernel_name : oneway_real_kernel_id::get_names())
    {
    std::string filename = "oneway_real/" + kernel_name + ".cu";
    result += read_file(filename);
    }

  return result;
  }



inline
std::string
get_cuda_oneway_integral_kernel_src()
  {
  std::string result = "";

  std::vector<std::string> aux_function_filenames = {
      "and_warp_reduce.cu",
      "or_warp_reduce.cu"
  };

  // First, load any auxiliary functions (e.g. device-specific functions).
  for (const std::string& filename : aux_function_filenames)
    {
    std::string full_filename = "oneway_integral/" + filename;
    result += read_file(full_filename);
    }

  // Now, load each file for each kernel.
  for (const std::string& kernel_name : oneway_integral_kernel_id::get_names())
    {
    std::string filename = "oneway_integral/" + kernel_name + ".cu";
    result += read_file(filename);
    }

  return result;
  }



inline
std::string
get_cuda_twoway_kernel_src()
  {
  // NOTE: kernel names must match the list in the kernel_id struct

  // TODO: adapt so that auxiliary terms have type eT1 not eT2
  // current dogma will be: eT2(x + val) *not* eT2(x) + eT2(val)
  // however, we should also add the overload eT2(x) + val for those situations
  // the operation, I guess, would look like Op<out_eT, eOp<...>, op_conv_to>
  // and we could add an auxiliary out_eT to Op that's 0 by default, but I guess we need bools to indicate usage?
  // they would need to be added to eOp too
  // need to look through Op to see if it's needed there

  std::vector<std::string> aux_function_filenames = {
      "dot_warp_reduce.cu"
  };

  std::string result = "";

  // First, load any auxiliary functions (e.g. device-specific functions).
  for (const std::string& filename : aux_function_filenames)
    {
    std::string full_filename = "twoway/" + filename;
    result += read_file(full_filename);
    }

  // Now, load each file for each kernel.
  for (const std::string& kernel_name : twoway_kernel_id::get_names())
    {
    std::string filename = "twoway/" + kernel_name + ".cu";
    result += read_file(filename);
    }

  return result;
  }



inline
std::string
get_cuda_threeway_kernel_src()
  {
  std::string result = "";

  // Load each file for each kernel.
  for (const std::string& kernel_name : threeway_kernel_id::get_names())
    {
    std::string filename = "threeway/" + kernel_name + ".cu";
    result += read_file(filename);
    }

  return result;
  }
