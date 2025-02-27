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



#define COOT_STRINGIFY(x) COOT_STRINGIFY_2(x)
#define COOT_STRINGIFY_2(x) #x



struct kernel_src
  {
  static inline const std::string&  get_src_preamble(const bool has_float64, const bool has_subgroups, const size_t subgroup_size, const bool must_synchronise_subgroups, const bool need_subgroup_extension);

  static inline const std::string&  get_zeroway_source();
  static inline       std::string  init_zeroway_source();

  static inline const std::string&  get_oneway_source();
  static inline       std::string  init_oneway_source();

  static inline const std::string&  get_oneway_real_source();
  static inline       std::string  init_oneway_real_source();

  static inline const std::string&  get_oneway_integral_source();
  static inline       std::string  init_oneway_integral_source();

  static inline const std::string&  get_twoway_source();
  static inline       std::string  init_twoway_source();

  static inline const std::string&  get_threeway_source();
  static inline       std::string  init_threeway_source();

  static inline const std::string&  get_magma_real_source();
  static inline       std::string  init_magma_real_source();

  static inline const std::string&  get_src_epilogue();
  };



inline
const std::string&
kernel_src::get_src_preamble(const bool has_float64, const bool has_subgroups, const size_t subgroup_size, const bool must_synchronise_subgroups, const bool need_subgroup_extension)
  {
  char u32_max[32];
  char u64_max[32];
  snprintf(u32_max, 32, "%llu", (unsigned long long) std::numeric_limits<u32>::max());
  snprintf(u64_max, 32, "%llu", (unsigned long long) std::numeric_limits<u64>::max());

  char s32_min[32];
  char s64_min[32];
  snprintf(s32_min, 32, "%llu", (unsigned long long) std::numeric_limits<s32>::min());
  snprintf(s64_min, 32, "%llu", (unsigned long long) std::numeric_limits<s64>::min());

  char s32_max[32];
  char s64_max[32];
  snprintf(s32_max, 32, "%llu", (unsigned long long) std::numeric_limits<s32>::max());
  snprintf(s64_max, 32, "%llu", (unsigned long long) std::numeric_limits<s64>::max());

  char subgroup_size_str[32];
  snprintf(subgroup_size_str, 32, "%zu", subgroup_size);

  static const std::string source = \

  "#ifdef cl_khr_pragma_unroll \n"
  "#pragma OPENCL EXTENSION cl_khr_pragma_unroll : enable \n"
  "#endif \n"
  "#ifdef cl_amd_pragma_unroll \n"
  "#pragma OPENCL EXTENSION cl_amd_pragma_unroll : enable \n"
  "#endif \n"
  "#ifdef cl_nv_pragma_unroll \n"
  "#pragma OPENCL EXTENSION cl_nv_pragma_unroll : enable \n"
  "#endif \n"
  "#ifdef cl_intel_pragma_unroll \n"
  "#pragma OPENCL EXTENSION cl_intel_pragma_unroll : enable \n"
  "#endif \n" +
  ((need_subgroup_extension) ?
      std::string("#pragma OPENCL EXTENSION cl_khr_subgroups : enable \n") :
      std::string("")) +
  "\n"
  "#define COOT_FN2(ARG1,ARG2)  ARG1 ## ARG2 \n"
  "#define COOT_FN(ARG1,ARG2) COOT_FN2(ARG1,ARG2) \n"
  "\n"
  "#define COOT_FN_3_2(ARG1,ARG2,ARG3) ARG1 ## ARG2 ## ARG3 \n"
  "#define COOT_FN_3(ARG1,ARG2,ARG3) COOT_FN_3_2(ARG1,ARG2,ARG3) \n" +
  ((has_float64) ?
      std::string("#define COOT_HAS_FP64 \n") :
      std::string("")) +
  "\n"
  // Define cx_float and cx_double in the same way that clMAGMA does.
  "typedef float2 cx_float; \n" +
  ((has_float64) ? "typedef double2 cx_double; \n" : "") +
  "\n"
  // Utility functions to return the correct min/max value for a given type.
  "inline uint coot_type_min_uint() { return 0; } \n"
  "inline ulong coot_type_min_ulong() { return 0; } \n"
  "inline uint coot_type_max_uint() { return " + std::string(u32_max) + "; } \n"
  "inline ulong coot_type_max_ulong() { return " + std::string(u64_max) + "; } \n"
  "\n"
  "inline int coot_type_min_int() { return " + std::string(s32_min) + "; } \n"
  "inline long coot_type_min_long() { return " + std::string(s64_min) + "; } \n"
  "inline int coot_type_max_int() { return " + std::string(s32_max) + "; } \n"
  "inline long coot_type_max_long() { return " + std::string(s64_max) + "; } \n"
  "\n"
  "inline float coot_type_min_float() { return FLT_MIN; } \n"
  "inline float coot_type_max_float() { return FLT_MAX; } \n" +
  ((has_float64) ?
      std::string("inline double coot_type_min_double() { return DBL_MIN; } \n"
                  "inline double coot_type_max_double() { return DBL_MAX; } \n") :
      std::string("")) +
  "\n"
  "inline bool coot_is_fp_uint() { return false; } \n"
  "inline bool coot_is_fp_int() { return false; } \n"
  "inline bool coot_is_fp_ulong() { return false; } \n"
  "inline bool coot_is_fp_long() { return false; } \n"
  "inline bool coot_is_fp_float() { return true; } \n" +
  ((has_float64) ?
      std::string("inline bool coot_is_fp_double() { return true; } \n") :
      std::string("")) +
  "\n"
  "inline bool coot_is_signed_uint() { return false; } \n"
  "inline bool coot_is_signed_int() { return true; } \n"
  "inline bool coot_is_signed_ulong() { return false; } \n"
  "inline bool coot_is_signed_long() { return true; } \n"
  "inline bool coot_is_signed_float() { return true; } \n" +
  ((has_float64) ?
      std::string("inline bool coot_is_signed_double() { return true; } \n") :
      std::string("")) +
  "\n"
  "inline bool coot_isnan_uint(const uint x)     { return false;    } \n"
  "inline bool coot_isnan_int(const int x)       { return false;    } \n"
  "inline bool coot_isnan_ulong(const ulong x)   { return false;    } \n"
  "inline bool coot_isnan_long(const long x)     { return false;    } \n"
  "inline bool coot_isnan_float(const float x)   { return isnan(x); } \n" +
  ((has_float64) ?
      std::string("inline bool coot_isnan_double(const double x) { return isnan(x); } \n") :
      std::string("")) +
  "\n"
  // MAGMA-specific macros.
  "#define MAGMABLAS_BLK_X " COOT_STRINGIFY(MAGMABLAS_BLK_X) " \n"
  "#define MAGMABLAS_BLK_Y " COOT_STRINGIFY(MAGMABLAS_BLK_Y) " \n"
  "#define MAGMABLAS_TRANS_NX " COOT_STRINGIFY(MAGMABLAS_TRANS_NX) " \n"
  "#define MAGMABLAS_TRANS_NY " COOT_STRINGIFY(MAGMABLAS_TRANS_NY) " \n"
  "#define MAGMABLAS_TRANS_NB " COOT_STRINGIFY(MAGMABLAS_TRANS_NB) " \n"
  "#define MAGMABLAS_TRANS_INPLACE_NB " COOT_STRINGIFY(MAGMABLAS_TRANS_INPLACE_NB) " \n"
  "#define MAGMABLAS_LASWP_MAX_PIVOTS " COOT_STRINGIFY(MAGMABLAS_LASWP_MAX_PIVOTS) " \n"
  "#define MAGMABLAS_LASWP_NTHREADS " COOT_STRINGIFY(MAGMABLAS_LASWP_NTHREADS) " \n"
  "#define MAGMABLAS_LASCL_NB " COOT_STRINGIFY(MAGMABLAS_LASCL_NB) " \n"
  "#define MAGMABLAS_LASET_BAND_NB " COOT_STRINGIFY(MAGMABLAS_LASET_BAND_NB) " \n"
  "#define MAGMABLAS_LANSY_INF_BS " COOT_STRINGIFY(MAGMABLAS_LANSY_INF_BS) " \n"
  "#define MAGMABLAS_LANSY_MAX_BS " COOT_STRINGIFY(MAGMABLAS_LANSY_MAX_BS) " \n"
  "\n"
  "typedef struct \n"
  "  { \n"
  "  int npivots; \n"
  "  int ipiv[" COOT_STRINGIFY(MAGMABLAS_LASWP_MAX_PIVOTS) "]; \n"
  "  } magmablas_laswp_params_t; \n"
  "\n"
  // Sometimes we need to approximate Armadillo functionality that uses
  // double---but double may not be available.  So we do our best...
  "#define ARMA_FP_TYPE " + ((has_float64) ? std::string("double") : std::string("float")) + " \n" +
  // Utility function for subgroup barriers; this is needed in case subgroups
  // are not available.
  ((has_subgroups) ?
      ((must_synchronise_subgroups) ? std::string("#define SUBGROUP_BARRIER sub_group_barrier") : std::string("#define SUBGROUP_BARRIER(x) ")) :
      std::string("#define SUBGROUP_BARRIER barrier")) + " \n"
  "#define SUBGROUP_SIZE " + std::string(subgroup_size_str) + " \n"
  "#define SUBGROUP_SIZE_NAME " + ((has_subgroups && subgroup_size < 128) ? std::string(subgroup_size_str) : "other") +
  "\n"
  // Forward declarations that may be needed.
  "void u32_or_subgroup_reduce_other(__local volatile uint* data, UWORD tid); \n"
  "void u32_or_subgroup_reduce_8(__local volatile uint* data, UWORD tid); \n"
  "void u32_or_subgroup_reduce_16(__local volatile uint* data, UWORD tid); \n"
  "void u32_or_subgroup_reduce_32(__local volatile uint* data, UWORD tid); \n"
  "void u32_or_subgroup_reduce_64(__local volatile uint* data, UWORD tid); \n"
  "void u32_or_subgroup_reduce_128(__local volatile uint* data, UWORD tid); \n"
  "\n"
  ;

  return source;
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
const std::string&
kernel_src::get_zeroway_source()
  {
  static const std::string source = init_zeroway_source();

  return source;
  }



inline
std::string
kernel_src::init_zeroway_source()
  {
  // NOTE: kernel names must match the list in the kernel_id struct

  std::vector<std::string> aux_function_filenames = {
      "xorwow_rng.cl",
      "philox_rng.cl",
      "absdiff.cl",
      "var_philox.cl",
      "conj.cl"
  };

  std::string source = "";

  // First, load any auxiliary functions.
  for (const std::string& filename : aux_function_filenames)
    {
    std::string full_filename = "zeroway/" + filename;
    source += read_file(full_filename);
    }

  // Now, load each file for each kernel.
  for (const std::string& kernel_name : zeroway_kernel_id::get_names())
    {
    std::string filename = "zeroway/" + kernel_name + ".cl";
    source += read_file(filename);
    }

  return source;
  }



inline
const std::string&
kernel_src::get_oneway_source()
  {
  static const std::string source = init_oneway_source();

  return source;
  }



// TODO: inplace_set_scalar() could be replaced with explicit call to clEnqueueFillBuffer()
// present in OpenCL 1.2: http://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clEnqueueFillBuffer.html

// TODO: need submat analogues of all functions

// TODO: need specialised handling for cx_float and cx_double
// for example (cx_double * cx_double) is not simply (double2 * double2)


inline
std::string
kernel_src::init_oneway_source()
  {
  // NOTE: kernel names must match the list in the kernel_id struct

  std::vector<std::string> aux_function_filenames = {
      "accu_subgroup_reduce.cl",
      "min_subgroup_reduce.cl",
      "max_subgroup_reduce.cl",
      "index_min_subgroup_reduce.cl",
      "index_max_subgroup_reduce.cl",
      "prod_subgroup_reduce.cl"
  };

  std::string source = "";

  // First, load any auxiliary functions (e.g. device-specific functions).
  for (const std::string& filename : aux_function_filenames)
    {
    std::string full_filename = "oneway/" + filename;
    source += read_file(full_filename);
    }

  // Now, load each file for each kernel.
  for (const std::string& kernel_name : oneway_kernel_id::get_names())
    {
    std::string filename = "oneway/" + kernel_name + ".cl";
    source += read_file(filename);
    }

  return source;
  }



inline
const std::string&
kernel_src::get_oneway_real_source()
  {
  static const std::string source = init_oneway_real_source();

  return source;
  }



inline
std::string
kernel_src::init_oneway_real_source()
  {
  // NOTE: kernel names must match the list in the kernel_id struct

  std::string source = "";

  // Load each file for each kernel.
  for (const std::string& kernel_name : oneway_real_kernel_id::get_names())
    {
    std::string filename = "oneway_real/" + kernel_name + ".cl";
    source += read_file(filename);
    }

  return source;
  }



inline
const std::string&
kernel_src::get_oneway_integral_source()
  {
  static const std::string source = init_oneway_integral_source();

  return source;
  }



inline
std::string
kernel_src::init_oneway_integral_source()
  {
  // NOTE: kernel names must match the list in the kernel_id struct

  std::vector<std::string> aux_function_filenames = {
      "and_subgroup_reduce.cl",
      "or_subgroup_reduce.cl"
  };

  std::string source = "";

  // First, load any auxiliary functions (e.g. device-specific functions).
  for (const std::string& filename : aux_function_filenames)
    {
    std::string full_filename = "oneway_integral/" + filename;
    source += read_file(full_filename);
    }

  // Load each file for each kernel.
  for (const std::string& kernel_name : oneway_integral_kernel_id::get_names())
    {
    std::string filename = "oneway_integral/" + kernel_name + ".cl";
    source += read_file(filename);
    }

  return source;
  }



inline
const std::string&
kernel_src::get_twoway_source()
  {
  static const std::string source = init_twoway_source();

  return source;
  }



inline
std::string
kernel_src::init_twoway_source()
  {
  // NOTE: kernel names must match the list in the kernel_id struct

  std::vector<std::string> aux_function_filenames = {
      "dot_subgroup_reduce.cl"
  };

  std::string source = "";

  // First, load any auxiliary functions (e.g. device-specific functions).
  for (const std::string& filename : aux_function_filenames)
    {
    std::string full_filename = "twoway/" + filename;
    source += read_file(full_filename);
    }

  // Now, load each file for each kernel.
  for (const std::string& kernel_name : twoway_kernel_id::get_names())
    {
    std::string filename = "twoway/" + kernel_name + ".cl";
    source += read_file(filename);
    }

  return source;
  }



inline
const std::string&
kernel_src::get_threeway_source()
  {
  static const std::string source = init_threeway_source();

  return source;
  }



inline
std::string
kernel_src::init_threeway_source()
  {
  // NOTE: kernel names must match the list in the kernel_id struct

  std::string source = "";

  // Load each file for each kernel.
  for (const std::string& kernel_name : threeway_kernel_id::get_names())
    {
    std::string filename = "threeway/" + kernel_name + ".cl";
    source += read_file(filename);
    }

  return source;
  }



inline
const std::string&
kernel_src::get_magma_real_source()
  {
  static const std::string source = init_magma_real_source();

  return source;
  }



inline
std::string
kernel_src::init_magma_real_source()
  {
  // NOTE: kernel names must match the list in the kernel_id struct

  std::string source = "";

  // Load each file for each kernel.
  for (const std::string& kernel_name : magma_real_kernel_id::get_names())
    {
    std::string filename = "magma_real/" + kernel_name + ".cl";
    source += read_file(filename);
    }

  return source;
  }



inline
const std::string&
kernel_src::get_src_epilogue()
  {
  static const std::string source = "";

  return source;
  }
