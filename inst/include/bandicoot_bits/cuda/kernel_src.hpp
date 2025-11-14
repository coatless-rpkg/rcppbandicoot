// Copyright 2019-2025 Ryan Curtin (http://www.ratml.org/)
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



struct kernel_src
  {
  static inline       std::string   init_src_preamble(const bool has_fp16);

  static inline       std::string   get_zeroway_source(const zeroway_kernel_id::enum_id num);

  template<typename eT>
  static inline       std::string   get_oneway_defines();
  static inline       std::string   get_oneway_source(const oneway_kernel_id::enum_id num);

  static inline       std::string   init_oneway_real_aux_functions();
  static inline const std::string&  get_oneway_real_aux_functions();
  static inline       std::string   get_oneway_real_source(const oneway_real_kernel_id::enum_id num);

  static inline       std::string   init_oneway_integral_aux_functions();
  static inline const std::string&  get_oneway_integral_aux_functions();
  static inline       std::string   get_oneway_integral_source(const oneway_integral_kernel_id::enum_id num);

  static inline       std::string   init_twoway_aux_functions();
  static inline const std::string&  get_twoway_aux_functions();
  template<typename eT1, typename eT2>
  static inline       std::string   get_twoway_defines();
  static inline       std::string   get_twoway_source(const twoway_kernel_id::enum_id num);

  template<typename eT1, typename eT2, typename eT3>
  static inline       std::string   get_threeway_defines();
  static inline       std::string   get_threeway_source(const threeway_kernel_id::enum_id num);
  };



// utility functions for compiled-on-the-fly CUDA kernels
inline
std::string
read_file(const std::string& filename)
  {
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
kernel_src::init_src_preamble(const bool has_fp16)
  {
  char u8_max[32];
  char u16_max[32];
  char u32_max[32];
  char u64_max[32];
  snprintf(u8_max,  32, "%hu",     (unsigned int) Datum<u8>::max);
  snprintf(u16_max, 32, "%hu",     (unsigned int) Datum<u16>::max);
  snprintf(u32_max, 32, "%uu",     (unsigned int) Datum<u32>::max);
  snprintf(u64_max, 32, "%llullu", (unsigned long long) Datum<u64>::max);

  char s8_min[32];
  char s16_min[32];
  char s32_min[32];
  char s64_min[32];
  snprintf(s8_min,  32, "%hd",    (int) std::numeric_limits<s8>::lowest());
  snprintf(s16_min, 32, "%hd",    (int) std::numeric_limits<s16>::lowest());
  snprintf(s32_min, 32, "%d",     (int) std::numeric_limits<s32>::lowest());
  snprintf(s64_min, 32, "%lldll", (long long) std::numeric_limits<s64>::lowest());

  char s8_max[32];
  char s16_max[32];
  char s32_max[32];
  char s64_max[32];
  snprintf(s8_max,  32, "%hd",    (int) Datum<s8>::max);
  snprintf(s16_max, 32, "%hd",    (int) Datum<s16>::max);
  snprintf(s32_max, 32, "%d",     (int) Datum<s32>::max);
  snprintf(s64_max, 32, "%lldll", (long long) Datum<s64>::max);

  std::string source = \

  ((has_fp16) ?
      std::string("#include <cuda_fp16.h> \n"
                  "#define COOT_HAVE_FP16 \n") :
      std::string("")) +
  "\n"
  "#define COOT_PI " STR2(M_PI) "\n"
  "\n"
  "#define COOT_S8_MIN "  + std::string(s8_min)  + " \n"
  "#define COOT_S16_MIN " + std::string(s16_min) + " \n"
  "#define COOT_S32_MIN " + std::string(s32_min) + " \n"
  "#define COOT_S64_MIN " + std::string(s64_min) + " \n"
  "\n"
  "#define COOT_U8_MAX "  + std::string(u8_max)  + " \n"
  "#define COOT_U16_MAX " + std::string(u16_max) + " \n"
  "#define COOT_U32_MAX " + std::string(u32_max) + " \n"
  "#define COOT_U64_MAX " + std::string(u64_max) + " \n"
  "#define COOT_S8_MAX "  + std::string(s8_max)  + " \n"
  "#define COOT_S16_MAX " + std::string(s16_max) + " \n"
  "#define COOT_S32_MAX " + std::string(s32_max) + " \n"
  "#define COOT_S64_MAX " + std::string(s64_max) + " \n"
  "\n"
  "#ifdef COOT_HAVE_FP16 \n"
  "  #define HALF_MIN " STR2(CUDART_MIN_DENORM_FP16) " \n"
  "  #define HALF_MAX " STR2(CUDART_MAX_NORMAL_FP16) " \n"
  "#endif \n"
  "#define FLT_MIN " STR2(FLT_MIN) " \n"
  "#define FLT_MAX " STR2(FLT_MAX) " \n"
  "#define DBL_MIN " STR2(DBL_MIN) " \n"
  "#define DBL_MAX " STR2(DBL_MAX) " \n"
  "#define SIZE_MAX " STR2(SIZE_MAX) " \n";

  source += read_file("defs/cuda_prelims.cu");

  return source;
  }



inline
std::string
kernel_src::get_zeroway_source(const zeroway_kernel_id::enum_id num)
  {
  const std::string kernel_name = zeroway_kernel_id::get_names()[num];
  const std::string filename = "zeroway/" + kernel_name + ".cu";

  std::string source =
      "extern \"C\" {\n"
      "\n"
      "#define PREFIX  \n"
      "\n";

  if (zeroway_kernel_id::get_deps().count(num) > 0)
    {
    const std::vector<std::string>& deps = zeroway_kernel_id::get_deps().at(num);
    for (const std::string& dep_f : deps)
      {
      source += read_file("deps/" + dep_f + ".cu");
      }
    }

  source += read_file(filename) +
      "\n"
      "}\n";

  return source;
  }



template<typename eT>
inline
std::string
kernel_src::get_oneway_defines()
  {
  typedef typename promote_type<eT, float>::result fp_eT;
  typedef typename uint_type<eT>::result           uint_eT;

  std::string source = \
      "#define PREFIX " + type_prefix<eT>() + "_ \n" +
      "#define eT1 " + type_to_dev_string::map<eT>() + " \n" +
      "#define fp_eT1 " + type_to_dev_string::map<fp_eT>() + " \n" +
      "#define uint_eT1 " + type_to_dev_string::map<uint_eT>() + " \n" +
      "#define ET1_ABS " + type_to_dev_string::abs_func<eT>() + " \n" +
      "#define TO_ET1(x) coot_to_" + type_to_dev_string::map<eT>() + "(x) \n" +
      "#define TO_FP_ET1(x) coot_to_" + type_to_dev_string::map<fp_eT>() + "(x) \n" +
      "#define TO_UINT_ET1(x) coot_to_" + type_to_dev_string::map<uint_eT>() + "(x) \n";

  source += read_file("defs/" + type_prefix<eT>() + "_defs.cu");
  if (is_same_type<eT, fp_eT>::no)
    source += read_file("defs/" + type_prefix<fp_eT>() + "_defs.cu");
  if (is_same_type<eT, uint_eT>::no)
    source += read_file("defs/" + type_prefix<uint_eT>() + "_defs.cu");

  return source;
  }



inline
std::string
kernel_src::get_oneway_source(const oneway_kernel_id::enum_id num)
  {
  const std::string kernel_name = oneway_kernel_id::get_names()[num];
  const std::string filename = "oneway/" + kernel_name + ".cu";

  std::string source = "extern \"C\" {\n";

  if (oneway_kernel_id::get_deps().count(num) > 0)
    {
    const std::vector<std::string>& deps = oneway_kernel_id::get_deps().at(num);
    for (const std::string& dep_f : deps)
      {
      source += read_file("deps/" + dep_f + ".cu");
      }
    }

  source += read_file(filename) +
      "\n"
      "}\n";

  return source;
  }



inline
std::string
kernel_src::get_oneway_integral_source(const oneway_integral_kernel_id::enum_id num)
  {
  const std::string kernel_name = oneway_integral_kernel_id::get_names()[num];
  const std::string filename = "oneway_integral/" + kernel_name + ".cu";

  std::string source = "extern \"C\" {\n";

  if (oneway_integral_kernel_id::get_deps().count(num) > 0)
    {
    const std::vector<std::string>& deps = oneway_integral_kernel_id::get_deps().at(num);
    for (const std::string& dep_f : deps)
      {
      source += read_file("deps/" + dep_f + ".cu");
      }
    }

  source += read_file(filename) +
      "\n"
      "}\n";

  return source;
  }



inline
std::string
kernel_src::get_oneway_real_source(const oneway_real_kernel_id::enum_id num)
  {
  const std::string kernel_name = oneway_real_kernel_id::get_names()[num];
  const std::string filename = "oneway_real/" + kernel_name + ".cu";

  std::string source = "extern \"C\" {\n";

  if (oneway_real_kernel_id::get_deps().count(num) > 0)
    {
    const std::vector<std::string>& deps = oneway_real_kernel_id::get_deps().at(num);
    for (const std::string& dep_f : deps)
      {
      source += read_file("deps/" + dep_f + ".cu");
      }
    }

  source += read_file(filename) +
      "\n"
      "}\n";

  return source;
  }



template<typename eT1, typename eT2>
inline
std::string
kernel_src::get_twoway_defines()
  {
  typedef typename promote_type<eT1, float>::result fp_eT1;
  typedef typename uint_type<eT1>::result           uint_eT1;
  typedef typename promote_type<eT2, float>::result fp_eT2;
  typedef typename uint_type<eT2>::result           uint_eT2;
  typedef typename promote_type<eT1, eT2>::result   twoway_promoted_eT;

  std::string source =
      "#define PREFIX " + type_prefix<eT1>() + "_" + type_prefix<eT2>() + "_\n" +
      "#define eT1 " + type_to_dev_string::map<eT1>() + "\n" +
      "#define fp_eT1 " + type_to_dev_string::map<fp_eT1>() + "\n" +
      "#define uint_eT1 " + type_to_dev_string::map<uint_eT1>() + "\n" +
      "#define ET1_ABS " + type_to_dev_string::abs_func<eT1>() + "\n" +
      "#define TO_ET1(x) coot_to_" + type_to_dev_string::map<eT1>() + "(x) \n" +
      "#define TO_FP_ET1(x) coot_to_" + type_to_dev_string::map<fp_eT1>() + "(x) \n" +
      "#define TO_UINT_ET1(x) coot_to_" + type_to_dev_string::map<uint_eT1>() + "(x) \n" +
      "#define eT2 " + type_to_dev_string::map<eT2>() + "\n" +
      "#define fp_eT2 " + type_to_dev_string::map<fp_eT2>() + "\n" +
      "#define uint_eT2 " + type_to_dev_string::map<uint_eT2>() + "\n" +
      "#define TO_ET2(x) coot_to_" + type_to_dev_string::map<eT2>() + "(x) \n" +
      "#define TO_FP_ET2(x) coot_to_" + type_to_dev_string::map<fp_eT2>() + "(x) \n" +
      "#define TO_UINT_ET2(x) coot_to_" + type_to_dev_string::map<uint_eT2>() + "(x) \n" +
      "#define twoway_promoted_eT " + type_to_dev_string::map<twoway_promoted_eT>() + "\n" +
      "#define TO_TWOWAY_PROMOTED_ET(x) coot_to_" + type_to_dev_string::map<twoway_promoted_eT>() + "(x) \n";

  source += read_file("defs/" + type_prefix<eT1>() + "_defs.cu");

  if (is_same_type<eT1, fp_eT1>::no)
    source += read_file("defs/" + type_prefix<fp_eT1>() + "_defs.cu");

  if (is_same_type<eT1, uint_eT1>::no)
    source += read_file("defs/" + type_prefix<uint_eT1>() + "_defs.cu");

  if (is_same_type<eT2, eT1>::no &&
      is_same_type<eT2, fp_eT1>::no &&
      is_same_type<eT2, uint_eT1>::no)
    source += read_file("defs/" + type_prefix<eT2>() + "_defs.cu");

  if (is_same_type<eT2, fp_eT2>::no &&
      is_same_type<eT1, fp_eT2>::no &&
      is_same_type<fp_eT1, fp_eT2>::no)
    source += read_file("defs/" + type_prefix<fp_eT2>() + "_defs.cu");

  if (is_same_type<eT2, uint_eT2>::no &&
      is_same_type<eT1, uint_eT2>::no &&
      is_same_type<uint_eT1, uint_eT2>::no)
    source += read_file("defs/" + type_prefix<uint_eT2>() + "_defs.cu");

  if (is_same_type<eT1, twoway_promoted_eT>::no &&
      is_same_type<eT2, twoway_promoted_eT>::no &&
      is_same_type<fp_eT1, twoway_promoted_eT>::no &&
      is_same_type<fp_eT2, twoway_promoted_eT>::no &&
      is_same_type<uint_eT1, twoway_promoted_eT>::no &&
      is_same_type<uint_eT2, twoway_promoted_eT>::no)
    source += read_file("defs/" + type_prefix<twoway_promoted_eT>() + "_defs.cu");

  return source;
  }



inline
std::string
kernel_src::get_twoway_source(const twoway_kernel_id::enum_id num)
  {
  const std::string kernel_name = twoway_kernel_id::get_names()[num];
  const std::string filename = "twoway/" + kernel_name + ".cu";

  std::string source = "extern \"C\" {\n";

  if (twoway_kernel_id::get_deps().count(num) > 0)
    {
    const std::vector<std::string>& deps = twoway_kernel_id::get_deps().at(num);
    for (const std::string& dep_f : deps)
      {
      source += read_file("deps/" + dep_f + ".cu");
      }
    }

  source += read_file(filename) +
      "\n"
      "}\n";

  return source;
  }



template<typename eT1, typename eT2, typename eT3>
inline
std::string
kernel_src::get_threeway_defines()
  {
  typedef typename promote_type<eT1, float>::result fp_eT1;
  typedef typename uint_type<eT1>::result           uint_eT1;
  typedef typename promote_type<eT2, float>::result fp_eT2;
  typedef typename uint_type<eT2>::result           uint_eT2;
  typedef typename promote_type<eT3, float>::result fp_eT3;
  typedef typename uint_type<eT3>::result           uint_eT3;

  typedef typename promote_type<eT1, eT2>::result                twoway_promoted_eT;
  typedef typename promote_type<twoway_promoted_eT, eT3>::result threeway_promoted_eT;

  std::string source =
      "#define PREFIX " + type_prefix<eT1>() + "_" + type_prefix<eT2>() + "_" + type_prefix<eT3>() + "_\n" +
      "#define eT1 " + type_to_dev_string::map<eT1>() + "\n" +
      "#define fp_eT1 " + type_to_dev_string::map<fp_eT1>() + "\n" +
      "#define uint_eT1 " + type_to_dev_string::map<uint_eT1>() + "\n" +
      "#define ET1_ABS " + type_to_dev_string::abs_func<eT1>() + "\n" +
      "#define TO_ET1(x) coot_to_" + type_to_dev_string::map<eT1>() + "(x) \n" +
      "#define TO_FP_ET1(x) coot_to_" + type_to_dev_string::map<fp_eT1>() + "(x) \n" +
      "#define TO_UINT_ET1(x) coot_to_" + type_to_dev_string::map<uint_eT1>() + "(x) \n" +
      "#define eT2 " + type_to_dev_string::map<eT2>() + "\n" +
      "#define fp_eT2 " + type_to_dev_string::map<fp_eT2>() + "\n" +
      "#define uint_eT2 " + type_to_dev_string::map<uint_eT2>() + "\n" +
      "#define TO_ET2(x) coot_to_" + type_to_dev_string::map<eT2>() + "(x) \n" +
      "#define TO_FP_ET2(x) coot_to_" + type_to_dev_string::map<fp_eT2>() + "(x) \n" +
      "#define TO_UINT_ET2(x) coot_to_" + type_to_dev_string::map<uint_eT2>() + "(x) \n" +
      "#define twoway_promoted_eT " + type_to_dev_string::map<twoway_promoted_eT>() + "\n" +
      "#define TO_TWOWAY_PROMOTED_ET(x) coot_to_" + type_to_dev_string::map<twoway_promoted_eT>() + "(x) \n" +
      "#define eT3 " + type_to_dev_string::map<eT3>() + "\n" +
      "#define fp_eT3 " + type_to_dev_string::map<fp_eT3>() + "\n" +
      "#define uint_eT3 " + type_to_dev_string::map<uint_eT3>() + "\n" +
      "#define TO_ET3(x) coot_to_" + type_to_dev_string::map<eT3>() + "(x) \n" +
      "#define TO_FP_ET3(x) coot_to_" + type_to_dev_string::map<fp_eT3>() + "(x) \n" +
      "#define TO_UINT_ET3(x) coot_to_" + type_to_dev_string::map<uint_eT3>() + "(x) \n" +
      "#define threeway_promoted_eT " + type_to_dev_string::map<threeway_promoted_eT>() + "\n" +
      "#define TO_THREEWAY_PROMOTED_ET(x) coot_to_" + type_to_dev_string::map<threeway_promoted_eT>() + "(x) \n";

  source += read_file("defs/" + type_prefix<eT1>() + "_defs.cu");

  if (is_same_type<eT1, fp_eT1>::no)
    source += read_file("defs/" + type_prefix<fp_eT1>() + "_defs.cu");

  if (is_same_type<eT1, uint_eT1>::no)
    source += read_file("defs/" + type_prefix<uint_eT1>() + "_defs.cu");

  if (is_same_type<eT2, eT1>::no &&
      is_same_type<eT2, fp_eT1>::no &&
      is_same_type<eT2, uint_eT1>::no)
    source += read_file("defs/" + type_prefix<eT2>() + "_defs.cu");

  if (is_same_type<eT2, fp_eT2>::no &&
      is_same_type<eT1, fp_eT2>::no &&
      is_same_type<fp_eT1, fp_eT2>::no)
    source += read_file("defs/" + type_prefix<fp_eT2>() + "_defs.cu");

  if (is_same_type<eT2, uint_eT2>::no &&
      is_same_type<eT1, uint_eT2>::no &&
      is_same_type<uint_eT1, uint_eT2>::no)
    source += read_file("defs/" + type_prefix<uint_eT2>() + "_defs.cu");

  if (is_same_type<eT1, twoway_promoted_eT>::no &&
      is_same_type<eT2, twoway_promoted_eT>::no &&
      is_same_type<fp_eT1, twoway_promoted_eT>::no &&
      is_same_type<fp_eT2, twoway_promoted_eT>::no &&
      is_same_type<uint_eT1, twoway_promoted_eT>::no &&
      is_same_type<uint_eT2, twoway_promoted_eT>::no)
    source += read_file("defs/" + type_prefix<twoway_promoted_eT>() + "_defs.cu");

  if (is_same_type<eT3, eT2>::no &&
      is_same_type<eT3, fp_eT2>::no &&
      is_same_type<eT3, uint_eT2>::no &&
      is_same_type<eT3, eT1>::no &&
      is_same_type<eT3, fp_eT1>::no &&
      is_same_type<eT3, uint_eT1>::no &&
      is_same_type<eT3, twoway_promoted_eT>::no)
    source += read_file("defs/" + type_prefix<eT3>() + "_defs.cu");

  if (is_same_type<eT3, fp_eT3>::no &&
      is_same_type<eT2, fp_eT3>::no &&
      is_same_type<eT1, fp_eT3>::no &&
      is_same_type<fp_eT2, fp_eT3>::no &&
      is_same_type<fp_eT1, fp_eT3>::no &&
      is_same_type<twoway_promoted_eT, fp_eT3>::no)
    source += read_file("defs/" + type_prefix<fp_eT3>() + "_defs.cu");

  if (is_same_type<eT3, uint_eT3>::no &&
      is_same_type<eT2, uint_eT3>::no &&
      is_same_type<eT1, uint_eT3>::no &&
      is_same_type<uint_eT2, uint_eT3>::no &&
      is_same_type<uint_eT1, uint_eT3>::no &&
      is_same_type<twoway_promoted_eT, uint_eT3>::no)
    source += read_file("defs/" + type_prefix<uint_eT3>() + "_defs.cu");

  if (is_same_type<eT1, threeway_promoted_eT>::no &&
      is_same_type<eT2, threeway_promoted_eT>::no &&
      is_same_type<eT3, threeway_promoted_eT>::no &&
      is_same_type<fp_eT1, threeway_promoted_eT>::no &&
      is_same_type<fp_eT2, threeway_promoted_eT>::no &&
      is_same_type<fp_eT3, threeway_promoted_eT>::no &&
      is_same_type<uint_eT1, threeway_promoted_eT>::no &&
      is_same_type<uint_eT2, threeway_promoted_eT>::no &&
      is_same_type<uint_eT3, threeway_promoted_eT>::no &&
      is_same_type<twoway_promoted_eT, threeway_promoted_eT>::no)
    source += read_file("defs/" + type_prefix<threeway_promoted_eT>() + "_defs.cu");

  return source;
  }



inline
std::string
kernel_src::get_threeway_source(const threeway_kernel_id::enum_id num)
  {
  const std::string kernel_name = threeway_kernel_id::get_names()[num];
  const std::string filename = "threeway/" + kernel_name + ".cu";

  std::string source = "extern \"C\" {\n";

  if (threeway_kernel_id::get_deps().count(num) > 0)
    {
    const std::vector<std::string>& deps = threeway_kernel_id::get_deps().at(num);
    for (const std::string& dep_f : deps)
      {
      source += read_file("deps/" + dep_f + ".cu");
      }
    }

  source += read_file(filename) +
      "\n"
      "}\n";

  return source;
  }



#undef STR2
#undef STR
