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
  static inline       std::string   init_src_preamble(const bool has_float64, const bool has_float16, const bool has_sizet64, const bool has_subgroups, const size_t subgroup_size, const bool must_synchronise_subgroups, const bool need_subgroup_extension);

  static inline       std::string   get_zeroway_source(const zeroway_kernel_id::enum_id num);

  template<typename eT>
  static inline       std::string   get_oneway_defines();
  static inline       std::string   get_oneway_source(const oneway_kernel_id::enum_id num);

  static inline       std::string   get_oneway_real_source(const oneway_real_kernel_id::enum_id num);

  static inline       std::string   get_oneway_integral_source(const oneway_integral_kernel_id::enum_id num);

  template<typename eT1, typename eT2>
  static inline       std::string   get_twoway_defines();
  static inline       std::string   get_twoway_source(const twoway_kernel_id::enum_id num);

  template<typename eT1, typename eT2, typename eT3>
  static inline       std::string   get_threeway_defines();
  static inline       std::string   get_threeway_source(const threeway_kernel_id::enum_id num);

  static inline       std::string   init_magma_defines();
  static inline const std::string&  get_magma_defines();
  static inline       std::string   get_magma_real_source(const magma_real_kernel_id::enum_id num);
  };



inline
std::string
read_file(const std::string& filename)
  {
  std::string full_filename;

  // Check if COOT_CL_KERNEL_PATH environment variable is set
  const char* kernel_path_env = std::getenv("COOT_CL_KERNEL_PATH");

  if (kernel_path_env != nullptr)
    {
    std::string kernel_path_str(kernel_path_env);
    if (!kernel_path_str.empty())
      {
      // Use the path from the environment variable
      full_filename = kernel_path_str + "/" + filename;
      }
    else
      {
      // Fall back to the original behavior using __FILE__
      const std::string this_file = __FILE__;
      // We need to strip the '_src.hpp' from __FILE__.
      full_filename = this_file.substr(0, this_file.size() - 8) + "s/" + filename;
      }
    }
  else
    {
    // Fall back to the original behavior using __FILE__
    const std::string this_file = __FILE__;
    // We need to strip the '_src.hpp' from __FILE__.
    full_filename = this_file.substr(0, this_file.size() - 8) + "s/" + filename;
    }

  std::ifstream f(full_filename);
  std::string file_contents = "";
  if (!f.is_open())
    {
    COOT_CERR_STREAM << "Failed to open " << full_filename << " (kernel source)!\n";
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
kernel_src::init_src_preamble(const bool has_float64, const bool has_float16, const bool has_sizet64, const bool has_subgroups, const size_t subgroup_size, const bool must_synchronise_subgroups, const bool need_subgroup_extension)
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

  char subgroup_size_str[32];
  snprintf(subgroup_size_str, 32, "%zu", subgroup_size);

  std::string source = \

  ((need_subgroup_extension) ?
      std::string("#pragma OPENCL EXTENSION cl_khr_subgroups : enable \n") :
      std::string("")) +
  "\n" +
  ((has_float64) ?
      std::string("#define COOT_HAVE_FP64 \n") :
      std::string("")) +
  ((has_float16) ?
      std::string("#pragma OPENCL EXTENSION cl_khr_fp16 : enable \n"
                  "#define COOT_HAVE_FP16 \n") :
      std::string("")) +
  "\n"
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
  "#define COOT_S64_MAX " + std::string(s64_max) + " \n" +

  ((has_sizet64) ?
      std::string("#define UWORD ulong \n"
                  "#define COOT_UWORD_MAX COOT_U64_MAX \n") :
      std::string("#define UWORD uint \n"
                  "#define COOT_UWORD_MAX COOT_U32_MAX \n")) +

  // Utility function for subgroup barriers; this is needed in case subgroups
  // are not available.
  ((has_subgroups) ?
      ((must_synchronise_subgroups) ? std::string("#define SUBGROUP_BARRIER sub_group_barrier") : std::string("#define SUBGROUP_BARRIER(x) ")) :
      std::string("#define SUBGROUP_BARRIER barrier")) + " \n"
  "#define SUBGROUP_SIZE " + std::string(subgroup_size_str) + " \n"
  "#define SUBGROUP_SIZE_NAME " + ((has_subgroups && subgroup_size < 128) ? std::string(subgroup_size_str) : "other") +
  "\n";

  source += read_file("defs/opencl_prelims.cl");

  return source;
  }



inline
std::string
kernel_src::get_zeroway_source(const zeroway_kernel_id::enum_id num)
  {
  const std::string kernel_name = zeroway_kernel_id::get_names()[num];
  const std::string filename = "zeroway/" + kernel_name + ".cl";

  std::string source;

  if (zeroway_kernel_id::get_deps().count(num) > 0)
    {
    const std::vector<std::string>& deps = zeroway_kernel_id::get_deps().at(num);
    for (const std::string& dep_f : deps)
      {
      source += read_file("deps/" + dep_f + ".cl");
      }
    }

  source += read_file(filename);
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
      "#define ET1_ABS " + type_to_dev_string::abs_func<eT>() + " \n";

  source += read_file("defs/" + type_prefix<eT>() + "_defs.cl");
  if (is_same_type<eT, fp_eT>::no)
    source += read_file("defs/" + type_prefix<fp_eT>() + "_defs.cl");
  if (is_same_type<eT, uint_eT>::no)
    source += read_file("defs/" + type_prefix<uint_eT>() + "_defs.cl");

  return source;
  }



inline
std::string
kernel_src::get_oneway_source(const oneway_kernel_id::enum_id num)
  {
  const std::string kernel_name = oneway_kernel_id::get_names()[num];
  const std::string filename = "oneway/" + kernel_name + ".cl";

  std::string source;

  if (oneway_kernel_id::get_deps().count(num) > 0)
    {
    const std::vector<std::string>& deps = oneway_kernel_id::get_deps().at(num);
    for (const std::string& dep_f : deps)
      {
      source += read_file("deps/" + dep_f + ".cl");
      }
    }

  source += read_file(filename);
  return source;
  }



inline
std::string
kernel_src::get_oneway_real_source(const oneway_real_kernel_id::enum_id num)
  {
  const std::string kernel_name = oneway_real_kernel_id::get_names()[num];
  const std::string filename = "oneway_real/" + kernel_name + ".cl";

  std::string source;

  if (oneway_real_kernel_id::get_deps().count(num) > 0)
    {
    const std::vector<std::string>& deps = oneway_real_kernel_id::get_deps().at(num);
    for (const std::string& dep_f : deps)
      {
      source += read_file("deps/" + dep_f + ".cl");
      }
    }

  source += read_file(filename);
  return source;
  }



inline
std::string
kernel_src::get_oneway_integral_source(const oneway_integral_kernel_id::enum_id num)
  {
  const std::string kernel_name = oneway_integral_kernel_id::get_names()[num];
  const std::string filename = "oneway_integral/" + kernel_name + ".cl";

  std::string source;

  if (oneway_integral_kernel_id::get_deps().count(num) > 0)
    {
    const std::vector<std::string>& deps = oneway_integral_kernel_id::get_deps().at(num);
    for (const std::string& dep_f : deps)
      {
      source += read_file("deps/" + dep_f + ".cl");
      }
    }

  source += read_file(filename);
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
      "#define eT2 " + type_to_dev_string::map<eT2>() + "\n" +
      "#define fp_eT2 " + type_to_dev_string::map<fp_eT2>() + "\n" +
      "#define uint_eT2 " + type_to_dev_string::map<uint_eT2>() + "\n" +
      "#define twoway_promoted_eT " + type_to_dev_string::map<twoway_promoted_eT>() + "\n";

  source += read_file("defs/" + type_prefix<eT1>() + "_defs.cl");

  if (is_same_type<eT1, fp_eT1>::no)
    source += read_file("defs/" + type_prefix<fp_eT1>() + "_defs.cl");

  if (is_same_type<eT1, uint_eT1>::no)
    source += read_file("defs/" + type_prefix<uint_eT1>() + "_defs.cl");

  if (is_same_type<eT2, eT1>::no &&
      is_same_type<eT2, fp_eT1>::no &&
      is_same_type<eT2, uint_eT1>::no)
    source += read_file("defs/" + type_prefix<eT2>() + "_defs.cl");

  if (is_same_type<eT2, fp_eT2>::no &&
      is_same_type<eT1, fp_eT2>::no &&
      is_same_type<fp_eT1, fp_eT2>::no)
    source += read_file("defs/" + type_prefix<fp_eT2>() + "_defs.cl");

  if (is_same_type<eT2, uint_eT2>::no &&
      is_same_type<eT1, uint_eT2>::no &&
      is_same_type<uint_eT1, uint_eT2>::no)
    source += read_file("defs/" + type_prefix<uint_eT2>() + "_defs.cl");

  if (is_same_type<eT1, twoway_promoted_eT>::no &&
      is_same_type<eT2, twoway_promoted_eT>::no &&
      is_same_type<fp_eT1, twoway_promoted_eT>::no &&
      is_same_type<fp_eT2, twoway_promoted_eT>::no &&
      is_same_type<uint_eT1, twoway_promoted_eT>::no &&
      is_same_type<uint_eT2, twoway_promoted_eT>::no)
    source += read_file("defs/" + type_prefix<twoway_promoted_eT>() + "_defs.cl");

  return source;
  }


inline
std::string
kernel_src::get_twoway_source(const twoway_kernel_id::enum_id num)
  {
  const std::string kernel_name = twoway_kernel_id::get_names()[num];
  const std::string filename = "twoway/" + kernel_name + ".cl";

  std::string source;

  if (twoway_kernel_id::get_deps().count(num) > 0)
    {
    const std::vector<std::string>& deps = twoway_kernel_id::get_deps().at(num);
    for (const std::string& dep_f : deps)
      {
      source += read_file("deps/" + dep_f + ".cl");
      }
    }

  source += read_file(filename);
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
      "#define eT2 " + type_to_dev_string::map<eT2>() + "\n" +
      "#define fp_eT2 " + type_to_dev_string::map<fp_eT2>() + "\n" +
      "#define uint_eT2 " + type_to_dev_string::map<uint_eT2>() + "\n" +
      "#define twoway_promoted_eT " + type_to_dev_string::map<twoway_promoted_eT>() + "\n" +
      "#define eT3 " + type_to_dev_string::map<eT3>() + "\n" +
      "#define fp_eT3 " + type_to_dev_string::map<fp_eT3>() + "\n" +
      "#define uint_eT3 " + type_to_dev_string::map<uint_eT3>() + "\n" +
      "#define threeway_promoted_eT " + type_to_dev_string::map<threeway_promoted_eT>() + "\n";

  source += read_file("defs/" + type_prefix<eT1>() + "_defs.cl");

  if (is_same_type<eT1, fp_eT1>::no)
    source += read_file("defs/" + type_prefix<fp_eT1>() + "_defs.cl");

  if (is_same_type<eT1, uint_eT1>::no)
    source += read_file("defs/" + type_prefix<uint_eT1>() + "_defs.cl");

  if (is_same_type<eT2, eT1>::no &&
      is_same_type<eT2, fp_eT1>::no &&
      is_same_type<eT2, uint_eT1>::no)
    source += read_file("defs/" + type_prefix<eT2>() + "_defs.cl");

  if (is_same_type<eT2, fp_eT2>::no &&
      is_same_type<eT1, fp_eT2>::no &&
      is_same_type<fp_eT1, fp_eT2>::no)
    source += read_file("defs/" + type_prefix<fp_eT2>() + "_defs.cl");

  if (is_same_type<eT2, uint_eT2>::no &&
      is_same_type<eT1, uint_eT2>::no &&
      is_same_type<uint_eT1, uint_eT2>::no)
    source += read_file("defs/" + type_prefix<uint_eT2>() + "_defs.cl");

  if (is_same_type<eT1, twoway_promoted_eT>::no &&
      is_same_type<eT2, twoway_promoted_eT>::no &&
      is_same_type<fp_eT1, twoway_promoted_eT>::no &&
      is_same_type<fp_eT2, twoway_promoted_eT>::no &&
      is_same_type<uint_eT1, twoway_promoted_eT>::no &&
      is_same_type<uint_eT2, twoway_promoted_eT>::no)
    source += read_file("defs/" + type_prefix<twoway_promoted_eT>() + "_defs.cl");

  if (is_same_type<eT3, eT2>::no &&
      is_same_type<eT3, fp_eT2>::no &&
      is_same_type<eT3, uint_eT2>::no &&
      is_same_type<eT3, eT1>::no &&
      is_same_type<eT3, fp_eT1>::no &&
      is_same_type<eT3, uint_eT1>::no &&
      is_same_type<eT3, twoway_promoted_eT>::no)
    source += read_file("defs/" + type_prefix<eT3>() + "_defs.cl");

  if (is_same_type<eT3, fp_eT3>::no &&
      is_same_type<eT2, fp_eT3>::no &&
      is_same_type<eT1, fp_eT3>::no &&
      is_same_type<fp_eT2, fp_eT3>::no &&
      is_same_type<fp_eT1, fp_eT3>::no &&
      is_same_type<twoway_promoted_eT, fp_eT3>::no)
    source += read_file("defs/" + type_prefix<fp_eT3>() + "_defs.cl");

  if (is_same_type<eT3, uint_eT3>::no &&
      is_same_type<eT2, uint_eT3>::no &&
      is_same_type<eT1, uint_eT3>::no &&
      is_same_type<uint_eT2, uint_eT3>::no &&
      is_same_type<uint_eT1, uint_eT3>::no &&
      is_same_type<twoway_promoted_eT, uint_eT3>::no)
    source += read_file("defs/" + type_prefix<uint_eT3>() + "_defs.cl");

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
    source += read_file("defs/" + type_prefix<threeway_promoted_eT>() + "_defs.cl");

  return source;
  }



inline
std::string
kernel_src::get_threeway_source(const threeway_kernel_id::enum_id num)
  {
  const std::string kernel_name = threeway_kernel_id::get_names()[num];
  const std::string filename = "threeway/" + kernel_name + ".cl";

  std::string source;

  if (threeway_kernel_id::get_deps().count(num) > 0)
    {
    const std::vector<std::string>& deps = threeway_kernel_id::get_deps().at(num);
    for (const std::string& dep_f : deps)
      {
      source += read_file("deps/" + dep_f + ".cl");
      }
    }

  source += read_file(filename);
  return source;
  }



inline
std::string
kernel_src::init_magma_defines()
  {
  return

  "typedef struct \n"
  "  { \n"
  "  int npivots; \n"
  "  int ipiv[" COOT_STRINGIFY(MAGMABLAS_LASWP_MAX_PIVOTS) "]; \n"
  "  } magmablas_laswp_params_t; \n"
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
  "#define MAGMABLAS_LANSY_MAX_BS " COOT_STRINGIFY(MAGMABLAS_LANSY_MAX_BS) " \n";
  }



inline
const std::string&
kernel_src::get_magma_defines()
  {
  static const std::string defines = init_magma_defines();

  return defines;
  }



inline
std::string
kernel_src::get_magma_real_source(const magma_real_kernel_id::enum_id num)
  {
  const std::string kernel_name = magma_real_kernel_id::get_names()[num];
  const std::string filename = "magma_real/" + kernel_name + ".cl";

  return read_file(filename);
  }
