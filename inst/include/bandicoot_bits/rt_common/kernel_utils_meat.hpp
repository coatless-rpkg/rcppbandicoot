// Copyright 2020 Ryan Curtin (http://www.ratml.org)
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

// utility functions required by all backends
// all in the coot::rt_common:: namespace



template<typename TypeMapper, typename eT1, typename eT2, typename eT3>
inline std::string substitute_types(const std::string& input, const std::string& prefix, const TypeMapper& type_map)
  {
  // for now: shitty implementation.  try again later...
  std::string output = input;

  size_t pos = output.find("ET1_ABS");
  while (pos != std::string::npos)
    {
    output.replace(pos, 7, type_map.template abs_func<eT1>());
    pos = output.find("ET1_ABS");
    }

  pos = output.find("fp_eT1");
  while (pos != std::string::npos)
    {
    output.replace(pos, 6, type_map.template map<typename promote_type<eT1, float>::result>());
    pos = output.find("fp_eT1");
    }

  pos = output.find("uint_eT1");
  while (pos != std::string::npos)
    {
    output.replace(pos, 8, type_map.template map<typename uint_type<eT1>::result>());
    pos = output.find("uint_eT1");
    }

  pos = output.find("eT1");
  while (pos != std::string::npos)
    {
    output.replace(pos, 3, type_map.template map<eT1>());
    pos = output.find("eT1");
    }

  // Add two-way types if needed.
  if (!is_same_type<eT2, void>::value)
    {
    typedef typename promote_type<eT1, eT2>::result twoway_promoted_eT;

    pos = output.find("fp_eT2");
    while (pos != std::string::npos)
      {
      output.replace(pos, 6, type_map.template map<typename promote_type<eT2, float>::result>());
      pos = output.find("fp_eT2");
      }

    pos = output.find("uint_eT2");
    while (pos != std::string::npos)
      {
      output.replace(pos, 8, type_map.template map<typename uint_type<eT2>::result>());
      pos = output.find("uint_eT2");
      }

    pos = output.find("eT2");
    while (pos != std::string::npos)
      {
      output.replace(pos, 3, type_map.template map<eT2>());
      pos = output.find("eT2");
      }

    pos = output.find("twoway_promoted_eT");
    while (pos != std::string::npos)
      {
      output.replace(pos, 18, type_map.template map<twoway_promoted_eT>());
      pos = output.find("twoway_promoted_eT");
      }

    // Add three-way types if needed.
    if (!is_same_type<eT3, void>::value)
      {
      typedef typename promote_type<twoway_promoted_eT, eT3>::result threeway_promoted_eT;

      pos = output.find("fp_eT3");
      while (pos != std::string::npos)
        {
        output.replace(pos, 6, type_map.template map<typename promote_type<eT3, float>::result>());
        pos = output.find("fp_eT3");
        }

      pos = output.find("uint_eT3");
      while (pos != std::string::npos)
        {
        output.replace(pos, 8, type_map.template map<typename uint_type<eT3>::result>());
        pos = output.find("uint_eT3");
        }

      pos = output.find("eT3");
      while (pos != std::string::npos)
        {
        output.replace(pos, 3, type_map.template map<eT3>());
        pos = output.find("eT3");
        }

      pos = output.find("threeway_promoted_eT");
      while (pos != std::string::npos)
        {
        output.replace(pos, 20, type_map.template map<threeway_promoted_eT>());
        pos = output.find("threeway_promoted_eT");
        }
      }
    }

  pos = output.find("PREFIX");
  while (pos != std::string::npos)
    {
    output.replace(pos, 6, prefix);
    pos = output.find("PREFIX");
    }

  return output;
  }



template<typename KernelType>
inline
void
init_zero_elem_kernel_map(std::vector<KernelType>& kernels,
                          std::vector<std::pair<std::string, KernelType*>>& name_map,
                          const std::vector<std::string>& kernel_names)
  {
  kernels.resize(kernel_names.size());

  for (size_t j = 0; j < kernel_names.size(); ++j)
    {
    name_map.push_back(std::make_pair(kernel_names[j], &kernels.at(j)));
    }
  }



template<typename KernelType, typename TypeMapper>
inline
std::string
get_zero_elem_kernel_src(std::vector<KernelType>& kernels,
                         const std::string& source,
                         const std::vector<std::string>& kernel_names,
                         std::vector<std::pair<std::string, KernelType*>>& name_map,
                         const TypeMapper& type_map)
  {
  const std::string src = substitute_types<TypeMapper, void, void, void>(source, "", type_map);

  init_zero_elem_kernel_map(kernels, name_map, kernel_names);

  return src;
  }



template<typename KernelType>
inline
void
init_one_elem_kernel_map(kernels_t<std::vector<KernelType>>& kernels,
                         std::vector<std::pair<std::string, KernelType*>>& name_map,
                         const std::vector<std::string>& kernel_names,
                         const std::string& prefix,
                         const bool has_float64)
  {
  kernels.u32_kernels.resize(kernel_names.size());
  kernels.s32_kernels.resize(kernel_names.size());
  kernels.u64_kernels.resize(kernel_names.size());
  kernels.s64_kernels.resize(kernel_names.size());
  kernels.f_kernels.resize(kernel_names.size());
  if (has_float64)
    {
    kernels.d_kernels.resize(kernel_names.size());
    }

  for (size_t j = 0; j < kernel_names.size(); ++j)
    {
    name_map.push_back(std::make_pair(prefix + "u32_" + kernel_names[j], &kernels.u32_kernels.at(j)));
    }
  for (size_t j = 0; j < kernel_names.size(); ++j)
    {
    name_map.push_back(std::make_pair(prefix + "s32_" + kernel_names[j], &kernels.s32_kernels.at(j)));
    }
  for (size_t j = 0; j < kernel_names.size(); ++j)
    {
    name_map.push_back(std::make_pair(prefix + "u64_" + kernel_names[j], &kernels.u64_kernels.at(j)));
    }
  for (size_t j = 0; j < kernel_names.size(); ++j)
    {
    name_map.push_back(std::make_pair(prefix + "s64_" + kernel_names[j], &kernels.s64_kernels.at(j)));
    }
  for (size_t j = 0; j < kernel_names.size(); ++j)
    {
    name_map.push_back(std::make_pair(prefix + "f_" + kernel_names[j], &kernels.f_kernels.at(j)));
    }
  if (has_float64)
    {
    for (size_t j = 0; j < kernel_names.size(); ++j)
      {
      name_map.push_back(std::make_pair(prefix + "d_" + kernel_names[j], &kernels.d_kernels.at(j)));
      }
    }
  }



template<typename KernelType, typename TypeMapper, typename higher_eT1, typename higher_eT2>
inline
std::string
get_one_elem_kernel_src(kernels_t<std::vector<KernelType>>& kernels,
                        const std::string& source,
                        const std::vector<std::string>& kernel_names,
                        const std::string& prefix,
                        std::vector<std::pair<std::string, KernelType*>>& name_map,
                        const TypeMapper& type_map,
                        const bool has_float64)
  {
  const std::string u32_src = substitute_types<TypeMapper, u32, higher_eT1, higher_eT2>(source, prefix + "u32_", type_map);
  const std::string s32_src = substitute_types<TypeMapper, s32, higher_eT1, higher_eT2>(source, prefix + "s32_", type_map);
  const std::string u64_src = substitute_types<TypeMapper, u64, higher_eT1, higher_eT2>(source, prefix + "u64_", type_map);
  const std::string s64_src = substitute_types<TypeMapper, s64, higher_eT1, higher_eT2>(source, prefix + "s64_", type_map);
  const std::string   f_src = substitute_types<TypeMapper, float, higher_eT1, higher_eT2>(source, prefix + "f_", type_map);
  const std::string   d_src = has_float64 ? substitute_types<TypeMapper, double, higher_eT1, higher_eT2>(source, prefix + "d_", type_map) : "";

  init_one_elem_kernel_map(kernels, name_map, kernel_names, prefix, has_float64);

  return u32_src + s32_src + u64_src + s64_src + f_src + d_src;
  }



template<typename KernelType>
inline
void
init_one_elem_real_kernel_map(kernels_t<std::vector<KernelType>>& kernels,
                              std::vector<std::pair<std::string, KernelType*>>& name_map,
                              const std::vector<std::string>& kernel_names,
                              const std::string& prefix,
                              const bool has_float64)
  {
  kernels.f_kernels.resize(kernel_names.size());
  if (has_float64)
    {
    kernels.d_kernels.resize(kernel_names.size());
    }

  for (size_t j = 0; j < kernel_names.size(); ++j)
    {
    name_map.push_back(std::make_pair(prefix + "f_" + kernel_names[j], &kernels.f_kernels.at(j)));
    }
  if (has_float64)
    {
    for (size_t j = 0; j < kernel_names.size(); ++j)
      {
      name_map.push_back(std::make_pair(prefix + "d_" + kernel_names[j], &kernels.d_kernels.at(j)));
      }
    }
  }



template<typename KernelType, typename TypeMapper, typename higher_eT1, typename higher_eT2>
inline
std::string
get_one_elem_real_kernel_src(kernels_t<std::vector<KernelType>>& kernels,
                             const std::string& source,
                             const std::vector<std::string>& kernel_names,
                             const std::string& prefix,
                             std::vector<std::pair<std::string, KernelType*>>& name_map,
                             const TypeMapper& type_map,
                             const bool has_float64)
  {
  const std::string   f_src = substitute_types<TypeMapper, float, higher_eT1, higher_eT2>(source, prefix + "f_", type_map);
  const std::string   d_src = has_float64 ? substitute_types<TypeMapper, double, higher_eT1, higher_eT2>(source, prefix + "d_", type_map) : "";

  init_one_elem_real_kernel_map(kernels, name_map, kernel_names, prefix, has_float64);

  return f_src + d_src;
  }



template<typename KernelType>
inline
void
init_one_elem_integral_kernel_map(kernels_t<std::vector<KernelType>>& kernels,
                                  std::vector<std::pair<std::string, KernelType*>>& name_map,
                                  const std::vector<std::string>& kernel_names,
                                  const std::string& prefix)
  {
  kernels.u32_kernels.resize(kernel_names.size());
  kernels.s32_kernels.resize(kernel_names.size());
  kernels.u64_kernels.resize(kernel_names.size());
  kernels.s64_kernels.resize(kernel_names.size());

  for (size_t j = 0; j < kernel_names.size(); ++j)
    {
    name_map.push_back(std::make_pair(prefix + "u32_" + kernel_names[j], &kernels.u32_kernels.at(j)));
    }
  for (size_t j = 0; j < kernel_names.size(); ++j)
    {
    name_map.push_back(std::make_pair(prefix + "s32_" + kernel_names[j], &kernels.s32_kernels.at(j)));
    }
  for (size_t j = 0; j < kernel_names.size(); ++j)
    {
    name_map.push_back(std::make_pair(prefix + "u64_" + kernel_names[j], &kernels.u64_kernels.at(j)));
    }
  for (size_t j = 0; j < kernel_names.size(); ++j)
    {
    name_map.push_back(std::make_pair(prefix + "s64_" + kernel_names[j], &kernels.s64_kernels.at(j)));
    }
  }



template<typename KernelType, typename TypeMapper, typename higher_eT1, typename higher_eT2>
inline
std::string
get_one_elem_integral_kernel_src(kernels_t<std::vector<KernelType>>& kernels,
                                 const std::string& source,
                                 const std::vector<std::string>& kernel_names,
                                 const std::string& prefix,
                                 std::vector<std::pair<std::string, KernelType*>>& name_map,
                                 const TypeMapper& type_map)
  {
  const std::string u32_src = substitute_types<TypeMapper, u32, higher_eT1, higher_eT2>(source, prefix + "u32_", type_map);
  const std::string s32_src = substitute_types<TypeMapper, s32, higher_eT1, higher_eT2>(source, prefix + "s32_", type_map);
  const std::string u64_src = substitute_types<TypeMapper, u64, higher_eT1, higher_eT2>(source, prefix + "u64_", type_map);
  const std::string s64_src = substitute_types<TypeMapper, s64, higher_eT1, higher_eT2>(source, prefix + "s64_", type_map);

  init_one_elem_integral_kernel_map(kernels, name_map, kernel_names, prefix);

  return u32_src + s32_src + u64_src + s64_src;
  }



template<typename KernelType>
inline
void
init_two_elem_kernel_map(kernels_t<kernels_t<std::vector<KernelType>>>& kernels,
                         std::vector<std::pair<std::string, KernelType*>>& name_map,
                         const std::vector<std::string>& kernel_names,
                         const std::string& prefix,
                         const bool has_float64)
  {
  init_one_elem_kernel_map<KernelType>(kernels.u32_kernels, name_map, kernel_names, prefix + "u32_", has_float64);
  init_one_elem_kernel_map<KernelType>(kernels.s32_kernels, name_map, kernel_names, prefix + "s32_", has_float64);
  init_one_elem_kernel_map<KernelType>(kernels.u64_kernels, name_map, kernel_names, prefix + "u64_", has_float64);
  init_one_elem_kernel_map<KernelType>(kernels.s64_kernels, name_map, kernel_names, prefix + "s64_", has_float64);
  init_one_elem_kernel_map<KernelType>(kernels.f_kernels, name_map, kernel_names, prefix + "f_", has_float64);
  if (has_float64)
    {
    init_one_elem_kernel_map<KernelType>(kernels.d_kernels, name_map, kernel_names, prefix + "d_", has_float64);
    }
  }



template<typename KernelType, typename TypeMapper, typename higher_eT>
inline
std::string
get_two_elem_kernel_src(kernels_t<kernels_t<std::vector<KernelType>>>& kernels,
                        const std::string& source,
                        const std::vector<std::string>& kernel_names,
                        const std::string& prefix,
                        std::vector<std::pair<std::string, KernelType*>>& name_map,
                        const TypeMapper& type_map,
                        const bool has_float64)
  {
  const std::string u32_src = get_one_elem_kernel_src<KernelType, TypeMapper, u32, higher_eT>(kernels.u32_kernels, source, kernel_names, prefix + "u32_", name_map, type_map, has_float64);
  const std::string s32_src = get_one_elem_kernel_src<KernelType, TypeMapper, s32, higher_eT>(kernels.s32_kernels, source, kernel_names, prefix + "s32_", name_map, type_map, has_float64);
  const std::string u64_src = get_one_elem_kernel_src<KernelType, TypeMapper, u64, higher_eT>(kernels.u64_kernels, source, kernel_names, prefix + "u64_", name_map, type_map, has_float64);
  const std::string s64_src = get_one_elem_kernel_src<KernelType, TypeMapper, s64, higher_eT>(kernels.s64_kernels, source, kernel_names, prefix + "s64_", name_map, type_map, has_float64);
  const std::string f_src = get_one_elem_kernel_src<KernelType, TypeMapper, float, higher_eT>(kernels.f_kernels, source, kernel_names, prefix + "f_", name_map, type_map, has_float64);
  const std::string d_src = has_float64 ? get_one_elem_kernel_src<KernelType, TypeMapper, double, higher_eT>(kernels.d_kernels, source, kernel_names, prefix + "d_", name_map, type_map, has_float64) : "";

  return u32_src + s32_src + u64_src + s64_src + f_src + d_src;
  }



template<typename KernelType>
inline
void
init_three_elem_kernel_map(kernels_t<kernels_t<kernels_t<std::vector<KernelType>>>>& kernels,
                           std::vector<std::pair<std::string, KernelType*>>& name_map,
                           const std::vector<std::string>& kernel_names,
                           const std::string& prefix,
                           const bool has_float64)
  {
  init_two_elem_kernel_map(kernels.u32_kernels, name_map, kernel_names, prefix + "u32_", has_float64);
  init_two_elem_kernel_map(kernels.s32_kernels, name_map, kernel_names, prefix + "s32_", has_float64);
  init_two_elem_kernel_map(kernels.u64_kernels, name_map, kernel_names, prefix + "u64_", has_float64);
  init_two_elem_kernel_map(kernels.s64_kernels, name_map, kernel_names, prefix + "s64_", has_float64);
  init_two_elem_kernel_map(kernels.f_kernels, name_map, kernel_names, prefix + "f_", has_float64);
  if (has_float64)
    {
    init_two_elem_kernel_map(kernels.d_kernels, name_map, kernel_names, prefix + "d_", has_float64);
    }
  }



template<typename KernelType, typename TypeMapper>
inline std::string get_three_elem_kernel_src(kernels_t<kernels_t<kernels_t<std::vector<KernelType>>>>& kernels,
                                             const std::string& source,
                                             const std::vector<std::string>& kernel_names,
                                             std::vector<std::pair<std::string, KernelType*>>& name_map,
                                             const TypeMapper& type_map,
                                             const bool has_float64)
  {
  const std::string u32_src = get_two_elem_kernel_src<KernelType, TypeMapper, u32>(kernels.u32_kernels, source, kernel_names, "u32_", name_map, type_map, has_float64);
  const std::string s32_src = get_two_elem_kernel_src<KernelType, TypeMapper, s32>(kernels.s32_kernels, source, kernel_names, "s32_", name_map, type_map, has_float64);
  const std::string u64_src = get_two_elem_kernel_src<KernelType, TypeMapper, u64>(kernels.u64_kernels, source, kernel_names, "u64_", name_map, type_map, has_float64);
  const std::string s64_src = get_two_elem_kernel_src<KernelType, TypeMapper, s64>(kernels.s64_kernels, source, kernel_names, "s64_", name_map, type_map, has_float64);
  const std::string f_src = get_two_elem_kernel_src<KernelType, TypeMapper, float>(kernels.f_kernels, source, kernel_names, "f_", name_map, type_map, has_float64);
  const std::string d_src = has_float64 ? get_two_elem_kernel_src<KernelType, TypeMapper, double>(kernels.d_kernels, source, kernel_names, "d_", name_map, type_map, has_float64) : "";

  return u32_src + s32_src + u64_src + s64_src + f_src + d_src;
  }
