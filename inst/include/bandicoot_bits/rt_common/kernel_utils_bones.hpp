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



/**
 * Given CUDA/OpenCL source `source`, and the types `eT1`, `eT2`, and `eT3`,
 * substitute those types into the `source` string, and use `prefix` as the
 * prefixed name for each kernel.  This is meant to be used during kernel
 * compilation, in order to generate kernels for each type.  The `TypeMapper`
 * object is used to map types (eT1/eT2/eT3) to strings representing the type
 * that should be used on the device.
 */
template<typename TypeMapper, typename eT1, typename eT2, typename eT3>
inline std::string substitute_types(const std::string& source, const std::string& prefix, const TypeMapper& type_map);



/**
 * Initialize `name_map` with names for all of the zero-element kernels given in
 * `kernels`.  This is equivalent to calling `get_zero_elem_kernel_src()`, except
 * that no source is collected.
 */
template<typename KernelType>
inline void init_zero_elem_kernel_map(std::vector<KernelType>& kernels,
                                      std::vector<std::pair<std::string, KernelType*>>& name_map,
                                      const std::vector<std::string>& kernel_names);



/**
 * Given one-element kernel source `source` and a collection of KernelTypes
 * `kernels`, generate a source string that contains those kernels.
 *
 * The `name_map` structure will be filled with pairs mapping the names of
 * generated kernels to KernelType objects.
 *
 * The `TypeMapper` object is used to map types to strings representing the type that should be used on the device.
 */
template<typename KernelType, typename TypeMapper>
inline std::string get_zero_elem_kernel_src(std::vector<KernelType>& kernels,
                                            const std::string& source,
                                            const std::vector<std::string>& kernel_names,
                                            std::vector<std::pair<std::string, KernelType*>>& name_map,
                                            const TypeMapper& type_map);



/**
 * Initialize `name_map` with names for all of the one-element kernels given in
 * `kernels`.  This is equivalent to calling `get_one_elem_kernel_src()`, except
 * that no source is collected.
 */
template<typename KernelType>
inline void init_one_elem_kernel_map(kernels_t<std::vector<KernelType>>& kernels,
                                     std::vector<std::pair<std::string, KernelType*>>& name_map,
                                     const std::string& prefix,
                                     const std::vector<std::string>& kernel_names,
                                     const bool has_float64);



/**
 * Given one-element kernel source `source` and a collection of KernelTypes
 * `kernels`, generate a source string that contains those kernels specialized
 * to each of the types supported by bandicoot, with the appropriate prefixes.
 *
 * An additional prefix can be specified with the `prefix` option; this extra
 * prefix will be prepended to the prefix generated for each kernel.
 *
 * The `name_map` structure will be filled with pairs mapping the names of
 * generated kernels to KernelType objects.
 *
 * The `TypeMapper` object is used to map types (eT1/eT2/eT3) to strings
 * representing the type that should be used on the device.
 *
 * `higher_eT1` and `higher_eT2` represent the other types for when this is
 * called to generate two- and three-element kernel source.
 */
template<typename KernelType, typename TypeMapper, typename higher_eT1 = void, typename higher_eT2 = void>
inline std::string get_one_elem_kernel_src(kernels_t<std::vector<KernelType>>& kernels,
                                           const std::string& source,
                                           const std::vector<std::string>& kernel_names,
                                           const std::string& prefix,
                                           std::vector<std::pair<std::string, KernelType*>>& name_map,
                                           const TypeMapper& type_map,
                                           const bool has_float64);



/**
 * Initialize `name_map` with names for all of the one-element real kernels
 * given in `kernels`.  This is equivalent to calling
 * `get_one_elem_real_kernel_src()`, except that no source is collected.
 */
template<typename KernelType>
inline void init_one_elem_real_kernel_map(kernels_t<std::vector<KernelType>>& kernels,
                                          std::vector<std::pair<std::string, KernelType*>>& name_map,
                                          const std::string& prefix,
                                          const std::vector<std::string>& kernel_names,
                                          const bool has_float64);



/**
 * Given one-element real kernel source `source` and a collection of KernelTypes
 * `kernels`, generate a source string that contains those kernels specialized
 * to each of the types supported by bandicoot, with the appropriate prefixes.
 *
 * An additional prefix can be specified with the `prefix` option; this extra
 * prefix will be prepended to the prefix generated for each kernel.
 *
 * The `name_map` structure will be filled with pairs mapping the names of
 * generated kernels to KernelType objects.
 *
 * The `TypeMapper` object is used to map types (eT1/eT2/eT3) to strings
 * representing the type that should be used on the device.
 *
 * `higher_eT1` and `higher_eT2` represent the other types for when this is
 * called to generate two- and three-element kernel source.
 */
template<typename KernelType, typename TypeMapper, typename higher_eT1 = void, typename higher_eT2 = void>
inline std::string get_one_elem_real_kernel_src(kernels_t<std::vector<KernelType>>& kernels,
                                                const std::string& source,
                                                const std::vector<std::string>& kernel_names,
                                                const std::string& prefix,
                                                std::vector<std::pair<std::string, KernelType*>>& name_map,
                                                const TypeMapper& type_map,
                                                const bool has_float64);



/**
 * Initialize `name_map` with names for all of the one-element integral kernels
 * given in `kernels`.  This is equivalent to calling
 * `get_one_elem_integral_kernel_src()`, except that no source is collected.
 */
template<typename KernelType>
inline void init_one_elem_integral_kernel_map(kernels_t<std::vector<KernelType>>& kernels,
                                              std::vector<std::pair<std::string, KernelType*>>& name_map,
                                              const std::string& prefix,
                                              const std::vector<std::string>& kernel_names);



/**
 * Given one-element integral kernel source `source` and a collection of KernelTypes
 * `kernels`, generate a source string that contains those kernels specialized
 * to each of the types supported by bandicoot, with the appropriate prefixes.
 *
 * An additional prefix can be specified with the `prefix` option; this extra
 * prefix will be prepended to the prefix generated for each kernel.
 *
 * The `name_map` structure will be filled with pairs mapping the names of
 * generated kernels to KernelType objects.
 *
 * The `TypeMapper` object is used to map types (eT1/eT2/eT3) to strings
 * representing the type that should be used on the device.
 *
 * `higher_eT1` and `higher_eT2` represent the other types for when this is
 * called to generate two- and three-element kernel source.
 */
template<typename KernelType, typename TypeMapper, typename higher_eT1 = void, typename higher_eT2 = void>
inline std::string get_one_elem_integral_kernel_src(kernels_t<std::vector<KernelType>>& kernels,
                                                    const std::string& source,
                                                    const std::vector<std::string>& kernel_names,
                                                    const std::string& prefix,
                                                    std::vector<std::pair<std::string, KernelType*>>& name_map,
                                                    const TypeMapper& type_map);



/**
 * Initialize `name_map` with names for all of the two-element kernels given in
 * `kernels`.  This is equivalent to calling `get_two_elem_kernel_src()`, except
 * that no source is collected.
 */
template<typename KernelType>
inline void init_two_elem_kernel_map(kernels_t<kernels_t<std::vector<KernelType>>>& kernels,
                                     std::vector<std::pair<std::string, KernelType*>>& name_map,
                                     const std::string& prefix,
                                     const std::vector<std::string>& kernel_names,
                                     const bool has_float64);



/**
 * Given two-element kernel source `source` and a collection of KernelTypes
 * `kernels`, generate a source string that contains those kernels specialized
 * to all pairs of the the types supported by bandicoot, with the appropriate
 * prefixes.
 *
 * An additional prefix can be specified with the `prefix` option; this extra
 * prefix will be prepended to the prefix generated for each kernel.
 *
 * The `name_map` structure will be filled with pairs mapping the names of
 * generated kernels to KernelType objects.
 *
 * The `TypeMapper` object is used to map types (eT1/eT2/eT3) to strings
 * representing the type that should be used on the device.
 *
 * `higher_eT1` represents the other type for when this is called as a part of
 * generating three-element kernel sources.
 */
template<typename KernelType, typename TypeMapper, typename higher_eT = void>
inline std::string get_two_elem_kernel_src(kernels_t<kernels_t<std::vector<KernelType>>>& kernels,
                                           const std::string& source,
                                           const std::vector<std::string>& kernel_names,
                                           const std::string& prefix,
                                           std::vector<std::pair<std::string, KernelType*>>& name_map,
                                           const TypeMapper& type_map,
                                           const bool has_float64);



/**
 * Initialize `name_map` with names for all of the three-element kernels given
 * in `kernels`.  This is equivalent to calling `get_three_elem_kernel_src()`,
 * except that no source is collected.
 */
template<typename KernelType>
inline void init_three_elem_kernel_map(kernels_t<kernels_t<kernels_t<std::vector<KernelType>>>>& kernels,
                                       std::vector<std::pair<std::string, KernelType*>>& name_map,
                                       const std::string& prefix,
                                       const std::vector<std::string>& kernel_names,
                                       const bool has_float64);



/**
 * Given three-element kernel source `source` and a collection of KernelTypes
 * `kernels`, generate a source string that contains those kernels specialized
 * to all triples of the types supported by bandicoot, with the appropriate
 * prefixes.
 *
 * The `name_map` structure will be filled with pairs mapping the names of
 * generated kernels to KernelType objects.
 *
 * The `TypeMapper` object is used to map types (eT1/eT2/eT3) to strings
 * representing the type that should be used on the device.
 */
template<typename KernelType, typename TypeMapper>
inline std::string get_three_elem_kernel_src(kernels_t<kernels_t<kernels_t<std::vector<KernelType>>>>& kernels,
                                             const std::string& source,
                                             const std::vector<std::string>& kernel_names,
                                             std::vector<std::pair<std::string, KernelType*>>& name_map,
                                             const TypeMapper& type_map,
                                             const bool has_float64);
