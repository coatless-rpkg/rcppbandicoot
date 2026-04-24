// Copyright 2025 Ryan Curtin (http://www.ratml.org)
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



//
// Utility function to append kernel macros for a type T to an existing list of macros.
//

template<coot_backend_t backend, size_t i>
void append_kernel_macros(std::vector<std::string>& result)
  {
  return;
  }



template<coot_backend_t backend, size_t i, typename T, typename... Ts>
void append_kernel_macros(std::vector<std::string>& result)
  {
  // All of these autos are char_array<N>s.
  const auto kernel_params_macro = macro_kernel_params<T, i, backend>::str();
  const auto bounds_check_macro = macro_bounds_check<T, i, backend>::str();
  const auto elem_access_macro = macro_elem_access<T, i, backend>::str();
  const auto elem_type_macro = macro_elem_type<typename T::elem_type, i, backend>::str();
  const auto elem_type_conv_macro = macro_conv_elem_type<typename T::elem_type, i, backend>::str();

  result.insert(result.end(),
      {
      std::string(kernel_params_macro.begin(), kernel_params_macro.end() - 1),
      std::string(bounds_check_macro.begin(), bounds_check_macro.end() - 1),
      std::string(elem_access_macro.begin(), elem_access_macro.end() - 1),
      std::string(elem_type_macro.begin(), elem_type_macro.end() - 1),
      std::string(elem_type_conv_macro.begin(), elem_type_conv_macro.end() - 1),
      });

  append_kernel_macros<backend, i + 1, Ts...>(result);
  }



//
// Vulkan: emit COOT_OBJECT_i_PARAMS(name) for each argument.
//

template<size_t i, size_t start_binding>
inline void append_buffer_macros(std::vector<std::string>&)
  {
  return;
  }

template<size_t i, size_t start_binding, typename T, typename... Ts>
inline void append_buffer_macros(std::vector<std::string>& result)
  {
  const auto buffers_macro = macro_kernel_push_params<T, i, start_binding>::str();
  result.push_back(std::string(buffers_macro.begin(), buffers_macro.end() - 1));
  constexpr const size_t next_start_binding = start_binding + vk_leaf_count<T>::value;
  append_buffer_macros<i + 1, next_start_binding, Ts...>(result);
  }

template<coot_backend_t backend, typename... Ts>
inline
typename enable_if2<backend == VULKAN_BACKEND, void>::result
append_buffer_macros_if_vulkan(std::vector<std::string>& result)
  {
  append_buffer_macros<0, 0, Ts...>(result);
  }

template<coot_backend_t backend, typename... Ts>
inline
typename enable_if2<backend != VULKAN_BACKEND, void>::result
append_buffer_macros_if_vulkan(std::vector<std::string>&)
  {
  return;
  }



template<coot_backend_t backend, kernel_id::enum_id num>
inline
typename std::enable_if<is_reduction_kernel<num>::value>::type
append_reduction_macros(std::vector<std::string>& result)
  {
  const auto init_macro = reduction_op_init_macro<num, backend>::str();
  const auto inner_macro = reduction_op_inner_macro<num, backend>::str();
  const auto final_macro = reduction_op_final_macro<num, backend>::str();

  result.insert(result.end(),
      {
      std::string(init_macro.begin(), init_macro.end() - 1),
      std::string(inner_macro.begin(), inner_macro.end() - 1),
      std::string(final_macro.begin(), final_macro.end() - 1),
      });
  }



template<coot_backend_t backend, kernel_id::enum_id num>
inline
typename std::enable_if<!is_reduction_kernel<num>::value>::type
append_reduction_macros(std::vector<std::string>& result)
  {
  return;
  }



//
// `generator` is a class that is used as a way to generate GPU kernel sources
// and the definitions needed to compile them correctly.
//
// So much as is possible, all of the strings computed by `generator` are done
// at compile-time.  In this way, we embed only the kernels we need for a program
// into its source, and only the definitions needed for the ways those kernels are
// used.
//



template
  <
  coot_backend_t backend, // the backend being used
  kernel_id::enum_id num, // the kernel being compiled
  typename... Ts          // the types of the arguments to the kernel, in order
  >
struct generator
  {
  // Get the concatenated kernel source, including:
  //
  //  * any preliminaries for the backend
  //  * any type definitions for the backend and types of the arguments
  //  * any utility functions needed by the kernel
  //  * the kernel source itself
  //
  // This is concatenated at runtime, but all of the elements are computed at compile time.
  static inline std::string kernel_source()
    {
    typedef typename elem_types<Ts...>::result eTs;
    typedef typename type_defs<backend, eTs>::result type_def_list;

    // For CUDA we need 'extern "C" {' to prevent name mangling.
    constexpr size_t extern_len = (backend == CUDA_BACKEND) ? 16 : 0;

    // Compute the total length.
    constexpr size_t total_len =
        preamble<backend>::len() +
        type_defs_len_helper<type_def_list>::len() +
        kernel_funcs<backend, Ts...>::len() +
        pre_kernel_src<backend>::len() +
        kernel_src<num, backend>::len() +
        extern_len +
        post_kernel_src<backend>::len() +
        1; // "\0"

    // Fill with null terminators.
    std::string result(total_len, '\0');

    // The preamble comes first.
    constexpr const size_t preamble_len = preamble<backend>::len();
    result.replace(0, preamble_len, &(preamble<backend>::str()[0]));

    // Now the definitions for the types we will use.
    size_t start_loc = type_defs_concat_helper<type_def_list>::concat_src(result, preamble_len);

    // Next, any utility functions used by the arguments of the kernel.
    result.replace(start_loc, kernel_funcs<backend, Ts...>::len(), &(kernel_funcs<backend, Ts...>::str()[0]));
    start_loc += kernel_funcs<backend, Ts...>::len();

    // Add anything required before the kernel.
    result.replace(start_loc, pre_kernel_src<backend>::len(), &(pre_kernel_src<backend>::str()[0]));
    start_loc += pre_kernel_src<backend>::len();

    // Now copy in the actual kernel source.
    result.replace(start_loc, kernel_src<num, backend>::len(), &(kernel_src<num, backend>::str()[0]));
    start_loc += kernel_src<num, backend>::len();

    // Add any post-kernel bits, if needed.
    result.replace(start_loc, post_kernel_src<backend>::len(), &(post_kernel_src<backend>::str()[0]));

    return result;
    }



  // Get a vector containing all of the preprocessor definitions necessary to compile
  // the kernel.  (This just collects all of the results of the functions above.)
  static inline std::vector<std::string> kernel_macros()
    {
    const auto kernel_name_macro = macro_kernel_name<num, Ts...>::str();
    std::vector<std::string> result = {{ std::string(kernel_name_macro.begin(), kernel_name_macro.end() - 1) }};

    append_kernel_macros<backend, 0, Ts...>(result);

    // Vulkan: COOT_OBJECT_i_PARAMS(name) declarations for each argument.
    append_buffer_macros_if_vulkan<backend, Ts...>(result);

    // For reduction kernels, append COOT_INIT_OP / COOT_INNER_OP / COOT_FINAL_OP macros.
    append_reduction_macros<backend, num>(result);

    return result;
    }

  };
