// Copyright 2026 Marcus Edel (http://www.kurg.org)
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



struct coot_vk_pointer_access_concat_name
  {
  static inline constexpr auto& str() { return "COOT_CONCAT(name,"; }
  };

struct coot_vk_pointer_access_buf_mid
  {
  static inline constexpr auto& str() { return "_buf).data[uint(COOT_CONCAT(name,"; }
  };

struct coot_vk_pointer_access_suffix1
  {
  static inline constexpr auto& str() { return "_offset) + "; }
  };

template<typename arg_name_prefix>
struct coot_pointer_access<VULKAN_BACKEND, arg_name_prefix>
  {
  using prefix = concat_str<
    coot_vk_pointer_access_concat_name,
    arg_name_prefix,
    coot_vk_pointer_access_buf_mid,
    arg_name_prefix,
    coot_vk_pointer_access_suffix1
    >;

  struct suffix
    {
    static inline constexpr auto& str() { return ")]"; }
    };
  };
