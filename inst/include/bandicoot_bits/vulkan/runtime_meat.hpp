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



inline
runtime_t::runtime_t()
  {
  }



inline
runtime_t::~runtime_t()
  {
  return;

#if 0
  const char* cleanup_env = std::getenv("COOT_VULKAN_ENABLE_CLEANUP");
  if (cleanup_env == nullptr || cleanup_env[0] == '\0')
    {
    return;
    }

  if (valid == false)
    {
    return;
    }

  if (device != VK_NULL_HANDLE)
    {
    coot_wrapper(vkDeviceWaitIdle)(device);
    }

  for (auto& entry : pipelines)
    {
    pipeline_t& p = entry.second;
    if (p.pipeline != VK_NULL_HANDLE) { coot_wrapper(vkDestroyPipeline)(device, p.pipeline, nullptr); }
    if (p.layout != VK_NULL_HANDLE) { coot_wrapper(vkDestroyPipelineLayout)(device, p.layout, nullptr); }
    if (p.set_layout != VK_NULL_HANDLE) { coot_wrapper(vkDestroyDescriptorSetLayout)(device, p.set_layout, nullptr); }
    if (p.shader != VK_NULL_HANDLE) { coot_wrapper(vkDestroyShaderModule)(device, p.shader, nullptr); }

    p.pipeline = VK_NULL_HANDLE;
    p.layout = VK_NULL_HANDLE;
    p.set_layout = VK_NULL_HANDLE;
    p.shader = VK_NULL_HANDLE;
    }

  pipelines.clear();

  if (descriptor_pool != VK_NULL_HANDLE) { coot_wrapper(vkDestroyDescriptorPool)(device, descriptor_pool, nullptr); }
  if (command_pool != VK_NULL_HANDLE) { coot_wrapper(vkDestroyCommandPool)(device, command_pool, nullptr); }
  if (device != VK_NULL_HANDLE) { coot_wrapper(vkDestroyDevice)(device, nullptr); }
  if (debug_messenger != VK_NULL_HANDLE)
    {
    auto destroy_messenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        coot_wrapper(vkGetInstanceProcAddr)(instance, "vkDestroyDebugUtilsMessengerEXT"));
    if (destroy_messenger)
      {
      destroy_messenger(instance, debug_messenger, nullptr);
      }
    }
  if (instance != VK_NULL_HANDLE) { coot_wrapper(vkDestroyInstance)(instance, nullptr); }

  descriptor_pool = VK_NULL_HANDLE;
  command_pool = VK_NULL_HANDLE;
  device = VK_NULL_HANDLE;
  instance = VK_NULL_HANDLE;
  debug_messenger = VK_NULL_HANDLE;
  physical_device = VK_NULL_HANDLE;
  queue = VK_NULL_HANDLE;
  valid = false;
#endif
  }


inline
bool
runtime_t::init(const bool, const uword wanted_platform, const uword wanted_device, const bool print_info)
  {
  coot_debug_sigprint();
  coot_ignore(wanted_platform);

  valid = false;

  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "bandicoot";
  app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.pEngineName = "bandicoot";
  app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.apiVersion = VK_API_VERSION_1_1;

  const char* validation_env = std::getenv("COOT_VULKAN_VALIDATION");
  validation_enabled = (validation_env != nullptr && validation_env[0] != '\0');

  std::vector<const char*> enabled_layers;
  std::vector<const char*> enabled_extensions;

  if (validation_enabled)
    {
    const char* validation_layer = "VK_LAYER_KHRONOS_validation";

    uint32_t layer_count = 0;
    coot_wrapper(vkEnumerateInstanceLayerProperties)(&layer_count, nullptr);
    std::vector<VkLayerProperties> layers(layer_count);
    coot_wrapper(vkEnumerateInstanceLayerProperties)(&layer_count, layers.data());

    bool has_validation = false;
    for (const auto& layer : layers)
      {
      if (std::strcmp(layer.layerName, validation_layer) == 0)
        {
        has_validation = true;
        break;
        }
      }

    if (has_validation)
      {
      enabled_layers.push_back(validation_layer);

      uint32_t ext_count = 0;
      coot_wrapper(vkEnumerateInstanceExtensionProperties)(nullptr, &ext_count, nullptr);
      std::vector<VkExtensionProperties> exts(ext_count);
      coot_wrapper(vkEnumerateInstanceExtensionProperties)(nullptr, &ext_count, exts.data());

      for (const auto& ext : exts)
        {
        if (std::strcmp(ext.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0)
          {
          enabled_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
          break;
          }
        }
      }
    }

  VkInstanceCreateInfo instance_info{};
  instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instance_info.pApplicationInfo = &app_info;
  instance_info.enabledLayerCount = static_cast<uint32_t>(enabled_layers.size());
  instance_info.ppEnabledLayerNames = enabled_layers.empty() ? nullptr : enabled_layers.data();
  instance_info.enabledExtensionCount = static_cast<uint32_t>(enabled_extensions.size());
  instance_info.ppEnabledExtensionNames = enabled_extensions.empty() ? nullptr : enabled_extensions.data();

  VkResult result = coot_wrapper(vkCreateInstance)(&instance_info, nullptr, &instance);
  coot_check_vk_error(result, "coot::vulkan::runtime_t::init(): vkCreateInstance() failed");

  if (!enabled_layers.empty())
    {
    auto create_messenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        coot_wrapper(vkGetInstanceProcAddr)(instance, "vkCreateDebugUtilsMessengerEXT"));
    if (create_messenger)
      {
      VkDebugUtilsMessengerCreateInfoEXT dbg_info{};
      dbg_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
      dbg_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
      dbg_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
      dbg_info.pfnUserCallback = [](VkDebugUtilsMessageSeverityFlagBitsEXT,
                                    VkDebugUtilsMessageTypeFlagsEXT,
                                    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
                                    void*) -> VkBool32
        {
        std::cerr << "Vulkan validation: " << callback_data->pMessage << std::endl;
        return VK_FALSE;
        };

      create_messenger(instance, &dbg_info, nullptr, &debug_messenger);
      }
    }

  uint32_t device_count = 0;
  result = coot_wrapper(vkEnumeratePhysicalDevices)(instance, &device_count, nullptr);
  coot_check_vk_error(result, "coot::vulkan::runtime_t::init(): vkEnumeratePhysicalDevices() failed");
  coot_check_runtime_error((device_count == 0), "coot::vulkan::runtime_t::init(): no Vulkan devices found");

  std::vector<VkPhysicalDevice> devices(device_count);
  result = coot_wrapper(vkEnumeratePhysicalDevices)(instance, &device_count, devices.data());
  coot_check_vk_error(result, "coot::vulkan::runtime_t::init(): vkEnumeratePhysicalDevices() failed");

  coot_check_runtime_error((wanted_device >= device_count), "coot::vulkan::runtime_t::init(): invalid wanted_device");
  physical_device = devices[wanted_device];

  uint32_t queue_family_count = 0;
  coot_wrapper(vkGetPhysicalDeviceQueueFamilyProperties)(physical_device, &queue_family_count, nullptr);
  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  coot_wrapper(vkGetPhysicalDeviceQueueFamilyProperties)(physical_device, &queue_family_count, queue_families.data());

  bool found_compute = false;
  for (uint32_t i = 0; i < queue_family_count; ++i)
    {
    if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
      {
      queue_family_index = i;
      found_compute = true;
      break;
      }
    }
  coot_check_runtime_error((found_compute == false), "coot::vulkan::runtime_t::init(): no compute-capable queue found");

  VkPhysicalDeviceFeatures features{};
  coot_wrapper(vkGetPhysicalDeviceFeatures)(physical_device, &features);
  supports_int64 = features.shaderInt64 == VK_TRUE;
  supports_fp64 = features.shaderFloat64 == VK_TRUE;

  VkPhysicalDeviceFeatures enabled_features{};
  enabled_features.shaderInt64 = supports_int64 ? VK_TRUE : VK_FALSE;
  enabled_features.shaderFloat64 = supports_fp64 ? VK_TRUE : VK_FALSE;

  float queue_priority = 1.0f;
  VkDeviceQueueCreateInfo queue_info{};
  queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_info.queueFamilyIndex = queue_family_index;
  queue_info.queueCount = 1;
  queue_info.pQueuePriorities = &queue_priority;

  VkDeviceCreateInfo device_info{};
  device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_info.queueCreateInfoCount = 1;
  device_info.pQueueCreateInfos = &queue_info;
  device_info.pEnabledFeatures = &enabled_features;

  result = coot_wrapper(vkCreateDevice)(physical_device, &device_info, nullptr, &device);
  coot_check_vk_error(result, "coot::vulkan::runtime_t::init(): vkCreateDevice() failed");

  coot_wrapper(vkGetDeviceQueue)(device, queue_family_index, 0, &queue);

  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.queueFamilyIndex = queue_family_index;
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

  result = coot_wrapper(vkCreateCommandPool)(device, &pool_info, nullptr, &command_pool);
  coot_check_vk_error(result, "coot::vulkan::runtime_t::init(): vkCreateCommandPool() failed");

  VkDescriptorPoolSize pool_size{};
  pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  pool_size.descriptorCount = 128;

  VkDescriptorPoolCreateInfo pool_create{};
  pool_create.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_create.poolSizeCount = 1;
  pool_create.pPoolSizes = &pool_size;
  pool_create.maxSets = 128;

  result = coot_wrapper(vkCreateDescriptorPool)(device, &pool_create, nullptr, &descriptor_pool);
  coot_check_vk_error(result, "coot::vulkan::runtime_t::init(): vkCreateDescriptorPool() failed");

  if (print_info)
    {
    std::cout << "coot::vulkan::runtime_t::init(): using Vulkan device " << wanted_device << std::endl;
    }

  // TODO, change the allocation strategy (persistent host-visible memory pool (256 MB)).
  {
  const VkDeviceSize pool_bytes = VkDeviceSize(256) * 1024 * 1024;

  VkBufferCreateInfo pool_buf_info{};
  pool_buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  pool_buf_info.size = pool_bytes;
  pool_buf_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  pool_buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  result = coot_wrapper(vkCreateBuffer)(device, &pool_buf_info, nullptr, &mem_pool_buffer);
  coot_check_vk_error(result, "coot::vulkan::runtime_t::init(): vkCreateBuffer() for memory pool failed");

  VkMemoryRequirements pool_mem_req;
  coot_wrapper(vkGetBufferMemoryRequirements)(device, mem_pool_buffer, &pool_mem_req);

  VkMemoryAllocateInfo pool_alloc{};
  pool_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  pool_alloc.allocationSize = pool_mem_req.size;
  pool_alloc.memoryTypeIndex = find_memory_type(pool_mem_req.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  result = coot_wrapper(vkAllocateMemory)(device, &pool_alloc, nullptr, &mem_pool_memory);
  coot_check_vk_error(result, "coot::vulkan::runtime_t::init(): vkAllocateMemory() for memory pool failed");

  result = coot_wrapper(vkBindBufferMemory)(device, mem_pool_buffer, mem_pool_memory, 0);
  coot_check_vk_error(result, "coot::vulkan::runtime_t::init(): vkBindBufferMemory() for memory pool failed");

  result = coot_wrapper(vkMapMemory)(device, mem_pool_memory, 0, VK_WHOLE_SIZE, 0, &mem_pool_mapped);
  coot_check_vk_error(result, "coot::vulkan::runtime_t::init(): vkMapMemory() for memory pool failed");

  mem_pool_used = 0;
  }

  valid = true;
  return true;
  }


inline
uint32_t
runtime_t::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) const
  {
  VkPhysicalDeviceMemoryProperties mem_properties;
  coot_wrapper(vkGetPhysicalDeviceMemoryProperties)(physical_device, &mem_properties);

  for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++)
    {
    if ((type_filter & (1 << i)) && (mem_properties.memoryTypes[i].propertyFlags & properties) == properties)
      {
      return i;
      }
    }

  coot_stop_runtime_error("coot::vulkan::runtime_t::find_memory_type(): suitable memory type not found");
  return 0;
  }


inline
VkCommandBuffer
runtime_t::begin_commands() const
  {
  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.commandPool = command_pool;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = 1;

  VkCommandBuffer command_buffer;
  VkResult result = coot_wrapper(vkAllocateCommandBuffers)(device, &alloc_info, &command_buffer);
  coot_check_vk_error(result, "coot::vulkan::runtime_t::begin_commands(): vkAllocateCommandBuffers() failed");

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  result = coot_wrapper(vkBeginCommandBuffer)(command_buffer, &begin_info);
  coot_check_vk_error(result, "coot::vulkan::runtime_t::begin_commands(): vkBeginCommandBuffer() failed");

  return command_buffer;
  }


inline
void
runtime_t::end_commands(VkCommandBuffer cmd) const
  {
  VkResult result = coot_wrapper(vkEndCommandBuffer)(cmd);
  coot_check_vk_error(result, "coot::vulkan::runtime_t::end_commands(): vkEndCommandBuffer() failed");

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &cmd;

  result = coot_wrapper(vkQueueSubmit)(queue, 1, &submit_info, VK_NULL_HANDLE);
  coot_check_vk_error(result, "coot::vulkan::runtime_t::end_commands(): vkQueueSubmit() failed");
  result = coot_wrapper(vkQueueWaitIdle)(queue);
  coot_check_vk_error(result, "coot::vulkan::runtime_t::end_commands(): vkQueueWaitIdle() failed");

  coot_wrapper(vkFreeCommandBuffers)(device, command_pool, 1, &cmd);
  }



inline
void
runtime_t::set_rng_seed(const u64 seed)
  {
  rng_seed = seed;
  rng_counter = 0;
  }



inline
u64
runtime_t::next_rng_seed()
  {
  ++rng_counter;
  return rng_seed + (rng_counter * 0x9E3779B97F4A7C15ULL);
  }



inline
void
runtime_t::synchronise()
  {
  coot_wrapper(vkQueueWaitIdle)(queue);
  }



template<typename eT>
inline
coot_vk_mem
runtime_t::acquire_memory(const uword n_elem)
  {
  const VkDeviceSize elem_bytes = VkDeviceSize(n_elem) * sizeof(eT);
  const VkDeviceSize alignment  = 64;

  for (size_t i = 0; i < free_list.size(); ++i)
    {
    const VkDeviceSize blk_off  = free_list[i].first;
    const VkDeviceSize blk_size = free_list[i].second;

    if (elem_bytes <= blk_size)
      {
      const VkDeviceSize remaining = blk_size - elem_bytes;
      if (remaining > 0)
        free_list[i] = { blk_off + elem_bytes, remaining };
      else
        free_list.erase(free_list.begin() + i);

      coot_vk_mem out;
      out.buffer = mem_pool_buffer;
      out.memory = mem_pool_memory;
      out.offset = size_t(blk_off / sizeof(eT));
      out.byte_offset = size_t(blk_off);
      out.byte_size = size_t(elem_bytes);
      return out;
      }
    }

  const VkDeviceSize byte_offset = (mem_pool_used + alignment - 1) / alignment * alignment;

  coot_check_runtime_error(
      (byte_offset + elem_bytes > VkDeviceSize(256) * 1024 * 1024),
      "coot::vulkan::runtime_t::acquire_memory(): 256 MB memory pool exhausted");

  mem_pool_used = byte_offset + elem_bytes;

  coot_vk_mem out;
  out.buffer = mem_pool_buffer;
  out.memory = mem_pool_memory;
  out.offset = size_t(byte_offset / sizeof(eT));
  out.byte_offset = size_t(byte_offset);
  out.byte_size = size_t(elem_bytes);
  return out;
  }



inline
void
runtime_t::release_memory(coot_vk_mem mem)
  {
  if (mem.byte_size == 0)
    return;

  const VkDeviceSize off  = VkDeviceSize(mem.byte_offset);
  const VkDeviceSize size = VkDeviceSize(mem.byte_size);

  auto it = std::lower_bound(free_list.begin(), free_list.end(), std::make_pair(off, VkDeviceSize(0)));
  it = free_list.insert(it, { off, size });

  {
  auto next = std::next(it);
  if (next != free_list.end() && it->first + it->second == next->first)
    {
    it->second += next->second;
    free_list.erase(next);
    }
  }

  if (it != free_list.begin())
    {
    auto prev = std::prev(it);
    if (prev->first + prev->second == it->first)
      {
      prev->second += it->second;
      it = free_list.erase(it);
      --it;
      }
    }

  if (it->first + it->second == mem_pool_used)
    {
    mem_pool_used = it->first;
    free_list.erase(it);
    }
  }



template<typename eT>
inline
bool
runtime_t::is_supported_type() const
  {
  if (is_float<eT>::value)
    {
    return true;
    }
  if (is_double<eT>::value)
    {
    return supports_fp64;
    }

  return false;
  }



#if defined(COOT_USE_SHADERC)

inline
std::vector<uint32_t>
runtime_t::compile_glsl_to_spirv(const std::string& name, const std::string& source, const std::vector<std::string>& macros)
  {
  coot_debug_sigprint();

  shaderc_compiler_t compiler = coot_wrapper(shaderc_compiler_initialize)();
  shaderc_compile_options_t options = coot_wrapper(shaderc_compile_options_initialize)();

  coot_wrapper(shaderc_compile_options_set_target_env)(options, shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_1);
  coot_wrapper(shaderc_compile_options_set_optimization_level)(options, shaderc_optimization_level_performance);

  for (const std::string& macro : macros)
    {
    const size_t eq_pos = macro.find('=');
    if (eq_pos != std::string::npos)
      {
      coot_wrapper(shaderc_compile_options_add_macro_definition)(
          options,
          macro.c_str(), eq_pos,
          macro.c_str() + eq_pos + 1, macro.size() - eq_pos - 1);
      }
    else
      {
      coot_wrapper(shaderc_compile_options_add_macro_definition)(
          options,
          macro.c_str(), macro.size(),
          nullptr, 0);
      }
    }

  #if defined(COOT_DEBUG_PRINT_KERNELS)
  get_cout_stream() << "vulkan::compile_glsl_to_spirv(): compiling kernel " << name << " with the following macros:" << std::endl;
  for (size_t i = 0; i < macros.size(); ++i)
    {
    get_cout_stream() << "  - " << macros[i] << std::endl;
    }
  get_cout_stream() << std::endl << "Kernel source:" << std::endl << source << std::endl;
  #endif

  shaderc_compilation_result_t result = coot_wrapper(shaderc_compile_into_spv)(
      compiler,
      source.c_str(),
      source.size(),
      shaderc_compute_shader,
      name.c_str(),
      "main",
      options);

  if (coot_wrapper(shaderc_result_get_compilation_status)(result) != shaderc_compilation_status_success)
    {
    const char* err = coot_wrapper(shaderc_result_get_error_message)(result);
    std::string error_str = (err != nullptr) ? std::string(err) : std::string("unknown error");

    get_cerr_stream() << "=== FAILED KERNEL: " << name << " ===" << std::endl;
    get_cerr_stream() << "Macros:" << std::endl;
    for (const std::string& macro : macros)
      {
      get_cerr_stream() << "  [" << macro << "]" << std::endl;
      }
    get_cerr_stream() << "Source:" << std::endl << source << std::endl;
    get_cerr_stream() << "=== END ===" << std::endl;

    coot_wrapper(shaderc_result_release)(result);
    coot_wrapper(shaderc_compile_options_release)(options);
    coot_wrapper(shaderc_compiler_release)(compiler);
    coot_stop_runtime_error("coot::vulkan::compile_glsl_to_spirv(): compilation of " + name + " failed: " + error_str);
    }

  const char* bytes = coot_wrapper(shaderc_result_get_bytes)(result);
  const size_t length = coot_wrapper(shaderc_result_get_length)(result);

  std::vector<uint32_t> spirv(length / sizeof(uint32_t));
  std::memcpy(spirv.data(), bytes, length);

  coot_wrapper(shaderc_result_release)(result);
  coot_wrapper(shaderc_compile_options_release)(options);
  coot_wrapper(shaderc_compiler_release)(compiler);

  return spirv;
  }



inline
pipeline_t&
runtime_t::get_pipeline_from_source(const std::string& name, const std::string& glsl_source, const size_t push_constant_size, const uint32_t num_buffers)
  {
  auto it = pipelines.find(name);
  if (it != pipelines.end())
    {
    return it->second;
    }

  std::vector<uint32_t> spirv = compile_glsl_to_spirv(name, glsl_source);

  pipeline_t p;

  VkShaderModuleCreateInfo shader_info{};
  shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shader_info.codeSize = spirv.size() * sizeof(uint32_t);
  shader_info.pCode = spirv.data();

  VkResult result = coot_wrapper(vkCreateShaderModule)(device, &shader_info, nullptr, &p.shader);
  coot_check_vk_error(result, "coot::vulkan::runtime_t::get_pipeline_from_source(): vkCreateShaderModule() failed for " + name);

  std::vector<VkDescriptorSetLayoutBinding> bindings(num_buffers);
  for (uint32_t i = 0; i < num_buffers; ++i)
    {
    bindings[i] = {};
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = num_buffers;
  layout_info.pBindings = bindings.data();

  result = coot_wrapper(vkCreateDescriptorSetLayout)(device, &layout_info, nullptr, &p.set_layout);
  coot_check_vk_error(result, "coot::vulkan::runtime_t::get_pipeline_from_source(): vkCreateDescriptorSetLayout() failed for " + name);

  VkPushConstantRange push_range{};
  push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push_range.offset = 0;
  push_range.size = (uint32_t) push_constant_size;

  VkPipelineLayoutCreateInfo pipeline_layout_info{};
  pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_info.setLayoutCount = 1;
  pipeline_layout_info.pSetLayouts = &p.set_layout;
  pipeline_layout_info.pushConstantRangeCount = 1;
  pipeline_layout_info.pPushConstantRanges = &push_range;

  result = coot_wrapper(vkCreatePipelineLayout)(device, &pipeline_layout_info, nullptr, &p.layout);
  coot_check_vk_error(result, "coot::vulkan::runtime_t::get_pipeline_from_source(): vkCreatePipelineLayout() failed for " + name);

  VkPipelineShaderStageCreateInfo stage_info{};
  stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage_info.module = p.shader;
  stage_info.pName = "main";

  VkComputePipelineCreateInfo pipeline_info{};
  pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline_info.stage = stage_info;
  pipeline_info.layout = p.layout;

  result = coot_wrapper(vkCreateComputePipelines)(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &p.pipeline);
  coot_check_vk_error(result, "coot::vulkan::runtime_t::get_pipeline_from_source(): vkCreateComputePipelines() failed for " + name);

  pipelines[name] = p;
  return pipelines[name];
  }



inline
pipeline_t&
runtime_t::get_pipeline_from_source(const std::string& name, const std::string& glsl_source, const std::vector<std::string>& macros, const size_t push_constant_size, const uint32_t num_buffers)
  {
  auto it = pipelines.find(name);
  if (it != pipelines.end())
    {
    return it->second;
    }

  std::vector<uint32_t> spirv = compile_glsl_to_spirv(name, glsl_source, macros);

  pipeline_t p;

  VkShaderModuleCreateInfo shader_info{};
  shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shader_info.codeSize = spirv.size() * sizeof(uint32_t);
  shader_info.pCode = spirv.data();

  VkResult result = coot_wrapper(vkCreateShaderModule)(device, &shader_info, nullptr, &p.shader);
  coot_check_vk_error(result, "coot::vulkan::runtime_t::get_pipeline_from_source(): vkCreateShaderModule() failed for " + name);

  std::vector<VkDescriptorSetLayoutBinding> bindings(num_buffers);
  for (uint32_t i = 0; i < num_buffers; ++i)
    {
    bindings[i] = {};
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = num_buffers;
  layout_info.pBindings = bindings.data();

  result = coot_wrapper(vkCreateDescriptorSetLayout)(device, &layout_info, nullptr, &p.set_layout);
  coot_check_vk_error(result, "coot::vulkan::runtime_t::get_pipeline_from_source(): vkCreateDescriptorSetLayout() failed for " + name);

  VkPushConstantRange push_range{};
  push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push_range.offset = 0;
  push_range.size = (uint32_t) push_constant_size;

  VkPipelineLayoutCreateInfo pipeline_layout_info{};
  pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_info.setLayoutCount = 1;
  pipeline_layout_info.pSetLayouts = &p.set_layout;
  pipeline_layout_info.pushConstantRangeCount = 1;
  pipeline_layout_info.pPushConstantRanges = &push_range;

  result = coot_wrapper(vkCreatePipelineLayout)(device, &pipeline_layout_info, nullptr, &p.layout);
  coot_check_vk_error(result, "coot::vulkan::runtime_t::get_pipeline_from_source(): vkCreatePipelineLayout() failed for " + name);

  VkPipelineShaderStageCreateInfo stage_info{};
  stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage_info.module = p.shader;
  stage_info.pName = "main";

  VkComputePipelineCreateInfo pipeline_info{};
  pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline_info.stage = stage_info;
  pipeline_info.layout = p.layout;

  result = coot_wrapper(vkCreateComputePipelines)(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &p.pipeline);
  coot_check_vk_error(result, "coot::vulkan::runtime_t::get_pipeline_from_source(): vkCreateComputePipelines() failed for " + name);

  pipelines[name] = p;
  return pipelines[name];
  }



template<typename... ProxyTypes>
inline
size_t
runtime_t::compute_gen_push_size()
  {
  if (has_sizet64())
    return push_aligned_proxies_size<u64, typename ProxyTypes::held_type...>();
  else
    return push_aligned_proxies_size<u32, typename ProxyTypes::held_type...>();
  }



template<kernel_id::enum_id num, typename... ProxyTypes>
inline
pipeline_t&
runtime_t::get_kernel()
  {
  coot_debug_sigprint();

  const std::string name = std::string(&(kernel_gen::full_name<num, typename ProxyTypes::held_type...>::str()[0]));

  auto it = pipelines.find(name);
  if (it != pipelines.end())
    {
    return it->second;
    }

  std::vector<std::string> raw_macros = kernel_gen::generator<VULKAN_BACKEND, num, typename ProxyTypes::held_type...>::kernel_macros();

  std::vector<std::string> macros;
  std::string source_defines;
  macros.reserve(raw_macros.size());
  for (const std::string& m : raw_macros)
    {
    std::string stripped = m;
    if (stripped.size() > 3 && stripped[0] == '-' && stripped[1] == 'D' && stripped[2] == ' ')
      {
      stripped = stripped.substr(3);
      }

    const size_t paren_pos = stripped.find('(');
    const size_t eq_pos = stripped.find('=');
    if (paren_pos != std::string::npos && (eq_pos == std::string::npos || paren_pos < eq_pos))
      {
      if (eq_pos != std::string::npos)
        {
        source_defines += "#define " + stripped.substr(0, eq_pos) + " " + stripped.substr(eq_pos + 1) + "\n";
        }
      else
        {
        source_defines += "#define " + stripped + "\n";
        }
      }
    else
      {
      macros.push_back(stripped);
      }
    }


  macros.push_back("UWORD=" + uword_str());
  if (has_sizet64())
    {
    macros.push_back("COOT_USE_INT64=1");
    }

  std::string kernel_source = kernel_gen::generator<VULKAN_BACKEND, num, typename ProxyTypes::held_type...>::kernel_source();

  // Strip trailing null bytes, which shaderc treats as unexpected tokens.
  while (!kernel_source.empty() && kernel_source.back() == '\0')
    {
    kernel_source.pop_back();
    }

  if (!source_defines.empty())
    {
    const size_t concat_pos = kernel_source.find("#define COOT_CONCAT");
    if (concat_pos != std::string::npos)
      {
      const size_t insert_pos = kernel_source.find('\n', concat_pos);
      if (insert_pos != std::string::npos)
        {
        kernel_source.insert(insert_pos + 1, source_defines);
        }
      }
    }

  const size_t push_size = compute_gen_push_size<ProxyTypes...>();

  const uint32_t num_buffers = (uint32_t) (kernel_gen::vk_leaf_count<typename ProxyTypes::held_type>::value + ... + 0);

  return get_pipeline_from_source(name, kernel_source, macros, push_size, num_buffers);
  }



template<typename eT>
inline
std::string
runtime_t::generate_eye_kernel()
  {
  coot_debug_sigprint();

  const std::string et_str = is_float<eT>::value ? "float" : "double";
  const std::string uw = uword_str();

  std::string source =
      "#version 450\n"
      + uword_ext() +
      "\n"
      "layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;\n"
      "\n"
      "layout(set = 0, binding = 0, std430) buffer Buf0 { " + et_str + " data[]; } out_buf;\n"
      "\n"
      "layout(push_constant) uniform PushConsts {\n"
      "  " + uw + " out_offset;\n"
      "  " + uw + " n_rows;\n"
      "  " + uw + " n_cols;\n"
      "} pc;\n"
      "\n"
      "void main() {\n"
      "  uint row = gl_GlobalInvocationID.x;\n"
      "  uint col = gl_GlobalInvocationID.y;\n"
      "  if (row < pc.n_rows && col < pc.n_cols) {\n"
      "    uint idx = uint(pc.out_offset) + row + col * uint(pc.n_rows);\n"
      "    out_buf.data[idx] = (row == col) ? " + et_str + "(1) : " + et_str + "(0);\n"
      "  }\n"
      "}\n";

  return source;
  }



template<typename eT>
inline
std::string
runtime_t::generate_fill_kernel()
  {
  coot_debug_sigprint();

  const std::string et_str = is_float<eT>::value ? "float" : "double";
  const std::string uw = uword_str();

  std::string source =
      "#version 450\n"
      + uword_ext() +
      "\n"
      "layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;\n"
      "\n"
      "layout(set = 0, binding = 0, std430) buffer Buf0 { " + et_str + " data[]; } out_buf;\n"
      "layout(set = 0, binding = 1, std430) readonly buffer Buf1 { " + et_str + " data[]; } val_buf;\n"
      "\n"
      "layout(push_constant) uniform PushConsts {\n"
      "  " + uw + " out_offset;\n"
      "  " + uw + " n_rows;\n"
      "  " + uw + " n_cols;\n"
      "  " + uw + " M_n_rows;\n"
      "  " + uw + " n_slices;\n"
      "  " + uw + " M_n_elem_slice;\n"
      "  " + uw + " val_offset;\n"
      "} pc;\n"
      "\n"
      "void main() {\n"
      "  uint row = gl_GlobalInvocationID.x;\n"
      "  uint col = gl_GlobalInvocationID.y;\n"
      "  uint slice = gl_GlobalInvocationID.z;\n"
      "  if (row < pc.n_rows && col < pc.n_cols && slice < pc.n_slices) {\n"
      "    uint idx = uint(pc.out_offset) + row + col * uint(pc.M_n_rows) + slice * uint(pc.M_n_elem_slice);\n"
      "    out_buf.data[idx] = val_buf.data[uint(pc.val_offset)];\n"
      "  }\n"
      "}\n";

  return source;
  }



template<typename eT_out, typename eT_in>
inline
std::string
runtime_t::generate_copy_kernel()
  {
  coot_debug_sigprint();

  const std::string et_out_str = is_float<eT_out>::value ? "float" : "double";
  const std::string et_in_str  = is_float<eT_in>::value  ? "float" : "double";
  const std::string uw = uword_str();

  std::string source =
      "#version 450\n"
      + uword_ext() +
      "\n"
      "layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;\n"
      "\n"
      "layout(set = 0, binding = 0, std430) buffer Buf0 { " + et_out_str + " data[]; } out_buf;\n"
      "layout(set = 0, binding = 1, std430) readonly buffer Buf1 { " + et_in_str + " data[]; } in_buf;\n"
      "\n"
      "layout(push_constant) uniform PushConsts {\n"
      "  " + uw + " out_offset;\n"
      "  " + uw + " out_n_rows;\n"
      "  " + uw + " out_n_cols;\n"
      "  " + uw + " out_M_n_rows;\n"
      "  " + uw + " in_offset;\n"
      "  " + uw + " in_n_rows;\n"
      "  " + uw + " in_n_cols;\n"
      "  " + uw + " in_M_n_rows;\n"
      "  " + uw + " n_slices;\n"
      "  " + uw + " out_M_n_elem_slice;\n"
      "  " + uw + " in_M_n_elem_slice;\n"
      "} pc;\n"
      "\n"
      "void main() {\n"
      "  uint row = gl_GlobalInvocationID.x;\n"
      "  uint col = gl_GlobalInvocationID.y;\n"
      "  uint slice = gl_GlobalInvocationID.z;\n"
      "  if (row < pc.out_n_rows && col < pc.out_n_cols && slice < pc.n_slices) {\n"
      "    uint out_idx = uint(pc.out_offset) + row + col * uint(pc.out_M_n_rows) + slice * uint(pc.out_M_n_elem_slice);\n"
      "    uint in_idx = uint(pc.in_offset) + row + col * uint(pc.in_M_n_rows) + slice * uint(pc.in_M_n_elem_slice);\n"
      "    out_buf.data[out_idx] = " + et_out_str + "(in_buf.data[in_idx]);\n"
      "  }\n"
      "}\n";

  return source;
  }



template<typename eT>
inline
std::string
runtime_t::generate_copy_replace_kernel()
  {
  coot_debug_sigprint();

  const std::string et_str = is_float<eT>::value ? "float" : "double";
  const std::string uw = uword_str();

  std::string source =
      "#version 450\n"
      + uword_ext() +
      "\n"
      "layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;\n"
      "\n"
      "layout(set = 0, binding = 0, std430) buffer Buf0 { " + et_str + " data[]; } out_buf;\n"
      "layout(set = 0, binding = 1, std430) readonly buffer Buf1 { " + et_str + " data[]; } in_buf;\n"
      "\n"
      "layout(push_constant) uniform PushConsts {\n"
      "  " + uw + " out_offset;\n"
      "  " + uw + " out_n_rows;\n"
      "  " + uw + " out_n_cols;\n"
      "  " + uw + " out_M_n_rows;\n"
      "  " + uw + " in_offset;\n"
      "  " + uw + " in_n_rows;\n"
      "  " + uw + " in_n_cols;\n"
      "  " + uw + " in_M_n_rows;\n"
      "  " + uw + " n_slices;\n"
      "  " + uw + " out_M_n_elem_slice;\n"
      "  " + uw + " in_M_n_elem_slice;\n"
      "  " + et_str + " old_val;\n"
      "  " + et_str + " new_val;\n"
      "} pc;\n"
      "\n"
      "void main() {\n"
      "  uint row = gl_GlobalInvocationID.x;\n"
      "  uint col = gl_GlobalInvocationID.y;\n"
      "  uint slice = gl_GlobalInvocationID.z;\n"
      "  if (row < pc.out_n_rows && col < pc.out_n_cols && slice < pc.n_slices) {\n"
      "    uint out_idx = uint(pc.out_offset) + row + col * uint(pc.out_M_n_rows) + slice * uint(pc.out_M_n_elem_slice);\n"
      "    uint in_idx = uint(pc.in_offset) + row + col * uint(pc.in_M_n_rows) + slice * uint(pc.in_M_n_elem_slice);\n"
      "    " + et_str + " val = in_buf.data[in_idx];\n"
      "    out_buf.data[out_idx] = ((val == pc.old_val) || (isnan(val) && isnan(pc.old_val))) ? pc.new_val : val;\n"
      "  }\n"
      "}\n";

  return source;
  }



template<typename eT>
inline
pipeline_t&
runtime_t::get_gen_eye_pipeline()
  {
  coot_debug_sigprint();

  const std::string name =
      (is_float<eT>::value) ? "gen_eye_f32" : "gen_eye_f64";

  auto it = pipelines.find(name);
  if (it != pipelines.end())
    {
    return it->second;
    }

  const std::string source = generate_eye_kernel<eT>();
  return get_pipeline_from_source(name, source, push_size<eye_push_t>(), 1);
  }



template<typename eT>
inline
pipeline_t&
runtime_t::get_gen_fill_pipeline()
  {
  coot_debug_sigprint();

  const std::string name =
      (is_float<eT>::value) ? "gen_fill_f32" : "gen_fill_f64";

  auto it = pipelines.find(name);
  if (it != pipelines.end())
    {
    return it->second;
    }

  const std::string source = generate_fill_kernel<eT>();
  return get_pipeline_from_source(name, source, push_size<fill_push_t>(), 2);
  }



template<typename eT_out, typename eT_in>
inline
pipeline_t&
runtime_t::get_gen_copy_pipeline()
  {
  coot_debug_sigprint();

  const std::string out_prefix = (is_float<eT_out>::value) ? "f32" : "f64";
  const std::string in_prefix = (is_float<eT_in>::value) ? "f32" : "f64";
  const std::string name = "gen_copy_" + out_prefix + "_" + in_prefix;

  auto it = pipelines.find(name);
  if (it != pipelines.end())
    {
    return it->second;
    }

  const std::string source = generate_copy_kernel<eT_out, eT_in>();
  return get_pipeline_from_source(name, source, push_size<copy_push_t>(), 2);
  }



template<typename eT>
inline
pipeline_t&
runtime_t::get_gen_copy_replace_pipeline()
  {
  coot_debug_sigprint();

  const std::string name =
      (is_float<eT>::value) ? "gen_copy_replace_f32" : "gen_copy_replace_f64";

  auto it = pipelines.find(name);
  if (it != pipelines.end())
    {
    return it->second;
    }

  const std::string source = generate_copy_replace_kernel<eT>();
  const size_t ps = has_sizet64() ? sizeof(copy_replace_push_t<u64, eT>) : sizeof(copy_replace_push_t<u32, eT>);
  return get_pipeline_from_source(name, source, ps, 2);
  }



template<typename eT>
inline
std::string
runtime_t::generate_accu_kernel()
  {
  coot_debug_sigprint();

  const bool use_fp64 = is_double<eT>::value || has_fp64();
  const std::string et_str = is_float<eT>::value ? "float" : "double";
  const std::string uw = uword_str();

  std::string source =
      "#version 450\n";

  if (use_fp64)
    source += "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n";

  source += uword_ext();

  source +=
      "\n"
      "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\n"
      "\n"
      "layout(set = 0, binding = 0, std430) buffer OutBuf { " + et_str + " data[]; } out_buf;\n"
      "layout(set = 0, binding = 1, std430) readonly buffer InBuf { " + et_str + " data[]; } in_buf;\n"
      "layout(set = 0, binding = 2, std430) readonly buffer UnusedBuf { " + et_str + " data[]; } unused_buf;\n"
      "\n"
      "layout(push_constant) uniform PushConsts {\n"
      "  " + uw + " out_offset;\n"
      "  " + uw + " in_offset;\n"
      "  " + uw + " n_elem;\n"
      "} pc;\n"
      "\n"
      "void main() {\n"
      "  if (gl_GlobalInvocationID.x == 0u) {\n";

  if (is_float<eT>::value && !has_fp64())
    {
    source +=
        "    float sum = 0.0;\n"
        "    float c = 0.0;\n"
        "    for (uint i = 0; i < pc.n_elem; ++i) {\n"
        "      float x = in_buf.data[uint(pc.in_offset) + i];\n"
        "      float y = x - c;\n"
        "      float t = sum + y;\n"
        "      c = (t - sum) - y;\n"
        "      sum = t;\n"
        "    }\n"
        "    out_buf.data[uint(pc.out_offset)] = sum;\n";
    }
  else if (is_float<eT>::value)
    {
    source +=
        "    double sum = 0.0;\n"
        "    for (uint i = 0; i < pc.n_elem; ++i) {\n"
        "      sum += double(in_buf.data[uint(pc.in_offset) + i]);\n"
        "    }\n"
        "    out_buf.data[uint(pc.out_offset)] = float(sum);\n";
    }
  else
    {
    source +=
        "    double sum = 0.0;\n"
        "    for (uint i = 0; i < pc.n_elem; ++i) {\n"
        "      sum += in_buf.data[uint(pc.in_offset) + i];\n"
        "    }\n"
        "    out_buf.data[uint(pc.out_offset)] = sum;\n";
    }

  source +=
      "  }\n"
      "}\n";

  return source;
  }



template<typename eT>
inline
std::string
runtime_t::generate_accu_subview_kernel()
  {
  coot_debug_sigprint();

  const bool use_fp64 = is_double<eT>::value || has_fp64();
  const std::string et_str = is_float<eT>::value ? "float" : "double";
  const std::string uw = uword_str();

  std::string source =
      "#version 450\n";

  if (use_fp64)
    source += "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n";

  source += uword_ext();

  source +=
      "\n"
      "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\n"
      "\n"
      "layout(set = 0, binding = 0, std430) buffer OutBuf { " + et_str + " data[]; } out_buf;\n"
      "layout(set = 0, binding = 1, std430) readonly buffer InBuf { " + et_str + " data[]; } in_buf;\n"
      "layout(set = 0, binding = 2, std430) readonly buffer UnusedBuf { " + et_str + " data[]; } unused_buf;\n"
      "\n"
      "layout(push_constant) uniform PushConsts {\n"
      "  " + uw + " out_offset;\n"
      "  " + uw + " in_offset;\n"
      "  " + uw + " n_rows;\n"
      "  " + uw + " n_cols;\n"
      "  " + uw + " M_n_rows;\n"
      "} pc;\n"
      "\n"
      "void main() {\n"
      "  if (gl_GlobalInvocationID.x == 0u) {\n";

  if (is_float<eT>::value && !has_fp64())
    {
    source +=
        "    float sum = 0.0;\n"
        "    float c = 0.0;\n"
        "    for (uint col = 0; col < pc.n_cols; ++col) {\n"
        "      uint base = uint(pc.in_offset) + col * uint(pc.M_n_rows);\n"
        "      for (uint row = 0; row < pc.n_rows; ++row) {\n"
        "        float x = in_buf.data[base + row];\n"
        "        float y = x - c;\n"
        "        float t = sum + y;\n"
        "        c = (t - sum) - y;\n"
        "        sum = t;\n"
        "      }\n"
        "    }\n"
        "    out_buf.data[uint(pc.out_offset)] = sum;\n";
    }
  else if (is_float<eT>::value)
    {
    source +=
        "    double sum = 0.0;\n"
        "    for (uint col = 0; col < pc.n_cols; ++col) {\n"
        "      uint base = uint(pc.in_offset) + col * uint(pc.M_n_rows);\n"
        "      for (uint row = 0; row < pc.n_rows; ++row) {\n"
        "        sum += double(in_buf.data[base + row]);\n"
        "      }\n"
        "    }\n"
        "    out_buf.data[uint(pc.out_offset)] = float(sum);\n";
    }
  else
    {
    source +=
        "    double sum = 0.0;\n"
        "    for (uint col = 0; col < pc.n_cols; ++col) {\n"
        "      uint base = uint(pc.in_offset) + col * uint(pc.M_n_rows);\n"
        "      for (uint row = 0; row < pc.n_rows; ++row) {\n"
        "        sum += in_buf.data[base + row];\n"
        "      }\n"
        "    }\n"
        "    out_buf.data[uint(pc.out_offset)] = sum;\n";
    }

  source +=
      "  }\n"
      "}\n";

  return source;
  }



template<typename eT>
inline
std::string
runtime_t::generate_dot_kernel()
  {
  coot_debug_sigprint();

  const bool use_fp64 = is_double<eT>::value || has_fp64();
  const std::string et_str = is_float<eT>::value ? "float" : "double";
  const std::string uw = uword_str();

  std::string source =
      "#version 450\n";

  if (use_fp64)
    source += "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n";

  source += uword_ext();

  source +=
      "\n"
      "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\n"
      "\n"
      "layout(set = 0, binding = 0, std430) buffer OutBuf { " + et_str + " data[]; } out_buf;\n"
      "layout(set = 0, binding = 1, std430) readonly buffer In1Buf { " + et_str + " data[]; } in1_buf;\n"
      "layout(set = 0, binding = 2, std430) readonly buffer In2Buf { " + et_str + " data[]; } in2_buf;\n"
      "\n"
      "layout(push_constant) uniform PushConsts {\n"
      "  " + uw + " out_offset;\n"
      "  " + uw + " in1_offset;\n"
      "  " + uw + " in2_offset;\n"
      "  " + uw + " n_elem;\n"
      "} pc;\n"
      "\n"
      "void main() {\n"
      "  if (gl_GlobalInvocationID.x == 0u) {\n";

  if (is_float<eT>::value && has_fp64())
    {
    source +=
        "    double sum = 0.0;\n"
        "    for (uint i = 0; i < pc.n_elem; ++i) {\n"
        "      sum += double(in1_buf.data[uint(pc.in1_offset) + i]) * double(in2_buf.data[uint(pc.in2_offset) + i]);\n"
        "    }\n"
        "    out_buf.data[uint(pc.out_offset)] = float(sum);\n";
    }
  else if (is_float<eT>::value)
    {
    source +=
        "    float sum = 0.0;\n"
        "    for (uint i = 0; i < pc.n_elem; ++i) {\n"
        "      sum += in1_buf.data[uint(pc.in1_offset) + i] * in2_buf.data[uint(pc.in2_offset) + i];\n"
        "    }\n"
        "    out_buf.data[uint(pc.out_offset)] = sum;\n";
    }
  else
    {
    source +=
        "    double sum = 0.0;\n"
        "    for (uint i = 0; i < pc.n_elem; ++i) {\n"
        "      sum += in1_buf.data[uint(pc.in1_offset) + i] * in2_buf.data[uint(pc.in2_offset) + i];\n"
        "    }\n"
        "    out_buf.data[uint(pc.out_offset)] = sum;\n";
    }

  source +=
      "  }\n"
      "}\n";

  return source;
  }



template<typename eT>
inline
std::string
runtime_t::generate_rel_eq_scalar_kernel()
  {
  coot_debug_sigprint();

  const bool need_fp64 = is_double<eT>::value;
  const std::string et_str = is_float<eT>::value ? "float" : "double";
  const std::string uw = uword_str();

  std::string source =
      "#version 450\n";

  if (need_fp64)
    source += "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n";

  source += uword_ext();

  source +=
      "\n"
      "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n"
      "\n"
      "layout(set = 0, binding = 0, std430) buffer OutBuf { " + uw + " data[]; } out_buf;\n"
      "layout(set = 0, binding = 1, std430) readonly buffer InBuf { " + et_str + " data[]; } in_buf;\n"
      "layout(set = 0, binding = 2, std430) readonly buffer ValBuf { " + et_str + " data[]; } val_buf;\n"
      "\n"
      "layout(push_constant) uniform PushConsts {\n"
      "  " + uw + " out_offset;\n"
      "  " + uw + " in_offset;\n"
      "  " + uw + " n_elem;\n"
      "  " + uw + " val_offset;\n"
      "} pc;\n"
      "\n"
      "void main() {\n"
      "  uint idx = gl_GlobalInvocationID.x;\n"
      "  if (idx < pc.n_elem) {\n"
      "    " + et_str + " v = in_buf.data[uint(pc.in_offset) + idx];\n"
      "    " + uw + " res = (v == val_buf.data[uint(pc.val_offset)]) ? " + uw + "(1) : " + uw + "(0);\n"
      "    out_buf.data[uint(pc.out_offset) + idx] = res;\n"
      "  }\n"
      "}\n";

  return source;
  }



template<typename eT>
inline
std::string
runtime_t::generate_rel_neq_scalar_kernel()
  {
  coot_debug_sigprint();

  const bool need_fp64 = is_double<eT>::value;
  const std::string et_str = is_float<eT>::value ? "float" : "double";
  const std::string uw = uword_str();

  std::string source = "#version 450\n";
  if (need_fp64)
    source += "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n";
  source += uword_ext();

  source +=
      "\n"
      "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n"
      "\n"
      "layout(set = 0, binding = 0, std430) buffer OutBuf { " + uw + " data[]; } out_buf;\n"
      "layout(set = 0, binding = 1, std430) readonly buffer InBuf { " + et_str + " data[]; } in_buf;\n"
      "layout(set = 0, binding = 2, std430) readonly buffer ValBuf { " + et_str + " data[]; } val_buf;\n"
      "\n"
      "layout(push_constant) uniform PushConsts {\n"
      "  " + uw + " out_offset;\n"
      "  " + uw + " in_offset;\n"
      "  " + uw + " n_elem;\n"
      "  " + uw + " val_offset;\n"
      "} pc;\n"
      "\n"
      "void main() {\n"
      "  uint idx = gl_GlobalInvocationID.x;\n"
      "  if (idx < pc.n_elem) {\n"
      "    " + et_str + " v = in_buf.data[uint(pc.in_offset) + idx];\n"
      "    " + uw + " res = (v != val_buf.data[uint(pc.val_offset)]) ? " + uw + "(1) : " + uw + "(0);\n"
      "    out_buf.data[uint(pc.out_offset) + idx] = res;\n"
      "  }\n"
      "}\n";

  return source;
  }



template<typename eT>
inline
std::string
runtime_t::generate_rel_gt_scalar_kernel()
  {
  coot_debug_sigprint();

  const bool need_fp64 = is_double<eT>::value;
  const std::string et_str = is_float<eT>::value ? "float" : "double";
  const std::string uw = uword_str();

  std::string source = "#version 450\n";
  if (need_fp64)
    source += "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n";
  source += uword_ext();

  source +=
      "\n"
      "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n"
      "\n"
      "layout(set = 0, binding = 0, std430) buffer OutBuf { " + uw + " data[]; } out_buf;\n"
      "layout(set = 0, binding = 1, std430) readonly buffer InBuf { " + et_str + " data[]; } in_buf;\n"
      "layout(set = 0, binding = 2, std430) readonly buffer ValBuf { " + et_str + " data[]; } val_buf;\n"
      "\n"
      "layout(push_constant) uniform PushConsts {\n"
      "  " + uw + " out_offset;\n"
      "  " + uw + " in_offset;\n"
      "  " + uw + " n_elem;\n"
      "  " + uw + " val_offset;\n"
      "} pc;\n"
      "\n"
      "void main() {\n"
      "  uint idx = gl_GlobalInvocationID.x;\n"
      "  if (idx < pc.n_elem) {\n"
      "    " + et_str + " v = in_buf.data[uint(pc.in_offset) + idx];\n"
      "    " + uw + " res = (v > val_buf.data[uint(pc.val_offset)]) ? " + uw + "(1) : " + uw + "(0);\n"
      "    out_buf.data[uint(pc.out_offset) + idx] = res;\n"
      "  }\n"
      "}\n";

  return source;
  }



template<typename eT>
inline
std::string
runtime_t::generate_rel_lt_scalar_kernel()
  {
  coot_debug_sigprint();

  const bool need_fp64 = is_double<eT>::value;
  const std::string et_str = is_float<eT>::value ? "float" : "double";
  const std::string uw = uword_str();

  std::string source = "#version 450\n";
  if (need_fp64)
    source += "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n";
  source += uword_ext();

  source +=
      "\n"
      "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n"
      "\n"
      "layout(set = 0, binding = 0, std430) buffer OutBuf { " + uw + " data[]; } out_buf;\n"
      "layout(set = 0, binding = 1, std430) readonly buffer InBuf { " + et_str + " data[]; } in_buf;\n"
      "layout(set = 0, binding = 2, std430) readonly buffer ValBuf { " + et_str + " data[]; } val_buf;\n"
      "\n"
      "layout(push_constant) uniform PushConsts {\n"
      "  " + uw + " out_offset;\n"
      "  " + uw + " in_offset;\n"
      "  " + uw + " n_elem;\n"
      "  " + uw + " val_offset;\n"
      "} pc;\n"
      "\n"
      "void main() {\n"
      "  uint idx = gl_GlobalInvocationID.x;\n"
      "  if (idx < pc.n_elem) {\n"
      "    " + et_str + " v = in_buf.data[uint(pc.in_offset) + idx];\n"
      "    " + uw + " res = (v < val_buf.data[uint(pc.val_offset)]) ? " + uw + "(1) : " + uw + "(0);\n"
      "    out_buf.data[uint(pc.out_offset) + idx] = res;\n"
      "  }\n"
      "}\n";

  return source;
  }



template<typename eT>
inline
std::string
runtime_t::generate_rel_gteq_scalar_kernel()
  {
  coot_debug_sigprint();

  const bool need_fp64 = is_double<eT>::value;
  const std::string et_str = is_float<eT>::value ? "float" : "double";
  const std::string uw = uword_str();

  std::string source = "#version 450\n";
  if (need_fp64)
    source += "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n";
  source += uword_ext();

  source +=
      "\n"
      "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n"
      "\n"
      "layout(set = 0, binding = 0, std430) buffer OutBuf { " + uw + " data[]; } out_buf;\n"
      "layout(set = 0, binding = 1, std430) readonly buffer InBuf { " + et_str + " data[]; } in_buf;\n"
      "layout(set = 0, binding = 2, std430) readonly buffer ValBuf { " + et_str + " data[]; } val_buf;\n"
      "\n"
      "layout(push_constant) uniform PushConsts {\n"
      "  " + uw + " out_offset;\n"
      "  " + uw + " in_offset;\n"
      "  " + uw + " n_elem;\n"
      "  " + uw + " val_offset;\n"
      "} pc;\n"
      "\n"
      "void main() {\n"
      "  uint idx = gl_GlobalInvocationID.x;\n"
      "  if (idx < pc.n_elem) {\n"
      "    " + et_str + " v = in_buf.data[uint(pc.in_offset) + idx];\n"
      "    " + uw + " res = (v >= val_buf.data[uint(pc.val_offset)]) ? " + uw + "(1) : " + uw + "(0);\n"
      "    out_buf.data[uint(pc.out_offset) + idx] = res;\n"
      "  }\n"
      "}\n";

  return source;
  }



template<typename eT>
inline
std::string
runtime_t::generate_rel_lteq_scalar_kernel()
  {
  coot_debug_sigprint();

  const bool need_fp64 = is_double<eT>::value;
  const std::string et_str = is_float<eT>::value ? "float" : "double";
  const std::string uw = uword_str();

  std::string source = "#version 450\n";
  if (need_fp64)
    source += "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n";
  source += uword_ext();

  source +=
      "\n"
      "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n"
      "\n"
      "layout(set = 0, binding = 0, std430) buffer OutBuf { " + uw + " data[]; } out_buf;\n"
      "layout(set = 0, binding = 1, std430) readonly buffer InBuf { " + et_str + " data[]; } in_buf;\n"
      "layout(set = 0, binding = 2, std430) readonly buffer ValBuf { " + et_str + " data[]; } val_buf;\n"
      "\n"
      "layout(push_constant) uniform PushConsts {\n"
      "  " + uw + " out_offset;\n"
      "  " + uw + " in_offset;\n"
      "  " + uw + " n_elem;\n"
      "  " + uw + " val_offset;\n"
      "} pc;\n"
      "\n"
      "void main() {\n"
      "  uint idx = gl_GlobalInvocationID.x;\n"
      "  if (idx < pc.n_elem) {\n"
      "    " + et_str + " v = in_buf.data[uint(pc.in_offset) + idx];\n"
      "    " + uw + " res = (v <= val_buf.data[uint(pc.val_offset)]) ? " + uw + "(1) : " + uw + "(0);\n"
      "    out_buf.data[uint(pc.out_offset) + idx] = res;\n"
      "  }\n"
      "}\n";

  return source;
  }



template<typename eT>
inline
std::string
runtime_t::generate_all_neq_vec_kernel()
  {
  coot_debug_sigprint();

  const bool need_fp64 = is_double<eT>::value;
  const std::string uw = uword_str();
  const std::string et_str = is_float<eT>::value ? "float" : (is_double<eT>::value ? "double" : uw);

  std::string source =
      "#version 450\n";

  if (need_fp64)
    source += "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n";

  source += uword_ext();

  source +=
      "\n"
      "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\n"
      "\n"
      "layout(set = 0, binding = 0, std430) buffer OutBuf { " + uw + " data[]; } out_buf;\n"
      "layout(set = 0, binding = 1, std430) readonly buffer InBuf { " + et_str + " data[]; } in_buf;\n"
      "layout(set = 0, binding = 2, std430) readonly buffer ValBuf { " + et_str + " data[]; } val_buf;\n"
      "\n"
      "layout(push_constant) uniform PushConsts {\n"
      "  " + uw + " out_offset;\n"
      "  " + uw + " in_offset;\n"
      "  " + uw + " n_elem;\n"
      "  " + uw + " val_offset;\n"
      "} pc;\n"
      "\n"
      "void main() {\n"
      "  if (gl_GlobalInvocationID.x == 0u) {\n"
      "    " + uw + " result = " + uw + "(1);\n"
      "    for (uint i = 0; i < pc.n_elem; ++i) {\n"
      "      " + et_str + " v = in_buf.data[uint(pc.in_offset) + i];\n"
      "      if (v == val_buf.data[uint(pc.val_offset)]) {\n"
      "        result = " + uw + "(0);\n"
      "        break;\n"
      "      }\n"
      "    }\n"
      "    out_buf.data[uint(pc.out_offset)] = result;\n"
      "  }\n"
      "}\n";

  return source;
  }



template<typename eT>
inline
std::string
runtime_t::generate_all_neq_colwise_kernel()
  {
  coot_debug_sigprint();

  const bool need_fp64 = is_double<eT>::value;
  const std::string uw = uword_str();
  const std::string et_str = is_float<eT>::value ? "float" : (is_double<eT>::value ? "double" : uw);

  std::string source =
      "#version 450\n";

  if (need_fp64)
    source += "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n";

  source += uword_ext();

  source +=
      "\n"
      "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n"
      "\n"
      "layout(set = 0, binding = 0, std430) buffer OutBuf { " + uw + " data[]; } out_buf;\n"
      "layout(set = 0, binding = 1, std430) readonly buffer InBuf { " + et_str + " data[]; } in_buf;\n"
      "layout(set = 0, binding = 2, std430) readonly buffer ValBuf { " + et_str + " data[]; } val_buf;\n"
      "\n"
      "layout(push_constant) uniform PushConsts {\n"
      "  " + uw + " out_offset;\n"
      "  " + uw + " in_offset;\n"
      "  " + uw + " n_rows;\n"
      "  " + uw + " n_cols;\n"
      "  " + uw + " val_offset;\n"
      "} pc;\n"
      "\n"
      "void main() {\n"
      "  uint col = gl_GlobalInvocationID.x;\n"
      "  if (col < pc.n_cols) {\n"
      "    " + uw + " result = " + uw + "(1);\n"
      "    uint base = uint(pc.in_offset) + col * uint(pc.n_rows);\n"
      "    for (uint row = 0; row < pc.n_rows; ++row) {\n"
      "      " + et_str + " v = in_buf.data[base + row];\n"
      "      if (v == val_buf.data[uint(pc.val_offset)]) {\n"
      "        result = " + uw + "(0);\n"
      "        break;\n"
      "      }\n"
      "    }\n"
      "    out_buf.data[uint(pc.out_offset) + col] = result;\n"
      "  }\n"
      "}\n";

  return source;
  }



template<typename eT>
inline
std::string
runtime_t::generate_all_neq_rowwise_kernel()
  {
  coot_debug_sigprint();

  const bool need_fp64 = is_double<eT>::value;
  const std::string uw = uword_str();
  const std::string et_str = is_float<eT>::value ? "float" : (is_double<eT>::value ? "double" : uw);

  std::string source =
      "#version 450\n";

  if (need_fp64)
    source += "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n";

  source += uword_ext();

  source +=
      "\n"
      "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n"
      "\n"
      "layout(set = 0, binding = 0, std430) buffer OutBuf { " + uw + " data[]; } out_buf;\n"
      "layout(set = 0, binding = 1, std430) readonly buffer InBuf { " + et_str + " data[]; } in_buf;\n"
      "layout(set = 0, binding = 2, std430) readonly buffer ValBuf { " + et_str + " data[]; } val_buf;\n"
      "\n"
      "layout(push_constant) uniform PushConsts {\n"
      "  " + uw + " out_offset;\n"
      "  " + uw + " in_offset;\n"
      "  " + uw + " n_rows;\n"
      "  " + uw + " n_cols;\n"
      "  " + uw + " val_offset;\n"
      "} pc;\n"
      "\n"
      "void main() {\n"
      "  uint row = gl_GlobalInvocationID.x;\n"
      "  if (row < pc.n_rows) {\n"
      "    " + uw + " result = " + uw + "(1);\n"
      "    for (uint col = 0; col < pc.n_cols; ++col) {\n"
      "      uint idx = uint(pc.in_offset) + row + col * uint(pc.n_rows);\n"
      "      " + et_str + " v = in_buf.data[idx];\n"
      "      if (v == val_buf.data[uint(pc.val_offset)]) {\n"
      "        result = " + uw + "(0);\n"
      "        break;\n"
      "      }\n"
      "    }\n"
      "    out_buf.data[uint(pc.out_offset) + row] = result;\n"
      "  }\n"
      "}\n";

  return source;
  }



template<typename eT>
inline
pipeline_t&
runtime_t::get_gen_accu_pipeline()
  {
  coot_debug_sigprint();

  const std::string name = (is_float<eT>::value) ? (has_fp64() ? "gen_accu_f32_f64" : "gen_accu_f32") : "gen_accu_f64";

  auto it = pipelines.find(name);
  if (it != pipelines.end())
    {
    return it->second;
    }

  const std::string source = generate_accu_kernel<eT>();
  return get_pipeline_from_source(name, source, push_size<accu_push_t>());
  }



template<typename eT>
inline
pipeline_t&
runtime_t::get_gen_accu_subview_pipeline()
  {
  coot_debug_sigprint();

  const std::string name = (is_float<eT>::value) ? (has_fp64() ? "gen_accu_subview_f32_f64" : "gen_accu_subview_f32") : "gen_accu_subview_f64";

  auto it = pipelines.find(name);
  if (it != pipelines.end())
    {
    return it->second;
    }

  const std::string source = generate_accu_subview_kernel<eT>();
  return get_pipeline_from_source(name, source, push_size<accu_subview_push_t>());
  }



template<typename eT>
inline
pipeline_t&
runtime_t::get_gen_dot_pipeline()
  {
  coot_debug_sigprint();

  const std::string name =
      (is_float<eT>::value) ? (has_fp64() ? "gen_dot_f32_f64" : "gen_dot_f32") : "gen_dot_f64";

  auto it = pipelines.find(name);
  if (it != pipelines.end())
    {
    return it->second;
    }

  const std::string source = generate_dot_kernel<eT>();
  return get_pipeline_from_source(name, source, push_size<dot_push_t>());
  }



template<typename eT>
inline
pipeline_t&
runtime_t::get_gen_rel_eq_scalar_pipeline()
  {
  coot_debug_sigprint();

  const std::string name = (is_float<eT>::value) ? "gen_rel_eq_scalar_f32" : "gen_rel_eq_scalar_f64";

  auto it = pipelines.find(name);
  if (it != pipelines.end())
    {
    return it->second;
    }

  const std::string source = generate_rel_eq_scalar_kernel<eT>();
  return get_pipeline_from_source(name, source, push_size<rel_scalar_push_t>());
  }



template<typename eT>
inline
pipeline_t&
runtime_t::get_gen_rel_neq_scalar_pipeline()
  {
  coot_debug_sigprint();

  const std::string name = (is_float<eT>::value) ? "gen_rel_neq_scalar_f32" : "gen_rel_neq_scalar_f64";

  auto it = pipelines.find(name);
  if (it != pipelines.end())
    return it->second;

  const std::string source = generate_rel_neq_scalar_kernel<eT>();
  return get_pipeline_from_source(name, source, push_size<rel_scalar_push_t>());
  }



template<typename eT>
inline
pipeline_t&
runtime_t::get_gen_rel_gt_scalar_pipeline()
  {
  coot_debug_sigprint();

  const std::string name = (is_float<eT>::value) ? "gen_rel_gt_scalar_f32" : "gen_rel_gt_scalar_f64";

  auto it = pipelines.find(name);
  if (it != pipelines.end())
    return it->second;

  const std::string source = generate_rel_gt_scalar_kernel<eT>();
  return get_pipeline_from_source(name, source, push_size<rel_scalar_push_t>());
  }



template<typename eT>
inline
pipeline_t&
runtime_t::get_gen_rel_lt_scalar_pipeline()
  {
  coot_debug_sigprint();

  const std::string name = (is_float<eT>::value) ? "gen_rel_lt_scalar_f32" : "gen_rel_lt_scalar_f64";

  auto it = pipelines.find(name);
  if (it != pipelines.end())
    return it->second;

  const std::string source = generate_rel_lt_scalar_kernel<eT>();
  return get_pipeline_from_source(name, source, push_size<rel_scalar_push_t>());
  }



template<typename eT>
inline
pipeline_t&
runtime_t::get_gen_rel_gteq_scalar_pipeline()
  {
  coot_debug_sigprint();

  const std::string name = (is_float<eT>::value) ? "gen_rel_gteq_scalar_f32" : "gen_rel_gteq_scalar_f64";

  auto it = pipelines.find(name);
  if (it != pipelines.end())
    return it->second;

  const std::string source = generate_rel_gteq_scalar_kernel<eT>();
  return get_pipeline_from_source(name, source, push_size<rel_scalar_push_t>());
  }



template<typename eT>
inline
pipeline_t&
runtime_t::get_gen_rel_lteq_scalar_pipeline()
  {
  coot_debug_sigprint();

  const std::string name = (is_float<eT>::value) ? "gen_rel_lteq_scalar_f32" : "gen_rel_lteq_scalar_f64";

  auto it = pipelines.find(name);
  if (it != pipelines.end())
    return it->second;

  const std::string source = generate_rel_lteq_scalar_kernel<eT>();
  return get_pipeline_from_source(name, source, push_size<rel_scalar_push_t>());
  }



template<typename eT>
inline
pipeline_t&
runtime_t::get_gen_all_neq_vec_pipeline()
  {
  coot_debug_sigprint();

  const std::string name = (is_uword<eT>::value) ? "gen_all_neq_vec_u64" : (is_float<eT>::value) ? "gen_all_neq_vec_f32" : "gen_all_neq_vec_f64";

  auto it = pipelines.find(name);
  if (it != pipelines.end())
    {
    return it->second;
    }

  const std::string source = generate_all_neq_vec_kernel<eT>();
  return get_pipeline_from_source(name, source, push_size<all_vec_push_t>());
  }



template<typename eT>
inline
pipeline_t&
runtime_t::get_gen_all_neq_pipeline(const bool colwise)
  {
  coot_debug_sigprint();

  const std::string type_str =
      (is_uword<eT>::value) ? "u64" : (is_float<eT>::value) ? "f32" : "f64";
  const std::string dir_str = colwise ? "colwise" : "rowwise";
  const std::string name = "gen_all_neq_" + dir_str + "_" + type_str;

  auto it = pipelines.find(name);
  if (it != pipelines.end())
    {
    return it->second;
    }

  const std::string source = colwise ? generate_all_neq_colwise_kernel<eT>() : generate_all_neq_rowwise_kernel<eT>();
  return get_pipeline_from_source(name, source, push_size<all_push_t>());
  }



template<typename eT>
inline
std::string
runtime_t::generate_min_vec_kernel()
  {
  coot_debug_sigprint();

  const std::string et_str = is_float<eT>::value ? "float" : "double";
  const std::string uw = uword_str();

  std::string source =
      "#version 450\n";

  if (is_double<eT>::value)
    source += "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n";

  source += uword_ext();

  source +=
      "\n"
      "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\n"
      "\n"
      "layout(set = 0, binding = 0, std430) buffer OutBuf { " + et_str + " data[]; } out_buf;\n"
      "layout(set = 0, binding = 1, std430) readonly buffer InBuf { " + et_str + " data[]; } in_buf;\n"
      "layout(set = 0, binding = 2, std430) readonly buffer UnusedBuf { " + et_str + " data[]; } unused_buf;\n"
      "\n"
      "layout(push_constant) uniform PushConsts {\n"
      "  " + uw + " out_offset;\n"
      "  " + uw + " in_offset;\n"
      "  " + uw + " n_elem;\n"
      "} pc;\n"
      "\n"
      "void main() {\n"
      "  if (gl_GlobalInvocationID.x == 0u) {\n"
      "    " + et_str + " val = in_buf.data[uint(pc.in_offset)];\n"
      "    for (uint i = 1; i < pc.n_elem; ++i) {\n"
      "      val = min(val, in_buf.data[uint(pc.in_offset) + i]);\n"
      "    }\n"
      "    out_buf.data[uint(pc.out_offset)] = val;\n"
      "  }\n"
      "}\n";

  return source;
  }



template<typename eT>
inline
std::string
runtime_t::generate_max_vec_kernel()
  {
  coot_debug_sigprint();

  const std::string et_str = is_float<eT>::value ? "float" : "double";
  const std::string uw = uword_str();

  std::string source =
      "#version 450\n";

  if (is_double<eT>::value)
    source += "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n";

  source += uword_ext();

  source +=
      "\n"
      "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\n"
      "\n"
      "layout(set = 0, binding = 0, std430) buffer OutBuf { " + et_str + " data[]; } out_buf;\n"
      "layout(set = 0, binding = 1, std430) readonly buffer InBuf { " + et_str + " data[]; } in_buf;\n"
      "layout(set = 0, binding = 2, std430) readonly buffer UnusedBuf { " + et_str + " data[]; } unused_buf;\n"
      "\n"
      "layout(push_constant) uniform PushConsts {\n"
      "  " + uw + " out_offset;\n"
      "  " + uw + " in_offset;\n"
      "  " + uw + " n_elem;\n"
      "} pc;\n"
      "\n"
      "void main() {\n"
      "  if (gl_GlobalInvocationID.x == 0u) {\n"
      "    " + et_str + " val = in_buf.data[uint(pc.in_offset)];\n"
      "    for (uint i = 1; i < pc.n_elem; ++i) {\n"
      "      val = max(val, in_buf.data[uint(pc.in_offset) + i]);\n"
      "    }\n"
      "    out_buf.data[uint(pc.out_offset)] = val;\n"
      "  }\n"
      "}\n";

  return source;
  }



template<typename eT>
inline
std::string
runtime_t::generate_min_colwise_kernel()
  {
  coot_debug_sigprint();

  const std::string et_str = is_float<eT>::value ? "float" : "double";
  const std::string uw = uword_str();

  std::string source =
      "#version 450\n";

  if (is_double<eT>::value)
    source += "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n";

  source += uword_ext();

  source +=
      "\n"
      "layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;\n"
      "\n"
      "layout(set = 0, binding = 0, std430) buffer OutBuf { " + et_str + " data[]; } out_buf;\n"
      "layout(set = 0, binding = 1, std430) readonly buffer InBuf { " + et_str + " data[]; } in_buf;\n"
      "\n"
      "layout(push_constant) uniform PushConsts {\n"
      "  " + uw + " dest_offset;\n"
      "  " + uw + " src_offset;\n"
      "  " + uw + " n_rows;\n"
      "  " + uw + " n_cols;\n"
      "  " + uw + " dest_mem_incr;\n"
      "  " + uw + " src_M_n_rows;\n"
      "} pc;\n"
      "\n"
      "void main() {\n"
      "  uint col = gl_GlobalInvocationID.x;\n"
      "  if (col < pc.n_cols) {\n"
      "    " + et_str + " val = in_buf.data[uint(pc.src_offset + col * pc.src_M_n_rows)];\n"
      "    for (uint row = 1; row < pc.n_rows; ++row) {\n"
      "      val = min(val, in_buf.data[uint(pc.src_offset + row + col * pc.src_M_n_rows)]);\n"
      "    }\n"
      "    out_buf.data[uint(pc.dest_offset + col * pc.dest_mem_incr)] = val;\n"
      "  }\n"
      "}\n";

  return source;
  }



template<typename eT>
inline
std::string
runtime_t::generate_min_rowwise_kernel()
  {
  coot_debug_sigprint();

  const std::string et_str = is_float<eT>::value ? "float" : "double";
  const std::string uw = uword_str();

  std::string source =
      "#version 450\n";

  if (is_double<eT>::value)
    source += "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n";

  source += uword_ext();

  source +=
      "\n"
      "layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;\n"
      "\n"
      "layout(set = 0, binding = 0, std430) buffer OutBuf { " + et_str + " data[]; } out_buf;\n"
      "layout(set = 0, binding = 1, std430) readonly buffer InBuf { " + et_str + " data[]; } in_buf;\n"
      "\n"
      "layout(push_constant) uniform PushConsts {\n"
      "  " + uw + " dest_offset;\n"
      "  " + uw + " src_offset;\n"
      "  " + uw + " n_rows;\n"
      "  " + uw + " n_cols;\n"
      "  " + uw + " dest_mem_incr;\n"
      "  " + uw + " src_M_n_rows;\n"
      "} pc;\n"
      "\n"
      "void main() {\n"
      "  uint row = gl_GlobalInvocationID.x;\n"
      "  if (row < pc.n_rows) {\n"
      "    " + et_str + " val = in_buf.data[uint(pc.src_offset + row)];\n"
      "    for (uint col = 1; col < pc.n_cols; ++col) {\n"
      "      val = min(val, in_buf.data[uint(pc.src_offset + row + col * pc.src_M_n_rows)]);\n"
      "    }\n"
      "    out_buf.data[uint(pc.dest_offset + row * pc.dest_mem_incr)] = val;\n"
      "  }\n"
      "}\n";

  return source;
  }



template<typename eT>
inline
std::string
runtime_t::generate_max_colwise_kernel()
  {
  coot_debug_sigprint();

  const std::string et_str = is_float<eT>::value ? "float" : "double";
  const std::string uw = uword_str();

  std::string source =
      "#version 450\n";

  if (is_double<eT>::value)
    source += "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n";

  source += uword_ext();

  source +=
      "\n"
      "layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;\n"
      "\n"
      "layout(set = 0, binding = 0, std430) buffer OutBuf { " + et_str + " data[]; } out_buf;\n"
      "layout(set = 0, binding = 1, std430) readonly buffer InBuf { " + et_str + " data[]; } in_buf;\n"
      "\n"
      "layout(push_constant) uniform PushConsts {\n"
      "  " + uw + " dest_offset;\n"
      "  " + uw + " src_offset;\n"
      "  " + uw + " n_rows;\n"
      "  " + uw + " n_cols;\n"
      "  " + uw + " dest_mem_incr;\n"
      "  " + uw + " src_M_n_rows;\n"
      "} pc;\n"
      "\n"
      "void main() {\n"
      "  uint col = gl_GlobalInvocationID.x;\n"
      "  if (col < pc.n_cols) {\n"
      "    " + et_str + " val = in_buf.data[uint(pc.src_offset + col * pc.src_M_n_rows)];\n"
      "    for (uint row = 1; row < pc.n_rows; ++row) {\n"
      "      val = max(val, in_buf.data[uint(pc.src_offset + row + col * pc.src_M_n_rows)]);\n"
      "    }\n"
      "    out_buf.data[uint(pc.dest_offset + col * pc.dest_mem_incr)] = val;\n"
      "  }\n"
      "}\n";

  return source;
  }



template<typename eT>
inline
std::string
runtime_t::generate_max_rowwise_kernel()
  {
  coot_debug_sigprint();

  const std::string et_str = is_float<eT>::value ? "float" : "double";
  const std::string uw = uword_str();

  std::string source =
      "#version 450\n";

  if (is_double<eT>::value)
    source += "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n";

  source += uword_ext();

  source +=
      "\n"
      "layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;\n"
      "\n"
      "layout(set = 0, binding = 0, std430) buffer OutBuf { " + et_str + " data[]; } out_buf;\n"
      "layout(set = 0, binding = 1, std430) readonly buffer InBuf { " + et_str + " data[]; } in_buf;\n"
      "\n"
      "layout(push_constant) uniform PushConsts {\n"
      "  " + uw + " dest_offset;\n"
      "  " + uw + " src_offset;\n"
      "  " + uw + " n_rows;\n"
      "  " + uw + " n_cols;\n"
      "  " + uw + " dest_mem_incr;\n"
      "  " + uw + " src_M_n_rows;\n"
      "} pc;\n"
      "\n"
      "void main() {\n"
      "  uint row = gl_GlobalInvocationID.x;\n"
      "  if (row < pc.n_rows) {\n"
      "    " + et_str + " val = in_buf.data[uint(pc.src_offset + row)];\n"
      "    for (uint col = 1; col < pc.n_cols; ++col) {\n"
      "      val = max(val, in_buf.data[uint(pc.src_offset + row + col * pc.src_M_n_rows)]);\n"
      "    }\n"
      "    out_buf.data[uint(pc.dest_offset + row * pc.dest_mem_incr)] = val;\n"
      "  }\n"
      "}\n";

  return source;
  }



template<typename eT>
inline
pipeline_t&
runtime_t::get_gen_min_vec_pipeline()
  {
  coot_debug_sigprint();

  const std::string name = is_float<eT>::value ? "gen_min_vec_f32" : "gen_min_vec_f64";

  auto it = pipelines.find(name);
  if (it != pipelines.end())
    {
    return it->second;
    }

  const std::string source = generate_min_vec_kernel<eT>();
  return get_pipeline_from_source(name, source, push_size<accu_push_t>());
  }



template<typename eT>
inline
pipeline_t&
runtime_t::get_gen_max_vec_pipeline()
  {
  coot_debug_sigprint();

  const std::string name = is_float<eT>::value ? "gen_max_vec_f32" : "gen_max_vec_f64";

  auto it = pipelines.find(name);
  if (it != pipelines.end())
    {
    return it->second;
    }

  const std::string source = generate_max_vec_kernel<eT>();
  return get_pipeline_from_source(name, source, push_size<accu_push_t>());
  }



template<typename eT>
inline
pipeline_t&
runtime_t::get_gen_min_colwise_pipeline()
  {
  coot_debug_sigprint();

  const std::string name = is_float<eT>::value ? "gen_min_colwise_f32" : "gen_min_colwise_f64";

  auto it = pipelines.find(name);
  if (it != pipelines.end())
    {
    return it->second;
    }

  const std::string source = generate_min_colwise_kernel<eT>();
  return get_pipeline_from_source(name, source, push_size<reduce_dim_push_t>(), 2);
  }



template<typename eT>
inline
pipeline_t&
runtime_t::get_gen_min_rowwise_pipeline()
  {
  coot_debug_sigprint();

  const std::string name = is_float<eT>::value ? "gen_min_rowwise_f32" : "gen_min_rowwise_f64";

  auto it = pipelines.find(name);
  if (it != pipelines.end())
    {
    return it->second;
    }

  const std::string source = generate_min_rowwise_kernel<eT>();
  return get_pipeline_from_source(name, source, push_size<reduce_dim_push_t>(), 2);
  }



template<typename eT>
inline
pipeline_t&
runtime_t::get_gen_max_colwise_pipeline()
  {
  coot_debug_sigprint();

  const std::string name = is_float<eT>::value ? "gen_max_colwise_f32" : "gen_max_colwise_f64";

  auto it = pipelines.find(name);
  if (it != pipelines.end())
    {
    return it->second;
    }

  const std::string source = generate_max_colwise_kernel<eT>();
  return get_pipeline_from_source(name, source, push_size<reduce_dim_push_t>(), 2);
  }



template<typename eT>
inline
pipeline_t&
runtime_t::get_gen_max_rowwise_pipeline()
  {
  coot_debug_sigprint();

  const std::string name = is_float<eT>::value ? "gen_max_rowwise_f32" : "gen_max_rowwise_f64";

  auto it = pipelines.find(name);
  if (it != pipelines.end())
    {
    return it->second;
    }

  const std::string source = generate_max_rowwise_kernel<eT>();
  return get_pipeline_from_source(name, source, push_size<reduce_dim_push_t>(), 2);
  }



template<typename eT>
inline
std::string
runtime_t::generate_rel_isfinite_kernel()
  {
  coot_debug_sigprint();
  const std::string et = is_float<eT>::value ? "float" : "double";
  const std::string uw = uword_str();
  std::string src = "#version 450\n";
  if (is_double<eT>::value) src += "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n";
  src += uword_ext();
  src +=
    "\nlayout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n"
    "layout(set=0,binding=0,std430) buffer OutBuf { " + uw + " data[]; } out_buf;\n"
    "layout(set=0,binding=1,std430) readonly buffer InBuf { " + et + " data[]; } in_buf;\n"
    "layout(push_constant) uniform PC { " + uw + " out_offset; " + uw + " in_offset; " + uw + " n_elem; } pc;\n"
    "void main() {\n"
    "  uint idx = gl_GlobalInvocationID.x;\n"
    "  if (idx < pc.n_elem) {\n"
    "    " + et + " v = in_buf.data[uint(pc.in_offset) + idx];\n"
    "    out_buf.data[uint(pc.out_offset) + idx] = (!isinf(v) && !isnan(v)) ? " + uw + "(1) : " + uw + "(0);\n"
    "  }\n"
    "}\n";
  return src;
  }



template<typename eT>
inline
std::string
runtime_t::generate_rel_isinf_kernel()
  {
  coot_debug_sigprint();
  const std::string et = is_float<eT>::value ? "float" : "double";
  const std::string uw = uword_str();
  std::string src = "#version 450\n";
  if (is_double<eT>::value) src += "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n";
  src += uword_ext();
  src +=
    "\nlayout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n"
    "layout(set=0,binding=0,std430) buffer OutBuf { " + uw + " data[]; } out_buf;\n"
    "layout(set=0,binding=1,std430) readonly buffer InBuf { " + et + " data[]; } in_buf;\n"
    "layout(push_constant) uniform PC { " + uw + " out_offset; " + uw + " in_offset; " + uw + " n_elem; } pc;\n"
    "void main() {\n"
    "  uint idx = gl_GlobalInvocationID.x;\n"
    "  if (idx < pc.n_elem) {\n"
    "    " + et + " v = in_buf.data[uint(pc.in_offset) + idx];\n"
    "    out_buf.data[uint(pc.out_offset) + idx] = isinf(v) ? " + uw + "(1) : " + uw + "(0);\n"
    "  }\n"
    "}\n";
  return src;
  }



template<typename eT>
inline
std::string
runtime_t::generate_rel_isnan_kernel()
  {
  coot_debug_sigprint();
  const std::string et = is_float<eT>::value ? "float" : "double";
  const std::string uw = uword_str();
  std::string src = "#version 450\n";
  if (is_double<eT>::value) src += "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n";
  src += uword_ext();
  src +=
    "\nlayout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n"
    "layout(set=0,binding=0,std430) buffer OutBuf { " + uw + " data[]; } out_buf;\n"
    "layout(set=0,binding=1,std430) readonly buffer InBuf { " + et + " data[]; } in_buf;\n"
    "layout(push_constant) uniform PC { " + uw + " out_offset; " + uw + " in_offset; " + uw + " n_elem; } pc;\n"
    "void main() {\n"
    "  uint idx = gl_GlobalInvocationID.x;\n"
    "  if (idx < pc.n_elem) {\n"
    "    " + et + " v = in_buf.data[uint(pc.in_offset) + idx];\n"
    "    out_buf.data[uint(pc.out_offset) + idx] = isnan(v) ? " + uw + "(1) : " + uw + "(0);\n"
    "  }\n"
    "}\n";
  return src;
  }



template<typename eT>
inline
pipeline_t&
runtime_t::get_gen_rel_isfinite_pipeline()
  {
  const std::string name = is_float<eT>::value ? "gen_rel_isfinite_f32" : "gen_rel_isfinite_f64";
  auto it = pipelines.find(name);
  if (it != pipelines.end()) return it->second;
  return get_pipeline_from_source(name, generate_rel_isfinite_kernel<eT>(), push_size<rel_scalar_push_t>(), 2);
  }



template<typename eT>
inline
pipeline_t&
runtime_t::get_gen_rel_isinf_pipeline()
  {
  const std::string name = is_float<eT>::value ? "gen_rel_isinf_f32" : "gen_rel_isinf_f64";
  auto it = pipelines.find(name);
  if (it != pipelines.end()) return it->second;
  return get_pipeline_from_source(name, generate_rel_isinf_kernel<eT>(), push_size<rel_scalar_push_t>(), 2);
  }



template<typename eT>
inline
pipeline_t&
runtime_t::get_gen_rel_isnan_pipeline()
  {
  const std::string name = is_float<eT>::value ? "gen_rel_isnan_f32" : "gen_rel_isnan_f64";
  auto it = pipelines.find(name);
  if (it != pipelines.end()) return it->second;
  return get_pipeline_from_source(name, generate_rel_isnan_kernel<eT>(), push_size<rel_scalar_push_t>(), 2);
  }



template<typename eT>
inline
std::string
runtime_t::generate_trans_kernel()
  {
  coot_debug_sigprint();
  const std::string et = is_float<eT>::value ? "float" : "double";
  const std::string uw = uword_str();
  std::string src = "#version 450\n";
  if (is_double<eT>::value) src += "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n";
  src += uword_ext();
  src +=
    "\nlayout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;\n"
    "layout(set=0,binding=0,std430) buffer OutBuf { " + et + " data[]; } out_buf;\n"
    "layout(set=0,binding=1,std430) readonly buffer InBuf { " + et + " data[]; } in_buf;\n"
    "layout(push_constant) uniform PC {\n"
    "  " + uw + " dest_offset; " + uw + " src_offset;\n"
    "  " + uw + " n_rows; " + uw + " n_cols;\n"
    "} pc;\n"
    "void main() {\n"
    "  uint row = gl_GlobalInvocationID.x;\n"
    "  uint col = gl_GlobalInvocationID.y;\n"
    "  if (row < pc.n_rows && col < pc.n_cols) {\n"
    "    " + et + " v = in_buf.data[uint(pc.src_offset) + col * uint(pc.n_rows) + row];\n"
    "    out_buf.data[uint(pc.dest_offset) + row * uint(pc.n_cols) + col] = v;\n"
    "  }\n"
    "}\n";
  return src;
  }



template<typename eT>
inline
pipeline_t&
runtime_t::get_gen_trans_pipeline()
  {
  const std::string name = is_float<eT>::value ? "gen_trans_f32" : "gen_trans_f64";
  auto it = pipelines.find(name);
  if (it != pipelines.end()) return it->second;
  return get_pipeline_from_source(name, generate_trans_kernel<eT>(), push_size<trans_push_t>(), 2);
  }



template<typename eT>
inline
std::string
runtime_t::generate_reorder_cols_kernel()
  {
  coot_debug_sigprint();
  const std::string et = is_float<eT>::value ? "float" : "double";
  const std::string uw = uword_str();
  std::string src = "#version 450\n";
  if (is_double<eT>::value) src += "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n";
  src += uword_ext();
  src +=
    "\nlayout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;\n"
    "layout(set=0,binding=0,std430) buffer OutBuf { " + et + " data[]; } out_buf;\n"
    "layout(set=0,binding=1,std430) readonly buffer InBuf { " + et + " data[]; } in_buf;\n"
    "layout(set=0,binding=2,std430) readonly buffer OrderBuf { " + uw + " data[]; } order_buf;\n"
    "layout(push_constant) uniform PC {\n"
    "  " + uw + " out_offset; " + uw + " in_offset;\n"
    "  " + uw + " order_offset; " + uw + " n_rows; " + uw + " out_n_cols;\n"
    "} pc;\n"
    "void main() {\n"
    "  uint out_col = gl_GlobalInvocationID.x;\n"
    "  if (out_col < pc.out_n_cols) {\n"
    "    " + uw + " in_col = order_buf.data[uint(pc.order_offset) + out_col];\n"
    "    for (uint i = 0; i < uint(pc.n_rows); ++i) {\n"
    "      out_buf.data[uint(pc.out_offset) + out_col * uint(pc.n_rows) + i] =\n"
    "          in_buf.data[uint(pc.in_offset) + uint(in_col) * uint(pc.n_rows) + i];\n"
    "    }\n"
    "  }\n"
    "}\n";
  return src;
  }



template<typename eT>
inline
pipeline_t&
runtime_t::get_gen_reorder_cols_pipeline()
  {
  const std::string name = is_float<eT>::value ? "gen_reorder_cols_f32" : "gen_reorder_cols_f64";
  auto it = pipelines.find(name);
  if (it != pipelines.end()) return it->second;
  return get_pipeline_from_source(name, generate_reorder_cols_kernel<eT>(), push_size<reorder_cols_push_t>(), 3);
  }



template<typename eT>
inline
std::string
runtime_t::generate_index_min_colwise_kernel()
  {
  coot_debug_sigprint();
  const std::string et = is_float<eT>::value ? "float" : "double";
  const std::string uw = uword_str();
  std::string src = "#version 450\n";
  if (is_double<eT>::value) src += "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n";
  src += uword_ext();
  src +=
    "\nlayout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;\n"
    "layout(set=0,binding=0,std430) buffer OutBuf { " + uw + " data[]; } out_buf;\n"
    "layout(set=0,binding=1,std430) readonly buffer InBuf { " + et + " data[]; } in_buf;\n"
    "layout(push_constant) uniform PC {\n"
    "  " + uw + " dest_offset; " + uw + " src_offset;\n"
    "  " + uw + " n_rows; " + uw + " n_cols;\n"
    "  " + uw + " dest_mem_incr; " + uw + " src_M_n_rows;\n"
    "} pc;\n"
    "void main() {\n"
    "  uint col = gl_GlobalInvocationID.x;\n"
    "  if (col < pc.n_cols) {\n"
    "    " + et + " best = in_buf.data[uint(pc.src_offset) + col * uint(pc.src_M_n_rows)];\n"
    "    " + uw + " best_idx = " + uw + "(0);\n"
    "    for (uint row = 1; row < uint(pc.n_rows); ++row) {\n"
    "      " + et + " v = in_buf.data[uint(pc.src_offset) + row + col * uint(pc.src_M_n_rows)];\n"
    "      if (v < best) { best = v; best_idx = " + uw + "(row); }\n"
    "    }\n"
    "    out_buf.data[uint(pc.dest_offset) + col * uint(pc.dest_mem_incr)] = best_idx;\n"
    "  }\n"
    "}\n";
  return src;
  }



template<typename eT>
inline
std::string
runtime_t::generate_index_min_rowwise_kernel()
  {
  coot_debug_sigprint();
  const std::string et = is_float<eT>::value ? "float" : "double";
  const std::string uw = uword_str();
  std::string src = "#version 450\n";
  if (is_double<eT>::value) src += "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n";
  src += uword_ext();
  src +=
    "\nlayout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;\n"
    "layout(set=0,binding=0,std430) buffer OutBuf { " + uw + " data[]; } out_buf;\n"
    "layout(set=0,binding=1,std430) readonly buffer InBuf { " + et + " data[]; } in_buf;\n"
    "layout(push_constant) uniform PC {\n"
    "  " + uw + " dest_offset; " + uw + " src_offset;\n"
    "  " + uw + " n_rows; " + uw + " n_cols;\n"
    "  " + uw + " dest_mem_incr; " + uw + " src_M_n_rows;\n"
    "} pc;\n"
    "void main() {\n"
    "  uint row = gl_GlobalInvocationID.x;\n"
    "  if (row < pc.n_rows) {\n"
    "    " + et + " best = in_buf.data[uint(pc.src_offset) + row];\n"
    "    " + uw + " best_idx = " + uw + "(0);\n"
    "    for (uint col = 1; col < uint(pc.n_cols); ++col) {\n"
    "      " + et + " v = in_buf.data[uint(pc.src_offset) + row + col * uint(pc.src_M_n_rows)];\n"
    "      if (v < best) { best = v; best_idx = " + uw + "(col); }\n"
    "    }\n"
    "    out_buf.data[uint(pc.dest_offset) + row * uint(pc.dest_mem_incr)] = best_idx;\n"
    "  }\n"
    "}\n";
  return src;
  }



template<typename eT>
inline
std::string
runtime_t::generate_index_min_vec_kernel()
  {
  coot_debug_sigprint();
  const std::string et = is_float<eT>::value ? "float" : "double";
  const std::string uw = uword_str();
  std::string src = "#version 450\n";
  if (is_double<eT>::value) src += "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n";
  src += uword_ext();
  src +=
    "\nlayout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\n"
    "layout(set=0,binding=0,std430) buffer OutBuf { " + uw + " data[]; } out_buf;\n"
    "layout(set=0,binding=1,std430) readonly buffer InBuf { " + et + " data[]; } in_buf;\n"
    "layout(set=0,binding=2,std430) buffer AuxBuf { " + et + " data[]; } aux_buf;\n"
    "layout(push_constant) uniform PC {\n"
    "  " + uw + " out_offset; " + uw + " in_offset; " + uw + " n_elem; " + uw + " aux_offset;\n"
    "} pc;\n"
    "void main() {\n"
    "  if (gl_GlobalInvocationID.x == 0u) {\n"
    "    " + et + " best = in_buf.data[uint(pc.in_offset)];\n"
    "    " + uw + " best_idx = " + uw + "(0);\n"
    "    for (uint i = 1; i < uint(pc.n_elem); ++i) {\n"
    "      " + et + " v = in_buf.data[uint(pc.in_offset) + i];\n"
    "      if (v < best) { best = v; best_idx = " + uw + "(i); }\n"
    "    }\n"
    "    out_buf.data[uint(pc.out_offset)] = best_idx;\n"
    "    aux_buf.data[uint(pc.aux_offset)] = best;\n"
    "  }\n"
    "}\n";
  return src;
  }



template<typename eT>
inline
pipeline_t&
runtime_t::get_gen_index_min_colwise_pipeline()
  {
  const std::string name = is_float<eT>::value ? "gen_index_min_colwise_f32" : "gen_index_min_colwise_f64";
  auto it = pipelines.find(name);
  if (it != pipelines.end()) return it->second;
  return get_pipeline_from_source(name, generate_index_min_colwise_kernel<eT>(), push_size<index_reduce_push_t>(), 2);
  }



template<typename eT>
inline
pipeline_t&
runtime_t::get_gen_index_min_rowwise_pipeline()
  {
  const std::string name = is_float<eT>::value ? "gen_index_min_rowwise_f32" : "gen_index_min_rowwise_f64";
  auto it = pipelines.find(name);
  if (it != pipelines.end()) return it->second;
  return get_pipeline_from_source(name, generate_index_min_rowwise_kernel<eT>(), push_size<index_reduce_push_t>(), 2);
  }



template<typename eT>
inline
pipeline_t&
runtime_t::get_gen_index_min_vec_pipeline()
  {
  const std::string name = is_float<eT>::value ? "gen_index_min_vec_f32" : "gen_index_min_vec_f64";
  auto it = pipelines.find(name);
  if (it != pipelines.end()) return it->second;
  return get_pipeline_from_source(name, generate_index_min_vec_kernel<eT>(), push_size<index_vec_push_t>(), 3);
  }



template<typename eT>
inline
std::string
runtime_t::generate_index_max_colwise_kernel()
  {
  coot_debug_sigprint();
  const std::string et = is_float<eT>::value ? "float" : "double";
  const std::string uw = uword_str();
  std::string src = "#version 450\n";
  if (is_double<eT>::value) src += "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n";
  src += uword_ext();
  src +=
    "\nlayout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;\n"
    "layout(set=0,binding=0,std430) buffer OutBuf { " + uw + " data[]; } out_buf;\n"
    "layout(set=0,binding=1,std430) readonly buffer InBuf { " + et + " data[]; } in_buf;\n"
    "layout(push_constant) uniform PC {\n"
    "  " + uw + " dest_offset; " + uw + " src_offset;\n"
    "  " + uw + " n_rows; " + uw + " n_cols;\n"
    "  " + uw + " dest_mem_incr; " + uw + " src_M_n_rows;\n"
    "} pc;\n"
    "void main() {\n"
    "  uint col = gl_GlobalInvocationID.x;\n"
    "  if (col < pc.n_cols) {\n"
    "    " + et + " best = in_buf.data[uint(pc.src_offset) + col * uint(pc.src_M_n_rows)];\n"
    "    " + uw + " best_idx = " + uw + "(0);\n"
    "    for (uint row = 1; row < uint(pc.n_rows); ++row) {\n"
    "      " + et + " v = in_buf.data[uint(pc.src_offset) + row + col * uint(pc.src_M_n_rows)];\n"
    "      if (v > best) { best = v; best_idx = " + uw + "(row); }\n"
    "    }\n"
    "    out_buf.data[uint(pc.dest_offset) + col * uint(pc.dest_mem_incr)] = best_idx;\n"
    "  }\n"
    "}\n";
  return src;
  }



template<typename eT>
inline
std::string
runtime_t::generate_index_max_rowwise_kernel()
  {
  coot_debug_sigprint();
  const std::string et = is_float<eT>::value ? "float" : "double";
  const std::string uw = uword_str();
  std::string src = "#version 450\n";
  if (is_double<eT>::value) src += "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n";
  src += uword_ext();
  src +=
    "\nlayout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;\n"
    "layout(set=0,binding=0,std430) buffer OutBuf { " + uw + " data[]; } out_buf;\n"
    "layout(set=0,binding=1,std430) readonly buffer InBuf { " + et + " data[]; } in_buf;\n"
    "layout(push_constant) uniform PC {\n"
    "  " + uw + " dest_offset; " + uw + " src_offset;\n"
    "  " + uw + " n_rows; " + uw + " n_cols;\n"
    "  " + uw + " dest_mem_incr; " + uw + " src_M_n_rows;\n"
    "} pc;\n"
    "void main() {\n"
    "  uint row = gl_GlobalInvocationID.x;\n"
    "  if (row < pc.n_rows) {\n"
    "    " + et + " best = in_buf.data[uint(pc.src_offset) + row];\n"
    "    " + uw + " best_idx = " + uw + "(0);\n"
    "    for (uint col = 1; col < uint(pc.n_cols); ++col) {\n"
    "      " + et + " v = in_buf.data[uint(pc.src_offset) + row + col * uint(pc.src_M_n_rows)];\n"
    "      if (v > best) { best = v; best_idx = " + uw + "(col); }\n"
    "    }\n"
    "    out_buf.data[uint(pc.dest_offset) + row * uint(pc.dest_mem_incr)] = best_idx;\n"
    "  }\n"
    "}\n";
  return src;
  }



template<typename eT>
inline
std::string
runtime_t::generate_index_max_vec_kernel()
  {
  coot_debug_sigprint();
  const std::string et = is_float<eT>::value ? "float" : "double";
  const std::string uw = uword_str();
  std::string src = "#version 450\n";
  if (is_double<eT>::value) src += "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n";
  src += uword_ext();
  src +=
    "\nlayout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\n"
    "layout(set=0,binding=0,std430) buffer OutBuf { " + uw + " data[]; } out_buf;\n"
    "layout(set=0,binding=1,std430) readonly buffer InBuf { " + et + " data[]; } in_buf;\n"
    "layout(set=0,binding=2,std430) buffer AuxBuf { " + et + " data[]; } aux_buf;\n"
    "layout(push_constant) uniform PC {\n"
    "  " + uw + " out_offset; " + uw + " in_offset; " + uw + " n_elem; " + uw + " aux_offset;\n"
    "} pc;\n"
    "void main() {\n"
    "  if (gl_GlobalInvocationID.x == 0u) {\n"
    "    " + et + " best = in_buf.data[uint(pc.in_offset)];\n"
    "    " + uw + " best_idx = " + uw + "(0);\n"
    "    for (uint i = 1; i < uint(pc.n_elem); ++i) {\n"
    "      " + et + " v = in_buf.data[uint(pc.in_offset) + i];\n"
    "      if (v > best) { best = v; best_idx = " + uw + "(i); }\n"
    "    }\n"
    "    out_buf.data[uint(pc.out_offset)] = best_idx;\n"
    "    aux_buf.data[uint(pc.aux_offset)] = best;\n"
    "  }\n"
    "}\n";
  return src;
  }



template<typename eT>
inline
pipeline_t&
runtime_t::get_gen_index_max_colwise_pipeline()
  {
  const std::string name = is_float<eT>::value ? "gen_index_max_colwise_f32" : "gen_index_max_colwise_f64";
  auto it = pipelines.find(name);
  if (it != pipelines.end()) return it->second;
  return get_pipeline_from_source(name, generate_index_max_colwise_kernel<eT>(), push_size<index_reduce_push_t>(), 2);
  }



template<typename eT>
inline
pipeline_t&
runtime_t::get_gen_index_max_rowwise_pipeline()
  {
  const std::string name = is_float<eT>::value ? "gen_index_max_rowwise_f32" : "gen_index_max_rowwise_f64";
  auto it = pipelines.find(name);
  if (it != pipelines.end()) return it->second;
  return get_pipeline_from_source(name, generate_index_max_rowwise_kernel<eT>(), push_size<index_reduce_push_t>(), 2);
  }



template<typename eT>
inline
pipeline_t&
runtime_t::get_gen_index_max_vec_pipeline()
  {
  const std::string name = is_float<eT>::value ? "gen_index_max_vec_f32" : "gen_index_max_vec_f64";
  auto it = pipelines.find(name);
  if (it != pipelines.end()) return it->second;
  return get_pipeline_from_source(name, generate_index_max_vec_kernel<eT>(), push_size<index_vec_push_t>(), 3);
  }



template<typename eT_src, typename eT_dest>
inline
std::string
runtime_t::generate_broadcast_kernel(const std::string& op_expr)
  {
  coot_debug_sigprint();
  const std::string et_src  = is_float<eT_src>::value  ? "float" : "double";
  const std::string et_dest = is_float<eT_dest>::value ? "float" : "double";
  const std::string uw = uword_str();
  std::string src = "#version 450\n";
  if (is_double<eT_src>::value || is_double<eT_dest>::value)
    src += "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n";
  src += uword_ext();
  src +=
    "\nlayout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;\n"
    "layout(set=0,binding=0,std430) buffer OutBuf { " + et_dest + " data[]; } out_buf;\n"
    "layout(set=0,binding=1,std430) readonly buffer SrcBuf { " + et_src + " data[]; } src_buf;\n"
    "layout(push_constant) uniform PC {\n"
    "  " + uw + " dest_offset;    " + uw + " dest_in_offset; " + uw + " src_offset;\n"
    "  " + uw + " src_n_rows;     " + uw + " src_n_cols;\n"
    "  " + uw + " copies_per_row; " + uw + " copies_per_col;\n"
    "  " + uw + " dest_M_n_rows;  " + uw + " dest_in_M_n_rows; " + uw + " src_M_n_rows;\n"
    "} pc;\n"
    "void main() {\n"
    "  uint out_row = gl_GlobalInvocationID.x;\n"
    "  uint out_col = gl_GlobalInvocationID.y;\n"
    "  uint new_n_rows = uint(pc.src_n_rows) * uint(pc.copies_per_row);\n"
    "  uint new_n_cols = uint(pc.src_n_cols) * uint(pc.copies_per_col);\n"
    "  if (out_row < new_n_rows && out_col < new_n_cols) {\n"
    "    uint in_row = out_row % uint(pc.src_n_rows);\n"
    "    uint in_col = out_col % uint(pc.src_n_cols);\n"
    "    " + et_dest + " d = out_buf.data[uint(pc.dest_in_offset) + out_col * uint(pc.dest_in_M_n_rows) + out_row];\n"
    "    " + et_dest + " s = " + et_dest + "(src_buf.data[uint(pc.src_offset) + in_col * uint(pc.src_M_n_rows) + in_row]);\n"
    "    out_buf.data[uint(pc.dest_offset) + out_col * uint(pc.dest_M_n_rows) + out_row] = " + op_expr + ";\n"
    "  }\n"
    "}\n";
  return src;
  }



template<typename eT_src, typename eT_dest> inline pipeline_t& runtime_t::get_gen_broadcast_set_pipeline()
  {
  const std::string name = std::string("gen_broadcast_set_") + (is_float<eT_src>::value ? "f32" : "f64") + "_" + (is_float<eT_dest>::value ? "f32" : "f64");
  auto it = pipelines.find(name);
  if (it != pipelines.end()) return it->second;
  return get_pipeline_from_source(name, generate_broadcast_kernel<eT_src, eT_dest>("s"), push_size<broadcast_push_t>(), 2);
  }

template<typename eT_src, typename eT_dest> inline pipeline_t& runtime_t::get_gen_broadcast_plus_pipeline()
  {
  const std::string name = std::string("gen_broadcast_plus_") + (is_float<eT_src>::value ? "f32" : "f64") + "_" + (is_float<eT_dest>::value ? "f32" : "f64");
  auto it = pipelines.find(name);
  if (it != pipelines.end()) return it->second;
  return get_pipeline_from_source(name, generate_broadcast_kernel<eT_src, eT_dest>("d + s"), push_size<broadcast_push_t>(), 2);
  }

template<typename eT_src, typename eT_dest> inline pipeline_t& runtime_t::get_gen_broadcast_minus_pre_pipeline()
  {
  const std::string name = std::string("gen_broadcast_minus_pre_") + (is_float<eT_src>::value ? "f32" : "f64") + "_" + (is_float<eT_dest>::value ? "f32" : "f64");
  auto it = pipelines.find(name);
  if (it != pipelines.end()) return it->second;
  return get_pipeline_from_source(name, generate_broadcast_kernel<eT_src, eT_dest>("s - d"), push_size<broadcast_push_t>(), 2);
  }

template<typename eT_src, typename eT_dest> inline pipeline_t& runtime_t::get_gen_broadcast_minus_post_pipeline()
  {
  const std::string name = std::string("gen_broadcast_minus_post_") + (is_float<eT_src>::value ? "f32" : "f64") + "_" + (is_float<eT_dest>::value ? "f32" : "f64");
  auto it = pipelines.find(name);
  if (it != pipelines.end()) return it->second;
  return get_pipeline_from_source(name, generate_broadcast_kernel<eT_src, eT_dest>("d - s"), push_size<broadcast_push_t>(), 2);
  }

template<typename eT_src, typename eT_dest> inline pipeline_t& runtime_t::get_gen_broadcast_schur_pipeline()
  {
  const std::string name = std::string("gen_broadcast_schur_") + (is_float<eT_src>::value ? "f32" : "f64") + "_" + (is_float<eT_dest>::value ? "f32" : "f64");
  auto it = pipelines.find(name);
  if (it != pipelines.end()) return it->second;
  return get_pipeline_from_source(name, generate_broadcast_kernel<eT_src, eT_dest>("d * s"), push_size<broadcast_push_t>(), 2);
  }

template<typename eT_src, typename eT_dest> inline pipeline_t& runtime_t::get_gen_broadcast_div_pre_pipeline()
  {
  const std::string name = std::string("gen_broadcast_div_pre_") + (is_float<eT_src>::value ? "f32" : "f64") + "_" + (is_float<eT_dest>::value ? "f32" : "f64");
  auto it = pipelines.find(name);
  if (it != pipelines.end()) return it->second;
  return get_pipeline_from_source(name, generate_broadcast_kernel<eT_src, eT_dest>("s / d"), push_size<broadcast_push_t>(), 2);
  }

template<typename eT_src, typename eT_dest> inline pipeline_t& runtime_t::get_gen_broadcast_div_post_pipeline()
  {
  const std::string name = std::string("gen_broadcast_div_post_") + (is_float<eT_src>::value ? "f32" : "f64") + "_" + (is_float<eT_dest>::value ? "f32" : "f64");
  auto it = pipelines.find(name);
  if (it != pipelines.end()) return it->second;
  return get_pipeline_from_source(name, generate_broadcast_kernel<eT_src, eT_dest>("d / s"), push_size<broadcast_push_t>(), 2);
  }



inline
runtime_t::adapt_uword::adapt_uword(const uword val)
  {
  if (get_rt().vk_rt.has_sizet64())
    {
    size = sizeof(u64);
    addr = (void*)(&val64);
    val64 = u64(val);
    }
  else
    {
    size = sizeof(u32);
    addr = (void*)(&val32);
    coot_check_runtime_error( ((sizeof(uword) >= 8) && (val > 0xffffffffU)), "given value doesn't fit into unsigned 32 bit integer" );
    val32 = u32(val);
    }
  }



inline
runtime_t::adapt_uword::adapt_uword(const adapt_uword& x)
  : size(x.size), val64(x.val64), val32(x.val32)
  {
  addr = (size == sizeof(u64)) ? (void*) &val64 : (void*) &val32;
  }



inline
runtime_t::adapt_uword::adapt_uword(adapt_uword&& x)
  : size(x.size), val64(x.val64), val32(x.val32)
  {
  addr = (size == sizeof(u64)) ? (void*) &val64 : (void*) &val32;
  }



inline
runtime_t::adapt_uword&
runtime_t::adapt_uword::operator=(const adapt_uword& x)
  {
  if (this != &x)
    {
    size = x.size;
    val64 = x.val64;
    val32 = x.val32;
    addr = (size == sizeof(u64)) ? (void*) &val64 : (void*) &val32;
    }
  return *this;
  }



inline
runtime_t::adapt_uword&
runtime_t::adapt_uword::operator=(adapt_uword&& x)
  {
  if (this != &x)
    {
    size = x.size;
    val64 = x.val64;
    val32 = x.val32;
    addr = (size == sizeof(u64)) ? (void*) &val64 : (void*) &val32;
    }
  return *this;
  }

template<typename eT_in, typename eT_out>
inline
std::string
runtime_t::generate_gather_kernel()
  {
  coot_debug_sigprint();

  const bool need_fp64 = is_double<eT_in>::value || is_double<eT_out>::value;
  const std::string et_in  = is_float<eT_in>::value  ? "float" : "double";
  const std::string et_out = is_float<eT_out>::value ? "float" : "double";
  const std::string uw = uword_str();

  std::string source = "#version 450\n";
  if (need_fp64)
    source += "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n";
  source += uword_ext();

  source +=
      "\n"
      "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n"
      "\n"
      "layout(set = 0, binding = 0, std430) buffer OutBuf { " + et_out + " data[]; } out_buf;\n"
      "layout(set = 0, binding = 1, std430) readonly buffer InBuf { " + et_in + " data[]; } in_buf;\n"
      "layout(set = 0, binding = 2, std430) readonly buffer IdxBuf { " + uw + " data[]; } idx_buf;\n"
      "\n"
      "layout(push_constant) uniform PushConsts {\n"
      "  " + uw + " out_offset;\n"
      "  " + uw + " in_offset;\n"
      "  " + uw + " idx_offset;\n"
      "  " + uw + " n_elem;\n"
      "} pc;\n"
      "\n"
      "void main() {\n"
      "  uint i = gl_GlobalInvocationID.x;\n"
      "  if (i < uint(pc.n_elem)) {\n"
      "    " + uw + " src = idx_buf.data[uint(pc.idx_offset) + i];\n"
      "    out_buf.data[uint(pc.out_offset) + i] = " + et_out + "(in_buf.data[uint(pc.in_offset) + uint(src)]);\n"
      "  }\n"
      "}\n";

  return source;
  }



template<typename eT_in, typename eT_out>
inline
pipeline_t&
runtime_t::get_gen_gather_pipeline()
  {
  coot_debug_sigprint();

  const std::string in_str  = is_float<eT_in>::value  ? "f32" : "f64";
  const std::string out_str = is_float<eT_out>::value ? "f32" : "f64";
  const std::string name = "gen_gather_" + in_str + "_to_" + out_str;

  auto it = pipelines.find(name);
  if (it != pipelines.end())
    return it->second;

  const std::string source = generate_gather_kernel<eT_in, eT_out>();
  return get_pipeline_from_source(name, source, push_size<gather_push_t>());
  }



template<typename eT, bool do_trans_A, bool do_trans_B>
inline
std::string
runtime_t::generate_gemm_kernel()
  {
  coot_debug_sigprint();

  const bool need_fp64 = is_double<eT>::value;
  const std::string et_str = is_float<eT>::value ? "float" : "double";
  const std::string uw = uword_str();

  std::string source = "#version 450\n";
  if (need_fp64)
    source += "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n";
  source += uword_ext();

  source +=
      "\n"
      "layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;\n"
      "\n"
      "layout(set = 0, binding = 0, std430) buffer CBuf { " + et_str + " data[]; } c_buf;\n"
      "layout(set = 0, binding = 1, std430) readonly buffer ABuf { " + et_str + " data[]; } a_buf;\n"
      "layout(set = 0, binding = 2, std430) readonly buffer BBuf { " + et_str + " data[]; } b_buf;\n"
      "layout(set = 0, binding = 3, std430) readonly buffer ScalarsBuf { " + et_str + " data[]; } sc_buf;\n"
      "\n"
      "layout(push_constant) uniform PushConsts {\n"
      "  " + uw + " c_offset;\n"
      "  " + uw + " c_row_offset;\n"
      "  " + uw + " c_col_offset;\n"
      "  " + uw + " c_M_n_rows;\n"
      "  " + uw + " a_offset;\n"
      "  " + uw + " a_row_offset;\n"
      "  " + uw + " a_col_offset;\n"
      "  " + uw + " a_M_n_rows;\n"
      "  " + uw + " b_offset;\n"
      "  " + uw + " b_row_offset;\n"
      "  " + uw + " b_col_offset;\n"
      "  " + uw + " b_M_n_rows;\n"
      "  " + uw + " c_n_rows;\n"
      "  " + uw + " c_n_cols;\n"
      "  " + uw + " K;\n"
      "  " + uw + " scalars_offset;\n"
      "} pc;\n"
      "\n"
      "void main() {\n"
      "  uint row = gl_GlobalInvocationID.x;\n"
      "  uint col = gl_GlobalInvocationID.y;\n"
      "  if (row >= uint(pc.c_n_rows) || col >= uint(pc.c_n_cols)) { return; }\n"
      "\n"
      "  " + et_str + " alpha = sc_buf.data[uint(pc.scalars_offset)];\n"
      "  " + et_str + " beta  = sc_buf.data[uint(pc.scalars_offset) + 1u];\n"
      "\n"
      "  " + et_str + " sum = " + et_str + "(0);\n"
      "  for (uint k = 0u; k < uint(pc.K); ++k) {\n";

  if (do_trans_A)
    source +=
      "    " + et_str + " a_val = a_buf.data[uint(pc.a_offset) + uint(pc.a_row_offset) + k + (uint(pc.a_col_offset) + row) * uint(pc.a_M_n_rows)];\n";
  else
    source +=
      "    " + et_str + " a_val = a_buf.data[uint(pc.a_offset) + uint(pc.a_row_offset) + row + (uint(pc.a_col_offset) + k) * uint(pc.a_M_n_rows)];\n";

  if (do_trans_B)
    source +=
      "    " + et_str + " b_val = b_buf.data[uint(pc.b_offset) + uint(pc.b_row_offset) + col + (uint(pc.b_col_offset) + k) * uint(pc.b_M_n_rows)];\n";
  else
    source +=
      "    " + et_str + " b_val = b_buf.data[uint(pc.b_offset) + uint(pc.b_row_offset) + k + (uint(pc.b_col_offset) + col) * uint(pc.b_M_n_rows)];\n";

  source +=
      "    sum += a_val * b_val;\n"
      "  }\n"
      "\n"
      "  uint c_idx = uint(pc.c_offset) + uint(pc.c_row_offset) + row + (uint(pc.c_col_offset) + col) * uint(pc.c_M_n_rows);\n"
      "  c_buf.data[c_idx] = alpha * sum + beta * c_buf.data[c_idx];\n"
      "}\n";

  return source;
  }



template<typename eT, bool do_trans_A, bool do_trans_B>
inline
pipeline_t&
runtime_t::get_gen_gemm_pipeline()
  {
  coot_debug_sigprint();

  const std::string type_str = is_float<eT>::value ? "f32" : "f64";
  const std::string ta_str = do_trans_A ? "t" : "f";
  const std::string tb_str = do_trans_B ? "t" : "f";
  const std::string name = "gen_gemm_" + type_str + "_" + ta_str + tb_str;

  auto it = pipelines.find(name);
  if (it != pipelines.end())
    return it->second;

  const std::string source = generate_gemm_kernel<eT, do_trans_A, do_trans_B>();
  return get_pipeline_from_source(name, source, push_size<gemm_push_t>(), 4);
  }



#endif // COOT_USE_SHADERC
