// Copyright 2025 Marcus Edel (https://kurg.org)
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



  //
  // instance and device setup
  //



  extern VkResult coot_wrapper(vkCreateInstance)(const VkInstanceCreateInfo* pCreateInfo,
                                                 const VkAllocationCallbacks* pAllocator,
                                                 VkInstance* pInstance);



  extern void coot_wrapper(vkDestroyInstance)(VkInstance instance,
                                              const VkAllocationCallbacks* pAllocator);



  extern VkResult coot_wrapper(vkEnumeratePhysicalDevices)(VkInstance instance,
                                                           uint32_t* pPhysicalDeviceCount,
                                                           VkPhysicalDevice* pPhysicalDevices);



  extern VkResult coot_wrapper(vkEnumerateInstanceLayerProperties)(uint32_t* pPropertyCount,
                                                                   VkLayerProperties* pProperties);



  extern VkResult coot_wrapper(vkEnumerateInstanceExtensionProperties)(const char* pLayerName,
                                                                      uint32_t* pPropertyCount,
                                                                      VkExtensionProperties* pProperties);



  extern void coot_wrapper(vkGetPhysicalDeviceFeatures)(VkPhysicalDevice physicalDevice,
                                                        VkPhysicalDeviceFeatures* pFeatures);



  extern void coot_wrapper(vkGetPhysicalDeviceMemoryProperties)(VkPhysicalDevice physicalDevice,
                                                                VkPhysicalDeviceMemoryProperties* pMemoryProperties);



  extern void coot_wrapper(vkGetPhysicalDeviceQueueFamilyProperties)(VkPhysicalDevice physicalDevice,
                                                                     uint32_t* pQueueFamilyPropertyCount,
                                                                     VkQueueFamilyProperties* pQueueFamilyProperties);



  extern VkResult coot_wrapper(vkCreateDevice)(VkPhysicalDevice physicalDevice,
                                               const VkDeviceCreateInfo* pCreateInfo,
                                               const VkAllocationCallbacks* pAllocator,
                                               VkDevice* pDevice);



  extern void coot_wrapper(vkDestroyDevice)(VkDevice device,
                                            const VkAllocationCallbacks* pAllocator);



  extern void coot_wrapper(vkGetDeviceQueue)(VkDevice device,
                                             uint32_t queueFamilyIndex,
                                             uint32_t queueIndex,
                                             VkQueue* pQueue);



  extern PFN_vkVoidFunction coot_wrapper(vkGetInstanceProcAddr)(VkInstance instance,
                                                                const char* pName);



  //
  // command pool and command buffer
  //



  extern VkResult coot_wrapper(vkCreateCommandPool)(VkDevice device,
                                                    const VkCommandPoolCreateInfo* pCreateInfo,
                                                    const VkAllocationCallbacks* pAllocator,
                                                    VkCommandPool* pCommandPool);



  extern void coot_wrapper(vkDestroyCommandPool)(VkDevice device,
                                                 VkCommandPool commandPool,
                                                 const VkAllocationCallbacks* pAllocator);



  extern VkResult coot_wrapper(vkAllocateCommandBuffers)(VkDevice device,
                                                         const VkCommandBufferAllocateInfo* pAllocateInfo,
                                                         VkCommandBuffer* pCommandBuffers);



  extern void coot_wrapper(vkFreeCommandBuffers)(VkDevice device,
                                                 VkCommandPool commandPool,
                                                 uint32_t commandBufferCount,
                                                 const VkCommandBuffer* pCommandBuffers);



  extern VkResult coot_wrapper(vkBeginCommandBuffer)(VkCommandBuffer commandBuffer,
                                                     const VkCommandBufferBeginInfo* pBeginInfo);



  extern VkResult coot_wrapper(vkEndCommandBuffer)(VkCommandBuffer commandBuffer);



  //
  // descriptor pool and descriptor sets
  //



  extern VkResult coot_wrapper(vkCreateDescriptorPool)(VkDevice device,
                                                       const VkDescriptorPoolCreateInfo* pCreateInfo,
                                                       const VkAllocationCallbacks* pAllocator,
                                                       VkDescriptorPool* pDescriptorPool);



  extern void coot_wrapper(vkDestroyDescriptorPool)(VkDevice device,
                                                    VkDescriptorPool descriptorPool,
                                                    const VkAllocationCallbacks* pAllocator);



  extern VkResult coot_wrapper(vkCreateDescriptorSetLayout)(VkDevice device,
                                                            const VkDescriptorSetLayoutCreateInfo* pCreateInfo,
                                                            const VkAllocationCallbacks* pAllocator,
                                                            VkDescriptorSetLayout* pSetLayout);



  extern void coot_wrapper(vkDestroyDescriptorSetLayout)(VkDevice device,
                                                         VkDescriptorSetLayout descriptorSetLayout,
                                                         const VkAllocationCallbacks* pAllocator);



  extern VkResult coot_wrapper(vkAllocateDescriptorSets)(VkDevice device,
                                                         const VkDescriptorSetAllocateInfo* pAllocateInfo,
                                                         VkDescriptorSet* pDescriptorSets);



  extern VkResult coot_wrapper(vkResetDescriptorPool)(VkDevice device,
                                                      VkDescriptorPool descriptorPool,
                                                      VkDescriptorPoolResetFlags flags);



  extern void coot_wrapper(vkUpdateDescriptorSets)(VkDevice device,
                                                   uint32_t descriptorWriteCount,
                                                   const VkWriteDescriptorSet* pDescriptorWrites,
                                                   uint32_t descriptorCopyCount,
                                                   const VkCopyDescriptorSet* pDescriptorCopies);



  //
  // pipeline
  //



  extern VkResult coot_wrapper(vkCreateShaderModule)(VkDevice device,
                                                     const VkShaderModuleCreateInfo* pCreateInfo,
                                                     const VkAllocationCallbacks* pAllocator,
                                                     VkShaderModule* pShaderModule);



  extern void coot_wrapper(vkDestroyShaderModule)(VkDevice device,
                                                  VkShaderModule shaderModule,
                                                  const VkAllocationCallbacks* pAllocator);



  extern VkResult coot_wrapper(vkCreatePipelineLayout)(VkDevice device,
                                                       const VkPipelineLayoutCreateInfo* pCreateInfo,
                                                       const VkAllocationCallbacks* pAllocator,
                                                       VkPipelineLayout* pPipelineLayout);



  extern void coot_wrapper(vkDestroyPipelineLayout)(VkDevice device,
                                                    VkPipelineLayout pipelineLayout,
                                                    const VkAllocationCallbacks* pAllocator);



  extern VkResult coot_wrapper(vkCreateComputePipelines)(VkDevice device,
                                                         VkPipelineCache pipelineCache,
                                                         uint32_t createInfoCount,
                                                         const VkComputePipelineCreateInfo* pCreateInfos,
                                                         const VkAllocationCallbacks* pAllocator,
                                                         VkPipeline* pPipelines);



  extern void coot_wrapper(vkDestroyPipeline)(VkDevice device,
                                              VkPipeline pipeline,
                                              const VkAllocationCallbacks* pAllocator);



  //
  // memory
  //



  extern VkResult coot_wrapper(vkCreateBuffer)(VkDevice device,
                                               const VkBufferCreateInfo* pCreateInfo,
                                               const VkAllocationCallbacks* pAllocator,
                                               VkBuffer* pBuffer);



  extern void coot_wrapper(vkDestroyBuffer)(VkDevice device,
                                            VkBuffer buffer,
                                            const VkAllocationCallbacks* pAllocator);



  extern VkResult coot_wrapper(vkAllocateMemory)(VkDevice device,
                                                 const VkMemoryAllocateInfo* pAllocateInfo,
                                                 const VkAllocationCallbacks* pAllocator,
                                                 VkDeviceMemory* pMemory);



  extern void coot_wrapper(vkFreeMemory)(VkDevice device,
                                         VkDeviceMemory memory,
                                         const VkAllocationCallbacks* pAllocator);



  extern VkResult coot_wrapper(vkBindBufferMemory)(VkDevice device,
                                                   VkBuffer buffer,
                                                   VkDeviceMemory memory,
                                                   VkDeviceSize memoryOffset);



  extern void coot_wrapper(vkGetBufferMemoryRequirements)(VkDevice device,
                                                          VkBuffer buffer,
                                                          VkMemoryRequirements* pMemoryRequirements);



  extern VkResult coot_wrapper(vkMapMemory)(VkDevice device,
                                            VkDeviceMemory memory,
                                            VkDeviceSize offset,
                                            VkDeviceSize size,
                                            VkMemoryMapFlags flags,
                                            void** ppData);



  extern void coot_wrapper(vkUnmapMemory)(VkDevice device,
                                          VkDeviceMemory memory);



  //
  // command recording
  //



  extern void coot_wrapper(vkCmdBindPipeline)(VkCommandBuffer commandBuffer,
                                              VkPipelineBindPoint pipelineBindPoint,
                                              VkPipeline pipeline);



  extern void coot_wrapper(vkCmdBindDescriptorSets)(VkCommandBuffer commandBuffer,
                                                    VkPipelineBindPoint pipelineBindPoint,
                                                    VkPipelineLayout layout,
                                                    uint32_t firstSet,
                                                    uint32_t descriptorSetCount,
                                                    const VkDescriptorSet* pDescriptorSets,
                                                    uint32_t dynamicOffsetCount,
                                                    const uint32_t* pDynamicOffsets);



  extern void coot_wrapper(vkCmdPushConstants)(VkCommandBuffer commandBuffer,
                                               VkPipelineLayout layout,
                                               VkShaderStageFlags stageFlags,
                                               uint32_t offset,
                                               uint32_t size,
                                               const void* pValues);



  extern void coot_wrapper(vkCmdDispatch)(VkCommandBuffer commandBuffer,
                                          uint32_t groupCountX,
                                          uint32_t groupCountY,
                                          uint32_t groupCountZ);



  //
  // synchronisation
  //



  extern VkResult coot_wrapper(vkQueueSubmit)(VkQueue queue,
                                              uint32_t submitCount,
                                              const VkSubmitInfo* pSubmits,
                                              VkFence fence);



  extern VkResult coot_wrapper(vkQueueWaitIdle)(VkQueue queue);



  extern VkResult coot_wrapper(vkDeviceWaitIdle)(VkDevice device);



#if defined(COOT_USE_SHADERC)

  //
  // shaderc (runtime GLSL -> SPIR-V compilation)
  //



  extern shaderc_compiler_t coot_wrapper(shaderc_compiler_initialize)(void);



  extern void coot_wrapper(shaderc_compiler_release)(shaderc_compiler_t);



  extern shaderc_compile_options_t coot_wrapper(shaderc_compile_options_initialize)(void);



  extern void coot_wrapper(shaderc_compile_options_release)(shaderc_compile_options_t options);



  extern void coot_wrapper(shaderc_compile_options_add_macro_definition)(
      shaderc_compile_options_t options,
      const char* name, size_t name_length,
      const char* value, size_t value_length);



  extern void coot_wrapper(shaderc_compile_options_set_optimization_level)(
      shaderc_compile_options_t options, shaderc_optimization_level level);



  extern void coot_wrapper(shaderc_compile_options_set_target_env)(
      shaderc_compile_options_t options,
      shaderc_target_env target,
      uint32_t version);



  extern shaderc_compilation_result_t coot_wrapper(shaderc_compile_into_spv)(
      const shaderc_compiler_t compiler, const char* source_text,
      size_t source_text_size, shaderc_shader_kind shader_kind,
      const char* input_file_name, const char* entry_point_name,
      const shaderc_compile_options_t additional_options);



  extern void coot_wrapper(shaderc_result_release)(shaderc_compilation_result_t result);



  extern size_t coot_wrapper(shaderc_result_get_length)(const shaderc_compilation_result_t result);



  extern shaderc_compilation_status coot_wrapper(shaderc_result_get_compilation_status)(
      const shaderc_compilation_result_t);



  extern const char* coot_wrapper(shaderc_result_get_bytes)(const shaderc_compilation_result_t result);



  extern const char* coot_wrapper(shaderc_result_get_error_message)(
      const shaderc_compilation_result_t result);



#endif // COOT_USE_SHADERC

  }
