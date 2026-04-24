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
VkDescriptorSet
create_descriptor_set(runtime_t& rt, pipeline_t& pipe, const VkDescriptorBufferInfo* infos, const uint32_t num_buffers)
  {
  coot_wrapper(vkResetDescriptorPool)(rt.get_device(), rt.get_descriptor_pool(), 0);

  VkDescriptorSetAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  alloc_info.descriptorPool = rt.get_descriptor_pool();
  alloc_info.descriptorSetCount = 1;
  alloc_info.pSetLayouts = &pipe.set_layout;

  VkDescriptorSet set = VK_NULL_HANDLE;
  VkResult result = coot_wrapper(vkAllocateDescriptorSets)(rt.get_device(), &alloc_info, &set);
  coot_check_vk_error(result, "coot::vulkan::create_descriptor_set(): vkAllocateDescriptorSets() failed");

  std::vector<VkWriteDescriptorSet> writes(num_buffers);
  for (uint32_t i = 0; i < num_buffers; ++i)
    {
    writes[i] = {};
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = set;
    writes[i].dstBinding = i;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].descriptorCount = 1;
    writes[i].pBufferInfo = &infos[i];
    }

  coot_wrapper(vkUpdateDescriptorSets)(rt.get_device(), num_buffers, writes.data(), 0, nullptr);

  return set;
  }


inline
void
dispatch(runtime_t& rt, pipeline_t& pipe, VkDescriptorSet set, const void* push_data, const size_t push_size, const uint32_t gx, const uint32_t gy, const uint32_t gz)
  {
  VkCommandBuffer cmd = rt.begin_commands();

  coot_wrapper(vkCmdBindPipeline)(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
  coot_wrapper(vkCmdBindDescriptorSets)(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.layout, 0, 1, &set, 0, nullptr);
  coot_wrapper(vkCmdPushConstants)(cmd, pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t) push_size, push_data);
  coot_wrapper(vkCmdDispatch)(cmd, gx, gy, gz);

  rt.end_commands(cmd);
  }



template<template<typename> class push_t, typename F>
inline
void
dispatch_push(runtime_t& rt, pipeline_t& pipe, VkDescriptorSet set, F&& fill_push, const uint32_t gx, const uint32_t gy, const uint32_t gz)
  {
  if (rt.has_sizet64())
    {
    push_t<u64> push{};
    fill_push(push);
    dispatch(rt, pipe, set, &push, sizeof(push), gx, gy, gz);
    }
  else
    {
    push_t<u32> push{};
    fill_push(push);
    dispatch(rt, pipe, set, &push, sizeof(push), gx, gy, gz);
    }
  }
