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





/**
 * Accumulate all elements in `mem`.
 */
template<typename eT>
inline
eT
accu(dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_debug_sigprint();

  if (n_elem == 0)
    {
    return eT(0);
    }

  runtime_t& rt = get_rt().vk_rt;

  if (!(is_float<eT>::value || is_double<eT>::value))
    {
    coot_stop_runtime_error("coot::vulkan::accu(): unsupported type");
    }

  coot_vk_mem out_mem = rt.acquire_memory<eT>(1);

  pipeline_t& pipe = rt.get_gen_accu_pipeline<eT>();

  VkDescriptorBufferInfo out_info{};
  out_info.buffer = out_mem.buffer;
  out_info.offset = 0;
  out_info.range = VK_WHOLE_SIZE;

  VkDescriptorBufferInfo in_info{};
  in_info.buffer = mem.vk_mem_ptr.buffer;
  in_info.offset = 0;
  in_info.range = VK_WHOLE_SIZE;

  VkDescriptorBufferInfo unused_info{};
  unused_info.buffer = out_mem.buffer;
  unused_info.offset = 0;
  unused_info.range = VK_WHOLE_SIZE;

  std::array<VkDescriptorBufferInfo, 3> infos = { out_info, in_info, unused_info };
  VkDescriptorSet set = create_descriptor_set(rt, pipe, infos.data(), (uint32_t) infos.size());

  dispatch_push<accu_push_t>(rt, pipe, set, [&](auto& push) {
    push.out_offset = out_mem.offset;
    push.in_offset = mem.vk_mem_ptr.offset;
    push.n_elem = n_elem;
  }, 1, 1, 1);

  eT host_val = eT(0);
  std::memcpy(&host_val, reinterpret_cast<eT*>(rt.get_pool_mapped()) + out_mem.offset, sizeof(eT));

  rt.release_memory(out_mem);

  return host_val;
  }


/**
 * Accumulate all elements in a subview.
 */
template<typename eT>
inline
eT
accu_subview(dev_mem_t<eT> mem, const uword M_n_rows, const uword aux_row1, const uword aux_col1, const uword n_rows, const uword n_cols)
  {
  coot_debug_sigprint();

  if (n_rows == 0 || n_cols == 0)
    {
    return eT(0);
    }

  runtime_t& rt = get_rt().vk_rt;

  if (!(is_float<eT>::value || is_double<eT>::value))
    {
    coot_stop_runtime_error("coot::vulkan::accu_subview(): unsupported type");
    }

  coot_vk_mem out_mem = rt.acquire_memory<eT>(1);

  pipeline_t& pipe = rt.get_gen_accu_subview_pipeline<eT>();

  VkDescriptorBufferInfo out_info{};
  out_info.buffer = out_mem.buffer;
  out_info.offset = 0;
  out_info.range = VK_WHOLE_SIZE;

  VkDescriptorBufferInfo in_info{};
  in_info.buffer = mem.vk_mem_ptr.buffer;
  in_info.offset = 0;
  in_info.range = VK_WHOLE_SIZE;

  VkDescriptorBufferInfo unused_info{};
  unused_info.buffer = out_mem.buffer;
  unused_info.offset = 0;
  unused_info.range = VK_WHOLE_SIZE;

  std::array<VkDescriptorBufferInfo, 3> infos = { out_info, in_info, unused_info };
  VkDescriptorSet set = create_descriptor_set(rt, pipe, infos.data(), (uint32_t) infos.size());

  const uword in_off = mem.vk_mem_ptr.offset + aux_row1 + aux_col1 * M_n_rows;
  dispatch_push<accu_subview_push_t>(rt, pipe, set, [&](auto& push) {
    push.out_offset = out_mem.offset;
    push.in_offset = in_off;
    push.n_rows = n_rows;
    push.n_cols = n_cols;
    push.M_n_rows = M_n_rows;
  }, 1, 1, 1);

  eT host_val = eT(0);
  std::memcpy(&host_val, reinterpret_cast<eT*>(rt.get_pool_mapped()) + out_mem.offset, sizeof(eT));

  rt.release_memory(out_mem);

  return host_val;
  }
