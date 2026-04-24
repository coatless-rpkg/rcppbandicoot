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




template<typename eT1, typename eT2>
inline
bool
all_vec(const dev_mem_t<eT1> mem, const uword n_elem, const eT2 val, const twoway_kernel_id::enum_id /* num */, const twoway_kernel_id::enum_id /* num_small */)
  {
  coot_debug_sigprint();

  if (n_elem == 0)
    {
    return true;
    }

  runtime_t& rt = get_rt().vk_rt;

  if (!(is_float<eT1>::value || is_double<eT1>::value || is_uword<eT1>::value))
    {
    coot_stop_runtime_error("coot::vulkan::all_vec(): unsupported type");
    }

  coot_vk_mem out_mem = rt.acquire_memory<uword>(1);
  const eT1 val_as_eT1 = static_cast<eT1>(val);
  coot_vk_mem val_mem = rt.acquire_memory<eT1>(1);
  std::memcpy(reinterpret_cast<eT1*>(rt.get_pool_mapped()) + val_mem.offset, &val_as_eT1, sizeof(eT1));

  pipeline_t& pipe = rt.get_gen_all_neq_vec_pipeline<eT1>();

  VkDescriptorBufferInfo out_info{};
  out_info.buffer = out_mem.buffer;
  out_info.offset = 0;
  out_info.range = VK_WHOLE_SIZE;

  VkDescriptorBufferInfo in_info{};
  in_info.buffer = mem.vk_mem_ptr.buffer;
  in_info.offset = 0;
  in_info.range = VK_WHOLE_SIZE;

  VkDescriptorBufferInfo val_info{};
  val_info.buffer = val_mem.buffer;
  val_info.offset = 0;
  val_info.range = VK_WHOLE_SIZE;

  std::array<VkDescriptorBufferInfo, 3> infos = { out_info, in_info, val_info };
  VkDescriptorSet set = create_descriptor_set(rt, pipe, infos.data(), (uint32_t) infos.size());

  dispatch_push<all_vec_push_t>(rt, pipe, set, [&](auto& push) {
    push.out_offset = out_mem.offset;
    push.in_offset = mem.vk_mem_ptr.offset;
    push.n_elem = n_elem;
    push.val_offset = val_mem.offset;
  }, 1, 1, 1);

  uword host_val = 0;
  std::memcpy(&host_val, reinterpret_cast<uword*>(rt.get_pool_mapped()) + out_mem.offset, sizeof(uword));

  rt.release_memory(out_mem);
  rt.release_memory(val_mem);

  return (host_val != uword(0));
  }



template<typename eT1, typename eT2>
inline
void
all(dev_mem_t<uword> out_mem, const dev_mem_t<eT1> in_mem, const uword n_rows, const uword n_cols, const eT2 val, const twoway_kernel_id::enum_id num, const bool colwise)
  {
  coot_debug_sigprint();

  if (n_rows == 0 || n_cols == 0)
    {
    return;
    }

  if (num != twoway_kernel_id::rel_all_neq_colwise && num != twoway_kernel_id::rel_all_neq_rowwise)
    {
    coot_stop_runtime_error("coot::vulkan::all(): unsupported kernel");
    }

  runtime_t& rt = get_rt().vk_rt;

  if (!(is_float<eT1>::value || is_double<eT1>::value || is_uword<eT1>::value))
    {
    coot_stop_runtime_error("coot::vulkan::all(): unsupported type");
    }

  coot_vk_mem val_mem = rt.acquire_memory<eT2>(1);
  std::memcpy(reinterpret_cast<eT2*>(rt.get_pool_mapped()) + val_mem.offset, &val, sizeof(eT2));

  pipeline_t& pipe = rt.get_gen_all_neq_pipeline<eT1>(colwise);

  VkDescriptorBufferInfo out_info{};
  out_info.buffer = out_mem.vk_mem_ptr.buffer;
  out_info.offset = 0;
  out_info.range = VK_WHOLE_SIZE;

  VkDescriptorBufferInfo in_info{};
  in_info.buffer = in_mem.vk_mem_ptr.buffer;
  in_info.offset = 0;
  in_info.range = VK_WHOLE_SIZE;

  VkDescriptorBufferInfo val_info{};
  val_info.buffer = val_mem.buffer;
  val_info.offset = 0;
  val_info.range = VK_WHOLE_SIZE;

  std::array<VkDescriptorBufferInfo, 3> infos = { out_info, in_info, val_info };
  VkDescriptorSet set = create_descriptor_set(rt, pipe, infos.data(), (uint32_t) infos.size());

  const uword work_items = colwise ? n_cols : n_rows;
  const uint32_t gx = (uint32_t) ((work_items + 255) / 256);

  dispatch_push<all_push_t>(rt, pipe, set, [&](auto& push) {
    push.out_offset = out_mem.vk_mem_ptr.offset;
    push.in_offset = in_mem.vk_mem_ptr.offset;
    push.n_rows = n_rows;
    push.n_cols = n_cols;
    push.val_offset = val_mem.offset;
  }, gx, 1, 1);

  rt.release_memory(val_mem);
  }
