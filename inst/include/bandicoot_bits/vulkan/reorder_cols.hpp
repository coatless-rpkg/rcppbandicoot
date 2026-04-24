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



template<typename eT>
inline
void
reorder_cols(dev_mem_t<eT> out,
             const dev_mem_t<eT> mem,
             const uword n_rows,
             const dev_mem_t<uword> order,
             const uword out_n_cols)
  {
  coot_debug_sigprint();

  if (n_rows == 0 || out_n_cols == 0) { return; }

  if (!(is_float<eT>::value || is_double<eT>::value))
    {
    coot_stop_runtime_error("coot::vulkan::reorder_cols(): unsupported type");
    }

  runtime_t& rt = get_rt().vk_rt;

  pipeline_t& pipe = rt.get_gen_reorder_cols_pipeline<eT>();

  VkDescriptorBufferInfo out_info{};
  out_info.buffer = out.vk_mem_ptr.buffer;
  out_info.offset = 0;
  out_info.range = VK_WHOLE_SIZE;

  VkDescriptorBufferInfo in_info{};
  in_info.buffer = mem.vk_mem_ptr.buffer;
  in_info.offset = 0;
  in_info.range = VK_WHOLE_SIZE;

  VkDescriptorBufferInfo order_info{};
  order_info.buffer = order.vk_mem_ptr.buffer;
  order_info.offset = 0;
  order_info.range = VK_WHOLE_SIZE;

  std::array<VkDescriptorBufferInfo, 3> infos = { out_info, in_info, order_info };
  VkDescriptorSet set = create_descriptor_set(rt, pipe, infos.data(), (uint32_t) infos.size());

  const uint32_t gx = (uint32_t) ((out_n_cols + 63) / 64);

  dispatch_push<reorder_cols_push_t>(rt, pipe, set, [&](auto& push) {
    push.out_offset = out.vk_mem_ptr.offset;
    push.in_offset = mem.vk_mem_ptr.offset;
    push.order_offset = order.vk_mem_ptr.offset;
    push.n_rows = n_rows;
    push.out_n_cols = out_n_cols;
  }, gx, 1, 1);
  }
