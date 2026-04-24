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



template<bool conj, typename eT>
inline
void
trans(dev_mem_t<eT> dest, const dev_mem_t<eT> src, const uword n_rows, const uword n_cols)
  {
  coot_debug_sigprint();

  if (n_rows == 0 || n_cols == 0) { return; }

  if (!(is_float<eT>::value || is_double<eT>::value))
    {
    coot_stop_runtime_error("coot::vulkan::trans(): unsupported type");
    }

  runtime_t& rt = get_rt().vk_rt;

  pipeline_t& pipe = rt.get_gen_trans_pipeline<eT>();

  VkDescriptorBufferInfo out_info{};
  out_info.buffer = dest.vk_mem_ptr.buffer;
  out_info.offset = 0;
  out_info.range = VK_WHOLE_SIZE;

  VkDescriptorBufferInfo in_info{};
  in_info.buffer = src.vk_mem_ptr.buffer;
  in_info.offset = 0;
  in_info.range = VK_WHOLE_SIZE;

  std::array<VkDescriptorBufferInfo, 2> infos = { out_info, in_info };
  VkDescriptorSet set = create_descriptor_set(rt, pipe, infos.data(), (uint32_t) infos.size());

  const uint32_t gx = (uint32_t) ((n_rows + 15) / 16);
  const uint32_t gy = (uint32_t) ((n_cols + 15) / 16);

  dispatch_push<trans_push_t>(rt, pipe, set, [&](auto& push) {
    push.dest_offset = dest.vk_mem_ptr.offset;
    push.src_offset = src.vk_mem_ptr.offset;
    push.n_rows = n_rows;
    push.n_cols = n_cols;
  }, gx, gy, 1);
  }
