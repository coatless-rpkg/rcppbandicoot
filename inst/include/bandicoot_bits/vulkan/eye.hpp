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
eye(dev_mem_t<eT> dest, const uword n_rows, const uword n_cols)
  {
  coot_debug_sigprint();

  runtime_t& rt = get_rt().vk_rt;

  pipeline_t& pipe = rt.get_gen_eye_pipeline<eT>();

  VkDescriptorBufferInfo out_info{};
  out_info.buffer = dest.vk_mem_ptr.buffer;
  out_info.offset = 0;
  out_info.range = VK_WHOLE_SIZE;

  std::array<VkDescriptorBufferInfo, 1> infos = { out_info };
  VkDescriptorSet set = create_descriptor_set(rt, pipe, infos.data(), (uint32_t) infos.size());

  const uint32_t gx = (uint32_t) ((n_rows + 15) / 16);
  const uint32_t gy = (uint32_t) ((n_cols + 15) / 16);

  dispatch_push<eye_push_t>(rt, pipe, set, [&](auto& push) {
    push.out_offset = dest.vk_mem_ptr.offset;
    push.n_rows = n_rows;
    push.n_cols = n_cols;
  }, gx, gy, 1);
  }
