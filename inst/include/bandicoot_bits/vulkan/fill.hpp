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



template<typename T1>
inline
void
fill(const Proxy<T1>& dest, const typename T1::elem_type val)
  {
  coot_debug_sigprint();

  typedef typename T1::elem_type eT;

  if (dest.is_empty())
    {
    return;
    }

  runtime_t& rt = get_rt().vk_rt;

  const dev_mem_t<eT>& out_mem = std::get<0>(dest.args());

  const uword n_rows = dest.get_n_rows();
  const uword n_cols = dest.get_n_cols();
  const uword M_n_rows = dest.get_M_n_rows();
  const uword n_slices = dest.get_n_slices();

  uword M_n_elem_slice;
  if constexpr (Proxy<T1>::num_dims >= 3)
    M_n_elem_slice = dest.get_M_n_elem_slice();
  else
    M_n_elem_slice = M_n_rows * n_cols;

  coot_vk_mem val_mem = rt.acquire_memory<eT>(1);
  std::memcpy(reinterpret_cast<eT*>(rt.get_pool_mapped()) + val_mem.offset, &val, sizeof(eT));

  pipeline_t& pipe = rt.get_gen_fill_pipeline<eT>();

  VkDescriptorBufferInfo out_info{};
  out_info.buffer = out_mem.vk_mem_ptr.buffer;
  out_info.offset = 0;
  out_info.range = VK_WHOLE_SIZE;

  VkDescriptorBufferInfo val_info{};
  val_info.buffer = val_mem.buffer;
  val_info.offset = 0;
  val_info.range = VK_WHOLE_SIZE;

  std::array<VkDescriptorBufferInfo, 2> infos = { out_info, val_info };
  VkDescriptorSet set = create_descriptor_set(rt, pipe, infos.data(), (uint32_t) infos.size());

  const uint32_t gx = (uint32_t) ((n_rows + 15) / 16);
  const uint32_t gy = (uint32_t) ((n_cols + 15) / 16);
  const uint32_t gz = (uint32_t) n_slices;

  dispatch_push<fill_push_t>(rt, pipe, set, [&](auto& push) {
    push.out_offset = out_mem.vk_mem_ptr.offset;
    push.n_rows = n_rows;
    push.n_cols = n_cols;
    push.M_n_rows = M_n_rows;
    push.n_slices = n_slices;
    push.M_n_elem_slice = M_n_elem_slice;
    push.val_offset = val_mem.offset;
  }, gx, gy, gz);

  rt.release_memory(val_mem);
  }
