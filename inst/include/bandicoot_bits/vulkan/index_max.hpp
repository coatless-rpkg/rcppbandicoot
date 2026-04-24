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
index_max(dev_mem_t<uword> dest,
          const dev_mem_t<eT> src,
          const uword n_rows,
          const uword n_cols,
          const uword dim,
          const uword dest_offset,
          const uword dest_mem_incr,
          const uword src_row_offset,
          const uword src_col_offset,
          const uword src_M_n_rows)
  {
  coot_debug_sigprint();

  if (!(is_float<eT>::value || is_double<eT>::value))
    {
    coot_stop_runtime_error("coot::vulkan::index_max(): unsupported type");
    }

  runtime_t& rt = get_rt().vk_rt;

  const uword src_offset = src.vk_mem_ptr.offset + src_row_offset + src_col_offset * src_M_n_rows;

  pipeline_t& pipe = (dim == 0)
      ? rt.get_gen_index_max_colwise_pipeline<eT>()
      : rt.get_gen_index_max_rowwise_pipeline<eT>();

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

  const uint32_t gx = (dim == 0)
      ? (uint32_t) ((n_cols + 63) / 64)
      : (uint32_t) ((n_rows + 63) / 64);

  dispatch_push<index_reduce_push_t>(rt, pipe, set, [&](auto& push) {
    push.dest_offset = dest.vk_mem_ptr.offset + dest_offset;
    push.src_offset = src_offset;
    push.n_rows = n_rows;
    push.n_cols = n_cols;
    push.dest_mem_incr = dest_mem_incr;
    push.src_M_n_rows = src_M_n_rows;
  }, gx, 1, 1);
  }



template<typename eT>
inline
uword
index_max_vec(const dev_mem_t<eT> mem, const uword n_elem, eT* max_val)
  {
  coot_debug_sigprint();

  if (n_elem == 0) { if (max_val) { *max_val = eT(0); } return 0; }

  if (!(is_float<eT>::value || is_double<eT>::value))
    {
    coot_stop_runtime_error("coot::vulkan::index_max_vec(): unsupported type");
    }

  runtime_t& rt = get_rt().vk_rt;

  coot_vk_mem out_mem = rt.acquire_memory<uword>(1);
  coot_vk_mem aux_mem = rt.acquire_memory<eT>(1);

  pipeline_t& pipe = rt.get_gen_index_max_vec_pipeline<eT>();

  VkDescriptorBufferInfo out_info{};
  out_info.buffer = out_mem.buffer;
  out_info.offset = 0;
  out_info.range  = VK_WHOLE_SIZE;

  VkDescriptorBufferInfo in_info{};
  in_info.buffer = mem.vk_mem_ptr.buffer;
  in_info.offset = 0;
  in_info.range  = VK_WHOLE_SIZE;

  VkDescriptorBufferInfo aux_info{};
  aux_info.buffer = aux_mem.buffer;
  aux_info.offset = 0;
  aux_info.range  = VK_WHOLE_SIZE;

  std::array<VkDescriptorBufferInfo, 3> infos = { out_info, in_info, aux_info };
  VkDescriptorSet set = create_descriptor_set(rt, pipe, infos.data(), (uint32_t) infos.size());

  dispatch_push<index_vec_push_t>(rt, pipe, set, [&](auto& push) {
    push.out_offset = out_mem.offset;
    push.in_offset  = mem.vk_mem_ptr.offset;
    push.n_elem     = n_elem;
    push.aux_offset = aux_mem.offset;
  }, 1, 1, 1);

  uword host_idx = 0;
  eT    host_val = eT(0);

  std::memcpy(&host_idx, reinterpret_cast<uword*>(rt.get_pool_mapped()) + out_mem.offset, sizeof(uword));

  if (max_val)
    {
    std::memcpy(&host_val, reinterpret_cast<eT*>(rt.get_pool_mapped()) + aux_mem.offset, sizeof(eT));
    *max_val = host_val;
    }

  rt.release_memory(out_mem);
  rt.release_memory(aux_mem);

  return host_idx;
  }



template<typename eT>
inline
void
index_max_cube_col(dev_mem_t<uword> dest,
                   const dev_mem_t<eT> src,
                   const uword n_rows,
                   const uword n_cols,
                   const uword n_slices)
  {
  coot_debug_sigprint();

  index_max(dest, src,
            n_rows, n_cols * n_slices,
            0,
            0, 1,
            0, 0, n_rows);
  }
