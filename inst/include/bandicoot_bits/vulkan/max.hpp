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
void
max(dev_mem_t<eT2> dest,
    const dev_mem_t<eT1> src,
    const uword n_rows,
    const uword n_cols,
    const uword dim,
    const bool post_conv_apply,
    const uword dest_offset,
    const uword dest_mem_incr,
    const uword src_row_offset,
    const uword src_col_offset,
    const uword src_M_n_rows)
  {
  coot_debug_sigprint();

  if (!(is_float<eT1>::value || is_double<eT1>::value))
    {
    coot_stop_runtime_error("coot::vulkan::max(): unsupported type");
    }

  runtime_t& rt = get_rt().vk_rt;

  const uword src_offset = src.vk_mem_ptr.offset + src_row_offset + src_col_offset * src_M_n_rows;
  const uword full_dest_offset = dest.vk_mem_ptr.offset + dest_offset;

  if (dim == 0)
    {
    pipeline_t& pipe = rt.get_gen_max_colwise_pipeline<eT1>();

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

    const uint32_t gx = (uint32_t) ((n_cols + 63) / 64);

    dispatch_push<reduce_dim_push_t>(rt, pipe, set, [&](auto& push) {
      push.dest_offset = full_dest_offset;
      push.src_offset = src_offset;
      push.n_rows = n_rows;
      push.n_cols = n_cols;
      push.dest_mem_incr = dest_mem_incr;
      push.src_M_n_rows = src_M_n_rows;
    }, gx, 1, 1);
    }
  else
    {
    pipeline_t& pipe = rt.get_gen_max_rowwise_pipeline<eT1>();

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

    const uint32_t gx = (uint32_t) ((n_rows + 63) / 64);

    dispatch_push<reduce_dim_push_t>(rt, pipe, set, [&](auto& push) {
      push.dest_offset = full_dest_offset;
      push.src_offset = src_offset;
      push.n_rows = n_rows;
      push.n_cols = n_cols;
      push.dest_mem_incr = dest_mem_incr;
      push.src_M_n_rows = src_M_n_rows;
    }, gx, 1, 1);
    }
  }



template<typename eT>
inline
eT
max_vec(dev_mem_t<eT> mem, const uword n_elem)
  {
  coot_debug_sigprint();

  if (n_elem == 0)
    {
    return eT(0);
    }

  runtime_t& rt = get_rt().vk_rt;

  if (!(is_float<eT>::value || is_double<eT>::value))
    {
    const eT* ptr = reinterpret_cast<const eT*>(rt.get_pool_mapped()) + mem.vk_mem_ptr.offset;
    eT result = ptr[0];
    for (uword i = 1; i < n_elem; ++i)
      {
      if (ptr[i] > result) { result = ptr[i]; }
      }
    return result;
    }

  coot_vk_mem out_mem = rt.acquire_memory<eT>(1);

  pipeline_t& pipe = rt.get_gen_max_vec_pipeline<eT>();

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



template<typename eT1, typename eT2>
inline
void
max_cube_col(dev_mem_t<eT2> dest,
             const dev_mem_t<eT1> src,
             const uword n_rows,
             const uword n_cols,
             const uword n_slices,
             const bool post_conv_apply)
  {
  coot_debug_sigprint();

  max(dest, src,
      n_rows, n_cols * n_slices,
      0, post_conv_apply,
      0, 1,
      0, 0, n_rows);
  }
