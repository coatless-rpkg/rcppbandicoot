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
copy_from_dev_mem(eT* dest,
                  const dev_mem_t<eT> src,
                  const uword n_rows,
                  const uword n_cols,
                  const uword src_row_offset,
                  const uword src_col_offset,
                  const uword src_M_n_rows)
  {
  coot_debug_sigprint();

  if (n_rows == 0 || n_cols == 0)
    {
    return;
    }

  runtime_t& rt = get_rt().vk_rt;

  const eT* src_ptr = reinterpret_cast<const eT*>(rt.get_pool_mapped()) + src.vk_mem_ptr.offset;

  for (uword col = 0; col < n_cols; ++col)
    {
    const uword src_col = src_col_offset + col;
    const uword dest_col_offset = col * n_rows;
    const uword src_col_offset_e = src_col * src_M_n_rows + src_row_offset;
    for (uword row = 0; row < n_rows; ++row)
      {
      dest[dest_col_offset + row] = src_ptr[src_col_offset_e + row];
      }
    }
  }


template<typename eT>
inline
void
copy_into_dev_mem(dev_mem_t<eT> dest, const eT* src, const uword N)
  {
  coot_debug_sigprint();

  if (N == 0)
    {
    return;
    }

  runtime_t& rt = get_rt().vk_rt;

  eT* dest_ptr = reinterpret_cast<eT*>(rt.get_pool_mapped()) + dest.vk_mem_ptr.offset;
  std::memcpy(dest_ptr, src, sizeof(eT) * N);
  }



template<typename T1, typename eT_in, typename TI>
inline
void
copy(const Proxy<T1>& out, const Proxy<subview_elem1<eT_in, TI>>& in)
  {
  coot_debug_sigprint();

  typedef typename T1::elem_type eT_out;

  if (in.is_empty())
    return;

  runtime_t& rt = get_rt().vk_rt;

  const dev_mem_t<eT_out>& out_dev_mem = std::get<0>(out.args());
  const dev_mem_t<eT_in>& in_data_mem = std::get<0>(in.args());
  const dev_mem_t<uword>& in_idx_mem = std::get<1>(in.args());
  const uword n_elem = in.get_n_elem();

  pipeline_t& pipe = rt.get_gen_gather_pipeline<eT_in, eT_out>();

  VkDescriptorBufferInfo out_info{};
  out_info.buffer = out_dev_mem.vk_mem_ptr.buffer;
  out_info.offset = 0;
  out_info.range = VK_WHOLE_SIZE;

  VkDescriptorBufferInfo in_info{};
  in_info.buffer = in_data_mem.vk_mem_ptr.buffer;
  in_info.offset = 0;
  in_info.range = VK_WHOLE_SIZE;

  VkDescriptorBufferInfo idx_info{};
  idx_info.buffer = in_idx_mem.vk_mem_ptr.buffer;
  idx_info.offset = 0;
  idx_info.range = VK_WHOLE_SIZE;

  std::array<VkDescriptorBufferInfo, 3> infos = { out_info, in_info, idx_info };
  VkDescriptorSet set = create_descriptor_set(rt, pipe, infos.data(), 3);

  const uint32_t gx = (uint32_t) ((n_elem + 255) / 256);
  dispatch_push<gather_push_t>(rt, pipe, set, [&](auto& push) {
    push.out_offset = out_dev_mem.vk_mem_ptr.offset;
    push.in_offset = in_data_mem.vk_mem_ptr.offset;
    push.idx_offset = in_idx_mem.vk_mem_ptr.offset;
    push.n_elem = n_elem;
  }, gx, 1, 1);
  }



template<typename T1, typename T2>
inline
void
copy(const Proxy<T1>& out, const Proxy<T2>& in)
  {
  coot_debug_sigprint();

  if (in.is_empty())
    {
    return;
    }

  coot_static_check( Proxy<T1>::num_dims != Proxy<T2>::num_dims, "coot::vulkan::copy(): objects must have the same number of dimensions" );

  runtime_t& rt = get_rt().vk_rt;

  pipeline_t& pipe = rt.get_kernel<kernel_id::copy, Proxy<T1>, Proxy<T2>>();

  std::vector<VkDescriptorBufferInfo> infos;
  infos.reserve(2);
  collect_dev_mem_infos(infos, out, in);

  VkDescriptorSet set = create_descriptor_set(rt, pipe, infos.data(), (uint32_t) infos.size());

  uint8_t push_buf[256];
  size_t push_off = 0;
  if (rt.has_sizet64())
    {
    fill_push_proxies<u64>(push_buf, push_off, out, in);
    }
  else
    {
    fill_push_proxies<u32>(push_buf, push_off, out, in);
    }

  const uword out_n_rows = out.get_n_rows();
  const uword out_n_cols = out.get_n_cols();
  const uword out_n_slices = out.get_n_slices();

  uint32_t gx, gy, gz;
  if constexpr (Proxy<T1>::num_dims == 1)
    {
    gx = (uint32_t) ((out.get_n_elem() + 15) / 16);
    gy = 1;
    gz = 1;
    }
  else
    {
    gx = (uint32_t) ((out_n_rows + 15) / 16);
    gy = (uint32_t) ((out_n_cols + 15) / 16);
    gz = (uint32_t) out_n_slices;
    }

  dispatch(rt, pipe, set, push_buf, push_off, gx, gy, gz);
  }
