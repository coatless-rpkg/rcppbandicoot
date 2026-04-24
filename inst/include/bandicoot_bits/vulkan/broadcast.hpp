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
broadcast_op(const twoway_kernel_id::enum_id num,
             dev_mem_t<eT2> dest,
             const dev_mem_t<eT2> dest_in,
             const dev_mem_t<eT1> src,
             const uword src_n_rows,
             const uword src_n_cols,
             const uword copies_per_row,
             const uword copies_per_col,
             const uword dest_row_offset,
             const uword dest_col_offset,
             const uword dest_M_n_rows,
             const uword dest_in_row_offset,
             const uword dest_in_col_offset,
             const uword dest_in_M_n_rows,
             const uword src_row_offset,
             const uword src_col_offset,
             const uword src_M_n_rows)
  {
  coot_debug_sigprint();

  if (!(is_float<eT1>::value || is_double<eT1>::value) ||
      !(is_float<eT2>::value || is_double<eT2>::value))
    {
    coot_stop_runtime_error("coot::vulkan::broadcast_op(): unsupported type");
    }

  runtime_t& rt = get_rt().vk_rt;

  pipeline_t* pipe_ptr = nullptr;
  switch (num)
    {
    case twoway_kernel_id::broadcast_set:        pipe_ptr = &rt.get_gen_broadcast_set_pipeline<eT1, eT2>();        break;
    case twoway_kernel_id::broadcast_plus:       pipe_ptr = &rt.get_gen_broadcast_plus_pipeline<eT1, eT2>();       break;
    case twoway_kernel_id::broadcast_minus_pre:  pipe_ptr = &rt.get_gen_broadcast_minus_pre_pipeline<eT1, eT2>();  break;
    case twoway_kernel_id::broadcast_minus_post: pipe_ptr = &rt.get_gen_broadcast_minus_post_pipeline<eT1, eT2>(); break;
    case twoway_kernel_id::broadcast_schur:      pipe_ptr = &rt.get_gen_broadcast_schur_pipeline<eT1, eT2>();      break;
    case twoway_kernel_id::broadcast_div_pre:    pipe_ptr = &rt.get_gen_broadcast_div_pre_pipeline<eT1, eT2>();    break;
    case twoway_kernel_id::broadcast_div_post:   pipe_ptr = &rt.get_gen_broadcast_div_post_pipeline<eT1, eT2>();   break;
    default:
      coot_stop_runtime_error("coot::vulkan::broadcast_op(): unsupported operation");
      return;
    }

  pipeline_t& pipe = *pipe_ptr;

  const uword dest_offset = dest.vk_mem_ptr.offset + dest_row_offset + dest_col_offset * dest_M_n_rows;
  const uword dest_in_offset = dest_in.vk_mem_ptr.offset + dest_in_row_offset + dest_in_col_offset * dest_in_M_n_rows;
  const uword src_offset = src.vk_mem_ptr.offset + src_row_offset + src_col_offset * src_M_n_rows;

  VkDescriptorBufferInfo out_info{};
  out_info.buffer = dest.vk_mem_ptr.buffer;
  out_info.offset = 0;
  out_info.range = VK_WHOLE_SIZE;

  VkDescriptorBufferInfo src_info{};
  src_info.buffer = src.vk_mem_ptr.buffer;
  src_info.offset = 0;
  src_info.range = VK_WHOLE_SIZE;

  std::array<VkDescriptorBufferInfo, 2> infos = { out_info, src_info };
  VkDescriptorSet set = create_descriptor_set(rt, pipe, infos.data(), (uint32_t) infos.size());

  const uword new_n_rows = src_n_rows * copies_per_row;
  const uword new_n_cols = src_n_cols * copies_per_col;
  const uint32_t gx = (uint32_t) ((new_n_rows + 15) / 16);
  const uint32_t gy = (uint32_t) ((new_n_cols + 15) / 16);

  dispatch_push<broadcast_push_t>(rt, pipe, set, [&](auto& push) {
    push.dest_offset = dest_offset;
    push.dest_in_offset = dest_in_offset;
    push.src_offset = src_offset;
    push.src_n_rows = src_n_rows;
    push.src_n_cols = src_n_cols;
    push.copies_per_row = copies_per_row;
    push.copies_per_col = copies_per_col;
    push.dest_M_n_rows = dest_M_n_rows;
    push.dest_in_M_n_rows = dest_in_M_n_rows;
    push.src_M_n_rows = src_M_n_rows;
  }, gx, gy, 1);
  }
