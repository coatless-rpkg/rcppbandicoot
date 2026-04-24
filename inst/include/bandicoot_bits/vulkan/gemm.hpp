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



template<const bool do_trans_A, const bool do_trans_B>
struct gemm
  {
  template<typename eT>
  static
  inline
  void apply(dev_mem_t<eT> C_mem,
             const uword C_n_rows,
             const uword C_n_cols,
             const dev_mem_t<eT> A_mem,
             const uword A_n_rows,
             const uword A_n_cols,
             const dev_mem_t<eT> B_mem,
             const eT alpha,
             const eT beta,
             const uword C_row_offset,
             const uword C_col_offset,
             const uword C_M_n_rows,
             const uword A_row_offset,
             const uword A_col_offset,
             const uword A_M_n_rows,
             const uword B_row_offset,
             const uword B_col_offset,
             const uword B_M_n_rows)
    {
    coot_debug_sigprint();

    if (!(is_float<eT>::value || is_double<eT>::value))
      {
      coot_stop_runtime_error("coot::vulkan::gemm(): unsupported type");
      }

    runtime_t& rt = get_rt().vk_rt;

    const uword K = do_trans_A ? A_n_rows : A_n_cols;

    coot_vk_mem scalars_mem = rt.acquire_memory<eT>(2);
    eT* scalars_ptr = reinterpret_cast<eT*>(rt.get_pool_mapped()) + scalars_mem.offset;
    scalars_ptr[0] = alpha;
    scalars_ptr[1] = beta;

    pipeline_t& pipe = rt.get_gen_gemm_pipeline<eT, do_trans_A, do_trans_B>();

    VkDescriptorBufferInfo c_info{};
    c_info.buffer = C_mem.vk_mem_ptr.buffer;
    c_info.offset = 0;
    c_info.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo a_info{};
    a_info.buffer = A_mem.vk_mem_ptr.buffer;
    a_info.offset = 0;
    a_info.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo b_info{};
    b_info.buffer = B_mem.vk_mem_ptr.buffer;
    b_info.offset = 0;
    b_info.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo sc_info{};
    sc_info.buffer = scalars_mem.buffer;
    sc_info.offset = 0;
    sc_info.range = VK_WHOLE_SIZE;

    std::array<VkDescriptorBufferInfo, 4> infos = { c_info, a_info, b_info, sc_info };
    VkDescriptorSet set = create_descriptor_set(rt, pipe, infos.data(), (uint32_t) infos.size());

    const uint32_t gx = (uint32_t) ((C_n_rows + 15) / 16);
    const uint32_t gy = (uint32_t) ((C_n_cols + 15) / 16);

    dispatch_push<gemm_push_t>(rt, pipe, set, [&](auto& push) {
      push.c_offset = C_mem.vk_mem_ptr.offset;
      push.c_row_offset = C_row_offset;
      push.c_col_offset = C_col_offset;
      push.c_M_n_rows = C_M_n_rows;
      push.a_offset = A_mem.vk_mem_ptr.offset;
      push.a_row_offset = A_row_offset;
      push.a_col_offset = A_col_offset;
      push.a_M_n_rows = A_M_n_rows;
      push.b_offset = B_mem.vk_mem_ptr.offset;
      push.b_row_offset = B_row_offset;
      push.b_col_offset = B_col_offset;
      push.b_M_n_rows = B_M_n_rows;
      push.c_n_rows = C_n_rows;
      push.c_n_cols = C_n_cols;
      push.K = K;
      push.scalars_offset = scalars_mem.offset;
    }, gx, gy, 1);

    rt.release_memory(scalars_mem);
    }
  };
