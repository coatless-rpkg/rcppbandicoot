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




template<typename eT1>
inline
void
relational_unary_array_op(dev_mem_t<uword> out_mem,
                          const dev_mem_t<eT1> in_mem,
                          const uword n_elem,
                          const oneway_real_kernel_id::enum_id num,
                          const std::string& name)
  {
  coot_debug_sigprint();

  if (n_elem == 0) { return; }

  if (!(is_float<eT1>::value || is_double<eT1>::value))
    {
    coot_stop_runtime_error("coot::vulkan::relational_unary_array_op(): unsupported type for " + name);
    }

  runtime_t& rt = get_rt().vk_rt;

  pipeline_t* pipe_ptr = nullptr;
  if (num == oneway_real_kernel_id::rel_isfinite)
    {
    pipe_ptr = &rt.get_gen_rel_isfinite_pipeline<eT1>();
    }
  else if (num == oneway_real_kernel_id::rel_isnonfinite)
    {
    pipe_ptr = &rt.get_gen_rel_isinf_pipeline<eT1>();
    }
  else if (num == oneway_real_kernel_id::rel_isnan)
    {
    pipe_ptr = &rt.get_gen_rel_isnan_pipeline<eT1>();
    }
  else
    {
    coot_stop_runtime_error("coot::vulkan::relational_unary_array_op(): unsupported kernel for " + name);
    return;
    }

  pipeline_t& pipe = *pipe_ptr;

  VkDescriptorBufferInfo out_info{};
  out_info.buffer = out_mem.vk_mem_ptr.buffer;
  out_info.offset = 0;
  out_info.range = VK_WHOLE_SIZE;

  VkDescriptorBufferInfo in_info{};
  in_info.buffer = in_mem.vk_mem_ptr.buffer;
  in_info.offset = 0;
  in_info.range = VK_WHOLE_SIZE;

  std::array<VkDescriptorBufferInfo, 2> infos = { out_info, in_info };
  VkDescriptorSet set = create_descriptor_set(rt, pipe, infos.data(), (uint32_t) infos.size());

  const uint32_t gx = (uint32_t) ((n_elem + 255) / 256);
  dispatch_push<rel_scalar_push_t>(rt, pipe, set, [&](auto& push) {
    push.out_offset = out_mem.vk_mem_ptr.offset;
    push.in_offset = in_mem.vk_mem_ptr.offset;
    push.n_elem = n_elem;
  }, gx, 1, 1);
  }
