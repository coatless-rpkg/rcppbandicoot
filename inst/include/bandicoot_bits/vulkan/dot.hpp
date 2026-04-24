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
typename promote_type<eT1, eT2>::result
dot(dev_mem_t<eT1> mem1, dev_mem_t<eT2> mem2, const uword n_elem)
  {
  coot_debug_sigprint();

  typedef typename promote_type<eT1, eT2>::result promoted_eT;

  if (n_elem == 0)
    {
    return promoted_eT(0);
    }

  runtime_t& rt = get_rt().vk_rt;

  if (!((is_float<eT1>::value && is_float<eT2>::value) || (is_double<eT1>::value && is_double<eT2>::value)))
    {
    coot_stop_runtime_error("coot::vulkan::dot(): unsupported type combination");
    }

  coot_vk_mem out_mem = rt.acquire_memory<promoted_eT>(1);

  pipeline_t& pipe = rt.get_gen_dot_pipeline<eT1>();

  VkDescriptorBufferInfo out_info{};
  out_info.buffer = out_mem.buffer;
  out_info.offset = 0;
  out_info.range = VK_WHOLE_SIZE;

  VkDescriptorBufferInfo in1_info{};
  in1_info.buffer = mem1.vk_mem_ptr.buffer;
  in1_info.offset = 0;
  in1_info.range = VK_WHOLE_SIZE;

  VkDescriptorBufferInfo in2_info{};
  in2_info.buffer = mem2.vk_mem_ptr.buffer;
  in2_info.offset = 0;
  in2_info.range = VK_WHOLE_SIZE;

  std::array<VkDescriptorBufferInfo, 3> infos = { out_info, in1_info, in2_info };
  VkDescriptorSet set = create_descriptor_set(rt, pipe, infos.data(), (uint32_t) infos.size());

  dispatch_push<dot_push_t>(rt, pipe, set, [&](auto& push) {
    push.out_offset = out_mem.offset;
    push.in1_offset = mem1.vk_mem_ptr.offset;
    push.in2_offset = mem2.vk_mem_ptr.offset;
    push.n_elem = n_elem;
  }, 1, 1, 1);

  promoted_eT host_val = promoted_eT(0);
  std::memcpy(&host_val, reinterpret_cast<promoted_eT*>(rt.get_pool_mapped()) + out_mem.offset, sizeof(promoted_eT));

  rt.release_memory(out_mem);

  return host_val;
  }
