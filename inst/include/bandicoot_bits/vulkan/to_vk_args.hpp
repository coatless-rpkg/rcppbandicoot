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



struct vk_dispatch_args
  {
  std::vector<VkDescriptorBufferInfo> buffer_infos;
  std::vector<u64> push_data;
  };



inline
void
to_vk_arg(vk_dispatch_args& args, const uword& val)
  {
  args.push_data.push_back(static_cast<u64>(val));
  }



template<typename eT>
inline
void
to_vk_arg(vk_dispatch_args& args, dev_mem_t<eT>& mem)
  {
  VkDescriptorBufferInfo info{};
  info.buffer = mem.vk_mem_ptr.buffer;
  info.offset = 0;
  info.range = VK_WHOLE_SIZE;
  args.buffer_infos.push_back(info);
  args.push_data.push_back(static_cast<u64>(mem.vk_mem_ptr.offset));
  }



template<size_t idx, typename... Ts>
inline
typename
enable_if2
  <
  idx == sizeof...(Ts),
  void
  >::result
to_vk_args_unpack(vk_dispatch_args& args, const std::tuple<Ts...>& t)
  {
  coot_ignore(args);
  coot_ignore(t);
  }



template<size_t idx, typename... Ts>
inline
typename
enable_if2
  <
  (idx < sizeof...(Ts)),
  void
  >::result
to_vk_args_unpack(vk_dispatch_args& args, const std::tuple<Ts...>& t)
  {
  to_vk_arg(args, std::get<idx>(t));
  to_vk_args_unpack<idx + 1>(args, t);
  }



template<typename T>
inline
void
to_vk_args(vk_dispatch_args& args, const Proxy<T>& proxy)
  {
  to_vk_args_unpack<0>(args, proxy.args());
  }



template<typename T1, typename T2>
inline
vk_dispatch_args
to_vk_args(const Proxy<T1>& p1, const Proxy<T2>& p2)
  {
  vk_dispatch_args args;
  to_vk_args(args, p1);
  to_vk_args(args, p2);
  return args;
  }



template<typename T>
inline
vk_dispatch_args
to_vk_args(const Proxy<T>& proxy)
  {
  vk_dispatch_args args;
  to_vk_args(args, proxy);
  return args;
  }
