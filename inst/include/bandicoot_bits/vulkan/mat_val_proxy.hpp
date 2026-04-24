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
eT
get_val(const dev_mem_t<eT> mem, const uword index)
  {
  coot_debug_sigprint();

  runtime_t& rt = get_rt().vk_rt;

  const eT* ptr = reinterpret_cast<const eT*>(rt.get_pool_mapped()) + mem.vk_mem_ptr.offset;
  eT val = ptr[index];

  return val;
  }



template<typename eT>
inline
void
set_val(dev_mem_t<eT> mem, const uword index, const eT in_val)
  {
  coot_debug_sigprint();

  runtime_t& rt = get_rt().vk_rt;

  eT* ptr = reinterpret_cast<eT*>(rt.get_pool_mapped()) + mem.vk_mem_ptr.offset;
  ptr[index] = in_val;
  }
