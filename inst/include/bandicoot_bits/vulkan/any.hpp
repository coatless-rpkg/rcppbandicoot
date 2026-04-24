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
bool
any_vec(const dev_mem_t<eT1> mem, const uword n_elem, const eT2 val, const twoway_kernel_id::enum_id num, const twoway_kernel_id::enum_id num_small)
  {
  coot_debug_sigprint();
  coot_ignore(num_small);

  if (n_elem == 0) { return false; }

  runtime_t& rt = get_rt().vk_rt;

  const eT1* ptr = reinterpret_cast<const eT1*>(rt.get_pool_mapped()) + mem.vk_mem_ptr.offset;

  bool found = false;

  if (num == twoway_kernel_id::rel_any_neq || num == twoway_kernel_id::rel_any_neq_small)
    {
    const eT1 casted_val = eT1(val);
    for (uword i = 0; i < n_elem; ++i)
      {
      if (ptr[i] != casted_val) { found = true; break; }
      }
    }

  return found;
  }



template<typename eT>
inline
bool
any_vec(const dev_mem_t<eT> mem, const uword n_elem, const eT val, const oneway_real_kernel_id::enum_id num, const oneway_real_kernel_id::enum_id num_small)
  {
  coot_debug_sigprint();
  coot_ignore(val);
  coot_ignore(num_small);

  if (n_elem == 0) { return false; }

  runtime_t& rt = get_rt().vk_rt;

  const eT* ptr = reinterpret_cast<const eT*>(rt.get_pool_mapped()) + mem.vk_mem_ptr.offset;

  bool found = false;

  if (num == oneway_real_kernel_id::rel_any_nan || num == oneway_real_kernel_id::rel_any_nan_small)
    {
    for (uword i = 0; i < n_elem; ++i)
      {
      if (coot_isnan(ptr[i])) { found = true; break; }
      }
    }
  else if (num == oneway_real_kernel_id::rel_any_inf || num == oneway_real_kernel_id::rel_any_inf_small)
    {
    for (uword i = 0; i < n_elem; ++i)
      {
      if (coot_isinf(ptr[i])) { found = true; break; }
      }
    }
  else if (num == oneway_real_kernel_id::rel_any_nonfinite || num == oneway_real_kernel_id::rel_any_nonfinite_small)
    {
    for (uword i = 0; i < n_elem; ++i)
      {
      if (coot_isnan(ptr[i]) || coot_isinf(ptr[i])) { found = true; break; }
      }
    }

  return found;
  }
