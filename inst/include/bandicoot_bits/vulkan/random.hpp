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
fill_randu(dev_mem_t<eT> dest, const uword n)
  {
  coot_debug_sigprint();

  if (n == 0) { return; }

  runtime_t& rt = get_rt().vk_rt;

  std::mt19937_64 rng(rt.next_rng_seed());
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  eT* out_ptr = reinterpret_cast<eT*>(rt.get_pool_mapped()) + dest.vk_mem_ptr.offset;
  for (uword i = 0; i < n; ++i)
    {
    out_ptr[i] = eT(dist(rng));
    }
  }



template<typename eT>
inline
void
fill_randn(dev_mem_t<eT> dest, const uword n, const double mu, const double sd)
  {
  coot_debug_sigprint();

  if (n == 0) { return; }

  runtime_t& rt = get_rt().vk_rt;

  std::mt19937_64 rng(rt.next_rng_seed());
  std::normal_distribution<double> dist(mu, sd);

  eT* out_ptr = reinterpret_cast<eT*>(rt.get_pool_mapped()) + dest.vk_mem_ptr.offset;
  for (uword i = 0; i < n; ++i)
    {
    out_ptr[i] = eT(dist(rng));
    }
  }



template<typename eT>
inline
void
fill_randi(dev_mem_t<eT> dest, const uword n, const int lo, const int hi)
  {
  coot_debug_sigprint();

  if (n == 0) { return; }

  runtime_t& rt = get_rt().vk_rt;

  std::mt19937_64 rng(rt.next_rng_seed());
  std::uniform_int_distribution<int> dist(lo, hi);

  eT* out_ptr = reinterpret_cast<eT*>(rt.get_pool_mapped()) + dest.vk_mem_ptr.offset;
  for (uword i = 0; i < n; ++i)
    {
    out_ptr[i] = eT(dist(rng));
    }
  }
