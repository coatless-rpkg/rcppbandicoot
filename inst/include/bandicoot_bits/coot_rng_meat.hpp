// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2019 Ryan Curtin <ryan@ratml.org>
// Copyright 2008-2023 Conrad Sanderson (https://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
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
inline void coot_rng::fill_randu(dev_mem_t<eT> dest, const uword n)
  {
  coot_extra_debug_sigprint();

  coot_rt_t::fill_randu(dest, n);
  }



template<typename eT>
inline void coot_rng::fill_randn(dev_mem_t<eT> dest, const uword n, const distr_param& param)
  {
  coot_extra_debug_sigprint();

  double mu, sd;
  if (param.state == 0)
    {
    // defaults
    mu = 0.0;
    sd = 1.0;
    }
  else if (param.state == 1)
    {
    // ints
    mu = (double) param.a_int;
    sd = (double) param.b_int;
    }
  else if (param.state == 2)
    {
    // doubles
    mu = param.a_double;
    sd = param.b_double;
    }
  else
    {
    coot_stop_runtime_error("randn(): incorrect distribution parameter settings");
    }

  coot_debug_check( (sd < 0.0), "randn(): incorrect distribution parameters: sd must be greater than or equal to 0" );

  coot_rt_t::fill_randn(dest, n, mu, sd);
  }



template<typename eT>
inline void coot_rng::fill_randi(dev_mem_t<eT> dest, const uword n, const distr_param& param)
  {
  coot_extra_debug_sigprint();

  int a = 0;
  int b = 0;

  if (param.state == 0)
    {
    b = std::numeric_limits<int>::max();
    }
  else if (param.state == 1)
    {
    a = param.a_int;
    b = param.b_int;
    }
  else if (param.state == 2)
    {
    a = int(param.a_double);
    b = int(param.b_double);
    }
  else
    {
    coot_stop_runtime_error("randi(): incorrect distribution parameter settings");
    }

  coot_debug_check( (a > b), "randi(): incorrect distribution parameters: a must be less than b" );

  coot_rt_t::fill_randi(dest, n, a, b);
  }



inline
void
coot_rng::set_seed(const u64 seed)
  {
  coot_rt_t::set_rng_seed(seed);
  }



inline
void
coot_rng::set_seed_random()
  {
  // This is borrowed from the Armadillo implementation.
  u64 seed1 = u64(0);
  u64 seed2 = u64(0);
  u64 seed3 = u64(0);
  u64 seed4 = u64(0);

  bool have_seed = false;

  try
    {
    std::random_device rd;

    if(rd.entropy() > double(0))  { seed1 = static_cast<u64>( rd() ); }

    if(seed1 != u64(0))  { have_seed = true; }
    }
  catch(...) {}


  if(have_seed == false)
    {
    try
      {
      union
        {
        u64           a;
        unsigned char b[sizeof(u64)];
        } tmp;

      tmp.a = u64(0);

      std::ifstream f("/dev/urandom", std::ifstream::binary);

      if(f.good())  { f.read((char*)(&(tmp.b[0])), sizeof(u64)); }

      if(f.good())
        {
        seed2 = tmp.a;

        if(seed2 != u64(0))  { have_seed = true; }
        }
      }
    catch(...) {}
    }
  if(have_seed == false)
    {
    // get better-than-nothing seeds in case reading /dev/urandom failed

    const std::chrono::system_clock::time_point tp_now = std::chrono::system_clock::now();

    auto since_epoch_usec = std::chrono::duration_cast<std::chrono::microseconds>(tp_now.time_since_epoch()).count();

    seed3 = static_cast<u64>( since_epoch_usec & 0xFFFF );

    union
      {
      uword*        a;
      unsigned char b[sizeof(uword*)];
      } tmp;

    tmp.a = (uword*)malloc(sizeof(uword));
    if(tmp.a != nullptr)
      {
      for(size_t i=0; i<sizeof(uword*); ++i)
        {
        seed4 += u64(tmp.b[i]);
        }

      free(tmp.a);
      }
    }

  coot_rng::set_seed(seed1 + seed2 + seed3 + seed4);
  }
