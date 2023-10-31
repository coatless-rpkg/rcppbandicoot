// Copyright 2023 Ryan Curtin (http://www.ratml.org)
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


struct magma_real_kernel_id
  {
  enum enum_id
    {
    laset_full,
    laset_lower,
    laset_upper,

    laset_band_lower,
    laset_band_upper,

    transpose_magma, // _magma is to avoid potential naming conflicts
    transpose_inplace_odd_magma,
    transpose_inplace_even_magma,

    laswp,

    lansy_inf_lower,
    lansy_inf_upper,
    lansy_max_lower,
    lansy_max_upper,

    lascl_full,
    lascl_lower,
    lascl_upper
    };



  static
  inline
  std::vector<std::string>
  init_names()
    {
    // NOTE: the order and names of kernels in "names" must match the order and names in the kernel_id enum

    std::vector<std::string> names;

    names.push_back("laset_full");
    names.push_back("laset_lower");
    names.push_back("laset_upper");

    names.push_back("laset_band_lower");
    names.push_back("laset_band_upper");

    names.push_back("transpose_magma");
    names.push_back("transpose_inplace_odd_magma");
    names.push_back("transpose_inplace_even_magma");

    names.push_back("laswp");

    names.push_back("lansy_inf_lower");
    names.push_back("lansy_inf_upper");
    names.push_back("lansy_max_lower");
    names.push_back("lansy_max_upper");

    names.push_back("lascl_full");
    names.push_back("lascl_lower");
    names.push_back("lascl_upper");

    return names;
    }



  static
  inline
  const std::vector<std::string>&
  get_names()
    {
    static const std::vector<std::string> names = init_names();

    return names;
    }
  };
