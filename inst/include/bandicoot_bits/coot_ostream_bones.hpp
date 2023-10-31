// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2017-2023 Ryan Curtin (https://www.ratml.org)
// Copyright 2008-2017 Conrad Sanderson (https://conradsanderson.id.au)
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



class coot_ostream_state
  {
  private:

  const ios::fmtflags   orig_flags;
  const std::streamsize orig_precision;
  const std::streamsize orig_width;
  const char            orig_fill;


  public:

  inline coot_ostream_state(const std::ostream& o);

  inline void restore(std::ostream& o) const;
  };



class coot_ostream
  {
  public:

  template<typename eT> inline static std::streamsize modify_stream(std::ostream& o, const eT*              data, const uword n_elem);

  template<typename eT> inline static void     print_elem_zero(std::ostream& o, const bool modify);
  template<typename eT> inline static void raw_print_elem_zero(std::ostream& o);

  template<typename eT> inline static void     print_elem(std::ostream& o, const eT& x, const bool modify);
  template<typename eT> inline static void raw_print_elem(std::ostream& o, const eT& x);

  template<typename eT> coot_cold inline static void print(std::ostream& o, const     Mat<eT>& m, const bool modify);
  template<typename eT> coot_cold inline static void print(std::ostream& o, const subview<eT>& m, const bool modify);

  coot_cold inline static void print(std::ostream& o, const SizeMat&  S);
  };
