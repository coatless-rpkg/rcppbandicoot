// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2017-2023 Ryan Curtin (https://www.ratml.org)
// Copyright 2017      Conrad Sanderson (http://conradsanderson.id.au)
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
class MatValProxy
  {
  private:

  coot_aligned Mat<eT>& M;
  coot_aligned uword    index;

  public:

  coot_inline MatValProxy(Mat<eT>& in_M, const uword in_index);
  coot_inline operator eT();

  static inline eT get_val(const Mat<eT>& in_M, const uword in_index);

  inline void operator= (const MatValProxy<eT>& in_val);
  inline void operator+=(const MatValProxy<eT>& in_val);
  inline void operator-=(const MatValProxy<eT>& in_val);
  inline void operator*=(const MatValProxy<eT>& in_val);
  inline void operator/=(const MatValProxy<eT>& in_val);

  inline void operator= (const eT in_val);
  inline void operator+=(const eT in_val);
  inline void operator-=(const eT in_val);
  inline void operator*=(const eT in_val);
  inline void operator/=(const eT in_val);
  };
