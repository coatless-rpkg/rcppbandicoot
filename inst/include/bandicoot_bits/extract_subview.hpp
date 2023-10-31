// SPDX-License-Identifier: Apache-2.0
//
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



// Template metaprogramming utility to extract a subview into a Mat;
// this is meant to be used as, e.g., extract_subview<unwrap<T1>>,
// so that the held object is ensured to be a Mat, not a subview.
//
// Note that the input object is expected to be a Mat or a subview!

template<typename T1>
struct extract_subview
  {
  explicit inline extract_subview(const T1& in)
    : M(in)
    {
    coot_extra_debug_sigprint();
    }

  Mat<typename T1::elem_type> M;
  };



template<typename eT>
struct extract_subview<Mat<eT>>
  {
  explicit inline extract_subview(const Mat<eT>& in)
    : M(in)
    {
    coot_extra_debug_sigprint();
    }

  const Mat<eT>& M;
  };
