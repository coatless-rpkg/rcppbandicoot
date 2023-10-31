// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2023 Ryan Curtin (http://www.ratml.org)
// Copyright 2023 Conrad Sanderson (http://conradsanderson.id.au)
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



template<typename T1>
coot_warn_unused
inline
typename T1::elem_type
det
  (
  const Base<typename T1::elem_type, T1>& X,
  const typename coot_real_only<typename T1::elem_type>::result* junk = 0
  )
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  typedef typename T1::elem_type eT;

  eT out_val = eT(0);

  const bool status = op_det::apply_direct(out_val, X.get_ref());

  if (status == false)
    {
    out_val = eT(0);
    coot_stop_runtime_error("det(): failed to find determinant");
    }

  return out_val;
  }



template<typename T1>
inline
bool
det
  (
  typename T1::elem_type& out_val,
  const Base<typename T1::elem_type, T1>& X,
  const typename coot_real_only<typename T1::elem_type>::result* junk = 0
  )
  {
  coot_extra_debug_sigprint();
  coot_ignore(junk);

  typedef typename T1::elem_type eT;

  const bool status = op_det::apply_direct(out_val, X.get_ref());

  if(status == false)
    {
    out_val = eT(0);
    coot_debug_warn_level(3, "det(): failed to find determinant");
    }

  return status;
  }
