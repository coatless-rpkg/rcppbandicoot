// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2020 Ryan Curtin (https://www.ratml.org/)
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



// A version of unwrap<> that avoids a final type conversion if possible.
// It does do type conversions if they are needed for an intermediate operation.
// This is useful for operations that can accept a different input type.

template<typename T1>
struct no_conv_unwrap : public unwrap<T1>
  {
  // By default we simply unwrap as normal.
  no_conv_unwrap(const T1& x) : unwrap<T1>(x) { }
  };



template<typename out_eT, typename T1>
struct no_conv_unwrap< mtOp<out_eT, T1, mtop_conv_to> > : public unwrap<T1>
  {
  // If we got a conversion operation, we only unwrap the inner operation and avoid the conversion.
  no_conv_unwrap(const mtOp<out_eT, T1, mtop_conv_to>& x) : unwrap<T1>(x.q) { }
  };
