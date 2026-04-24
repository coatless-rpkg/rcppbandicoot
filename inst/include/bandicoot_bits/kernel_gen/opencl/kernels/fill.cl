// Copyright 2017 Conrad Sanderson (http://conradsanderson.id.au)
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
R"(


__kernel
void
(COOT_KERNEL_FUNC)(COOT_OBJECT_0(out),
                   const ET0 val)
  {
  // Not all of these may be used, depending on what COOT_OBJECT_0 is.
  const UWORD row   = get_global_id(0);
  const UWORD col   = get_global_id(1);
  const UWORD slice = get_global_id(2);

  if (COOT_OBJECT_0_BOUNDS_CHECK(out, row, col, slice))
    {
    COOT_OBJECT_0_AT(out, row, col, slice) = val;
    }
  }


)"
