// Copyright 2026 Marcus Edel (http://www.kurg.org/)
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

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

COOT_OBJECT_0_PARAMS(out);
COOT_OBJECT_1_PARAMS(in);

layout(push_constant) uniform PushConsts {
  COOT_OBJECT_0(out);
  COOT_OBJECT_1(in);
};

void main()
  {
  const UWORD row = gl_GlobalInvocationID.x;
  const UWORD col = gl_GlobalInvocationID.y;
  const UWORD slice = gl_GlobalInvocationID.z;

  if (COOT_OBJECT_0_BOUNDS_CHECK(out, row, col, slice))
    {
    COOT_OBJECT_0_AT(out, row, col, slice) = COOT_TO_ET0(COOT_OBJECT_1_AT(in, row, col, slice));
    }
  }

)"
