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

layout(set = 0, binding = 0, std430) buffer Buf0 { ET0 data[]; } out_buf;
layout(set = 0, binding = 1, std430) readonly buffer Buf1 { ET0 data[]; } val_buf;

layout(push_constant) uniform PushConsts {
  UWORD out_offset;
  UWORD out_n_rows;
  UWORD out_n_cols;
  UWORD out_M_n_rows;
  UWORD out_n_slices;
  UWORD out_M_n_elem_slice;
} pc;

bool coot_isnan(float  x) { return isnan(x); }
bool coot_isnan(double x) { return isnan(x); }

void main()
  {
  const UWORD row = gl_GlobalInvocationID.x;
  const UWORD col = gl_GlobalInvocationID.y;
  const UWORD slice = gl_GlobalInvocationID.z;

  if (row < pc.out_n_rows && col < pc.out_n_cols && slice < pc.out_n_slices)
    {
    UWORD idx = pc.out_offset + row + col * pc.out_M_n_rows + slice * pc.out_M_n_elem_slice;
    out_buf.data[uint(idx)] = val_buf.data[0];
    }
  }

)"
