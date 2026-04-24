// Copyright 2026 Marcus Edel (http://www.kurg.org/)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------
R"(

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer Buf0 { ET0 data[]; } out_buf;
layout(set = 0, binding = 1, std430) readonly buffer Buf1 { ET0 data[]; } in_buf;

layout(push_constant) uniform PushConsts {
  UWORD dest_offset;
  UWORD src_offset;
  UWORD n_rows;
  UWORD n_cols;
} pc;

void main()
  {
  const UWORD row = gl_GlobalInvocationID.x;
  const UWORD col = gl_GlobalInvocationID.y;

  if (row < pc.n_rows && col < pc.n_cols)
    {
    UWORD src_idx = pc.src_offset + row + col * pc.n_rows;
    UWORD dest_idx = pc.dest_offset + col + row * pc.n_cols;
    out_buf.data[uint(dest_idx)] = in_buf.data[uint(src_idx)];
    }
  }

)"
