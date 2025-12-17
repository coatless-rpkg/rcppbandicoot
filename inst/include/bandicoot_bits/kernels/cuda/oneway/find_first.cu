// Copyright 2023 Ryan Curtin (http://www.ratml.org/)
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



__global__
void
COOT_FN(PREFIX,find_first)(const eT1* A,
                           const UWORD* thread_counts,
                           UWORD* out,
                           const UWORD k,
                           const UWORD n_elem)
  {
  // Our goal is to fill `out` with the first `k` indices of nonzero values.
  // It is assumed that `k != 0`; if `k` is `0`, use the `find` kernel instead.
  // Since the kernel is multithreaded, each thread will handle a different (contiguous) part of `A`.
  // We expect that we already have the starting position for each thread in `thread_counts`.
  // (It should have been filled with the `count_nonzeros` kernel.)

  const UWORD tid = threadIdx.x;

  const UWORD num_threads = blockDim.x;
  const UWORD elems_per_thread = (n_elem + num_threads - 1) / num_threads; // this is ceil(n_elem / num_threads)
  const UWORD start_elem = tid * elems_per_thread;
  UWORD end_elem = min((tid + 1) * elems_per_thread, n_elem);

  UWORD out_index = thread_counts[tid];

  UWORD i = start_elem;

  // We only want to find the first k points.
  if (out_index < k)
    {
    while (i + 1 < end_elem)
      {
      if (A[i] != TO_ET1(0) && out_index < k)
        {
        out[out_index++] = i;
        }
      if (A[i + 1] != TO_ET1(0) && out_index < k)
        {
        out[out_index++] = (i + 1);
        }

      i += 2;
      }
    if (i < end_elem)
      {
      if (A[i] != TO_ET1(0) && out_index < k)
        {
        out[out_index++] = i;
        }
      }
    }
  }
