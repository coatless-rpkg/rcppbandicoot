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
COOT_FN(PREFIX,find_last)(const eT1* A,
                          const UWORD* thread_counts,
                          UWORD* out,
                          const UWORD m,
                          const UWORD n_elem)
  {
  // Our goal is to fill `out` with the last `k` indices of nonzero values.
  // (Note that to match Armadillo's behavior, we want the last `k` indices in ascending order.)
  // Instead of accepting `k` as a parameter, we instead accept `m = nnz - k`.
  // This gives us the first index we should be putting an output value in.
  // It is also assumed that `k != 0`; if `k` is `0`, use the `find` kernel instead.

  // Since the kernel is multithreaded, each thread will handle a different (contiguous) part of `A`.
  // We expect that we already have the starting position for each thread in `thread_counts`.
  // (It should have been filled with the `count_nonzeros` kernel.)

  const UWORD tid = threadIdx.x;

  const UWORD num_threads = blockDim.x;
  const UWORD elems_per_thread = (n_elem + num_threads - 1) / num_threads; // this is ceil(n_elem / num_threads)
  const UWORD start_elem = tid * elems_per_thread;
  UWORD end_elem = min((tid + 1) * elems_per_thread, n_elem);

  UWORD out_index = thread_counts[tid];
  UWORD last_out_index = thread_counts[tid + 1];

  UWORD i = start_elem;

  // We only want to find points with index `m` or higher.
  if (last_out_index >= m)
    {
    while (i + 1 < end_elem)
      {
      if (A[i] != (eT1) 0)
        {
        if (out_index >= m)
          {
          out[out_index - m] = i;
          }

        ++out_index;
        }
      if (A[i + 1] != (eT1) 0)
        {
        if (out_index >= m)
          {
          out[out_index - m] = (i + 1);
          }

        ++out_index;
        }

      i += 2;
      }

    if (i < end_elem)
      {
      if (A[i] != (eT1) 0)
        {
        if (out_index >= m)
          {
          out[out_index - m] = i;
          }

        ++out_index;
        }
      }
    }
  }
