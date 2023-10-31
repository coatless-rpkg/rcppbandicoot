// Copyright 2020 Ryan Curtin (http://www.ratml.org)
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



/**
 * Compute the dot product of two vectors.
 */
template<typename eT1, typename eT2>
inline
typename promote_type<eT1, eT2>::result
dot(dev_mem_t<eT1> mem1, dev_mem_t<eT2> mem2, const uword n_elem)
  {
  coot_extra_debug_sigprint();

  coot_debug_check( (get_rt().cl_rt.is_valid() == false), "coot::opencl::dot(): OpenCL runtime not valid" );

  // We could use clblasSdot() and clblasDdot(), but the slowness of the sasum() and dasum() implementations
  // makes me think we're better off using our own kernel here.

  typedef typename promote_type<eT1, eT2>::result promoted_eT;

  cl_int status = 0;

  cl_kernel k = get_rt().cl_rt.get_kernel<eT2, eT1>(twoway_kernel_id::dot);
  cl_kernel k_small = get_rt().cl_rt.get_kernel<eT2, eT1>(twoway_kernel_id::dot_small);

  // Compute workgroup sizes.  We use CL_KERNEL_WORK_GROUP_SIZE as an upper bound, which
  // depends on the compiled kernel.  I assume that the results for k will be identical to k_small.
  size_t kernel_wg_size;
  status = coot_wrapper(clGetKernelWorkGroupInfo)(k, get_rt().cl_rt.get_device(), CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &kernel_wg_size, NULL);
  coot_check_cl_error(status, "dot()");

  const size_t k1_work_dim       = 1;
  const size_t k1_work_offset    = 0;
  const uword subgroup_size = get_rt().cl_rt.get_subgroup_size();

  uword total_num_threads = std::ceil(n_elem / std::max(1.0, (2 * std::ceil(std::log2(n_elem)))));
  uword local_group_size = std::min(kernel_wg_size, total_num_threads);

  // Create auxiliary memory.
  const uword aux_size = std::ceil((total_num_threads + (local_group_size - 1)) / local_group_size);
  Mat<promoted_eT> aux(aux_size, 1);
  aux.zeros();
  dev_mem_t<promoted_eT> aux_mem = aux.get_dev_mem(false);

  // We'll only run once with the dot kernel, and if this still needs further reduction, we can use accu().
  runtime_t::cq_guard guard;

  runtime_t::adapt_uword dev_n_elem(n_elem);

  // We need to round total_num_threads up to the next power of 2.  (The kernel assumes this.)
  const uword pow2_group_size = (uword) std::pow(2.0f, std::ceil(std::log2((float) local_group_size)));
  const uword pow2_total_num_threads = (total_num_threads % pow2_group_size == 0) ? total_num_threads : ((total_num_threads / pow2_group_size) + 1) * pow2_group_size;

  // If the number of threads is less than the subgroup size, we need to use the small kernel.
  cl_kernel* k_use = (pow2_group_size <= subgroup_size) ? &k_small : &k;

  status |= coot_wrapper(clSetKernelArg)(*k_use, 0, sizeof(cl_mem),                        &(aux_mem.cl_mem_ptr));
  status |= coot_wrapper(clSetKernelArg)(*k_use, 1, sizeof(cl_mem),                        &(mem1.cl_mem_ptr));
  status |= coot_wrapper(clSetKernelArg)(*k_use, 2, sizeof(cl_mem),                        &(mem2.cl_mem_ptr));
  status |= coot_wrapper(clSetKernelArg)(*k_use, 3, dev_n_elem.size,                       dev_n_elem.addr);
  status |= coot_wrapper(clSetKernelArg)(*k_use, 4, sizeof(promoted_eT) * pow2_group_size, NULL);

  status |= coot_wrapper(clEnqueueNDRangeKernel)(get_rt().cl_rt.get_cq(), *k_use, k1_work_dim, &k1_work_offset, &pow2_total_num_threads, &pow2_group_size, 0, NULL, NULL);

  coot_check_cl_error(status, "dot()");

  if (aux.n_elem == 1)
    {
    return promoted_eT(aux[0]);
    }
  else
    {
    return accu(aux_mem, aux.n_elem);
    }
  }
