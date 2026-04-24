// Copyright 2026 Marcus Edel (http://www.kurg.org)
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



struct pipeline_t
  {
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
  VkShaderModule shader = VK_NULL_HANDLE;
  };



class runtime_t
  {
  public:

  inline runtime_t();
  inline ~runtime_t();

  inline bool init(const bool manual_selection, const uword wanted_platform, const uword wanted_device, const bool print_info);

  inline bool is_valid() const { return valid; }

  inline bool has_fp64() const { return supports_fp64; }
  inline bool has_int64() const { return supports_int64; }

  inline bool has_sizet64() const { return (sizeof(uword) >= 8) && supports_int64; }

  inline std::string uword_str() const { return has_sizet64() ? "uint64_t" : "uint"; }

  inline std::string uword_ext() const { return has_sizet64() ? "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n" : ""; }

  template<template<typename> class push_t>
  inline size_t push_size() const { return has_sizet64() ? sizeof(push_t<u64>) : sizeof(push_t<u32>); }

  template<typename eT>
  inline bool is_supported_type() const;

  template<typename eT>
  inline coot_vk_mem acquire_memory(const uword n_elem);

  inline void release_memory(coot_vk_mem mem);

  inline void synchronise();

  inline pipeline_t& get_pipeline_from_source(const std::string& name, const std::string& glsl_source, const size_t push_constant_size, const uint32_t num_buffers = 3);
  inline pipeline_t& get_pipeline_from_source(const std::string& name, const std::string& glsl_source, const std::vector<std::string>& macros, const size_t push_constant_size, const uint32_t num_buffers = 3);

  inline std::vector<uint32_t> compile_glsl_to_spirv(const std::string& name, const std::string& source, const std::vector<std::string>& macros = {});

  template<typename eT>
  inline std::string generate_eye_kernel();

  template<typename eT>
  inline std::string generate_fill_kernel();

  template<typename eT_out, typename eT_in>
  inline std::string generate_copy_kernel();

  template<typename eT>
  inline std::string generate_copy_replace_kernel();

  template<typename eT>
  inline std::string generate_accu_kernel();

  template<typename eT>
  inline std::string generate_accu_subview_kernel();

  template<typename eT>
  inline std::string generate_dot_kernel();

  template<typename eT>
  inline std::string generate_rel_eq_scalar_kernel();

  template<typename eT>
  inline std::string generate_rel_neq_scalar_kernel();

  template<typename eT>
  inline std::string generate_rel_gt_scalar_kernel();

  template<typename eT>
  inline std::string generate_rel_lt_scalar_kernel();

  template<typename eT>
  inline std::string generate_rel_gteq_scalar_kernel();

  template<typename eT>
  inline std::string generate_rel_lteq_scalar_kernel();

  template<typename eT>
  inline pipeline_t& get_gen_rel_neq_scalar_pipeline();

  template<typename eT>
  inline pipeline_t& get_gen_rel_gt_scalar_pipeline();

  template<typename eT>
  inline pipeline_t& get_gen_rel_lt_scalar_pipeline();

  template<typename eT>
  inline pipeline_t& get_gen_rel_gteq_scalar_pipeline();

  template<typename eT>
  inline pipeline_t& get_gen_rel_lteq_scalar_pipeline();

  template<typename eT>
  inline std::string generate_all_neq_vec_kernel();

  template<typename eT>
  inline std::string generate_all_neq_colwise_kernel();

  template<typename eT>
  inline std::string generate_all_neq_rowwise_kernel();

  template<kernel_id::enum_id num, typename... ProxyTypes>
  inline pipeline_t& get_kernel();

  template<typename... ProxyTypes>
  inline size_t compute_gen_push_size();

  template<typename eT>
  inline pipeline_t& get_gen_eye_pipeline();

  template<typename eT>
  inline pipeline_t& get_gen_fill_pipeline();

  template<typename eT_out, typename eT_in>
  inline pipeline_t& get_gen_copy_pipeline();

  template<typename eT>
  inline pipeline_t& get_gen_copy_replace_pipeline();

  template<typename eT>
  inline pipeline_t& get_gen_accu_pipeline();

  template<typename eT>
  inline pipeline_t& get_gen_accu_subview_pipeline();

  template<typename eT>
  inline pipeline_t& get_gen_dot_pipeline();

  template<typename eT>
  inline pipeline_t& get_gen_rel_eq_scalar_pipeline();

  template<typename eT>
  inline pipeline_t& get_gen_all_neq_vec_pipeline();

  template<typename eT>
  inline pipeline_t& get_gen_all_neq_pipeline(const bool colwise);

  template<typename eT>
  inline std::string generate_min_vec_kernel();

  template<typename eT>
  inline std::string generate_max_vec_kernel();

  template<typename eT>
  inline std::string generate_min_colwise_kernel();

  template<typename eT>
  inline std::string generate_min_rowwise_kernel();

  template<typename eT>
  inline std::string generate_max_colwise_kernel();

  template<typename eT>
  inline std::string generate_max_rowwise_kernel();

  template<typename eT>
  inline pipeline_t& get_gen_min_vec_pipeline();

  template<typename eT>
  inline pipeline_t& get_gen_max_vec_pipeline();

  template<typename eT>
  inline pipeline_t& get_gen_min_colwise_pipeline();

  template<typename eT>
  inline pipeline_t& get_gen_min_rowwise_pipeline();

  template<typename eT>
  inline pipeline_t& get_gen_max_colwise_pipeline();

  template<typename eT>
  inline pipeline_t& get_gen_max_rowwise_pipeline();

  template<typename eT>
  inline std::string generate_rel_isfinite_kernel();

  template<typename eT>
  inline std::string generate_rel_isinf_kernel();

  template<typename eT>
  inline std::string generate_rel_isnan_kernel();

  template<typename eT>
  inline pipeline_t& get_gen_rel_isfinite_pipeline();

  template<typename eT>
  inline pipeline_t& get_gen_rel_isinf_pipeline();

  template<typename eT>
  inline pipeline_t& get_gen_rel_isnan_pipeline();

  template<typename eT>
  inline std::string generate_trans_kernel();

  template<typename eT>
  inline pipeline_t& get_gen_trans_pipeline();

  template<typename eT>
  inline std::string generate_reorder_cols_kernel();

  template<typename eT>
  inline pipeline_t& get_gen_reorder_cols_pipeline();

  template<typename eT>
  inline std::string generate_index_min_colwise_kernel();

  template<typename eT>
  inline std::string generate_index_min_rowwise_kernel();

  template<typename eT>
  inline std::string generate_index_min_vec_kernel();

  template<typename eT>
  inline pipeline_t& get_gen_index_min_colwise_pipeline();

  template<typename eT>
  inline pipeline_t& get_gen_index_min_rowwise_pipeline();

  template<typename eT>
  inline pipeline_t& get_gen_index_min_vec_pipeline();

  template<typename eT>
  inline std::string generate_index_max_colwise_kernel();

  template<typename eT>
  inline std::string generate_index_max_rowwise_kernel();

  template<typename eT>
  inline std::string generate_index_max_vec_kernel();

  template<typename eT>
  inline pipeline_t& get_gen_index_max_colwise_pipeline();

  template<typename eT>
  inline pipeline_t& get_gen_index_max_rowwise_pipeline();

  template<typename eT>
  inline pipeline_t& get_gen_index_max_vec_pipeline();

  template<typename eT_src, typename eT_dest>
  inline std::string generate_broadcast_kernel(const std::string& op_expr);

  template<typename eT_src, typename eT_dest>
  inline pipeline_t& get_gen_broadcast_set_pipeline();

  template<typename eT_src, typename eT_dest>
  inline pipeline_t& get_gen_broadcast_plus_pipeline();

  template<typename eT_src, typename eT_dest>
  inline pipeline_t& get_gen_broadcast_minus_pre_pipeline();

  template<typename eT_src, typename eT_dest>
  inline pipeline_t& get_gen_broadcast_minus_post_pipeline();

  template<typename eT_src, typename eT_dest>
  inline pipeline_t& get_gen_broadcast_schur_pipeline();

  template<typename eT_src, typename eT_dest>
  inline pipeline_t& get_gen_broadcast_div_pre_pipeline();

  template<typename eT_src, typename eT_dest>
  inline pipeline_t& get_gen_broadcast_div_post_pipeline();

  template<typename eT_in, typename eT_out>
  inline std::string generate_gather_kernel();

  template<typename eT_in, typename eT_out>
  inline pipeline_t& get_gen_gather_pipeline();

  template<typename eT, bool do_trans_A, bool do_trans_B>
  inline std::string generate_gemm_kernel();

  template<typename eT, bool do_trans_A, bool do_trans_B>
  inline pipeline_t& get_gen_gemm_pipeline();

  inline VkDevice get_device() const { return device; }
  inline VkQueue get_queue() const { return queue; }
  inline VkCommandPool get_command_pool() const { return command_pool; }
  inline VkDescriptorPool get_descriptor_pool() const { return descriptor_pool; }
  inline void* get_pool_mapped() const { return mem_pool_mapped; }

  inline VkCommandBuffer begin_commands() const;
  inline void end_commands(VkCommandBuffer cmd) const;

  inline void set_rng_seed(const u64 seed);
  inline u64 next_rng_seed();

  class adapt_uword;

  private:

  inline uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) const;

  coot_aligned bool valid = false;
  coot_aligned bool supports_fp64 = false;
  coot_aligned bool supports_int64 = false;
  coot_aligned bool validation_enabled = false;
  coot_aligned u64 rng_seed = 1;
  coot_aligned u64 rng_counter = 0;

  VkInstance instance = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t queue_family_index = 0;

  VkCommandPool command_pool = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;

  VkBuffer mem_pool_buffer = VK_NULL_HANDLE;
  VkDeviceMemory mem_pool_memory = VK_NULL_HANDLE;
  VkDeviceSize mem_pool_used = 0;
  void* mem_pool_mapped = nullptr;

  std::unordered_map<std::string, pipeline_t> pipelines;
  std::vector<std::pair<VkDeviceSize, VkDeviceSize>> free_list;
  };



class runtime_t::adapt_uword
  {
  public:

  inline adapt_uword(const uword val);
  inline adapt_uword(const adapt_uword& x);
  inline adapt_uword(adapt_uword&& x);
  inline adapt_uword& operator=(const adapt_uword& x);
  inline adapt_uword& operator=(adapt_uword&& x);

  coot_aligned size_t size;
  coot_aligned void*  addr;

  coot_aligned u64 val64;
  coot_aligned u32 val32;
  };
