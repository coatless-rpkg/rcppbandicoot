// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2017-2025 Ryan Curtin (https://www.ratml.org)
// Copyright 2017      Conrad Sanderson (https://conradsanderson.id.au)
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



struct zeroway_kernel_id
  {
  enum enum_id
    {
    shuffle_large_compute_locs,
    //
    invalid_kernel
    };


  static
  inline
  std::vector<std::string>
  init_names()
    {
    // NOTE: the order and names of kernels in "names" must match the order and names in the kernel_id enum

    std::vector<std::string> names;

    names.push_back("shuffle_large_compute_locs");

    return names;
    };


  static
  inline
  const std::vector<std::string>&
  get_names()
    {
    static const std::vector<std::string> names = init_names();

    return names;
    }



  static
  inline
  std::unordered_map<enum_id, std::vector<std::string>>
  init_deps()
    {
    std::unordered_map<enum_id, std::vector<std::string>> deps;

    deps[shuffle_large_compute_locs] = { "var_philox" };

    return deps;
    }



  static
  inline
  const std::unordered_map<enum_id, std::vector<std::string>>&
  get_deps()
    {
    static const std::unordered_map<enum_id, std::vector<std::string>> deps = init_deps();

    return deps;
    }

  };




struct oneway_kernel_id
  {
  enum enum_id
    {
    fill = 0,
    fill_sve1,
    fill_sve2,
    //
    mul_colwise,
    mul_rowwise,
    mul_colwise_trans,
    mul_rowwise_trans,
    //
    inplace_set_eye,
    linspace,
    logspace,
    regspace_desc,
    //
    accu_simple,
    accu,
    accu_small,
    //
    min,
    min_small,
    max,
    max_small,
    max_abs,
    max_abs_small,
    index_min,
    index_min_small,
    index_max,
    index_max_small,
    index_min_rowwise,
    index_min_colwise,
    index_min_cube_col,
    index_max_rowwise,
    index_max_colwise,
    index_max_cube_col,
    //
    prod,
    prod_small,
    //
    trace,
    //
    ltri_set_zero,
    //
    inplace_xorwow32_randu,
    inplace_xorwow64_randu,
    inplace_philox_randn,
    inplace_xorwow32_randi,
    inplace_xorwow64_randi,
    shuffle,
    shuffle_large,
    //
    var_colwise,
    var_rowwise,
    var,
    var_small,
    submat_var,
    submat_var_small,
    //
    radix_sort_colwise_asc,
    radix_sort_rowwise_asc,
    radix_sort_asc,
    radix_sort_colwise_desc,
    radix_sort_rowwise_desc,
    radix_sort_desc,
    radix_sort_index_asc,
    radix_sort_index_desc,
    stable_radix_sort_index_asc,
    stable_radix_sort_index_desc,
    shifted_prefix_sum_small,
    shifted_prefix_sum_subgroups,
    shifted_prefix_sum_add_offset,
    radix_sort_multi_wg_bit_count,
    radix_sort_multi_wg_shuffle,
    radix_sort_index_multi_wg_shuffle,
    //
    count_nonzeros,
    find,
    find_first,
    find_last,
    //
    symmatu_inplace,
    symmatl_inplace,
    //
    reorder_cols,
    //
    rotate_180,
    //
    approx_equal,
    approx_equal_small,
    approx_equal_cube,
    approx_equal_cube_small,
    //
    invalid_kernel
    };


  static
  inline
  std::vector<std::string>
  init_names()
    {
    // NOTE: the order and names of kernels in "names" must match the order and names in the kernel_id enum

    std::vector<std::string> names;

    names.push_back("fill");
    names.push_back("fill_sve1");
    names.push_back("fill_sve2");

    names.push_back("mul_colwise");
    names.push_back("mul_rowwise");
    names.push_back("mul_colwise_trans");
    names.push_back("mul_rowwise_trans");

    names.push_back("inplace_set_eye");
    names.push_back("linspace");
    names.push_back("logspace");
    names.push_back("regspace_desc");

    names.push_back("accu_simple");
    names.push_back("accu");
    names.push_back("accu_small");

    names.push_back("min");
    names.push_back("min_small");
    names.push_back("max");
    names.push_back("max_small");
    names.push_back("max_abs");
    names.push_back("max_abs_small");
    names.push_back("index_min");
    names.push_back("index_min_small");
    names.push_back("index_max");
    names.push_back("index_max_small");
    names.push_back("index_min_rowwise");
    names.push_back("index_min_colwise");
    names.push_back("index_min_cube_col");
    names.push_back("index_max_rowwise");
    names.push_back("index_max_colwise");
    names.push_back("index_max_cube_col");

    names.push_back("prod");
    names.push_back("prod_small");

    names.push_back("trace");

    names.push_back("ltri_set_zero");

    names.push_back("inplace_xorwow32_randu");
    names.push_back("inplace_xorwow64_randu");
    names.push_back("inplace_philox_randn");
    names.push_back("inplace_xorwow32_randi");
    names.push_back("inplace_xorwow64_randi");
    names.push_back("shuffle");
    names.push_back("shuffle_large");

    names.push_back("var_colwise");
    names.push_back("var_rowwise");
    names.push_back("var");
    names.push_back("var_small");
    names.push_back("submat_var");
    names.push_back("submat_var_small");

    names.push_back("radix_sort_colwise_asc");
    names.push_back("radix_sort_rowwise_asc");
    names.push_back("radix_sort_asc");
    names.push_back("radix_sort_colwise_desc");
    names.push_back("radix_sort_rowwise_desc");
    names.push_back("radix_sort_desc");
    names.push_back("radix_sort_index_asc");
    names.push_back("radix_sort_index_desc");
    names.push_back("stable_radix_sort_index_asc");
    names.push_back("stable_radix_sort_index_desc");
    names.push_back("shifted_prefix_sum_small");
    names.push_back("shifted_prefix_sum_subgroups");
    names.push_back("shifted_prefix_sum_add_offset");
    names.push_back("radix_sort_multi_wg_bit_count");
    names.push_back("radix_sort_multi_wg_shuffle");
    names.push_back("radix_sort_index_multi_wg_shuffle");

    names.push_back("count_nonzeros");
    names.push_back("find");
    names.push_back("find_first");
    names.push_back("find_last");

    names.push_back("symmatu_inplace");
    names.push_back("symmatl_inplace");

    names.push_back("reorder_cols");

    names.push_back("rotate_180");

    names.push_back("approx_equal");
    names.push_back("approx_equal_small");
    names.push_back("approx_equal_cube");
    names.push_back("approx_equal_cube_small");

    return names;
    }



  static
  inline
  const std::vector<std::string>&
  get_names()
    {
    static const std::vector<std::string> names = init_names();

    return names;
    }



  static
  inline
  std::unordered_map<enum_id, std::vector<std::string>>
  init_deps()
    {
    std::unordered_map<enum_id, std::vector<std::string>> deps;

    deps[approx_equal_cube] = { "and_subgroup_reduce_u32" };
    deps[approx_equal]      = { "and_subgroup_reduce_u32" };
    deps[shuffle]           = { "var_philox"              };
    deps[shuffle_large]     = { "var_philox"              };
    deps[prod]              = { "prod_subgroup_reduce"    };
    deps[min]               = { "min_subgroup_reduce"     };
    deps[max]               = { "max_subgroup_reduce"     };
    deps[max_abs]           = { "max_subgroup_reduce"     };
    deps[accu]              = { "accu_subgroup_reduce"    };
    deps[var]               = { "accu_subgroup_reduce"    };
    deps[submat_var]        = { "accu_subgroup_reduce"    };

    return deps;
    }



  static
  inline
  const std::unordered_map<enum_id, std::vector<std::string>>&
  get_deps()
    {
    static const std::unordered_map<enum_id, std::vector<std::string>> deps = init_deps();

    return deps;
    }

  };



// These kernels should only be used with float or double element types.
struct oneway_real_kernel_id
  {
  enum enum_id
    {
    vec_norm_1,
    vec_norm_1_small,
    vec_norm_2,
    vec_norm_2_small,
    vec_norm_2_robust,
    vec_norm_2_robust_small,
    vec_norm_k,
    vec_norm_k_small,
    vec_norm_min,
    vec_norm_min_small,
    //
    rel_isfinite,
    rel_isnonfinite,
    rel_isnan,
    rel_any_nan,
    rel_any_nan_small,
    rel_any_inf,
    rel_any_inf_small,
    rel_any_nonfinite,
    rel_any_nonfinite_small,
    //
    lu_extract_l,
    lu_extract_pivoted_l,
    lu_extract_p,
    //
    diag_prod,
    diag_prod_small,
    //
    extract_cx,
    //
    invalid_kernel
    };


  static
  inline
  std::vector<std::string>
  init_names()
    {
    // NOTE: the order and names of kernels in "names" must match the order and names in the kernel_id enum

    std::vector<std::string> names;

    names.push_back("vec_norm_1");
    names.push_back("vec_norm_1_small");
    names.push_back("vec_norm_2");
    names.push_back("vec_norm_2_small");
    names.push_back("vec_norm_2_robust");
    names.push_back("vec_norm_2_robust_small");
    names.push_back("vec_norm_k");
    names.push_back("vec_norm_k_small");
    names.push_back("vec_norm_min");
    names.push_back("vec_norm_min_small");

    names.push_back("rel_isfinite");
    names.push_back("rel_isnonfinite");
    names.push_back("rel_isnan");
    names.push_back("rel_any_nan");
    names.push_back("rel_any_nan_small");
    names.push_back("rel_any_inf");
    names.push_back("rel_any_inf_small");
    names.push_back("rel_any_nonfinite");
    names.push_back("rel_any_nonfinite_small");

    names.push_back("lu_extract_l");
    names.push_back("lu_extract_pivoted_l");
    names.push_back("lu_extract_p");

    names.push_back("diag_prod");
    names.push_back("diag_prod_small");

    names.push_back("extract_cx");

    return names;
    }



  static
  inline
  const std::vector<std::string>&
  get_names()
    {
    static const std::vector<std::string> names = init_names();

    return names;
    }



  static
  inline
  std::unordered_map<enum_id, std::vector<std::string>>
  init_deps()
    {
    std::unordered_map<enum_id, std::vector<std::string>> deps;

    deps[rel_any_nan]       = { "or_subgroup_reduce_u32" };
    deps[rel_any_inf]       = { "or_subgroup_reduce_u32" };
    deps[rel_any_nonfinite] = { "or_subgroup_reduce_u32" };
    deps[diag_prod]         = { "prod_subgroup_reduce"   };
    deps[vec_norm_min]      = { "min_subgroup_reduce"    };
    deps[vec_norm_1]        = { "accu_subgroup_reduce"   };
    deps[vec_norm_2]        = { "accu_subgroup_reduce"   };
    deps[vec_norm_2_robust] = { "accu_subgroup_reduce"   };
    deps[vec_norm_k]        = { "accu_subgroup_reduce"   };

    return deps;
    }



  static
  inline
  const std::unordered_map<enum_id, std::vector<std::string>>&
  get_deps()
    {
    static const std::unordered_map<enum_id, std::vector<std::string>> deps = init_deps();

    return deps;
    }

  };



// These kernels should only be used with integral types (u32/s32/u64/s64/etc.).
struct oneway_integral_kernel_id
  {
  enum enum_id
    {
    and_reduce,
    and_reduce_small,
    or_reduce,
    or_reduce_small,
    //
    ipiv_det,
    ipiv_det_small,
    //
    invalid_kernel
    };



  static
  inline
  std::vector<std::string>
  init_names()
    {
    // NOTE: the order and names of kernels in "names" must match the order and names in the kernel_id enum

    std::vector<std::string> names;

    names.push_back("and_reduce");
    names.push_back("and_reduce_small");
    names.push_back("or_reduce");
    names.push_back("or_reduce_small");

    names.push_back("ipiv_det");
    names.push_back("ipiv_det_small");

    return names;
    }



  static
  inline
  const std::vector<std::string>&
  get_names()
    {
    static const std::vector<std::string> names = init_names();

    return names;
    }



  static
  inline
  std::unordered_map<enum_id, std::vector<std::string>>
  init_deps()
    {
    std::unordered_map<enum_id, std::vector<std::string>> deps;

    deps[ipiv_det] = { "prod_subgroup_reduce" };

    return deps;
    }



  static
  inline
  const std::unordered_map<enum_id, std::vector<std::string>>&
  get_deps()
    {
    static const std::unordered_map<enum_id, std::vector<std::string>> deps = init_deps();

    return deps;
    }

  };



struct twoway_kernel_id
  {
  enum enum_id
    {
    equ_array_plus_scalar = 0,
    equ_array_neg_pre,
    equ_array_neg_post,
    equ_array_minus_scalar_pre_pre,
    equ_array_minus_scalar_pre_post,
    equ_array_minus_scalar_post,
    equ_array_mul_scalar,
    equ_array_div_scalar_pre,
    equ_array_div_scalar_post,
    equ_array_mod_scalar,
    equ_array_square_pre,
    equ_array_square_post,
    equ_array_sqrt_pre,
    equ_array_sqrt_post,
    equ_array_exp_pre,
    equ_array_exp_post,
    equ_array_exp2_pre,
    equ_array_exp2_post,
    equ_array_exp10_pre,
    equ_array_exp10_post,
    equ_array_trunc_exp_pre,
    equ_array_trunc_exp_post,
    equ_array_log_pre,
    equ_array_log_post,
    equ_array_log2_pre,
    equ_array_log2_post,
    equ_array_log10_pre,
    equ_array_log10_post,
    equ_array_trunc_log_pre,
    equ_array_trunc_log_post,
    equ_array_cos_pre,
    equ_array_cos_post,
    equ_array_sin_pre,
    equ_array_sin_post,
    equ_array_tan_pre,
    equ_array_tan_post,
    equ_array_acos_pre,
    equ_array_acos_post,
    equ_array_asin_pre,
    equ_array_asin_post,
    equ_array_atan_pre,
    equ_array_atan_post,
    equ_array_cosh_pre,
    equ_array_cosh_post,
    equ_array_sinh_pre,
    equ_array_sinh_post,
    equ_array_tanh_pre,
    equ_array_tanh_post,
    equ_array_acosh_pre,
    equ_array_acosh_post,
    equ_array_asinh_pre,
    equ_array_asinh_post,
    equ_array_atanh_pre,
    equ_array_atanh_post,
    equ_array_sinc_pre,
    equ_array_sinc_post,
    equ_array_abs,
    equ_array_pow_pre,
    equ_array_pow_post,
    equ_array_floor_pre,
    equ_array_floor_post,
    equ_array_ceil_pre,
    equ_array_ceil_post,
    equ_array_round_pre,
    equ_array_round_post,
    equ_array_trunc_pre,
    equ_array_trunc_post,
    equ_array_sign_pre,
    equ_array_sign_post,
    equ_array_erf_pre,
    equ_array_erf_post,
    equ_array_erfc_pre,
    equ_array_erfc_post,
    equ_array_lgamma_pre,
    equ_array_lgamma_post,
    replace,
    clamp,
    //
    extract_sve1,
    equ_array_plus_scalar_sve1,
    equ_array_minus_scalar_pre_pre_sve1,
    equ_array_minus_scalar_pre_post_sve1,
    equ_array_minus_scalar_post_sve1,
    equ_array_mul_scalar_sve1,
    equ_array_div_scalar_pre_sve1,
    equ_array_div_scalar_post_sve1,
    //
    inplace_sve1_eq_array,
    inplace_sve1_plus_array,
    inplace_sve1_minus_array,
    inplace_sve1_mul_array,
    inplace_sve1_div_array,
    //
    inplace_sve1_eq_sve1,
    inplace_sve1_plus_sve1,
    inplace_sve1_minus_sve1,
    inplace_sve1_mul_sve1,
    inplace_sve1_div_sve1,
    //
    extract_sve2,
    equ_array_plus_scalar_sve2,
    equ_array_minus_scalar_pre_pre_sve2,
    equ_array_minus_scalar_pre_post_sve2,
    equ_array_minus_scalar_post_sve2,
    equ_array_mul_scalar_sve2,
    equ_array_div_scalar_pre_sve2,
    equ_array_div_scalar_post_sve2,
    //
    inplace_sve2_eq_array,
    inplace_sve2_plus_array,
    inplace_sve2_minus_array,
    inplace_sve2_mul_array,
    inplace_sve2_div_array,
    //
    inplace_sve2_eq_sve2,
    inplace_sve2_plus_sve2,
    inplace_sve2_minus_sve2,
    inplace_sve2_mul_sve2,
    inplace_sve2_div_sve2,
    //
    equ_array_min_array_cube,
    equ_array_max_array_cube,
    //
    sum_colwise_conv_pre,
    sum_rowwise_conv_pre,
    sum_colwise_conv_post,
    sum_rowwise_conv_post,
    min_colwise_conv_pre,
    min_rowwise_conv_pre,
    min_colwise_conv_post,
    min_rowwise_conv_post,
    min_cube_col_conv_pre,
    min_cube_col_conv_post,
    max_colwise_conv_pre,
    max_rowwise_conv_pre,
    max_colwise_conv_post,
    max_rowwise_conv_post,
    max_cube_col_conv_pre,
    max_cube_col_conv_post,
    mean_colwise_conv_pre,
    mean_rowwise_conv_pre,
    mean_colwise_conv_post,
    mean_rowwise_conv_post,
    //
    dot,
    dot_small,
    //
    convert_type,
    convert_type_cube,
    //
    broadcast_set,
    broadcast_plus,
    broadcast_minus_pre,
    broadcast_minus_post,
    broadcast_schur,
    broadcast_div_pre,
    broadcast_div_post,
    broadcast_subset_set,
    broadcast_subset_plus,
    broadcast_subset_minus_pre,
    broadcast_subset_minus_post,
    broadcast_subset_schur,
    broadcast_subset_div_pre,
    broadcast_subset_div_post,
    //
    htrans,
    strans,
    //
    rel_gt_scalar,
    rel_lt_scalar,
    rel_gteq_scalar,
    rel_lteq_scalar,
    rel_eq_scalar,
    rel_neq_scalar,
    rel_gt_array,
    rel_lt_array,
    rel_gteq_array,
    rel_lteq_array,
    rel_eq_array,
    rel_neq_array,
    rel_and_array,
    rel_or_array,
    rel_all_neq,
    rel_all_neq_small,
    rel_all_neq_colwise,
    rel_all_neq_rowwise,
    rel_any_neq,
    rel_any_neq_small,
    rel_any_neq_colwise,
    rel_any_neq_rowwise,
    //
    symmatu,
    symmatl,
    //
    cross,
    //
    invalid_kernel
    };



  static
  inline
  std::vector<std::string>
  init_names()
    {
    std::vector<std::string> names;

    names.push_back("equ_array_plus_scalar");
    names.push_back("equ_array_neg_pre");
    names.push_back("equ_array_neg_post");
    names.push_back("equ_array_minus_scalar_pre_pre");
    names.push_back("equ_array_minus_scalar_pre_post");
    names.push_back("equ_array_minus_scalar_post");
    names.push_back("equ_array_mul_scalar");
    names.push_back("equ_array_div_scalar_pre");
    names.push_back("equ_array_div_scalar_post");
    names.push_back("equ_array_mod_scalar");
    names.push_back("equ_array_square_pre");
    names.push_back("equ_array_square_post");
    names.push_back("equ_array_sqrt_pre");
    names.push_back("equ_array_sqrt_post");
    names.push_back("equ_array_exp_pre");
    names.push_back("equ_array_exp_post");
    names.push_back("equ_array_exp2_pre");
    names.push_back("equ_array_exp2_post");
    names.push_back("equ_array_exp10_pre");
    names.push_back("equ_array_exp10_post");
    names.push_back("equ_array_trunc_exp_pre");
    names.push_back("equ_array_trunc_exp_post");
    names.push_back("equ_array_log_pre");
    names.push_back("equ_array_log_post");
    names.push_back("equ_array_log2_pre");
    names.push_back("equ_array_log2_post");
    names.push_back("equ_array_log10_pre");
    names.push_back("equ_array_log10_post");
    names.push_back("equ_array_trunc_log_pre");
    names.push_back("equ_array_trunc_log_post");
    names.push_back("equ_array_cos_pre");
    names.push_back("equ_array_cos_post");
    names.push_back("equ_array_sin_pre");
    names.push_back("equ_array_sin_post");
    names.push_back("equ_array_tan_pre");
    names.push_back("equ_array_tan_post");
    names.push_back("equ_array_acos_pre");
    names.push_back("equ_array_acos_post");
    names.push_back("equ_array_asin_pre");
    names.push_back("equ_array_asin_post");
    names.push_back("equ_array_atan_pre");
    names.push_back("equ_array_atan_post");
    names.push_back("equ_array_cosh_pre");
    names.push_back("equ_array_cosh_post");
    names.push_back("equ_array_sinh_pre");
    names.push_back("equ_array_sinh_post");
    names.push_back("equ_array_tanh_pre");
    names.push_back("equ_array_tanh_post");
    names.push_back("equ_array_acosh_pre");
    names.push_back("equ_array_acosh_post");
    names.push_back("equ_array_asinh_pre");
    names.push_back("equ_array_asinh_post");
    names.push_back("equ_array_atanh_pre");
    names.push_back("equ_array_atanh_post");
    names.push_back("equ_array_sinc_pre");
    names.push_back("equ_array_sinc_post");
    names.push_back("equ_array_abs");
    names.push_back("equ_array_pow_pre");
    names.push_back("equ_array_pow_post");
    names.push_back("equ_array_floor_pre");
    names.push_back("equ_array_floor_post");
    names.push_back("equ_array_ceil_pre");
    names.push_back("equ_array_ceil_post");
    names.push_back("equ_array_round_pre");
    names.push_back("equ_array_round_post");
    names.push_back("equ_array_trunc_pre");
    names.push_back("equ_array_trunc_post");
    names.push_back("equ_array_sign_pre");
    names.push_back("equ_array_sign_post");
    names.push_back("equ_array_erf_pre");
    names.push_back("equ_array_erf_post");
    names.push_back("equ_array_erfc_pre");
    names.push_back("equ_array_erfc_post");
    names.push_back("equ_array_lgamma_pre");
    names.push_back("equ_array_lgamma_post");
    names.push_back("replace");
    names.push_back("clamp");

    names.push_back("extract_sve1");
    names.push_back("equ_array_plus_scalar_sve1");
    names.push_back("equ_array_minus_scalar_pre_pre_sve1");
    names.push_back("equ_array_minus_scalar_pre_post_sve1");
    names.push_back("equ_array_minus_scalar_post_sve1");
    names.push_back("equ_array_mul_scalar_sve1");
    names.push_back("equ_array_div_scalar_pre_sve1");
    names.push_back("equ_array_div_scalar_post_sve1");

    names.push_back("inplace_sve1_eq_array");
    names.push_back("inplace_sve1_plus_array");
    names.push_back("inplace_sve1_minus_array");
    names.push_back("inplace_sve1_mul_array");
    names.push_back("inplace_sve1_div_array");

    names.push_back("inplace_sve1_eq_sve1");
    names.push_back("inplace_sve1_plus_sve1");
    names.push_back("inplace_sve1_minus_sve1");
    names.push_back("inplace_sve1_mul_sve1");
    names.push_back("inplace_sve1_div_sve1");

    names.push_back("extract_sve2");
    names.push_back("equ_array_plus_scalar_sve2");
    names.push_back("equ_array_minus_scalar_pre_pre_sve2");
    names.push_back("equ_array_minus_scalar_pre_post_sve2");
    names.push_back("equ_array_minus_scalar_post_sve2");
    names.push_back("equ_array_mul_scalar_sve2");
    names.push_back("equ_array_div_scalar_pre_sve2");
    names.push_back("equ_array_div_scalar_post_sve2");

    names.push_back("inplace_sve2_eq_array");
    names.push_back("inplace_sve2_plus_array");
    names.push_back("inplace_sve2_minus_array");
    names.push_back("inplace_sve2_mul_array");
    names.push_back("inplace_sve2_div_array");

    names.push_back("inplace_sve2_eq_sve2");
    names.push_back("inplace_sve2_plus_sve2");
    names.push_back("inplace_sve2_minus_sve2");
    names.push_back("inplace_sve2_mul_sve2");
    names.push_back("inplace_sve2_div_sve2");

    names.push_back("equ_array_min_array_cube");
    names.push_back("equ_array_max_array_cube");

    names.push_back("sum_colwise_conv_pre");
    names.push_back("sum_rowwise_conv_pre");
    names.push_back("sum_colwise_conv_post");
    names.push_back("sum_rowwise_conv_post");
    names.push_back("min_colwise_conv_pre");
    names.push_back("min_rowwise_conv_pre");
    names.push_back("min_colwise_conv_post");
    names.push_back("min_rowwise_conv_post");
    names.push_back("min_cube_col_conv_pre");
    names.push_back("min_cube_col_conv_post");
    names.push_back("max_colwise_conv_pre");
    names.push_back("max_rowwise_conv_pre");
    names.push_back("max_colwise_conv_post");
    names.push_back("max_rowwise_conv_post");
    names.push_back("max_cube_col_conv_pre");
    names.push_back("max_cube_col_conv_post");
    names.push_back("mean_colwise_conv_pre");
    names.push_back("mean_rowwise_conv_pre");
    names.push_back("mean_colwise_conv_post");
    names.push_back("mean_rowwise_conv_post");

    names.push_back("dot");
    names.push_back("dot_small");

    names.push_back("convert_type");
    names.push_back("convert_type_cube");

    names.push_back("broadcast_set");
    names.push_back("broadcast_plus");
    names.push_back("broadcast_minus_pre");
    names.push_back("broadcast_minus_post");
    names.push_back("broadcast_schur");
    names.push_back("broadcast_div_pre");
    names.push_back("broadcast_div_post");
    names.push_back("broadcast_subset_set");
    names.push_back("broadcast_subset_plus");
    names.push_back("broadcast_subset_minus_pre");
    names.push_back("broadcast_subset_minus_post");
    names.push_back("broadcast_subset_schur");
    names.push_back("broadcast_subset_div_pre");
    names.push_back("broadcast_subset_div_post");

    names.push_back("htrans");
    names.push_back("strans");

    names.push_back("rel_gt_scalar");
    names.push_back("rel_lt_scalar");
    names.push_back("rel_gteq_scalar");
    names.push_back("rel_lteq_scalar");
    names.push_back("rel_eq_scalar");
    names.push_back("rel_neq_scalar");
    names.push_back("rel_gt_array");
    names.push_back("rel_lt_array");
    names.push_back("rel_gteq_array");
    names.push_back("rel_lteq_array");
    names.push_back("rel_eq_array");
    names.push_back("rel_neq_array");
    names.push_back("rel_and_array");
    names.push_back("rel_or_array");
    names.push_back("rel_all_neq");
    names.push_back("rel_all_neq_small");
    names.push_back("rel_all_neq_colwise");
    names.push_back("rel_all_neq_rowwise");
    names.push_back("rel_any_neq");
    names.push_back("rel_any_neq_small");
    names.push_back("rel_any_neq_colwise");
    names.push_back("rel_any_neq_rowwise");

    names.push_back("symmatu");
    names.push_back("symmatl");

    names.push_back("cross");

    return names;
    }



  static
  inline
  const std::vector<std::string>&
  get_names()
    {
    static const std::vector<std::string> names = init_names();

    return names;
    }



  static
  inline
  std::unordered_map<enum_id, std::vector<std::string>>
  init_deps()
    {
    std::unordered_map<enum_id, std::vector<std::string>> deps;

    deps[rel_any_neq] = { "or_subgroup_reduce_u32" };
    deps[rel_all_neq] = { "and_subgroup_reduce_u32" };

    return deps;
    }



  static
  inline
  const std::unordered_map<enum_id, std::vector<std::string>>&
  get_deps()
    {
    static const std::unordered_map<enum_id, std::vector<std::string>> deps = init_deps();

    return deps;
    }

  };



struct threeway_kernel_id
  {
  enum enum_id
    {
    equ_array_plus_array = 0,
    equ_array_minus_array,
    equ_array_mul_array,
    equ_array_div_array,
    //
    equ_array_min_array,
    equ_array_max_array,
    //
    equ_array_atan2,
    equ_array_hypot,
    //
    equ_array_plus_array_cube,
    equ_array_minus_array_cube,
    equ_array_mul_array_cube,
    equ_array_div_array_cube,
    //
    invalid_kernel
    };



  static
  inline
  std::vector<std::string>
  init_names()
    {
    std::vector<std::string> names;

    names.push_back("equ_array_plus_array");
    names.push_back("equ_array_minus_array");
    names.push_back("equ_array_mul_array");
    names.push_back("equ_array_div_array");

    names.push_back("equ_array_min_array");
    names.push_back("equ_array_max_array");

    names.push_back("equ_array_atan2");
    names.push_back("equ_array_hypot");

    names.push_back("equ_array_plus_array_cube");
    names.push_back("equ_array_minus_array_cube");
    names.push_back("equ_array_mul_array_cube");
    names.push_back("equ_array_div_array_cube");

    return names;
    }



  static
  inline
  const std::vector<std::string>&
  get_names()
    {
    static const std::vector<std::string> names = init_names();

    return names;
    }



  static
  inline
  std::unordered_map<enum_id, std::vector<std::string>>
  init_deps()
    {
    std::unordered_map<enum_id, std::vector<std::string>> deps;

    return deps;
    }



  static
  inline
  const std::unordered_map<enum_id, std::vector<std::string>>&
  get_deps()
    {
    static const std::unordered_map<enum_id, std::vector<std::string>> deps = init_deps();

    return deps;
    }

  };
