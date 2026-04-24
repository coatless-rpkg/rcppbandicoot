// SPDX-License-Identifier: Apache-2.0
//
// Copyright 2017-2026 Ryan Curtin (https://www.ratml.org)
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



template<size_t i, typename arg_name1 = kernel_gen::empty_str, typename arg_name2 = kernel_gen::eop_scalar_arg_name>
struct eop_arg_names { };

template<typename arg_name1, typename arg_name2> struct eop_arg_names<0, arg_name1, arg_name2> : public arg_name1 { };
template<typename arg_name1, typename arg_name2> struct eop_arg_names<1, arg_name1, arg_name2> : public arg_name2 { };

template<size_t i> using eop_empty_arg_names = eop_arg_names<i>;




template<typename eop_type>
class eop_core
  {
  public:

  //
  // default implementation: use a Proxy and the copy skeleton kernel
  //

  template<typename eT, typename T1> inline static void apply(Mat<eT>& out,  const eOp<T1, eop_type>& x);
  template<typename eT, typename T1> inline static void apply(Cube<eT>& out, const eOpCube<T1, eop_type>& x);

  // default: no extra types necessary
  template<typename T1>
  using extra_kernel_types = std::tuple<>;
  };



// every eop has the ability to apply a conversion before or after; if 'chainable' is true, then
// it is possible to apply a conversion *between* two eops of the same type



class eop_neg               : public eop_core<eop_neg>
  {
  public:

  // no extra arguments needed for the kernel
  const static size_t num_args = 0;

  // inline eT coot_neg(const eT x) { return -x; }
  struct prefix    { static inline constexpr auto& str() { return "n";        } };
  struct func_name { static inline constexpr auto& str() { return "coot_neg"; } };
  struct func_body { static inline constexpr auto& str() { return "-x";       } };

  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, eop_empty_arg_names, func_body > { };
  };



class eop_scalar_plus : public eop_core<eop_scalar_plus>
  {
  public:

  // one extra argument needed for the kernel
  const static size_t num_args = 1;

  // no extra types necessary
  template<typename T1>
  using extra_kernel_types = std::tuple<>;

  // no need for a definition---this is already in the basic definitions for the type
  struct prefix    { static inline constexpr auto& str() { return "p";         } };
  struct func_name { static inline constexpr auto& str() { return "coot_plus"; } };

  template<typename eT, coot_backend_t backend>
  using aux_functions = kernel_gen::empty_str;
  };



class eop_scalar_minus_pre  : public eop_core<eop_scalar_minus_pre>
  {
  public:

  // one scalar argument needed for the kernel
  const static size_t num_args = 1;

  template<size_t i> using arg_names = eop_arg_names<i, kernel_gen::eop_scalar_arg_name>;

  // inline eT coot_minus_pre(const eT x, const eT a) { return a - x; }
  struct prefix    { static inline constexpr auto& str() { return "mP";             } };
  struct func_name { static inline constexpr auto& str() { return "coot_minus_pre"; } };
  struct func_body { static inline constexpr auto& str() { return "a - x";          } };

  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, arg_names, func_body > { };
  };



class eop_scalar_minus_post : public eop_core<eop_scalar_minus_post>
  {
  public:

  // one scalar argument needed for the kernel
  const static size_t num_args = 1;

  template<size_t i> using arg_names = eop_arg_names<i, kernel_gen::eop_scalar_arg_name>;

  // no need for a definition---this is already in the basic definitions for the type
  struct prefix    { static inline constexpr auto& str() { return "mp";         } };
  struct func_name { static inline constexpr auto& str() { return "coot_minus"; } };

  template<typename eT, coot_backend_t backend>
  using aux_functions = kernel_gen::empty_str;
  };



class eop_scalar_times      : public eop_core<eop_scalar_times>
  {
  public:

  // one scalar argument needed for the kernel
  const static size_t num_args = 1;

  template<size_t i> using arg_names = eop_arg_names<i, kernel_gen::eop_scalar_arg_name>;

  // no need for a definition---this is already in the basic definitions for the type
  struct prefix    { static inline constexpr auto& str() { return "t";          } };
  struct func_name { static inline constexpr auto& str() { return "coot_times"; } };

  template<typename eT, coot_backend_t backend>
  using aux_functions = kernel_gen::empty_str;
  };



class eop_scalar_div_pre    : public eop_core<eop_scalar_div_pre>
  {
  public:

  // one scalar argument needed for the kernel
  const static size_t num_args = 1;

  template<size_t i> using arg_names = eop_arg_names<i, kernel_gen::eop_scalar_arg_name>;

  // inline eT coot_div_pre(const eT x, const eT a) { return x / a; }
  struct prefix    { static inline constexpr auto& str() { return "dP";           } };
  struct func_name { static inline constexpr auto& str() { return "coot_div_pre"; } };
  struct func_body { static inline constexpr auto& str() { return "a / x";        } };

  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, arg_names, func_body > { };
  };



class eop_scalar_div_post   : public eop_core<eop_scalar_div_post>
  {
  public:

  // one scalar argument needed for the kernel
  const static size_t num_args = 1;

  template<size_t i> using arg_names = eop_arg_names<i, kernel_gen::eop_scalar_arg_name>;

  // no need for a definition---this is already in the basic definitions for the type
  struct prefix    { static inline constexpr auto& str() { return "dp";       } };
  struct func_name { static inline constexpr auto& str() { return "coot_div"; } };

  template<typename eT, coot_backend_t backend>
  using aux_functions = kernel_gen::empty_str;
  };

class eop_square            : public eop_core<eop_square>
  {
  public:

  // no extra arguments needed for the kernel
  const static size_t num_args = 0;

  // inline eT coot_square(const eT x) { return x * x; }
  struct prefix    { static inline constexpr auto& str() { return "sqr";         } };
  struct func_name { static inline constexpr auto& str() { return "coot_square"; } };
  struct func_body { static inline constexpr auto& str() { return "x * x";       } };

  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, eop_empty_arg_names, func_body > { };
  };



// for generic types that need to be converted to floating point
template<typename eT, typename fp_eT, typename fp_func_name, coot_backend_t backend, bool is_fp = false>
struct eop_fp_cast_func_body_helper : public kernel_gen::concat_str
  <
  // cast back from FP type
  typename kernel_gen::conv_elem_type_str<eT, backend>,
  kernel_gen::open_paren,
  // call the function on the FP representation
  fp_func_name,
  kernel_gen::open_paren,
  // cast to FP type
  typename kernel_gen::conv_elem_type_str<fp_eT, backend>,
  kernel_gen::paren_x, // (x)
  kernel_gen::double_close_paren
  > { };

template<typename eT, typename fp_eT, typename fp_func_name, coot_backend_t backend>
struct eop_fp_cast_func_body_helper<eT, fp_eT, fp_func_name, backend, true> : public kernel_gen::concat_str
  <
  fp_func_name,
  kernel_gen::paren_x
  > { };

template<typename eT, typename fp_func_name, coot_backend_t backend, typename fp_eT = typename promote_fp_type<eT>::result>
struct eop_fp_cast_func_body : public eop_fp_cast_func_body_helper<eT, fp_eT, fp_func_name, backend, is_real<eT>::value> { };

class eop_sqrt              : public eop_core<eop_sqrt>
  {
  public:

  // no extra arguments needed for the kernel
  const static size_t num_args = 0;

  template<typename T1>
  using extra_kernel_types = std::tuple< typename promote_fp_type<typename T1::elem_type>::result >;

  // for floating-point types: (more complicated for other types)
  // inline eT coot_sqrt(const eT x) { return sqrt(x); }
  struct prefix    { static inline constexpr auto& str() { return "rt";        } };
  struct func_name { static inline constexpr auto& str() { return "coot_sqrt"; } };
  struct func_body { static inline constexpr auto& str() { return "sqrt";      } };

  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, eop_empty_arg_names, eop_fp_cast_func_body<eT, func_body, backend> > { };
  };



class eop_log               : public eop_core<eop_log>
  {
  public:

  // no extra arguments needed for the kernel
  const static size_t num_args = 0;

  template<typename T1>
  using extra_kernel_types = std::tuple< typename promote_fp_type<typename T1::elem_type>::result >;

  // for floating-point types: (more complicated for other types)
  // inline eT coot_log(const eT x) { return log(x); }
  struct prefix    { static inline constexpr auto& str() { return "l";        } };
  struct func_name { static inline constexpr auto& str() { return "coot_log"; } };
  struct func_body { static inline constexpr auto& str() { return "log";      } };

  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, eop_empty_arg_names, eop_fp_cast_func_body<eT, func_body, backend> > { };
  };



class eop_log2              : public eop_core<eop_log2>
  {
  public:

  // no extra arguments needed for the kernel
  const static size_t num_args = 0;

  template<typename T1>
  using extra_kernel_types = std::tuple< typename promote_fp_type<typename T1::elem_type>::result >;

  // for floating-point types: (more complicated for other types)
  // inline eT coot_log2(const eT x) { return log2(x); }
  struct prefix    { static inline constexpr auto& str() { return "l2";        } };
  struct func_name { static inline constexpr auto& str() { return "coot_log2"; } };
  struct func_body { static inline constexpr auto& str() { return "log2";      } };

  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, eop_empty_arg_names, eop_fp_cast_func_body<eT, func_body, backend> > { };
  };



class eop_log10             : public eop_core<eop_log10>
  {
  public:

  // no extra arguments needed for the kernel
  const static size_t num_args = 0;

  template<typename T1>
  using extra_kernel_types = std::tuple< typename promote_fp_type<typename T1::elem_type>::result >;

  // for floating-point types: (more complicated for other types)
  // inline eT coot_log10(const eT x) { return log10(x); }
  struct prefix    { static inline constexpr auto& str() { return "l10";        } };
  struct func_name { static inline constexpr auto& str() { return "coot_log10"; } };
  struct func_body { static inline constexpr auto& str() { return "log10";      } };

  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, eop_empty_arg_names, eop_fp_cast_func_body<eT, func_body, backend> > { };
  };



struct eop_trunc_log_func_inner1 { static inline constexpr auto& str() { return "(x <= ";                      } };
struct eop_trunc_log_func_inner2 { static inline constexpr auto& str() { return "(0)) ? log(coot_type_minpos"; } };
struct eop_trunc_log_func_inner3 { static inline constexpr auto& str() { return "(0))) : (coot_isinf";         } };
struct eop_trunc_log_func_inner4 { static inline constexpr auto& str() { return "(x)) ? log(coot_type_max";    } };
struct eop_trunc_log_func_inner5 { static inline constexpr auto& str() { return "(0))) : log(x))";             } };

template<bool is_fp, typename eT, coot_backend_t backend>
struct eop_trunc_log_func : public kernel_gen::concat_str
  <
  // (x <= 0) ? log(coot_type_minpos(eT(0))) : (coot_isinf(x) ? log(coot_type_max(fp_eT(0))) : log(x))
  eop_trunc_log_func_inner1,
  kernel_gen::conv_elem_type_str<eT, backend>,
  eop_trunc_log_func_inner2,
  kernel_gen::func_name_suffix<eT, backend>,
  kernel_gen::open_paren,
  kernel_gen::conv_elem_type_str<eT, backend>,
  eop_trunc_log_func_inner3,
  kernel_gen::func_name_suffix<eT, backend>,
  kernel_gen::open_paren,
  kernel_gen::conv_elem_type_str<eT, backend>,
  eop_trunc_log_func_inner4,
  kernel_gen::func_name_suffix<eT, backend>,
  kernel_gen::open_paren,
  kernel_gen::conv_elem_type_str<eT, backend>,
  eop_trunc_log_func_inner5
  > { };

struct eop_trunc_log_func_inner6 { static inline constexpr auto& str() { return "(x) <= 0) ? log(DBL_MIN) : (coot_isinf"; } };
struct eop_trunc_log_func_inner7 { static inline constexpr auto& str() { return "(x)) ? log(DBL_MAX) : log(";             } };
struct eop_trunc_log_func_inner8 { static inline constexpr auto& str() { return "(x)))";                                  } };

template<typename eT, coot_backend_t backend>
struct eop_trunc_log_func<false, eT, backend> : public kernel_gen::concat_str
  <
  // eT((double(x) <= 0) ? eT(log(DBL_MIN)) : (coot_isinf(double(x)) ? log(DBL_MAX) : log(double(x))))
  kernel_gen::conv_elem_type_str<eT, backend>,
  kernel_gen::open_paren,
  kernel_gen::conv_elem_type_str<double, backend>,
  eop_trunc_log_func_inner6,
  kernel_gen::func_name_suffix<double, backend>,
  kernel_gen::open_paren,
  kernel_gen::conv_elem_type_str<double, backend>,
  eop_trunc_log_func_inner7,
  kernel_gen::conv_elem_type_str<double, backend>,
  eop_trunc_log_func_inner8
  > { };



class eop_trunc_log         : public eop_core<eop_trunc_log>
  {
  public:

  // no extra arguments needed for the kernel
  const static size_t num_args = 0;

  template<typename T1>
  using extra_kernel_types = std::tuple< double >;

  // coot_trunc_log
  struct prefix    { static inline constexpr auto& str() { return "trl";            } };
  struct func_name { static inline constexpr auto& str() { return "coot_trunc_log"; } };

  // different implementations depending on whether eT is integral or floating-point
  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, eop_empty_arg_names, eop_trunc_log_func<is_real<eT>::value, eT, backend> > { };
  };



class eop_exp               : public eop_core<eop_exp>
  {
  public:

  // no extra arguments needed for the kernel
  const static size_t num_args = 0;

  template<typename T1>
  using extra_kernel_types = std::tuple< typename promote_fp_type<typename T1::elem_type>::result >;

  // for floating-point types: (more complicated for other types)
  // inline eT coot_exp(const eT x) { return exp(x); }
  struct prefix    { static inline constexpr auto& str() { return "e";        } };
  struct func_name { static inline constexpr auto& str() { return "coot_exp"; } };
  struct func_body { static inline constexpr auto& str() { return "exp";      } };

  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, eop_empty_arg_names, eop_fp_cast_func_body<eT, func_body, backend> > { };
  };



class eop_exp2              : public eop_core<eop_exp2>
  {
  public:

  // no extra arguments needed for the kernel
  const static size_t num_args = 0;

  template<typename T1>
  using extra_kernel_types = std::tuple< typename promote_fp_type<typename T1::elem_type>::result >;

  // for floating-point types: (more complicated for other types)
  // inline eT coot_exp2(const eT x) { return exp2(x); }
  struct prefix    { static inline constexpr auto& str() { return "e2";        } };
  struct func_name { static inline constexpr auto& str() { return "coot_exp2"; } };
  struct func_body { static inline constexpr auto& str() { return "exp2";      } };

  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, eop_empty_arg_names, eop_fp_cast_func_body<eT, func_body, backend> > { };
  };



class eop_exp10             : public eop_core<eop_exp10>
  {
  public:

  // no extra arguments needed for the kernel
  const static size_t num_args = 0;

  template<typename T1>
  using extra_kernel_types = std::tuple< typename promote_fp_type<typename T1::elem_type>::result >;

  // for floating-point types: (more complicated for other types)
  // inline eT coot_exp10(const eT x) { return exp10(x); }
  struct prefix    { static inline constexpr auto& str() { return "e10";        } };
  struct func_name { static inline constexpr auto& str() { return "coot_exp10"; } };
  struct func_body { static inline constexpr auto& str() { return "exp10";      } };

  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, eop_empty_arg_names, eop_fp_cast_func_body<eT, func_body, backend> > { };
  };



struct eop_trunc_exp_func_inner1 { static inline constexpr auto& str() { return "((x >= log(coot_type_max"; } };
struct eop_trunc_exp_func_inner2 { static inline constexpr auto& str() { return "(0)))) ? coot_type_max";   } };
struct eop_trunc_exp_func_inner3 { static inline constexpr auto& str() { return "(0)) : exp(x))";           } };

template<bool is_fp, typename eT, coot_backend_t backend>
struct eop_trunc_exp_func : public kernel_gen::concat_str
  <
  // ((x >= log(coot_type_max(eT(0)))) ? coot_type_max(eT(0)) : exp(x))
  eop_trunc_exp_func_inner1,
  kernel_gen::func_name_suffix<eT, backend>,
  kernel_gen::open_paren,
  kernel_gen::conv_elem_type_str<eT, backend>,
  eop_trunc_exp_func_inner2,
  kernel_gen::func_name_suffix<eT, backend>,
  kernel_gen::open_paren,
  kernel_gen::conv_elem_type_str<eT, backend>,
  eop_trunc_exp_func_inner3
  > { };

struct eop_trunc_exp_func_inner4 { static inline constexpr auto& str() { return "(x) >= log(DBL_MAX)) ? DBL_MAX : exp("; } };
struct eop_trunc_exp_func_inner5 { static inline constexpr auto& str() { return "(x)))";                                 } };

template<typename eT, coot_backend_t backend>
struct eop_trunc_exp_func<false, eT, backend> : public kernel_gen::concat_str
  <
  // eT((double(x) >= log(DBL_MAX)) ? DBL_MAX : exp(double(x)))
  kernel_gen::conv_elem_type_str<eT, backend>,
  kernel_gen::double_open_paren,
  kernel_gen::conv_elem_type_str<double, backend>,
  eop_trunc_exp_func_inner4,
  kernel_gen::conv_elem_type_str<double, backend>,
  eop_trunc_exp_func_inner5
  > { };

class eop_trunc_exp         : public eop_core<eop_trunc_exp>
  {
  public:

  // no extra arguments needed for the kernel
  const static size_t num_args = 0;

  template<typename T1>
  using extra_kernel_types = std::tuple< double >;

  // coot_trunc_exp
  struct prefix    { static inline constexpr auto& str() { return "tre";            } };
  struct func_name { static inline constexpr auto& str() { return "coot_trunc_exp"; } };

  // different implementations depending on whether eT is integral or floating-point
  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, eop_empty_arg_names, eop_trunc_exp_func<is_real<eT>::value, eT, backend> > { };
  };



class eop_cos               : public eop_core<eop_cos>
  {
  public:

  // no extra arguments needed for the kernel
  const static size_t num_args = 0;

  template<typename T1>
  using extra_kernel_types = std::tuple< typename promote_fp_type<typename T1::elem_type>::result >;

  // for floating-point types: (more complicated for other types)
  // inline eT coot_cos(const eT x) { return cos(x); }
  struct prefix    { static inline constexpr auto& str() { return "Tc";       } };
  struct func_name { static inline constexpr auto& str() { return "coot_cos"; } };
  struct func_body { static inline constexpr auto& str() { return "cos";      } };

  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, eop_empty_arg_names, eop_fp_cast_func_body<eT, func_body, backend> > { };
  };



class eop_sin               : public eop_core<eop_sin>
  {
  public:

  // no extra arguments needed for the kernel
  const static size_t num_args = 0;

  template<typename T1>
  using extra_kernel_types = std::tuple< typename promote_fp_type<typename T1::elem_type>::result >;

  // for floating-point types: (more complicated for other types)
  // inline eT coot_sin(const eT x) { return sin(x); }
  struct prefix    { static inline constexpr auto& str() { return "Ts";       } };
  struct func_name { static inline constexpr auto& str() { return "coot_sin"; } };
  struct func_body { static inline constexpr auto& str() { return "sin";      } };

  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, eop_empty_arg_names, eop_fp_cast_func_body<eT, func_body, backend> > { };
  };



class eop_tan               : public eop_core<eop_tan>
  {
  public:

  // no extra arguments needed for the kernel
  const static size_t num_args = 0;

  template<typename T1>
  using extra_kernel_types = std::tuple< typename promote_fp_type<typename T1::elem_type>::result >;

  // for floating-point types: (more complicated for other types)
  // inline eT coot_tan(const eT x) { return tan(x); }
  struct prefix    { static inline constexpr auto& str() { return "Tt";       } };
  struct func_name { static inline constexpr auto& str() { return "coot_tan"; } };
  struct func_body { static inline constexpr auto& str() { return "tan";      } };

  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, eop_empty_arg_names, eop_fp_cast_func_body<eT, func_body, backend> > { };
  };



class eop_acos              : public eop_core<eop_acos>
  {
  public:

  // no extra arguments needed for the kernel
  const static size_t num_args = 0;

  template<typename T1>
  using extra_kernel_types = std::tuple< typename promote_fp_type<typename T1::elem_type>::result >;

  // for floating-point types: (more complicated for other types)
  // inline eT coot_acos(const eT x) { return acos(x); }
  struct prefix    { static inline constexpr auto& str() { return "Tac";       } };
  struct func_name { static inline constexpr auto& str() { return "coot_acos"; } };
  struct func_body { static inline constexpr auto& str() { return "acos";      } };

  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, eop_empty_arg_names, eop_fp_cast_func_body<eT, func_body, backend> > { };
  };



class eop_asin              : public eop_core<eop_asin>
  {
  public:

  // no extra arguments needed for the kernel
  const static size_t num_args = 0;

  template<typename T1>
  using extra_kernel_types = std::tuple< typename promote_fp_type<typename T1::elem_type>::result >;

  // for floating-point types: (more complicated for other types)
  // inline eT coot_asin(const eT x) { return asin(x); }
  struct prefix    { static inline constexpr auto& str() { return "Tas";       } };
  struct func_name { static inline constexpr auto& str() { return "coot_asin"; } };
  struct func_body { static inline constexpr auto& str() { return "asin";      } };

  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, eop_empty_arg_names, eop_fp_cast_func_body<eT, func_body, backend> > { };
  };



class eop_atan              : public eop_core<eop_atan>
  {
  public:

  // no extra arguments needed for the kernel
  const static size_t num_args = 0;

  template<typename T1>
  using extra_kernel_types = std::tuple< typename promote_fp_type<typename T1::elem_type>::result >;

  // for floating-point types: (more complicated for other types)
  // inline eT coot_atan(const eT x) { return atan(x); }
  struct prefix    { static inline constexpr auto& str() { return "Tat";       } };
  struct func_name { static inline constexpr auto& str() { return "coot_atan"; } };
  struct func_body { static inline constexpr auto& str() { return "atan";      } };

  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, eop_empty_arg_names, eop_fp_cast_func_body<eT, func_body, backend> > { };
  };



class eop_cosh              : public eop_core<eop_cosh>
  {
  public:

  // no extra arguments needed for the kernel
  const static size_t num_args = 0;

  template<typename T1>
  using extra_kernel_types = std::tuple< typename promote_fp_type<typename T1::elem_type>::result >;

  // for floating-point types: (more complicated for other types)
  // inline eT coot_cosh(const eT x) { return cosh(x); }
  struct prefix    { static inline constexpr auto& str() { return "Tch";       } };
  struct func_name { static inline constexpr auto& str() { return "coot_cosh"; } };
  struct func_body { static inline constexpr auto& str() { return "cosh";      } };

  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, eop_empty_arg_names, eop_fp_cast_func_body<eT, func_body, backend> > { };
  };



class eop_sinh              : public eop_core<eop_sinh>
  {
  public:

  // no extra arguments needed for the kernel
  const static size_t num_args = 0;

  template<typename T1>
  using extra_kernel_types = std::tuple< typename promote_fp_type<typename T1::elem_type>::result >;

  // for floating-point types: (more complicated for other types)
  // inline eT coot_sinh(const eT x) { return sinh(x); }
  struct prefix    { static inline constexpr auto& str() { return "Tsh";       } };
  struct func_name { static inline constexpr auto& str() { return "coot_sinh"; } };
  struct func_body { static inline constexpr auto& str() { return "sinh";      } };

  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, eop_empty_arg_names, eop_fp_cast_func_body<eT, func_body, backend> > { };
  };



class eop_tanh              : public eop_core<eop_tanh>
  {
  public:

  // no extra arguments needed for the kernel
  const static size_t num_args = 0;

  template<typename T1>
  using extra_kernel_types = std::tuple< typename promote_fp_type<typename T1::elem_type>::result >;

  // for floating-point types: (more complicated for other types)
  // inline eT coot_tanh(const eT x) { return tanh(x); }
  struct prefix    { static inline constexpr auto& str() { return "Tth";       } };
  struct func_name { static inline constexpr auto& str() { return "coot_tanh"; } };
  struct func_body { static inline constexpr auto& str() { return "tanh";      } };

  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, eop_empty_arg_names, eop_fp_cast_func_body<eT, func_body, backend> > { };
  };



class eop_acosh             : public eop_core<eop_acosh>
  {
  public:

  // no extra arguments needed for the kernel
  const static size_t num_args = 0;

  template<typename T1>
  using extra_kernel_types = std::tuple< typename promote_fp_type<typename T1::elem_type>::result >;

  // for floating-point types: (more complicated for other types)
  // inline eT coot_acosh(const eT x) { return acosh(x); }
  struct prefix    { static inline constexpr auto& str() { return "TAc";       } };
  struct func_name { static inline constexpr auto& str() { return "coot_acosh"; } };
  struct func_body { static inline constexpr auto& str() { return "acosh";      } };

  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, eop_empty_arg_names, eop_fp_cast_func_body<eT, func_body, backend> > { };
  };



class eop_asinh             : public eop_core<eop_asinh>
  {
  public:

  // no extra arguments needed for the kernel
  const static size_t num_args = 0;

  template<typename T1>
  using extra_kernel_types = std::tuple< typename promote_fp_type<typename T1::elem_type>::result >;

  // for floating-point types: (more complicated for other types)
  // inline eT coot_asinh(const eT x) { return asinh(x); }
  struct prefix    { static inline constexpr auto& str() { return "TAs";       } };
  struct func_name { static inline constexpr auto& str() { return "coot_asinh"; } };
  struct func_body { static inline constexpr auto& str() { return "asinh";      } };

  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, eop_empty_arg_names, eop_fp_cast_func_body<eT, func_body, backend> > { };
  };



class eop_atanh             : public eop_core<eop_atanh>
  {
  public:

  // no extra arguments needed for the kernel
  const static size_t num_args = 0;

  template<typename T1>
  using extra_kernel_types = std::tuple< typename promote_fp_type<typename T1::elem_type>::result >;

  // for floating-point types: (more complicated for other types)
  // inline eT coot_atanh(const eT x) { return atanh(x); }
  struct prefix    { static inline constexpr auto& str() { return "TAt";       } };
  struct func_name { static inline constexpr auto& str() { return "coot_atanh"; } };
  struct func_body { static inline constexpr auto& str() { return "atanh";      } };

  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, eop_empty_arg_names, eop_fp_cast_func_body<eT, func_body, backend> > { };
  };


// ((x == eT(0)) ? eT(1) : (sin(x * COOT_PI) / (x * COOT_PI)))
struct eop_sinc_func_body_inner1 { static inline constexpr auto& str() { return "((x == ";                                   } };
struct eop_sinc_func_body_inner2 { static inline constexpr auto& str() { return "(0)) ? ";                                   } };
struct eop_sinc_func_body_inner3 { static inline constexpr auto& str() { return "(1) : (sin(x * COOT_PI) / (x * COOT_PI)))"; } };

template<typename eT, coot_backend_t backend, bool is_fp>
struct eop_sinc_func_body_inner : public kernel_gen::concat_str
  <
  eop_sinc_func_body_inner1,
  kernel_gen::conv_elem_type_str<eT, backend>,
  eop_sinc_func_body_inner2,
  kernel_gen::conv_elem_type_str<eT, backend>,
  eop_sinc_func_body_inner3
  > { };

struct eop_sinc_func_body_inner4 { static inline constexpr auto& str() { return "(1) : ";             } };
struct eop_sinc_func_body_inner5 { static inline constexpr auto& str() { return "(sin(";              } };
struct eop_sinc_func_body_inner6 { static inline constexpr auto& str() { return "(x) * COOT_PI) / ("; } };
struct eop_sinc_func_body_inner7 { static inline constexpr auto& str() { return "(x) * COOT_PI)))";   } };

// ((x == eT(0)) ? eT(1) : eT(sin(double(x) * COOT_PI) / (double(x) * COOT_PI)))
template<typename eT, coot_backend_t backend>
struct eop_sinc_func_body_inner<eT, backend, false> : public kernel_gen::concat_str
  <
  eop_sinc_func_body_inner1,
  kernel_gen::conv_elem_type_str<eT, backend>,
  eop_sinc_func_body_inner2,
  kernel_gen::conv_elem_type_str<eT, backend>,
  eop_sinc_func_body_inner4,
  kernel_gen::conv_elem_type_str<eT, backend>,
  eop_sinc_func_body_inner5,
  kernel_gen::conv_elem_type_str<double, backend>,
  eop_sinc_func_body_inner6,
  kernel_gen::conv_elem_type_str<double, backend>,
  eop_sinc_func_body_inner7
  > { };

template<typename eT, coot_backend_t backend>
struct eop_sinc_func_body : public eop_sinc_func_body_inner< eT, backend, is_real<eT>::value > { };

class eop_sinc              : public eop_core<eop_sinc>
  {
  public:

  // no extra arguments needed for the kernel
  const static size_t num_args = 0;

  template<typename T1>
  using extra_kernel_types = std::tuple< double >;

  // coot_sinc
  struct prefix    { static inline constexpr auto& str() { return "sc";        } };
  struct func_name { static inline constexpr auto& str() { return "coot_sinc"; } };

  // for floating-point types: (more complicated for other types)
  // inline eT coot_sinc(const eT x) { return ((x == eT(0)) ? eT(1) : (sin(x * COOT_PI) / (x * COOT_PI))); }
  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, eop_empty_arg_names, eop_sinc_func_body<eT, backend> > { };
  };



class eop_abs               : public eop_core<eop_abs>
  {
  public:

  // no extra arguments needed for the kernel
  const static size_t num_args = 0;

  // coot_eop_abs (to differentiate from utility coot_abs function)
  struct prefix        { static inline constexpr auto& str() { return "A";       } };
  struct func_name     { static inline constexpr auto& str() { return "coot_eop_abs"; } };
  struct abs_func_name { static inline constexpr auto& str() { return "coot_abs";     } };

  // return coot_abs(x);
  template<typename eT, coot_backend_t backend>
  struct func_body : public kernel_gen::concat_str
    <
    abs_func_name,
    kernel_gen::func_name_suffix<eT, backend>,
    kernel_gen::paren_x // (x)
    > { };

  // inline eT coot_eop_abs(const eT x) { return coot_abs(x); }
  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, eop_empty_arg_names, func_body<eT, backend> > { };
  };



// class eop_arg               : public eop_core<eop_arg>               {};
// class eop_conj              : public eop_core<eop_conj>              {};



struct eop_pow_func_body_inner1 { static inline constexpr auto& str() { return "pow(x, "; } };
struct eop_pow_func_body_inner2 { static inline constexpr auto& str() { return "(a))";    } };

template<typename eT, typename fp_eT, coot_backend_t backend, bool is_fp>
struct eop_pow_func_body_inner : public kernel_gen::concat_str
  <
  eop_pow_func_body_inner1,
  kernel_gen::conv_elem_type_str<eT, backend>,
  eop_pow_func_body_inner2
  > { };

template<typename eT, typename fp_eT, coot_backend_t backend>
struct eop_pow_func_body_inner<eT, fp_eT, backend, false> : public kernel_gen::concat_str
  <
  kernel_gen::conv_elem_type_str<eT, backend>,
  kernel_gen::open_paren,
  eop_pow_func_body_inner1,
  kernel_gen::conv_elem_type_str<fp_eT, backend>,
  eop_pow_func_body_inner2,
  kernel_gen::close_paren
  > { };

template<typename eT, coot_backend_t backend>
struct eop_pow_func_body : public eop_pow_func_body_inner< eT, typename promote_fp_type<eT>::result, backend, is_real<eT>::value > { };

class eop_pow               : public eop_core<eop_pow>
  {
  public:

  // one scalar argument needed for the kernel
  const static size_t num_args = 1;

  template<typename T1>
  using extra_kernel_types = std::tuple< typename promote_fp_type<typename T1::elem_type>::result >;

  // coot_pow
  struct prefix    { static inline constexpr auto& str() { return "P";        } };
  struct func_name { static inline constexpr auto& str() { return "coot_pow"; } };

  template<size_t i> using arg_names = eop_arg_names<i, kernel_gen::eop_scalar_arg_name>;

  // for floating-point types: (more complicated for other types)
  // inline eT coot_pow(const eT x, const eT a) { return pow(x, a); }
  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, arg_names, eop_pow_func_body<eT, backend> > { };
  };



class eop_floor             : public eop_core<eop_floor>
  {
  public:

  // no extra arguments needed for the kernel
  const static size_t num_args = 0;

  template<typename T1>
  using extra_kernel_types = std::tuple< typename promote_fp_type<typename T1::elem_type>::result >;

  // for floating-point types: (more complicated for other types)
  // inline eT coot_floor(const eT x) { return floor(x); }
  struct prefix    { static inline constexpr auto& str() { return "f";          } };
  struct func_name { static inline constexpr auto& str() { return "coot_floor"; } };
  struct func_body { static inline constexpr auto& str() { return "floor";      } };

  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, eop_empty_arg_names, eop_fp_cast_func_body<eT, func_body, backend> > { };
  };



class eop_ceil              : public eop_core<eop_ceil>
  {
  public:

  // no extra arguments needed for the kernel
  const static size_t num_args = 0;

  template<typename T1>
  using extra_kernel_types = std::tuple< typename promote_fp_type<typename T1::elem_type>::result >;

  // for floating-point types: (more complicated for other types)
  // inline eT coot_ceil(const eT x) { return ceil(x); }
  struct prefix    { static inline constexpr auto& str() { return "c"  ;       } };
  struct func_name { static inline constexpr auto& str() { return "coot_ceil"; } };
  struct func_body { static inline constexpr auto& str() { return "ceil";      } };

  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, eop_empty_arg_names, eop_fp_cast_func_body<eT, func_body, backend> > { };
  };



class eop_round             : public eop_core<eop_round>
  {
  public:

  // no extra arguments needed for the kernel
  const static size_t num_args = 0;

  template<typename T1>
  using extra_kernel_types = std::tuple< typename promote_fp_type<typename T1::elem_type>::result >;

  // for floating-point types: (more complicated for other types)
  // inline eT coot_round(const eT x) { return round(x); }
  struct prefix    { static inline constexpr auto& str() { return "r";          } };
  struct func_name { static inline constexpr auto& str() { return "coot_round"; } };
  struct func_body { static inline constexpr auto& str() { return "round";      } };

  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, eop_empty_arg_names, eop_fp_cast_func_body<eT, func_body, backend> > { };
  };



template<bool is_fp> struct eop_trunc_body_inner        { static inline constexpr auto& str() { return "trunc(x)"; } };
template<>           struct eop_trunc_body_inner<false> { static inline constexpr auto& str() { return "x";        } };

class eop_trunc             : public eop_core<eop_trunc>
  {
  public:

  // no extra arguments needed for the kernel
  const static size_t num_args = 0;

  template<typename T1>
  using extra_kernel_types = std::tuple< typename promote_fp_type<typename T1::elem_type>::result >;

  // coot_trunc
  struct prefix    { static inline constexpr auto& str() { return "tr";         } };
  struct func_name { static inline constexpr auto& str() { return "coot_trunc"; } };

  // for floating-point types:
  // inline eT coot_trunc(const eT x) { return trunc(x); }
  //
  // for integral types:
  // inline eT coot_trunc(const eT x) { return x; }
  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, eop_empty_arg_names, eop_trunc_body_inner<is_real<eT>::value> > { };
  };



class eop_sign              : public eop_core<eop_sign>
  {
  public:

  // no extra arguments needed for the kernel
  const static size_t num_args = 0;

  template<typename T1>
  using extra_kernel_types = std::tuple< typename promote_fp_type<typename T1::elem_type>::result >;

  // coot_sign
  struct prefix     { static inline constexpr auto& str() { return "s";         } };
  struct func_name  { static inline constexpr auto& str() { return "coot_sign";     } };
  struct func_body1 { static inline constexpr auto& str() { return "((x > ";        } };
  struct func_body2 { static inline constexpr auto& str() { return "(0)) ? ";       } };
  struct func_body3 { static inline constexpr auto& str() { return "(1) : ((x == "; } };
  struct func_body4 { static inline constexpr auto& str() { return "(0) : ";        } };
  struct func_body5 { static inline constexpr auto& str() { return "(-1)))";        } };

  template<typename eT, coot_backend_t backend>
  struct func_body : public kernel_gen::concat_str
    <
    func_body1,
    kernel_gen::conv_elem_type_str<eT, backend>,
    func_body2,
    kernel_gen::conv_elem_type_str<eT, backend>,
    func_body3,
    kernel_gen::conv_elem_type_str<eT, backend>,
    func_body2,
    kernel_gen::conv_elem_type_str<eT, backend>,
    func_body4,
    kernel_gen::conv_elem_type_str<eT, backend>,
    func_body5
    > { };

  // for floating-point types: (more complicated for other types)
  // inline eT coot_sign(const eT x) { return ((x > eT(0)) ? eT(1) : ((x == eT(0)) ? eT(0) : eT(-1))); }
  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, eop_empty_arg_names, func_body<eT, backend> > { };
  };



class eop_erf               : public eop_core<eop_erf>
  {
  public:

  // no extra arguments needed for the kernel
  const static size_t num_args = 0;

  template<typename T1>
  using extra_kernel_types = std::tuple< typename promote_fp_type<typename T1::elem_type>::result >;

  // for floating-point types: (more complicated for other types)
  // inline eT coot_erf(const eT x) { return erf(x); }
  struct prefix    { static inline constexpr auto& str() { return "erf";      } };
  struct func_name { static inline constexpr auto& str() { return "coot_erf"; } };
  struct func_body { static inline constexpr auto& str() { return "erf";      } };

  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, eop_empty_arg_names, eop_fp_cast_func_body<eT, func_body, backend> > { };
  };



class eop_erfc              : public eop_core<eop_erfc>
  {
  public:

  // no extra arguments needed for the kernel
  const static size_t num_args = 0;

  template<typename T1>
  using extra_kernel_types = std::tuple< typename promote_fp_type<typename T1::elem_type>::result >;

  // for floating-point types: (more complicated for other types)
  // inline eT coot_erfc(const eT x) { return erfc(x); }
  struct prefix    { static inline constexpr auto& str() { return "Erf";       } };
  struct func_name { static inline constexpr auto& str() { return "coot_erfc"; } };
  struct func_body { static inline constexpr auto& str() { return "erfc";      } };

  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, eop_empty_arg_names, eop_fp_cast_func_body<eT, func_body, backend> > { };
  };



class eop_lgamma            : public eop_core<eop_lgamma>
  {
  public:

  // no extra arguments needed for the kernel
  const static size_t num_args = 0;

  template<typename T1>
  using extra_kernel_types = std::tuple< typename promote_fp_type<typename T1::elem_type>::result >;

  // for floating-point types: (more complicated for other types)
  // inline eT coot_lgamma(const eT x) { return lgamma(x); }
  struct prefix    { static inline constexpr auto& str() { return "lg";          } };
  struct func_name { static inline constexpr auto& str() { return "coot_lgamma"; } };
  struct func_body { static inline constexpr auto& str() { return "lgamma";      } };

  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, eop_empty_arg_names, eop_fp_cast_func_body<eT, func_body, backend> > { };
  };



// NOTE: this is not exposed to users and will not work for floating-point eTs
class eop_modulo : public eop_core<eop_modulo>
  {
  public:

  // one scalar argument needed for the kernel
  const static size_t num_args = 1;

  template<size_t i> using arg_names = eop_arg_names<i, kernel_gen::eop_scalar_arg_name>;

  // inline eT coot_modulo(const eT x, const eT a) { return x % a; }
  struct prefix    { static inline constexpr auto& str() { return "mod";         } };
  struct func_name { static inline constexpr auto& str() { return "coot_modulo"; } };
  struct func_body { static inline constexpr auto& str() { return "x % a";       } };

  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, arg_names, func_body > { };

  };



class eop_replace : public eop_core<eop_replace>
  {
  public:

  // two extra arguments needed for the kernel
  const static size_t num_args = 2;

  // no extra types necessary
  template<typename T1>
  using extra_kernel_types = std::tuple<>;

  // "coot_replace()"
  struct prefix          { static inline constexpr auto& str() { return "R";                              } };
  struct func_name       { static inline constexpr auto& str() { return "coot_replace";                   } };
  struct func_body_part1 { static inline constexpr auto& str() { return "((x == val_find) || (";          } };
  struct func_coot_isnan { static inline constexpr auto& str() { return "coot_isnan";                     } };
  struct func_body_part2 { static inline constexpr auto& str() { return "(x) && ";                        } };
  struct func_body_part3 { static inline constexpr auto& str() { return "(val_find))) ? val_replace : x"; } };

  template<typename eT, coot_backend_t backend>
  struct func_body : public kernel_gen::concat_str
    <
    func_body_part1,
    func_coot_isnan,
    kernel_gen::func_name_suffix<eT, backend>,
    func_body_part2,
    func_coot_isnan,
    kernel_gen::func_name_suffix<eT, backend>,
    func_body_part3
    > { };

  struct val_find_name    { static inline constexpr auto& str() { return "val_find";    } };
  struct val_replace_name { static inline constexpr auto& str() { return "val_replace"; } };

  template<size_t i> using arg_names = eop_arg_names<i, val_find_name, val_replace_name>;

  // inline eT coot_replace(const eT x, const eT val_find, const eT val_replace) { return ((x == val_find) || (coot_isnan(x) && coot_isnan(val_find))) ? val_replace : x; }
  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, arg_names, func_body<eT, backend> > { };
  };



template<size_t arg>
struct eop_clamp_arg_names { };

template<> struct eop_clamp_arg_names<0> { static inline constexpr auto& str() { return "min_val"; } };
template<> struct eop_clamp_arg_names<1> { static inline constexpr auto& str() { return "max_val"; } };

struct eop_clamp : public eop_core<eop_clamp>
  {
  public:

  // two extra arguments needed for the kernel
  const static size_t num_args = 2;

  // no extra types necessary
  template<typename T1>
  using extra_kernel_types = std::tuple<>;

  // return coot_max(coot_min(x, max_val), max_val);
  struct prefix             { static inline constexpr auto& str() { return "C";                      } };
  struct func_name          { static inline constexpr auto& str() { return "coot_clamp";             } };
  struct func_coot_min_name { static inline constexpr auto& str() { return "coot_min";               } };
  struct func_coot_max_name { static inline constexpr auto& str() { return "coot_max";               } };
  struct func_body_end      { static inline constexpr auto& str() { return "(x, max_val), min_val)"; } };

  template<typename eT, coot_backend_t backend>
  struct func_body : public kernel_gen::concat_str
    <
    func_coot_max_name,
    kernel_gen::func_name_suffix<eT, backend>,
    kernel_gen::open_paren,
    func_coot_min_name,
    kernel_gen::func_name_suffix<eT, backend>,
    func_body_end
    > { };

  template<size_t i> using arg_names = eop_clamp_arg_names<i>;

  // inline eT coot_clamp(const eT x, const eT min_val, const eT max_val) { return coot_max(coot_min(x, max_val), min_val); }
  template<typename eT, coot_backend_t backend>
  struct aux_functions : public kernel_gen::eop_inline_function< eT, backend, func_name, num_args, eop_clamp_arg_names, func_body<eT, backend> > { };
  };
