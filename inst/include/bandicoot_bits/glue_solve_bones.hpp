// SPDX-License-Identifier: Apache-2.0
// 
// Copyright 2023 Ryan Curtin (https://www.ratml.org)
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



class glue_solve
  : public traits_glue_default
  {
  public:

  template<typename out_eT, typename T1, typename T2>
  inline static void apply(Mat<out_eT>& out, const Glue<T1, T2, glue_solve>& in);

  template<typename out_eT, typename eT, typename T1, typename T2>
  inline static std::tuple<bool, std::string> apply(Mat<out_eT>& out, const Base<eT, T1>& A_expr, const Base<eT, T2>& B_expr, const uword flags, const typename enable_if<!is_same_type<eT, out_eT>::value>::result* junk = 0);

  template<typename eT, typename T1, typename T2>
  inline static std::tuple<bool, std::string> apply(Mat<eT>& out, const Base<eT,                 T1>& A_expr, const Base<eT, T2>& B_expr, const uword flags);

  template<typename eT, typename T1, typename T2>
  inline static std::tuple<bool, std::string> apply(Mat<eT>& out, const Base<eT,  Op<T1, op_htrans>>& A_expr, const Base<eT, T2>& B_expr, const uword flags);

  template<typename eT, typename T1, typename T2>
  inline static std::tuple<bool, std::string> apply(Mat<eT>& out, const Base<eT, Op<T1, op_htrans2>>& A_expr, const Base<eT, T2>& B_expr, const uword flags);

  template<typename T1, typename T2> inline static uword compute_n_rows(const Glue<T1, T2, glue_solve>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols);
  template<typename T1, typename T2> inline static uword compute_n_cols(const Glue<T1, T2, glue_solve>& glue, const uword A_n_rows, const uword A_n_cols, const uword B_n_rows, const uword B_n_cols);
  };



namespace solve_opts
  {
  struct opts
    {
    const uword flags;

    inline constexpr explicit opts(const uword in_flags);

    inline const opts operator+(const opts& rhs) const;
    };

  inline
  constexpr
  opts::opts(const uword in_flags)
    : flags(in_flags)
    {}

  inline
  const opts
  opts::operator+(const opts& rhs) const
    {
    const opts result( flags | rhs.flags );

    return result;
    }

  // The values below (e.g. 1u << 1) are for internal Bandicoot use only.
  // The values can change without notice.

  static constexpr uword flag_none         = uword(0       );
  static constexpr uword flag_fast         = uword(1u <<  0);
  // TODO: port more Armadillo solve options

  struct opts_none         : public opts { inline constexpr opts_none()         : opts(flag_none        ) {} };
  struct opts_fast         : public opts { inline constexpr opts_fast()         : opts(flag_fast        ) {} };

  static constexpr opts_none         none;
  static constexpr opts_fast         fast;
  }
