/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2026
        Data Intensive Applications and Systems Laboratory (DIAS)
                École Polytechnique Fédérale de Lausanne

                            All Rights Reserved.

    Permission to use, copy, modify and distribute this software and
    its documentation is hereby granted, provided that both the
    copyright notice and this permission notice appear in all copies of
    the software, derivative works or modified versions, and any
    portions thereof, and that both notices appear in supporting
    documentation.

    This code is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. THE AUTHORS
    DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER
    RESULTING FROM THE USE OF THIS SOFTWARE.
*/

#include <stack/query.hpp>

// Stack SO-Q6 (Bao template q6) -- 7 tables, column-vs-column comparison,
// COUNT(DISTINCT). Pulls in the large `comment` table (103M rows, the probe).
//   select COUNT(distinct account.display_name)
//   from tag t1, site s1, question q1, tag_question tq1, so_user u1,
//        comment c1, account
//   where s1.site_name = 'dba' and t1.name in ('cloud','contained-database')
//     and t1.site_id = s1.site_id and q1.site_id = s1.site_id
//     and tq1.site_id = s1.site_id
//     and tq1.question_id = q1.id and tq1.tag_id = t1.id
//     and q1.owner_user_id = u1.id and q1.site_id = u1.site_id
//     and c1.site_id = q1.site_id and c1.post_id = q1.id
//     and c1.score > q1.score
//     and account.id = u1.account_id
//
// Same COUNT(DISTINCT)/account rewrite as SO-Q4 (q04.cpp).
// `c1.score > q1.score` is an ordinary column-vs-column residual after the
// comment |><| question join. Probe = comment; the question/tag_question/tag/
// user sides are (filtered) build hash tables, with every site equality as a
// residual on the carried build-side site_id.

constexpr auto query = "stack_Q6";

PreparedStatement stack::Query::prepareQ6(proteus::QueryShaper &morph) {
  morph.setQueryName(query);

  auto rel_site = morph.scan("site", {"site_id", "site_name"});
  auto rel_tag = morph.scan("tag", {"id", "site_id", "name"});
  auto rel_question =
      morph.scan("question", {"id", "site_id", "owner_user_id", "score"});
  auto rel_tq = morph.scan("tag_question", {"question_id", "tag_id", "site_id"});
  auto rel_user = morph.scan("so_user", {"id", "site_id", "account_id"});
  auto rel = morph.scan("comment", {"post_id", "site_id", "score"});

  return morph
      .parallel(
          rel, {rel_site, rel_tag, rel_question, rel_tq, rel_user},
          [&](RelBuilder probe, std::vector<RelBuilder> build) {
            auto site_d =
                build.at(0)
                    .unpack()
                    .filter([&](const auto &arg) -> expression_t {
                      return expressions::hint(
                          eq(arg["site_name"], "dba"),
                          expressions::Selectivity(1.0 / 170));
                    })
                    .project([&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["site_id"].as("site.csv", "s_site")};
                    });

            auto tag_d =
                build.at(1)
                    .unpack()
                    .filter([&](const auto &arg) -> expression_t {
                      return expressions::hint(
                          eq(arg["name"], "cloud") |
                              eq(arg["name"], "contained-database"),
                          expressions::Selectivity(2.0 / 100'000));
                    })
                    .project([&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["id"].as("tag.csv", "t_id"),
                              arg["site_id"].as("tag.csv", "t_site")};
                    });

            auto question_d =
                build.at(2).unpack().project(
                    [&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["id"].as("question.csv", "q_id"),
                              arg["site_id"].as("question.csv", "q_site"),
                              arg["owner_user_id"].as("question.csv",
                                                      "q_owner"),
                              arg["score"].as("question.csv", "q_score")};
                    });

            auto tq_d =
                build.at(3).unpack().project(
                    [&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["question_id"].as("tag_question.csv", "tq_q"),
                              arg["tag_id"].as("tag_question.csv", "tq_tag"),
                              arg["site_id"].as("tag_question.csv", "tq_site")};
                    });

            auto user_d =
                build.at(4)
                    .unpack()
                    .filter([&](const auto &arg) -> expression_t {
                      return expressions::hint(
                          ne(arg["account_id"], -1),
                          expressions::Selectivity(0.95));
                    })
                    .project([&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["id"].as("so_user.csv", "u_id"),
                              arg["site_id"].as("so_user.csv", "u_site"),
                              arg["account_id"].as("so_user.csv", "u_acct")};
                    });

            return probe.unpack()
                .join(
                    site_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["s_site"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["site_id"];
                    },
                    4, 16)
                // c1.post_id = q1.id, then site equality and the
                // column-vs-column c1.score > q1.score as residuals
                .join(
                    question_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["q_id"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["post_id"];
                    },
                    25, 14'000'000)
                .filter([&](const auto &arg) -> expression_t {
                  return expressions::hint(
                      eq(arg["q_site"], arg["site_id"]) &
                          gt(arg["score"], arg["q_score"]),
                      expressions::Selectivity(0.2));
                })
                // tq1.question_id = q1.id (multi-match: all tags of the
                // question), site as residual
                .join(
                    tq_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["tq_q"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["post_id"];
                    },
                    26, 40'000'000)
                .filter([&](const auto &arg) -> expression_t {
                  return expressions::hint(eq(arg["tq_site"], arg["site_id"]),
                                           expressions::Selectivity(0.9));
                })
                .join(
                    tag_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["t_id"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["tq_tag"];
                    },
                    9, 512)
                .filter([&](const auto &arg) -> expression_t {
                  return expressions::hint(eq(arg["t_site"], arg["site_id"]),
                                           expressions::Selectivity(0.9));
                })
                .join(
                    user_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["u_id"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["q_owner"];
                    },
                    25, 22'000'000)
                .filter([&](const auto &arg) -> expression_t {
                  return expressions::hint(eq(arg["u_site"], arg["site_id"]),
                                           expressions::Selectivity(0.9));
                })
                .groupby(
                    [&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["u_acct"].as("tmp", "u_acct")};
                    },
                    [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                      return {GpuAggrMatExpr{
                          expression_t{1}.as("tmp", "cnt"), 1, 0, SUM}};
                    },
                    16, 1u << 16)
                .pack();
          })
      .unpack()
      .groupby(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["u_acct"]};
          },
          [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
            return {GpuAggrMatExpr{arg["cnt"], 1, 0, SUM}};
          },
          16, 1u << 16)
      .reduce(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {expression_t{1}.as("tmp", "distinct_accounts")};
          },
          {SUM})
      .print(pg{"pm-csv"})
      .prepare();
}
