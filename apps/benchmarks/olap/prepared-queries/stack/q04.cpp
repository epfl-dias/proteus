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

// Stack SO-Q4 (Bao template q4) -- 7 tables, date-interval predicate,
// COUNT(DISTINCT).
//   select COUNT(distinct account.display_name)
//   from tag t1, site s1, question q1, answer a1, tag_question tq1,
//        so_user u1, account
//   where s1.site_name = 'physics' and t1.name = 'homework-and-exercises'
//     and t1.site_id = s1.site_id and q1.site_id = s1.site_id
//     and tq1.site_id = s1.site_id
//     and tq1.question_id = q1.id and tq1.tag_id = t1.id
//     and a1.site_id = q1.site_id and a1.question_id = q1.id
//     and a1.owner_user_id = u1.id and a1.site_id = u1.site_id
//     and a1.creation_date >= q1.creation_date + '1 year'::interval
//     and account.id = u1.account_id
//
// REWRITES (verified in the DuckDB baseline, see
// The benchmark uses the accepted-answer predicate below:
//  * COUNT(DISTINCT account.display_name) -> COUNT(DISTINCT u1.account_id)
//    and the account table dropped: the join only fetches display_name.
//    Baseline checks: account.id is unique and no u1.account_id dangles, so
//    the inner join is exactly an `account_id IS NOT NULL` filter -> here
//    `account_id <> -1` (the export's NULL sentinel).
//  * '+ 1 year' on int YYYYMMDD dates = + 10000 (baseline uses the same int
//    arithmetic, so verification is exact; time-of-day is dropped by the
//    export convention).
// COUNT(DISTINCT) = groupby(account_id) partials -> final groupby -> count of
// groups, the TPC-H Q13-test pattern.

constexpr auto query = "stack_Q4";

PreparedStatement stack::Query::prepareQ4(proteus::QueryShaper &morph) {
  morph.setQueryName(query);

  auto rel_site = morph.scan("site", {"site_id", "site_name"});
  auto rel_tag = morph.scan("tag", {"id", "site_id", "name"});
  auto rel_question = morph.scan("question", {"id", "site_id", "creation_date"});
  auto rel_answer = morph.scan(
      "answer", {"question_id", "site_id", "owner_user_id", "creation_date"});
  auto rel_user = morph.scan("so_user", {"id", "site_id", "account_id"});
  auto rel = morph.scan("tag_question", {"question_id", "tag_id", "site_id"});

  return morph
      .parallel(
          rel, {rel_site, rel_tag, rel_question, rel_answer, rel_user},
          [&](RelBuilder probe, std::vector<RelBuilder> build) {
            auto site_d =
                build.at(0)
                    .unpack()
                    .filter([&](const auto &arg) -> expression_t {
                      return expressions::hint(
                          eq(arg["site_name"], "physics"),
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
                          eq(arg["name"], "homework-and-exercises"),
                          expressions::Selectivity(1.0 / 100'000));
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
                              arg["creation_date"].as("question.csv",
                                                      "q_creat")};
                    });

            auto answer_d =
                build.at(3).unpack().project(
                    [&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["question_id"].as("answer.csv", "a_q"),
                              arg["site_id"].as("answer.csv", "a_site"),
                              arg["owner_user_id"].as("answer.csv", "a_owner"),
                              arg["creation_date"].as("answer.csv", "a_creat")};
                    });

            auto user_d =
                build.at(4)
                    .unpack()
                    // stands in for the account inner join (see header)
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
                .join(
                    tag_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["t_id"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["tag_id"];
                    },
                    9, 512)
                .filter([&](const auto &arg) -> expression_t {
                  return expressions::hint(eq(arg["t_site"], arg["site_id"]),
                                           expressions::Selectivity(0.9));
                })
                .join(
                    question_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["q_id"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["question_id"];
                    },
                    25, 14'000'000)
                .filter([&](const auto &arg) -> expression_t {
                  return expressions::hint(eq(arg["q_site"], arg["site_id"]),
                                           expressions::Selectivity(0.9));
                })
                // all answers of the qualifying questions (multi-match), then
                // site + the one-year interval residuals
                .join(
                    answer_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["a_q"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["question_id"];
                    },
                    23, 7'000'000)
                .filter([&](const auto &arg) -> expression_t {
                  return expressions::hint(
                      eq(arg["a_site"], arg["site_id"]) &
                          ge(arg["a_creat"], arg["q_creat"] + 10000),
                      expressions::Selectivity(0.2));
                })
                .join(
                    user_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["u_id"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["a_owner"];
                    },
                    25, 22'000'000)
                .filter([&](const auto &arg) -> expression_t {
                  return expressions::hint(eq(arg["u_site"], arg["site_id"]),
                                           expressions::Selectivity(0.9));
                })
                // COUNT(DISTINCT u_acct): partial dedup per partition
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
      // final dedup across partitions, then count the groups
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
