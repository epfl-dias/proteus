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

// Stack SO-Q14 (Bao template q14) -- the joins-heavy instance.
//   SELECT COUNT(*)
//   FROM site s, so_user u1, question q1, answer a1, tag t1,
//        tag_question tq1, badge b, account acc
//   WHERE s.site_id = {q1,u1,a1,t1,tq1,b}.site_id
//     AND q1.id = tq1.question_id AND q1.id = a1.question_id
//     AND a1.owner_user_id = u1.id AND t1.id = tq1.tag_id
//     AND b.user_id = u1.id AND acc.id = u1.account_id
//     AND s.site_name IN ('stackoverflow','superuser')
//     AND t1.name IN ('android-asynctask','asp.net-mvc-4','css3',
//                     'entity-framework-4','inner-join','jquery',
//                     'sublimetext3')
//     AND q1.view_count >= 10 AND q1.view_count <= 1000
//     AND u1.downvotes >= 0 AND u1.downvotes <= 10
//     AND b.name IN ('Analytical','Beta','Deputy','Fanatic','Great Question',
//                    'Not a Robot','Unsung Hero')
//
// REWRITE: the account join carries no filter or projection; the DuckDB
// baseline proves account.id unique + no dangling u1.account_id, and running
// the query both ways gives the identical count -- so the join is dropped and
// replaced by `u1.account_id <> -1` (the export's NULL sentinel), exactly as
// in SO-Q4/Q6.
// PROBE ORDER (2026-07-18): probe = tag_question. The original probe=badge
// (51M rows) hit join-fan-out skew: GpuHashJoinChained emits matches serially
// in the probing thread, and a single power user's badge row fans into tens
// of thousands of answers -> per-launch tail latency (nsys: probe launches
// median 1.2 ms, max 26 ms; probe pipeline 219 of 266 ms). Probing tq and
// building badge NAME-FILTERED (215,844 rows) bounds the per-row fan-out
// (answers/question x 7-name badges/user): 266 -> ~40 ms class.
// Multi-site (2 sites) -> id joins carry the build-side site_id and enforce
// site equality as residual filters.

constexpr auto query = "stack_Q14";

PreparedStatement stack::Query::prepareQ14(proteus::QueryShaper &morph) {
  morph.setQueryName(query);

  auto rel_site = morph.scan("site", {"site_id", "site_name"});
  auto rel_tag = morph.scan("tag", {"id", "site_id", "name"});
  auto rel_question = morph.scan("question", {"id", "site_id", "view_count"});
  auto rel_answer = morph.scan(
      "answer", {"question_id", "site_id", "owner_user_id"});
  auto rel_user =
      morph.scan("so_user", {"id", "site_id", "downvotes", "account_id"});
  auto rel_badge = morph.scan("badge", {"user_id", "site_id", "name"});
  auto rel = morph.scan("tag_question", {"question_id", "tag_id", "site_id"});

  return morph
      .parallel(
          rel,
          {rel_site, rel_tag, rel_question, rel_answer, rel_user, rel_badge},
          [&](RelBuilder probe, std::vector<RelBuilder> build) {
            auto site_d =
                build.at(0)
                    .unpack()
                    .filter([&](const auto &arg) -> expression_t {
                      return expressions::hint(
                          eq(arg["site_name"], "stackoverflow") |
                              eq(arg["site_name"], "superuser"),
                          expressions::Selectivity(2.0 / 170));
                    })
                    .project([&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["site_id"].as("site.csv", "s_site")};
                    });

            auto tag_d =
                build.at(1)
                    .unpack()
                    .filter([&](const auto &arg) -> expression_t {
                      return expressions::hint(
                          eq(arg["name"], "android-asynctask") |
                              eq(arg["name"], "asp.net-mvc-4") |
                              eq(arg["name"], "css3") |
                              eq(arg["name"], "entity-framework-4") |
                              eq(arg["name"], "inner-join") |
                              eq(arg["name"], "jquery") |
                              eq(arg["name"], "sublimetext3"),
                          expressions::Selectivity(7.0 / 100'000));
                    })
                    .project([&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["id"].as("tag.csv", "t_id"),
                              arg["site_id"].as("tag.csv", "t_site")};
                    });

            auto question_d =
                build.at(2)
                    .unpack()
                    .filter([&](const auto &arg) -> expression_t {
                      return expressions::hint(
                          ge(arg["view_count"], 10) &
                              le(arg["view_count"], 1000),
                          expressions::Selectivity(0.5));
                    })
                    .project([&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["id"].as("question.csv", "q_id"),
                              arg["site_id"].as("question.csv", "q_site")};
                    });

            auto answer_d =
                build.at(3).unpack().project(
                    [&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["question_id"].as("answer.csv", "a_q"),
                              arg["site_id"].as("answer.csv", "a_site"),
                              arg["owner_user_id"].as("answer.csv", "a_owner")};
                    });

            auto user_d =
                build.at(4)
                    .unpack()
                    .filter([&](const auto &arg) -> expression_t {
                      return expressions::hint(
                          ge(arg["downvotes"], expression_t{0}) &
                              le(arg["downvotes"], 10) &
                              ne(arg["account_id"], -1),
                          expressions::Selectivity(0.9));
                    })
                    .project([&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["id"].as("so_user.csv", "u_id"),
                              arg["site_id"].as("so_user.csv", "u_site")};
                    });

            // b.name IN (...) applied on the BUILD: 215,844 of 51M rows.
            auto badge_d =
                build.at(5)
                    .unpack()
                    .filter([&](const auto &arg) -> expression_t {
                      return expressions::hint(
                          eq(arg["name"], "Analytical") |
                              eq(arg["name"], "Beta") |
                              eq(arg["name"], "Deputy") |
                              eq(arg["name"], "Fanatic") |
                              eq(arg["name"], "Great Question") |
                              eq(arg["name"], "Not a Robot") |
                              eq(arg["name"], "Unsung Hero"),
                          expressions::Selectivity(0.005));
                    })
                    .project([&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["user_id"].as("badge.csv", "b_user"),
                              arg["site_id"].as("badge.csv", "b_site")};
                    });

            return probe.unpack()
                // s.site_id = tq1.site_id (2 sites after the name filter)
                .join(
                    site_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["s_site"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["site_id"];
                    },
                    4, 16)
                // t1.id = tq1.tag_id, site as residual
                .join(
                    tag_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["t_id"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["tag_id"];
                    },
                    12, 4096)
                .filter([&](const auto &arg) -> expression_t {
                  return expressions::hint(eq(arg["t_site"], arg["site_id"]),
                                           expressions::Selectivity(0.9));
                })
                // q1.id = tq1.question_id (view-range-filtered build)
                .join(
                    question_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["q_id"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["question_id"];
                    },
                    24, 10'000'000)
                .filter([&](const auto &arg) -> expression_t {
                  return expressions::hint(eq(arg["q_site"], arg["site_id"]),
                                           expressions::Selectivity(0.9));
                })
                // a1.question_id = q1.id (multi-match: answers per question)
                .join(
                    answer_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["a_q"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["q_id"];
                    },
                    23, 7'000'000)
                .filter([&](const auto &arg) -> expression_t {
                  return expressions::hint(eq(arg["a_site"], arg["site_id"]),
                                           expressions::Selectivity(0.9));
                })
                // u1.id = a1.owner_user_id, site as residual
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
                // b.user_id = u1.id (multi-match: this user's 7-name badges),
                // site as residual
                .join(
                    badge_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["b_user"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["u_id"];
                    },
                    19, 300'000)
                .filter([&](const auto &arg) -> expression_t {
                  return expressions::hint(eq(arg["b_site"], arg["site_id"]),
                                           expressions::Selectivity(0.9));
                })
                .reduce(
                    [&](const auto &arg) -> std::vector<expression_t> {
                      return {expression_t{1}.as("tmp", "cnt")};
                    },
                    {SUM});
          })
      .reduce(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["cnt"]};
          },
          {SUM})
      .print(pg{"pm-csv"})
      .prepare();
}
