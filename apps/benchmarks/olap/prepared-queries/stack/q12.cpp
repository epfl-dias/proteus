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

// Stack SO-Q12 (Bao template q12) -- 6 tables, GROUP BY on a dict column.
//   SELECT t1.name, count(*)
//   FROM site s, so_user u1, question q1, answer a1, tag t1, tag_question tq1
//   WHERE q1.owner_user_id = u1.id AND a1.question_id = q1.id
//     AND a1.owner_user_id = u1.id            -- users answering their own q
//     AND s.site_id = {q1,a1,u1,tq1,t1}.site_id
//     AND q1.id = tq1.question_id AND t1.id = tq1.tag_id
//     AND s.site_name IN ('android','gamedev','judaism','sharepoint')
//     AND t1.name IN ('java','opengl','powershell','sharepoint-enterprise',
//                     'sharepoint-foundation','sharepoint-online',
//                     'sources-mekorot','web-part','xna')
//     AND q1.view_count >= 10 AND q1.view_count <= 1000
//     AND u1.reputation >= 10 AND u1.reputation <= 100000
//   GROUP BY t1.name
//
// Multi-site instance (4 sites), so id joins can collide across sites; every
// id join carries the build side's site_id and enforces the site equality as
// a residual filter (same technique as q01.cpp -- correct for any number of
// sites, just more spurious hash matches to discard).
// Probe = tag_question; group-by key is the carried dstring t_name (<=9
// groups; the dict is lexicographically sorted, so ordering by the code
// orders by name).

constexpr auto query = "stack_Q12";

PreparedStatement stack::Query::prepareQ12(proteus::QueryShaper &morph) {
  morph.setQueryName(query);

  auto rel_site = morph.scan("site", {"site_id", "site_name"});
  auto rel_tag = morph.scan("tag", {"id", "site_id", "name"});
  auto rel_question =
      morph.scan("question", {"id", "site_id", "owner_user_id", "view_count"});
  auto rel_user = morph.scan("so_user", {"id", "site_id", "reputation"});
  auto rel_answer =
      morph.scan("answer", {"question_id", "site_id", "owner_user_id"});
  auto rel = morph.scan("tag_question", {"question_id", "tag_id", "site_id"});

  return morph
      .parallel(
          rel, {rel_site, rel_tag, rel_question, rel_user, rel_answer},
          [&](RelBuilder probe, std::vector<RelBuilder> build) {
            auto site_d =
                build.at(0)
                    .unpack()
                    .filter([&](const auto &arg) -> expression_t {
                      return expressions::hint(
                          eq(arg["site_name"], "android") |
                              eq(arg["site_name"], "gamedev") |
                              eq(arg["site_name"], "judaism") |
                              eq(arg["site_name"], "sharepoint"),
                          expressions::Selectivity(4.0 / 170));
                    })
                    .project([&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["site_id"].as("site.csv", "s_site")};
                    });

            auto tag_d =
                build.at(1)
                    .unpack()
                    .filter([&](const auto &arg) -> expression_t {
                      return expressions::hint(
                          eq(arg["name"], "java") | eq(arg["name"], "opengl") |
                              eq(arg["name"], "powershell") |
                              eq(arg["name"], "sharepoint-enterprise") |
                              eq(arg["name"], "sharepoint-foundation") |
                              eq(arg["name"], "sharepoint-online") |
                              eq(arg["name"], "sources-mekorot") |
                              eq(arg["name"], "web-part") |
                              eq(arg["name"], "xna"),
                          expressions::Selectivity(9.0 / 100'000));
                    })
                    .project([&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["id"].as("tag.csv", "t_id"),
                              arg["site_id"].as("tag.csv", "t_site"),
                              arg["name"].as("tag.csv", "t_name")};
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
                              arg["site_id"].as("question.csv", "q_site"),
                              arg["owner_user_id"].as("question.csv",
                                                      "q_owner")};
                    });

            auto user_d =
                build.at(3)
                    .unpack()
                    .filter([&](const auto &arg) -> expression_t {
                      return expressions::hint(
                          ge(arg["reputation"], 10) &
                              le(arg["reputation"], 100'000),
                          expressions::Selectivity(0.5));
                    })
                    .project([&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["id"].as("so_user.csv", "u_id"),
                              arg["site_id"].as("so_user.csv", "u_site")};
                    });

            auto answer_d =
                build.at(4).unpack().project(
                    [&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["question_id"].as("answer.csv", "a_q"),
                              arg["site_id"].as("answer.csv", "a_site"),
                              arg["owner_user_id"].as("answer.csv", "a_owner")};
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
                    12, 4096)
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
                    24, 10'000'000)
                .filter([&](const auto &arg) -> expression_t {
                  return expressions::hint(eq(arg["q_site"], arg["site_id"]),
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
                    25, 12'000'000)
                .filter([&](const auto &arg) -> expression_t {
                  return expressions::hint(eq(arg["u_site"], arg["site_id"]),
                                           expressions::Selectivity(0.9));
                })
                // a1.question_id = q1.id (multi-match: all answers of the
                // question), then the self-answer condition
                // a1.owner_user_id = u1.id and the site equality as residuals.
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
                          eq(arg["a_owner"], arg["u_id"]),
                      expressions::Selectivity(0.2));
                })
                .groupby(
                    [&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["t_name"]};
                    },
                    [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                      return {GpuAggrMatExpr{
                          expression_t{1}.as("tmp", "cnt"), 1, 0, SUM}};
                    },
                    10, 16)
                .pack();
          })
      .unpack()
      .groupby(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["t_name"]};
          },
          [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
            return {GpuAggrMatExpr{arg["cnt"], 1, 0, SUM}};
          },
          10, 16)
      .sort(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["t_name"], arg["cnt"]};
          },
          {direction::ASC, direction::NONE})
      .print(pg{"pm-csv"})
      .prepare();
}
