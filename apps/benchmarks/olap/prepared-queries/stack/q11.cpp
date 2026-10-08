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

// Stack SO-Q11 (Bao template q11) -- Q1's star + IN lists + int range.
//   SELECT COUNT(*) FROM tag t, site s, question q, tag_question tq
//   WHERE t.site_id = s.site_id AND q.site_id = s.site_id
//     AND tq.site_id = s.site_id
//     AND tq.question_id = q.id AND tq.tag_id = t.id
//     AND s.site_name IN ('stackoverflow')
//     AND t.name IN ('api','audio','events','file-io','forms',
//                    'functional-programming','java-ee','serialization',
//                    'sqlite','ubuntu','unicode')
//     AND q.favorite_count >= 5 AND q.favorite_count <= 5000
//
// Same shape as SO-Q1 (see q01.cpp): probe = tag_question; site join first,
// then tag (IN = OR of dict equalities, as TPC-H Q19), then question (now
// filtered by the favorite_count range; exported NULLs are -1 and fail the
// >= 5 bound, matching SQL NULL semantics). 'stackoverflow' is by far the
// largest site, so the probe side stays big after the site join.

constexpr auto query = "stack_Q11";

PreparedStatement stack::Query::prepareQ11(proteus::QueryShaper &morph) {
  morph.setQueryName(query);

  auto rel_site = morph.scan("site", {"site_id", "site_name"});
  auto rel_tag = morph.scan("tag", {"id", "site_id", "name"});
  auto rel_question = morph.scan("question", {"id", "site_id", "favorite_count"});
  auto rel = morph.scan("tag_question", {"question_id", "tag_id", "site_id"});

  return morph
      .parallel(
          rel, {rel_site, rel_tag, rel_question},
          [&](RelBuilder probe, std::vector<RelBuilder> build) {
            auto site_d =
                build.at(0)
                    .unpack()
                    .filter([&](const auto &arg) -> expression_t {
                      return expressions::hint(
                          eq(arg["site_name"], "stackoverflow"),
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
                          eq(arg["name"], "api") | eq(arg["name"], "audio") |
                              eq(arg["name"], "events") |
                              eq(arg["name"], "file-io") |
                              eq(arg["name"], "forms") |
                              eq(arg["name"], "functional-programming") |
                              eq(arg["name"], "java-ee") |
                              eq(arg["name"], "serialization") |
                              eq(arg["name"], "sqlite") |
                              eq(arg["name"], "ubuntu") |
                              eq(arg["name"], "unicode"),
                          expressions::Selectivity(11.0 / 100'000));
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
                          ge(arg["favorite_count"], 5) &
                              le(arg["favorite_count"], 5000),
                          expressions::Selectivity(0.05));
                    })
                    .project([&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["id"].as("question.csv", "q_id"),
                              arg["site_id"].as("question.csv", "q_site")};
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
                    22, 4'000'000)
                .filter([&](const auto &arg) -> expression_t {
                  return expressions::hint(eq(arg["q_site"], arg["site_id"]),
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
