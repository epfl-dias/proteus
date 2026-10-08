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

// Stack SO-Q1 (Bao template q1) -- 4-table star, COUNT(*).
//   select count(*) from tag, site, question, tag_question
//   where site.site_name = 'scifi'
//     and tag.name = 'steins-gate'
//     and tag.site_id = site.site_id
//     and question.site_id = site.site_id
//     and tag_question.site_id = site.site_id
//     and tag_question.question_id = question.id
//     and tag_question.tag_id = tag.id
//
// Stack ids are only unique per (site_id, id), so every id join carries a
// site_id equality. Shape used here (all four joins preserved):
//   probe = tag_question (the bridge, the largest scanned relation here);
//   builds = site (filtered to 'scifi': 1 row), tag (filtered to
//   'steins-gate': one row per site that has the tag), question (unfiltered:
//   id + site_id of every question, the big build).
// Join order on the probe stream: site first (cuts tag_question to the one
// site early), then tag, then question. The site_id equalities of the tag and
// question joins are enforced as residual filters on the carried build-side
// site_id -- same technique as the TPC-H Q5 nation cycle.
//
// Attribute naming: tag/question/site all expose `id`/`site_id`, so every
// build projection renames its columns (t_id/t_site, q_id/q_site, s_site) to
// keep the probe stream unambiguous; tag_question keeps its original names.

constexpr auto query = "stack_Q1";

PreparedStatement stack::Query::prepareQ1(proteus::QueryShaper &morph) {
  morph.setQueryName(query);

  auto rel_site = morph.scan("site", {"site_id", "site_name"});
  auto rel_tag = morph.scan("tag", {"id", "site_id", "name"});
  auto rel_question = morph.scan("question", {"id", "site_id"});
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
                          eq(arg["site_name"], "scifi"),
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
                          eq(arg["name"], "steins-gate"),
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
                              arg["site_id"].as("question.csv", "q_site")};
                    });

            return probe.unpack()
                // tag_question.site_id = site.site_id (site pre-filtered)
                .join(
                    site_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["s_site"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["site_id"];
                    },
                    4, 16)
                // tag_question.tag_id = tag.id (tag pre-filtered to the name)
                .join(
                    tag_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["t_id"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["tag_id"];
                    },
                    9, 512)
                // ... and tag.site_id = site.site_id (== tag_question.site_id)
                .filter([&](const auto &arg) -> expression_t {
                  return expressions::hint(eq(arg["t_site"], arg["site_id"]),
                                           expressions::Selectivity(0.9));
                })
                // tag_question.question_id = question.id
                .join(
                    question_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["q_id"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["question_id"];
                    },
                    25, 19'000'000)
                // ... and question.site_id = site.site_id
                .filter([&](const auto &arg) -> expression_t {
                  return expressions::hint(eq(arg["q_site"], arg["site_id"]),
                                           expressions::Selectivity(0.9))
                      ;
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
