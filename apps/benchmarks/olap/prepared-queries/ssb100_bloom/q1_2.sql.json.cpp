/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2019
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

// AUTOGENERATED FILE. DO NOT EDIT.

constexpr auto query = "ssb100_Q1_2";

#include "query.cpp.inc"

PreparedStatement Query::prepare12(bool memmv, SLAZY conf, size_t bloomSize) {
  auto rel2337 =
      getBuilder<Tplugin>()
          .scan(
              "inputs/ssbm100/date.csv", {"d_datekey", "d_yearmonthnum"},
              getCatalog(),
              pg{Tplugin::type})  // (table=[[SSB, ssbm_date]], fields=[[0, 4]],
          // traits=[Pelago.[].packed.X86_64.homSingle.hetSingle])
          //          .to_gpu()
          .membrdcst(dop, true, true)
          .router(
              [&](const auto &arg) -> expression_t {
                return arg["__broadcastTarget"];
              },
              dop, 128, dev,
              aff_parallel())  // (trait=[Pelago.[].packed.X86_64.homBrdcst.hetSingle])
          .unpack()  // (trait=[Pelago.[].unpckd.NVPTX.homBrdcst.hetSingle])
          .filter([&](const auto &arg) -> expression_t {
            return eq(arg["d_yearmonthnum"], 199401);
          })
          .bloomfilter_build(
              [&](const auto &arg) -> expression_t { return arg["d_datekey"]; },
              bloomSize, 0)
          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {arg["d_datekey"]};
          })
          .pack()
          .to_gpu()
          .unpack();

  auto rel =
      getBuilder<Tplugin>()
          .scan("inputs/ssbm100/lineorder.csv",
                {"lo_orderdate", "lo_quantity", "lo_extendedprice",
                 "lo_discount"},
                getCatalog(),
                pg{Tplugin::type})  // (table=[[SSB, ssbm_lineorder]],
                                    // fields=[[5, 8, 9, 11]],
          // traits=[Pelago.[].packed.X86_64.homSingle.hetSingle])
          .router(DegreeOfParallelism{48}, 8, RoutingPolicy::LOCAL,
                  DeviceType::CPU)
          .unpack()  // (trait=[Pelago.[].unpckd.NVPTX.homRandom.hetSingle])
          .bloomfilter_probe(
              [&](const auto &arg) { return arg["lo_orderdate"]; }, bloomSize,
              0);

  if (conf == BLOOM_CPUFILTER_PROJECT || conf == BLOOM_CPUFILTER_NOPROJECT) {
    rel = rel.filter([&](const auto &arg) -> expression_t {
      return (ge(arg["lo_discount"], 4) & le(arg["lo_discount"], 6) &
              ge(arg["lo_quantity"], 26) & le(arg["lo_quantity"], 35));
    })  // (condition=[AND(>=($3, 1), <=($3, 3), <($1, 25))],
        ;
  }

  if (conf == BLOOM_CPUFILTER_PROJECT) {
    // trait=[Pelago.[].unpckd.NVPTX.homRandom.hetSingle],
    // isS=[false])
    rel = rel.project([&](const auto &arg) -> std::vector<expression_t> {
      return {arg["lo_orderdate"], arg["lo_extendedprice"], arg["lo_discount"]};
    });
  }

  rel =
      rel.pack()
          //          .router(DegreeOfParallelism{1}, 8, RoutingPolicy::RANDOM,
          //                  DeviceType::CPU)
          .router(DegreeOfParallelism{2}, 8, RoutingPolicy::LOCAL,
                  DeviceType::GPU);

  if (memmv) rel = rel.memmove(8, dev);

  rel = rel.to_gpu().unpack();

  if (conf == BLOOM_GPUFILTER_NOPROJECT) {
    rel = rel.filter([&](const auto &arg) -> expression_t {
      return (ge(arg["lo_discount"], 4) & le(arg["lo_discount"], 6) &
              ge(arg["lo_quantity"], 26) & le(arg["lo_quantity"], 35));
    })  // (condition=[AND(>=($3, 1), <=($3, 3), <($1, 25))],
        ;
  }

  rel = rel.join(
               rel2337,
               [&](const auto &build_arg) -> expression_t {
                 return build_arg["d_datekey"];
               },
               [&](const auto &probe_arg) -> expression_t {
                 return probe_arg["lo_orderdate"];
               },
               10, 512)
            .reduce(
                [&](const auto &arg) -> std::vector<expression_t> {
                  return {(arg["lo_extendedprice"] * arg["lo_discount"])
                              .as("PelagoAggregate#2345", "revenue")};
                },
                {SUM})
            .to_cpu()
            .router(DegreeOfParallelism{1}, 8, RoutingPolicy::RANDOM,
                    DeviceType::CPU)
            .reduce(
                [&](const auto &arg) -> std::vector<expression_t> {
                  return {arg["revenue"]};
                },
                {SUM})
            .print(pg{"pm-csv"});
  return rel.prepare();
}

PreparedStatement Query::prepare12_b(bool memmv, size_t bloomSize) {
  auto rel =
      getBuilder<Tplugin>()
          .scan("inputs/ssbm100/lineorder.csv",
                {"lo_orderdate", "lo_quantity", "lo_extendedprice",
                 "lo_discount"},
                getCatalog(),
                pg{Tplugin::type})  // (table=[[SSB, ssbm_lineorder]],
                                    // fields=[[5, 8, 9, 11]],
          // traits=[Pelago.[].packed.X86_64.homSingle.hetSingle])
          .router(DegreeOfParallelism{48}, 8, RoutingPolicy::LOCAL,
                  DeviceType::CPU)
          .unpack()  // (trait=[Pelago.[].unpckd.NVPTX.homRandom.hetSingle])
          .bloomfilter_probe(
              [&](const auto &arg) { return arg["lo_orderdate"]; }, bloomSize,
              0)
          .filter([&](const auto &arg) -> expression_t {
            return (ge(arg["lo_discount"], 4) & le(arg["lo_discount"], 6) &
                    ge(arg["lo_quantity"], 26) & le(arg["lo_quantity"], 35));
          })
          .reduce(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {expression_t{1}.as("tmp", "cnt")};
              },
              {SUM})
          .router(
              DegreeOfParallelism{1}, 128, RoutingPolicy::RANDOM,
              DeviceType::CPU,
              aff_reduce())  // (trait=[Pelago.[].packed.X86_64.homSingle.hetSingle])
          .reduce(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["cnt"]};
              },
              {SUM})
          .print(pg{"pm-csv"});

  return rel.prepare();
}
