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

constexpr auto query = "ssb1000_Q3_2";

#include "query.cpp.inc"

PreparedStatement Query::prepare32(bool memmv) {
  auto rel34584 =
      getBuilder<Tplugin>()
          .scan<Tplugin>(
              "inputs/ssbm1000/date.csv", {"d_datekey", "d_year"},
              getCatalog())  // (table=[[SSB, ssbm_date]], fields=[[0, 4]],
                             // traits=[Pelago.[].packed.X86_64.homSingle.hetSingle])
          .membrdcst(dop, true, true)
          .router(
              [&](const auto &arg) -> std::optional<expression_t> {
                return arg["__broadcastTarget"];
              },
              dop, 1, RoutingPolicy::HASH_BASED, dev,
              aff_parallel())  // (trait=[Pelago.[].packed.X86_64.homBrdcst.hetSingle])
          .to_gpu()  // (trait=[Pelago.[].packed.NVPTX.homBrdcst.hetSingle])
          .unpack()  // (trait=[Pelago.[].unpckd.NVPTX.homBrdcst.hetSingle])
          .filter([&](const auto &arg) -> expression_t {
            return (ge(arg["$1"], 1992) & le(arg["$1"], 1997));
          })  // (condition=[AND(>=($1, 1992), <=($1, 1997))],
              // trait=[Pelago.[].unpckd.NVPTX.homBrdcst.hetSingle],
              // isS=[false])
      ;
  auto rel34589 =
      getBuilder<Tplugin>()
          .scan<Tplugin>(
              "inputs/ssbm1000/customer.csv",
              {"c_custkey", "c_city", "c_nation"},
              getCatalog())  // (table=[[SSB, ssbm_customer]], fields=[[0, 3,
                             // 4]],
                             // traits=[Pelago.[].packed.X86_64.homSingle.hetSingle])
          .membrdcst(dop, true, true)
          .router(
              [&](const auto &arg) -> std::optional<expression_t> {
                return arg["__broadcastTarget"];
              },
              dop, 1, RoutingPolicy::HASH_BASED, dev,
              aff_parallel())  // (trait=[Pelago.[].packed.X86_64.homBrdcst.hetSingle])
          .to_gpu()  // (trait=[Pelago.[].packed.NVPTX.homBrdcst.hetSingle])
          .unpack()  // (trait=[Pelago.[].unpckd.NVPTX.homBrdcst.hetSingle])
          .filter([&](const auto &arg) -> expression_t {
            return eq(arg["$2"], "UNITED STATES");
          })  // (condition=[=($2, 'UNITED STATES')],
              // trait=[Pelago.[].unpckd.NVPTX.homBrdcst.hetSingle],
              // isS=[false])
          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {(arg["$0"]).as("PelagoProject#34589", "$0"),
                    (arg["$1"]).as("PelagoProject#34589", "c_city")};
          })  // (c_custkey=[$0], c_city=[$1],
              // trait=[Pelago.[].unpckd.NVPTX.homBrdcst.hetSingle])
      ;
  auto rel34594 =
      getBuilder<Tplugin>()
          .scan<Tplugin>(
              "inputs/ssbm1000/supplier.csv",
              {"s_suppkey", "s_city", "s_nation"},
              getCatalog())  // (table=[[SSB, ssbm_supplier]], fields=[[0, 3,
                             // 4]],
                             // traits=[Pelago.[].packed.X86_64.homSingle.hetSingle])
          .membrdcst(dop, true, true)
          .router(
              [&](const auto &arg) -> std::optional<expression_t> {
                return arg["__broadcastTarget"];
              },
              dop, 1, RoutingPolicy::HASH_BASED, dev,
              aff_parallel())  // (trait=[Pelago.[].packed.X86_64.homBrdcst.hetSingle])
          .to_gpu()  // (trait=[Pelago.[].packed.NVPTX.homBrdcst.hetSingle])
          .unpack()  // (trait=[Pelago.[].unpckd.NVPTX.homBrdcst.hetSingle])
          .filter([&](const auto &arg) -> expression_t {
            return eq(arg["$2"], "UNITED STATES");
          })  // (condition=[=($2, 'UNITED STATES')],
              // trait=[Pelago.[].unpckd.NVPTX.homBrdcst.hetSingle],
              // isS=[false])
          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {(arg["$0"]).as("PelagoProject#34594", "$0"),
                    (arg["$1"]).as("PelagoProject#34594", "s_city")};
          })  // (s_suppkey=[$0], s_city=[$1],
              // trait=[Pelago.[].unpckd.NVPTX.homBrdcst.hetSingle])
      ;
  auto rel =
      getBuilder<Tplugin>()
          .scan<Tplugin>(
              "inputs/ssbm1000/lineorder.csv",
              {"lo_custkey", "lo_suppkey", "lo_orderdate", "lo_revenue"},
              getCatalog())  // (table=[[SSB, ssbm_lineorder]], fields=[[2, 4,
                             // 5, 12]],
                             // traits=[Pelago.[].packed.X86_64.homSingle.hetSingle])
          .router(
              dop, 1, RoutingPolicy::LOCAL, dev,
              aff_parallel())  // (trait=[Pelago.[].packed.X86_64.homRandom.hetSingle])
      ;

  if (memmv) rel = rel.memmove(8, dev == DeviceType::CPU);

  rel =
      rel.to_gpu()   // (trait=[Pelago.[].packed.NVPTX.homRandom.hetSingle])
          .unpack()  // (trait=[Pelago.[].unpckd.NVPTX.homRandom.hetSingle])
          .join(
              rel34594,
              [&](const auto &build_arg) -> expression_t {
                return build_arg["$0"];
              },
              [&](const auto &probe_arg) -> expression_t {
                return probe_arg["lo_suppkey"];
              },
              18,
              131072)  // (condition=[=($3, $0)], joinType=[inner],
                       // rowcnt=[8192000.0], maxrow=[200000.0],
                       // maxEst=[200000.0], h_bits=[25],
                       // build=[RecordType(INTEGER s_suppkey, VARCHAR s_city)],
                       // lcount=[2.7216799017896134E8],
                       // rcount=[3.0721953024E11], buildcountrow=[8192000.0],
                       // probecountrow=[3.0721953024E11])
          .join(
              rel34589,
              [&](const auto &build_arg) -> expression_t {
                return build_arg["$0"];
              },
              [&](const auto &probe_arg) -> expression_t {
                return probe_arg["lo_custkey"];
              },
              21,
              2097152)  // (condition=[=($2, $0)], joinType=[inner],
                        // rowcnt=[1.2288E8], maxrow=[3000000.0],
                        // maxEst=[3000000.0], h_bits=[28],
                        // build=[RecordType(INTEGER c_custkey, VARCHAR
                        // c_city)], lcount=[4.7480502701073E9],
                        // rcount=[1.22887812096E10], buildcountrow=[1.2288E8],
                        // probecountrow=[1.22887812096E10])
          .join(
              rel34584,
              [&](const auto &build_arg) -> expression_t {
                return build_arg["$0"];
              },
              [&](const auto &probe_arg) -> expression_t {
                return probe_arg["lo_orderdate"];
              },
              14,
              2556)  // (condition=[=($2, $0)], joinType=[inner],
                     // rowcnt=[654336.0], maxrow=[2556.0], maxEst=[2556.0],
                     // h_bits=[22], build=[RecordType(INTEGER d_datekey,
                     // INTEGER d_year)], lcount=[1.8432021459974352E7],
                     // rcount=[4.91551248384E8], buildcountrow=[654336.0],
                     // probecountrow=[4.91551248384E8])
          .groupby(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["c_city"].as("PelagoAggregate#34604", "$0"),
                        arg["s_city"].as("PelagoAggregate#34604", "$1"),
                        arg["d_year"].as("PelagoAggregate#34604", "$2")};
              },
              [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                return {GpuAggrMatExpr{
                    (arg["lo_revenue"]).as("PelagoAggregate#34604", "$3"), 1, 0,
                    SUM}};
              },
              10,
              131072)  // (group=[{0, 1, 2}], lo_revenue=[SUM($3)],
                       // trait=[Pelago.[].unpckd.NVPTX.homRandom.hetSingle])
          .pack()      // (trait=[Pelago.[].packed.NVPTX.homRandom.hetSingle],
                       // intrait=[Pelago.[].unpckd.NVPTX.homRandom.hetSingle],
                       // inputRows=[1.22887812096E7], cost=[{19.2192
                       // rows, 19.200000000000003 cpu, 0.0 io}])
          .to_cpu()    // (trait=[Pelago.[].packed.X86_64.homRandom.hetSingle])
          .router(
              DegreeOfParallelism{1}, 128, RoutingPolicy::RANDOM,
              DeviceType::CPU,
              aff_reduce())  // (trait=[Pelago.[].packed.X86_64.homSingle.hetSingle])
          .memmove(8, true)
          .unpack()  // (trait=[Pelago.[].unpckd.NVPTX.homSingle.hetSingle])
          .groupby(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["$0"].as("PelagoAggregate#34610", "$0"),
                        arg["$1"].as("PelagoAggregate#34610", "$1"),
                        arg["$2"].as("PelagoAggregate#34610", "$2")};
              },
              [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                return {GpuAggrMatExpr{
                    (arg["$3"]).as("PelagoAggregate#34610", "$3"), 1, 0, SUM}};
              },
              10,
              131072)  // (group=[{0, 1, 2}], lo_revenue=[SUM($3)],
                       // trait=[Pelago.[].unpckd.NVPTX.homSingle.hetSingle])
          .sort(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["$0"], arg["$1"], arg["$2"], arg["$3"]};
              },
              {direction::NONE, direction::NONE, direction::ASC,
               direction::DESC})  // (sort0=[$2], sort1=[$3], dir0=[ASC],
                                  // dir1=[DESC], trait=[Pelago.[2, 3
                                  // DESC].unpckd.X86_64.homSingle.hetSingle])
          .print(
              [&](const auto &arg,
                  std::string outrel) -> std::vector<expression_t> {
                return {arg["$0"].as(outrel, "c_city"),
                        arg["$1"].as(outrel, "s_city"),
                        arg["$2"].as(outrel, "d_year"),
                        arg["$3"].as(outrel, "lo_revenue")};
              },
              std::string{query} +
                  (memmv ? "mv" : "nmv"))  // (trait=[ENUMERABLE.[2, 3
      // DESC].unpckd.X86_64.homSingle.hetSingle])
      ;
  return rel.prepare();
}
