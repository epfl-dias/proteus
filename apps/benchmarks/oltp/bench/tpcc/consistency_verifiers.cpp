/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2020
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

#include <chrono>
#include <functional>
#include <iostream>
#include <olap/operators/relbuilder-factory.hpp>
#include <olap/plan/catalog-parser.hpp>
#include <olap/plan/prepared-statement.hpp>
#include <olap/routing/degree-of-parallelism.hpp>
#include <oltp/common/utils.hpp>
#include <oltp/storage/layout/column_store.hpp>
#include <platform/threadpool/thread.hpp>
#include <string>

#include "tpcc/tpcc_64.hpp"

namespace bench {

PreparedStatement TPCC::consistency_check_1_query_builder(bool return_aggregate,
                                                          string olap_plugin) {
  // SQL: (EXPECTED 0 ROWS)
  //     SELECT sum(w_ytd), sum(d_ytd)
  //     FROM tpcc_warehouse, tpcc_district
  //     WHERE tpcc_warehouse.w_id = tpcc_district.d_w_id
  //     GROUP BY tpcc_warehouse.w_id
  //     HAVING sum(w_ytd) != sum(d_ytd)

  // auto dop = DegreeOfParallelism {topology::getInstance().getCoreCount()};

  RelBuilderFactory ctx{"tpcc_consistency_check_1"};
  CatalogParser &catalog = CatalogParser::getInstance();
  auto wh_scan =
      ctx.getBuilder()
          .scan("tpcc_warehouse<" + olap_plugin + ">", {"w_id", "w_ytd"},
                CatalogParser::getInstance(), pg{olap_plugin})
          .router(32, RoutingPolicy::LOCAL, DeviceType::CPU)
          .unpack();

  auto tpcc_check_1 =
      ctx.getBuilder()
          .scan("tpcc_district<" + olap_plugin + ">", {"d_w_id", "d_ytd"},
                CatalogParser::getInstance(), pg{olap_plugin})
          .router(32, RoutingPolicy::LOCAL, DeviceType::CPU)
          .unpack()
          .join(
              wh_scan,
              [&](const auto &build_arg) -> expression_t {
                return expressions::RecordConstruction{
                    build_arg["w_id"].as("PelagoJoin#2030", "bk_0")}
                    .as("PelagoJoin#2030", "bk");
              },
              [&](const auto &probe_arg) -> expression_t {
                return expressions::RecordConstruction{
                    probe_arg["d_w_id"].as("PelagoJoin#2030", "pk_0")}
                    .as("PelagoJoin#2030", "pk");
              },
              log2(this->num_warehouse) + 1, this->num_warehouse + 10)
          .groupby(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {(arg["w_id"]).as("tmpRelation", "w_id")};
              },
              [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                return {GpuAggrMatExpr{
                            (arg["w_ytd"]).as("tmpRelation", "warehouse_ytd"),
                            1, 0, SUM},
                        GpuAggrMatExpr{
                            (arg["d_ytd"]).as("tmpRelation", "district_ytd"), 2,
                            0, SUM}};
              },
              log2(this->num_warehouse) + 1, this->num_warehouse + 10)
          // Cast to floatType as pack-requires same width of data-types.
          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {(arg["w_id"].template as<FloatType>())
                        .as("tmpRelation#2", "w_id"),
                    arg["warehouse_ytd"].as("tmpRelation#2", "warehouse_ytd"),
                    arg["district_ytd"].as("tmpRelation#2", "district_ytd")

            };
          })
          //.pack()
          .router(DegreeOfParallelism{1}, 128, RoutingPolicy::RANDOM,
                  DeviceType::CPU)
          //.unpack()
          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {(arg["w_id"].template as<Int64Type>())
                        .as("tmpRelation#3", "w_id"),
                    arg["warehouse_ytd"].as("tmpRelation#3", "warehouse_ytd"),
                    arg["district_ytd"].as("tmpRelation#3", "district_ytd")

            };
          })
          .groupby(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {(arg["w_id"])};
              },
              [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                return {GpuAggrMatExpr{(arg["warehouse_ytd"]), 1, 0, SUM},
                        GpuAggrMatExpr{(arg["district_ytd"]), 2, 0, SUM}};
              },
              log2(this->num_warehouse) + 1, this->num_warehouse)
          // Cast as Int64 to avoid precision mis-matches.
          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {(arg["w_id"])
                        .template as<Int64Type>()
                        .as("PelagoProject#13144", "w_id"),
                    (arg["warehouse_ytd"].template as<Int64Type>())
                        .as("PelagoProject#13144", "warehouse_ytd"),
                    (arg["district_ytd"].template as<Int64Type>())
                        .as("PelagoProject#13144", "district_ytd")

            };
          })
          .filter([&](const auto &arg) -> expression_t {
            return ne(arg["warehouse_ytd"], arg["district_ytd"]);
          });

  if (return_aggregate) {
    tpcc_check_1 = tpcc_check_1.reduce(
        [&](const auto &arg) -> std::vector<expression_t> {
          return {(expression_t{INT32_C(1)})
                      .as("PelagoProject#6", "count_inconsistent_rec")};
        },
        {SUM});
  }

  tpcc_check_1 = tpcc_check_1.print(pg{"pm-csv"});

  return tpcc_check_1.prepare();
}

bool TPCC::consistency_check_1(bool print_inconsistent_rows) {
  // Check-1
  // Entries in the WAREHOUSE and DISTRICT tables must satisfy the relationship:
  // W_YTD = sum(D_YTD)

  // SQL: (EXPECTED 0 ROWS)
  //     SELECT sum(w_ytd), sum(d_ytd)
  //     FROM tpcc_warehouse, tpcc_district
  //     WHERE tpcc_warehouse.w_id = tpcc_district.d_w_id
  //     GROUP BY tpcc_warehouse.w_id
  //     HAVING sum(w_ytd) != sum(d_ytd)

  LOG(INFO) << "##########\tConsistency Check - 1";
  bool db_consistent = true;

  std::ostringstream check1_result;
  check1_result << consistency_check_1_query_builder().execute();

  std::string n_orders_str = check1_result.str();
  int n_failed_rows = stoi(n_orders_str);

  if (n_failed_rows) {
    LOG(INFO) << "Consistency Check # 1 failed."
              << "\n\t# of inconsistent rows: " << n_failed_rows;
    db_consistent = false;
    if (print_inconsistent_rows) {
      LOG(INFO) << consistency_check_1_query_builder(false).execute();
    }
  }
  LOG(INFO) << "##########\tConsistency Check - 1 Completed: " << db_consistent;

  return db_consistent;
}

std::vector<PreparedStatement> TPCC::consistency_check_2_query_builder(
    bool return_aggregate, string olap_plugin) {
  RelBuilderFactory ctx_one{"tpcc_consistency_check_2_1"};
  RelBuilderFactory ctx_two{"tpcc_consistency_check_2_2"};
  RelBuilderFactory ctx_three{"tpcc_consistency_check_2_3"};
  CatalogParser &catalog = CatalogParser::getInstance();
  PreparedStatement o_max =
      ctx_one.getBuilder()
          .scan("tpcc_order<block-remote>", {"o_w_id", "o_d_id", "o_id"},
                CatalogParser::getInstance(), pg{"block-remote"})
          .router(DegreeOfParallelism{24}, 32, RoutingPolicy::LOCAL,
                  DeviceType::CPU)
          .unpack()
          .groupby(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["o_w_id"], arg["o_d_id"]};
              },
              [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                return {
                    GpuAggrMatExpr{(arg["o_id"]), 1, 0, MAX},
                };
              },
              log2(this->num_warehouse * TPCC_NDIST_PER_WH) + 1,
              log2(this->num_warehouse * TPCC_NDIST_PER_WH) + (1024 * 1024))
          //.pack()
          .router(DegreeOfParallelism{1}, 128, RoutingPolicy::RANDOM,
                  DeviceType::CPU)
          //.unpack()
          .groupby(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["o_w_id"], arg["o_d_id"]};
              },
              [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                return {
                    GpuAggrMatExpr{(arg["o_id"]), 1, 0, MAX},
                };
              },
              log2(this->num_warehouse * TPCC_NDIST_PER_WH) + 1,
              log2(this->num_warehouse * TPCC_NDIST_PER_WH) + (1024 * 1024))
          .sort(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["o_w_id"], arg["o_d_id"], arg["o_id"]};
              },
              {direction::ASC, direction::ASC, direction::NONE})
          .print(pg{"pm-csv"})
          .prepare();

  PreparedStatement no_max =
      ctx_two.getBuilder()
          .scan("tpcc_neworder<block-remote>",
                {"no_w_id", "no_d_id", "no_o_id"}, CatalogParser::getInstance(),
                pg{"block-remote"})
          .router(DegreeOfParallelism{24}, 32, RoutingPolicy::LOCAL,
                  DeviceType::CPU)
          .unpack()
          .groupby(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["no_w_id"], arg["no_d_id"]};
              },
              [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                return {
                    GpuAggrMatExpr{(arg["no_o_id"]), 1, 0, MAX},
                };
              },
              log2(this->num_warehouse * TPCC_NDIST_PER_WH) + 1,
              log2(this->num_warehouse * TPCC_NDIST_PER_WH) + (1024 * 1024))
          //.pack()
          .router(DegreeOfParallelism{1}, 128, RoutingPolicy::RANDOM,
                  DeviceType::CPU)
          //.unpack()
          .groupby(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["no_w_id"], arg["no_d_id"]};
              },
              [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                return {
                    GpuAggrMatExpr{(arg["no_o_id"]), 1, 0, MAX},
                };
              },
              log2(this->num_warehouse * TPCC_NDIST_PER_WH) + 1,
              log2(this->num_warehouse * TPCC_NDIST_PER_WH) + (1024 * 1024))
          .sort(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["no_w_id"], arg["no_d_id"], arg["no_o_id"]};
              },
              {direction::ASC, direction::ASC, direction::NONE})
          .print(pg{"pm-csv"})
          .prepare();

  PreparedStatement d_next_oid =
      ctx_three.getBuilder()
          .scan("tpcc_district<block-remote>",
                {"d_w_id", "d_id", "d_next_o_id"}, CatalogParser::getInstance(),
                pg{"block-remote"})
          .unpack()
          .sort(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["d_w_id"], arg["d_id"], arg["d_next_o_id"]};
              },
              {direction::ASC, direction::ASC, direction::NONE})
          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {(arg["d_w_id"]).as("PelagoProject#13144", "d_w_id"),
                    (arg["d_id"]).as("PelagoProject#13144", "d_id"),
                    (arg["d_next_o_id"] - (INT32_C(1)))
                        .as("PelagoProject#13144", "d_next_o_id")};
          })
          .print(pg{"pm-csv"})
          .prepare();

  return {o_max, no_max, d_next_oid};
}

static void print_inconsistency(std::string a, std::string b) {
  std::stringstream check1(a);
  std::string i1;

  std::stringstream check2(b);
  std::string i2;

  // Tokenizing w.r.t. space ' '
  while (getline(check1, i1, '\n') && getline(check2, i2, '\n')) {
    if (i1 != i2) {
      LOG(INFO) << i1 << " | " << i2;
    }
  }
}

bool TPCC::consistency_check_2(bool print_inconsistent_rows) {
  // Check-2
  // Entries in the DISTRICT, ORDER, and NEW-ORDER tables must satisfy the
  // relationship: D_NEXT_O_ID - 1 = max(O_ID) = max(NO_O_ID)
  // for each district defined by
  // (D_W_ID = O_W_ID = NO_W_ID) and (D_ID = O_D_ID = NO_D_ID).
  // This condition does not apply to the NEW-ORDER table for any districts
  // which have no outstanding new orders (i.e., the number of rows is zero).

  //  Comment 2: For Consistency Conditions 2 and 4 (Clauses 3.3.2.2
  //  and 3.3.2.4), sampling the first, last, and two random warehouses is
  //  sufficient.

  LOG(INFO) << "##########\tConsistency Check - 2";
  bool db_consistent = true;

  auto queries = consistency_check_2_query_builder();

  std::ostringstream stream_orders;
  std::ostringstream stream_new_orders;
  std::ostringstream stream_district_orders;

  stream_orders << queries[0].execute();
  stream_new_orders << queries[1].execute();
  stream_district_orders << queries[2].execute();

  std::string s_orders = stream_orders.str();
  std::string s_new_orders = stream_new_orders.str();
  std::string s_dist_orders = stream_district_orders.str();

  if (s_orders == s_dist_orders) {
    if (s_new_orders != s_dist_orders) {
      LOG(INFO) << "NewOrders and District Orders doesnt match.";
      db_consistent = false;
      LOG(INFO) << "NewOrders, DistOrders";
      print_inconsistency(s_new_orders, s_dist_orders);
    }
  } else {
    LOG(INFO) << "Orders and District Orders doesnt match.";
    db_consistent = false;
    LOG(INFO) << "Orders, DistOrders";
    LOG(INFO) << s_orders;
    LOG(INFO) << "##########";
    LOG(INFO) << s_dist_orders;
    print_inconsistency(s_orders, s_dist_orders);
  }
  LOG(INFO) << "##########\tConsistency Check - 2 Completed: " << db_consistent;
  return db_consistent;
}

PreparedStatement TPCC::consistency_check_3_query_builder(bool return_aggregate,
                                                          string olap_plugin) {
  RelBuilderFactory ctx{"tpcc_consistency_check_3"};
  CatalogParser &catalog = CatalogParser::getInstance();
  auto tpcc_check_3 =
      ctx.getBuilder()
          .scan("tpcc_neworder<block-remote>",
                {"no_w_id", "no_d_id", "no_o_id"}, CatalogParser::getInstance(),
                pg{"block-remote"})
          .router(32, RoutingPolicy::LOCAL, DeviceType::CPU)
          .unpack()
          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {
                (arg["no_w_id"]).as("tmpRelationProject", "no_w_id"),
                (arg["no_d_id"]).as("tmpRelationProject", "no_d_id"),
                (arg["no_o_id"]).as("tmpRelationProject", "no_o_id_min"),
                (arg["no_o_id"]).as("tmpRelationProject", "no_o_id_max"),
            };
          })
          .groupby(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {(arg["no_w_id"]).as("tmpRelation", "no_w_id"),
                        (arg["no_d_id"]).as("tmpRelation", "no_d_id")};
              },
              [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                return {
                    GpuAggrMatExpr{
                        (arg["no_o_id_max"]).as("tmpRelation", "maxCount"), 1,
                        0, MAX},
                    GpuAggrMatExpr{
                        (arg["no_o_id_min"]).as("tmpRelation", "minCount"), 2,
                        0, MIN},
                    GpuAggrMatExpr{
                        (expression_t{INT32_C(1)}).as("tmpRelation", "noCount"),
                        3, 0, SUM}};
              },
              log2(this->num_warehouse * TPCC_NDIST_PER_WH) + 1,
              this->num_warehouse * TPCC_NDIST_PER_WH)
          //.pack()
          .router(DegreeOfParallelism{1}, 128, RoutingPolicy::RANDOM,
                  DeviceType::CPU)
          //.unpack()
          .groupby(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {(arg["no_w_id"]), (arg["no_d_id"])};
              },
              [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                return {GpuAggrMatExpr{(arg["maxCount"]), 1, 0, MAX},
                        GpuAggrMatExpr{(arg["minCount"]), 2, 0, MIN},
                        GpuAggrMatExpr{(arg["noCount"]), 3, 0, SUM}};
              },
              log2(this->num_warehouse * TPCC_NDIST_PER_WH) + 1,
              this->num_warehouse * TPCC_NDIST_PER_WH)

          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {(arg["no_w_id"]).as("PelagoProject#13144", "no_w_id"),
                    (arg["no_d_id"]).as("PelagoProject#13144", "no_d_id"),
                    (arg["maxCount"]).as("PelagoProject#13144", "maxCount"),
                    (arg["minCount"]).as("PelagoProject#13144", "minCount"),
                    (arg["maxCount"] - 2100 + 1)
                        .as("PelagoProject#13144", "expected_count"),
                    (arg["noCount"]).as("PelagoProject#13144", "actual_count")

            };
          })
          .filter([&](const auto &arg) -> expression_t {
            return ne(arg["expected_count"], arg["actual_count"]);
          });

  if (return_aggregate)
    tpcc_check_3 = tpcc_check_3.reduce(
        [&](const auto &arg) -> std::vector<expression_t> {
          return {(expression_t{INT32_C(1)})
                      .as("tmpRelation#6", "count_inconsistent_rec")};
        },
        {SUM});

  tpcc_check_3 = tpcc_check_3.print(pg{"pm-csv"});

  return tpcc_check_3.prepare();
}

bool TPCC::consistency_check_3(bool print_inconsistent_rows) {
  // Check-3
  // Entries in the NEW-ORDER table must satisfy the relationship:
  // max(NO_O_ID) - min(NO_O_ID) + 1 =
  //                       [# of rows in the NEW-ORDER table for this district]
  // for each district defined by NO_W_ID and NO_D_ID. This condition does not
  // apply to any districts which have no outstanding new orders
  // (i.e., the number of rows is zero).

  // SQL: (EXPECTED 0 ROWS)
  //     SELECT no_w_id, no_d_id, max(no_o_id)-min(no_o_id)+1, count(no_o_id)
  //     FROM tpcc_new_order
  //     GROUP BY no_w_id, no_d_id
  //     HAVING max(no_o_id)-min(no_o_id)+1 != count(no_o_id)

  LOG(INFO) << "##########\tConsistency Check - 3";
  bool db_consistent = true;

  auto query = consistency_check_3_query_builder();
  std::ostringstream stream_new_orders;
  stream_new_orders << query.execute();
  std::string n_orders = stream_new_orders.str();

  int n_failed_rows = stoi(n_orders);

  if (n_failed_rows) {
    LOG(INFO) << "Consistency Check # 3 failed."
              << "\n\t# of inconsistent rows: " << n_failed_rows;
    db_consistent = false;
    if (print_inconsistent_rows) {
      LOG(INFO) << consistency_check_3_query_builder(false).execute();
    }
  }
  LOG(INFO) << "##########\tConsistency Check - 3 Completed: " << db_consistent;

  return db_consistent;
}

PreparedStatement TPCC::consistency_check_4_query_builder(bool return_aggregate,
                                                          string olap_plugin) {
  // orders
  // o_w_id, o_d_id, sum(o_ol_cnt)

  RelBuilderFactory ctx_o{"tpcc_consistency_check_4_order"};
  CatalogParser &catalog = CatalogParser::getInstance();
  auto tpcc_check_4_order =
      ctx_o.getBuilder()
          .scan("tpcc_order<" + olap_plugin + ">",
                {"o_w_id", "o_d_id", "o_ol_cnt"}, CatalogParser::getInstance(),
                pg{olap_plugin})
          .router(32, RoutingPolicy::LOCAL, DeviceType::CPU)
          .unpack()
          .groupby(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {(arg["o_w_id"]).as("tmpRelation", "o_w_id"),
                        (arg["o_d_id"]).as("tmpRelation", "o_d_id")};
              },
              [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                return {GpuAggrMatExpr{
                    (arg["o_ol_cnt"]).as("tmpRelation", "sum_o_ol_cnt"), 1, 0,
                    SUM}};
              },
              log2(this->num_warehouse * TPCC_NDIST_PER_WH) + 1,
              this->num_warehouse * TPCC_NDIST_PER_WH)
          //.pack()
          .router(DegreeOfParallelism{1}, 128, RoutingPolicy::RANDOM,
                  DeviceType::CPU)
          //.unpack()
          .groupby(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {(arg["o_w_id"]), (arg["o_d_id"])};
              },
              [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                return {GpuAggrMatExpr{(arg["sum_o_ol_cnt"]), 1, 0, SUM}};
              },
              log2(this->num_warehouse * TPCC_NDIST_PER_WH) + 1,
              this->num_warehouse * TPCC_NDIST_PER_WH)

          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {
                (arg["o_w_id"]).as("tmpRelation#2", "o_w_id"),
                (arg["o_d_id"]).as("tmpRelation#2", "o_d_id"),
                (arg["sum_o_ol_cnt"]).as("tmpRelation#2", "sum_o_ol_cnt"),
            };
          });

  // orderline
  // ol_w_id, ol_d_id, count(ol_o_id)
  auto tpcc_check_4_orderLine =
      ctx_o.getBuilder()
          .scan("tpcc_orderline<" + olap_plugin + ">",
                {"ol_w_id", "ol_d_id", "ol_o_id"}, CatalogParser::getInstance(),
                pg{olap_plugin})
          .router(32, RoutingPolicy::LOCAL, DeviceType::CPU)
          .unpack()
          .groupby(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {(arg["ol_w_id"]).as("tmpRelation#3", "ol_w_id"),
                        (arg["ol_d_id"]).as("tmpRelation#3", "ol_d_id")};
              },
              [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                return {GpuAggrMatExpr{(expression_t{INT32_C(1)})
                                           .as("tmpRelation#3", "cnt_ol_o_id"),
                                       1, 0, SUM}};
              },
              log2(this->num_warehouse * TPCC_NDIST_PER_WH) + 1,
              this->num_warehouse * TPCC_NDIST_PER_WH)
          // .pack()
          .router(DegreeOfParallelism{1}, 128, RoutingPolicy::RANDOM,
                  DeviceType::CPU)
          // .unpack()
          .groupby(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {(arg["ol_w_id"]), (arg["ol_d_id"])};
              },
              [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                return {GpuAggrMatExpr{(arg["cnt_ol_o_id"]), 1, 0, SUM}};
              },
              log2(this->num_warehouse * TPCC_NDIST_PER_WH) + 1,
              this->num_warehouse * TPCC_NDIST_PER_WH)

          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {
                (arg["ol_w_id"]).as("tmpRelation#4", "ol_w_id"),
                (arg["ol_d_id"]).as("tmpRelation#4", "ol_d_id"),
                (arg["cnt_ol_o_id"]).as("tmpRelation#4", "cnt_ol_o_id"),
            };
          });

  auto tpcc_check_4 =
      tpcc_check_4_orderLine
          .join(
              tpcc_check_4_order,
              [&](const auto &build_arg) -> expression_t {
                return expressions::RecordConstruction{
                    build_arg["o_w_id"].as("PelagoJoin#2030", "bk_0"),
                    build_arg["o_d_id"].as("PelagoJoin#2030", "bk_1")}
                    .as("PelagoJoin#2030", "bk");
              },
              [&](const auto &probe_arg) -> expression_t {
                return expressions::RecordConstruction{
                    probe_arg["ol_w_id"].as("PelagoJoin#2030", "pk_0"),
                    probe_arg["ol_d_id"].as("PelagoJoin#2030", "pk_1")}
                    .as("PelagoJoin#2030", "pk");
              },
              log2(this->num_warehouse * TPCC_NDIST_PER_WH) + 1,
              this->num_warehouse * TPCC_NDIST_PER_WH)
          .filter([&](const auto &arg) -> expression_t {
            return ne(arg["sum_o_ol_cnt"], arg["cnt_ol_o_id"]);
          })
          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {
                (arg["o_w_id"]).as("tmpRelation#5", "w_id"),
                (arg["o_d_id"]).as("tmpRelation#5", "d_id"),
                (arg["sum_o_ol_cnt"]).as("tmpRelation#5", "o_count"),
                (arg["cnt_ol_o_id"]).as("tmpRelation#5", "ol_count"),
            };
          });

  if (return_aggregate)
    tpcc_check_4 = tpcc_check_4.reduce(
        [&](const auto &arg) -> std::vector<expression_t> {
          // return {INT32_C(1)}.as("tmpRelation#5", "ol_count");
          return {(expression_t{INT32_C(1)})
                      .as("tmpRelation#6", "count_inconsistent_rec")};
        },
        {SUM});

  tpcc_check_4 = tpcc_check_4.print(pg{"pm-csv"});

  return tpcc_check_4.prepare();
}

bool TPCC::consistency_check_4(bool print_inconsistent_rows) {
  // Check-4
  // Entries in the ORDER and ORDER-LINE tables must satisfy the relationship:
  // sum(O_OL_CNT) = [number of rows in the ORDER-LINE table for this district]
  // for each district defined by (O_W_ID = OL_W_ID) and (O_D_ID = OL_D_ID).

  //  Comment 2: For Consistency Conditions 2 and 4 (Clauses 3.3.2.2
  //  and 3.3.2.4), sampling the first, last, and two random warehouses is
  //  sufficient.

  // SQL: (EXPECTED 0 ROWS)
  //     SELECT o_w_id, o_d_id, o_count, ol_count
  //     FROM (
  //	    SELECT o_w_id, o_d_id, sum(o_ol_cnt) as o_count
  //	    FROM tpcc_orders
  //	    GROUP BY o_w_id, o_d_id) o,
  //	    (SELECT ol_w_id, ol_d_id, count(o_ol_id) as ol_count
  //	    FROM tpcc_orderline
  //	    GROUP BY ol_w_id, ol_d_id) ol
  //     WHERE o.o_w_id = ol.ol_w_id
  //     AND o.o_d_id = ol.ol_d_id
  //     AND o_count != ol_count

  LOG(INFO) << "##########\tConsistency Check - 4";
  bool db_consistent = true;

  std::ostringstream check4_result;
  check4_result << consistency_check_4_query_builder().execute();
  std::string n_orders_str = check4_result.str();

  int n_failed_rows = stoi(n_orders_str);

  if (n_failed_rows) {
    LOG(INFO) << "Consistency Check # 4 failed."
              << "\n\t# of inconsistent rows: " << n_failed_rows;
    db_consistent = false;
    if (print_inconsistent_rows) {
      LOG(INFO) << consistency_check_4_query_builder(false).execute();
    }
  }
  LOG(INFO) << "##########\tConsistency Check - 4 Completed: " << db_consistent;

  return db_consistent;
}

class OltpInconsistentException : public std::runtime_error {
 public:
  explicit OltpInconsistentException(const char *message)
      : std::runtime_error(message) {}
};

void TPCC::verify_consistency() {
  LOG(INFO) << "##############################################################";
  LOG(INFO) << "Verifying consistency...";
  LOG(INFO) << "##############################################################";
  // NOTE: Only first-four are required, others are extra.

  //  Comment 1: The consistency conditions were chosen so that they would
  //  remain valid within the context of a larger order-entry application that
  //  includes the five TPC-C transactions (See Clause 1.1.). They are designed
  //  to be independent of the length of time for which such an application
  //  would be executed. Thus, for example, a condition involving I_PRICE was
  //  not included here since it is conceivable that within a larger application
  //  I_PRICE is modified from time to time.
  //
  //  Comment 2: For Consistency Conditions 2 and 4 (Clauses 3.3.2.2
  //  and 3.3.2.4), sampling the first, last, and two random warehouses is
  //  sufficient.

  // Set execution affinity to everywhere..
  //  cpu_set_t all_cpu_set;
  //  CPU_ZERO(&all_cpu_set);
  //  for (uint32_t i =0; i < topology::getInstance().getCoreCount(); i++)
  //    CPU_SET(i, &all_cpu_set);
  //  set_exec_location_on_scope d{all_cpu_set};

  // execute consistency checks.
  if (!consistency_check_1()) {
    throw OltpInconsistentException("DB IS NOT CONSISTENT: Check 1 failed");
  }
  if (!consistency_check_2()) {
    throw OltpInconsistentException("DB IS NOT CONSISTENT: Check 2 failed");
  }
  if (!consistency_check_3()) {
    throw OltpInconsistentException("DB IS NOT CONSISTENT: Check 3 failed");
  }
  if (!consistency_check_4()) {
    throw OltpInconsistentException("DB IS NOT CONSISTENT: Check 4 failed");
  }
  LOG(INFO) << "##############################################################";
}

}  // namespace bench
