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

bool TPCC::consistency_check_1() {
  // Check-1
  // Entries in the WAREHOUSE and DISTRICT tables must satisfy the relationship:
  // W_YTD = sum(D_YTD)

  // SQL: (EXPECTED 0 ROWS)
  //     SELECT sum(w_ytd), sum(d_ytd)
  //     FROM tpcc_warehouse, tpcc_district
  //     WHERE tpcc_warehouse.w_id = tpcc_district.d_w_id
  //     GROUP BY tpcc_warehouse.w_id
  //     HAVING sum(w_ytd) != sum(d_ytd)

  bool db_consistent = true;
  std::vector<proteus::thread> workers;
  // TODO: parallelize

  //  loaders.emplace_back(
  //      [this]() { this->create_tbl_warehouse(this->num_warehouse); });

  for (uint64_t i = 0; i < this->num_warehouse; i++) {
    // Get wh ytd.
    double wh_ytd = 0.0;
    const ushort w_col_scan[] = {8};  // position in columns
    auto w_idx_ptr = (global_conf::IndexVal *)table_warehouse->p_index->find(i);

    table_warehouse->getRecordByKey(w_idx_ptr, UINT64_MAX, w_col_scan, 1,
                                    &wh_ytd);

    double district_ytd = 0.0;
    for (uint j = 0; j < TPCC_NDIST_PER_WH; j++) {
      // Get all district ytd and sum it.
      double tmp_d_ytd = 0.0;
      const ushort d_col_scan[] = {9};  // position in columns
      auto d_idx_ptr = (global_conf::IndexVal *)table_district->p_index->find(
          MAKE_DIST_KEY(i, j));
      table_district->getRecordByKey(d_idx_ptr, UINT64_MAX, d_col_scan, 1,
                                     &tmp_d_ytd);
      district_ytd += tmp_d_ytd;
    }

    if ((size_t)wh_ytd != (size_t)district_ytd) {
      LOG(INFO) << "FAILED CONSISTENCY CHECK-1"
                << "\n\tWH: " << i << "\n\twh_ytd: " << wh_ytd << " | "
                << "district_ytd: " << district_ytd;
      db_consistent = false;
    }
  }

  for (auto &th : workers) {
    th.join();
  }
  return db_consistent;
}

std::vector<PreparedStatement> TPCC::consistency_check_2_query_builder() {
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
          .pack()
          .router(DegreeOfParallelism{1}, 128, RoutingPolicy::RANDOM,
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
          .pack()
          .router(DegreeOfParallelism{1}, 128, RoutingPolicy::RANDOM,
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
                    (arg["d_next_o_id"] - ((int64_t)1))
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

bool TPCC::consistency_check_2() {
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
    print_inconsistency(s_orders, s_dist_orders);
  }
  return db_consistent;
}

std::vector<PreparedStatement> TPCC::consistency_check_3_query_builder() {
  RelBuilderFactory ctx{"tpcc_consistency_check_3"};
  CatalogParser &catalog = CatalogParser::getInstance();
  PreparedStatement new_order_stats =
      ctx.getBuilder()
          .scan("tpcc_neworder<block-remote>",
                {"no_o_id", "no_d_id", "no_w_id"}, CatalogParser::getInstance(),
                pg{"block-remote"})
          .router(DegreeOfParallelism{48}, 32, RoutingPolicy::LOCAL,
                  DeviceType::CPU)

          .groupby(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["no_w_id"], arg["no_d_id"]};
              },
              [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                return {GpuAggrMatExpr{(arg["no_o_id"]), 1, 0, MAX},
                        GpuAggrMatExpr{(arg["no_o_id"]), 2, 0, MIN},
                        GpuAggrMatExpr{(expression_t{INT64_C(1)})
                                           .as("PelagoAggregate#1052", "count"),
                                       3, 0, SUM}};
              },
              log2(this->num_warehouse * TPCC_NDIST_PER_WH) + 1,
              log2(this->num_warehouse * TPCC_NDIST_PER_WH) + (1024 * 1024))
          .pack()
          .router(DegreeOfParallelism{1}, 128, RoutingPolicy::RANDOM,
                  DeviceType::CPU)
          .unpack()
          .groupby(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {(arg["no_w_id"]).as("PelagoAggregate#1052", "no_w_id"),
                        (arg["no_d_id"]).as("PelagoAggregate#1052", "no_d_id")};
              },
              [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                return {GpuAggrMatExpr{
                            (arg["no_o_id"])
                                .as("PelagoAggregate#1052", "max_neworder"),
                            1, 0, MAX},
                        GpuAggrMatExpr{
                            (arg["no_o_id"])
                                .as("PelagoAggregate#1052", "min_neworder"),
                            2, 0, MIN},
                        GpuAggrMatExpr{
                            (expression_t{1})
                                .as("PelagoAggregate#1052", "count_neworder"),
                            3, 0, SUM}};
              },
              log2(this->num_warehouse * TPCC_NDIST_PER_WH) + 1, 1024 * 1024)

          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {(arg["no_w_id"]).as("PelagoProject#13144", "no_w_id"),
                    (arg["no_d_id"]).as("PelagoProject#13144", "no_d_id"),
                    (arg["max_neworder"] - arg["min_neworder"] + 1)
                        .as("PelagoProject#13144", "expected_count"),
                    (arg["count_neworder"])
                        .as("PelagoProject#13144", "actual_count")

            };
          })
          .sort(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["no_w_id"], arg["no_d_id"], arg["expected_count"],
                        arg["actual_count"]};
              },
              {direction::ASC, direction::ASC, direction::NONE,
               direction::NONE})
          .print(pg{"pm-csv"})
          .prepare();
  return {new_order_stats};
}

bool TPCC::consistency_check_3() {
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

  bool db_consistent = true;

  auto query = consistency_check_3_query_builder();
  LOG(INFO) << "CHECK-3";
  LOG(INFO) << query[0].execute();
  LOG(INFO) << "CHECK-3";

  return db_consistent;
}

std::vector<PreparedStatement> TPCC::consistency_check_4_query_builder() {
  return {};
}

bool TPCC::consistency_check_4() {
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

  bool db_consistent = true;
  std::vector<proteus::thread> workers;

  // TODO: parallelize

  for (uint i = 0; i < this->num_warehouse; i++) {
    for (uint j = 0; j < TPCC_NDIST_PER_WH; j++) {
      size_t o_ol_sum = 0;
      // TODO: Get sum(o_ol_cnt)

      // TODO: Get count(ol_o_id)
      size_t ol_count = 0;

      if (o_ol_sum != ol_count) {
        LOG(INFO) << "FAILED CONSISTENCY CHECK-4"
                  << "\n\tWH: " << i << " | DT: " << j
                  << "\n\to_ol_count: " << o_ol_sum
                  << " | ol_count: " << ol_count;
        db_consistent = false;
      }
    }
  }

  for (auto &th : workers) {
    th.join();
  }
  return db_consistent;
}

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
  if (consistency_check_1() && consistency_check_2() && consistency_check_3() /*&&
      consistency_check_4()*/) {
    LOG(INFO) << "DB IS CONSISTENT.";
  } else {
    LOG(FATAL) << "DB IS NOT CONSISTENT.";
  }
  LOG(INFO) << "##############################################################";
}

}  // namespace bench
