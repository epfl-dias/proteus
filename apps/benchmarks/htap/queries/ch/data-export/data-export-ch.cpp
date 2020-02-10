/*
    Harmonia -- High-performance elastic HTAP on heterogeneous hardware.

                            Copyright (c) 2017
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

#include "data-export-ch.hpp"

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#else
// TODO: remove as soon as the default GCC moves filesystem out of experimental
//  GCC 8.3 has made the transition, but the default GCC in Ubuntu 18.04 is 7.4
#include <experimental/filesystem>
namespace std {
namespace filesystem = std::experimental::filesystem;
}
#endif

#include <fstream>
#include <system_error>

void DataExporter_CH::exportAll(std::string& output_dir) {
  std::vector<QueryResult> queryResults;
  std::vector<PreparedStatement> relations;

  // order
  relations.emplace_back(export_tpcc_order(true));
  relations.emplace_back(export_tpcc_order(false));

  // orderline
  relations.emplace_back(export_tpcc_orderline(true));
  relations.emplace_back(export_tpcc_orderline(false));

  for (auto& q : relations) {
    queryResults.emplace_back(q.execute());
  }

  // TODO: move the results to output_dir
  std::filesystem::path src_path{"/dev/shm"};
  std::filesystem::path dst_path{output_dir};

  assert(std::filesystem::is_directory(dst_path));

  std::filesystem::copy(src_path, dst_path,
                        std::filesystem::copy_options::recursive);
}
PreparedStatement DataExporter_CH::export_ch_nation(bool output_binary) {
  throw std::runtime_error("unimplemented");
}
PreparedStatement DataExporter_CH::export_ch_region(bool output_binary) {
  throw std::runtime_error("unimplemented");
}
PreparedStatement DataExporter_CH::export_ch_supplier(bool output_binary) {
  throw std::runtime_error("unimplemented");
}
PreparedStatement DataExporter_CH::export_tpcc_stock(bool output_binary) {
  throw std::runtime_error("unimplemented");
}
PreparedStatement DataExporter_CH::export_tpcc_item(bool output_binary) {
  throw std::runtime_error("unimplemented");
}
PreparedStatement DataExporter_CH::export_tpcc_warehouse(bool output_binary) {
  throw std::runtime_error("unimplemented");
}
PreparedStatement DataExporter_CH::export_tpcc_district(bool output_binary) {
  throw std::runtime_error("unimplemented");
}
PreparedStatement DataExporter_CH::export_tpcc_history(bool output_binary) {
  throw std::runtime_error("unimplemented");
}
PreparedStatement DataExporter_CH::export_tpcc_customer(bool output_binary) {
  throw std::runtime_error("unimplemented");
}
PreparedStatement DataExporter_CH::export_tpcc_new_order(bool output_binary) {
  throw std::runtime_error("unimplemented");
}

PreparedStatement DataExporter_CH::export_tpcc_orderline(bool output_binary) {
  auto rel =
      getBuilder()
          .scan<ch_access_plugin>(
              "tpcc_orderline<block-remote>",
              {"ol_o_id", "ol_d_id", "ol_w_id", "ol_number", "ol_i_id",
               "ol_supply_w_id", "ol_delivery_d", "ol_quantity", "ol_amount"},
              CatalogParser::getInstance())
          .unpack()
          .print(
              [&](const auto& arg) -> std::vector<expression_t> {
                return {
                    arg["ol_o_id"],       arg["ol_d_id"],
                    arg["ol_w_id"],       arg["ol_number"],
                    arg["ol_i_id"],       arg["ol_supply_w_id"],
                    arg["ol_delivery_d"], arg["ol_quantity"],
                    arg["ol_amount"],
                };
              },
              (output_binary ? pg(bin_plugin) : pg(csv_plugin)),
              (output_binary ? "tpcc_orderline" : "tpcc_orderline.tbl"));
  return rel.prepare();
}

PreparedStatement DataExporter_CH::export_tpcc_order(bool output_binary) {
  auto rel =
      getBuilder()
          .scan<ch_access_plugin>(
              "tpcc_order<block-remote>",
              {"o_id", "o_d_id", "o_w_id", "o_c_id", "o_entry_d",
               "o_carrier_id", "o_ol_cnt", "o_all_local"},
              CatalogParser::getInstance())
          .unpack()
          .print(
              [&](const auto& arg) -> std::vector<expression_t> {
                return {arg["o_id"],     arg["o_d_id"],     arg["o_w_id"],
                        arg["o_c_id"],   arg["o_entry_d"],  arg["o_carrier_id"],
                        arg["o_ol_cnt"], arg["o_all_local"]};
              },
              (output_binary ? pg(bin_plugin) : pg(csv_plugin)),
              (output_binary ? "tpcc_order" : "tpcc_order.tbl"));
  return rel.prepare();
}
