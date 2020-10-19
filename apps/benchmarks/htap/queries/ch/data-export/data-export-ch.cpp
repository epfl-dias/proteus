/*
    Proteus -- High-performance query processing on heterogeneous hardware.

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

#include <system_error>

void DataExporter_CH::exportAll(std::string& output_dir) {
  std::vector<QueryResult> queryResults;
  std::vector<PreparedStatement> relations;

  // Order
  relations.emplace_back(export_tpcc_order(true));
  relations.emplace_back(export_tpcc_order(false));

  // Orderline
  relations.emplace_back(export_tpcc_orderline(true));
  relations.emplace_back(export_tpcc_orderline(false));

  // New-Order
  relations.emplace_back(export_tpcc_new_order(true));
  relations.emplace_back(export_tpcc_new_order(false));

  // Warehouse
  relations.emplace_back(export_tpcc_warehouse(true));
  relations.emplace_back(export_tpcc_warehouse(false));

  // District
  relations.emplace_back(export_tpcc_district(true));
  relations.emplace_back(export_tpcc_district(false));

  // item
  relations.emplace_back(export_tpcc_item(true));
  relations.emplace_back(export_tpcc_item(false));
  // stock
  relations.emplace_back(export_tpcc_stock(true));
  relations.emplace_back(export_tpcc_stock(false));

  // Customer
  relations.emplace_back(export_tpcc_customer(true));
  relations.emplace_back(export_tpcc_customer(false));
  // History
  relations.emplace_back(export_tpcc_history(true));
  relations.emplace_back(export_tpcc_history(false));

  // supplier
  relations.emplace_back(export_ch_supplier(true));
  relations.emplace_back(export_ch_supplier(false));

  // region
  relations.emplace_back(export_ch_region(true));
  relations.emplace_back(export_ch_region(false));

  // nation
  relations.emplace_back(export_ch_nation(true));
  relations.emplace_back(export_ch_nation(false));

  queryResults.reserve(relations.size());
  for (auto& q : relations) {
    queryResults.emplace_back(q.execute());
  }

  // move the results to output_dir
  // FIXME: /dev/shm might contain other files than the outputs.

  std::filesystem::path src_path{"/dev/shm"};
  std::filesystem::path dst_path{output_dir};
  assert(std::filesystem::is_directory(dst_path));
  std::filesystem::copy(src_path, dst_path,
                        std::filesystem::copy_options::recursive);
}
PreparedStatement DataExporter_CH::export_ch_nation(bool output_binary) {
  /*
   * FIXME: Missing string attributes: [n_name, n_comment]
   * */

  auto rel =
      getBuilder()
          .scan("tpcc_nation<block-remote>", {"n_nationkey", "n_regionkey"},
                CatalogParser::getInstance(), pg{ch_access_plugin::type})
          .unpack()
          .print((output_binary ? pg(bin_plugin) : pg(csv_plugin)),
                 (output_binary ? "tpcc_nation" : "tpcc_nation.tbl"));
  return rel.prepare();
}
PreparedStatement DataExporter_CH::export_ch_region(bool output_binary) {
  /*
   * FIXME: Missing string attributes: [r_name, r_comment]
   * */

  auto rel = getBuilder()
                 .scan("tpcc_region<block-remote>", {"r_regionkey"},
                       CatalogParser::getInstance(), pg{ch_access_plugin::type})
                 .unpack()
                 .print((output_binary ? pg(bin_plugin) : pg(csv_plugin)),
                        (output_binary ? "tpcc_region" : "tpcc_region.tbl"));
  return rel.prepare();
}
PreparedStatement DataExporter_CH::export_ch_supplier(bool output_binary) {
  /*
   * FIXME: Missing string attributes: [su_name, su_address, su_phone,
   * su_comment]
   * */

  auto rel =
      getBuilder()
          .scan("tpcc_supplier<block-remote>",
                {"su_suppkey", "su_nationkey", "su_acctbal"},
                CatalogParser::getInstance(), pg{ch_access_plugin::type})
          .unpack()
          .print((output_binary ? pg(bin_plugin) : pg(csv_plugin)),
                 (output_binary ? "tpcc_supplier" : "tpcc_supplier.tbl"));
  return rel.prepare();
}
PreparedStatement DataExporter_CH::export_tpcc_stock(bool output_binary) {
  /*
   * FIXME: Missing string attributes: [s_dist, s_data]
   * */

  auto rel = getBuilder()
                 .scan("tpcc_stock<block-remote>",
                       {"s_i_id", "s_w_id", "s_quantity", "s_ytd",
                        "s_order_cnt", "s_remote_cnt", "s_su_suppkey"},
                       CatalogParser::getInstance(), pg{ch_access_plugin::type})
                 .unpack()
                 .print((output_binary ? pg(bin_plugin) : pg(csv_plugin)),
                        (output_binary ? "tpcc_stock" : "tpcc_stock.tbl"));
  return rel.prepare();
}
PreparedStatement DataExporter_CH::export_tpcc_item(bool output_binary) {
  /*
   * FIXME: Missing string attributes: [i_name, i_data]
   * */
  auto rel =
      getBuilder()
          .scan("tpcc_item<block-remote>", {"i_id", "i_im_id", "i_price"},
                CatalogParser::getInstance(), pg{ch_access_plugin::type})
          .unpack()
          .print((output_binary ? pg(bin_plugin) : pg(csv_plugin)),
                 (output_binary ? "tpcc_item" : "tpcc_item.tbl"));
  return rel.prepare();
}
PreparedStatement DataExporter_CH::export_tpcc_warehouse(bool output_binary) {
  /*
   * FIXME: Missing string attributes: [w_name, w_street_1, w_street_2, w_city,
   * w_state, w_zip]
   * */
  auto rel =
      getBuilder()
          .scan("tpcc_warehouse<block-remote>", {"w_id", "w_tax", "w_ytd"},
                CatalogParser::getInstance(), pg{ch_access_plugin::type})
          .unpack()
          .print((output_binary ? pg(bin_plugin) : pg(csv_plugin)),
                 (output_binary ? "tpcc_warehouse" : "tpcc_warehouse.tbl"));
  return rel.prepare();
}
PreparedStatement DataExporter_CH::export_tpcc_district(bool output_binary) {
  /*
   * FIXME: Missing string attributes: [d_name, d_street_1, d_street_2, d_city,
   * d_state, d_zip]
   * */

  auto rel =
      getBuilder()
          .scan("tpcc_district<block-remote>",
                {"d_id", "d_w_id", "d_tax", "d_ytd", "d_next_o_id"},
                CatalogParser::getInstance(), pg{ch_access_plugin::type})
          .unpack()
          .print((output_binary ? pg(bin_plugin) : pg(csv_plugin)),
                 (output_binary ? "tpcc_district" : "tpcc_district.tbl"));
  return rel.prepare();
}
PreparedStatement DataExporter_CH::export_tpcc_history(bool output_binary) {
  /*
   * FIXME: Missing string attributes: [h_data]
   * */

  auto rel = getBuilder()
                 .scan("tpcc_history<block-remote>",
                       {"h_c_id", "h_c_d_id", "h_c_w_id", "h_d_id", "h_w_id",
                        "h_date", "h_amount"},
                       CatalogParser::getInstance(), pg{ch_access_plugin::type})
                 .unpack()
                 .print((output_binary ? pg(bin_plugin) : pg(csv_plugin)),
                        (output_binary ? "tpcc_history" : "tpcc_history.tbl"));
  return rel.prepare();
}
PreparedStatement DataExporter_CH::export_tpcc_customer(bool output_binary) {
  /*
   * FIXME: Missing string attributes:
   * [c_first, c_middle, c_last, c_street, c_city, c_state, c_zip, c_phone,
   * c_credit, c_data]
   * */

  auto rel =
      getBuilder()
          .scan("tpcc_customer<block-remote>",
                {"c_id", "c_w_id", "c_d_id", "c_since", "c_credit_lim",
                 "c_discount", "c_balance", "c_ytd_payment", "c_payment_cnt",
                 "c_delivery_cnt", "c_n_nationkey"},
                CatalogParser::getInstance(), pg{ch_access_plugin::type})
          .unpack()
          .print((output_binary ? pg(bin_plugin) : pg(csv_plugin)),
                 (output_binary ? "tpcc_customer" : "tpcc_customer.tbl"));
  return rel.prepare();
}

PreparedStatement DataExporter_CH::export_tpcc_orderline(bool output_binary) {
  auto rel =
      getBuilder()
          .scan("tpcc_orderline<block-remote>",
                {"ol_o_id", "ol_d_id", "ol_w_id", "ol_number", "ol_i_id",
                 "ol_supply_w_id", "ol_delivery_d", "ol_quantity", "ol_amount"},
                CatalogParser::getInstance(), pg{ch_access_plugin::type})
          .unpack()
          .print((output_binary ? pg(bin_plugin) : pg(csv_plugin)),
                 (output_binary ? "tpcc_orderline" : "tpcc_orderline.tbl"));
  return rel.prepare();
}

PreparedStatement DataExporter_CH::export_tpcc_order(bool output_binary) {
  auto rel = getBuilder()
                 .scan("tpcc_order<block-remote>",
                       {"o_id", "o_d_id", "o_w_id", "o_c_id", "o_entry_d",
                        "o_carrier_id", "o_ol_cnt", "o_all_local"},
                       CatalogParser::getInstance(), pg{ch_access_plugin::type})
                 .unpack()
                 .print((output_binary ? pg(bin_plugin) : pg(csv_plugin)),
                        (output_binary ? "tpcc_order" : "tpcc_order.tbl"));
  return rel.prepare();
}

PreparedStatement DataExporter_CH::export_tpcc_new_order(bool output_binary) {
  auto rel =
      getBuilder()
          .scan("tpcc_neworder<block-remote>",
                {"no_o_id", "no_d_id", "no_w_id"}, CatalogParser::getInstance(),
                pg{ch_access_plugin::type})
          .unpack()
          .print((output_binary ? pg(bin_plugin) : pg(csv_plugin)),
                 (output_binary ? "tpcc_neworder" : "tpcc_neworder.tbl"));
  return rel.prepare();
}
