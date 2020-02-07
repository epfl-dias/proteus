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

#ifndef HARMONIA_DATA_EXPORT_TPCC_ORDER_HPP_
#define HARMONIA_DATA_EXPORT_TPCC_ORDER_HPP_

#include "../../queries.hpp"

using ch_access_plugin = AeolusRemotePlugin;

class DataExporter_CH {
 private:
  constexpr static auto csv_plugin = "pm-csv";
  constexpr static auto bin_plugin = "block";

  inline static auto getBuilder() {
    static RelBuilderFactory ctx{"DataExporter_CH"};
    return ctx.getBuilder();
  }

  DataExporter_CH() {}

 public:
  static void exportAll(std::string output_dir);

  static PreparedStatement export_tpcc_order(bool output_binary = false);
  static PreparedStatement export_tpcc_orderline(bool output_binary = false);
  static PreparedStatement export_ch_nation(bool output_binary = false);
  static PreparedStatement export_ch_region(bool output_binary = false);
  static PreparedStatement export_ch_supplier(bool output_binary = false);
  static PreparedStatement export_tpcc_stock(bool output_binary = false);
  static PreparedStatement export_tpcc_item(bool output_binary = false);
  static PreparedStatement export_tpcc_warehouse(bool output_binary = false);
  static PreparedStatement export_tpcc_district(bool output_binary = false);
  static PreparedStatement export_tpcc_history(bool output_binary = false);
  static PreparedStatement export_tpcc_customer(bool output_binary = false);
  static PreparedStatement export_tpcc_new_order(bool output_binary = false);
};

#endif /* HARMONIA_DATA_EXPORT_TPCC_ORDER_HPP_ */
