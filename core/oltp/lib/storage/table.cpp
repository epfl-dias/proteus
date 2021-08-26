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

#include "oltp/storage/table.hpp"

#include <iostream>
#include <mutex>
#include <olap/plan/catalog-parser.hpp>
#include <olap/values/expressionTypes.hpp>
#include <platform/threadpool/threadpool.hpp>
#include <platform/util/timing.hpp>
#include <string>

#include "oltp/common/constants.hpp"
#include "oltp/common/numa-partition-policy.hpp"
#include "oltp/storage/layout/column_store.hpp"

namespace storage {

static std::mutex m_catalog;

Table::~Table() { LOG(INFO) << "Destructing table: " << this->name; };

Table::Table(table_id_t table_id, std::string& name, layout_type storage_layout,
             TableDef& columns)
    : name(name),
      table_id(table_id),
      total_memory_reserved(0),
      storage_layout(storage_layout) {
  for (int i = 0; i < topology::getInstance().getCpuNumaNodeCount(); i++) {
    // vid.emplace_back();
    vid[i].vid_part.store(0);
  }
  std::vector<RecordAttribute*> attrs;
  attrs.reserve(columns.size());
  for (const auto& c : columns.getColumns()) {
    attrs.emplace_back(
        new RecordAttribute(this->name, c.getName(), getProteusType(c)));
    attrs.emplace_back(new RecordAttribute(this->name, c.getName() + "_bitmask",
                                           new BoolType()));
  }

  auto exprType = new BagType(
      *(new RecordType(attrs)));  // new and derefernce is needed due to the
  // BagType getting a reference
  std::lock_guard<std::mutex> lock{m_catalog};
  // FIXME: the table should not require knowledge of all the plugin types
  //  one possible solution is to register the tables "variants" when asking
  //  for the plugin, just before the plan creation, where either way the
  //  scheduler knows about the available plugins
  for (const auto& s :
       {"block-local", "block-remote", "block-elastic", "block-elastic-ni"}) {
    auto tName = name + "<" + s + ">";
    LOG(INFO) << "Registering table " << tName << " to OLAP";
    CatalogParser::getInstance().registerInput(tName, exprType);
  }
}

void Table::reportUsage() {
  LOG(INFO) << "Table: " << this->name;
  for (auto i = 0; i < g_num_partitions; i++) {
    auto curr = vid[i].vid_part.load();
    double percent =
        ((double)curr / ((double)(record_capacity / g_num_partitions))) * 100;

    LOG(INFO) << "P" << i << ": " << curr << " / "
              << (record_capacity / g_num_partitions) << " | " << percent
              << "%";
  }
}

ExpressionType* Table::getProteusType(const ColumnDef& col) {
  switch (col.getType()) {
    case INTEGER: {
      switch (col.getSize()) {
        case 4:
          return new IntType();
        case 8:
          return new Int64Type();
        default: {
          auto msg = std::string{"Unknown integer type of size: "} +
                     std::to_string(col.getSize());
          LOG(ERROR) << msg;
          throw std::runtime_error(msg);
        }
      }
    }
    case FLOAT: {
      switch (col.getSize()) {
        case 8:
          return new FloatType();
        default: {
          auto msg = std::string{"Unknown float type of size: "} +
                     std::to_string(col.getSize());
          LOG(ERROR) << msg;
          throw std::runtime_error(msg);
        }
      }
    }
    case VARCHAR:
    case STRING: {
      return new DStringType(new std::map<int, std::string>());
    }
    case DSTRING: {
      if (col.getDict() == nullptr) {
        auto msg = std::string{"Column[" + col.getName() +
                               "] with type DSTRING with no dictionary."};
        LOG(ERROR) << msg;
        throw std::runtime_error(msg);
      }
      // std::map<int, std::string> *d = new std::map<int, std::string>;
      return new DStringType(col.getDict());
    }
    case DATE: {
      switch (col.getSize()) {
        case 8:
          return new DateType();
        default: {
          auto msg = std::string{"Unknown date type of size: "} +
                     std::to_string((col.getSize()));
          LOG(ERROR) << msg;
          throw std::runtime_error(msg);
        }
      }
    }
    case MV:
    case META: {
      auto msg = std::string{"Illegal  type"};
      LOG(ERROR) << msg;
      throw std::runtime_error(msg);
    }
    default: {
      auto msg = std::string{"Unknown type"};
      LOG(ERROR) << msg;
      throw std::runtime_error(msg);
    }
  }
}

};  // namespace storage
