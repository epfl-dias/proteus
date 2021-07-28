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

#include <olap/plan/catalog-parser.hpp>
#include <query-shaping/input-prefix-query-shaper.hpp>

namespace proteus {
InputPrefixQueryShaper::InputPrefixQueryShaper(
    std::string base_path, decltype(input_sizes) input_sizes, bool allowMoves,
    size_t slack)
    : QueryShaperControlMoves(allowMoves, slack),
      base_path(std::move(base_path)),
      input_sizes(std::move(input_sizes)),
      sf_(100) {
  try {
    sf_ = this->input_sizes.at("sf")(*this);
  } catch (const std::out_of_range&) {
  }
}

std::string InputPrefixQueryShaper::getRelName(const std::string& base) {
  return base_path + base + ".csv";
}

double InputPrefixQueryShaper::getRowHint(const std::string& relName) {
  return input_sizes.at(relName)(*this);
}

RelBuilder InputPrefixQueryShaper::scan(
    const std::string& relName, std::initializer_list<std::string> relAttrs) {
  auto rel = getBuilder().scan(getRelName(relName), relAttrs,
                               CatalogParser::getInstance(), getPlugin());
  //  try {
  rel = rel.hintRowCount(getRowHint(relName));
  //  } catch (const std::out_of_range&) {}

  return rel;
}

size_t InputPrefixQueryShaper::sf() { return sf_; }

[[nodiscard]] pg InputPrefixQueryShaper::getPlugin() const {
  return pg{"block"};
}

[[nodiscard]] DeviceType InputPrefixQueryShaper::getDevice() {
  return DeviceType::GPU;
}
}  // namespace proteus
