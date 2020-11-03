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

#ifndef PROTEUS_INPUT_PREFIX_QUERY_SHAPER_HPP
#define PROTEUS_INPUT_PREFIX_QUERY_SHAPER_HPP

#include <query-shaping/query-shaper.hpp>

namespace proteus {
class InputPrefixQueryShaper : public QueryShaper {
  std::string base_path;

 public:
  explicit InputPrefixQueryShaper(std::string base_path);

  std::string getRelName(const std::string& base) override;

  int sf() override;

  [[nodiscard]] pg getPlugin() const override;

  [[nodiscard]] DeviceType getDevice() override;
};
}  // namespace proteus

#endif /* PROTEUS_INPUT_PREFIX_QUERY_SHAPER_HPP */
