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

#ifndef PROTEUS_SCALE_OUT_QUERY_SHAPER_HPP
#define PROTEUS_SCALE_OUT_QUERY_SHAPER_HPP

#include <query-shaping/input-prefix-query-shaper.hpp>

namespace proteus {

class ScaleOutQueryShaper : public proteus::InputPrefixQueryShaper {
 public:
  explicit ScaleOutQueryShaper(std::string base_path,
                               decltype(input_sizes) input_sizes);

  [[nodiscard]] pg getPlugin() const override;

  [[nodiscard]] DeviceType getDevice() override;
  [[nodiscard]] int getSlack() override;

  [[nodiscard]] RelBuilder getBuilder() const override;

  [[nodiscard]] virtual DegreeOfParallelism getServerDOP();

  [[nodiscard]] RelBuilder distribute_build(RelBuilder input) override;
  [[nodiscard]] RelBuilder distribute_probe(RelBuilder input) override;

 protected:
  [[nodiscard]] RelBuilder collect_unpacked(RelBuilder input) override;
  [[nodiscard]] RelBuilder collect_packed(RelBuilder input) override;

  double getRowHint(const std::string &relName) override;

  [[nodiscard]] virtual RelBuilder distribute_probe_interserver(
      RelBuilder input);
  [[nodiscard]] virtual RelBuilder distribute_probe_intraserver(
      RelBuilder input);
};

}  // namespace proteus

#endif /* PROTEUS_SCALE_OUT_QUERY_SHAPER_HPP */
