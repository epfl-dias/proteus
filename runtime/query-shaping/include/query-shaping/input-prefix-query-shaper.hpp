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
class InputPrefixQueryShaper : public QueryShaperControlMoves {
 private:
  size_t sf_;

 protected:
  std::string base_path;

  std::map<std::string, std::function<double(InputPrefixQueryShaper&)>>
      input_sizes;
  virtual double getRowHint(const std::string& relName);

 public:
  /**
   * @param base_path The base system path to use to locate input data files
   * @param input_sizes A function which produces a map from relation name to
   * number of tuples in the relation. The function takes a
   * InputPrefixQueryShaper argument which can be used for additional context to
   * calculate the row counts. For example the function may call sf() to learn
   * the scale factor
   * @param allowMoves Whether to insert mem-move operators into the plan
   * @param slack see @see QueryShaper
   */
  explicit InputPrefixQueryShaper(std::string base_path,
                                  decltype(input_sizes) input_sizes = {},
                                  bool allowMoves = true, size_t slack = 64);

  RelBuilder scan(const std::string& relName,
                  std::initializer_list<std::string> relAttrs) override;

  std::string getRelName(const std::string& base) override;

  size_t sf() override;

  [[nodiscard]] pg getPlugin() const override;

  [[nodiscard]] DeviceType getDevice() override;
};
}  // namespace proteus

#endif /* PROTEUS_INPUT_PREFIX_QUERY_SHAPER_HPP */
