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

#ifndef PROTEUS_QUERY_SHAPER_HPP
#define PROTEUS_QUERY_SHAPER_HPP

#include <olap/operators/relbuilder.hpp>
#include <olap/plugins/binary-block-plugin.hpp>

namespace proteus {
class QueryShaper {
 protected:
  std::string query;

  [[nodiscard]] virtual std::string getRelName(const std::string &base);
  [[nodiscard]] virtual RelBuilder getBuilder() const;
  [[nodiscard]] virtual pg getPlugin() const;

  [[nodiscard]] virtual std::unique_ptr<Affinitizer> getAffinitizer();
  [[nodiscard]] virtual std::unique_ptr<Affinitizer> getAffinitizerReduce();

  [[nodiscard]] virtual DeviceType getDevice();
  [[nodiscard]] virtual int getSlack();
  [[nodiscard]] virtual int getSlackReduce();
  [[nodiscard]] virtual bool doMove();

  [[nodiscard]] virtual DegreeOfParallelism getDOP();

 public:
  virtual void setQueryName(std::string name);
  virtual ~QueryShaper() = default;

  [[nodiscard]] virtual int sf();

  [[nodiscard]] virtual RelBuilder distribute_probe(RelBuilder input);
  [[nodiscard]] virtual RelBuilder distribute_build(RelBuilder input);
  [[nodiscard]] virtual RelBuilder collect_unpacked(RelBuilder input);
  [[nodiscard]] virtual RelBuilder collect(RelBuilder input);
  [[nodiscard]] virtual RelBuilder scan(
      const std::string &relName, std::initializer_list<std::string> relAttrs);
};

class QueryShaperControlMoves : public QueryShaper {
  bool allow_moves;

 public:
  explicit QueryShaperControlMoves(bool allow_moves);

  [[nodiscard]] bool doMove() override;
  void setQueryName(std::string name) override;
};
}  // namespace proteus

#endif /* PROTEUS_QUERY_SHAPER_HPP */
