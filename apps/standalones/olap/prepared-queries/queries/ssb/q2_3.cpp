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

#include <ssb/query.hpp>

constexpr auto query = "ssb100_Q2_3";

class DanglingAttr;

class RelWithAttributes;

class rel {
 private:
  std::string relName;

 public:
  explicit rel(std::string relName) : relName(std::move(relName)) {}

  RelWithAttributes operator()(std::initializer_list<DanglingAttr>);
  template <typename... T>
  RelWithAttributes operator()(T... x);

  operator std::string() const { return relName; }
};

class DanglingAttr {
 private:
  std::string attrName;
  ExpressionType *type;

 protected:
  explicit DanglingAttr(std::string attrName, ExpressionType *type)
      : attrName(std::move(attrName)), type(type) {}

  friend DanglingAttr attr(std::string attrName, ExpressionType *type);

 public:
  operator std::string() const { return attrName; }

  auto getName() const { return attrName; }
  auto getType() const { return type; }
};

DanglingAttr attr(std::string attrName, ExpressionType *type) {
  return DanglingAttr(std::move(attrName), type);
}

DanglingAttr Int(std::string attrName) {
  return attr(std::move(attrName), new IntType());
}

DanglingAttr Float(std::string attrName) {
  return attr(std::move(attrName), new FloatType());
}

DanglingAttr String(std::string attrName) {
  return attr(std::move(attrName), new StringType());
}

class RelWithAttributes {
 private:
  rel r;
  std::vector<DanglingAttr> attrs;

 private:
  RelWithAttributes(rel relName, decltype(attrs) attrs)
      : r(std::move(relName)), attrs(std::move(attrs)) {}

  friend class rel;

 public:
  operator RecordType() const {
    std::vector<RecordAttribute *> recattrs;
    recattrs.reserve(attrs.size());
    for (const auto &da : attrs) {
      LOG(INFO) << recattrs.size() << ((std::string)r) << da.getName()
                << *(da.getType());
      recattrs.emplace_back(new RecordAttribute(recattrs.size() + 1, r,
                                                da.getName(), da.getType()));
    }
    return {recattrs};
  }
};

RelWithAttributes rel::operator()(std::initializer_list<DanglingAttr> attrs) {
  return {*this, attrs};
}

template <typename... T>
RelWithAttributes rel::operator()(T... x) {
  return (*this)({x...});
}

static auto lineorder2 =
    rel("inputs/ssbm100/lineorder.csv")(Int("lo_partkey"), Int("lo_suppkey"),
                                        Int("lo_orderdate"), Int("lo_revenue"));

PreparedStatement ssb::Query::prepare23(proteus::QueryShaper &morph) {
  morph.setQueryName(query);

  auto rel23990 =
      morph.distribute_build(morph.scan("date", {"d_datekey", "d_year"}))
          .unpack();

  auto rel23995 =
      morph.distribute_build(morph.scan("supplier", {"s_suppkey", "s_region"}))
          .unpack()
          .filter([&](const auto &arg) -> expression_t {
            return expressions::hint(eq(arg["s_region"], "EUROPE"),
                                     expressions::Selectivity{1.0 / 5});
          })
          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {arg["s_suppkey"]};
          });

  auto rel23999 =
      morph.distribute_build(morph.scan("part", {"p_partkey", "p_brand1"}))
          .unpack()
          .filter([&](const auto &arg) -> expression_t {
            return expressions::hint(eq(arg["p_brand1"], "MFGR#2239"),
                                     expressions::Selectivity{1.0 / 1000});
          });

  auto rel =
      morph
          .distribute_probe(morph.scan(
              "lineorder",
              {"lo_partkey", "lo_suppkey", "lo_orderdate", "lo_revenue"}))
          .unpack()
          .join(
              rel23999,
              [&](const auto &build_arg) -> expression_t {
                return build_arg["p_partkey"];
              },
              [&](const auto &probe_arg) -> expression_t {
                return probe_arg["lo_partkey"];
              })
          .join(
              rel23995,
              [&](const auto &build_arg) -> expression_t {
                return build_arg["s_suppkey"];
              },
              [&](const auto &probe_arg) -> expression_t {
                return probe_arg["lo_suppkey"];
              })
          .join(
              rel23990,
              [&](const auto &build_arg) -> expression_t {
                return build_arg["d_datekey"];
              },
              [&](const auto &probe_arg) -> expression_t {
                return probe_arg["lo_orderdate"];
              })
          .groupby(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {
                    arg["d_year"].as("PelagoAggregate#24009", "d_year"),
                    arg["p_brand1"].as("PelagoAggregate#24009", "p_brand1")};
              },
              [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                return {GpuAggrMatExpr{
                    (arg["lo_revenue"]).as("PelagoAggregate#24009", "EXPR$0"),
                    1, 0, SUM}};
              },
              4, 128 * 1024);
  rel = morph.collect_unpacked(rel)
            .groupby(
                [&](const auto &arg) -> std::vector<expression_t> {
                  return {arg["d_year"], arg["p_brand1"]};
                },
                [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                  return {GpuAggrMatExpr{arg["EXPR$0"], 1, 0, SUM}};
                },
                4, 128 * 1024)
            .project([&](const auto &arg) -> std::vector<expression_t> {
              return {arg["EXPR$0"], arg["d_year"], arg["p_brand1"]};
            })
            .sort(
                [&](const auto &arg) -> std::vector<expression_t> {
                  return {arg["EXPR$0"], arg["d_year"], arg["p_brand1"]};
                },
                {direction::NONE, direction::ASC, direction::ASC})
            .print(pg{"pm-csv"});
  return rel.prepare();
}
