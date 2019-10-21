/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2018
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

#ifndef RELBUILDER_HPP_
#define RELBUILDER_HPP_

#include "operators/gpu/gpu-materializer-expr.hpp"
#include "operators/hash-group-by-chained.hpp"
#include "operators/operators.hpp"
#include "plan/prepared-statement.hpp"
#include "routing/routing-policy.hpp"
#include "topology/topology.hpp"
#include "util/parallel-context.hpp"
#include "util/sort/sort-direction.hpp"

class CatalogParser;

class RelBuilder {
 private:
  ParallelContext* ctx;
  Operator* root;

  RelBuilder(ParallelContext* ctx, Operator* root) : ctx(ctx), root(root) {}
  RelBuilder(const RelBuilder& builder, Operator* root)
      : RelBuilder(builder.ctx, root) {
    if (builder.root) builder.root->setParent(root);
  }

  RelBuilder apply(Operator* op) const;

  expressions::InputArgument getOutputArg() const {
    return new RecordType(root->getRowType());
  }

  expressions::InputArgument getOutputArgUnnested() const {
    auto args = root->getRowType().getArgs();
    std::vector<RecordAttribute*> attrs;
    attrs.reserve(args.size());
    for (const auto& arg : args) {
      const BlockType* block =
          dynamic_cast<const BlockType*>(arg->getOriginalType());
      if (block) {
        attrs.emplace_back(
            new RecordAttribute(arg->getAttrNo(), arg->getRelationName(),
                                arg->getAttrName(), &(block->getNestedType())));
      } else {
        attrs.emplace_back(arg);
      }
    }
    return new RecordType(attrs);
  }

  const RecordType& getRecordType(CatalogParser& catalog,
                                  std::string relName) const;
  void setOIDType(CatalogParser& catalog, std::string relName,
                  ExpressionType* type) const;

 public:
  RelBuilder() : RelBuilder(new ParallelContext("main", false)) {}
  RelBuilder(ParallelContext* ctx) : RelBuilder(ctx, nullptr) {}

  RelBuilder scan(Plugin& pg) const;

  template <typename Tplugin>
  RelBuilder scan(std::string relName, std::vector<std::string> relAttrs,
                  CatalogParser& catalog) const {
    const RecordType& recType_ = getRecordType(catalog, relName);

    std::vector<RecordAttribute*> v;

    for (const auto& s : relAttrs) {
      auto attr = (new RecordType(recType_.getArgs()))->getArg(s);
      v.emplace_back(new RecordAttribute(*attr));
    }

    Plugin* pg = new Tplugin(ctx, relName, recType_, v);

    setOIDType(catalog, relName, pg->getOIDType());
    Catalog::getInstance().registerPlugin(relName, pg);

    return scan(*pg);
  }

  RelBuilder memmove(size_t slack, bool to_cpu) const;

  template <typename T>
  RelBuilder memmove(T attr, size_t slack, bool to_cpu) const {
    return memmove(attr(getOutputArg()), slack, to_cpu);
  }

  template <typename T>
  RelBuilder membrdcst(T attr, DegreeOfParallelism fanout, bool to_cpu,
                       bool always_share = false) const {
    return membrdcst(attr(getOutputArg()), fanout, to_cpu, always_share);
  }

  RelBuilder membrdcst(DegreeOfParallelism fanout, bool to_cpu,
                       bool always_share = false) const;

  template <typename T, typename Thash>
  RelBuilder router(T attr, Thash hash, DegreeOfParallelism fanout,
                    size_t slack, RoutingPolicy p, DeviceType target,
                    std::unique_ptr<Affinitizer> aff = nullptr) const {
    return router(attr(getOutputArg()), hash(getOutputArg()), fanout, slack, p,
                  target, std::move(aff));
  }

  template <typename Thash>
  RelBuilder router(Thash hash, DegreeOfParallelism fanout, size_t slack,
                    RoutingPolicy p, DeviceType target,
                    std::unique_ptr<Affinitizer> aff = nullptr) const {
    return router(
        [&](const auto& arg) -> std::vector<RecordAttribute*> {
          std::vector<RecordAttribute*> attrs;
          for (const auto& attr : arg.getProjections()) {
            if (p == RoutingPolicy::HASH_BASED &&
                attr.getAttrName() == "__broadcastTarget") {
              continue;
            }
            attrs.emplace_back(new RecordAttribute{attr});
          }
          return attrs;
        },
        hash, fanout, slack, p, target, std::move(aff));
  }

  RelBuilder router(DegreeOfParallelism fanout, size_t slack, RoutingPolicy p,
                    DeviceType target,
                    std::unique_ptr<Affinitizer> aff = nullptr) const {
    assert(p != RoutingPolicy::HASH_BASED);
    return router(
        [&](const auto& arg) -> std::optional<expression_t> {
          return std::nullopt;
        },
        fanout, slack, p, target, std::move(aff));
  }

  RelBuilder router(size_t slack, RoutingPolicy p, DeviceType target,
                    std::unique_ptr<Affinitizer> aff = nullptr) const {
    size_t dop = (target == DeviceType::CPU)
                     ? topology::getInstance().getCoreCount()
                     : topology::getInstance().getGpuCount();
    return router(DegreeOfParallelism{dop}, slack, p, target, std::move(aff));
  }

  RelBuilder to_gpu() const;

  template <typename T>
  RelBuilder to_gpu(T attr) const {
    return to_gpu(attr(getOutputArg()));
  }

  RelBuilder to_cpu(gran_t granularity = gran_t::THREAD,
                    size_t size = 1024 * 1024 / 4) const;

  template <typename T>
  RelBuilder to_cpu(T attr, gran_t granularity = gran_t::THREAD,
                    size_t size = 1024 * 1024 / 4) const {
    return to_cpu(attr(getOutputArg()), granularity, size);
  }

  template <typename T>
  RelBuilder unpack(T expr) const {
    return unpack(expr(getOutputArgUnnested()));
  }

  RelBuilder unpack() const {
    return unpack([&](const auto& arg) -> std::vector<expression_t> {
      std::vector<expression_t> attrs;
      for (const auto& attr : arg.getProjections()) {
        attrs.emplace_back(arg[attr]);
      }
      return attrs;
    });
  }

  template <typename T>
  RelBuilder filter(T pred) const {
    return filter(pred(getOutputArg()));
  }

  template <typename Tbk, typename Tbe, typename Tpk, typename Tpe>
  RelBuilder join(RelBuilder build, Tbk build_k, Tbe build_e,
                  std::vector<size_t> build_w, Tpk probe_k, Tpe probe_e,
                  std::vector<size_t> probe_w, int hash_bits,
                  size_t maxBuildInputSize) const {
    return join(build, build_k(build.getOutputArg()),
                build_e(build.getOutputArg()), build_w, probe_k(getOutputArg()),
                probe_e(getOutputArg()), probe_w, hash_bits, maxBuildInputSize);
  }

  template <typename Tbk, typename Tpk>
  RelBuilder join(RelBuilder build, Tbk build_k, Tpk probe_k, int hash_bits,
                  size_t maxBuildInputSize) const {
    return join(build, build_k(build.getOutputArg()), probe_k(getOutputArg()),
                hash_bits, maxBuildInputSize);
  }

  template <typename T>
  RelBuilder unpack(T expr, gran_t granularity) const {
    return unpack(expr(getOutputArgUnnested()), granularity);
  }

  template <typename T, typename Th>
  RelBuilder pack(T expr, Th hashExpr, size_t numOfBuckets) const {
    return pack(expr(getOutputArg()), hashExpr(getOutputArg()), numOfBuckets);
  }

  template <typename T>
  RelBuilder pack(T expr) const {
    return pack(
        expr, [](const auto& arg) { return expression_t{0}; }, 1);
  }

  RelBuilder pack() const {
    return pack([&](const auto& arg) -> std::vector<expression_t> {
      std::vector<expression_t> attrs;
      for (const auto& attr : arg.getProjections()) {
        attrs.emplace_back(arg[attr]);
      }
      return attrs;
    });
  }

  template <typename T>
  RelBuilder project(T expr) const {
    return project(expr(getOutputArg()));
  }

  template <typename T>
  RelBuilder reduce(T expr, const vector<Monoid>& accs) const {
    return reduce(expr(getOutputArg()), accs);
  }

  template <typename Tk, typename Te>
  RelBuilder groupby(Tk k, Te e, size_t hash_bits, size_t maxInputSize) const {
    return groupby(k(getOutputArg()), e(getOutputArg()), hash_bits,
                   maxInputSize);
  }

  template <typename T>
  RelBuilder sort(T e, const vector<direction>& dirs) const {
    return sort(e(getOutputArg()), dirs);
  }

  template <typename T>
  RelBuilder print(T expr, std::string outrel) const {
    const auto vec = expr(getOutputArg(), outrel);
#ifndef NDEBUG
    for (const auto& e : vec) {
      assert(e.isRegistered());
      assert(e.getRegisteredRelName() == outrel);
    }
#endif
    return print(vec);
  }

  template <typename T>
  RelBuilder print(T expr) const {
    return print(expr, ctx->getModuleName());
  }

  template <typename T>
  RelBuilder unnest(T expr) const {
    return unnest(expr(getOutputArg()));
  }

  Operator* operator->() { return root; }

  PreparedStatement prepare();

 private:
  RelBuilder memmove(const vector<RecordAttribute*>& wantedFields, size_t slack,
                     bool to_cpu) const;

  RelBuilder to_gpu(const vector<RecordAttribute*>& wantedFields) const;

  RelBuilder to_cpu(const vector<RecordAttribute*>& wantedFields,
                    gran_t granularity = gran_t::THREAD,
                    size_t size = 1024 * 1024 / 4) const;

  RelBuilder unpack(const vector<expression_t>& projections) const;

  RelBuilder unpack(const vector<expression_t>& projections,
                    gran_t granularity) const;

  RelBuilder pack(const vector<expression_t>& projections,
                  expression_t hashExpr, size_t numOfBuckets) const;

  RelBuilder filter(expression_t pred) const;

  RelBuilder project(const vector<expression_t>& e) const;

  RelBuilder reduce(const vector<expression_t>& e,
                    const vector<Monoid>& accs) const;

  RelBuilder groupby(const std::vector<expression_t>& e,
                     const std::vector<GpuAggrMatExpr>& agg_exprs,
                     size_t hash_bits, size_t maxInputSize) const;

  RelBuilder sort(const vector<expression_t>& orderByFields,
                  const vector<direction>& dirs) const;

  RelBuilder print(const vector<expression_t>& e) const;

  RelBuilder unnest(expression_t e) const;

  RelBuilder router(const vector<RecordAttribute*>& wantedFields,
                    std::optional<expression_t> hash,
                    DegreeOfParallelism fanout, size_t slack, RoutingPolicy p,
                    DeviceType target, std::unique_ptr<Affinitizer> aff) const;

  RelBuilder membrdcst(const vector<RecordAttribute*>& wantedFields,
                       DegreeOfParallelism fanout, bool to_cpu,
                       bool always_share = false) const;

  RelBuilder join(RelBuilder build, expression_t build_k,
                  const std::vector<GpuMatExpr>& build_e,
                  const std::vector<size_t>& build_w, expression_t probe_k,
                  const std::vector<GpuMatExpr>& probe_e,
                  const std::vector<size_t>& probe_w, int hash_bits,
                  size_t maxBuildInputSize) const;

  RelBuilder join(RelBuilder build, expression_t build_k, expression_t probe_k,
                  int hash_bits, size_t maxBuildInputSize) const;
};

#endif /* RELBUILDER_HPP_ */
