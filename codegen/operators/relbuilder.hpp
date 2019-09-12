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
#include "operators/operators.hpp"
#include "plan/prepared-statement.hpp"
#include "util/parallel-context.hpp"

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

 public:
  RelBuilder() : RelBuilder(new ParallelContext("main", false)) {}
  RelBuilder(ParallelContext* ctx) : RelBuilder(ctx, nullptr) {}

  RelBuilder scan(Plugin& pg) const;

  template <typename T>
  RelBuilder memmove(T attr, size_t slack, bool to_cpu) const {
    return memmove(attr(getOutputArg()), slack, to_cpu);
  }

  template <typename T>
  RelBuilder membrdcst(T attr, size_t fanout, bool to_cpu,
                       bool always_share = false) const {
    return membrdcst(attr(getOutputArg()), fanout, to_cpu, always_share);
  }

  template <typename T, typename Thash>
  RelBuilder router(T attr, Thash hash, size_t fanout, size_t fanin,
                    size_t slack, bool numa_local = true,
                    bool rand_local_cpu = false, bool cpu_targets = false,
                    int numa_socket_id = -1) const {
    return router(attr(getOutputArg()), hash(getOutputArg()), fanout, fanin,
                  slack, numa_local, rand_local_cpu, cpu_targets,
                  numa_socket_id);
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
    return unpack(expr(getOutputArg()));
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

  template <typename T>
  RelBuilder unpack(T expr, gran_t granularity) const {
    return unpack(expr(getOutputArg()), granularity);
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

  template <typename T>
  RelBuilder reduce(T expr, const vector<Monoid>& accs) const {
    return reduce(expr(getOutputArg()), accs);
  }

  template <typename T>
  RelBuilder print(T expr) const {
    return print(expr(getOutputArg()));
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

  RelBuilder reduce(const vector<expression_t>& e,
                    const vector<Monoid>& accs) const;

  RelBuilder print(const vector<expression_t>& e) const;

  RelBuilder unnest(expression_t e) const;

  RelBuilder router(const vector<RecordAttribute*>& wantedFields,
                    std::optional<expression_t> hash, size_t fanout,
                    size_t fanin, size_t slack, bool numa_local,
                    bool rand_local_cpu, bool cpu_targets,
                    int numa_socket_id) const;

  RelBuilder membrdcst(const vector<RecordAttribute*>& wantedFields,
                       size_t fanout, bool to_cpu,
                       bool always_share = false) const;

  RelBuilder join(RelBuilder build, expression_t build_k,
                  const std::vector<GpuMatExpr>& build_e,
                  const std::vector<size_t>& build_w, expression_t probe_k,
                  const std::vector<GpuMatExpr>& probe_e,
                  const std::vector<size_t>& probe_w, int hash_bits,
                  size_t maxBuildInputSize) const;
};

#endif /* RELBUILDER_HPP_ */
