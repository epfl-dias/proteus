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

#include <olap/operators/gpu-aggr-mat-expr.hpp>
#include <olap/operators/gpu/gpu-materializer-expr.hpp>
#include <olap/plan/prepared-statement.hpp>
#include <olap/plugins/plugins.hpp>
#include <olap/routing/affinitizers.hpp>
#include <olap/routing/degree-of-parallelism.hpp>
#include <olap/routing/routing-policy-types.hpp>
#include <olap/values/types.hpp>
#include <platform/util/sort/sort-direction.hpp>
#include <utility>

class PreparedStatement;
class CatalogParser;
class ParallelContext;

class [[nodiscard]] pg {
 private:
  const std::string pgType;

 public:
  explicit pg(std::string pgType) : pgType(std::move(pgType)) {}

  [[nodiscard]] auto getType() const { return pgType; }
};

/**
 * @class RelBuilder
 * The RelBuilder class is used to construct PreparedStatements
 *
 * It is used to build a physical query plan of operators. The code for the plan
 * is then generated and compiled in the prepare method.
 * It is an abstraction/syntactic sugar for building a query plan in cpp by
 * hand
 */
class RelBuilder {
 private:
  ParallelContext* ctx;
  Operator* root;

  RelBuilder(ParallelContext* ctx, Operator* root);
  RelBuilder(const RelBuilder& builder, Operator* root);

  RelBuilder apply(Operator* op) const;

  [[nodiscard]] expressions::InputArgument getOutputArg() const;

  [[nodiscard]] expressions::InputArgument getOutputArgUnnested() const;

  static const RecordType& getRecordType(CatalogParser& catalog,
                                         std::string relName);
  static void setOIDType(CatalogParser& catalog, std::string relName,
                         ExpressionType* type);

  [[nodiscard]] std::string getModuleName() const;

  [[nodiscard]] Plugin* createPlugin(const RecordType& rec,
                                     const std::vector<RecordAttribute*>& projs,
                                     const std::string& pgType) const;

 private:
  RelBuilder();
  explicit RelBuilder(ParallelContext* ctx);

  friend class RelBuilderFactory;

  static void registerPlugin(const std::string& relName, Plugin* pg);

 protected:
  [[nodiscard]] static DegreeOfParallelism getDefaultDOP(DeviceType targetType);

 public:
  typedef std::function<std::vector<RecordAttribute*>(
      const expressions::InputArgument&)>
      MultiAttributeFactory;
  typedef std::function<expression_t(const expressions::InputArgument&)>
      ExpressionFactory;
  typedef std::function<std::vector<expression_t>(
      const expressions::InputArgument&)>
      MultiExpressionFactory;
  typedef std::function<std::optional<expression_t>(
      const expressions::InputArgument&)>
      OptionalExpressionFactory;

  RelBuilder scan(Plugin& pg) const;

  /**
   * All (global) plans start with scans.
   * Scans read the metadata from the catalogs and allow data format
   * plugins to inject their logic for interpreting data.
   * The latter happens by the plugin registering itself as the source
   * of tuples. Then during expression evaluation the source of each
   * input is invoked to determine how a field is processed.
   *
   * @param   relName   The name of the table we access. In general, this is
   *                    an arbitrary string, but for files fetched using the
   *                    same path across all servers, it can be that path.
   *                    The file name extension provides no information
   *                    the plugin that will be used for this table.
   *
   *                    For example, the name may end with .csv for a file
   *                    generated from a csv file as input, but containing
   *                    the binary data in columnar formats.
   *                    The "block" plugin is Proteus default plugin for
   *                    high-performance analytics and it signifies that the
   *                    data are in binary columnar format.
   *                    The plugin will use the relName as teh basis for
   *                    finding the rest of the column, but the actual files
   *                    will be "inputs/ssbm100/date.bin.d_datekey" etc.
   *
   * @param   relAttrs  The columns participating in this query.
   *                    For hierarchical data, this would usually be the
   *                    top level columns and unnest will follow for inner
   *                    attributes.
   *                    In general any plugin is allowed to interpret the
   *                    attribute names as it prefers.
   *
   * @param   catalog   Both the scan and the plugin may require to fetch
   *                    extra information from the catalog, as for example
   *                    the rest of the participating columns, statistics
   *                    and/or plugin-specific information (like which
   *                    global partitions is on this server for this rel)
   *
   * @param   pg        The plugin type that will be used for this relation.
   *                    Example plugins are "block", "json", "pm-csv",
   *                    "distributed-block".
   *                    The plugin name is used to locate the factory
   *                    function for instantiating the plugin.
   *                    If the name is a "::"-separated list of strings,
   *                    the last string is interpreted as the plugin type,
   *                    the string before the last is interpreted as the
   *                    dynamic library name and any additional strings
   *                    will cause the 1-to-(N-1) strings to be interpreted
   *                    as a path.
   *
   * @see Input plugins: Karpathiotakis et al, VLDB2016
   */
  RelBuilder scan(std::string relName, const std::vector<std::string>& relAttrs,
                  CatalogParser& catalog, const pg& pg) const;

  [[nodiscard]] RelBuilder scan(const RecordType& rec,
                                const std::vector<std::string>& relAttrs,
                                const std::string& pg) const;

  [[nodiscard]] RelBuilder scan(
      const std::vector<
          std::pair<RecordAttribute*, std::shared_ptr<proteus_any_vector>>>&
          data) const;

  template <typename Tplugin>
  [[deprecated]] RelBuilder scan(std::string relName,
                                 std::vector<std::string> relAttrs,
                                 CatalogParser& catalog) const {
    const RecordType& recType_ = getRecordType(catalog, relName);

    std::vector<RecordAttribute*> v;

    for (const auto& s : relAttrs) {
      auto attr = (new RecordType(recType_.getArgs()))->getArg(s);
      assert(attr && "Unknown attribute");
      v.emplace_back(new RecordAttribute(*attr));
    }

    auto pg = new Tplugin(ctx, relName, recType_, v);

    setOIDType(catalog, relName, pg->getOIDType());
    registerPlugin(relName, pg);

    return scan(*pg);
  }

  [[nodiscard]] RelBuilder update(
      const std::function<expression_t(const expressions::InputArgument&)>&
          expr) const;

  [[nodiscard]] RelBuilder update(expression_t e) const;

  [[nodiscard]] RelBuilder print(pg pgType, std::string outrel) const;

  /**
   * Print the results using the given plugin.
   *
   * @param     pgType  Plugin to be used for output.
   */
  [[nodiscard]] RelBuilder print(pg pgType) const;

  [[deprecated]] RelBuilder print(std::function<std::vector<expression_t>(
                                      const expressions::InputArgument&)>
                                      exprs) const;

  RelBuilder print(std::function<std::vector<expression_t>(
                       const expressions::InputArgument&)>
                       exprs,
                   pg pgType) const;

  [[deprecated]] RelBuilder print(
      std::function<std::vector<expression_t>(const expressions::InputArgument&,
                                              std::string)>
          expr,
      std::string outrel, Plugin* pg = nullptr) const {
    const auto vec = expr(getOutputArg(), outrel);
#ifndef NDEBUG
    for (const auto& e : vec) {
      assert(e.isRegistered());
      assert(e.getRegisteredRelName() == outrel);
    }
#endif
    assert(pg == nullptr || outrel == pg->getName());
    return print(vec, pg);
  }

  template <typename Tplugin>
  [[deprecated]] RelBuilder print(
      std::function<std::vector<expression_t>(const expressions::InputArgument&,
                                              std::string)>
          expr,
      std::string outrel) const {
    auto pg = new Tplugin(ctx, outrel, {});
    registerPlugin(outrel, pg);

    return print(expr, outrel, pg);
  }

  [[deprecated]] RelBuilder print(
      std::function<std::vector<expression_t>(const expressions::InputArgument&,
                                              std::string)>
          expr,
      Plugin* pg) const {
    return print(std::move(expr), pg->getName(), pg);
  }

  [[deprecated]] RelBuilder print(
      std::function<std::vector<expression_t>(const expressions::InputArgument&,
                                              std::string)>
          expr) const {
    return print(std::move(expr), getModuleName());
  }

  /**
   * Pull data into the current NUMA node, if not already there.
   *
   * @param     slack   slack between the two ends of the memmove pipe,
   *                    used for load balancing
   *                    (limits the on-the-fly transfers)
   * @param     to      the type of the current NUMA node
   *
   * @see Chrysogelos et al, VLDB2019
   */
  [[nodiscard]] RelBuilder memmove(size_t slack, DeviceType to) const;

  [[nodiscard]] RelBuilder memmove_scaleout(size_t slack) const;

  [[nodiscard]] RelBuilder memmove_scaleout(const MultiAttributeFactory& attr,
                                            size_t slack) const;
  /**
   * Broadcast to all compute units.
   * @see Chrysogelos et al, VLDB2019
   */
  template <typename T>
  RelBuilder membrdcst(T attr, DegreeOfParallelism fanout, bool to_cpu,
                       bool always_share = false) const {
    return membrdcst(attr(getOutputArg()), fanout, to_cpu, always_share);
  }

  [[nodiscard]] RelBuilder membrdcst(DegreeOfParallelism fanout, bool to_cpu,
                                     bool always_share = false) const;

  [[nodiscard]] RelBuilder membrdcst(DeviceType target,
                                     bool always_share = false) const;

  template <typename T>
  RelBuilder membrdcst_scaleout(T attr, size_t fanout, bool to_cpu,
                                bool always_share = false) const {
    return membrdcst_scaleout(attr(getOutputArg()), fanout, to_cpu,
                              always_share);
  }

  [[nodiscard]] RelBuilder membrdcst_scaleout(size_t fanout, bool to_cpu,
                                              bool always_share = false) const;

  template <typename T, typename Thash>
  [[nodiscard]] RelBuilder router(
      T attr, Thash hash, DegreeOfParallelism fanout, size_t slack,
      RoutingPolicy p, DeviceType target,
      std::unique_ptr<Affinitizer> aff = nullptr) const {
    return router(attr(getOutputArg()), hash(getOutputArg()), fanout, slack, p,
                  target, std::move(aff));
  }

  [[nodiscard]] RelBuilder router_scaleout(
      const MultiAttributeFactory& attr, const OptionalExpressionFactory& hash,
      DegreeOfParallelism fanout, size_t slack, RoutingPolicy p,
      DeviceType targets) const;

  /**
   * Similar to router, but the scale-out case.
   *
   * RouterScaleout distributes the inputs to @p fanout executors (machines)
   *
   * How inputs are distributed across executors depends on the
   * RoutingPolicies.
   *
   * The RoutingPolicy::HASH_BASED specifies that we are going to send the
   * data to a specific node, given by the @p hash expression.
   *
   * Note that routers do not transfer data but only commands, such as
   * which data are going to be pulled from the other side.
   *
   * @param     hash    expression used for directing the routing, either to
   *                    specific machines through a constant, or using
   *                    a data property (for example for hash-based routing)
   * @param     fanout  Number of target executors
   * @param     slack   Slack in each pipe, used for load-balancing and
   *                    backpressure. Limits the number of on-the-fly
   *                    requests.
   */
  [[nodiscard]] RelBuilder router_scaleout(
      const OptionalExpressionFactory& hash, DegreeOfParallelism fanout,
      size_t slack, RoutingPolicy p, DeviceType target) const;

  [[nodiscard]] RelBuilder router_scaleout(const ExpressionFactory& hash,
                                           DegreeOfParallelism fanout,
                                           size_t slack,
                                           DeviceType target) const;

  [[nodiscard]] RelBuilder router(
      ExpressionFactory hash, DegreeOfParallelism fanout, size_t slack,
      DeviceType target, std::unique_ptr<Affinitizer> aff = nullptr) const {
    return router(
        [&](const auto& arg) -> std::vector<RecordAttribute*> {
          std::vector<RecordAttribute*> attrs;
          for (const auto& attr : arg.getProjections()) {
            if (attr.getAttrName() == "__broadcastTarget") {
              continue;
            }
            attrs.emplace_back(new RecordAttribute{attr});
          }
          return attrs;
        },
        [&](const auto& arg) -> std::optional<expression_t> {
          return hash(arg);
        },
        fanout, slack, RoutingPolicy::HASH_BASED, target, std::move(aff));
  }

  [[nodiscard]] RelBuilder router_scaleout(DegreeOfParallelism fanout,
                                           size_t slack, RoutingPolicy p,
                                           DeviceType target) const;

  /**
   * Router distributes the inputs to @p fanout workers and handles the
   * affinity of the workers.
   *
   * How inputs are distributed across workers depends on the
   * RoutingPolicies.
   * Affinity of the workers depends on the target and the affinitization
   * policy. The default affinitization policy does a round-robin across
   * NUMA nodes.
   *
   * @see Chrysogelos et al, VLDB2019
   */
  [[nodiscard]] RelBuilder router(
      DegreeOfParallelism fanout, size_t slack, RoutingPolicy p,
      DeviceType target, std::unique_ptr<Affinitizer> aff = nullptr) const;

  [[nodiscard]] RelBuilder router(
      size_t slack, RoutingPolicy p, DeviceType target,
      std::unique_ptr<Affinitizer> aff = nullptr) const;

  /**
   * Splits the data flow into multiple ones to allow different alternatives.
   * The policy will determine which alternative is selected for each input, in
   * combination with the target type and the affinitization policies
   *
   * The slack provides load balancing and backpressure.
   *
   * @param     alternatives    number of output flows
   * @param     slack           max number of on-the-fly
   *                            tasks on each flow
   * @param     p               policy determine how
   *                            the target flow is selected
   */
  [[nodiscard]] RelBuilder split(
      size_t alternatives, size_t slack, RoutingPolicy p,
      std::unique_ptr<Affinitizer> aff = nullptr) const;

  /**
   * Union the items from the current flow and the others
   *
   * @param     others  flows to unify with current one
   */
  [[nodiscard]] RelBuilder unionAll(
      const std::vector<RelBuilder>& others) const;

  [[nodiscard]] RelBuilder to_gpu() const;

  [[nodiscard]] RelBuilder to_cpu(gran_t granularity = gran_t::THREAD,
                                  size_t size = 1024 * 1024 / 4) const;

  [[nodiscard]] RelBuilder unpack() const;

  [[nodiscard]] RelBuilder bloomfilter_repack(
      std::function<expression_t(expressions::InputArgument)> pred,
      size_t filterSize, uint64_t bloomId) {
    auto arg = getOutputArg();
    std::vector<expression_t> attrs;
    for (const auto& attr : arg.getProjections()) {
      attrs.emplace_back(arg[attr]);
    }
    return bloomfilter_repack(pred(arg), attrs, filterSize, bloomId);
  }

  /**
   * Filters tuples using a predicate.
   *
   * Yields each tuple for which the predicate is true, to the next
   * operator.
   *
   * @param     pred    Predicate used for filtering.
   */
  template <typename T>
  RelBuilder filter(T pred) const {
    return filter(pred(getOutputArg()));
  }

  RelBuilder bloomfilter_probe(
      std::function<expression_t(expressions::InputArgument)> pred,
      size_t filterSize, uint64_t bloomId) const {
    if (true)
      return bloomfilter_probe(pred(getOutputArg()), filterSize, bloomId);
    return *this;
  }

  RelBuilder bloomfilter_build(
      std::function<expression_t(expressions::InputArgument)> pred,
      size_t filterSize, uint64_t bloomId) const {
    return bloomfilter_build(pred(getOutputArg()), filterSize, bloomId);
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

  [[nodiscard]] RelBuilder hintRowCount(double expectedRowCount) const;

  template <typename Tbk, typename Tpk>
  RelBuilder join(RelBuilder build, Tbk build_k, Tpk probe_k) const {
    return join(build, build_k(build.getOutputArg()), probe_k(getOutputArg()));
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
  [[nodiscard]] [[nodiscard]] RelBuilder pack(T expr, Th hashExpr,
                                              size_t numOfBuckets) const {
    return pack(expr(getOutputArg()), hashExpr(getOutputArg()), numOfBuckets);
  }

  [[nodiscard]] [[nodiscard]] RelBuilder pack(ExpressionFactory hashExpr,
                                              size_t numOfBuckets) const;

  /**
   * Pack tuples to blocks
   *
   * @see Chrysogelos et al, VLDB2019
   */
  [[nodiscard]] RelBuilder pack() const;

  /**
   * Project (calculate) tuple-wise expressions.
   *
   * @param     expr    List of expressions to evaluate.
   *
   * @note  expressions usually perserve types and they look like c++
   *        default expression in terms of type convergences, without
   *        c++'s defeult automatic casts.
   */
  template <typename T>
  RelBuilder project(T expr) const {
    return project(expr(getOutputArg()));
  }

  /**
   * Performs simple aggregations with a single group.
   *
   * @param     expr    Inputs to the (monoid) aggregates
   * @param     accs    Aggregate type (ie. SUM for a summation)
   *                    The attribute name of an aggregate is the same as
   *                    the input name.
   */
  [[nodiscard]] RelBuilder reduce(const MultiExpressionFactory& expr,
                                  const vector<Monoid>& accs) const {
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
  RelBuilder unnest(T expr) const {
    return unnest(expr(getOutputArg()));
  }

  Operator* operator->() { return root; }
  Operator* operator->() const { return root; }

  /**
   * Compile the plan
   * @return a PreparedStatement which can then be executed
   */
  PreparedStatement prepare();

  [[nodiscard]] bool isPacked() const;

 private:
  [[nodiscard]] RelBuilder memmove(const vector<RecordAttribute*>& wantedFields,
                                   size_t slack, DeviceType to) const;

  [[nodiscard]] RelBuilder memmove_scaleout(
      const vector<RecordAttribute*>& wantedFields, size_t slack) const;

  [[nodiscard]] RelBuilder to_gpu(
      const vector<RecordAttribute*>& wantedFields) const;

  [[nodiscard]] RelBuilder to_cpu(const vector<RecordAttribute*>& wantedFields,
                                  gran_t granularity = gran_t::THREAD,
                                  size_t size = 1024 * 1024 / 4) const;

  [[nodiscard]] RelBuilder unpack(
      const vector<expression_t>& projections) const;

  [[nodiscard]] RelBuilder unpack(const vector<expression_t>& projections,
                                  gran_t granularity) const;

  [[nodiscard]] RelBuilder pack(const vector<expression_t>& projections,
                                expression_t hashExpr,
                                size_t numOfBuckets) const;

  [[nodiscard]] RelBuilder filter(expression_t pred) const;

  [[nodiscard]] RelBuilder bloomfilter_probe(expression_t pred,
                                             size_t filterSize,
                                             uint64_t bloomId) const;
  [[nodiscard]] RelBuilder bloomfilter_build(expression_t pred,
                                             size_t filterSize,
                                             uint64_t bloomId) const;

  [[nodiscard]] RelBuilder bloomfilter_repack(expression_t pred,
                                              std::vector<expression_t> attr,
                                              size_t filterSize,
                                              uint64_t bloomId) const;

  [[nodiscard]] RelBuilder project(const vector<expression_t>& e) const;

  [[nodiscard]] RelBuilder reduce(const vector<expression_t>& e,
                                  const vector<Monoid>& accs) const;

  [[nodiscard]] RelBuilder groupby(const std::vector<expression_t>& e,
                                   const std::vector<GpuAggrMatExpr>& agg_exprs,
                                   size_t hash_bits, size_t maxInputSize) const;

  [[nodiscard]] RelBuilder sort(const vector<expression_t>& orderByFields,
                                const vector<direction>& dirs) const;

  RelBuilder print(const vector<expression_t>& e, Plugin* pg,
                   bool may_overwrite = false) const;

  RelBuilder print(std::function<std::vector<expression_t>(
                       const expressions::InputArgument&)>
                       exprs,
                   pg pgType, std::string outrel) const;

  [[nodiscard]] RelBuilder unnest(expression_t e) const;

  [[nodiscard]] RelBuilder router(const vector<RecordAttribute*>& wantedFields,
                                  std::optional<expression_t> hash,
                                  DegreeOfParallelism fanout, size_t slack,
                                  RoutingPolicy p, DeviceType target,
                                  std::unique_ptr<Affinitizer> aff) const;

  [[nodiscard]] RelBuilder router_scaleout(
      const vector<RecordAttribute*>& wantedFields,
      std::optional<expression_t> hash, DegreeOfParallelism fanout,
      size_t slack, RoutingPolicy p, DeviceType cpu_targets) const;

  [[nodiscard]] RelBuilder unionAll(
      const std::vector<RelBuilder>& children,
      const vector<RecordAttribute*>& wantedFields) const;

  [[nodiscard]] RelBuilder membrdcst(
      const vector<RecordAttribute*>& wantedFields, DegreeOfParallelism fanout,
      bool to_cpu, bool always_share = false) const;

  [[nodiscard]] RelBuilder join(RelBuilder build, expression_t build_k,
                                const std::vector<GpuMatExpr>& build_e,
                                const std::vector<size_t>& build_w,
                                expression_t probe_k,
                                const std::vector<GpuMatExpr>& probe_e,
                                const std::vector<size_t>& probe_w,
                                int hash_bits, size_t maxBuildInputSize) const;

  [[nodiscard]] RelBuilder membrdcst_scaleout(
      const vector<RecordAttribute*>& wantedFields, size_t fanout, bool to_cpu,
      bool always_share = false) const;

  [[nodiscard]] RelBuilder join(RelBuilder build, expression_t build_k,
                                expression_t probe_k, int hash_bits,
                                size_t maxBuildInputSize) const;

  [[nodiscard]] RelBuilder join(RelBuilder build, expression_t build_k,
                                expression_t probe_k) const;

  // Helpers

  template <typename T>
  [[nodiscard]] RelBuilder unpack(T expr) const {
    return unpack(expr(getOutputArg()));
  }

  friend class PlanExecutor;
};

std::ostream& operator<<(std::ostream& out, const RelBuilder& builder);

#endif /* RELBUILDER_HPP_ */
