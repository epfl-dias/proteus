/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2014
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

#include "plan-parser.hpp"

#include <dirent.h>
#include <dlfcn.h>

#include <cstdlib>

#include "lib/plugins/binary-col-plugin.hpp"
#include "lib/plugins/binary-row-plugin.hpp"
#include "lib/plugins/csv-plugin-pm.hpp"
#include "olap/plugins/binary-block-plugin.hpp"
#ifndef NCUDA
#include "lib/operators/cpu-to-gpu.hpp"
#include "lib/operators/gpu/gpu-hash-group-by-chained.hpp"
#include "lib/operators/gpu/gpu-hash-join-chained.hpp"
#include "lib/operators/gpu/gpu-hash-rearrange.hpp"
#include "lib/operators/gpu/gpu-partitioned-hash-join-chained.hpp"
#include "lib/operators/gpu/gpu-reduce.hpp"
#include "lib/operators/gpu/gpu-to-cpu.hpp"
#endif
#include <olap/routing/affinitization-factory.hpp>

#include "lib/operators/block-to-tuples.hpp"
#include "lib/operators/dict-scan.hpp"
#include "lib/operators/flush.hpp"
#include "lib/operators/hash-group-by-chained.hpp"
#include "lib/operators/hash-join-chained.hpp"
#include "lib/operators/hash-rearrange.hpp"
#include "lib/operators/mem-broadcast-device.hpp"
#include "lib/operators/mem-move-device.hpp"
#include "lib/operators/mem-move-local-to.hpp"
#include "lib/operators/outer-unnest.hpp"
#include "lib/operators/packet-zip.hpp"
#include "lib/operators/print.hpp"
#include "lib/operators/radix-join.hpp"
#include "lib/operators/radix-nest.hpp"
#include "lib/operators/reduce-opt.hpp"
#include "lib/operators/root.hpp"
#include "lib/operators/router.hpp"
#include "lib/operators/scan.hpp"
#include "lib/operators/select.hpp"
#include "lib/operators/split.hpp"
#include "lib/operators/unnest.hpp"
#include "olap/operators/gpu/gpu-materializer-expr.hpp"
#include "rapidjson/error/en.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

std::string hyphenatedPluginToCamel(const std::string &line) {
  size_t len = line.size();
  char conv[len + 1];
  bool active = true;
  int j = 0;
  for (int i = 0; line[i] != '\0'; i++) {
    if (std::isalpha(line[i])) {
      if (active) {
        conv[j] = std::toupper(line[i]);
        active = false;
      } else {
        conv[j] = std::tolower(line[i]);
      }
      j++;
    } else if (line[i] == '-') {
      active = true;
    }
  }
  conv[j] = '\0';
  return {conv};
}

PlanExecutor::PlanExecutor(const char *planPath, CatalogParser &cat,
                           const char *moduleName)
    : PlanExecutor(planPath, cat,
                   std::make_unique<ParserAffinitizationFactory>(),
                   moduleName) {}

PlanExecutor::PlanExecutor(
    const char *planPath, CatalogParser &cat,
    std::unique_ptr<ParserAffinitizationFactory> parFactory,
    const char *moduleName)
    : handle(dlopen(nullptr, 0)),
      moduleName(moduleName),
      catalogParser(cat),
      factory(moduleName),
      ctx(factory.getBuilder().ctx),
      parFactory(std::move(parFactory)) {
  // Input Path
  const char *nameJSON = planPath;
  // Prepare Input
  struct stat statbuf;
  stat(nameJSON, &statbuf);
  size_t fsize = statbuf.st_size;

  int fd = open(nameJSON, O_RDONLY);
  if (fd == -1) {
    throw runtime_error(string("json.open"));
  }

  const char *bufJSON =
      (const char *)mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
  if (bufJSON == MAP_FAILED) {
    const char *err = "json.mmap";
    LOG(ERROR) << err;
    throw runtime_error(err);
  }

  rapidjson::Document document;  // Default template parameter uses UTF8 and
                                 // MemoryPoolAllocator.
  auto &parsed = document.Parse(bufJSON);
  if (parsed.HasParseError()) {
    auto ok = (rapidjson::ParseResult)parsed;
    auto *err =
        "[PlanExecutor: ] Error parsing physical plan (JSON parsing error)";
    LOG(ERROR) << err << "JSON parse error: "
               << RAPIDJSON_NAMESPACE::GetParseError_En(ok.Code()) << " ("
               << ok.Offset() << ")";
    throw runtime_error(err);
  }

  /* Start plan traversal. */
  printf("\nParsing physical plan:\n");
  assert(document.IsObject());

  assert(document.HasMember("operator"));
  assert(document["operator"].IsString());
  printf("operator = %s\n", document["operator"].GetString());

  parsePlan(document, false);
}

void PlanExecutor::parsePlan(const rapidjson::Document &doc, bool execute) {
  splitOps.clear();
  auto root = parseOperator(doc);

  root->produce(ctx);
  ctx->prepareFunction(ctx->getGlobalFunction());

  if (execute) {
    Catalog &catalog = Catalog::getInstance();
    /* XXX Remove when testing caches (?) */
    catalog.clear();
  }
}

void PlanExecutor::cleanUp() {
  Catalog &catalog = Catalog::getInstance();
  /* XXX Remove when testing caches (?) */
  catalog.clear();

  /* Cleanup */
  for (const auto &currPg : activePlugins) {
    currPg->finish();
  }
}

class ParserAdapterAffinitizationFactory : public ParserAffinitizationFactory {
  std::unique_ptr<AffinitizationFactory> aff;

 public:
  ParserAdapterAffinitizationFactory(std::unique_ptr<AffinitizationFactory> aff)
      : aff(std::move(aff)) {}

  DegreeOfParallelism getDOP(DeviceType trgt, const rapidjson::Value &val,
                             RelBuilder &input) override {
    return aff->getDOP(trgt, input);
  }

  RoutingPolicy getRoutingPolicy(DeviceType trgt, const rapidjson::Value &val,
                                 RelBuilder &input) override {
    return aff->getRoutingPolicy(trgt, val.HasMember("target"), input);
  }

  std::unique_ptr<Affinitizer> getAffinitizer(DeviceType trgt,
                                              RoutingPolicy policy,
                                              const rapidjson::Value &val,
                                              RelBuilder &input) override {
    return aff->getAffinitizer(trgt, policy, input);
  }

  std::string getDynamicPgName(const std::string &relName) override {
    return aff->getDynamicPgName(relName);
  }
};

DegreeOfParallelism ParserAffinitizationFactory::getDOP(
    DeviceType trgt, const rapidjson::Value &val, RelBuilder &input) {
  auto attr = [&]() -> std::string {
    if (val.HasMember("numOfParents")) {
      return "numOfParents";
    } else {
      assert(val.HasMember("num_of_targets"));
      return "num_of_targets";
    }
  }();
  assert(val[attr.c_str()].IsUint64());
  return DegreeOfParallelism{val[attr.c_str()].GetUint64()};
}

RoutingPolicy ParserAffinitizationFactory::getRoutingPolicy(
    DeviceType trgt, const rapidjson::Value &val, RelBuilder &input) {
  if (val.HasMember("target")) return RoutingPolicy::HASH_BASED;

  bool numa_local = true;
  bool rand_local_cpu = false;
  if (val.HasMember("rand_local_cpu")) {
    assert(val["rand_local_cpu"].IsBool());
    rand_local_cpu = val["rand_local_cpu"].GetBool();
    numa_local = false;
  }

  if (val.HasMember("numa_local")) {
    assert(!rand_local_cpu);
    assert(numa_local);
    assert(val["numa_local"].IsBool());
    numa_local = val["numa_local"].GetBool();
  }

  return (numa_local || rand_local_cpu) ? RoutingPolicy::LOCAL
                                        : RoutingPolicy::RANDOM;
}

std::unique_ptr<Affinitizer> ParserAffinitizationFactory::getAffinitizer(
    DeviceType trgt, RoutingPolicy policy, const rapidjson::Value &val,
    RelBuilder &input) {
  return (trgt == DeviceType::GPU)
             ? (std::unique_ptr<Affinitizer>)std::make_unique<GPUAffinitizer>()
             : (std::unique_ptr<Affinitizer>)
                   std::make_unique<CpuCoreAffinitizer>(); /* FIXME */
}

std::string ParserAffinitizationFactory::getDynamicPgName(
    const std::string &relName) {
  return "binary";
}

PlanExecutor::PlanExecutor(const char *planPath, CatalogParser &cat,
                           std::unique_ptr<AffinitizationFactory> parFactory,
                           const char *moduleName)
    : PlanExecutor(planPath, cat,
                   std::make_unique<ParserAdapterAffinitizationFactory>(
                       std::move(parFactory)),
                   moduleName) {}

RelBuilder PlanExecutor::parseOperator(const rapidjson::Value &val) {
  const char *keyPg = "plugin";
  const char *keyOp = "operator";

  assert(val.HasMember(keyOp));
  assert(val[keyOp].IsString());
  const char *opName = val["operator"].GetString();

  Operator *newOp = nullptr;

  if (strcmp(opName, "reduce") == 0) {
    /* "Multi - reduce"! */
    /* parse operator input */
    auto childOp = parseOperator(val["input"]);

    /* get monoid(s) */
    assert(val.HasMember("accumulator"));
    assert(val["accumulator"].IsArray());
    vector<Monoid> accs;
    for (const auto &accm : val["accumulator"].GetArray()) {
      assert(accm.IsString());
      Monoid acc = parseAccumulator(accm.GetString(), childOp.getOutputArg());
      accs.push_back(acc);
    }

    /* parse filtering expression */
    assert(!val.HasMember("p") ||
           dynamic_cast<const expressions::BoolConstant &>(
               *parseExpression(val["p"], childOp.getOutputArg())
                    .getUnderlyingExpression())
               .getVal());

    return childOp.reduce(
        [&](const auto &arg) {
          /*
           * parse output expressions
           * XXX Careful: Assuming numerous output expressions!
           */
          assert(val.HasMember("e"));
          assert(val["e"].IsArray());
          std::vector<expression_t> e;
          e.reserve(val["e"].Size());
          for (const auto &v : val["e"].GetArray()) {
            e.emplace_back(parseExpression(v, arg));
          }

          return e;
        },
        accs);
  } else if (strcmp(opName, "print") == 0) {
    /* parse operator input */
    auto childOp = parseOperator(val["input"]);
    auto pgType = [&]() {
      if (val.HasMember("plugin")) {
        assert(val["plugin"].HasMember("type"));
        assert(val["plugin"]["type"].IsString());
        return pg{val["plugin"]["type"].GetString()};
      } else {
        return pg{"pm-csv"};
      }
    }();

    if (val.HasMember("e")) {
      return childOp.print(
          [&](const auto &arg) {
            /*
             * parse output expressions
             * XXX Careful: Assuming numerous output expressions!
             */
            assert(val.HasMember("e"));
            assert(val["e"].IsArray());
            std::vector<expression_t> e;
            e.reserve(val["e"].Size());
            for (const auto &v : val["e"].GetArray()) {
              e.emplace_back(parseExpression(v, arg));
              assert(e.back().isRegistered());
            }
            return e;
          },
          pgType);
    } else {
      return childOp.print(pgType);
    }
  } else if (strcmp(opName, "sort") == 0) {
    auto childOp = parseOperator(val["input"]);

    assert(val.HasMember("e"));
    assert(val["e"].IsArray());
    vector<direction> d;
    for (const auto &v : val["e"].GetArray()) {
      assert(v.HasMember("direction"));
      assert(v["direction"].IsString());
      std::string dir = v["direction"].GetString();
      if (dir == "ASC")
        d.emplace_back(direction::ASC);
      else if (dir == "NONE")
        d.emplace_back(direction::NONE);
      else if (dir == "DESC")
        d.emplace_back(direction::DESC);
      else
        assert(false);
    }

    return childOp.sort(
        [&](const auto &arg) {
          /*
           * parse output expressions
           * XXX Careful: Assuming numerous output expressions!
           */
          vector<expression_t> e;
          for (const auto &v : val["e"].GetArray()) {
            assert(v.IsObject());
            assert(v.HasMember("expression"));
            assert(v["expression"].IsObject());
            expression_t outExpr = parseExpression(v["expression"], arg);
            e.emplace_back(outExpr);
          }
          return e;
        },
        d);
  } else if (strcmp(opName, "project") == 0) {
    /* parse operator input */
    auto childOp = parseOperator(val["input"]);

    return childOp.project([&](const auto &arg) {
      /*
       * parse output expressions
       * XXX Careful: Assuming numerous output expressions!
       */
      assert(val.HasMember("e"));
      assert(val["e"].IsArray());
      std::vector<expression_t> e;
      for (const auto &v : val["e"].GetArray()) {
        e.emplace_back(parseExpression(v, arg));
      }

      return e;
    });
  } else if (strcmp(opName, "unnest") == 0) {
    /* parse operator input */
    auto childOp = parseOperator(val["input"]);

    /* parse filtering expression */
    assert(val.HasMember("p"));
    assert(val["p"].IsObject());
    expression_t p = parseExpression(val["p"], childOp.getOutputArg());

    /* parse path expression */
    assert(val.HasMember("path"));
    assert(val["path"].IsObject());

    assert(val["path"]["e"].IsObject());
    auto exprToUnnest =
        parseExpression(val["path"]["e"], childOp.getOutputArg());
    auto proj = dynamic_cast<const expressions::RecordProjection *>(
        exprToUnnest.getUnderlyingExpression());
    if (proj == nullptr) {
      string error_msg = string("[Unnest: ] Cannot cast to record projection");
      LOG(ERROR) << error_msg;
      throw runtime_error(string(error_msg));
    }

    string pathAlias;
    if (exprToUnnest.isRegistered()) {
      pathAlias = exprToUnnest.getRegisteredAttrName();
    } else {
      assert(val["path"]["name"].IsString());
      pathAlias = val["path"]["name"].GetString();
    }

    Path projPath{pathAlias, proj};

    auto newOp = new Unnest(p, projPath, childOp.root);
    childOp->setParent(newOp);

    return RelBuilder(ctx, newOp);
  } else if (strcmp(opName, "outer_unnest") == 0) {
    /* parse operator input */
    auto c = parseOperator(val["input"]);
    auto childOp = c.root;

    /* parse filtering expression */
    assert(val.HasMember("p"));
    assert(val["p"].IsObject());
    auto p = parseExpression(val["p"], c.getOutputArg());

    /* parse path expression */
    assert(val.HasMember("path"));
    assert(val["path"].IsObject());

    assert(val["path"]["name"].IsString());
    string pathAlias = val["path"]["name"].GetString();

    assert(val["path"]["e"].IsObject());
    auto exprToUnnest = parseExpression(val["path"]["e"], c.getOutputArg());
    auto proj = dynamic_cast<const expressions::RecordProjection *>(
        exprToUnnest.getUnderlyingExpression());
    if (proj == nullptr) {
      string error_msg = string("[Unnest: ] Cannot cast to record projection");
      LOG(ERROR) << error_msg;
      throw runtime_error(string(error_msg));
    }

    Path projPath{pathAlias, proj};

    newOp = new OuterUnnest(p, projPath, childOp);
    childOp->setParent(newOp);

    return RelBuilder(ctx, newOp);
  } else if (strcmp(opName, "groupby") == 0 ||
             strcmp(opName, "hashgroupby-chained") == 0) {
    /* parse operator input */
    assert(val.HasMember("input"));
    assert(val["input"].IsObject());
    auto childOp = parseOperator(val["input"]);

    // if (val.HasMember("gpu") && val["gpu"].GetBool()){
    assert(val.HasMember("hash_bits"));
    assert(val["hash_bits"].IsUint64());
    size_t hash_bits = val["hash_bits"].GetUint64();

    assert(val.HasMember("maxInputSize"));
    assert(val["maxInputSize"].IsUint64());

    size_t maxInputSize = val["maxInputSize"].GetUint64();

    assert(dynamic_cast<ParallelContext *>(this->ctx));
    // newOp = new GpuHashGroupByChained(e, widths, key_expr, child, hash_bits,
    //                     dynamic_cast<ParallelContext *>(this->ctx),
    //                     maxInputSize);

    return childOp.groupby(
        [&](const auto &arg) {
          assert(val.HasMember("k"));
          assert(val["k"].IsArray());
          vector<expression_t> key_expr;
          for (const auto &k : val["k"].GetArray()) {
            key_expr.emplace_back(parseExpression(k, arg));
          }
          return key_expr;
        },
        [&](const auto &arg) {
          /*
           * parse output expressions
           * XXX Careful: Assuming numerous output expressions!
           */
          assert(val.HasMember("e"));
          assert(val["e"].IsArray());
          std::vector<GpuAggrMatExpr> e;
          for (const auto &v : val["e"].GetArray()) {
            assert(v.HasMember("e"));
            assert(v.HasMember("m"));
            assert(v["m"].IsString());
            assert(v.HasMember("packet"));
            assert(v["packet"].IsInt());
            assert(v.HasMember("offset"));
            assert(v["offset"].IsInt());
            auto outExpr = parseExpression(v["e"], arg);

            e.emplace_back(outExpr, v["packet"].GetInt(), v["offset"].GetInt(),
                           parseAccumulator(v["m"].GetString(), arg));
          }

          return e;
        },
        hash_bits, maxInputSize);
  } else if (strcmp(opName, "out-of-gpu-join") == 0) {
    /* parse operator input */
    assert(val.HasMember("probe_input"));
    assert(val["probe_input"].IsObject());
    auto p = parseOperator(val["probe_input"]);
    auto probe_arg = p.getOutputArg();
    Operator *probe_op = p.root;

    /* parse operator input */
    assert(val.HasMember("build_input"));
    assert(val["build_input"].IsObject());
    auto b = parseOperator(val["build_input"]);
    auto build_arg = b.getOutputArg();
    Operator *build_op = b.root;

    /*number of cpu partitions*/
    assert(val.HasMember("numOfBuckets"));
    assert(val["numOfBuckets"].IsInt());
    size_t numOfBuckets = val["numOfBuckets"].GetInt();
    /*number of cpu threads in partitioning*/
    assert(val.HasMember("numPartitioners"));
    assert(val["numPartitioners"].IsInt());
    size_t numPartitioners = val["numPartitioners"].GetInt();
    /*number of tasks running concurrently in join phase*/
    assert(val.HasMember("numConcurrent"));
    assert(val["numConcurrent"].IsInt());
    size_t numConcurrent = val["numConcurrent"].GetInt();

    /*parameters for join buffers*/
    assert(val.HasMember("maxBuildInputSize"));
    assert(val["maxBuildInputSize"].IsInt());
    int maxBuildInputSize = val["maxBuildInputSize"].GetInt();

    assert(val.HasMember("maxProbeInputSize"));
    assert(val["maxProbeInputSize"].IsInt());
    int maxProbeInputSize = val["maxProbeInputSize"].GetInt();

    assert(val.HasMember("slack"));
    assert(val["slack"].IsInt());
    int slack = val["slack"].GetInt();

    assert(val["build_e"].IsArray());
    vector<RecordAttribute *> build_attr;
    vector<RecordAttribute *> build_attr_block;
    vector<expression_t> build_expr;
    vector<RecordAttribute *> build_hashed_attr;
    vector<RecordAttribute *> build_hashed_attr_block;
    vector<expression_t> build_hashed_expr;
    vector<expression_t> build_hashed_expr_block;
    vector<expression_t> build_prejoin_expr;
    vector<RecordAttribute *> build_join_attr;
    vector<RecordAttribute *> build_join_attr_block;
    vector<GpuMatExpr> build_join_expr;

    for (const auto &e : val["build_e"].GetArray()) {
      assert(e.IsObject());

      assert(e["original"].IsObject());
      auto outExpr = parseExpression(e["original"], build_arg);
      build_expr.emplace_back(outExpr);

      assert(e["original"]["attribute"].IsObject());
      RecordAttribute *recAttr =
          this->parseRecordAttr(e["original"]["attribute"], build_arg);
      build_attr.push_back(recAttr);
      build_attr_block.push_back(new RecordAttribute(*recAttr, true));

      assert(e["hashed"].IsObject());
      auto outHashedExpr = parseExpression(e["hashed"], build_arg);
      build_hashed_expr.emplace_back(outHashedExpr);

      assert(e["hashed-block"].IsObject());
      auto outHashedBlockExpr = parseExpression(e["hashed-block"], build_arg);
      build_hashed_expr_block.emplace_back(outHashedBlockExpr);

      assert(e["hashed"]["attribute"].IsObject());
      RecordAttribute *recHashedAttr =
          this->parseRecordAttr(e["hashed"]["register_as"], build_arg);
      build_hashed_attr.push_back(recHashedAttr);
      build_hashed_attr_block.push_back(
          new RecordAttribute(*recHashedAttr, true));

      assert(e["join"].IsObject());
      assert(e["join"].HasMember("e"));
      assert(e["join"].HasMember("packet"));
      assert(e["join"]["packet"].IsInt());
      assert(e["join"].HasMember("offset"));
      assert(e["join"]["offset"].IsInt());
      auto outJoinExpr = parseExpression(e["join"]["e"], build_arg);
      build_join_expr.emplace_back(outJoinExpr, e["join"]["packet"].GetInt(),
                                   e["join"]["offset"].GetInt());

      assert(e["join"]["e"]["attribute"].IsObject());
      RecordAttribute *recJoinAttr =
          this->parseRecordAttr(e["join"]["e"]["attribute"], build_arg);
      build_join_attr.push_back(recJoinAttr);
      build_join_attr_block.push_back(new RecordAttribute(*recJoinAttr, true));
      auto outPreJoinExpr = parseExpression(e["join"]["e"], build_arg);
      outPreJoinExpr.registerAs(recJoinAttr);
      build_prejoin_expr.push_back(outPreJoinExpr);
    }

    assert(val.HasMember("build_hash"));
    RecordAttribute *build_hash_attr =
        this->parseRecordAttr(val["build_hash"]["attribute"], build_arg);

    assert(val.HasMember("build_w"));
    assert(val["build_w"].IsArray());
    vector<size_t> build_widths;

    for (const auto &w : val["build_w"].GetArray()) {
      assert(w.IsInt());
      build_widths.push_back(w.GetInt());
    }

    assert(val["probe_e"].IsArray());
    vector<RecordAttribute *> probe_attr;
    vector<RecordAttribute *> probe_attr_block;
    vector<expression_t> probe_expr;
    vector<RecordAttribute *> probe_hashed_attr;
    vector<RecordAttribute *> probe_hashed_attr_block;
    vector<expression_t> probe_hashed_expr;
    vector<expression_t> probe_hashed_expr_block;
    vector<expression_t> probe_prejoin_expr;
    vector<RecordAttribute *> probe_join_attr;
    vector<RecordAttribute *> probe_join_attr_block;
    vector<GpuMatExpr> probe_join_expr;

    for (const auto &e : val["probe_e"].GetArray()) {
      assert(e.IsObject());

      assert(e["original"].IsObject());
      auto outExpr = parseExpression(e["original"], probe_arg);
      probe_expr.emplace_back(outExpr);

      assert(e["original"]["attribute"].IsObject());
      RecordAttribute *recAttr =
          this->parseRecordAttr(e["original"]["attribute"], probe_arg);
      probe_attr.push_back(recAttr);
      probe_attr_block.push_back(new RecordAttribute(*recAttr, true));

      assert(e["hashed"].IsObject());
      auto outHashedExpr = parseExpression(e["hashed"], probe_arg);
      probe_hashed_expr.emplace_back(outHashedExpr);

      assert(e["hashed-block"].IsObject());
      auto outHashedBlockExpr = parseExpression(e["hashed-block"], probe_arg);
      probe_hashed_expr_block.emplace_back(outHashedBlockExpr);

      assert(e["hashed"]["attribute"].IsObject());
      RecordAttribute *recHashedAttr =
          this->parseRecordAttr(e["hashed"]["register_as"], probe_arg);
      probe_hashed_attr.push_back(recHashedAttr);
      probe_hashed_attr_block.push_back(
          new RecordAttribute(*recHashedAttr, true));

      assert(e["join"].IsObject());
      assert(e["join"].HasMember("e"));
      assert(e["join"].HasMember("packet"));
      assert(e["join"]["packet"].IsInt());
      assert(e["join"].HasMember("offset"));
      assert(e["join"]["offset"].IsInt());
      auto outJoinExpr = parseExpression(e["join"]["e"], probe_arg);
      probe_join_expr.emplace_back(outJoinExpr, e["join"]["packet"].GetInt(),
                                   e["join"]["offset"].GetInt());

      assert(e["join"]["e"]["attribute"].IsObject());
      RecordAttribute *recJoinAttr =
          this->parseRecordAttr(e["join"]["e"]["attribute"], probe_arg);
      probe_join_attr.push_back(recJoinAttr);
      probe_join_attr_block.push_back(new RecordAttribute(*recJoinAttr, true));
      auto outPreJoinExpr = parseExpression(e["join"]["e"], probe_arg);
      outPreJoinExpr.registerAs(recJoinAttr);
      probe_prejoin_expr.push_back(outPreJoinExpr);
    }

    assert(val.HasMember("probe_hash"));
    RecordAttribute *probe_hash_attr =
        this->parseRecordAttr(val["probe_hash"]["attribute"], probe_arg);

    assert(val.HasMember("probe_w"));
    assert(val["probe_w"].IsArray());
    vector<size_t> probe_widths;

    for (const auto &w : val["build_w"].GetArray()) {
      assert(w.IsInt());
      probe_widths.push_back(w.GetInt());
    }

    Router *xch_build = new Router(
        build_op, DegreeOfParallelism{numPartitioners}, build_attr_block, slack,
        std::nullopt, RoutingPolicy::LOCAL, DeviceType::CPU);
    build_op->setParent(xch_build);
    Operator *btt_build =
        new BlockToTuples(xch_build, build_expr, false, gran_t::THREAD);
    xch_build->setParent(btt_build);
    Operator *part_build = new HashRearrange(
        btt_build, numOfBuckets, build_expr, build_expr[0], build_hash_attr);
    btt_build->setParent(part_build);
    build_attr_block.push_back(build_hash_attr);
    Router *xch_build2 =
        new Router(part_build, DegreeOfParallelism{1}, build_attr_block, slack,
                   std::nullopt, RoutingPolicy::LOCAL, DeviceType::GPU);
    part_build->setParent(xch_build2);

    Router *xch_probe = new Router(
        probe_op, DegreeOfParallelism{numPartitioners}, probe_attr_block, slack,
        std::nullopt, RoutingPolicy::LOCAL, DeviceType::CPU);
    probe_op->setParent(xch_probe);
    Operator *btt_probe =
        new BlockToTuples(xch_probe, probe_expr, false, gran_t::THREAD);
    xch_probe->setParent(btt_probe);
    Operator *part_probe = new HashRearrange(
        btt_probe, numOfBuckets, probe_expr, probe_expr[0], probe_hash_attr);
    btt_probe->setParent(part_probe);
    probe_attr_block.push_back(probe_hash_attr);
    Router *xch_probe2 =
        new Router(part_probe, DegreeOfParallelism{1}, probe_attr_block, slack,
                   std::nullopt, RoutingPolicy::LOCAL, DeviceType::GPU);
    part_probe->setParent(xch_probe2);

    RecordAttribute *attr_ptr =
        new RecordAttribute(1, "coordinator", "ptr", new IntType(), true);
    RecordAttribute *attr_target =
        new RecordAttribute(1, "coordinator", "target", new IntType(), false);
    RecordAttribute *attr_splitter =
        new RecordAttribute(2, "coordinator", "splitter", new IntType(), false);

    InputInfo *datasetInfoCoord =
        catalogParser.getOrCreateInputInfo(attr_target->getRelationName());
    RecordType *coord_rec = new RecordType{dynamic_cast<const RecordType &>(
        dynamic_cast<CollectionType *>(datasetInfoCoord->exprType)
            ->getNestedType())};
    coord_rec->appendAttribute(attr_ptr);
    coord_rec->appendAttribute(attr_target);
    coord_rec->appendAttribute(attr_splitter);
    datasetInfoCoord->exprType = new BagType{*coord_rec};

    list<RecordAttribute *> f_atts_target;
    f_atts_target.push_back(attr_ptr);
    f_atts_target.push_back(attr_target);
    f_atts_target.push_back(attr_splitter);
    RecordType *recTypeTarget = new RecordType(f_atts_target);

    list<RecordAttribute> f_atts_hash_d;
    f_atts_hash_d.push_back(*attr_target);
    expressions::InputArgument fHtarg{recTypeTarget, -1, f_atts_hash_d};
    expressions::RecordProjection expr_target =
        expression_t{fHtarg}[*attr_target];

    vector<RecordAttribute *> f_atts_target_v;
    f_atts_target_v.push_back(attr_ptr);
    f_atts_target_v.push_back(attr_target);
    f_atts_target_v.push_back(attr_splitter);

    ZipCollect *coord =
        new ZipCollect(attr_ptr, attr_splitter, attr_target,
                       new RecordAttribute(*build_attr[0], true),
                       new RecordAttribute(*probe_attr[0], true), xch_build2,
                       xch_probe2, (ParallelContext *)ctx, numOfBuckets,
                       build_hash_attr, build_hashed_expr_block,
                       probe_hash_attr, probe_hashed_expr_block, "coordinator");
    xch_build2->setParent(coord);
    xch_probe2->setParent(coord);

    Router *xch_proc = new Router(coord, DegreeOfParallelism{numConcurrent},
                                  f_atts_target_v, slack, expr_target,
                                  RoutingPolicy::HASH_BASED, DeviceType::GPU);
    coord->setParent(xch_proc);
    ZipInitiate *initiator = new ZipInitiate(
        attr_ptr, attr_splitter, attr_target, xch_proc, (ParallelContext *)ctx,
        numOfBuckets, coord->getStateLeft(), coord->getStateRight(),
        "launcher");
    xch_proc->setParent(initiator);
    PipelineGen **pip_rcv = initiator->pipeSocket();

    ZipForward *fwd_build =
        new ZipForward(attr_target, initiator, (ParallelContext *)ctx,
                       build_hashed_expr, "forwarder", coord->getStateLeft());

    Operator *mml_build =
        new MemMoveLocalTo(fwd_build, build_hashed_attr_block, 4);
    fwd_build->setParent(mml_build);
    Operator *mmd_build = new MemMoveDevice(mml_build, (ParallelContext *)ctx,
                                            build_hashed_attr_block, 4, false);
    mml_build->setParent(mmd_build);
    Operator *ctg_build = new CpuToGpu(mmd_build, build_hashed_attr_block);
    mmd_build->setParent(ctg_build);
    Operator *btt_build2 =
        new BlockToTuples(ctg_build, build_prejoin_expr, true, gran_t::GRID);
    ctg_build->setParent(btt_build2);
    HashPartitioner *hpart1 = new HashPartitioner(
        build_join_expr, build_widths, build_prejoin_expr[0], btt_build2,
        (ParallelContext *)ctx, maxBuildInputSize, 13, "partition_hash_1");
    btt_build2->setParent(hpart1);

    ZipForward *fwd_probe =
        new ZipForward(attr_target, initiator, (ParallelContext *)ctx,
                       probe_hashed_expr, "forwarder", coord->getStateRight());

    Operator *mml_probe =
        new MemMoveLocalTo(fwd_probe, probe_hashed_attr_block, 4);
    fwd_probe->setParent(mml_probe);
    Operator *mmd_probe = new MemMoveDevice(mml_probe, (ParallelContext *)ctx,
                                            probe_hashed_attr_block, 4, false);
    mml_probe->setParent(mmd_probe);
    Operator *ctg_probe = new CpuToGpu(mmd_probe, probe_hashed_attr_block);
    mmd_probe->setParent(ctg_probe);
    Operator *btt_probe2 =
        new BlockToTuples(ctg_probe, probe_prejoin_expr, true, gran_t::GRID);
    ctg_probe->setParent(btt_probe2);
    HashPartitioner *hpart2 = new HashPartitioner(
        probe_join_expr, probe_widths, probe_prejoin_expr[0], btt_probe2,
        (ParallelContext *)ctx, maxProbeInputSize, 13, "partition_hash_2");
    btt_probe2->setParent(hpart2);

    newOp = new GpuPartitionedHashJoinChained(
        build_join_expr, build_widths, build_join_expr[0].expr, std::nullopt,
        hpart1, probe_join_expr, probe_widths, probe_join_expr[0].expr,
        std::nullopt, hpart2, hpart1->getState(), hpart2->getState(),
        maxBuildInputSize, maxProbeInputSize, 13, (ParallelContext *)ctx,
        "hj_part", pip_rcv, nullptr);
    hpart1->setParent(newOp);
    hpart2->setParent(newOp);

    return RelBuilder(ctx, newOp);
  } else if (strcmp(opName, "partitioned-hashjoin-chained") == 0) {
    /* parse operator input */
    assert(val.HasMember("probe_input"));
    assert(val["probe_input"].IsObject());
    auto p = parseOperator(val["probe_input"]);
    auto probe_arg = p.getOutputArg();
    Operator *probe_op = p.root;

    /* parse operator input */
    assert(val.HasMember("build_input"));
    assert(val["build_input"].IsObject());
    auto b = parseOperator(val["build_input"]);
    auto build_arg = b.getOutputArg();
    Operator *build_op = b.root;

    assert(val.HasMember("build_k"));
    auto build_key_expr = parseExpression(val["build_k"], build_arg);

    assert(val.HasMember("probe_k"));
    auto probe_key_expr = parseExpression(val["probe_k"], probe_arg);

    std::optional<expression_t> build_minorkey_expr{
        (val.HasMember("build_k_minor")) ? std::make_optional(parseExpression(
                                               val["build_k_minor"], build_arg))
                                         : std::nullopt};

    std::optional<expression_t> probe_minorkey_expr{
        (val.HasMember("probe_k_minor")) ? std::make_optional(parseExpression(
                                               val["probe_k_minor"], probe_arg))
                                         : std::nullopt};

    assert(val.HasMember("build_w"));
    assert(val["build_w"].IsArray());
    vector<size_t> build_widths;

    for (const auto &w : val["build_w"].GetArray()) {
      assert(w.IsInt());
      build_widths.push_back(w.GetInt());
    }

    /*
     * parse output expressions
     * XXX Careful: Assuming numerous output expressions!
     */
    assert(val.HasMember("build_e"));
    assert(val["build_e"].IsArray());
    vector<GpuMatExpr> build_e;
    for (const auto &e : val["build_e"].GetArray()) {
      assert(e.HasMember("e"));
      assert(e.HasMember("packet"));
      assert(e["packet"].IsInt());
      assert(e.HasMember("offset"));
      assert(e["offset"].IsInt());
      auto outExpr = parseExpression(e["e"], build_arg);

      build_e.emplace_back(outExpr, e["packet"].GetInt(), e["offset"].GetInt());
    }

    assert(val.HasMember("probe_w"));
    assert(val["probe_w"].IsArray());
    vector<size_t> probe_widths;

    for (const auto &w : val["probe_w"].GetArray()) {
      assert(w.IsInt());
      probe_widths.push_back(w.GetInt());
    }

    /*
     * parse output expressions
     * XXX Careful: Assuming numerous output expressions!
     */
    assert(val.HasMember("probe_e"));
    assert(val["probe_e"].IsArray());
    vector<GpuMatExpr> probe_e;
    for (const auto &e : val["probe_e"].GetArray()) {
      assert(e.HasMember("e"));
      assert(e.HasMember("packet"));
      assert(e["packet"].IsInt());
      assert(e.HasMember("offset"));
      assert(e["offset"].IsInt());
      auto outExpr = parseExpression(e["e"], probe_arg);

      probe_e.emplace_back(outExpr, e["packet"].GetInt(), e["offset"].GetInt());
    }

    assert(val.HasMember("maxBuildInputSize"));
    assert(val["maxBuildInputSize"].IsUint64());

    size_t maxBuildInputSize = val["maxBuildInputSize"].GetUint64();

    assert(val.HasMember("maxProbeInputSize"));
    assert(val["maxProbeInputSize"].IsUint64());

    size_t maxProbeInputSize = val["maxProbeInputSize"].GetUint64();

    assert(dynamic_cast<ParallelContext *>(this->ctx));

    int log_parts = 13;

    HashPartitioner *part_left =
        new HashPartitioner(build_e, build_widths, build_key_expr, build_op,
                            dynamic_cast<ParallelContext *>(this->ctx),
                            maxBuildInputSize, log_parts, "part1");

    HashPartitioner *part_right =
        new HashPartitioner(probe_e, probe_widths, probe_key_expr, probe_op,
                            dynamic_cast<ParallelContext *>(this->ctx),
                            maxProbeInputSize, log_parts, "part1");

    newOp = new GpuPartitionedHashJoinChained(
        build_e, build_widths, build_key_expr, build_minorkey_expr, part_left,
        probe_e, probe_widths, probe_key_expr, probe_minorkey_expr, part_right,
        part_left->getState(), part_right->getState(), maxBuildInputSize,
        maxProbeInputSize, log_parts,
        dynamic_cast<ParallelContext *>(this->ctx), "phjc", nullptr, nullptr);

    build_op->setParent(part_left);
    probe_op->setParent(part_right);

    build_op = part_left;
    probe_op = part_right;

    build_op->setParent(newOp);
    probe_op->setParent(newOp);

    return RelBuilder(ctx, newOp);
  } else if (strcmp(opName, "hashjoin-chained") == 0) {
    /* parse operator input */
    assert(val.HasMember("probe_input"));
    assert(val["probe_input"].IsObject());
    auto probe_op = parseOperator(val["probe_input"]);
    /* parse operator input */
    assert(val.HasMember("build_input"));
    assert(val["build_input"].IsObject());
    auto build_op = parseOperator(val["build_input"]);

    assert(val.HasMember("hash_bits"));
    assert(val["hash_bits"].IsUint64());
    size_t hash_bits = val["hash_bits"].GetUint64();

    assert(val.HasMember("maxBuildInputSize"));
    assert(val["maxBuildInputSize"].IsUint64());

    size_t maxBuildInputSize = val["maxBuildInputSize"].GetUint64();

    if (val.HasMember("morsel") && val["morsel"].GetBool()) {
      return probe_op.morsel_join(
          build_op,
          [&](const auto &build_arg) {
            assert(val.HasMember("build_k"));
            return parseExpression(val["build_k"], build_arg);
          },
          [&](const auto &probe_arg) {
            assert(val.HasMember("probe_k"));
            return parseExpression(val["probe_k"], probe_arg);
          },
          hash_bits, maxBuildInputSize);
    } else {
      return probe_op.join(
          build_op,
          [&](const auto &build_arg) {
            assert(val.HasMember("build_k"));
            return parseExpression(val["build_k"], build_arg);
          },
          [&](const auto &probe_arg) {
            assert(val.HasMember("probe_k"));
            return parseExpression(val["probe_k"], probe_arg);
          },
          hash_bits, maxBuildInputSize);
    }
  } else if (strcmp(opName, "join") == 0) {
    assert(!(val.HasMember("gpu") && val["gpu"].GetBool()));
    const char *keyMatLeft = "leftFields";
    const char *keyMatRight = "rightFields";
    const char *keyPred = "p";

    /* parse operator input */
    auto left = parseOperator(val["leftInput"]);
    Operator *leftOp = left.root;
    auto right = parseOperator(val["rightInput"]);
    Operator *rightOp = right.root;

    // Predicate
    assert(val.HasMember(keyPred));
    assert(val[keyPred].IsObject());

    assert(val[keyPred]["expression"].GetString() == std::string{"eq"});
    auto el = parseExpression(val[keyPred]["left"], left.getOutputArg());
    auto er = parseExpression(val[keyPred]["right"], right.getOutputArg());

    auto pred = eq(el, er);

    /*
     * *** WHAT TO MATERIALIZE ***
     * XXX v0: JSON file contains a list of **RecordProjections**
     * EXPLICIT OIDs injected by PARSER (not in json file by default)
     * XXX Eager materialization atm
     *
     * XXX Why am I not using minimal constructor for materializer yet, as use
     * cases do?
     *     -> Because then I would have to encode the OID type in JSON -> can be
     * messy
     */

    // LEFT SIDE
    assert(val.HasMember(keyMatLeft));
    assert(val[keyMatLeft].IsArray());
    vector<expression_t> exprsLeft;
    map<string, RecordAttribute *> mapOidsLeft =
        map<string, RecordAttribute *>();
    vector<RecordAttribute *> fieldsLeft = vector<RecordAttribute *>();
    vector<materialization_mode> outputModesLeft;
    for (const auto &keyMat : val[keyMatLeft].GetArray()) {
      auto exprL = parseExpression(keyMat, left.getOutputArg());

      exprsLeft.push_back(exprL);
      outputModesLeft.insert(outputModesLeft.begin(), EAGER);

      // XXX STRONG ASSUMPTION: Expression is actually a record projection!
      const expressions::RecordProjection *projL =
          dynamic_cast<const expressions::RecordProjection *>(
              exprL.getUnderlyingExpression());
      if (projL == nullptr) {
        string error_msg =
            string("[Join: ] Cannot cast to rec projection. Original: ") +
            exprL.getExpressionType()->getType();
        LOG(ERROR) << error_msg;
        throw runtime_error(string(error_msg));
      }
      // Added in 'wanted fields'
      RecordAttribute *recAttr = new RecordAttribute(projL->getAttribute());
      fieldsLeft.push_back(recAttr);

      string relName = recAttr->getRelationName();
      if (mapOidsLeft.find(relName) == mapOidsLeft.end()) {
        InputInfo *datasetInfo = (this->catalogParser).getInputInfo(relName);
        RecordAttribute *oid = new RecordAttribute(
            recAttr->getRelationName(), activeLoop, datasetInfo->oidType);
        mapOidsLeft[relName] = oid;
        auto oidL = projL->getExpr()[*oid];
        // Added in 'wanted expressions'
        cout << "Injecting left OID for " << relName << endl;
        exprsLeft.insert(exprsLeft.begin(), oidL);
        outputModesLeft.insert(outputModesLeft.begin(), EAGER);
      }
    }
    vector<RecordAttribute *> oidsLeft = vector<RecordAttribute *>();
    MapToVec(mapOidsLeft, oidsLeft);
    Materializer *matLeft =
        new Materializer(fieldsLeft, exprsLeft, oidsLeft, outputModesLeft);

    // RIGHT SIDE
    assert(val.HasMember(keyMatRight));
    assert(val[keyMatRight].IsArray());
    vector<expression_t> exprsRight;
    map<string, RecordAttribute *> mapOidsRight =
        map<string, RecordAttribute *>();
    vector<RecordAttribute *> fieldsRight = vector<RecordAttribute *>();
    vector<materialization_mode> outputModesRight;
    for (const auto &keyMat : val[keyMatRight].GetArray()) {
      auto exprR = parseExpression(keyMat, right.getOutputArg());

      exprsRight.push_back(exprR);
      outputModesRight.insert(outputModesRight.begin(), EAGER);

      // XXX STRONG ASSUMPTION: Expression is actually a record projection!
      auto projR = dynamic_cast<const expressions::RecordProjection *>(
          exprR.getUnderlyingExpression());
      if (projR == nullptr) {
        string error_msg =
            string("[Join: ] Cannot cast to rec projection. Original: ") +
            exprR.getExpressionType()->getType();
        LOG(ERROR) << error_msg;
        throw runtime_error(string(error_msg));
      }

      // Added in 'wanted fields'
      RecordAttribute *recAttr = new RecordAttribute(projR->getAttribute());
      fieldsRight.push_back(recAttr);

      string relName = recAttr->getRelationName();
      if (mapOidsRight.find(relName) == mapOidsRight.end()) {
        InputInfo *datasetInfo = (this->catalogParser).getInputInfo(relName);
        RecordAttribute *oid = new RecordAttribute(
            recAttr->getRelationName(), activeLoop, datasetInfo->oidType);
        mapOidsRight[relName] = oid;
        expressions::RecordProjection oidR = projR->getExpr()[*oid];
        // Added in 'wanted expressions'
        exprsRight.insert(exprsRight.begin(), oidR);
        cout << "Injecting right OID for " << relName << endl;
        outputModesRight.insert(outputModesRight.begin(), EAGER);
      }
    }
    vector<RecordAttribute *> oidsRight = vector<RecordAttribute *>();
    MapToVec(mapOidsRight, oidsRight);
    Materializer *matRight =
        new Materializer(fieldsRight, exprsRight, oidsRight, outputModesRight);

    newOp = new RadixJoin(pred, leftOp, rightOp, this->ctx, "radixHashJoin",
                          *matLeft, *matRight);
    leftOp->setParent(newOp);
    rightOp->setParent(newOp);

    return RelBuilder(ctx, newOp);
  } else if (strcmp(opName, "nest") == 0) {
    const char *keyGroup = "f";
    const char *keyNull = "g";
    const char *keyPred = "p";
    const char *keyExprs = "e";
    const char *keyAccum = "accumulator";
    /* Physical Level Info */
    const char *keyAggrNames = "aggrLabels";
    // Materializer
    const char *keyMat = "fields";

    /* parse operator input */
    assert(val.HasMember("input"));
    assert(val["input"].IsObject());
    auto child = parseOperator(val["input"]);
    auto arg = child.getOutputArg();
    auto childOp = child.root;

    /* get monoid(s) */
    assert(val.HasMember(keyAccum));
    assert(val[keyAccum].IsArray());
    vector<Monoid> accs;
    for (const auto &accm : val[keyAccum].GetArray()) {
      assert(accm.IsString());
      Monoid acc = parseAccumulator(accm.GetString(), arg);
      accs.push_back(acc);
    }
    /* get label for each of the aggregate values */
    vector<string> aggrLabels;
    assert(val.HasMember(keyAggrNames));
    assert(val[keyAggrNames].IsArray());
    for (const auto &label : val[keyAggrNames].GetArray()) {
      assert(label.IsString());
      aggrLabels.emplace_back(label.GetString());
    }

    /* Predicate */
    assert(val.HasMember(keyPred));
    assert(val[keyPred].IsObject());
    auto predExpr = parseExpression(val[keyPred], arg);

    /* Group By */
    assert(val.HasMember(keyGroup));
    assert(val[keyGroup].IsObject());
    auto groupByExpr = parseExpression(val[keyGroup], arg);

    /* Null-to-zero Checks */
    // FIXME not used in radix nest yet!
    assert(val.HasMember(keyNull));
    assert(val[keyNull].IsObject());
    auto nullsToZerosExpr = parseExpression(val[keyNull], arg);

    /* Output aggregate expression(s) */
    assert(val.HasMember(keyExprs));
    assert(val[keyExprs].IsArray());
    vector<expression_t> outputExprs;
    for (const auto &v : val[keyExprs].GetArray()) {
      outputExprs.emplace_back(parseExpression(v, arg));
    }

    /*
     * *** WHAT TO MATERIALIZE ***
     * XXX v0: JSON file contains a list of **RecordProjections**
     * EXPLICIT OIDs injected by PARSER (not in json file by default)
     * XXX Eager materialization atm
     *
     * XXX Why am I not using minimal constructor for materializer yet, as use
     * cases do?
     *     -> Because then I would have to encode the OID type in JSON -> can be
     * messy
     */

    assert(val.HasMember(keyMat));
    assert(val[keyMat].IsArray());
    vector<expression_t> exprsToMat;
    map<string, RecordAttribute *> mapOids;
    vector<RecordAttribute *> fieldsToMat;
    vector<materialization_mode> outputModes;
    for (const auto &v : val[keyMat].GetArray()) {
      auto expr = parseExpression(v, arg);
      exprsToMat.push_back(expr);
      outputModes.insert(outputModes.begin(), EAGER);

      // XXX STRONG ASSUMPTION: Expression is actually a record projection!
      auto proj = dynamic_cast<const expressions::RecordProjection *>(
          expr.getUnderlyingExpression());
      if (proj == nullptr) {
        string error_msg =
            string("[Nest: ] Cannot cast to rec projection. Original: ") +
            expr.getExpressionType()->getType();
        LOG(ERROR) << error_msg;
        throw runtime_error(string(error_msg));
      }
      // Added in 'wanted fields'
      RecordAttribute *recAttr = new RecordAttribute(proj->getAttribute());
      fieldsToMat.push_back(recAttr);

      string relName = recAttr->getRelationName();
      if (mapOids.find(relName) == mapOids.end()) {
        InputInfo *datasetInfo = (this->catalogParser).getInputInfo(relName);
        auto *oid = new RecordAttribute(recAttr->getRelationName(), activeLoop,
                                        datasetInfo->oidType);
        mapOids[relName] = oid;
        auto oidProj = proj->getExpr()[*oid];
        // Added in 'wanted expressions'
        LOG(INFO) << "[Plan Parser: ] Injecting OID for " << relName;
        //                std::cout << "[Plan Parser: ] Injecting OID for " <<
        //                relName << std::endl;
        /* ORDER OF expression fields matters!! OIDs need to be placed first! */
        exprsToMat.insert(exprsToMat.begin(), oidProj);
        outputModes.insert(outputModes.begin(), EAGER);
      }
    }
    vector<RecordAttribute *> oids;
    MapToVec(mapOids, oids);
    /* FIXME This constructor breaks nest use cases that trigger caching */
    /* Check similar hook in radix-nest.cpp */
    auto *matCoarse =
        new Materializer(fieldsToMat, exprsToMat, oids, outputModes);

    // Materializer* matCoarse = new Materializer(exprsToMat);

    // Put operator together
    newOp = new radix::Nest(this->ctx, accs, outputExprs, aggrLabels, predExpr,
                            std::vector<expression_t>{groupByExpr},
                            nullsToZerosExpr, childOp, "radixNest", *matCoarse);
    childOp->setParent(newOp);

    return RelBuilder(ctx, newOp);
  } else if (strcmp(opName, "select") == 0) {
    /* parse operator input */
    assert(val.HasMember("input"));
    assert(val["input"].IsObject());
    auto childOp = parseOperator(val["input"]);

    return childOp.filter([&](const auto &arg) {
      /* parse filtering expression */
      assert(val.HasMember("p"));
      assert(val["p"].IsObject());
      return parseExpression(val["p"], arg);
    });
  } else if (strcmp(opName, "scan") == 0) {
    assert(val.HasMember(keyPg));
    assert(val[keyPg].IsObject());
    Plugin *pg = this->parsePlugin(val[keyPg]);

    return factory.getBuilder().scan(*pg);
  } else if (strcmp(opName, "dict-scan") == 0) {
    assert(val.HasMember("relName"));
    assert(val["relName"].IsString());
    auto relName = val["relName"].GetString();

    assert(val.HasMember("attrName"));
    assert(val["attrName"].IsString());
    auto attrName = val["attrName"].GetString();

    assert(val.HasMember("regex"));
    assert(val["regex"].IsString());
    auto regex = val["regex"].GetString();

    auto dictRelName = relName + std::string{"$dict$"} + attrName;

    void *dict = StorageManager::getInstance().getDictionaryOf(
        relName + std::string{"."} + attrName);

    InputInfo *datasetInfo =
        (this->catalogParser).getOrCreateInputInfo(dictRelName);
    auto *rec = new RecordType{dynamic_cast<const RecordType &>(
        dynamic_cast<CollectionType *>(datasetInfo->exprType)
            ->getNestedType())};
    auto *reg_as =
        new RecordAttribute(dictRelName, attrName, new DStringType(dict));
    LOG(INFO) << "Registered: " << reg_as->getRelationName() << "."
              << reg_as->getAttrName();
    rec->appendAttribute(reg_as);

    datasetInfo->exprType = new BagType{*rec};

    newOp = new DictScan(
        this->ctx, RecordAttribute{relName, attrName, new DStringType(dict)},
        regex, *reg_as);

    return RelBuilder(ctx, newOp);
#ifndef NCUDA
  } else if (strcmp(opName, "cpu-to-gpu") == 0) {
    /* parse operator input */
    assert(val.HasMember("input"));
    assert(val["input"].IsObject());
    auto childOp = parseOperator(val["input"]);

    return childOp.to_gpu();
#endif
  } else if (strcmp(opName, "block-to-tuples") == 0 ||
             strcmp(opName, "unpack") == 0) {
    /* parse operator input */
    assert(val.HasMember("input"));
    assert(val["input"].IsObject());
    auto childOp = parseOperator(val["input"]);

    return childOp.unpack();
#ifndef NCUDA
  } else if (strcmp(opName, "gpu-to-cpu") == 0) {
    /* parse operator input */
    assert(val.HasMember("input"));
    assert(val["input"].IsObject());
    auto child = parseOperator(val["input"]);

    assert(val.HasMember("queueSize"));
    assert(val["queueSize"].IsInt());
    int size = val["queueSize"].GetInt();

    assert(val.HasMember("granularity"));
    assert(val["granularity"].IsString());
    std::string gran = val["granularity"].GetString();
    std::transform(gran.begin(), gran.end(), gran.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    gran_t g = gran_t::GRID;
    if (gran == "grid")
      g = gran_t::GRID;
    else if (gran == "block")
      g = gran_t::BLOCK;
    else if (gran == "thread")
      g = gran_t::THREAD;
    else
      assert(false && "granularity must be one of GRID, BLOCK, THREAD");

    return child.to_cpu(g, size);
#endif
  } else if (strcmp(opName, "tuples-to-block") == 0 ||
             strcmp(opName, "pack") == 0) {
    /* parse operator input */
    assert(val.HasMember("input"));
    assert(val["input"].IsObject());
    auto childOp = parseOperator(val["input"]);

    return childOp.pack();
  } else if (strcmp(opName, "hash-rearrange") == 0 ||
             strcmp(opName, "hash-pack") == 0) {
    assert(val.HasMember("input"));
    assert(val["input"].IsObject());
    auto childOp = parseOperator(val["input"]);

    int numOfBuckets = 2;  // FIXME: !!!
    if (val.HasMember("buckets")) {
      assert(val["buckets"].IsInt());
      numOfBuckets = val["buckets"].GetInt();
    }

    return childOp.pack(
        [&](const auto &arg) -> std::vector<expression_t> {
          assert(val.HasMember("projections"));
          assert(val["projections"].IsArray());
          vector<expression_t> projections;
          for (const auto &v : val["projections"].GetArray()) {
            projections.emplace_back(parseExpression(v, arg));
          }
          return projections;
        },
        [&](const auto &arg) -> expression_t {
          assert(val.HasMember("e"));
          assert(val["e"].IsObject());

          auto hashExpr = parseExpression(val["e"], arg);

          if (val.HasMember("hashProject")) {
            hashExpr.as(parseRecordAttr(val["hashProject"], arg));
          }
          return hashExpr;
        },
        numOfBuckets);
  } else if (strcmp(opName, "mem-move-device") == 0) {
    /* parse operator input */
    assert(val.HasMember("input"));
    assert(val["input"].IsObject());
    auto childOp = parseOperator(val["input"]);

    bool to_cpu = false;
    if (val.HasMember("to_cpu")) {
      assert(val["to_cpu"].IsBool());
      to_cpu = val["to_cpu"].GetBool();
    }

    int slack = 8;
    if (val.HasMember("slack")) {
      assert(val["slack"].IsInt());
      slack = val["slack"].GetInt();
    }

    size_t width = childOp.getOutputArg().getProjections().size();
    std::vector<bool> do_transfer;
    if (false && val.HasMember("do_transfer")) {
      assert(val["do_transfer"].IsArray());
      assert(val["do_transfer"].Size() == width);
      for (const auto &a : val["do_transfer"].GetArray()) {
        assert(a.IsBool());
        do_transfer.emplace_back(a.GetBool());
        assert(do_transfer.back() && "TODO implement parsing lazy");
      }
    } else {
      for (size_t i = 0; i < width; ++i) {
        do_transfer.emplace_back(true);
      }
    }

    return childOp.memmove(slack, to_cpu ? DeviceType::CPU : DeviceType::GPU);
  } else if (strcmp(opName, "mem-broadcast-device") == 0) {
    /* parse operator input */
    assert(val.HasMember("input"));
    assert(val["input"].IsObject());
    auto childOp = parseOperator(val["input"]);

    bool to_cpu = false;
    if (val.HasMember("to_cpu")) {
      assert(val["to_cpu"].IsBool());
      to_cpu = val["to_cpu"].GetBool();
    }

    bool always_share = false;
    if (val.HasMember("always_share")) {
      assert(val["always_share"].IsBool());
      always_share = val["always_share"].GetBool();
    }

    assert(parFactory->getDOP(to_cpu ? DeviceType::CPU : DeviceType::GPU, val,
                              childOp) != DegreeOfParallelism{1});

    return childOp.membrdcst(
        parFactory->getDOP(to_cpu ? DeviceType::CPU : DeviceType::GPU, val,
                           childOp),
        to_cpu, always_share);
  } else if (strcmp(opName, "mem-move-local-to") == 0) {
    /* parse operator input */
    assert(val.HasMember("input"));
    assert(val["input"].IsObject());
    auto child = parseOperator(val["input"]);
    auto arg = child.getOutputArg();
    auto childOp = child.root;

    assert(val.HasMember("projections"));
    assert(val["projections"].IsArray());

    vector<RecordAttribute *> projections;
    for (const auto &proj : val["projections"].GetArray()) {
      assert(proj.IsObject());
      RecordAttribute *recAttr = this->parseRecordAttr(proj, arg);
      projections.push_back(recAttr);
    }

    int slack = 8;
    if (val.HasMember("slack")) {
      assert(val["slack"].IsInt());
      slack = val["slack"].GetInt();
    }

    assert(dynamic_cast<ParallelContext *>(this->ctx));
    newOp = new MemMoveLocalTo(childOp, projections, slack);
    childOp->setParent(newOp);

    return RelBuilder(ctx, newOp);
  } else if (strcmp(opName, "exchange") == 0 || strcmp(opName, "router") == 0) {
    /* parse operator input */
    assert(val.HasMember("input"));
    assert(val["input"].IsObject());
    auto childOp = parseOperator(val["input"]);

    size_t slack = 8;
    if (val.HasMember("slack")) {
      assert(val["slack"].IsUint64());
      slack = val["slack"].GetUint64();
    }

    auto targets = DeviceType::GPU;
    if (val.HasMember("cpu_targets")) {
      assert(val["cpu_targets"].IsBool());
      targets =
          val["cpu_targets"].GetBool() ? DeviceType::CPU : DeviceType::GPU;
    }

    auto dop = parFactory->getDOP(targets, val, childOp);

    auto policy_type = parFactory->getRoutingPolicy(targets, val, childOp);

    auto aff = parFactory->getAffinitizer(targets, policy_type, val, childOp);

    if (val.HasMember("target")) {
      assert(policy_type == RoutingPolicy::HASH_BASED);
      return childOp.router(
          [&](const auto &arg) -> std::optional<expression_t> {
            assert(val["target"].IsObject());
            return parseExpression(val["target"], arg);
          },
          dop, slack, policy_type, targets, std::move(aff));
    } else {
      return childOp.router(dop, slack, policy_type, targets, std::move(aff));
    }
  } else if (strcmp(opName, "union-all") == 0) {
    /* parse operator input */
    assert(val.HasMember("input"));
    assert(val["input"].IsArray());

    std::vector<RelBuilder> children;
    for (const auto &child : val["input"].GetArray()) {
      assert(child.IsObject());
      children.emplace_back(parseOperator(child));
    }

    assert(children.size() >= 2);

    return children[0].unionAll({children.begin() + 1, children.end()});
  } else if (strcmp(opName, "split") == 0) {
    assert(val.HasMember("split_id"));
    assert(val["split_id"].IsInt());
    size_t split_id = val["split_id"].GetInt();

    if (splitOps.count(split_id) == 0) {
      /* parse operator input */
      assert(val.HasMember("input"));
      assert(val["input"].IsObject());
      auto child = parseOperator(val["input"]);
      auto arg = child.getOutputArg();
      auto childOp = child.root;

      assert(val.HasMember("numOfParents"));
      assert(val["numOfParents"].IsInt());
      int numOfParents = val["numOfParents"].GetInt();

      std::vector<RecordAttribute *> projections;
      for (const auto &attr : arg.getProjections()) {
        if (attr.getAttrName() == "__broadcastTarget") {
          continue;
        }
        projections.emplace_back(new RecordAttribute{attr});
      }

      int slack = 8;
      if (val.HasMember("slack")) {
        assert(val["slack"].IsInt());
        slack = val["slack"].GetInt();
      }

      // Does it make sense to have anything rather than rand local ?
      bool numa_local = false;  // = true;
      bool rand_local_cpu = false;
      std::optional<expression_t> hash;
      if (val.HasMember("target")) {
        assert(val["target"].IsObject());
        hash = parseExpression(val["target"], arg);
        numa_local = false;
      }

      if (val.HasMember("rand_local_cpu")) {
        assert(!hash.has_value() && "Can not have both flags set");
        assert(val["rand_local_cpu"].IsBool());
        rand_local_cpu = val["rand_local_cpu"].GetBool();
        numa_local = false;
      }

      if (val.HasMember("numa_local")) {
        assert(!hash.has_value() && "Can not have both flags set");
        assert(!rand_local_cpu);
        assert(numa_local);
        assert(val["numa_local"].IsBool());
        numa_local = val["numa_local"].GetBool();
      }

      RoutingPolicy policy_type =
          (numa_local || rand_local_cpu)
              ? RoutingPolicy::LOCAL
              : ((hash.has_value()) ? RoutingPolicy::HASH_BASED
                                    : RoutingPolicy::RANDOM);

      assert(dynamic_cast<ParallelContext *>(this->ctx));
      newOp = new Split(childOp, numOfParents, projections, slack, hash,
                        policy_type);
      splitOps[split_id] = newOp;
      childOp->setParent(newOp);
    } else {
      newOp = splitOps[split_id];
      // Splits from common subtrees may have the same split_id, but subtrees
      // will, at least with the current planner, not be nested, so cleanup
      // the list to avoid Splits with 4 parents.
      splitOps.erase(split_id);
    }
    return RelBuilder(ctx, newOp);
  }
  string err = string("Unknown Operator: ") + opName;
  LOG(ERROR) << err;
  throw runtime_error(err);
}

/**
 * {"name": "foo", "type": "csv", ... }
 * FIXME / TODO If we introduce more plugins, this code must be extended
 */
Plugin *PlanExecutor::parsePlugin(const rapidjson::Value &val) {
  Plugin *newPg = nullptr;

  const char *keyInputName = "name";
  const char *keyPgType = "type";

  /*
   * CSV-specific
   */
  // which fields to project
  const char *keyProjections = "projections";
  // pm policy
  const char *keyPolicy = "policy";
  // line hint
  const char *keyLineHint = "lines";
  // OPTIONAL: which delimiter to use
  const char *keyDelimiter = "delimiter";
  // OPTIONAL: are string values wrapped in brackets?
  const char *keyBrackets = "brackets";

  assert(val.HasMember(keyInputName));
  assert(val[keyInputName].IsString());
  std::string datasetName = val[keyInputName].GetString();

  assert(val.HasMember(keyPgType));
  assert(val[keyPgType].IsString());
  std::string pgType = val[keyPgType].GetString();

  if (pgType == "dynamic") {
    pgType = parFactory->getDynamicPgName(pgType);
    datasetName = datasetName + "<" + pgType + ">";
  }

  // Lookup in catalog based on name
  InputInfo *datasetInfo =
      (this->catalogParser).getInputInfoIfKnown(datasetName);
  bool pluginExisted = true;

  if (!datasetInfo) {
    RecordType *rec = new RecordType();

    if (val.HasMember("schema")) {
      size_t attrNo = 1;
      for (const auto &attr : val["schema"].GetArray()) {
        assert(attr.IsObject());
        auto recAttr = parseRecordAttr(attr, datasetName, nullptr, attrNo++);

        LOG(INFO) << "Plugin Registered: " << *recAttr;

        rec->appendAttribute(recAttr);
      }
    }

    datasetInfo = new InputInfo();
    datasetInfo->exprType = new BagType(*rec);
    datasetInfo->path = datasetName;

    if (val.HasMember("schema")) {
      // Register it to make it visible to the plugin
      datasetInfo->oidType = nullptr;
      (this->catalogParser).setInputInfo(datasetName, datasetInfo);
    }

    pluginExisted = false;
  }

  // Dynamic allocation because I have to pass reference later on
  string *pathDynamicCopy = new string(datasetInfo->path);

  /* Retrieve RecordType */
  /* Extract inner type of collection */
  CollectionType *collType =
      dynamic_cast<CollectionType *>(datasetInfo->exprType);
  if (collType == nullptr) {
    string error_msg = string(
                           "[Plugin Parser: ] Cannot cast to collection "
                           "type. Original intended type: ") +
                       datasetInfo->exprType->getType();
    LOG(ERROR) << error_msg;
    throw runtime_error(string(error_msg));
  }
  /* For the current plugins, the expression type is unambiguously RecordType */
  const ExpressionType &nestedType = collType->getNestedType();
  const RecordType &recType_ = dynamic_cast<const RecordType &>(nestedType);
  // circumventing the presence of const
  RecordType *recType = new RecordType(recType_.getArgs());

  if (pgType == "csv") {
    //        cout<<"Original intended type: " <<
    //        datasetInfo.exprType->getType()<<endl; cout<<"File path: " <<
    //        datasetInfo.path<<endl;

    /* Projections come in an array of Record Attributes */
    assert(val.HasMember(keyProjections));
    assert(val[keyProjections].IsArray());

    vector<RecordAttribute *> projections;
    for (const auto &attr : val[keyProjections].GetArray()) {
      projections.push_back(parseRecordAttr(attr, {recType}));
    }

    assert(val.HasMember(keyLineHint));
    assert(val[keyLineHint].IsInt());
    int linehint = val[keyLineHint].GetInt();

    assert(val.HasMember(keyPolicy));
    assert(val[keyPolicy].IsInt());
    int policy = val[keyPolicy].GetInt();

    char delim = ',';
    if (val.HasMember(keyDelimiter)) {
      assert(val[keyDelimiter].IsString());
      delim = (val[keyDelimiter].GetString())[0];
    } else {
      string err =
          string("WARNING - NO DELIMITER SPECIFIED. FALLING BACK TO DEFAULT");
      LOG(WARNING) << err;
      cout << err << endl;
    }

    bool stringBrackets = true;
    if (val.HasMember(keyBrackets)) {
      assert(val[keyBrackets].IsBool());
      stringBrackets = val[keyBrackets].GetBool();
    }

    bool hasHeader = false;
    if (val.HasMember("hasHeader")) {
      assert(val["hasHeader"].IsBool());
      hasHeader = val["hasHeader"].GetBool();
    }

    newPg =
        new pm::CSVPlugin(this->ctx, *pathDynamicCopy, *recType, projections,
                          delim, linehint, policy, stringBrackets, hasHeader);
  } else if (pgType == "json") {
    assert(val.HasMember(keyLineHint));
    assert(val[keyLineHint].IsInt());
    int linehint = val[keyLineHint].GetInt();

    newPg = new jsonPipelined::JSONPlugin(this->ctx, *pathDynamicCopy,
                                          datasetInfo->exprType, linehint);
  } else if (pgType == "binrow") {
    assert(val.HasMember(keyProjections));
    assert(val[keyProjections].IsArray());

    vector<RecordAttribute *> projections;
    for (const auto &attr : val[keyProjections].GetArray()) {
      projections.push_back(parseRecordAttr(attr, {recType}));
    }

    newPg =
        new BinaryRowPlugin(this->ctx, *pathDynamicCopy, *recType, projections);
  } else if (pgType == "bincol") {
    assert(val.HasMember(keyProjections));
    assert(val[keyProjections].IsArray());

    vector<RecordAttribute *> projections;
    for (const auto &attr : val[keyProjections].GetArray()) {
      projections.push_back(parseRecordAttr(attr, {recType}));
    }

    bool sizeInFile = true;
    if (val.HasMember("sizeInFile")) {
      assert(val["sizeInFile"].IsBool());
      sizeInFile = val["sizeInFile"].GetBool();
    }
    newPg = new BinaryColPlugin(this->ctx, *pathDynamicCopy, *recType,
                                projections, sizeInFile);
  } else if (pgType == "block") {
    assert(val.HasMember(keyProjections));
    assert(val[keyProjections].IsArray());

    vector<RecordAttribute *> projections;
    for (const auto &attr : val[keyProjections].GetArray()) {
      projections.push_back(parseRecordAttr(attr, {recType}));
    }

    assert(dynamic_cast<ParallelContext *>(this->ctx));

    newPg = new BinaryBlockPlugin(dynamic_cast<ParallelContext *>(this->ctx),
                                  *pathDynamicCopy, *recType, projections);
  } else {
    assert(dynamic_cast<ParallelContext *>(this->ctx));

    typedef Plugin *(*plugin_creator_t)(ParallelContext *, std::string,
                                        RecordType,
                                        std::vector<RecordAttribute *> &);

    //    if (std::string{pgType} == "dynamic"){
    //      pgType = "block-elastic";
    //    }

    std::string conv = "create" + hyphenatedPluginToCamel(pgType) + "Plugin";

    std::cout << "PluginName: " << hyphenatedPluginToCamel(pgType) << std::endl;

    auto create = (plugin_creator_t)dlsym(handle, conv.c_str());

    if (!create) {
      newPg = Catalog::getInstance().getPlugin(*pathDynamicCopy);  // TODO fix
      if (!newPg) {
        auto err = "Unknown Plugin Type: " + pgType;
        LOG(ERROR) << err;
        throw runtime_error(err);
      }
    } else {
      assert(val.HasMember(keyProjections));
      assert(val[keyProjections].IsArray());

      vector<RecordAttribute *> projections;
      for (const auto &attr : val[keyProjections].GetArray()) {
        projections.push_back(parseRecordAttr(attr, {recType}));
      }

      newPg =
          create(dynamic_cast<ParallelContext *>(this->ctx), *pathDynamicCopy,
                 *recType, projections /*, const rapidjson::Value &val */);
      // FIXME: a better interface would be to also pass the current json value,
      //  so that plugins can read their own attributes.
    }
  }

  activePlugins.push_back(newPg);
  Catalog &catalog = Catalog::getInstance();
  catalog.registerPlugin(*pathDynamicCopy, newPg);
  datasetInfo->oidType = newPg->getOIDType();
  (this->catalogParser).setInputInfo(datasetName, datasetInfo);
  return newPg;
}

void CatalogParser::parseCatalogFile(const std::filesystem::path &file) {
  // key aliases
  const char *keyInputPath = "path";
  const char *keyExprType = "type";

  auto fsize = std::filesystem::file_size(file);

  auto f = std::filesystem::absolute(file).string();
  int fd = open(f.c_str(), O_RDONLY);
  if (fd < 0) {
    const auto err = "failed to open file: " + f;
    LOG(ERROR) << err;
    throw runtime_error(err);
  }
  auto bufJSON =
      (const char *)mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
  if (bufJSON == MAP_FAILED) {
    const auto err = "failed to mmap file: " + f;
    LOG(ERROR) << err;
    throw runtime_error(err);
  }

  rapidjson::Document document;  // Default template parameter uses UTF8 and
                                 // MemoryPoolAllocator.
  auto &parsed = document.Parse(bufJSON);
  if (parsed.HasParseError()) {
    auto ok = (rapidjson::ParseResult)parsed;
    fprintf(stderr, "[CatalogParser: ] Error parsing physical plan: %s (%lu)",
            RAPIDJSON_NAMESPACE::GetParseError_En(ok.Code()), ok.Offset());
    const char *err = "[CatalogParser: ] Error parsing physical plan";
    LOG(ERROR) << err << ": "
               << RAPIDJSON_NAMESPACE::GetParseError_En(ok.Code()) << "("
               << ok.Offset() << ")";
    throw runtime_error(err);
  }

  // Start plan traversal.
  assert(document.IsObject());

  for (const auto &member : document.GetObject()) {
    assert(member.value.IsObject());
    assert((member.value)[keyInputPath].IsString());
    string inputPath = ((member.value)[keyInputPath]).GetString();
    assert((member.value)[keyExprType].IsObject());
    ExpressionType *exprType =
        ExpressionParser{*this,
                         new RecordType{std::vector<RecordAttribute *>{}}}
            .parseExpressionType((member.value)[keyExprType]);
    auto *info = new InputInfo();
    info->exprType = exprType;
    info->path = inputPath;
    // Initialized by parsePlugin() later on
    info->oidType = nullptr;
    //            (this->inputs)[itr->name.GetString()] = info;
    (this->inputs)[info->path] = info;

    setInputInfo(info->path, info);
  }
}

void CatalogParser::clear() {
  auto it = inputs.begin();
  while (it != inputs.end()) {
    if (it->first.substr(0, 5) != "tpcc_") {
      LOG(WARNING) << "FIXME: CLEANING TABLE " << it->first;
      inputs.erase(it++);
    } else {
      LOG(WARNING) << "FIXME: KEEPING TABLE " << it->first;
      ++it;
    }
  }
  parseDir("inputs");
}

void CatalogParser::parseDir(const std::filesystem::path &dir) {
  LOG(INFO) << "Scanning for catalogs: " << dir;

  for (const auto &entry : std::filesystem::directory_iterator(dir)) {
    if (std::filesystem::is_directory(entry)) {
      parseDir(entry);
    } else if (entry.path().filename() == "catalog.json") {
      parseCatalogFile(entry);
    }
  }
}

/**
 * {"datasetname": {"path": "foo", "type": { ... } }
 */
CatalogParser::CatalogParser(const char *catalogPath, ParallelContext *context)
    : context(context) {
  parseDir(catalogPath);
}

CatalogParser &CatalogParser::getInstance() {
  static CatalogParser instance{"inputs", nullptr};
  return instance;
}

InputInfo *CatalogParser::getOrCreateInputInfo(string inputName) {
  return getOrCreateInputInfo(inputName, context);
}

InputInfo *CatalogParser::getOrCreateInputInfo(string inputName,
                                               ParallelContext *context) {
  InputInfo *ret = getInputInfoIfKnown(inputName);

  if (!ret) {
    RecordType *rec = new RecordType();

    ret = new InputInfo();
    ret->exprType = new BagType(*rec);
    ret->path = inputName;

    Catalog &catalog = Catalog::getInstance();

    assert(
        context &&
        "A ParallelContext is required to register relationships on the fly");
    vector<RecordAttribute *> projs;
    //    Plugin *newPg =
    //        new pm::CSVPlugin(context, inputName, *rec, projs, ',', 10, 1,
    //        false);
    Plugin *newPg = new BinaryBlockPlugin(context, inputName, *rec, projs);
    catalog.registerPlugin(inputName, newPg);
    ret->oidType = newPg->getOIDType();

    setInputInfo(inputName, ret);
  }

  return ret;
}

void PlanExecutor::compileAndLoad() { return ctx->compileAndLoad(); }

std::ostream &operator<<(std::ostream &out, const rapidjson::Value &val) {
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  val.Accept(writer);
  out << buffer.GetString();
  return out;
}
