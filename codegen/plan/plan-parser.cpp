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

#include "plan/plan-parser.hpp"

#include <dlfcn.h>

#include "plugins/binary-block-plugin.hpp"
#ifndef NCUDA
#include "operators/cpu-to-gpu.hpp"
#include "operators/gpu/gpu-hash-group-by-chained.hpp"
#include "operators/gpu/gpu-hash-join-chained.hpp"
#include "operators/gpu/gpu-hash-rearrange.hpp"
#include "operators/gpu/gpu-partitioned-hash-join-chained.hpp"
#include "operators/gpu/gpu-reduce.hpp"
#include "operators/gpu/gpu-to-cpu.hpp"
#endif
#include "operators/block-to-tuples.hpp"
#include "operators/dict-scan.hpp"
#include "operators/flush.hpp"
#include "operators/gpu/gpu-materializer-expr.hpp"
#include "operators/gpu/gpu-sort.hpp"
#include "operators/hash-group-by-chained.hpp"
#include "operators/hash-join-chained.hpp"
#include "operators/hash-rearrange.hpp"
#include "operators/mem-broadcast-device.hpp"
#include "operators/mem-move-device.hpp"
#include "operators/mem-move-local-to.hpp"
#include "operators/outer-unnest.hpp"
#include "operators/packet-zip.hpp"
#include "operators/print.hpp"
#include "operators/project.hpp"
#include "operators/radix-join.hpp"
#include "operators/radix-nest.hpp"
#include "operators/reduce-opt.hpp"
#include "operators/root.hpp"
#include "operators/router.hpp"
#include "operators/scan.hpp"
#include "operators/select.hpp"
#include "operators/sort.hpp"
#include "operators/split.hpp"
#include "operators/unionall.hpp"
#include "operators/unnest.hpp"
#include "rapidjson/error/en.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

// std::string hyphenatedPluginToCamel(const char *name) {
//   size_t len = strlen(name);
//   bool make_capital = true;
//   char conv[len + 1];
//   size_t j = 0;
//   for (size_t i = 0; i < len - 1; ++i) {
//     if (name[i] == '-') {
//       ++i;
//       make_capital = true;
//     }
//     if (make_capital) {
//       conv[j++] = name[i];
//     }
//     make_capital = false;
//   }
//   conv[j] = '\0';
//   return {conv};
// }

std::string hyphenatedPluginToCamel(const char *line) {
  size_t len = strlen(line);
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
    : PlanExecutor(planPath, cat, moduleName, prepareContext(moduleName)) {}

PlanExecutor::PlanExecutor(const char *planPath, CatalogParser &cat,
                           const char *moduleName, Context *ctx)
    : handle(dlopen(nullptr, 0)),
      exprParser(cat),
      planPath(planPath),
      moduleName(moduleName),
      catalogParser(cat),
      ctx(ctx) {
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
    fprintf(stderr, "JSON parse error: %s (%lu)",
            RAPIDJSON_NAMESPACE::GetParseError_En(ok.Code()), ok.Offset());
    const char *err =
        "[PlanExecutor: ] Error parsing physical plan (JSON parsing error)";
    LOG(ERROR) << err;
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
  Operator *planRootOp = parseOperator(doc);

  planRootOp->produce();

  // Run function
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

Operator *PlanExecutor::parseOperator(const rapidjson::Value &val) {
  const char *keyPg = "plugin";
  const char *keyOp = "operator";

  assert(val.HasMember(keyOp));
  assert(val[keyOp].IsString());
  const char *opName = val["operator"].GetString();

  Operator *newOp = nullptr;

  if (strcmp(opName, "reduce") == 0) {
    /* "Multi - reduce"! */
    /* parse operator input */
    Operator *childOp = parseOperator(val["input"]);

    /* get monoid(s) */
    assert(val.HasMember("accumulator"));
    assert(val["accumulator"].IsArray());
    vector<Monoid> accs;
    for (const auto &accm : val["accumulator"].GetArray()) {
      assert(accm.IsString());
      Monoid acc = parseAccumulator(accm.GetString());
      accs.push_back(acc);
    }

    /*
     * parse output expressions
     * XXX Careful: Assuming numerous output expressions!
     */
    assert(val.HasMember("e"));
    assert(val["e"].IsArray());
    std::vector<expression_t> e;
    e.reserve(val["e"].Size());
    for (const auto &v : val["e"].GetArray()) {
      e.emplace_back(parseExpression(v));
    }

    /* parse filtering expression */
    assert(val.HasMember("p"));
    assert(val["p"].IsObject());
    expression_t p = parseExpression(val["p"]);

    /* 'Multi-reduce' used */
#ifndef NCUDA
    if (val.HasMember("gpu") && val["gpu"].GetBool()) {
      assert(dynamic_cast<ParallelContext *>(this->ctx));
      newOp = new opt::GpuReduce(accs, e, p, childOp,
                                 dynamic_cast<ParallelContext *>(this->ctx));
    } else {
#endif
      newOp =
          new opt::Reduce(accs, e, p, childOp, this->ctx, false, moduleName);
#ifndef NCUDA
    }
#endif
    childOp->setParent(newOp);
  } else if (strcmp(opName, "print") == 0) {
    /* "Multi - reduce"! */
    if (val.HasMember("plugin")) {
      assert(val["plugin"].IsObject());
      parsePlugin(val["plugin"]);
    }

    /* parse operator input */
    Operator *childOp = parseOperator(val["input"]);

    /*
     * parse output expressions
     * XXX Careful: Assuming numerous output expressions!
     */
    assert(val.HasMember("e"));
    assert(val["e"].IsArray());
    std::vector<expression_t> e;
    e.reserve(val["e"].Size());
    for (const auto &v : val["e"].GetArray()) {
      e.emplace_back(parseExpression(v));
      assert(e.back().isRegistered());
    }

    newOp = new Flush(e, childOp, this->ctx, moduleName);
    childOp->setParent(newOp);
  } else if (strcmp(opName, "sort") == 0) {
    /* "Multi - reduce"! */
    /* parse operator input */
    Operator *childOp = parseOperator(val["input"]);

    /*
     * parse output expressions
     * XXX Careful: Assuming numerous output expressions!
     */
    assert(val.HasMember("e"));
    assert(val["e"].IsArray());
    vector<expression_t> e;
    vector<direction> d;
    vector<RecordAttribute *> recattr;
    for (const auto &v : val["e"].GetArray()) {
      assert(v.IsObject());
      assert(v.HasMember("expression"));
      assert(v["expression"].IsObject());
      expression_t outExpr = parseExpression(v["expression"]);
      e.emplace_back(outExpr);
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

      recattr.emplace_back(new RecordAttribute{outExpr.getRegisteredAs()});
    }

    std::string relName = e[0].getRegisteredRelName();

    InputInfo *datasetInfo =
        (this->catalogParser).getOrCreateInputInfo(relName);
    RecordType *rec = new RecordType{dynamic_cast<const RecordType &>(
        dynamic_cast<CollectionType *>(datasetInfo->exprType)
            ->getNestedType())};
    RecordAttribute *reg_as =
        new RecordAttribute(relName, "__sorted", new RecordType(recattr));
    std::cout << "Registered: " << reg_as->getRelationName() << "."
              << reg_as->getAttrName() << std::endl;
    rec->appendAttribute(reg_as);

    datasetInfo->exprType = new BagType{*rec};

#ifndef NCUDA
    if (val.HasMember("gpu") && val["gpu"].GetBool()) {
      gran_t granularity = gran_t::GRID;

      if (val.HasMember("granularity")) {
        assert(val["granularity"].IsString());
        std::string gran = val["granularity"].GetString();
        std::transform(gran.begin(), gran.end(), gran.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (gran == "grid")
          granularity = gran_t::GRID;
        else if (gran == "block")
          granularity = gran_t::BLOCK;
        else if (gran == "thread")
          granularity = gran_t::THREAD;
        else
          assert(false && "granularity must be one of GRID, BLOCK, THREAD");
      }

      assert(dynamic_cast<ParallelContext *>(this->ctx));
      newOp = new GpuSort(childOp, dynamic_cast<ParallelContext *>(this->ctx),
                          e, d, granularity);
    } else {
#endif
      newOp =
          new Sort(childOp, dynamic_cast<ParallelContext *>(this->ctx), e, d);
#ifndef NCUDA
    }
#endif
    childOp->setParent(newOp);
  } else if (strcmp(opName, "project") == 0) {
    /* "Multi - reduce"! */
    /* parse operator input */
    Operator *childOp = parseOperator(val["input"]);

    /*
     * parse output expressions
     * XXX Careful: Assuming numerous output expressions!
     */
    assert(val.HasMember("e"));
    assert(val["e"].IsArray());
    std::vector<expression_t> e;
    for (const auto &v : val["e"].GetArray()) {
      e.emplace_back(parseExpression(v));
    }

    assert(val.HasMember("relName"));
    assert(val["relName"].IsString());

    newOp = new Project(e, val["relName"].GetString(), childOp, this->ctx);
    childOp->setParent(newOp);
  } else if (strcmp(opName, "unnest") == 0) {
    /* parse operator input */
    Operator *childOp = parseOperator(val["input"]);

    /* parse filtering expression */
    assert(val.HasMember("p"));
    assert(val["p"].IsObject());
    expression_t p = parseExpression(val["p"]);

    /* parse path expression */
    assert(val.HasMember("path"));
    assert(val["path"].IsObject());

    assert(val["path"]["e"].IsObject());
    auto exprToUnnest = parseExpression(val["path"]["e"]);
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

    newOp = new Unnest(p, projPath, childOp);
    childOp->setParent(newOp);

    auto inputInfo = new InputInfo();
    inputInfo->exprType = new BagType(
        dynamic_cast<const CollectionType &>(*proj->getExpressionType())
            .getNestedType());
    inputInfo->path = projPath.toString();
    inputInfo->oidType = projPath.getRelevantPlugin()->getOIDType();
    catalogParser.setInputInfo(projPath.toString(), inputInfo);
  } else if (strcmp(opName, "outer_unnest") == 0) {
    /* parse operator input */
    Operator *childOp = parseOperator(val["input"]);

    /* parse filtering expression */
    assert(val.HasMember("p"));
    assert(val["p"].IsObject());
    auto p = parseExpression(val["p"]);

    /* parse path expression */
    assert(val.HasMember("path"));
    assert(val["path"].IsObject());

    assert(val["path"]["name"].IsString());
    string pathAlias = val["path"]["name"].GetString();

    assert(val["path"]["e"].IsObject());
    auto exprToUnnest = parseExpression(val["path"]["e"]);
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
  } else if (strcmp(opName, "groupby") == 0 ||
             strcmp(opName, "hashgroupby-chained") == 0) {
    /* parse operator input */
    assert(val.HasMember("input"));
    assert(val["input"].IsObject());
    Operator *child = parseOperator(val["input"]);

#ifndef NCUDA
    // if (val.HasMember("gpu") && val["gpu"].GetBool()){
    assert(val.HasMember("hash_bits"));
    assert(val["hash_bits"].IsInt());
    int hash_bits = val["hash_bits"].GetInt();

    // assert(val.HasMember("w"));
    // assert(val["w"].IsArray());
    // vector<size_t> widths;

    // const rapidjson::Value& wJSON = val["w"];
    // for (SizeType i = 0; i < wJSON.Size(); i++){
    //     assert(wJSON[i].IsInt());
    //     widths.push_back(wJSON[i].GetInt());
    // }

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
      auto outExpr = parseExpression(v["e"]);

      e.emplace_back(outExpr, v["packet"].GetInt(), v["offset"].GetInt(),
                     parseAccumulator(v["m"].GetString()));
    }

    assert(val.HasMember("k"));
    assert(val["k"].IsArray());
    vector<expression_t> key_expr;
    for (const auto &k : val["k"].GetArray()) {
      key_expr.emplace_back(parseExpression(k));
    }

    assert(val.HasMember("maxInputSize"));
    assert(val["maxInputSize"].IsUint64());

    size_t maxInputSize = val["maxInputSize"].GetUint64();

    assert(dynamic_cast<ParallelContext *>(this->ctx));
    // newOp = new GpuHashGroupByChained(e, widths, key_expr, child, hash_bits,
    //                     dynamic_cast<ParallelContext *>(this->ctx),
    //                     maxInputSize);

    if (val.HasMember("gpu") && val["gpu"].GetBool()) {
      newOp = new GpuHashGroupByChained(
          e, key_expr, child, hash_bits,
          dynamic_cast<ParallelContext *>(this->ctx), maxInputSize);
    } else {
      newOp = new HashGroupByChained(e, key_expr, child, hash_bits,
                                     dynamic_cast<ParallelContext *>(this->ctx),
                                     maxInputSize);
    }
    // } else {
#endif
    //         assert(val.HasMember("hash_bits"));
    //         assert(val["hash_bits"].IsInt());
    //         int hash_bits = val["hash_bits"].GetInt();

    //         // assert(val.HasMember("w"));
    //         // assert(val["w"].IsArray());
    //         // vector<size_t> widths;

    //         // const rapidjson::Value& wJSON = val["w"];
    //         // for (SizeType i = 0; i < wJSON.Size(); i++){
    //         //     assert(wJSON[i].IsInt());
    //         //     widths.push_back(wJSON[i].GetInt());
    //         // }

    //         /*
    //          * parse output expressions
    //          * XXX Careful: Assuming numerous output expressions!
    //          */
    //         assert(val.HasMember("e"));
    //         assert(val["e"].IsArray());
    //         // vector<GpuAggrMatExpr> e;
    //         const rapidjson::Value& aggrJSON = val["e"];
    //         vector<Monoid> accs;
    //         vector<string> aggrLabels;
    //         vector<expressions::Expression*> outputExprs;
    //         vector<expressions::Expression*> exprsToMat;
    //         vector<materialization_mode> outputModes;
    //         map<string, RecordAttribute*> mapOids;
    //         vector<RecordAttribute*> fieldsToMat;
    //         for (SizeType i = 0; i < aggrJSON.Size(); i++){
    //             assert(aggrJSON[i].HasMember("e"     ));
    //             assert(aggrJSON[i].HasMember("m"     ));
    //             assert(aggrJSON[i]["m"].IsString()    );
    //             assert(aggrJSON[i].HasMember("packet"));
    //             assert(aggrJSON[i]["packet"].IsInt()  );
    //             assert(aggrJSON[i].HasMember("offset"));
    //             assert(aggrJSON[i]["offset"].IsInt()  );
    //             expressions::Expression *outExpr =
    //             parseExpression(aggrJSON[i]["e"]);

    //             // e.emplace_back(outExpr, aggrJSON[i]["packet"].GetInt(),
    //             aggrJSON[i]["offset"].GetInt(), );

    //             aggrLabels.push_back(outExpr->getRegisteredAttrName());
    //             accs.push_back(parseAccumulator(aggrJSON[i]["m"].GetString()));

    //             outputExprs.push_back(outExpr);

    //             //XXX STRONG ASSUMPTION: Expression is actually a record
    //             projection! expressions::RecordProjection *proj =
    //                     dynamic_cast<expressions::RecordProjection
    //                     *>(outExpr);

    //             if (proj == nullptr) {
    //                 if (outExpr->getTypeID() != expressions::CONSTANT){
    //                     string error_msg = string(
    //                             "[Nest: ] Cannot cast to rec projection.
    //                             Original: ")
    //                             + outExpr->getExpressionType()->getType();
    //                     LOG(ERROR)<< error_msg;
    //                     throw runtime_error(string(error_msg));
    //                 }
    //             } else {
    //                 exprsToMat.push_back(outExpr);
    //                 outputModes.insert(outputModes.begin(), EAGER);

    //                 //Added in 'wanted fields'
    //                 RecordAttribute *recAttr = new
    //                 RecordAttribute(proj->getAttribute());
    //                 fieldsToMat.push_back(new
    //                 RecordAttribute(outExpr->getRegisteredAs()));

    //                 string relName = recAttr->getRelationName();
    //                 if (mapOids.find(relName) == mapOids.end()) {
    //                     InputInfo *datasetInfo =
    //                             (this->catalogParser).getInputInfo(relName);
    //                     RecordAttribute *oid =
    //                             new
    //                             RecordAttribute(recAttr->getRelationName(),
    //                             activeLoop, datasetInfo->oidType);
    //                     mapOids[relName] = oid;
    //                     expressions::RecordProjection *oidProj =
    //                             new expressions::RecordProjection(outExpr,
    //                             *oid);
    //                     //Added in 'wanted expressions'
    //                     LOG(INFO)<< "[Plan Parser: ] Injecting OID for " <<
    //                     relName; std::cout << "[Plan Parser: ] Injecting OID
    //                     for " << relName << std::endl;
    //                     /* ORDER OF expression fields matters!! OIDs need to
    //                     be placed first! */
    //                     exprsToMat.insert(exprsToMat.begin(), oidProj);
    //                     outputModes.insert(outputModes.begin(), EAGER);
    //                 }
    //             }
    //         }

    //         /* Predicate */
    //         expressions::Expression * predExpr = new
    //         expressions::BoolConstant(true);

    //         assert(val.HasMember("k"));
    //         assert(val["k"].IsArray());
    //         vector<expressions::Expression *> key_expr;
    //         const rapidjson::Value& keyJSON = val["k"];
    //         for (SizeType i = 0; i < keyJSON.Size(); i++){
    //             key_expr.emplace_back(parseExpression(keyJSON[i]));
    //         }

    //         assert(val.HasMember("maxInputSize"));
    //         assert(val["maxInputSize"].IsUint64());

    //         size_t maxInputSize = val["maxInputSize"].GetUint64();

    //         const char *keyGroup = "f";
    //         const char *keyNull  = "g";
    //         const char *keyPred  = "p";
    //         const char *keyExprs = "e";
    //         const char *keyAccum = "accumulator";
    //         /* Physical Level Info */
    //         const char *keyAggrNames = "aggrLabels";
    //         //Materializer
    //         const char *keyMat = "fields";

    //         /* Group By */
    //         // assert(key_expr.size() == 1);
    //         // expressions::Expression *groupByExpr = key_expr[0];

    //         for (const auto &e: key_expr){
    //             //XXX STRONG ASSUMPTION: Expression is actually a record
    //             projection! expressions::RecordProjection *proj =
    //                     dynamic_cast<expressions::RecordProjection *>(e);

    //             if (proj == nullptr) {
    //                 if (e->getTypeID() != expressions::CONSTANT){
    //                     string error_msg = string(
    //                             "[Nest: ] Cannot cast to rec projection.
    //                             Original: ")
    //                             + e->getExpressionType()->getType();
    //                     LOG(ERROR)<< error_msg;
    //                     throw runtime_error(string(error_msg));
    //                 }
    //             } else {
    //                 exprsToMat.push_back(e);
    //                 outputModes.insert(outputModes.begin(), EAGER);

    //                 //Added in 'wanted fields'
    //                 RecordAttribute *recAttr = new
    //                 RecordAttribute(proj->getAttribute());
    //                 fieldsToMat.push_back(new
    //                 RecordAttribute(e->getRegisteredAs()));

    //                 string relName = recAttr->getRelationName();
    //                 if (mapOids.find(relName) == mapOids.end()) {
    //                     InputInfo *datasetInfo =
    //                             (this->catalogParser).getInputInfo(relName);
    //                     RecordAttribute *oid =
    //                             new
    //                             RecordAttribute(recAttr->getRelationName(),
    //                             activeLoop, datasetInfo->oidType);
    //                     std::cout << datasetInfo->oidType->getType() <<
    //                     std::endl; mapOids[relName] = oid;
    //                     expressions::RecordProjection *oidProj =
    //                             new expressions::RecordProjection(e, *oid);
    //                     //Added in 'wanted expressions'
    //                     LOG(INFO)<< "[Plan Parser: ] Injecting OID for " <<
    //                     relName; std::cout << "[Plan Parser: ] Injecting OID
    //                     for " << relName << std::endl;
    //                     /* ORDER OF expression fields matters!! OIDs need to
    //                     be placed first! */
    //                     exprsToMat.insert(exprsToMat.begin(), oidProj);
    //                     outputModes.insert(outputModes.begin(), EAGER);
    //                 }
    //             }
    //         }
    //         /* Null-to-zero Checks */
    //         //FIXME not used in radix nest yet!
    //         // assert(val.HasMember(keyNull));
    //         // assert(val[keyNull].IsObject());
    //         expressions::Expression *nullsToZerosExpr =
    //         nullptr;//parseExpression(val[keyNull]);

    //         /* Output aggregate expression(s) */
    //         // assert(val.HasMember(keyExprs));
    //         // assert(val[keyExprs].IsArray());
    //         // vector<expressions::Expression*> outputExprs;
    //         // for (SizeType i = 0; i < val[keyExprs].Size(); i++) {
    //         //     expressions::Expression *expr =
    //         parseExpression(val[keyExprs][i]);
    //         // }

    //         /*
    //          * *** WHAT TO MATERIALIZE ***
    //          * XXX v0: JSON file contains a list of **RecordProjections**
    //          * EXPLICIT OIDs injected by PARSER (not in json file by default)
    //          * XXX Eager materialization atm
    //          *
    //          * XXX Why am I not using minimal constructor for materializer
    //          yet, as use cases do?
    //          *     -> Because then I would have to encode the OID type in
    //          JSON -> can be messy
    //          */
    //         vector<RecordAttribute*> oids;
    //         MapToVec(mapOids, oids);
    //         /* FIXME This constructor breaks nest use cases that trigger
    //         caching */
    //         /* Check similar hook in radix-nest.cpp */
    // //        Materializer *mat =
    // //                new Materializer(fieldsToMat, exprsToMat, oids,
    // outputModes);
    //         // for (const auto &e: exprsToMat) {
    //         //     std::cout << "mat: " << e->getRegisteredRelName() << " "
    //         << e->getRegisteredAttrName() << std::endl;
    //         // }
    //         Materializer* matCoarse = new Materializer(fieldsToMat,
    //         exprsToMat,
    //                 oids, outputModes);

    //         //Put operator together
    //         auto opLabel = key_expr[0]->getRegisteredRelName();
    //         std::cout << "regRelNAme" << opLabel << std::endl;
    //         newOp = new radix::Nest(this->ctx, accs, outputExprs, aggrLabels,
    //         predExpr,
    //                 key_expr, nullsToZerosExpr, child, opLabel, *matCoarse);
#ifndef NCUDA
    // }
#endif
    child->setParent(newOp);
  } else if (strcmp(opName, "out-of-gpu-join") == 0) {
    /* parse operator input */
    assert(val.HasMember("probe_input"));
    assert(val["probe_input"].IsObject());
    Operator *probe_op = parseOperator(val["probe_input"]);

    /* parse operator input */
    assert(val.HasMember("build_input"));
    assert(val["build_input"].IsObject());
    Operator *build_op = parseOperator(val["build_input"]);

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
      auto outExpr = parseExpression(e["original"]);
      build_expr.emplace_back(outExpr);

      assert(e["original"]["attribute"].IsObject());
      RecordAttribute *recAttr =
          this->parseRecordAttr(e["original"]["attribute"]);
      build_attr.push_back(recAttr);
      build_attr_block.push_back(new RecordAttribute(*recAttr, true));

      assert(e["hashed"].IsObject());
      auto outHashedExpr = parseExpression(e["hashed"]);
      build_hashed_expr.emplace_back(outHashedExpr);

      assert(e["hashed-block"].IsObject());
      auto outHashedBlockExpr = parseExpression(e["hashed-block"]);
      build_hashed_expr_block.emplace_back(outHashedBlockExpr);

      assert(e["hashed"]["attribute"].IsObject());
      RecordAttribute *recHashedAttr =
          this->parseRecordAttr(e["hashed"]["register_as"]);
      build_hashed_attr.push_back(recHashedAttr);
      build_hashed_attr_block.push_back(
          new RecordAttribute(*recHashedAttr, true));

      assert(e["join"].IsObject());
      assert(e["join"].HasMember("e"));
      assert(e["join"].HasMember("packet"));
      assert(e["join"]["packet"].IsInt());
      assert(e["join"].HasMember("offset"));
      assert(e["join"]["offset"].IsInt());
      auto outJoinExpr = parseExpression(e["join"]["e"]);
      build_join_expr.emplace_back(outJoinExpr, e["join"]["packet"].GetInt(),
                                   e["join"]["offset"].GetInt());

      assert(e["join"]["e"]["attribute"].IsObject());
      RecordAttribute *recJoinAttr =
          this->parseRecordAttr(e["join"]["e"]["attribute"]);
      build_join_attr.push_back(recJoinAttr);
      build_join_attr_block.push_back(new RecordAttribute(*recJoinAttr, true));
      auto outPreJoinExpr = parseExpression(e["join"]["e"]);
      outPreJoinExpr.registerAs(recJoinAttr);
      build_prejoin_expr.push_back(outPreJoinExpr);
    }

    assert(val.HasMember("build_hash"));
    RecordAttribute *build_hash_attr =
        this->parseRecordAttr(val["build_hash"]["attribute"]);

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
      auto outExpr = parseExpression(e["original"]);
      probe_expr.emplace_back(outExpr);

      assert(e["original"]["attribute"].IsObject());
      RecordAttribute *recAttr =
          this->parseRecordAttr(e["original"]["attribute"]);
      probe_attr.push_back(recAttr);
      probe_attr_block.push_back(new RecordAttribute(*recAttr, true));

      assert(e["hashed"].IsObject());
      auto outHashedExpr = parseExpression(e["hashed"]);
      probe_hashed_expr.emplace_back(outHashedExpr);

      assert(e["hashed-block"].IsObject());
      auto outHashedBlockExpr = parseExpression(e["hashed-block"]);
      probe_hashed_expr_block.emplace_back(outHashedBlockExpr);

      assert(e["hashed"]["attribute"].IsObject());
      RecordAttribute *recHashedAttr =
          this->parseRecordAttr(e["hashed"]["register_as"]);
      probe_hashed_attr.push_back(recHashedAttr);
      probe_hashed_attr_block.push_back(
          new RecordAttribute(*recHashedAttr, true));

      assert(e["join"].IsObject());
      assert(e["join"].HasMember("e"));
      assert(e["join"].HasMember("packet"));
      assert(e["join"]["packet"].IsInt());
      assert(e["join"].HasMember("offset"));
      assert(e["join"]["offset"].IsInt());
      auto outJoinExpr = parseExpression(e["join"]["e"]);
      probe_join_expr.emplace_back(outJoinExpr, e["join"]["packet"].GetInt(),
                                   e["join"]["offset"].GetInt());

      assert(e["join"]["e"]["attribute"].IsObject());
      RecordAttribute *recJoinAttr =
          this->parseRecordAttr(e["join"]["e"]["attribute"]);
      probe_join_attr.push_back(recJoinAttr);
      probe_join_attr_block.push_back(new RecordAttribute(*recJoinAttr, true));
      auto outPreJoinExpr = parseExpression(e["join"]["e"]);
      outPreJoinExpr.registerAs(recJoinAttr);
      probe_prejoin_expr.push_back(outPreJoinExpr);
    }

    assert(val.HasMember("probe_hash"));
    RecordAttribute *probe_hash_attr =
        this->parseRecordAttr(val["probe_hash"]["attribute"]);

    assert(val.HasMember("probe_w"));
    assert(val["probe_w"].IsArray());
    vector<size_t> probe_widths;

    for (const auto &w : val["build_w"].GetArray()) {
      assert(w.IsInt());
      probe_widths.push_back(w.GetInt());
    }

    Router *xch_build =
        new Router(build_op, (ParallelContext *)ctx,
                   DegreeOfParallelism{numPartitioners}, build_attr_block,
                   slack, std::nullopt, RoutingPolicy::LOCAL, DeviceType::CPU);
    build_op->setParent(xch_build);
    Operator *btt_build = new BlockToTuples(xch_build, (ParallelContext *)ctx,
                                            build_expr, false, gran_t::THREAD);
    xch_build->setParent(btt_build);
    Operator *part_build =
        new HashRearrange(btt_build, (ParallelContext *)ctx, numOfBuckets,
                          build_expr, build_expr[0], build_hash_attr);
    btt_build->setParent(part_build);
    build_attr_block.push_back(build_hash_attr);
    Router *xch_build2 =
        new Router(part_build, (ParallelContext *)ctx, DegreeOfParallelism{1},
                   build_attr_block, slack, std::nullopt, RoutingPolicy::LOCAL,
                   DeviceType::GPU);
    part_build->setParent(xch_build2);

    Router *xch_probe =
        new Router(probe_op, (ParallelContext *)ctx,
                   DegreeOfParallelism{numPartitioners}, probe_attr_block,
                   slack, std::nullopt, RoutingPolicy::LOCAL, DeviceType::CPU);
    probe_op->setParent(xch_probe);
    Operator *btt_probe = new BlockToTuples(xch_probe, (ParallelContext *)ctx,
                                            probe_expr, false, gran_t::THREAD);
    xch_probe->setParent(btt_probe);
    Operator *part_probe =
        new HashRearrange(btt_probe, (ParallelContext *)ctx, numOfBuckets,
                          probe_expr, probe_expr[0], probe_hash_attr);
    btt_probe->setParent(part_probe);
    probe_attr_block.push_back(probe_hash_attr);
    Router *xch_probe2 =
        new Router(part_probe, (ParallelContext *)ctx, DegreeOfParallelism{1},
                   probe_attr_block, slack, std::nullopt, RoutingPolicy::LOCAL,
                   DeviceType::GPU);
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

    Router *xch_proc =
        new Router(coord, (ParallelContext *)ctx,
                   DegreeOfParallelism{numConcurrent}, f_atts_target_v, slack,
                   expr_target, RoutingPolicy::HASH_BASED, DeviceType::GPU);
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

    Operator *mml_build = new MemMoveLocalTo(fwd_build, (ParallelContext *)ctx,
                                             build_hashed_attr_block, 4);
    fwd_build->setParent(mml_build);
    Operator *mmd_build = new MemMoveDevice(mml_build, (ParallelContext *)ctx,
                                            build_hashed_attr_block, 4, false);
    mml_build->setParent(mmd_build);
    Operator *ctg_build = new CpuToGpu(mmd_build, (ParallelContext *)ctx,
                                       build_hashed_attr_block);
    mmd_build->setParent(ctg_build);
    Operator *btt_build2 =
        new BlockToTuples(ctg_build, (ParallelContext *)ctx, build_prejoin_expr,
                          true, gran_t::GRID);
    ctg_build->setParent(btt_build2);
    HashPartitioner *hpart1 = new HashPartitioner(
        build_join_expr, build_widths, build_prejoin_expr[0], btt_build2,
        (ParallelContext *)ctx, maxBuildInputSize, 13, "partition_hash_1");
    btt_build2->setParent(hpart1);

    ZipForward *fwd_probe =
        new ZipForward(attr_target, initiator, (ParallelContext *)ctx,
                       probe_hashed_expr, "forwarder", coord->getStateRight());

    Operator *mml_probe = new MemMoveLocalTo(fwd_probe, (ParallelContext *)ctx,
                                             probe_hashed_attr_block, 4);
    fwd_probe->setParent(mml_probe);
    Operator *mmd_probe = new MemMoveDevice(mml_probe, (ParallelContext *)ctx,
                                            probe_hashed_attr_block, 4, false);
    mml_probe->setParent(mmd_probe);
    Operator *ctg_probe = new CpuToGpu(mmd_probe, (ParallelContext *)ctx,
                                       probe_hashed_attr_block);
    mmd_probe->setParent(ctg_probe);
    Operator *btt_probe2 =
        new BlockToTuples(ctg_probe, (ParallelContext *)ctx, probe_prejoin_expr,
                          true, gran_t::GRID);
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
  } else if (strcmp(opName, "partitioned-hashjoin-chained") == 0) {
    /* parse operator input */
    assert(val.HasMember("probe_input"));
    assert(val["probe_input"].IsObject());
    Operator *probe_op = parseOperator(val["probe_input"]);
    /* parse operator input */
    assert(val.HasMember("build_input"));
    assert(val["build_input"].IsObject());
    Operator *build_op = parseOperator(val["build_input"]);

    assert(val.HasMember("build_k"));
    auto build_key_expr = parseExpression(val["build_k"]);

    assert(val.HasMember("probe_k"));
    auto probe_key_expr = parseExpression(val["probe_k"]);

    std::optional<expression_t> build_minorkey_expr{
        (val.HasMember("build_k_minor"))
            ? std::make_optional(parseExpression(val["build_k_minor"]))
            : std::nullopt};

    std::optional<expression_t> probe_minorkey_expr{
        (val.HasMember("probe_k_minor"))
            ? std::make_optional(parseExpression(val["probe_k_minor"]))
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
      auto outExpr = parseExpression(e["e"]);

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
      auto outExpr = parseExpression(e["e"]);

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

  } else if (strcmp(opName, "hashjoin-chained") == 0) {
    /* parse operator input */
    assert(val.HasMember("probe_input"));
    assert(val["probe_input"].IsObject());
    Operator *probe_op = parseOperator(val["probe_input"]);
    /* parse operator input */
    assert(val.HasMember("build_input"));
    assert(val["build_input"].IsObject());
    Operator *build_op = parseOperator(val["build_input"]);

    assert(val.HasMember("build_k"));
    auto build_key_expr = parseExpression(val["build_k"]);

    assert(val.HasMember("probe_k"));
    auto probe_key_expr = parseExpression(val["probe_k"]);

    // #ifndef NCUDA
    //         if (val.HasMember("gpu") && val["gpu"].GetBool()){
    assert(val.HasMember("hash_bits"));
    assert(val["hash_bits"].IsInt());
    int hash_bits = val["hash_bits"].GetInt();

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
      auto outExpr = parseExpression(e["e"]);

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
      auto outExpr = parseExpression(e["e"]);

      probe_e.emplace_back(outExpr, e["packet"].GetInt(), e["offset"].GetInt());
    }

    assert(val.HasMember("maxBuildInputSize"));
    assert(val["maxBuildInputSize"].IsUint64());

    size_t maxBuildInputSize = val["maxBuildInputSize"].GetUint64();

    assert(dynamic_cast<ParallelContext *>(this->ctx));
#ifndef NCUDA
    if (val.HasMember("gpu") && val["gpu"].GetBool()) {
      newOp = new GpuHashJoinChained(
          build_e, build_widths, build_key_expr, build_op, probe_e,
          probe_widths, probe_key_expr, probe_op, hash_bits,
          dynamic_cast<ParallelContext *>(this->ctx), maxBuildInputSize);
    } else {
#endif
      newOp = new HashJoinChained(
          build_e, build_widths, build_key_expr, build_op, probe_e,
          probe_widths, probe_key_expr, probe_op, hash_bits,
          dynamic_cast<ParallelContext *>(this->ctx), maxBuildInputSize);
#ifndef NCUDA
    }
#endif
    //         } else {
    // #endif
    //             expressions::BinaryExpression *predExpr = new
    //             expressions::EqExpression(build_key_expr, probe_key_expr);

    //             /*
    //              * *** WHAT TO MATERIALIZE ***
    //              * XXX v0: JSON file contains a list of **RecordProjections**
    //              * EXPLICIT OIDs injected by PARSER (not in json file by
    //              default)
    //              * XXX Eager materialization atm
    //              *
    //              * XXX Why am I not using minimal constructor for
    //              materializer yet, as use cases do?
    //              *     -> Because then I would have to encode the OID type in
    //              JSON -> can be messy
    //              */

    //             //LEFT SIDE

    //             /*
    //              * parse output expressions
    //              * XXX Careful: Assuming numerous output expressions!
    //              */
    //             assert(val.HasMember("build_e"));
    //             assert(val["build_e"].IsArray());
    //             vector<expressions::Expression *> exprBuild   ;
    //             vector<RecordAttribute         *> fieldsBuild ;
    //             map<string, RecordAttribute    *> mapOidsBuild;
    //             vector<materialization_mode>      outputModesBuild;

    //             {
    //                 exprBuild.emplace_back(build_key_expr);

    //                 expressions::Expression * exprR = exprBuild.back();

    //                 outputModesBuild.insert(outputModesBuild.begin(), EAGER);

    //                 expressions::RecordProjection *projBuild =
    //                         dynamic_cast<expressions::RecordProjection
    //                         *>(exprR);
    //                 if(projBuild == nullptr)
    //                 {
    //                     string error_msg = string(
    //                             "[Join: ] Cannot cast to rec projection.
    //                             Original: ")
    //                             + exprR->getExpressionType()->getType();
    //                     LOG(ERROR)<< error_msg;
    //                     throw runtime_error(string(error_msg));
    //                 }

    //                 //Added in 'wanted fields'
    //                 RecordAttribute *recAttr = new
    //                 RecordAttribute(projBuild->getAttribute());
    //                 fieldsBuild.push_back(new
    //                 RecordAttribute(exprR->getRegisteredAs()));

    //                 string relName = recAttr->getRelationName();
    //                 if (mapOidsBuild.find(relName) == mapOidsBuild.end()) {
    //                     InputInfo *datasetInfo =
    //                     (this->catalogParser).getInputInfo(
    //                             relName);
    //                     RecordAttribute *oid = new RecordAttribute(
    //                             recAttr->getRelationName(), activeLoop,
    //                             datasetInfo->oidType);
    //                     mapOidsBuild[relName] = oid;
    //                     expressions::RecordProjection *oidR =
    //                             new expressions::RecordProjection(exprR,
    //                             *oid);
    //                     // oidR->registerAs(exprR->getRegisteredRelName(),
    //                     exprR->getRegisteredAttrName());
    //                     //Added in 'wanted expressions'
    //                     exprBuild.insert(exprBuild.begin(),oidR);
    //                     cout << "Injecting build OID for " << relName <<
    //                     endl;
    //                     outputModesBuild.insert(outputModesBuild.begin(),
    //                     EAGER);
    //                 }
    //             }

    //             const rapidjson::Value& build_exprsJSON = val["build_e"];
    //             for (SizeType i = 0; i < build_exprsJSON.Size(); i++){
    //                 assert(build_exprsJSON[i].HasMember("e"     ));
    //                 assert(build_exprsJSON[i].HasMember("packet"));
    //                 assert(build_exprsJSON[i]["packet"].IsInt());
    //                 assert(build_exprsJSON[i].HasMember("offset"));
    //                 assert(build_exprsJSON[i]["offset"].IsInt());
    //                 exprBuild.emplace_back(parseExpression(build_exprsJSON[i]["e"]));

    //                 expressions::Expression * exprR = exprBuild.back();

    //                 outputModesBuild.insert(outputModesBuild.begin(), EAGER);

    //                 expressions::RecordProjection *projBuild =
    //                         dynamic_cast<expressions::RecordProjection
    //                         *>(exprR);
    //                 if(projBuild == nullptr)
    //                 {
    //                     string error_msg = string(
    //                             "[Join: ] Cannot cast to rec projection.
    //                             Original: ")
    //                             + exprR->getExpressionType()->getType();
    //                     LOG(ERROR)<< error_msg;
    //                     throw runtime_error(string(error_msg));
    //                 }

    //                 //Added in 'wanted fields'
    //                 RecordAttribute *recAttr = new
    //                 RecordAttribute(projBuild->getAttribute());
    //                 fieldsBuild.push_back(new
    //                 RecordAttribute(exprR->getRegisteredAs()));

    //                 string relName = recAttr->getRelationName();
    //                 if (mapOidsBuild.find(relName) == mapOidsBuild.end()) {
    //                     InputInfo *datasetInfo =
    //                     (this->catalogParser).getInputInfo(
    //                             relName);
    //                     RecordAttribute *oid = new RecordAttribute(
    //                             recAttr->getRelationName(), activeLoop,
    //                             datasetInfo->oidType);
    //                     mapOidsBuild[relName] = oid;
    //                     expressions::RecordProjection *oidR =
    //                             new expressions::RecordProjection(exprR,
    //                             *oid);
    //                     // oidR->registerAs(exprR->getRegisteredRelName(),
    //                     exprR->getRegisteredAttrName());
    //                     //Added in 'wanted expressions'
    //                     exprBuild.insert(exprBuild.begin(),oidR);
    //                     cout << "Injecting build OID for " << relName <<
    //                     endl;
    //                     outputModesBuild.insert(outputModesBuild.begin(),
    //                     EAGER);
    //                 }
    //             }
    //             vector<RecordAttribute*> oidsBuild;
    //             MapToVec(mapOidsBuild, oidsBuild);
    //             Materializer* matBuild = new Materializer(fieldsBuild,
    //             exprBuild,
    //                     oidsBuild, outputModesBuild);

    //             /*
    //              * parse output expressions
    //              * XXX Careful: Assuming numerous output expressions!
    //              */
    //             assert(val.HasMember("probe_e"));
    //             assert(val["probe_e"].IsArray());
    //             vector<expressions::Expression *> exprProbe   ;
    //             vector<RecordAttribute         *> fieldsProbe ;
    //             map<string, RecordAttribute    *> mapOidsProbe;
    //             vector<materialization_mode>      outputModesProbe;

    //             {
    //                 exprProbe.emplace_back(probe_key_expr);

    //                 expressions::Expression * exprR = exprProbe.back();

    //                 outputModesProbe.insert(outputModesProbe.begin(), EAGER);

    //                 expressions::RecordProjection *projProbe =
    //                         dynamic_cast<expressions::RecordProjection
    //                         *>(exprR);
    //                 if(projProbe == nullptr)
    //                 {
    //                     string error_msg = string(
    //                             "[Join: ] Cannot cast to rec projection.
    //                             Original: ")
    //                             + exprR->getExpressionType()->getType();
    //                     LOG(ERROR)<< error_msg;
    //                     throw runtime_error(string(error_msg));
    //                 }

    //                 //Added in 'wanted fields'
    //                 RecordAttribute *recAttr = new
    //                 RecordAttribute(projProbe->getAttribute());
    //                 fieldsProbe.push_back(new
    //                 RecordAttribute(exprR->getRegisteredAs()));

    //                 string relName = recAttr->getRelationName();
    //                 std::cout << "relName" << " " << relName << std::endl;
    //                 if (mapOidsProbe.find(relName) == mapOidsProbe.end()) {
    //                     InputInfo *datasetInfo =
    //                     (this->catalogParser).getInputInfo(
    //                             relName);
    //                     RecordAttribute *oid = new RecordAttribute(
    //                             recAttr->getRelationName(), activeLoop,
    //                             datasetInfo->oidType);
    //                     mapOidsProbe[relName] = oid;
    //                     expressions::RecordProjection *oidR =
    //                             new expressions::RecordProjection(exprR,
    //                             *oid);
    //                     // oidR->registerAs(exprR->getRegisteredRelName(),
    //                     exprR->getRegisteredAttrName());
    //                     //Added in 'wanted expressions'
    //                     exprProbe.insert(exprProbe.begin(),oidR);
    //                     cout << "Injecting probe OID for " << relName <<
    //                     endl;
    //                     outputModesProbe.insert(outputModesProbe.begin(),
    //                     EAGER);
    //                 }
    //             }

    //             const rapidjson::Value& probe_exprsJSON = val["probe_e"];
    //             for (SizeType i = 0; i < probe_exprsJSON.Size(); i++){
    //                 assert(probe_exprsJSON[i].HasMember("e"     ));
    //                 assert(probe_exprsJSON[i].HasMember("packet"));
    //                 assert(probe_exprsJSON[i]["packet"].IsInt());
    //                 assert(probe_exprsJSON[i].HasMember("offset"));
    //                 assert(probe_exprsJSON[i]["offset"].IsInt());
    //                 exprProbe.emplace_back(parseExpression(probe_exprsJSON[i]["e"]));

    //                 expressions::Expression * exprR = exprProbe.back();

    //                 outputModesProbe.insert(outputModesProbe.begin(), EAGER);

    //                 expressions::RecordProjection *projProbe =
    //                         dynamic_cast<expressions::RecordProjection
    //                         *>(exprR);
    //                 if(projProbe == nullptr)
    //                 {
    //                     string error_msg = string(
    //                             "[Join: ] Cannot cast to rec projection.
    //                             Original: ")
    //                             + exprR->getExpressionType()->getType();
    //                     LOG(ERROR)<< error_msg;
    //                     throw runtime_error(string(error_msg));
    //                 }

    //                 //Added in 'wanted fields'
    //                 RecordAttribute *recAttr = new
    //                 RecordAttribute(projProbe->getAttribute());
    //                 fieldsProbe.push_back(new
    //                 RecordAttribute(exprR->getRegisteredAs()));

    //                 string relName = recAttr->getRelationName();
    //                 std::cout << "relName" << " " << relName << std::endl;
    //                 if (mapOidsProbe.find(relName) == mapOidsProbe.end()) {
    //                     InputInfo *datasetInfo =
    //                     (this->catalogParser).getInputInfo(
    //                             relName);
    //                     RecordAttribute *oid = new RecordAttribute(
    //                             recAttr->getRelationName(), activeLoop,
    //                             datasetInfo->oidType);
    //                     mapOidsProbe[relName] = oid;
    //                     expressions::RecordProjection *oidR =
    //                             new expressions::RecordProjection(exprR,
    //                             *oid);
    //                     // oidR->registerAs(exprR->getRegisteredRelName(),
    //                     exprR->getRegisteredAttrName());
    //                     //Added in 'wanted expressions'
    //                     exprProbe.insert(exprProbe.begin(),oidR);
    //                     cout << "Injecting probe OID for " << relName <<
    //                     endl;
    //                     outputModesProbe.insert(outputModesProbe.begin(),
    //                     EAGER);
    //                 }
    //             }
    //             vector<RecordAttribute*> oidsProbe;
    //             MapToVec(mapOidsProbe, oidsProbe);
    //             Materializer* matProbe = new Materializer(fieldsProbe,
    //             exprProbe,
    //                     oidsProbe, outputModesProbe);

    //             newOp = new RadixJoin(predExpr, build_op, probe_op,
    //             this->ctx, "radixHashJoin", *matBuild, *matProbe);
    // #ifndef NCUDA
    //         }
    // #endif
    build_op->setParent(newOp);
    probe_op->setParent(newOp);
  } else if (strcmp(opName, "join") == 0) {
    assert(!(val.HasMember("gpu") && val["gpu"].GetBool()));
    const char *keyMatLeft = "leftFields";
    const char *keyMatRight = "rightFields";
    const char *keyPred = "p";

    /* parse operator input */
    Operator *leftOp = parseOperator(val["leftInput"]);
    Operator *rightOp = parseOperator(val["rightInput"]);

    // Predicate
    assert(val.HasMember(keyPred));
    assert(val[keyPred].IsObject());

    auto predExpr = parseExpression(val[keyPred]);
    const expressions::BinaryExpression *pred =
        dynamic_cast<const expressions::BinaryExpression *>(
            predExpr.getUnderlyingExpression());
    if (pred == nullptr) {
      string error_msg =
          string("[JOIN: ] Cannot cast to binary predicate. Original: ") +
          predExpr.getExpressionType()->getType();
      LOG(ERROR) << error_msg;
      throw runtime_error(string(error_msg));
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

    // LEFT SIDE
    assert(val.HasMember(keyMatLeft));
    assert(val[keyMatLeft].IsArray());
    vector<expression_t> exprsLeft;
    map<string, RecordAttribute *> mapOidsLeft =
        map<string, RecordAttribute *>();
    vector<RecordAttribute *> fieldsLeft = vector<RecordAttribute *>();
    vector<materialization_mode> outputModesLeft;
    for (const auto &keyMat : val[keyMatLeft].GetArray()) {
      auto exprL = parseExpression(keyMat);

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
      auto exprR = parseExpression(keyMat);

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

    newOp = new RadixJoin(*pred, leftOp, rightOp, this->ctx, "radixHashJoin",
                          *matLeft, *matRight);
    leftOp->setParent(newOp);
    rightOp->setParent(newOp);
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
    Operator *childOp = parseOperator(val["input"]);

    /* get monoid(s) */
    assert(val.HasMember(keyAccum));
    assert(val[keyAccum].IsArray());
    vector<Monoid> accs;
    for (const auto &accm : val[keyAccum].GetArray()) {
      assert(accm.IsString());
      Monoid acc = parseAccumulator(accm.GetString());
      accs.push_back(acc);
    }
    /* get label for each of the aggregate values */
    vector<string> aggrLabels;
    assert(val.HasMember(keyAggrNames));
    assert(val[keyAggrNames].IsArray());
    for (const auto &label : val[keyAggrNames].GetArray()) {
      assert(label.IsString());
      aggrLabels.push_back(label.GetString());
    }

    /* Predicate */
    assert(val.HasMember(keyPred));
    assert(val[keyPred].IsObject());
    auto predExpr = parseExpression(val[keyPred]);

    /* Group By */
    assert(val.HasMember(keyGroup));
    assert(val[keyGroup].IsObject());
    auto groupByExpr = parseExpression(val[keyGroup]);

    /* Null-to-zero Checks */
    // FIXME not used in radix nest yet!
    assert(val.HasMember(keyNull));
    assert(val[keyNull].IsObject());
    auto nullsToZerosExpr = parseExpression(val[keyNull]);

    /* Output aggregate expression(s) */
    assert(val.HasMember(keyExprs));
    assert(val[keyExprs].IsArray());
    vector<expression_t> outputExprs;
    for (const auto &v : val[keyExprs].GetArray()) {
      outputExprs.emplace_back(parseExpression(v));
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
    map<string, RecordAttribute *> mapOids = map<string, RecordAttribute *>();
    vector<RecordAttribute *> fieldsToMat = vector<RecordAttribute *>();
    vector<materialization_mode> outputModes;
    for (const auto &v : val[keyMat].GetArray()) {
      auto expr = parseExpression(v);
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
        RecordAttribute *oid = new RecordAttribute(
            recAttr->getRelationName(), activeLoop, datasetInfo->oidType);
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
    vector<RecordAttribute *> oids = vector<RecordAttribute *>();
    MapToVec(mapOids, oids);
    /* FIXME This constructor breaks nest use cases that trigger caching */
    /* Check similar hook in radix-nest.cpp */
    Materializer *matCoarse =
        new Materializer(fieldsToMat, exprsToMat, oids, outputModes);

    // Materializer* matCoarse = new Materializer(exprsToMat);

    // Put operator together
    newOp = new radix::Nest(this->ctx, accs, outputExprs, aggrLabels, predExpr,
                            std::vector<expression_t>{groupByExpr},
                            nullsToZerosExpr, childOp, "radixNest", *matCoarse);
    childOp->setParent(newOp);
  } else if (strcmp(opName, "select") == 0) {
    /* parse operator input */
    assert(val.HasMember("input"));
    assert(val["input"].IsObject());
    Operator *childOp = parseOperator(val["input"]);

    /* parse filtering expression */
    assert(val.HasMember("p"));
    assert(val["p"].IsObject());
    auto p = parseExpression(val["p"]);

    newOp = new Select(p, childOp);
    childOp->setParent(newOp);
  } else if (strcmp(opName, "scan") == 0) {
    assert(val.HasMember(keyPg));
    assert(val[keyPg].IsObject());
    Plugin *pg = this->parsePlugin(val[keyPg]);

    newOp = new Scan(this->ctx, *pg);
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

    void *dict =
        StorageManager::getDictionaryOf(relName + std::string{"."} + attrName);

    InputInfo *datasetInfo =
        (this->catalogParser).getOrCreateInputInfo(dictRelName);
    RecordType *rec = new RecordType{dynamic_cast<const RecordType &>(
        dynamic_cast<CollectionType *>(datasetInfo->exprType)
            ->getNestedType())};
    RecordAttribute *reg_as =
        new RecordAttribute(dictRelName, attrName, new DStringType(dict));
    std::cout << "Registered: " << reg_as->getRelationName() << "."
              << reg_as->getAttrName() << std::endl;
    rec->appendAttribute(reg_as);

    datasetInfo->exprType = new BagType{*rec};

    newOp = new DictScan(
        this->ctx, RecordAttribute{relName, attrName, new DStringType(dict)},
        regex, *reg_as);
#ifndef NCUDA
  } else if (strcmp(opName, "cpu-to-gpu") == 0) {
    /* parse operator input */
    assert(val.HasMember("input"));
    assert(val["input"].IsObject());
    Operator *childOp = parseOperator(val["input"]);

    assert(val.HasMember("projections"));
    assert(val["projections"].IsArray());

    vector<RecordAttribute *> projections;
    for (const auto &proj : val["projections"].GetArray()) {
      assert(proj.IsObject());
      RecordAttribute *recAttr = this->parseRecordAttr(proj);
      projections.push_back(recAttr);
    }

    assert(dynamic_cast<ParallelContext *>(this->ctx));
    newOp = new CpuToGpu(childOp, ((ParallelContext *)this->ctx), projections);
    childOp->setParent(newOp);
#endif
  } else if (strcmp(opName, "block-to-tuples") == 0 ||
             strcmp(opName, "unpack") == 0) {
    /* parse operator input */
    assert(val.HasMember("input"));
    assert(val["input"].IsObject());
    Operator *childOp = parseOperator(val["input"]);

    assert(val.HasMember("projections"));
    assert(val["projections"].IsArray());

    gran_t granularity = gran_t::GRID;
    bool gpu = true;
    if (val.HasMember("gpu")) {
      assert(val["gpu"].IsBool());
      gpu = val["gpu"].GetBool();
      if (!gpu) granularity = gran_t::THREAD;
    }

    if (val.HasMember("granularity")) {
      assert(val["granularity"].IsString());
      std::string gran = val["granularity"].GetString();
      std::transform(gran.begin(), gran.end(), gran.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      if (gran == "grid")
        granularity = gran_t::GRID;
      else if (gran == "block")
        granularity = gran_t::BLOCK;
      else if (gran == "thread")
        granularity = gran_t::THREAD;
      else
        assert(false && "granularity must be one of GRID, BLOCK, THREAD");
    }

    vector<expression_t> projections;
    for (const auto &v : val["projections"].GetArray()) {
      projections.emplace_back(parseExpression(v));
    }

    assert(dynamic_cast<ParallelContext *>(this->ctx));
    newOp = new BlockToTuples(childOp, ((ParallelContext *)this->ctx),
                              projections, gpu, granularity);
    childOp->setParent(newOp);
#ifndef NCUDA
  } else if (strcmp(opName, "gpu-to-cpu") == 0) {
    /* parse operator input */
    assert(val.HasMember("input"));
    assert(val["input"].IsObject());
    Operator *childOp = parseOperator(val["input"]);

    assert(val.HasMember("projections"));
    assert(val["projections"].IsArray());

    vector<RecordAttribute *> projections;
    for (const auto &proj : val["projections"].GetArray()) {
      assert(proj.IsObject());
      RecordAttribute *recAttr = this->parseRecordAttr(proj);
      projections.push_back(recAttr);
    }

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

    assert(dynamic_cast<ParallelContext *>(this->ctx));
    newOp = new GpuToCpu(childOp, ((ParallelContext *)this->ctx), projections,
                         size, g);
    childOp->setParent(newOp);
#endif
  } else if (strcmp(opName, "tuples-to-block") == 0 ||
             strcmp(opName, "pack") == 0) {
    bool gpu = false;
    if (val.HasMember("gpu")) {
      assert(val["gpu"].IsBool());
      gpu = val["gpu"].GetBool();
    }

    /* parse operator input */
    assert(val.HasMember("input"));
    assert(val["input"].IsObject());
    Operator *childOp = parseOperator(val["input"]);

    assert(val.HasMember("projections"));
    assert(val["projections"].IsArray());

    int numOfBuckets = 1;
    expression_t hashExpr = expressions::IntConstant(0);

    vector<expression_t> projections;
    for (const auto &v : val["projections"].GetArray()) {
      projections.emplace_back(parseExpression(v));
    }

    assert(dynamic_cast<ParallelContext *>(this->ctx));
#ifndef NCUDA
    if (gpu) {
      newOp = new GpuHashRearrange(childOp, ((ParallelContext *)this->ctx),
                                   numOfBuckets, projections, hashExpr);
    } else {
#endif
      newOp = new HashRearrange(childOp, ((ParallelContext *)this->ctx),
                                numOfBuckets, projections, hashExpr);
#ifndef NCUDA
    }
#endif
    childOp->setParent(newOp);
  } else if (strcmp(opName, "hash-rearrange") == 0 ||
             strcmp(opName, "hash-pack") == 0) {
    bool gpu = false;
    if (val.HasMember("gpu")) {
      assert(val["gpu"].IsBool());
      gpu = val["gpu"].GetBool();
    }
    /* parse operator input */
    assert(val.HasMember("input"));
    assert(val["input"].IsObject());
    Operator *childOp = parseOperator(val["input"]);

    assert(val.HasMember("projections"));
    assert(val["projections"].IsArray());

    int numOfBuckets = 2;
    if (val.HasMember("buckets")) {
      assert(val["buckets"].IsInt());
      numOfBuckets = val["buckets"].GetInt();
    }

    RecordAttribute *hashAttr = nullptr;
    // register hash as an attribute
    if (val.HasMember("hashProject")) {
      assert(val["hashProject"].IsObject());

      hashAttr = parseRecordAttr(val["hashProject"]);

      InputInfo *datasetInfo =
          (this->catalogParser).getInputInfo(hashAttr->getRelationName());
      RecordType *rec = new RecordType{dynamic_cast<const RecordType &>(
          dynamic_cast<CollectionType *>(datasetInfo->exprType)
              ->getNestedType())};

      rec->appendAttribute(hashAttr);

      datasetInfo->exprType = new BagType{*rec};
    }

    assert(val.HasMember("e"));
    assert(val["e"].IsObject());

    auto hashExpr = parseExpression(val["e"]);

    vector<expression_t> projections;
    for (const auto &v : val["projections"].GetArray()) {
      projections.emplace_back(parseExpression(v));
    }

    assert(dynamic_cast<ParallelContext *>(this->ctx));

#ifndef NCUDA
    if (gpu) {
      newOp =
          new GpuHashRearrange(childOp, ((ParallelContext *)this->ctx),
                               numOfBuckets, projections, hashExpr, hashAttr);
    } else {
#endif
      newOp = new HashRearrange(childOp, ((ParallelContext *)this->ctx),
                                numOfBuckets, projections, hashExpr, hashAttr);
#ifndef NCUDA
    }
#endif
    childOp->setParent(newOp);
  } else if (strcmp(opName, "mem-move-device") == 0) {
    /* parse operator input */
    assert(val.HasMember("input"));
    assert(val["input"].IsObject());
    Operator *childOp = parseOperator(val["input"]);

    assert(val.HasMember("projections"));
    assert(val["projections"].IsArray());

    vector<RecordAttribute *> projections;
    for (const auto &proj : val["projections"].GetArray()) {
      assert(proj.IsObject());
      RecordAttribute *recAttr = this->parseRecordAttr(proj);
      projections.push_back(recAttr);
    }

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

    assert(dynamic_cast<ParallelContext *>(this->ctx));
    newOp = new MemMoveDevice(childOp, ((ParallelContext *)this->ctx),
                              projections, slack, to_cpu);
    childOp->setParent(newOp);
  } else if (strcmp(opName, "mem-broadcast-device") == 0) {
    /* parse operator input */
    assert(val.HasMember("input"));
    assert(val["input"].IsObject());
    Operator *childOp = parseOperator(val["input"]);

    assert(val.HasMember("projections"));
    assert(val["projections"].IsArray());

    vector<RecordAttribute *> projections;
    for (const auto &proj : val["projections"].GetArray()) {
      assert(proj.IsObject());
      RecordAttribute *recAttr = this->parseRecordAttr(proj);
      projections.push_back(recAttr);
    }

    bool to_cpu = false;
    if (val.HasMember("to_cpu")) {
      assert(val["to_cpu"].IsBool());
      to_cpu = val["to_cpu"].GetBool();
    }

    int num_of_targets = 1;
    if (val.HasMember("num_of_targets")) {
      assert(val["num_of_targets"].IsInt());
      num_of_targets = val["num_of_targets"].GetInt();
    }

    bool always_share = false;
    if (val.HasMember("always_share")) {
      assert(val["always_share"].IsBool());
      always_share = val["always_share"].GetBool();
    }

    std::string relName = projections[0]->getRelationName();

    InputInfo *datasetInfo =
        (this->catalogParser).getOrCreateInputInfo(relName);
    RecordType *rec = new RecordType{dynamic_cast<const RecordType &>(
        dynamic_cast<CollectionType *>(datasetInfo->exprType)
            ->getNestedType())};
    RecordAttribute *reg_as =
        new RecordAttribute(relName, "__broadcastTarget", new IntType());
    std::cout << "Registered: " << reg_as->getRelationName() << "."
              << reg_as->getAttrName() << std::endl;
    rec->appendAttribute(reg_as);

    datasetInfo->exprType = new BagType{*rec};

    assert(dynamic_cast<ParallelContext *>(this->ctx));
    newOp = new MemBroadcastDevice(childOp, ((ParallelContext *)this->ctx),
                                   projections, num_of_targets, to_cpu,
                                   always_share);
    childOp->setParent(newOp);
  } else if (strcmp(opName, "mem-move-local-to") == 0) {
    /* parse operator input */
    assert(val.HasMember("input"));
    assert(val["input"].IsObject());
    Operator *childOp = parseOperator(val["input"]);

    assert(val.HasMember("projections"));
    assert(val["projections"].IsArray());

    vector<RecordAttribute *> projections;
    for (const auto &proj : val["projections"].GetArray()) {
      assert(proj.IsObject());
      RecordAttribute *recAttr = this->parseRecordAttr(proj);
      projections.push_back(recAttr);
    }

    int slack = 8;
    if (val.HasMember("slack")) {
      assert(val["slack"].IsInt());
      slack = val["slack"].GetInt();
    }

    assert(dynamic_cast<ParallelContext *>(this->ctx));
    newOp = new MemMoveLocalTo(childOp, ((ParallelContext *)this->ctx),
                               projections, slack);
    childOp->setParent(newOp);
  } else if (strcmp(opName, "exchange") == 0 || strcmp(opName, "router") == 0) {
    /* parse operator input */
    assert(val.HasMember("input"));
    assert(val["input"].IsObject());
    Operator *childOp = parseOperator(val["input"]);

    assert(val.HasMember("projections"));
    assert(val["projections"].IsArray());

    vector<RecordAttribute *> projections;
    for (const auto &proj : val["projections"].GetArray()) {
      assert(proj.IsObject());
      RecordAttribute *recAttr = this->parseRecordAttr(proj);
      projections.push_back(recAttr);
    }

    assert(val.HasMember("numOfParents"));
    assert(val["numOfParents"].IsInt());
    size_t numOfParents = val["numOfParents"].GetInt();

    int slack = 8;
    if (val.HasMember("slack")) {
      assert(val["slack"].IsInt());
      slack = val["slack"].GetInt();
    }

#ifndef NDEBUG
    if (val.HasMember("producers")) {
      assert(val["producers"].IsInt());
      assert(childOp->getDOP() ==
             DegreeOfParallelism{(size_t)val["producers"].GetInt()});
    }
#endif

    bool numa_local = true;
    bool rand_local_cpu = false;
    std::optional<expression_t> hash;
    if (val.HasMember("target")) {
      assert(val["target"].IsObject());
      hash = parseExpression(val["target"]);
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

    auto targets = DeviceType::GPU;
    if (val.HasMember("cpu_targets")) {
      assert(val["cpu_targets"].IsBool());
      targets =
          val["cpu_targets"].GetBool() ? DeviceType::CPU : DeviceType::GPU;
    }

    RoutingPolicy policy_type =
        (numa_local || rand_local_cpu)
            ? RoutingPolicy::LOCAL
            : ((hash.has_value()) ? RoutingPolicy::HASH_BASED
                                  : RoutingPolicy::RANDOM);

    assert(!val.HasMember("numa_socket_id"));

    assert(dynamic_cast<ParallelContext *>(this->ctx));
    newOp = new Router(childOp, ((ParallelContext *)this->ctx),
                       DegreeOfParallelism{numOfParents}, projections, slack,
                       hash, policy_type, targets);
    childOp->setParent(newOp);
  } else if (strcmp(opName, "union-all") == 0) {
    /* parse operator input */
    assert(val.HasMember("input"));
    assert(val["input"].IsArray());
    std::vector<Operator *> children;
    for (const auto &child : val["input"].GetArray()) {
      assert(child.IsObject());
      children.push_back(parseOperator(child));
    }

    assert(val.HasMember("projections"));
    assert(val["projections"].IsArray());

    vector<RecordAttribute *> projections;
    for (const auto &proj : val["projections"].GetArray()) {
      assert(proj.IsObject());
      RecordAttribute *recAttr = this->parseRecordAttr(proj);
      projections.push_back(recAttr);
    }

    assert(dynamic_cast<ParallelContext *>(this->ctx));
    newOp = new UnionAll(children, ((ParallelContext *)this->ctx), projections);
    for (const auto &childOp : children) childOp->setParent(newOp);
  } else if (strcmp(opName, "split") == 0) {
    assert(val.HasMember("split_id"));
    assert(val["split_id"].IsInt());
    size_t split_id = val["split_id"].GetInt();

    if (splitOps.count(split_id) == 0) {
      /* parse operator input */
      assert(val.HasMember("input"));
      assert(val["input"].IsObject());
      Operator *childOp = parseOperator(val["input"]);

      assert(val.HasMember("numOfParents"));
      assert(val["numOfParents"].IsInt());
      int numOfParents = val["numOfParents"].GetInt();

      assert(val.HasMember("projections"));
      assert(val["projections"].IsArray());

      vector<RecordAttribute *> projections;
      for (const auto &proj : val["projections"].GetArray()) {
        assert(proj.IsObject());
        RecordAttribute *recAttr = this->parseRecordAttr(proj);
        projections.push_back(recAttr);
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
        hash = parseExpression(val["target"]);
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
      newOp = new Split(childOp, ((ParallelContext *)this->ctx), numOfParents,
                        projections, slack, hash, policy_type);
      splitOps[split_id] = newOp;
      childOp->setParent(newOp);
    } else {
      newOp = splitOps[split_id];
    }
  } else {
    string err = string("Unknown Operator: ") + opName;
    LOG(ERROR) << err;
    throw runtime_error(err);
  }

  return newOp;
}

inline bool ends_with(std::string const &value, std::string const &ending) {
  if (ending.size() > value.size()) return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

int lookupInDictionary(string s, const rapidjson::Value &val) {
  assert(val.IsObject());
  assert(val.HasMember("path"));
  assert(val["path"].IsString());

  // Input Path
  const char *nameJSON = val["path"].GetString();
  if (ends_with(nameJSON, ".dict")) {
    ifstream is(nameJSON);
    string str;
    string prefix = s + ":";
    while (getline(is, str)) {
      if (strncmp(str.c_str(), prefix.c_str(), prefix.size()) == 0) {
        string encoding{str.c_str() + prefix.size()};
        try {
          size_t pos;
          int enc = stoi(encoding, &pos);
          if (pos + prefix.size() == str.size()) return enc;
          const char *err = "encoded value has extra characters";
          LOG(ERROR) << err;
          throw runtime_error(err);
        } catch (const std::invalid_argument &) {
          const char *err = "invalid dict encoding";
          LOG(ERROR) << err;
          throw runtime_error(err);
        } catch (const std::out_of_range &) {
          const char *err = "out of range dict encoding";
          LOG(ERROR) << err;
          throw runtime_error(err);
        }
      }
    }
    return -1;  // FIXME: this is wrong, we need a binary search, otherwise it
                // breaks ordering
  } else {
    // Prepare Input
    struct stat statbuf;
    stat(nameJSON, &statbuf);
    size_t fsize = statbuf.st_size;

    int fd = open(nameJSON, O_RDONLY);
    if (fd == -1) {
      throw runtime_error(string("json.dict.open"));
    }

    const char *bufJSON =
        (const char *)mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (bufJSON == MAP_FAILED) {
      const char *err = "json.dict.mmap";
      LOG(ERROR) << err;
      throw runtime_error(err);
    }

    rapidjson::Document document;  // Default template parameter uses UTF8 and
                                   // MemoryPoolAllocator.
    if (document.Parse(bufJSON).HasParseError()) {
      const char *err = (string("[CatalogParser: ] Error parsing dictionary ") +
                         string(val["path"].GetString()))
                            .c_str();
      LOG(ERROR) << err;
      throw runtime_error(err);
    }

    assert(document.IsObject());

    if (!document.HasMember(s.c_str()))
      return -1;  // FIXME: this is wrong, we need a binary search, otherwise it
                  // breaks ordering

    assert(document[s.c_str()].IsInt());
    return document[s.c_str()].GetInt();
  }
}

expressions::extract_unit ExpressionParser::parseUnitRange(std::string range,
                                                           Context *ctx) {
  if (range == "YEAR") return expressions::extract_unit::YEAR;
  if (range == "MONTH") return expressions::extract_unit::MONTH;
  if (range == "DAY") return expressions::extract_unit::DAYOFMONTH;
  if (range == "HOUR") return expressions::extract_unit::HOUR;
  if (range == "MINUTE") return expressions::extract_unit::MINUTE;
  if (range == "SECOND") return expressions::extract_unit::SECOND;
  if (range == "QUARTER") return expressions::extract_unit::QUARTER;
  if (range == "WEEK") return expressions::extract_unit::WEEK;
  if (range == "MILLISECOND") return expressions::extract_unit::MILLISECOND;
  if (range == "DOW") return expressions::extract_unit::DAYOFWEEK;
  if (range == "DOY") return expressions::extract_unit::DAYOFYEAR;
  if (range == "DECADE") return expressions::extract_unit::DECADE;
  if (range == "CENTURY") return expressions::extract_unit::CENTURY;
  if (range == "MILLENNIUM") return expressions::extract_unit::MILLENNIUM;
  // case "YEAR_TO_MONTH"     :
  // case "DAY_TO_HOUR"         :
  // case "DAY_TO_MINUTE"     :
  // case "DAY_TO_SECOND"     :
  // case "HOUR_TO_MINUTE"     :
  // case "HOUR_TO_SECOND"     :
  // case "MINUTE_TO_SECOND" :
  // case "EPOCH"             :
  // default:{
  string err = string("Unsupoport TimeUnitRange: ") + range;
  LOG(ERROR) << err;
  throw runtime_error(err);
  // }
}

expression_t ExpressionParser::parseExpression(const rapidjson::Value &val,
                                               Context *ctx) {
  assert(val.IsObject());
  expression_t ret = parseExpressionWithoutRegistering(val, ctx);
  if (val.HasMember("register_as")) {
    assert(val["register_as"].IsObject());
    RecordAttribute *reg_as =
        parseRecordAttr(val["register_as"], ret.getExpressionType());
    assert(reg_as && "Error registering expression as attribute");

    InputInfo *datasetInfo =
        (this->catalogParser).getOrCreateInputInfo(reg_as->getRelationName());
    RecordType *rec = new RecordType{dynamic_cast<const RecordType &>(
        dynamic_cast<CollectionType *>(datasetInfo->exprType)
            ->getNestedType())};
    std::cout << "Registered: " << reg_as->getRelationName() << "."
              << reg_as->getAttrName() << std::endl;
    rec->appendAttribute(reg_as);

    datasetInfo->exprType = new BagType{*rec};

    ret.registerAs(reg_as);
  }
  return ret;
}

/*
 *    enum ExpressionId    { CONSTANT, ARGUMENT, RECORD_PROJECTION,
 * RECORD_CONSTRUCTION, IF_THEN_ELSE, BINARY, MERGE };
 *    FIXME / TODO No Merge yet!! Will be needed for parallelism!
 *    TODO Add NotExpression ?
 */

expression_t ExpressionParser::parseExpressionWithoutRegistering(
    const rapidjson::Value &val, Context *ctx) {
  assert(val.IsObject());

  const char *keyExpression = "expression";
  const char *keyArgNo = "argNo";
  const char *keyExprType = "type";

  /* Input Argument specifics */
  const char *keyAtts = "attributes";

  /* Record Projection specifics */
  const char *keyInnerExpr = "e";
  const char *keyProjectedAttr = "attribute";

  /* Record Construction specifics */
  const char *keyAttsConstruction = "attributes";
  const char *keyAttrName = "name";
  const char *keyAttrExpr = "e";

  /* If-else specifics */
  const char *keyCond = "cond";
  const char *keyThen = "then";
  const char *keyElse = "else";

  /*Binary operator(s) specifics */
  const char *leftArg = "left";
  const char *rightArg = "right";

  assert(val.HasMember(keyExpression));
  assert(val[keyExpression].IsString());
  const char *valExpression = val[keyExpression].GetString();

  expressions::Expression *retValue = nullptr;

  assert(!val.HasMember("isNull") || val["isNull"].IsBool());
  bool isNull = val.HasMember("isNull") && val["isNull"].GetBool();

  const auto &createNull = [&](ExpressionType *b) {
    ProteusValue rv{
        llvm::UndefValue::get(b->getLLVMType(ctx->getLLVMContext())),
        ctx->createTrue()};

    return new expressions::ProteusValueExpression(b, rv);
  };

  if (strcmp(valExpression, "bool") == 0) {
    if (isNull) {
      retValue = createNull(new BoolType());
    } else {
      assert(val.HasMember("v"));
      assert(val["v"].IsBool());
      retValue = new expressions::BoolConstant(val["v"].GetBool());
    }
  } else if (strcmp(valExpression, "int") == 0) {
    if (isNull) {
      retValue = createNull(new IntType());
    } else {
      assert(val.HasMember("v"));
      assert(val["v"].IsInt());
      retValue = new expressions::IntConstant(val["v"].GetInt());
    }
  } else if (strcmp(valExpression, "int64") == 0) {
    if (isNull) {
      retValue = createNull(new Int64Type());
    } else {
      assert(val.HasMember("v"));
      assert(val["v"].IsInt64());
      retValue = new expressions::Int64Constant(val["v"].GetInt64());
    }
  } else if (strcmp(valExpression, "float") == 0) {
    if (isNull) {
      retValue = createNull(new FloatType());
    } else {
      assert(val.HasMember("v"));
      assert(val["v"].IsDouble());
      retValue = new expressions::FloatConstant(val["v"].GetDouble());
    }
  } else if (strcmp(valExpression, "date") == 0) {
    if (isNull) {
      retValue = createNull(new DateType());
    } else {
      assert(val.HasMember("v"));
      assert(val["v"].IsInt64());
      retValue = new expressions::DateConstant(val["v"].GetInt64());
    }
  } else if (strcmp(valExpression, "datetime") == 0) {
    if (isNull) {
      retValue = createNull(new DateType());
    } else {
      assert(val.HasMember("v"));
      assert(val["v"].IsInt64());
      retValue = new expressions::DateConstant(val["v"].GetInt64());
    }
  } else if (strcmp(valExpression, "string") == 0) {
    if (isNull) {
      retValue = createNull(new StringType());
    } else {
      assert(val.HasMember("v"));
      assert(val["v"].IsString());
      string *stringVal = new string(val["v"].GetString());
      retValue = new expressions::StringConstant(*stringVal);
    }
  } else if (strcmp(valExpression, "dstring") ==
             0) {  // FIMXE: do something better, include the dictionary
    if (isNull) {
      retValue = createNull(new DStringType());
    } else {
      assert(val.HasMember("v"));
      if (val["v"].IsInt()) {
        retValue = new expressions::IntConstant(val["v"].GetInt());
      } else {
        assert(val["v"].IsString());
        assert(val.HasMember("dict"));

        int sVal = lookupInDictionary(val["v"].GetString(), val["dict"]);
        retValue = new expressions::IntConstant(sVal);
      }
    }
  } else if (strcmp(valExpression, "argument") == 0) {
    assert(!isNull);
    /* exprType */
    assert(val.HasMember(keyExprType));
    assert(val[keyExprType].IsObject());
    ExpressionType *exprType = parseExpressionType(val[keyExprType]);

    /* argNo */
    assert(val.HasMember(keyArgNo));
    assert(val[keyArgNo].IsInt());
    int argNo = val[keyArgNo].GetInt();

    /* 'projections' / attributes */
    assert(val.HasMember(keyAtts));
    assert(val[keyAtts].IsArray());

    list<RecordAttribute> atts;
    for (const auto &v : val[keyAtts].GetArray()) {
      atts.emplace_back(*parseRecordAttr(v));
    }

    return expression_t::make<expressions::InputArgument>(exprType, argNo,
                                                          atts);
  } else if (strcmp(valExpression, "recordProjection") == 0) {
    assert(!isNull);

    /* e: expression over which projection is calculated */
    assert(val.HasMember(keyInnerExpr));
    expression_t expr = parseExpression(val[keyInnerExpr], ctx);

    /* projected attribute */
    assert(val.HasMember(keyProjectedAttr));
    assert(val[keyProjectedAttr].IsObject());
    RecordAttribute *recAttr = parseRecordAttr(val[keyProjectedAttr]);

    /* exprType */
    if (val.HasMember(keyExprType)) {
      string err{"deprecated type in recordProjection ignored"};
      LOG(WARNING) << err;
      std::cerr << err << endl;
    }

    return expression_t::make<expressions::RecordProjection>(expr, *recAttr);
  } else if (strcmp(valExpression, "recordConstruction") == 0) {
    assert(!isNull);
    /* exprType */
    // assert(val.HasMember(keyExprType));
    // assert(val[keyExprType].IsObject());
    // ExpressionType *exprType = parseExpressionType(val[keyExprType]);

    /* attribute construction(s) */
    assert(val.HasMember(keyAttsConstruction));
    assert(val[keyAttsConstruction].IsArray());

    list<expressions::AttributeConstruction> newAtts;
    for (const auto &attrConst : val[keyAttsConstruction].GetArray()) {
      assert(attrConst.HasMember(keyAttrName));
      assert(attrConst[keyAttrName].IsString());
      string newAttrName = attrConst[keyAttrName].GetString();

      assert(attrConst.HasMember(keyAttrExpr));
      expression_t newAttrExpr = parseExpression(attrConst[keyAttrExpr], ctx);

      expressions::AttributeConstruction *newAttr =
          new expressions::AttributeConstruction(newAttrName, newAttrExpr);
      newAtts.push_back(*newAttr);
    }
    return expression_t::make<expressions::RecordConstruction>(newAtts);
  } else if (strcmp(valExpression, "extract") == 0) {
    assert(val.HasMember("unitrange"));
    assert(val["unitrange"].IsString());

    assert(val.HasMember(keyInnerExpr));
    expression_t expr = parseExpression(val[keyInnerExpr], ctx);

    auto u = parseUnitRange(val["unitrange"].GetString(), ctx);
    return expression_t::make<expressions::ExtractExpression>(expr, u);
  } else if (strcmp(valExpression, "if") == 0) {
    assert(!isNull);
    /* if cond */
    assert(val.HasMember(keyCond));
    expression_t condExpr = parseExpression(val[keyCond], ctx);

    /* then expression */
    assert(val.HasMember(keyThen));
    expression_t thenExpr = parseExpression(val[keyThen], ctx);

    /* else expression */
    assert(val.HasMember(keyElse));
    expression_t elseExpr = parseExpression(val[keyElse], ctx);

    return expression_t::make<expressions::IfThenElse>(condExpr, thenExpr,
                                                       elseExpr);
  }
  /*
   * BINARY EXPRESSIONS
   */
  else if (strcmp(valExpression, "eq") == 0) {
    assert(!isNull);
    /* left child */
    assert(val.HasMember(leftArg));
    expression_t leftExpr = parseExpression(val[leftArg], ctx);

    /* right child */
    assert(val.HasMember(rightArg));
    expression_t rightExpr = parseExpression(val[rightArg], ctx);

    return eq(leftExpr, rightExpr);
  } else if (strcmp(valExpression, "neq") == 0) {
    assert(!isNull);
    /* left child */
    assert(val.HasMember(leftArg));
    expression_t leftExpr = parseExpression(val[leftArg], ctx);

    /* right child */
    assert(val.HasMember(rightArg));
    expression_t rightExpr = parseExpression(val[rightArg], ctx);

    return ne(leftExpr, rightExpr);
  } else if (strcmp(valExpression, "lt") == 0) {
    assert(!isNull);
    /* left child */
    assert(val.HasMember(leftArg));
    expression_t leftExpr = parseExpression(val[leftArg], ctx);

    /* right child */
    assert(val.HasMember(rightArg));
    expression_t rightExpr = parseExpression(val[rightArg], ctx);

    return lt(leftExpr, rightExpr);
  } else if (strcmp(valExpression, "le") == 0) {
    assert(!isNull);
    /* left child */
    assert(val.HasMember(leftArg));
    expression_t leftExpr = parseExpression(val[leftArg], ctx);

    /* right child */
    assert(val.HasMember(rightArg));
    expression_t rightExpr = parseExpression(val[rightArg], ctx);

    return le(leftExpr, rightExpr);
  } else if (strcmp(valExpression, "gt") == 0) {
    assert(!isNull);
    /* left child */
    assert(val.HasMember(leftArg));
    expression_t leftExpr = parseExpression(val[leftArg], ctx);

    /* right child */
    assert(val.HasMember(rightArg));
    expression_t rightExpr = parseExpression(val[rightArg], ctx);

    return gt(leftExpr, rightExpr);
  } else if (strcmp(valExpression, "ge") == 0) {
    assert(!isNull);
    /* left child */
    assert(val.HasMember(leftArg));
    expression_t leftExpr = parseExpression(val[leftArg], ctx);

    /* right child */
    assert(val.HasMember(rightArg));
    expression_t rightExpr = parseExpression(val[rightArg], ctx);

    return ge(leftExpr, rightExpr);
  } else if (strcmp(valExpression, "and") == 0) {
    assert(!isNull);
    /* left child */
    assert(val.HasMember(leftArg));
    expression_t leftExpr = parseExpression(val[leftArg], ctx);

    /* right child */
    assert(val.HasMember(rightArg));
    expression_t rightExpr = parseExpression(val[rightArg], ctx);

    return leftExpr & rightExpr;
  } else if (strcmp(valExpression, "or") == 0) {
    assert(!isNull);
    /* left child */
    assert(val.HasMember(leftArg));
    expression_t leftExpr = parseExpression(val[leftArg], ctx);

    /* right child */
    assert(val.HasMember(rightArg));
    expression_t rightExpr = parseExpression(val[rightArg], ctx);

    return leftExpr | rightExpr;
  } else if (strcmp(valExpression, "add") == 0) {
    assert(!isNull);
    /* left child */
    assert(val.HasMember(leftArg));
    expression_t leftExpr = parseExpression(val[leftArg], ctx);

    /* right child */
    assert(val.HasMember(rightArg));
    expression_t rightExpr = parseExpression(val[rightArg], ctx);

    // ExpressionType *exprType =
    // const_cast<ExpressionType*>(leftExpr->getExpressionType());
    return leftExpr + rightExpr;
  } else if (strcmp(valExpression, "sub") == 0) {
    assert(!isNull);
    /* left child */
    assert(val.HasMember(leftArg));
    expression_t leftExpr = parseExpression(val[leftArg], ctx);

    /* right child */
    assert(val.HasMember(rightArg));
    expression_t rightExpr = parseExpression(val[rightArg], ctx);

    return leftExpr - rightExpr;
  } else if (strcmp(valExpression, "neg") == 0) {
    assert(!isNull);
    /* right child */
    assert(val.HasMember(keyInnerExpr));
    return -parseExpression(val[keyInnerExpr], ctx);
  } else if (strcmp(valExpression, "is_not_null") == 0) {
    assert(!isNull);
    /* right child */
    assert(val.HasMember(keyInnerExpr));
    expression_t expr = parseExpression(val[keyInnerExpr], ctx);

    return expression_t::make<expressions::TestNullExpression>(expr, false);
  } else if (strcmp(valExpression, "mod") == 0) {
    assert(!isNull);
    /* left child */
    assert(val.HasMember(leftArg));
    expression_t leftExpr = parseExpression(val[leftArg], ctx);

    /* right child */
    assert(val.HasMember(rightArg));
    expression_t rightExpr = parseExpression(val[rightArg], ctx);

    // ExpressionType *exprType =
    // const_cast<ExpressionType*>(leftExpr->getExpressionType());
    return leftExpr % rightExpr;
  } else if (strcmp(valExpression, "is_null") == 0) {
    assert(!isNull);
    /* right child */
    assert(val.HasMember(keyInnerExpr));
    expression_t expr = parseExpression(val[keyInnerExpr], ctx);

    return expression_t::make<expressions::TestNullExpression>(expr, true);
  } else if (strcmp(valExpression, "cast") == 0) {
    assert(!isNull);
    /* right child */
    assert(val.HasMember(keyInnerExpr));
    expression_t expr = parseExpression(val[keyInnerExpr], ctx);

    assert(val.HasMember(keyExprType));
    ExpressionType *t = parseExpressionType(val[keyExprType]);

    return expression_t::make<expressions::CastExpression>(t, expr);
  } else if (strcmp(valExpression, "multiply") == 0) {
    assert(!isNull);
    /* left child */
    assert(val.HasMember(leftArg));
    expression_t leftExpr = parseExpression(val[leftArg], ctx);

    /* right child */
    assert(val.HasMember(rightArg));
    expression_t rightExpr = parseExpression(val[rightArg], ctx);

    return leftExpr * rightExpr;
  } else if (strcmp(valExpression, "div") == 0) {
    assert(!isNull);
    /* left child */
    assert(val.HasMember(leftArg));
    expression_t leftExpr = parseExpression(val[leftArg], ctx);

    /* right child */
    assert(val.HasMember(rightArg));
    expression_t rightExpr = parseExpression(val[rightArg], ctx);

    return leftExpr / rightExpr;
  } else if (strcmp(valExpression, "merge") == 0) {
    assert(!isNull);
    string err = string("(Still) unsupported expression: ") + valExpression;
    LOG(ERROR) << err;
    throw runtime_error(err);
  } else {
    string err = string("Unknown expression: ") + valExpression;
    LOG(ERROR) << err;
    throw runtime_error(err);
  }

  return retValue;
}

/*
 * enum typeID    { BOOL, STRING, FLOAT, INT, RECORD, LIST, BAG, SET, INT64,
 * COMPOSITE };
 * FIXME / TODO: Do I need to cater for 'composite' types?
 * IIRC, they only occur as OIDs / complex caches
 */
ExpressionType *ExpressionParser::parseExpressionType(
    const rapidjson::Value &val) {
  /* upper-level keys */
  const char *keyExprType = "type";
  const char *keyCollectionType = "inner";

  /* Related to record types */
  const char *keyRecordAttrs = "attributes";

  assert(val.HasMember(keyExprType));
  assert(val[keyExprType].IsString());
  const char *valExprType = val[keyExprType].GetString();

  if (strcmp(valExprType, "bool") == 0) {
    return new BoolType();
  } else if (strcmp(valExprType, "int") == 0) {
    return new IntType();
  } else if (strcmp(valExprType, "int64") == 0) {
    return new Int64Type();
  } else if (strcmp(valExprType, "float") == 0) {
    return new FloatType();
  } else if (strcmp(valExprType, "date") == 0) {
    return new DateType();
  } else if (strcmp(valExprType, "datetime") == 0) {
    return new DateType();
  } else if (strcmp(valExprType, "string") == 0) {
    return new StringType();
  } else if (strcmp(valExprType, "dstring") == 0) {
    return new DStringType(nullptr);
  } else if (strcmp(valExprType, "set") == 0) {
    assert(val.HasMember("inner"));
    assert(val["inner"].IsObject());
    ExpressionType *innerType = parseExpressionType(val["inner"]);
    return new SetType(*innerType);
  } else if (strcmp(valExprType, "bag") == 0) {
    assert(val.HasMember("inner"));
    assert(val["inner"].IsObject());
    ExpressionType *innerType = parseExpressionType(val["inner"]);
    return new BagType(*innerType);
  } else if (strcmp(valExprType, "list") == 0) {
    assert(val.HasMember("inner"));
    assert(val["inner"].IsObject());
    ExpressionType *innerType = parseExpressionType(val["inner"]);
    return new ListType(*innerType);
  } else if (strcmp(valExprType, "record") == 0) {
    if (val.HasMember("attributes")) {
      assert(val["attributes"].IsArray());

      list<RecordAttribute *> atts;
      for (const auto &attr : val["attributes"].GetArray()) {
        RecordAttribute *recAttr = parseRecordAttr(attr);
        atts.push_back(recAttr);
      }
      return new RecordType(atts);
    } else if (val.HasMember("relName")) {
      assert(val["relName"].IsString());

      return getRecordType(val["relName"].GetString());
    } else {
      return new RecordType();
    }
  } else if (strcmp(valExprType, "composite") == 0) {
    string err = string("(Still) Unsupported expression type: ") + valExprType;
    LOG(ERROR) << err;
    throw runtime_error(err);
  } else {
    string err = string("Unknown expression type: ") + valExprType;
    LOG(ERROR) << err;
    throw runtime_error(err);
  }
}

RecordType *ExpressionParser::getRecordType(string relName,
                                            bool createIfNeeded) {
  // Lookup in catalog based on name
  InputInfo *datasetInfo =
      (createIfNeeded) ? (this->catalogParser).getOrCreateInputInfo(relName)
                       : (this->catalogParser).getInputInfoIfKnown(relName);
  if (datasetInfo == nullptr) return nullptr;

  /* Retrieve RecordType */
  /* Extract inner type of collection */
  CollectionType *collType =
      dynamic_cast<CollectionType *>(datasetInfo->exprType);
  if (collType == nullptr) {
    string error_msg = string(
                           "[Type Parser: ] Cannot cast to collection type. "
                           "Original intended type: ") +
                       datasetInfo->exprType->getType();
    LOG(ERROR) << error_msg;
    throw runtime_error(string(error_msg));
  }
  /* For the current plugins, the expression type is unambiguously RecordType */
  const ExpressionType &nestedType = collType->getNestedType();
  const RecordType &recType_ = dynamic_cast<const RecordType &>(nestedType);
  return new RecordType(recType_.getArgs());
}

const RecordAttribute *ExpressionParser::getAttribute(string relName,
                                                      string attrName,
                                                      bool createIfNeeded) {
  RecordType *recType = getRecordType(relName, createIfNeeded);
  if (recType == nullptr) return nullptr;

  return recType->getArg(attrName);
}

RecordAttribute *ExpressionParser::parseRecordAttr(
    const rapidjson::Value &val, const ExpressionType *defaultType,
    int defaultAttrNo) {
  assert(val.IsObject());
  const char *keyRecAttrType = "type";
  const char *keyRelName = "relName";
  const char *keyAttrName = "attrName";
  const char *keyAttrNo = "attrNo";

  assert(val.HasMember(keyRelName));
  assert(val[keyRelName].IsString());
  string relName = val[keyRelName].GetString();

  assert(val.HasMember(keyAttrName));
  assert(val[keyAttrName].IsString());
  string attrName = val[keyAttrName].GetString();

  const RecordAttribute *attr = getAttribute(relName, attrName, false);

  int attrNo;
  if (val.HasMember(keyAttrNo)) {
    assert(val[keyAttrNo].IsInt());
    attrNo = val[keyAttrNo].GetInt();
  } else {
    attrNo = (attr) ? attr->getAttrNo() : defaultAttrNo;
  }

  const ExpressionType *recArgType;
  if (val.HasMember(keyRecAttrType)) {
    assert(val[keyRecAttrType].IsObject());
    recArgType = parseExpressionType(val[keyRecAttrType]);
  } else {
    if (attr) {
      recArgType = attr->getOriginalType();
    } else {
      if (defaultType) {
        recArgType = defaultType;
      } else {
        std::cerr << relName << "." << attrName << std::endl;
        assert(false && "Attribute not found");
      }
    }
  }

  bool is_block = false;
  if (val.HasMember("isBlock")) {
    assert(val["isBlock"].IsBool());
    is_block = val["isBlock"].GetBool();
  }

  return new RecordAttribute(attrNo, relName, attrName, recArgType, is_block);
}

Monoid ExpressionParser::parseAccumulator(const char *acc) {
  if (strcmp(acc, "sum") == 0) {
    return SUM;
  } else if (strcmp(acc, "max") == 0) {
    return MAX;
  } else if (strcmp(acc, "min") == 0) {
    return MIN;
  } else if (strcmp(acc, "multiply") == 0) {
    return MULTIPLY;
  } else if (strcmp(acc, "or") == 0) {
    return OR;
  } else if (strcmp(acc, "and") == 0) {
    return AND;
  } else if (strcmp(acc, "union") == 0) {
    return UNION;
  } else if (strcmp(acc, "bagunion") == 0) {
    return BAGUNION;
  } else if (strcmp(acc, "append") == 0) {
    return APPEND;
  } else {
    string err = string("Unknown Monoid: ") + acc;
    LOG(ERROR) << err;
    throw runtime_error(err);
  }
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
  const char *keyProjectionsCSV = "projections";
  // pm policy
  const char *keyPolicy = "policy";
  // line hint
  const char *keyLineHint = "lines";
  // OPTIONAL: which delimiter to use
  const char *keyDelimiter = "delimiter";
  // OPTIONAL: are string values wrapped in brackets?
  const char *keyBrackets = "brackets";

  /*
   * BinRow
   */
  const char *keyProjectionsBinRow = "projections";

  /*
   * GPU
   */
  const char *keyProjectionsGPU = "projections";

  /*
   * BinCol
   */
  const char *keyProjectionsBinCol = "projections";

  assert(val.HasMember(keyInputName));
  assert(val[keyInputName].IsString());
  string datasetName = val[keyInputName].GetString();

  assert(val.HasMember(keyPgType));
  assert(val[keyPgType].IsString());
  const char *pgType = val[keyPgType].GetString();

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
        RecordAttribute *recAttr = parseRecordAttr(attr, nullptr, attrNo++);

        std::cout << "Plugin Registered: " << recAttr->getRelationName() << "."
                  << recAttr->getAttrName() << std::endl;

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

  if (strcmp(pgType, "csv") == 0) {
    //        cout<<"Original intended type: " <<
    //        datasetInfo.exprType->getType()<<endl; cout<<"File path: " <<
    //        datasetInfo.path<<endl;

    /* Projections come in an array of Record Attributes */
    assert(val.HasMember(keyProjectionsCSV));
    assert(val[keyProjectionsCSV].IsArray());

    vector<RecordAttribute *> projections;
    for (const auto &attr : val[keyProjectionsCSV].GetArray()) {
      projections.push_back(parseRecordAttr(attr));
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
  } else if (strcmp(pgType, "json") == 0) {
    assert(val.HasMember(keyLineHint));
    assert(val[keyLineHint].IsInt());
    int linehint = val[keyLineHint].GetInt();

    newPg = new jsonPipelined::JSONPlugin(this->ctx, *pathDynamicCopy,
                                          datasetInfo->exprType, linehint);
  } else if (strcmp(pgType, "binrow") == 0) {
    assert(val.HasMember(keyProjectionsBinRow));
    assert(val[keyProjectionsBinRow].IsArray());

    vector<RecordAttribute *> projections;
    for (const auto &attr : val[keyProjectionsBinRow].GetArray()) {
      projections.push_back(parseRecordAttr(attr));
    }

    newPg =
        new BinaryRowPlugin(this->ctx, *pathDynamicCopy, *recType, projections);
  } else if (strcmp(pgType, "bincol") == 0) {
    assert(val.HasMember(keyProjectionsBinCol));
    assert(val[keyProjectionsBinCol].IsArray());

    vector<RecordAttribute *> projections;
    for (const auto &attr : val[keyProjectionsBinCol].GetArray()) {
      projections.push_back(parseRecordAttr(attr));
    }

    bool sizeInFile = true;
    if (val.HasMember("sizeInFile")) {
      assert(val["sizeInFile"].IsBool());
      sizeInFile = val["sizeInFile"].GetBool();
    }
    newPg = new BinaryColPlugin(this->ctx, *pathDynamicCopy, *recType,
                                projections, sizeInFile);
  } else if (strcmp(pgType, "block") == 0) {
    assert(val.HasMember(keyProjectionsGPU));
    assert(val[keyProjectionsGPU].IsArray());

    vector<RecordAttribute *> projections;
    for (const auto &attr : val[keyProjectionsGPU].GetArray()) {
      projections.push_back(parseRecordAttr(attr));
    }

    assert(dynamic_cast<ParallelContext *>(this->ctx));

    newPg = new BinaryBlockPlugin(dynamic_cast<ParallelContext *>(this->ctx),
                                  *pathDynamicCopy, *recType, projections);
  } else {
    assert(dynamic_cast<ParallelContext *>(this->ctx));

    typedef Plugin *(*plugin_creator_t)(ParallelContext *, std::string,
                                        RecordType,
                                        std::vector<RecordAttribute *> &);

    std::string conv = "create" + hyphenatedPluginToCamel(pgType) + "Plugin";

    std::cout << "PluginName: " << hyphenatedPluginToCamel(pgType) << std::endl;

    plugin_creator_t create = (plugin_creator_t)dlsym(handle, conv.c_str());

    if (!create) {
      string err = string("Unknown Plugin Type: ") + pgType;
      LOG(ERROR) << err;
      throw runtime_error(err);
    }

    assert(val.HasMember(keyProjectionsGPU));
    assert(val[keyProjectionsGPU].IsArray());

    vector<RecordAttribute *> projections;
    for (const auto &attr : val[keyProjectionsGPU].GetArray()) {
      projections.push_back(parseRecordAttr(attr));
    }

    newPg = create(dynamic_cast<ParallelContext *>(this->ctx), *pathDynamicCopy,
                   *recType, projections /*, const rapidjson::Value &val */);
    // FIXME: a better interface would be to also pass the current json value,
    // so that plugins can read their own attributes.
  }

  activePlugins.push_back(newPg);
  Catalog &catalog = Catalog::getInstance();
  catalog.registerPlugin(*pathDynamicCopy, newPg);
  datasetInfo->oidType = newPg->getOIDType();
  (this->catalogParser).setInputInfo(datasetName, datasetInfo);
  return newPg;
}

#include <dirent.h>
#include <stdlib.h>

void CatalogParser::parseCatalogFile(std::string file) {
  // key aliases
  const char *keyInputPath = "path";
  const char *keyExprType = "type";

  // Prepare Input
  struct stat statbuf;
  stat(file.c_str(), &statbuf);
  size_t fsize = statbuf.st_size;

  int fd = open(file.c_str(), O_RDONLY);
  if (fd < 0) {
    std::string err = "failed to open file: " + file;
    LOG(ERROR) << err;
    throw runtime_error(err);
  }
  const char *bufJSON =
      (const char *)mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
  if (bufJSON == MAP_FAILED) {
    std::string err = "json.mmap";
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

  ExpressionParser exprParser{*this};

  // Start plan traversal.
  assert(document.IsObject());

  for (const auto &member : document.GetObject()) {
    assert(member.value.IsObject());
    assert((member.value)[keyInputPath].IsString());
    string inputPath = ((member.value)[keyInputPath]).GetString();
    assert((member.value)[keyExprType].IsObject());
    ExpressionType *exprType =
        exprParser.parseExpressionType((member.value)[keyExprType]);
    InputInfo *info = new InputInfo();
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

void CatalogParser::parseDir(std::string dir) {
  // FIXME: we can do that in a portable way with C++17, but for now because we
  // are using libstdc++, upgrading to C++17 and using <filesystem> causes
  // linking problems in machines with old gcc version
  DIR *d = opendir(dir.c_str());
  if (!d) {
    LOG(WARNING) << "Open dir " << dir << " failed (" << strerror(errno) << ")";
    LOG(WARNING) << "Ignoring directory: " << dir;
    return;
  }

  dirent *entry;
  while ((entry = readdir(d)) != nullptr) {
    std::string fname{entry->d_name};

    if (strcmp(entry->d_name, "..") == 0) continue;
    if (strcmp(entry->d_name, ".") == 0) continue;

    std::string origd{dir + "/" + fname};
    // Use this to canonicalize paths:
    // std::string pathd{realpath(origd.c_str(), nullptr)};
    std::string pathd{origd};

    struct stat s;
    stat(pathd.c_str(), &s);

    if (S_ISDIR(s.st_mode)) {
      parseDir(pathd);
    } else if (fname == "catalog.json" && S_ISREG(s.st_mode)) {
      parseCatalogFile(pathd);
    } /* else skipping */
  }
  closedir(d);
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
    Plugin *newPg =
        new pm::CSVPlugin(context, inputName, *rec, projs, ',', 10, 1, false);
    catalog.registerPlugin(*(new string(inputName)), newPg);
    ret->oidType = newPg->getOIDType();

    setInputInfo(inputName, ret);
  }

  return ret;
}

std::ostream &operator<<(std::ostream &out, const rapidjson::Value &val) {
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  val.Accept(writer);
  out << buffer.GetString();
  return out;
}
