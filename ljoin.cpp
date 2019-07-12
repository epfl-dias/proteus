/*
    RAW -- High-performance querying over raw, never-seen-before data.

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

#include "codegen/util/parallel-context.hpp"
#include "operators/block-to-tuples.hpp"
#include "operators/cpu-to-gpu.hpp"
#include "operators/flush.hpp"
#include "operators/gpu/gpu-hash-join-chained.hpp"
#include "operators/gpu/gpu-hash-rearrange.hpp"
#include "operators/gpu/gpu-partitioned-hash-join-chained.hpp"
#include "operators/gpu/gpu-reduce.hpp"
#include "operators/gpu/gpu-to-cpu.hpp"
#include "operators/hash-join-chained.hpp"
#include "operators/hash-rearrange-buffered.hpp"
#include "operators/hash-rearrange.hpp"
#include "operators/mem-move-device.hpp"
#include "operators/mem-move-local-to.hpp"
#include "operators/packet-zip.hpp"
#include "operators/reduce-opt.hpp"
#include "operators/split-opt.hpp"
#include "operators/split.hpp"
#include "operators/unionall.hpp"
#include "plan/plan-parser.hpp"
#include "plugins/gpu-col-scan-plugin.hpp"
#include "plugins/gpu-col-scan-to-blocks-plugin.hpp"
#include "plugins/scan-to-blocks-sm-plugin.hpp"
#include "storage/raw-storage-manager.hpp"
#include "util/raw-memory-manager.hpp"
#include "util/raw-pipeline.hpp"

void Query() {
  {
    RawCatalog *catalog = &RawCatalog::getInstance();
    CachingService *caches = &CachingService::getInstance();
    catalog->clear();
    caches->clear();
  }

  ParallelContext *ctx = new ParallelContext("Out_of_GPU", false);
  CatalogParser catalog = CatalogParser("inputs/plans/catalog.json", ctx);

  RawCatalog &rawcatalog = RawCatalog::getInstance();

  string key_str("d_key");

  string relName1("inputs/micro/A_128000000.bin");

  InputInfo *datasetInfo1 = catalog.getInputInfoIfKnown(relName1);
  std::cout << datasetInfo1 << std::endl;
  std::cout << datasetInfo1->path << std::endl;
  string *path1 = new string(datasetInfo1->path);
  CollectionType *collType1 =
      dynamic_cast<CollectionType *>(datasetInfo1->exprType);
  const ExpressionType &nestedType1 = collType1->getNestedType();
  const RecordType &recType_1 = dynamic_cast<const RecordType &>(nestedType1);

  RecordType *recType1 = new RecordType(recType_1.getArgs());

  RecordAttribute *d_key1 =
      new RecordAttribute(1, relName1, key_str, new IntType(),
                          false);  // recType1->getArg(key_str);
  std::cout << d_key1->getType() << std::endl;
  int attrNo = 1;
  const ExpressionType *recArgType = d_key1->getOriginalType();
  RecordAttribute *d_key_block1 =
      new RecordAttribute(attrNo, relName1, key_str, recArgType, true);
  std::cout << d_key_block1->getType() << std::endl;

  vector<RecordAttribute> expr_projections1;
  expr_projections1.push_back(*d_key1);

  vector<RecordAttribute *> expr_projections_ptr1;
  expr_projections_ptr1.push_back(d_key1);

  Plugin *pg1 =
      new ScanToBlockSMPlugin(ctx, *path1, *recType1, expr_projections_ptr1);
  rawcatalog.registerPlugin(*path1, pg1);
  RawOperator *scan1 = new Scan(ctx, *pg1);

  vector<RecordAttribute> expr_projections_block1;
  expr_projections_block1.push_back(*d_key_block1);

  vector<RecordAttribute *> expr_projections_block_ptr1;
  expr_projections_block_ptr1.push_back(d_key_block1);

  RawOperator *mml1 =
      new MemMoveLocalTo(scan1, ctx, expr_projections_block_ptr1, 4);
  scan1->setParent(mml1);

  RawOperator *mmd1 =
      new MemMoveDevice(mml1, ctx, expr_projections_block_ptr1, 4, false);
  mml1->setParent(mmd1);

  RawOperator *ctg1 = new CpuToGpu(mmd1, ctx, expr_projections_block_ptr1);
  mmd1->setParent(ctg1);

  list<RecordAttribute> atts1 = list<RecordAttribute>();
  atts1.push_back(*d_key1);
  expressions::Expression *e1 =
      new expressions::InputArgument(recType1, -1, atts1);
  expressions::Expression *e11 = new expressions::RecordProjection(e1, *d_key1);
  expressions::Expression *e3 =
      new expressions::InputArgument(recType1, -1, atts1);
  expressions::Expression *e31 = new expressions::RecordProjection(e3, *d_key1);

  vector<expressions::Expression *> exprs1;
  exprs1.push_back(e11);

  RawOperator *btt1 = new BlockToTuples(ctg1, ctx, exprs1, true, gran_t::GRID);
  ctg1->setParent(btt1);

  expressions::Expression *lhs1 = e11;
  expressions::Expression *rhs1 = new expressions::IntConstant(129000000);
  expressions::Expression *pred1 =
      new expressions::LtExpression(new BoolType(), lhs1, rhs1);

  RawOperator *select1 = new Select(pred1, btt1);
  btt1->setParent(select1);

  string relName2("inputs/micro/B_128000000.bin");

  InputInfo *datasetInfo2 = catalog.getInputInfoIfKnown(relName2);
  std::cout << datasetInfo2 << std::endl;
  std::cout << datasetInfo2->path << std::endl;
  string *path2 = new string(datasetInfo2->path);
  CollectionType *collType2 =
      dynamic_cast<CollectionType *>(datasetInfo2->exprType);
  const ExpressionType &nestedType2 = collType2->getNestedType();
  const RecordType &recType_2 = dynamic_cast<const RecordType &>(nestedType2);

  RecordType *recType2 = new RecordType(recType_2.getArgs());

  RecordAttribute *d_key2 =
      new RecordAttribute(1, relName2, key_str, new IntType(),
                          false);  // 2recType1->getArg(key_str);
  std::cout << d_key2->getType() << std::endl;
  int attrNo2 = 1;
  const ExpressionType *recArgType2 = d_key2->getOriginalType();
  RecordAttribute *d_key_block2 =
      new RecordAttribute(attrNo2, relName2, key_str, recArgType2, true);
  std::cout << d_key_block2->getType() << std::endl;

  vector<RecordAttribute> expr_projections2;
  expr_projections2.push_back(*d_key2);

  vector<RecordAttribute *> expr_projections_ptr2;
  expr_projections_ptr2.push_back(d_key2);

  Plugin *pg2 =
      new ScanToBlockSMPlugin(ctx, *path2, *recType2, expr_projections_ptr2);
  rawcatalog.registerPlugin(*path2, pg2);
  RawOperator *scan2 = new Scan(ctx, *pg2);

  vector<RecordAttribute> expr_projections_block2;
  expr_projections_block2.push_back(*d_key_block2);

  vector<RecordAttribute *> expr_projections_block_ptr2;
  expr_projections_block_ptr2.push_back(d_key_block2);

  RawOperator *mml2 =
      new MemMoveLocalTo(scan2, ctx, expr_projections_block_ptr2, 4);
  scan2->setParent(mml2);

  RawOperator *mmd2 =
      new MemMoveDevice(mml2, ctx, expr_projections_block_ptr2, 4, false);
  mml2->setParent(mmd2);

  RawOperator *ctg2 = new CpuToGpu(mmd2, ctx, expr_projections_block_ptr2);
  mmd2->setParent(ctg2);

  list<RecordAttribute> atts2 = list<RecordAttribute>();
  atts2.push_back(*d_key2);
  expressions::Expression *e2 =
      new expressions::InputArgument(recType2, -1, atts2);
  expressions::Expression *e21 = new expressions::RecordProjection(e2, *d_key2);
  expressions::Expression *e4 =
      new expressions::InputArgument(recType2, -1, atts2);
  expressions::Expression *e41 = new expressions::RecordProjection(e4, *d_key2);

  vector<expressions::Expression *> exprs2;
  exprs2.push_back(e21);

  RawOperator *btt2 = new BlockToTuples(ctg2, ctx, exprs2, true, gran_t::GRID);
  ctg2->setParent(btt2);

  expressions::Expression *lhs2 = e21;
  expressions::Expression *rhs2 = new expressions::IntConstant(129000000);
  expressions::Expression *pred2 =
      new expressions::LtExpression(new BoolType(), lhs2, rhs2);

  RawOperator *select2 = new Select(pred2, btt2);
  btt2->setParent(select2);

  RecordAttribute *join_key1 =
      new RecordAttribute(-1, "join198", key_str + "0", new IntType(), false);
  InputInfo *datasetInfo11 =
      catalog.getOrCreateInputInfo(join_key1->getRelationName());
  RecordType *rec1 = new RecordType{dynamic_cast<const RecordType &>(
      dynamic_cast<CollectionType *>(datasetInfo11->exprType)
          ->getNestedType())};
  rec1->appendAttribute(join_key1);
  datasetInfo11->exprType = new BagType{*rec1};
  e31->registerAs(join_key1);

  RecordAttribute *join_key2 =
      new RecordAttribute(-1, "join198", key_str, new IntType(), false);
  InputInfo *datasetInfo21 =
      catalog.getOrCreateInputInfo(join_key2->getRelationName());
  RecordType *rec2 = new RecordType{dynamic_cast<const RecordType &>(
      dynamic_cast<CollectionType *>(datasetInfo21->exprType)
          ->getNestedType())};
  rec2->appendAttribute(join_key2);
  datasetInfo21->exprType = new BagType{*rec2};
  e41->registerAs(join_key2);

  vector<GpuMatExpr> build_e;
  vector<size_t> build_widths;
  build_widths.push_back(64);
  vector<GpuMatExpr> probe_e;
  vector<size_t> probe_widths;
  probe_widths.push_back(64);

  RecordAttribute *attr_target =
      new RecordAttribute(-1, "coordinator", "target", new IntType(), false);
  InputInfo *datasetInfoCoord =
      catalog.getOrCreateInputInfo(attr_target->getRelationName());
  RecordType *coord_rec = new RecordType{dynamic_cast<const RecordType &>(
      dynamic_cast<CollectionType *>(datasetInfoCoord->exprType)
          ->getNestedType())};
  coord_rec->appendAttribute(attr_target);
  datasetInfoCoord->exprType = new BagType{*coord_rec};

  HashPartitioner *hpart1 =
      new HashPartitioner(attr_target, build_e, build_widths, e31, select1, ctx,
                          15, "partition_hash_1");
  select1->setParent(hpart1);
  HashPartitioner *hpart2 =
      new HashPartitioner(attr_target, probe_e, probe_widths, e41, select2, ctx,
                          15, "partition_hash_2");
  select2->setParent(hpart2);

  RawOperator *join = new GpuPartitionedHashJoinChained(
      build_e, build_widths, e31, hpart1, probe_e, probe_widths, e41, hpart2,
      hpart1->getState(), hpart2->getState(), 15, ctx);
  hpart1->setParent(join);
  hpart2->setParent(join);

  list<RecordAttribute *> join_atts;
  join_atts.push_back(join_key1);
  join_atts.push_back(join_key2);
  list<RecordAttribute> join_atts_d;
  // join_atts_d.push_back(*join_key1);
  join_atts_d.push_back(*join_key2);

  RecordType *recTypeJoin = new RecordType(join_atts);
  expressions::Expression *e5 =
      new expressions::InputArgument(recTypeJoin, -1, join_atts_d);
  expressions::Expression *e51 =
      new expressions::RecordProjection(e5, *join_key2);

  RecordAttribute *aggr =
      new RecordAttribute(-1, "agg199", "EXPR$0", new IntType(), false);
  InputInfo *datasetInfo31 =
      catalog.getOrCreateInputInfo(aggr->getRelationName());
  RecordType *rec3 = new RecordType{dynamic_cast<const RecordType &>(
      dynamic_cast<CollectionType *>(datasetInfo31->exprType)
          ->getNestedType())};
  rec3->appendAttribute(aggr);
  datasetInfo31->exprType = new BagType{*rec3};
  e51->registerAs(aggr);

  vector<Monoid> accs;
  vector<expressions::Expression *> outputExprs;
  accs.push_back(SUM);
  outputExprs.push_back(e51);

  RawOperator *reduce = new opt::GpuReduce(
      accs, outputExprs, new expressions::BoolConstant(true), join, ctx);
  join->setParent(reduce);

  vector<RecordAttribute *> reduce_attrs;
  reduce_attrs.push_back(aggr);
  RawOperator *gtc =
      new GpuToCpu(reduce, ctx, reduce_attrs, 131072, gran_t::GRID);
  reduce->setParent(gtc);

  list<RecordAttribute *> aggr_atts;
  aggr_atts.push_back(aggr);
  list<RecordAttribute> aggr_atts_d;
  aggr_atts_d.push_back(*aggr);

  RecordType *recTypeAggr = new RecordType(aggr_atts);
  expressions::Expression *e6 =
      new expressions::InputArgument(recTypeAggr, -1, aggr_atts_d);
  expressions::Expression *e61 = new expressions::RecordProjection(e6, *aggr);

  vector<expressions::Expression *> printExprs;
  printExprs.push_back(e61);
  RawOperator *flush = new Flush(printExprs, gtc, ctx);
  gtc->setParent(flush);

  RawOperator *root = flush;
  root->produce();

  std::cout << "produced" << std::endl;

  ctx->prepareFunction(ctx->getGlobalFunction());
  {
    RawCatalog &catalog = RawCatalog::getInstance();
    /* XXX Remove when testing caches (?) */
    catalog.clear();
  }

  ctx->compileAndLoad();

  std::vector<RawPipeline *> pipelines;

  pipelines = ctx->getPipelines();

  for (RawPipeline *p : pipelines) {
    {
      time_block t("T: ");

      p->open();
      p->consume(0);
      p->close();
    }
  }
}

void Query2() {
  {
    RawCatalog *catalog = &RawCatalog::getInstance();
    CachingService *caches = &CachingService::getInstance();
    catalog->clear();
    caches->clear();
  }

  ParallelContext *ctx = new ParallelContext("Out_of_GPU", false);
  CatalogParser catalog = CatalogParser("inputs/plans/catalog.json", ctx);

  RawCatalog &rawcatalog = RawCatalog::getInstance();

  int numOfBuckets = 16;
  int numPartitioners = 48;
  int numConcurrent = 4;

  string key_str("d_key");
  string payload_str1("d_val2");
  string payload_str2("d_val");

  string part_str("partition_hash_");
  string relName1("inputs/micro/A_256000000.bin");

  InputInfo *datasetInfo1 = catalog.getInputInfoIfKnown(relName1);
  std::cout << datasetInfo1 << std::endl;
  std::cout << datasetInfo1->path << std::endl;
  string *path1 = new string(datasetInfo1->path);
  CollectionType *collType1 =
      dynamic_cast<CollectionType *>(datasetInfo1->exprType);
  const ExpressionType &nestedType1 = collType1->getNestedType();
  const RecordType &recType_1 = dynamic_cast<const RecordType &>(nestedType1);

  RecordType *recType1 = new RecordType(recType_1.getArgs());

  RecordAttribute *d_key1 =
      new RecordAttribute(1, relName1, key_str, new IntType(),
                          false);  // recType1->getArg(key_str);
  RecordAttribute *d_payload1 =
      new RecordAttribute(2, relName1, payload_str1, new IntType(), false);
  std::cout << d_key1->getType() << std::endl;
  int attrNo = 1;
  const ExpressionType *recArgType = d_key1->getOriginalType();
  RecordAttribute *d_key_block1 =
      new RecordAttribute(1, relName1, key_str, recArgType, true);
  RecordAttribute *d_payload_block1 =
      new RecordAttribute(2, relName1, payload_str1, recArgType, true);
  std::cout << d_key_block1->getType() << std::endl;

  vector<RecordAttribute> expr_projections1;
  expr_projections1.push_back(*d_key1);
  expr_projections1.push_back(*d_payload1);

  vector<RecordAttribute *> expr_projections_ptr1;
  expr_projections_ptr1.push_back(d_key1);
  expr_projections_ptr1.push_back(d_payload1);

  vector<RecordAttribute *> expr_projections_blockd_ptr1;
  expr_projections_blockd_ptr1.push_back(d_key_block1);
  expr_projections_blockd_ptr1.push_back(d_payload_block1);

  list<RecordAttribute> atts1 = list<RecordAttribute>();
  atts1.push_back(*d_key1);
  atts1.push_back(*d_payload1);

  expressions::Expression *e1 =
      new expressions::InputArgument(recType1, -1, atts1);
  expressions::Expression *e11 = new expressions::RecordProjection(e1, *d_key1);
  expressions::Expression *e12 =
      new expressions::RecordProjection(e1, *d_payload1);
  expressions::Expression *e3 =
      new expressions::InputArgument(recType1, -1, atts1);
  expressions::Expression *e31 = new expressions::RecordProjection(e3, *d_key1);
  expressions::Expression *e32 =
      new expressions::RecordProjection(e3, *d_payload1);

  Plugin *pg1 =
      new ScanToBlockSMPlugin(ctx, *path1, *recType1, expr_projections_ptr1);
  rawcatalog.registerPlugin(*path1, pg1);
  RawOperator *scan1 = new Scan(ctx, *pg1);

  vector<expressions::Expression *> exprs1;
  exprs1.push_back(e11);
  exprs1.push_back(e12);

  Exchange *in_xch11 = new Exchange(
      scan1, ctx, numPartitioners, expr_projections_blockd_ptr1, 4, NULL, true);
  scan1->setParent(in_xch11);

  // RawOperator* btt11 =  new BlockToTuples(scan1, ctx, exprs1, false,
  // gran_t::THREAD); scan1->setParent(btt11);

  RawOperator *btt11 =
      new BlockToTuples(in_xch11, ctx, exprs1, false, gran_t::THREAD);
  in_xch11->setParent(btt11);

  RecordAttribute *hash_key1 = new RecordAttribute(
      -1, relName1, part_str + relName1, new IntType(), false);
  InputInfo *datasetInfoH1 =
      catalog.getOrCreateInputInfo(hash_key1->getRelationName());
  RecordType *hrec1 = new RecordType{dynamic_cast<const RecordType &>(
      dynamic_cast<CollectionType *>(datasetInfoH1->exprType)
          ->getNestedType())};
  hrec1->appendAttribute(hash_key1);
  datasetInfoH1->exprType = new BagType{*hrec1};
  // e31->registerAs(join_key1);

  list<RecordAttribute *> hash_atts;
  hash_atts.push_back(d_key1);
  hash_atts.push_back(d_payload1);
  hash_atts.push_back(hash_key1);

  list<RecordAttribute> hash_atts_d;
  hash_atts_d.push_back(*d_key1);
  hash_atts_d.push_back(*d_payload1);

  RecordType *recTypeHash = new RecordType(hash_atts);
  expressions::Expression *h1 =
      new expressions::InputArgument(recTypeHash, -1, hash_atts_d);
  expressions::Expression *h11 = new expressions::RecordProjection(h1, *d_key1);
  expressions::Expression *h12 =
      new expressions::RecordProjection(h1, *d_payload1);

  vector<expressions::Expression *> hash_projections;
  hash_projections.push_back(h11);
  hash_projections.push_back(h12);

  RawOperator *part1 = new HashRearrange(btt11, ctx, numOfBuckets,
                                         hash_projections, h11, hash_key1);
  btt11->setParent(part1);

  expr_projections_blockd_ptr1.push_back(hash_key1);

  Exchange *in_xch12 = new Exchange(part1, ctx, 1, expr_projections_blockd_ptr1,
                                    4, NULL, true, false, numPartitioners);
  part1->setParent(in_xch12);

  ////////////////////////////////////////////////////////////////////////////////////////////

  string relName2("inputs/micro/B_256000000.bin");

  InputInfo *datasetInfo2 = catalog.getInputInfoIfKnown(relName2);
  std::cout << datasetInfo2 << std::endl;
  std::cout << datasetInfo2->path << std::endl;
  string *path2 = new string(datasetInfo2->path);
  CollectionType *collType2 =
      dynamic_cast<CollectionType *>(datasetInfo2->exprType);
  const ExpressionType &nestedType2 = collType2->getNestedType();
  const RecordType &recType_2 = dynamic_cast<const RecordType &>(nestedType2);

  RecordType *recType2 = new RecordType(recType_2.getArgs());

  RecordAttribute *d_key2 =
      new RecordAttribute(1, relName2, key_str, new IntType(),
                          false);  // recType1->getArg(key_str);
  RecordAttribute *d_payload2 =
      new RecordAttribute(2, relName2, payload_str2, new IntType(), false);
  std::cout << d_key2->getType() << std::endl;
  int attrNo2 = 1;
  const ExpressionType *recArgType2 = d_key2->getOriginalType();
  RecordAttribute *d_key_block2 =
      new RecordAttribute(1, relName2, key_str, recArgType2, true);
  RecordAttribute *d_payload_block2 =
      new RecordAttribute(2, relName2, payload_str2, recArgType2, true);

  std::cout << d_key_block2->getType() << std::endl;

  vector<RecordAttribute> expr_projections2;
  expr_projections2.push_back(*d_key2);
  expr_projections2.push_back(*d_payload2);

  vector<RecordAttribute *> expr_projections_ptr2;
  expr_projections_ptr2.push_back(d_key2);
  expr_projections_ptr2.push_back(d_payload2);

  vector<RecordAttribute *> expr_projections_blockd_ptr2;
  expr_projections_blockd_ptr2.push_back(d_key_block2);
  expr_projections_blockd_ptr2.push_back(d_payload_block2);

  list<RecordAttribute> atts2 = list<RecordAttribute>();
  atts2.push_back(*d_key2);
  atts2.push_back(*d_payload2);

  expressions::Expression *e2 =
      new expressions::InputArgument(recType2, -1, atts2);
  expressions::Expression *e21 = new expressions::RecordProjection(e2, *d_key2);
  expressions::Expression *e22 =
      new expressions::RecordProjection(e2, *d_payload2);
  expressions::Expression *e4 =
      new expressions::InputArgument(recType2, -1, atts2);
  expressions::Expression *e41 = new expressions::RecordProjection(e4, *d_key2);
  expressions::Expression *e42 =
      new expressions::RecordProjection(e4, *d_payload2);

  Plugin *pg2 =
      new ScanToBlockSMPlugin(ctx, *path2, *recType2, expr_projections_ptr2);
  rawcatalog.registerPlugin(*path2, pg2);
  RawOperator *scan2 = new Scan(ctx, *pg2);

  vector<expressions::Expression *> exprs2;
  exprs2.push_back(e21);
  exprs2.push_back(e22);

  // RawOperator* btt21 =  new BlockToTuples(scan2, ctx, exprs2, false,
  // gran_t::THREAD); scan2->setParent(btt21);

  Exchange *in_xch21 = new Exchange(
      scan2, ctx, numPartitioners, expr_projections_blockd_ptr2, 4, NULL, true);
  scan2->setParent(in_xch21);

  RawOperator *btt21 =
      new BlockToTuples(in_xch21, ctx, exprs2, false, gran_t::THREAD);
  in_xch21->setParent(btt21);

  RecordAttribute *hash_key2 = new RecordAttribute(
      -1, relName2, part_str + relName2, new IntType(), false);
  InputInfo *datasetInfoH2 =
      catalog.getOrCreateInputInfo(hash_key2->getRelationName());
  RecordType *hrec2 = new RecordType{dynamic_cast<const RecordType &>(
      dynamic_cast<CollectionType *>(datasetInfoH2->exprType)
          ->getNestedType())};
  hrec2->appendAttribute(hash_key2);
  datasetInfoH2->exprType = new BagType{*hrec2};
  // e31->registerAs(join_key1);

  list<RecordAttribute *> hash_atts2;
  hash_atts2.push_back(d_key2);
  hash_atts2.push_back(d_payload2);
  hash_atts2.push_back(hash_key2);
  list<RecordAttribute> hash_atts_d2;
  // join_atts_d.push_back(*join_key1);
  hash_atts_d2.push_back(*d_key2);
  hash_atts_d2.push_back(*d_payload2);

  RecordType *recTypeHash2 = new RecordType(hash_atts2);
  expressions::Expression *h2 =
      new expressions::InputArgument(recTypeHash2, -1, hash_atts_d2);
  expressions::Expression *h21 = new expressions::RecordProjection(h2, *d_key2);
  expressions::Expression *h22 =
      new expressions::RecordProjection(h2, *d_payload2);

  vector<expressions::Expression *> hash_projections2;
  hash_projections2.push_back(h21);
  hash_projections2.push_back(h22);

  RawOperator *part2 = new HashRearrange(btt21, ctx, numOfBuckets,
                                         hash_projections2, h21, hash_key2);
  btt21->setParent(part2);

  expr_projections_blockd_ptr2.push_back(hash_key2);

  Exchange *in_xch22 = new Exchange(part2, ctx, 1, expr_projections_blockd_ptr2,
                                    4, NULL, true, false, numPartitioners);
  part2->setParent(in_xch22);

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

  list<RecordAttribute *> f_atts;
  f_atts.push_back(d_key_block1);
  f_atts.push_back(d_payload_block1);
  f_atts.push_back(hash_key1);
  list<RecordAttribute> f_atts_d;
  // join_atts_d.push_back(*join_key1);
  f_atts_d.push_back(*d_key_block1);
  f_atts_d.push_back(*d_payload_block1);

  RecordType *recTypeF = new RecordType(f_atts);
  expressions::Expression *f1 =
      new expressions::InputArgument(recTypeF, -1, f_atts_d);
  expressions::Expression *f11 =
      new expressions::RecordProjection(f1, *d_key_block1);
  expressions::Expression *f12 =
      new expressions::RecordProjection(f1, *d_payload_block1);
  expressions::Expression *f3 =
      new expressions::InputArgument(recTypeF, -1, f_atts_d);
  expressions::Expression *f31 =
      new expressions::RecordProjection(f3, *d_key_block1);
  expressions::Expression *f32 =
      new expressions::RecordProjection(f3, *d_payload_block1);

  RecordAttribute *f_key1 =
      new RecordAttribute(1, "coordinator_A", "f_key1", new IntType(), false);
  RecordAttribute *f_key_block1 = new RecordAttribute(*f_key1, true);
  RecordAttribute *f_payload1 = new RecordAttribute(
      2, "coordinator_A", "f_payload1", new IntType(), false);
  RecordAttribute *f_payload_block1 = new RecordAttribute(*f_payload1, true);

  InputInfo *datasetInfoC1 =
      catalog.getOrCreateInputInfo(f_key1->getRelationName());
  RecordType *crec1 = new RecordType{dynamic_cast<const RecordType &>(
      dynamic_cast<CollectionType *>(datasetInfoC1->exprType)
          ->getNestedType())};
  crec1->appendAttribute(f_key1);
  crec1->appendAttribute(f_payload1);
  datasetInfoC1->exprType = new BagType{*crec1};
  f11->registerAs(f_key1);
  f12->registerAs(f_payload1);

  list<RecordAttribute *> f_atts2;
  f_atts2.push_back(d_key_block2);
  f_atts2.push_back(d_payload_block2);
  f_atts2.push_back(hash_key2);
  list<RecordAttribute> f_atts_d2;
  // join_atts_d.push_back(*join_key1);
  f_atts_d2.push_back(*d_key_block2);
  f_atts_d2.push_back(*d_payload_block2);

  RecordType *recTypeF2 = new RecordType(f_atts2);
  expressions::Expression *f2 =
      new expressions::InputArgument(recTypeF2, -1, f_atts_d2);
  expressions::Expression *f21 =
      new expressions::RecordProjection(f2, *d_key_block2);
  expressions::Expression *f22 =
      new expressions::RecordProjection(f2, *d_payload_block2);
  expressions::Expression *f4 =
      new expressions::InputArgument(recTypeF2, -1, f_atts_d2);
  expressions::Expression *f41 =
      new expressions::RecordProjection(f4, *d_key_block2);
  expressions::Expression *f42 =
      new expressions::RecordProjection(f4, *d_payload_block2);

  RecordAttribute *f_key2 =
      new RecordAttribute(-1, "coordinator_B", "f_key2", new IntType(), false);
  RecordAttribute *f_key_block2 = new RecordAttribute(*f_key2, true);
  RecordAttribute *f_payload2 = new RecordAttribute(
      -1, "coordinator_B", "f_payload2", new IntType(), false);
  RecordAttribute *f_payload_block2 = new RecordAttribute(*f_payload2, true);
  InputInfo *datasetInfoC2 =
      catalog.getOrCreateInputInfo(f_key2->getRelationName());
  RecordType *crec2 = new RecordType{dynamic_cast<const RecordType &>(
      dynamic_cast<CollectionType *>(datasetInfoC2->exprType)
          ->getNestedType())};
  crec2->appendAttribute(f_key2);
  crec2->appendAttribute(f_payload2);
  datasetInfoC2->exprType = new BagType{*crec2};
  f21->registerAs(f_key2);
  f22->registerAs(f_payload2);

  RecordAttribute *attr_ptr =
      new RecordAttribute(1, "coordinator", "ptr", new IntType(), true);
  RecordAttribute *attr_target =
      new RecordAttribute(1, "coordinator", "target", new IntType(), false);
  RecordAttribute *attr_target_block = new RecordAttribute(*attr_target, true);
  RecordAttribute *attr_splitter =
      new RecordAttribute(2, "coordinator", "splitter", new IntType(), false);
  InputInfo *datasetInfoCoord =
      catalog.getOrCreateInputInfo(attr_target->getRelationName());
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
  list<RecordAttribute> f_atts_target_d;
  f_atts_target_d.push_back(*attr_splitter);
  RecordType *recTypeTarget = new RecordType(f_atts_target);
  expressions::Expression *fTarg =
      new expressions::InputArgument(recTypeTarget, -1, f_atts_target_d);
  expressions::Expression *dest_expr =
      new expressions::RecordProjection(fTarg, *attr_splitter);

  list<RecordAttribute> f_atts_hash_d;
  f_atts_hash_d.push_back(*attr_target);
  expressions::Expression *fHtarg =
      new expressions::InputArgument(recTypeTarget, -1, f_atts_hash_d);
  expressions::Expression *expr_target =
      new expressions::RecordProjection(fHtarg, *attr_target);

  vector<expressions::Expression *> expr_v1;
  expr_v1.push_back(f11);
  expr_v1.push_back(f12);
  vector<expressions::Expression *> expr_v2;
  expr_v2.push_back(f21);
  expr_v2.push_back(f22);

  vector<RecordAttribute> expr_projections_block1;
  expr_projections_block1.push_back(*f_key_block1);
  expr_projections_block1.push_back(*f_payload_block1);

  vector<RecordAttribute *> expr_projections_block_ptr1;
  expr_projections_block_ptr1.push_back(f_key_block1);
  expr_projections_block_ptr1.push_back(f_payload_block1);

  vector<RecordAttribute> expr_projections_block2;
  expr_projections_block2.push_back(*f_key_block2);
  expr_projections_block2.push_back(*f_payload_block2);

  vector<RecordAttribute *> expr_projections_block_ptr2;
  expr_projections_block_ptr2.push_back(f_key_block2);
  expr_projections_block_ptr2.push_back(f_payload_block2);

  vector<RecordAttribute *> f_atts_target_v;
  f_atts_target_v.push_back(attr_ptr);
  f_atts_target_v.push_back(attr_target);
  f_atts_target_v.push_back(attr_splitter);

  /*ZipCollect* coord = new ZipCollect (attr_ptr, attr_splitter, attr_target,
     d_key_block1, d_key_block2, part1, part2, ctx, numOfBuckets, hash_key1,
     expr_v1, hash_key2, expr_v2, "coordinator");*/

  ZipCollect *coord =
      new ZipCollect(attr_ptr, attr_splitter, attr_target, d_key_block1,
                     d_key_block2, in_xch12, in_xch22, ctx, numOfBuckets,
                     hash_key1, expr_v1, hash_key2, expr_v2, "coordinator");

  // part1->setParent(coord);
  // part2->setParent(coord);

  in_xch12->setParent(coord);
  in_xch22->setParent(coord);

  // Exchange* xch1 = new Exchange (coord, ctx, numConcurrent, f_atts_target_v,
  // 4, expr_target, false); coord->setParent(xch1);

  // std::cout << "exch1" << xch1 << std::endl;

  ZipInitiate *initiator = new ZipInitiate(
      attr_ptr, attr_splitter, attr_target, coord, ctx, numOfBuckets,
      coord->getStateLeft(), coord->getStateRight(), "launcher");
  coord->setParent(initiator);
  // ZipInitiate* initiator = new ZipInitiate (attr_ptr, attr_splitter,
  // attr_target, xch1, ctx, numOfBuckets, coord->getStateLeft(),
  // coord->getStateRight(), "launcher"); xch1->setParent(initiator);

  vector<RecordAttribute *> targets;
  targets.push_back(attr_ptr);
  targets.push_back(attr_target);
  targets.push_back(attr_splitter);

  // SplitOpt* splitop = new SplitOpt(initiator, ctx, 2, targets, 4, dest_expr,
  // false, true); initiator->setParent(splitop);

  // std::cout << "split " << splitop << std::endl;

  ZipForward *fwd1 =
      new ZipForward(attr_splitter, attr_target, d_key_block1, initiator, ctx,
                     numOfBuckets, expr_v1, "forwarder", coord->getStateLeft());
  // splitop->setParent(fwd1);

  ZipForward *fwd2 = new ZipForward(attr_splitter, attr_target, d_key_block2,
                                    initiator, ctx, numOfBuckets, expr_v2,
                                    "forwarder", coord->getStateRight());
  // splitop->setParent(fwd2);

  RawPipelineGen **pip_rcv = initiator->pipeSocket();

  ///////////////////////////////////////////////////

  CollectionType *collTypeC1 =
      dynamic_cast<CollectionType *>(datasetInfoC1->exprType);
  const ExpressionType &nestedTypeC1 = collTypeC1->getNestedType();
  const RecordType &recType_C1 = dynamic_cast<const RecordType &>(nestedTypeC1);

  list<RecordAttribute> atts3 = list<RecordAttribute>();
  atts3.push_back(*f_key1);
  atts3.push_back(*f_payload1);

  expressions::Expression *df1 = new expressions::InputArgument(
      new RecordType(recType_C1.getArgs()), -1, atts3);
  expressions::Expression *df11 =
      new expressions::RecordProjection(df1, *f_key1);
  expressions::Expression *df12 =
      new expressions::RecordProjection(df1, *f_payload1);
  expressions::Expression *df3 = new expressions::InputArgument(
      new RecordType(recType_C1.getArgs()), -1, atts3);
  expressions::Expression *df31 =
      new expressions::RecordProjection(df3, *f_key1);
  expressions::Expression *df32 =
      new expressions::RecordProjection(df3, *f_payload1);

  vector<expressions::Expression *> exprs3;
  exprs3.push_back(df11);
  exprs3.push_back(df12);

  RawOperator *mml1 =
      new MemMoveLocalTo(fwd1, ctx, expr_projections_block_ptr1, 4);
  fwd1->setParent(mml1);

  RawOperator *mmd1 =
      new MemMoveDevice(mml1, ctx, expr_projections_block_ptr1, 4, false);
  mml1->setParent(mmd1);

  RawOperator *ctg1 = new CpuToGpu(mmd1, ctx, expr_projections_block_ptr1);
  mmd1->setParent(ctg1);

  RawOperator *btt12 = new BlockToTuples(ctg1, ctx, exprs3, true, gran_t::GRID);
  ctg1->setParent(btt12);

  expressions::Expression *lhs1 = df11;
  expressions::Expression *rhs1 = new expressions::IntConstant(2048000001);
  expressions::Expression *pred1 =
      new expressions::LtExpression(new BoolType(), lhs1, rhs1);

  RawOperator *select1 = new Select(pred1, btt12);
  btt12->setParent(select1);

  /////////////////////////////////////////////////////////////////////////

  CollectionType *collTypeC2 =
      dynamic_cast<CollectionType *>(datasetInfoC2->exprType);
  const ExpressionType &nestedTypeC2 = collTypeC2->getNestedType();
  const RecordType &recType_C2 = dynamic_cast<const RecordType &>(nestedTypeC2);

  list<RecordAttribute> atts4 = list<RecordAttribute>();
  atts4.push_back(*f_key2);
  atts4.push_back(*f_payload2);

  expressions::Expression *df2 = new expressions::InputArgument(
      new RecordType(recType_C2.getArgs()), -1, atts4);
  expressions::Expression *df21 =
      new expressions::RecordProjection(df2, *f_key2);
  expressions::Expression *df22 =
      new expressions::RecordProjection(df2, *f_payload2);
  expressions::Expression *df4 = new expressions::InputArgument(
      new RecordType(recType_C2.getArgs()), -1, atts4);
  expressions::Expression *df41 =
      new expressions::RecordProjection(df4, *f_key2);
  expressions::Expression *df42 =
      new expressions::RecordProjection(df4, *f_payload2);

  vector<expressions::Expression *> exprs4;
  exprs4.push_back(df21);
  exprs4.push_back(df22);

  RawOperator *mml2 =
      new MemMoveLocalTo(fwd2, ctx, expr_projections_block_ptr2, 4);
  fwd2->setParent(mml2);

  RawOperator *mmd2 =
      new MemMoveDevice(mml2, ctx, expr_projections_block_ptr2, 4, false);
  mml2->setParent(mmd2);

  RawOperator *ctg2 = new CpuToGpu(mmd2, ctx, expr_projections_block_ptr2);
  mmd2->setParent(ctg2);

  RawOperator *btt22 = new BlockToTuples(ctg2, ctx, exprs4, true, gran_t::GRID);
  ctg2->setParent(btt22);

  expressions::Expression *lhs2 = df21;
  expressions::Expression *rhs2 = new expressions::IntConstant(2048000001);
  expressions::Expression *pred2 =
      new expressions::LtExpression(new BoolType(), lhs2, rhs2);

  RawOperator *select2 = new Select(pred2, btt22);
  btt22->setParent(select2);

  ///////////////////////////////////////////////////////////////////////////////

  vector<RecordAttribute *> f_atts_dummy_v;
  f_atts_dummy_v.push_back(attr_target);
  list<RecordAttribute *> f_atts_dummy;
  f_atts_dummy.push_back(attr_target);
  list<RecordAttribute> f_atts_dummy_d;
  // join_atts_d.push_back(*join_key1);
  f_atts_dummy_d.push_back(*attr_target);
  RecordType *recTypeTarget2 = new RecordType(f_atts_dummy);
  expressions::Expression *dummy =
      new expressions::InputArgument(recTypeTarget2, -1, f_atts_dummy_d);
  expressions::Expression *dummy1 =
      new expressions::RecordProjection(dummy, *attr_target);

  vector<expressions::Expression *> expr_dummy;
  expr_dummy.push_back(dummy1);

  RecordAttribute *join_key1 =
      new RecordAttribute(1, "join198", key_str + "0", new IntType(), false);
  RecordAttribute *join_payload1 =
      new RecordAttribute(2, "join198", payload_str1, new IntType(), false);
  InputInfo *datasetInfo11 =
      catalog.getOrCreateInputInfo(join_key1->getRelationName());
  RecordType *rec1 = new RecordType{dynamic_cast<const RecordType &>(
      dynamic_cast<CollectionType *>(datasetInfo11->exprType)
          ->getNestedType())};
  rec1->appendAttribute(join_key1);
  rec1->appendAttribute(join_payload1);
  datasetInfo11->exprType = new BagType{*rec1};
  df31->registerAs(join_key1);
  df32->registerAs(join_payload1);

  RecordAttribute *join_key2 =
      new RecordAttribute(1, "join198", key_str, new IntType(), false);
  RecordAttribute *join_payload2 =
      new RecordAttribute(2, "join198", payload_str2, new IntType(), false);
  InputInfo *datasetInfo21 =
      catalog.getOrCreateInputInfo(join_key2->getRelationName());
  RecordType *rec2 = new RecordType{dynamic_cast<const RecordType &>(
      dynamic_cast<CollectionType *>(datasetInfo21->exprType)
          ->getNestedType())};
  rec2->appendAttribute(join_key2);
  rec2->appendAttribute(join_payload2);
  datasetInfo21->exprType = new BagType{*rec2};
  df41->registerAs(join_key2);
  df42->registerAs(join_payload2);

  vector<GpuMatExpr> build_e;
  GpuMatExpr gexpr1(df32, 1, 0);
  build_e.push_back(gexpr1);
  vector<size_t> build_widths;
  build_widths.push_back(64);
  vector<GpuMatExpr> probe_e;
  GpuMatExpr gexpr2(df42, 1, 0);
  probe_e.push_back(gexpr2);
  vector<size_t> probe_widths;
  probe_widths.push_back(64);

  vector<RecordAttribute *> unionFields;
  unionFields.push_back(attr_target_block);

  HashPartitioner *hpart1 =
      new HashPartitioner(attr_target, build_e, build_widths, df31, select1,
                          ctx, 13, "partition_hash_1");
  select1->setParent(hpart1);

  expressions::Expression *lhs3 = dummy1;
  expressions::Expression *rhs3 = new expressions::IntConstant(-1);
  expressions::Expression *pred3 =
      new expressions::LtExpression(new BoolType(), lhs3, rhs3);

  // RawOperator* select3 = new Select(pred3, hpart1);
  // hpart1->setParent(select3);
  // GpuHashRearrange* ttb1 = new GpuHashRearrange(select3, ctx, 1, expr_dummy,
  // dummy1); select3->setParent(ttb1);

  // GpuToCpu* gtc1 = new GpuToCpu(ttb1, ctx, unionFields, 131072);
  // ttb1->setParent(gtc1);

  HashPartitioner *hpart2 =
      new HashPartitioner(attr_target, probe_e, probe_widths, df41, select2,
                          ctx, 13, "partition_hash_2");
  select2->setParent(hpart2);

  expressions::Expression *lhs4 = dummy1;
  expressions::Expression *rhs4 = new expressions::IntConstant(-1);
  expressions::Expression *pred4 =
      new expressions::LtExpression(new BoolType(), lhs4, rhs4);

  // RawOperator* select4 = new Select(pred4, hpart2);
  // hpart2->setParent(select4);
  // GpuHashRearrange* ttb2 = new GpuHashRearrange(select4, ctx, 1, expr_dummy,
  // dummy1); select4->setParent(ttb2);

  // GpuToCpu* gtc2 = new GpuToCpu(ttb2, ctx, unionFields, 131072);
  // ttb2->setParent(gtc2);

  // vector<RawOperator*> children;

  // children.push_back(gtc1);
  // children.push_back(gtc2);

  // RawOperator* unionop = new UnionAll(children, ctx, unionFields);

  // std::cout << "union " << unionop << std::endl;

  // RawOperator* ctg = new CpuToGpu(unionop, ctx, unionFields);

  // hpart1->setParent(unionop);
  // hpart2->setParent(unionop);

  // RawOperator* btt =  new BlockToTuples(ctg, ctx, expr_dummy, true,
  // gran_t::GRID); ctg->setParent(btt);

  RawOperator *join = new GpuPartitionedHashJoinChained(
      build_e, build_widths, e31, hpart1, probe_e, probe_widths, e41, hpart2,
      hpart1->getState(), hpart2->getState(), 13, ctx, "hj_part", pip_rcv,
      NULL);
  // gtc1->setParent(unionop);
  // gtc2->setParent(unionop);
  // unionop->setParent(ctg);
  // btt->setParent(join);
  hpart1->setParent(join);
  hpart2->setParent(join);

  list<RecordAttribute *> join_atts;
  join_atts.push_back(join_key1);
  join_atts.push_back(join_key2);
  join_atts.push_back(join_payload1);
  join_atts.push_back(join_payload2);
  list<RecordAttribute> join_atts_d;
  // join_atts_d.push_back(*join_key1);
  join_atts_d.push_back(*join_payload1);
  join_atts_d.push_back(*join_payload2);

  RecordType *recTypeJoin = new RecordType(join_atts);
  expressions::Expression *e5 =
      new expressions::InputArgument(recTypeJoin, -1, join_atts_d);
  expressions::Expression *e51 =
      new expressions::RecordProjection(e5, *join_payload1);
  expressions::Expression *e52 =
      new expressions::RecordProjection(e5, *join_payload2);

  RecordAttribute *aggr1 =
      new RecordAttribute(1, "agg199", "EXPR$0", new IntType(), false);
  RecordAttribute *aggr2 =
      new RecordAttribute(2, "agg199", "EXPR$1", new IntType(), false);
  InputInfo *datasetInfo31 =
      catalog.getOrCreateInputInfo(aggr1->getRelationName());
  RecordType *rec3 = new RecordType{dynamic_cast<const RecordType &>(
      dynamic_cast<CollectionType *>(datasetInfo31->exprType)
          ->getNestedType())};
  rec3->appendAttribute(aggr1);
  rec3->appendAttribute(aggr2);
  datasetInfo31->exprType = new BagType{*rec3};
  e51->registerAs(aggr1);
  e52->registerAs(aggr2);

  vector<Monoid> accs;
  vector<expressions::Expression *> outputExprs;
  accs.push_back(SUM);
  outputExprs.push_back(e51);
  accs.push_back(SUM);
  outputExprs.push_back(e52);

  RawOperator *reduce = new opt::GpuReduce(
      accs, outputExprs, new expressions::BoolConstant(true), join, ctx);
  join->setParent(reduce);

  vector<RecordAttribute *> reduce_attrs;
  reduce_attrs.push_back(aggr1);
  reduce_attrs.push_back(aggr2);

  RawOperator *gtc =
      new GpuToCpu(reduce, ctx, reduce_attrs, 131072, gran_t::GRID);
  reduce->setParent(gtc);

  list<RecordAttribute *> aggr_atts;
  aggr_atts.push_back(aggr1);
  aggr_atts.push_back(aggr2);
  list<RecordAttribute> aggr_atts_d;
  aggr_atts_d.push_back(*aggr1);
  aggr_atts_d.push_back(*aggr2);

  RecordType *recTypeAggr = new RecordType(aggr_atts);
  expressions::Expression *e6 =
      new expressions::InputArgument(recTypeAggr, -1, aggr_atts_d);
  expressions::Expression *e61 = new expressions::RecordProjection(e6, *aggr1);
  expressions::Expression *e62 = new expressions::RecordProjection(e6, *aggr2);

  // Exchange* xch2 = new Exchange (gtc, ctx, 1, reduce_attrs, 4, new
  // expressions::IntConstant(0), false, false, numConcurrent);
  // gtc->setParent(xch2);

  RecordAttribute *aggr3 =
      new RecordAttribute(1, "agg200", "EXPR$0", new IntType(), false);
  RecordAttribute *aggr4 =
      new RecordAttribute(2, "agg200", "EXPR$1", new IntType(), false);
  InputInfo *datasetInfo41 =
      catalog.getOrCreateInputInfo(aggr3->getRelationName());
  RecordType *rec4 = new RecordType{dynamic_cast<const RecordType &>(
      dynamic_cast<CollectionType *>(datasetInfo41->exprType)
          ->getNestedType())};
  rec4->appendAttribute(aggr3);
  rec4->appendAttribute(aggr4);
  datasetInfo41->exprType = new BagType{*rec4};
  e61->registerAs(aggr3);
  e62->registerAs(aggr4);

  // std::cout << "exch2" << xch2 << std::endl;

  vector<expressions::Expression *> printExprs;
  printExprs.push_back(e61);
  printExprs.push_back(e62);

  // RawOperator* reduce2 = new opt::Reduce(accs, printExprs, new
  // expressions::BoolConstant(true), xch2, ctx); xch2->setParent(reduce2);
  RawOperator *reduce2 = new opt::Reduce(
      accs, printExprs, new expressions::BoolConstant(true), gtc, ctx);
  gtc->setParent(reduce2);

  list<RecordAttribute *> aggr_atts2;
  aggr_atts2.push_back(aggr3);
  aggr_atts2.push_back(aggr4);
  list<RecordAttribute> aggr_atts_d2;
  aggr_atts_d2.push_back(*aggr3);
  aggr_atts_d2.push_back(*aggr4);

  RecordType *recTypeAggr2 = new RecordType(aggr_atts2);
  expressions::Expression *e7 =
      new expressions::InputArgument(recTypeAggr2, -1, aggr_atts_d2);
  expressions::Expression *e71 = new expressions::RecordProjection(e7, *aggr3);
  expressions::Expression *e72 = new expressions::RecordProjection(e7, *aggr4);

  vector<expressions::Expression *> printExprs2;
  printExprs2.push_back(e71);
  printExprs2.push_back(e72);

  // RawOperator* flush = new Flush(printExprs, gtc, ctx, "out1.json");
  // gtc->setParent(flush);
  RawOperator *flush = new Flush(printExprs2, reduce2, ctx, "out1.json");
  reduce2->setParent(flush);

  std::cout << "tree built" << std::endl;

  /*RecordAttribute* aggr2 = new RecordAttribute(-1, "agg199", "EXPR$0", new
  IntType(), false); InputInfo * datasetInfo101 =
  catalog.getOrCreateInputInfo(aggr2->getRelationName()); RecordType * rec101 =
  new RecordType{dynamic_cast<const RecordType &>(dynamic_cast<CollectionType
  *>(datasetInfo101->exprType)->getNestedType())};
  rec101->appendAttribute(aggr2);
  datasetInfo101->exprType = new BagType{*rec101};
  e31->registerAs(aggr);

  vector<expressions::Expression*> outputExprs2;
  outputExprs2.push_back(e31);

  RawOperator* red = new opt::Reduce(accs, outputExprs2, new
  expressions::BoolConstant(true), btt11, ctx, false,
  "/home/sioulas/pelago/pelago/res.json"); btt11->setParent(red);*/

  RawOperator *root = flush;
  root->produce();

  std::cout << "produced" << std::endl;

  ctx->prepareFunction(ctx->getGlobalFunction());
  {
    RawCatalog &catalog = RawCatalog::getInstance();
    /* XXX Remove when testing caches (?) */
    catalog.clear();
  }

  ctx->compileAndLoad();

  std::vector<RawPipeline *> pipelines;

  pipelines = ctx->getPipelines();

  for (RawPipeline *p : pipelines) {
    {
      time_block t("T: ");
      // std::cout << p->getName() << std::endl;
      p->open();
      p->consume(0);
      p->close();
    }
  }
}

/*void Query3 () {
    {
        RawCatalog     * catalog = &RawCatalog::getInstance();
        CachingService * caches  = &CachingService::getInstance();
        catalog->clear();
        caches->clear();
    }

    ParallelContext * ctx   = new ParallelContext("Out_of_GPU", false);
    CatalogParser catalog = CatalogParser("inputs/plans/catalog.json", ctx);

    RawCatalog& rawcatalog = RawCatalog::getInstance();

    int numOfBuckets = 8;

    string key_str("d_key");
    string part_str("partition_hash_");
    string relName1("inputs/A_128000000.bin");


    InputInfo *datasetInfo1 = catalog.getInputInfoIfKnown(relName1);
    std::cout << datasetInfo1 << std::endl;
    std::cout << datasetInfo1->path << std::endl;
    string *path1 = new string(datasetInfo1->path);
    CollectionType *collType1 =
dynamic_cast<CollectionType*>(datasetInfo1->exprType); const ExpressionType&
nestedType1 = collType1->getNestedType(); const RecordType& recType_1 =
dynamic_cast<const RecordType&>(nestedType1);

    RecordType *recType1 = new RecordType(recType_1.getArgs());

    RecordAttribute* d_key1 = new RecordAttribute(1, relName1, key_str, new
IntType(), false); //recType1->getArg(key_str); std::cout << d_key1->getType()
<< std::endl; int attrNo = 1; const ExpressionType* recArgType =
d_key1->getOriginalType(); RecordAttribute* d_key_block1 = new
RecordAttribute(attrNo, relName1, key_str, recArgType, true); std::cout <<
d_key_block1->getType() << std::endl;

    vector<RecordAttribute> expr_projections1;
    expr_projections1.push_back(*d_key1);

    vector<RecordAttribute*> expr_projections_ptr1;
    expr_projections_ptr1.push_back(d_key1);

    list<RecordAttribute> atts1 = list<RecordAttribute>();
    atts1.push_back(*d_key1);
    expressions::Expression* e1 = new expressions::InputArgument(recType1, -1,
atts1); expressions::Expression* e11 = new expressions::RecordProjection(e1,
*d_key1); expressions::Expression* e3 = new expressions::InputArgument(recType1,
-1, atts1); expressions::Expression* e31 = new expressions::RecordProjection(e3,
*d_key1);


    Plugin* pg1 = new ScanToBlockSMPlugin(ctx, *path1, *recType1,
expr_projections_ptr1); rawcatalog.registerPlugin(*path1, pg1); RawOperator*
scan1 = new Scan(ctx, *pg1);

    vector<expressions::Expression*> exprs1;
    exprs1.push_back(e11);

    RawOperator* btt11 =  new BlockToTuples(scan1, ctx, exprs1, false,
gran_t::THREAD); scan1->setParent(btt11);

    RecordAttribute* hash_key1 = new RecordAttribute(-1, relName1, part_str +
relName1, new IntType(), false); InputInfo * datasetInfoH1 =
catalog.getOrCreateInputInfo(hash_key1->getRelationName()); RecordType * hrec1 =
new RecordType{dynamic_cast<const RecordType &>(dynamic_cast<CollectionType
*>(datasetInfoH1->exprType)->getNestedType())};
    hrec1->appendAttribute(hash_key1);
    datasetInfoH1->exprType = new BagType{*hrec1};
    //e31->registerAs(join_key1);

    list<RecordAttribute*> hash_atts;
    hash_atts.push_back(d_key1);
    hash_atts.push_back(hash_key1);
    list<RecordAttribute> hash_atts_d;
    //join_atts_d.push_back(*join_key1);
    hash_atts_d.push_back(*d_key1);

    RecordType *recTypeHash = new RecordType(hash_atts);
    expressions::Expression* h1 = new expressions::InputArgument(recTypeHash,
-1, hash_atts_d); expressions::Expression* h11 = new
expressions::RecordProjection(h1, *d_key1);

    vector<expressions::Expression*> hash_projections;
    hash_projections.push_back(h11);

    RawOperator* part1 = new HashRearrange(btt11, ctx, numOfBuckets,
hash_projections, h11, hash_key1); btt11->setParent(part1);

    ////////////////////////////////////////////////////////////////////////////////////////////

    string relName2("inputs/B_128000000.bin");


    InputInfo *datasetInfo2 = catalog.getInputInfoIfKnown(relName2);
    std::cout << datasetInfo2 << std::endl;
    std::cout << datasetInfo2->path << std::endl;
    string *path2 = new string(datasetInfo2->path);
    CollectionType *collType2 =
dynamic_cast<CollectionType*>(datasetInfo2->exprType); const ExpressionType&
nestedType2 = collType2->getNestedType(); const RecordType& recType_2 =
dynamic_cast<const RecordType&>(nestedType2);

    RecordType *recType2 = new RecordType(recType_2.getArgs());

    RecordAttribute* d_key2 = new RecordAttribute(1, relName2, key_str, new
IntType(), false); //recType1->getArg(key_str); std::cout << d_key2->getType()
<< std::endl; int attrNo2 = 1; const ExpressionType* recArgType2 =
d_key2->getOriginalType(); RecordAttribute* d_key_block2 = new
RecordAttribute(attrNo2, relName2, key_str, recArgType2, true); std::cout <<
d_key_block2->getType() << std::endl;

    vector<RecordAttribute> expr_projections2;
    expr_projections2.push_back(*d_key2);

    vector<RecordAttribute*> expr_projections_ptr2;
    expr_projections_ptr2.push_back(d_key2);

    list<RecordAttribute> atts2 = list<RecordAttribute>();
    atts2.push_back(*d_key2);
    expressions::Expression* e2 = new expressions::InputArgument(recType2, -1,
atts2); expressions::Expression* e21 = new expressions::RecordProjection(e2,
*d_key2); expressions::Expression* e4 = new expressions::InputArgument(recType2,
-1, atts2); expressions::Expression* e41 = new expressions::RecordProjection(e4,
*d_key2);

    Plugin* pg2 = new ScanToBlockSMPlugin(ctx, *path2, *recType2,
expr_projections_ptr2); rawcatalog.registerPlugin(*path2, pg2); RawOperator*
scan2 = new Scan(ctx, *pg2);

    vector<expressions::Expression*> exprs2;
    exprs2.push_back(e21);

    RawOperator* btt21 =  new BlockToTuples(scan2, ctx, exprs2, false,
gran_t::THREAD); scan2->setParent(btt21);

    RecordAttribute* hash_key2 = new RecordAttribute(-1, relName2, part_str +
relName2, new IntType(), false); InputInfo * datasetInfoH2 =
catalog.getOrCreateInputInfo(hash_key2->getRelationName()); RecordType * hrec2 =
new RecordType{dynamic_cast<const RecordType &>(dynamic_cast<CollectionType
*>(datasetInfoH2->exprType)->getNestedType())};
    hrec2->appendAttribute(hash_key2);
    datasetInfoH2->exprType = new BagType{*hrec2};
    //e31->registerAs(join_key1);

    list<RecordAttribute*> hash_atts2;
    hash_atts2.push_back(d_key2);
    hash_atts2.push_back(hash_key2);
    list<RecordAttribute> hash_atts_d2;
    //join_atts_d.push_back(*join_key1);
    hash_atts_d2.push_back(*d_key2);

    RecordType *recTypeHash2 = new RecordType(hash_atts2);
    expressions::Expression* h2 = new expressions::InputArgument(recTypeHash2,
-1, hash_atts_d2); expressions::Expression* h21 = new
expressions::RecordProjection(h2, *d_key2);

    vector<expressions::Expression*> hash_projections2;
    hash_projections2.push_back(h21);

    RawOperator* part2 = new HashRearrange(btt21, ctx, numOfBuckets,
hash_projections2, h21, hash_key2); btt21->setParent(part2);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

    list<RecordAttribute*> f_atts;
    f_atts.push_back(d_key_block1);
    f_atts.push_back(hash_key1);
    list<RecordAttribute> f_atts_d;
    //join_atts_d.push_back(*join_key1);
    f_atts_d.push_back(*d_key_block1);
    RecordType *recTypeF = new RecordType(f_atts);
    expressions::Expression* f1 = new expressions::InputArgument(recTypeF, -1,
f_atts_d); expressions::Expression* f11 = new expressions::RecordProjection(f1,
*d_key_block1); expressions::Expression* f3 = new
expressions::InputArgument(recTypeF, -1, f_atts_d); expressions::Expression* f31
= new expressions::RecordProjection(f3, *d_key_block1);

    RecordAttribute* f_key1 = new RecordAttribute(-1, "coordinator_A", "f_key1",
new IntType(), false); RecordAttribute* f_key_block1 = new
RecordAttribute(*f_key1, true); InputInfo * datasetInfoC1 =
catalog.getOrCreateInputInfo(f_key1->getRelationName()); RecordType * crec1 =
new RecordType{dynamic_cast<const RecordType &>(dynamic_cast<CollectionType
*>(datasetInfoC1->exprType)->getNestedType())}; crec1->appendAttribute(f_key1);
    datasetInfoC1->exprType = new BagType{*crec1};
    f11->registerAs(f_key1);

    list<RecordAttribute*> f_atts2;
    f_atts2.push_back(d_key_block2);
    f_atts2.push_back(hash_key2);
    list<RecordAttribute> f_atts_d2;
    //join_atts_d.push_back(*join_key1);
    f_atts_d2.push_back(*d_key_block2);
    RecordType *recTypeF2 = new RecordType(f_atts2);
    expressions::Expression* f2 = new expressions::InputArgument(recTypeF2, -1,
f_atts_d2); expressions::Expression* f21 = new expressions::RecordProjection(f2,
*d_key_block2); expressions::Expression* f4 = new
expressions::InputArgument(recTypeF2, -1, f_atts_d2); expressions::Expression*
f41 = new expressions::RecordProjection(f4, *d_key_block2);

    RecordAttribute* f_key2 = new RecordAttribute(-1, "coordinator_B", "f_key2",
new IntType(), false); RecordAttribute* f_key_block2 = new
RecordAttribute(*f_key2, true); InputInfo * datasetInfoC2 =
catalog.getOrCreateInputInfo(f_key2->getRelationName()); RecordType * crec2 =
new RecordType{dynamic_cast<const RecordType &>(dynamic_cast<CollectionType
*>(datasetInfoC2->exprType)->getNestedType())}; crec2->appendAttribute(f_key2);
    datasetInfoC2->exprType = new BagType{*crec2};
    f21->registerAs(f_key2);

    vector<expressions::Expression*> expr_v1;
    expr_v1.push_back(f11);
    vector<expressions::Expression*> expr_v2;
    expr_v2.push_back(f21);

    vector<RecordAttribute> expr_projections_block1;
    expr_projections_block1.push_back(*f_key_block1);

    vector<RecordAttribute*> expr_projections_block_ptr1;
    expr_projections_block_ptr1.push_back(f_key_block1);

    vector<RecordAttribute> expr_projections_block2;
    expr_projections_block2.push_back(*f_key_block2);

    vector<RecordAttribute*> expr_projections_block_ptr2;
    expr_projections_block_ptr2.push_back(f_key_block2);

    ZipCollect* coord = new ZipCollect (    d_key_block1, d_key_block2,
                                        part1, part2, ctx, numOfBuckets,
                                        hash_key1, expr_v1,
                                        hash_key2, expr_v2, "coordinator");

    part1->setParent(coord);
    part2->setParent(coord);

    ZipForward* fwd = new ZipForward (d_key_block1, d_key_block2,
                                        coord, ctx, numOfBuckets,
                                        expr_v1, expr_v2, "forwarder");
    coord->setParent(fwd);


    RawOperator* ch_left = new ZipLeft (fwd);
    RawOperator* ch_right = new ZipRight (fwd);

    ///////////////////////////////////////////////////

    CollectionType *collTypeC1 =
dynamic_cast<CollectionType*>(datasetInfoC1->exprType); const ExpressionType&
nestedTypeC1 = collTypeC1->getNestedType(); const RecordType& recType_C1 =
dynamic_cast<const RecordType&>(nestedTypeC1);

    list<RecordAttribute> atts3 = list<RecordAttribute>();
    atts3.push_back(*f_key1);
    expressions::Expression* df1 = new expressions::InputArgument(new
RecordType(recType_C1.getArgs()), -1, atts3); expressions::Expression* df11 =
new expressions::RecordProjection(df1, *f_key1); expressions::Expression* df3 =
new expressions::InputArgument(new RecordType(recType_C1.getArgs()), -1, atts3);
    expressions::Expression* df31 = new expressions::RecordProjection(df3,
*f_key1);

    vector<expressions::Expression*> exprs3;
    exprs3.push_back(df11);

    //RawOperator* mml1 =  new MemMoveLocalTo(ch_left, ctx,
expr_projections_block_ptr1, 4);
    //ch_left->setParent(mml1);

    RawOperator* btt12 =  new BlockToTuples(ch_left, ctx, exprs3, false,
gran_t::THREAD); ch_left->setParent(btt12);

    expressions::Expression* lhs1 = df11;
    expressions::Expression* rhs1 = new expressions::IntConstant(129000000);
    expressions::Expression* pred1 = new expressions::LtExpression(new
BoolType(), lhs1, rhs1);

    RawOperator* select1 = new Select(pred1, btt12);
    btt12->setParent(select1);

    /////////////////////////////////////////////////////////////////////////

    CollectionType *collTypeC2 =
dynamic_cast<CollectionType*>(datasetInfoC2->exprType); const ExpressionType&
nestedTypeC2 = collTypeC2->getNestedType(); const RecordType& recType_C2 =
dynamic_cast<const RecordType&>(nestedTypeC2);

    list<RecordAttribute> atts4 = list<RecordAttribute>();
    atts4.push_back(*f_key2);
    expressions::Expression* df2 = new expressions::InputArgument(new
RecordType(recType_C2.getArgs()), -1, atts4); expressions::Expression* df21 =
new expressions::RecordProjection(df2, *f_key2); expressions::Expression* df4 =
new expressions::InputArgument(new RecordType(recType_C2.getArgs()), -1, atts4);
    expressions::Expression* df41 = new expressions::RecordProjection(df4,
*f_key2);

    vector<expressions::Expression*> exprs4;
    exprs4.push_back(df21);

    //RawOperator* mml2 =  new MemMoveLocalTo(ch_right, ctx,
expr_projections_block_ptr2, 4);
    //ch_right->setParent(mml2);

    RawOperator* btt22 =  new BlockToTuples(ch_right, ctx, exprs4, false,
gran_t::THREAD); ch_right->setParent(btt22);

    expressions::Expression* lhs2 = df21;
    expressions::Expression* rhs2 = new expressions::IntConstant(129000000);
    expressions::Expression* pred2 = new expressions::LtExpression(new
BoolType(), lhs2, rhs2);

    RawOperator* select2 = new Select(pred2, btt22);
    btt22->setParent(select2);

    ///////////////////////////////////////////////////////////////////////////////

    RecordAttribute* join_key1 = new RecordAttribute(-1, "join198", key_str+"0",
new IntType(), false); InputInfo * datasetInfo11 =
catalog.getOrCreateInputInfo(join_key1->getRelationName()); RecordType * rec1 =
new RecordType{dynamic_cast<const RecordType &>(dynamic_cast<CollectionType
*>(datasetInfo11->exprType)->getNestedType())};
    rec1->appendAttribute(join_key1);
    datasetInfo11->exprType = new BagType{*rec1};
    df31->registerAs(join_key1);

    RecordAttribute* join_key2 = new RecordAttribute(-1, "join198", key_str, new
IntType(), false); InputInfo * datasetInfo21 =
catalog.getOrCreateInputInfo(join_key2->getRelationName()); RecordType * rec2 =
new RecordType{dynamic_cast<const RecordType &>(dynamic_cast<CollectionType
*>(datasetInfo21->exprType)->getNestedType())};
    rec2->appendAttribute(join_key2);
    datasetInfo21->exprType = new BagType{*rec2};
    df41->registerAs(join_key2);

    vector<GpuMatExpr> build_e;
    vector<size_t> build_widths;
    build_widths.push_back(64);
    vector<GpuMatExpr> probe_e;
    vector<size_t> probe_widths;
    probe_widths.push_back(64);

    RawOperator* join = new HashJoinChained(build_e, build_widths, df31,
select1, probe_e, probe_widths, df41, select2, 28, ctx, 128000000);
    select1->setParent(join);
    select2->setParent(join);

    list<RecordAttribute*> join_atts;
    join_atts.push_back(join_key1);
    join_atts.push_back(join_key2);
    list<RecordAttribute> join_atts_d;
    //join_atts_d.push_back(*join_key1);
    join_atts_d.push_back(*join_key2);

    RecordType *recTypeJoin = new RecordType(join_atts);
    expressions::Expression* e5 = new expressions::InputArgument(recTypeJoin,
-1, join_atts_d); expressions::Expression* e51 = new
expressions::RecordProjection(e5, *join_key2);

    RecordAttribute* aggr = new RecordAttribute(-1, "agg199", "EXPR$0", new
IntType(), false); InputInfo * datasetInfo31 =
catalog.getOrCreateInputInfo(aggr->getRelationName()); RecordType * rec3 = new
RecordType{dynamic_cast<const RecordType &>(dynamic_cast<CollectionType
*>(datasetInfo31->exprType)->getNestedType())}; rec3->appendAttribute(aggr);
    datasetInfo31->exprType = new BagType{*rec3};
    e51->registerAs(aggr);

    vector<Monoid> accs;
    vector<expressions::Expression*> outputExprs;
    accs.push_back(SUM);
    outputExprs.push_back(e51);

    RawOperator* reduce = new opt::Reduce(accs, outputExprs, new
expressions::BoolConstant(true), join, ctx, false,
"/home/sioulas/pelago/pelago/res.json"); join->setParent(reduce);

    list<RecordAttribute*> aggr_atts;
    aggr_atts.push_back(aggr);
    list<RecordAttribute> aggr_atts_d;
    aggr_atts_d.push_back(*aggr);

    RecordType *recTypeAggr = new RecordType(aggr_atts);
    expressions::Expression* e6 = new expressions::InputArgument(recTypeAggr,
-1, aggr_atts_d); expressions::Expression* e61 = new
expressions::RecordProjection(e6, *aggr);

    vector<expressions::Expression*> printExprs;
    printExprs.push_back(e61);
    RawOperator* flush = new Flush(printExprs, reduce, ctx);
    reduce->setParent(flush);

    std::cout << "tree built" << std::endl;


    RawOperator* root = flush;
    root->produce();

    std::cout << "produced" << std::endl;

    ctx->prepareFunction(ctx->getGlobalFunction());
    {
        RawCatalog& catalog = RawCatalog::getInstance();

        catalog.clear();
    }

    ctx->compileAndLoad();

    std::vector<RawPipeline *> pipelines;

    pipelines = ctx->getPipelines();

    for (RawPipeline * p: pipelines) {
                {
                    time_block t("T: ");
                    //std::cout << p->getName() << std::endl;
                    p->open();
                    p->consume(0);
                    p->close();
                }

    }
}*/

int main(void) {
  RawPipelineGen::init();
  RawMemoryManager::init();

  gpu_run(cudaSetDevice(0));

  {
    auto load = [](string filename) {
      // StorageManager::load(filename, PINNED);
      StorageManager::loadToCpus(filename);
    };

    load("inputs/micro/A_256000000.bin.d_key");
    load("inputs/micro/A_256000000.bin.d_val2");
    load("inputs/micro/B_256000000.bin.d_key");
    load("inputs/micro/B_256000000.bin.d_val");
  }
  gpu_run(cudaSetDevice(0));

  std::cout << "ready" << std::endl;

  Query2();

  return 0;
}
