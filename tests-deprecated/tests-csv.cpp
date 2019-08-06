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

#include "common/common.hpp"
#include "expressions/binary-operators.hpp"
#include "expressions/expressions.hpp"
#include "gtest/gtest.h"
#include "operators/join.hpp"
#include "operators/outer-unnest.hpp"
#include "operators/print.hpp"
#include "operators/reduce.hpp"
#include "operators/root.hpp"
#include "operators/scan.hpp"
#include "operators/select.hpp"
#include "operators/unnest.hpp"
#include "plugins/csv-plugin-pm.hpp"
#include "plugins/csv-plugin.hpp"
#include "plugins/json-jsmn-plugin.hpp"
#include "util/context.hpp"
#include "util/functions.hpp"
#include "values/expressionTypes.hpp"

TEST(CSV, ScanCsvPM) {
  Context &ctx = *prepareContext("ScanCsvPM");
  Catalog &catalog = Catalog::getInstance();

  /**
   * SCAN
   */
  string filename = string("inputs/sailors.csv");
  PrimitiveType *intType = new IntType();
  PrimitiveType *floatType = new FloatType();
  PrimitiveType *stringType = new StringType();
  RecordAttribute *sid =
      new RecordAttribute(1, filename, string("sid"), intType);
  RecordAttribute *sname =
      new RecordAttribute(2, filename, string("sname"), stringType);
  RecordAttribute *rating =
      new RecordAttribute(3, filename, string("rating"), intType);
  RecordAttribute *age =
      new RecordAttribute(4, filename, string("age"), floatType);

  list<RecordAttribute *> attrList;
  attrList.push_back(sid);
  attrList.push_back(sname);
  attrList.push_back(rating);
  attrList.push_back(age);

  RecordType rec1 = RecordType(attrList);

  vector<RecordAttribute *> whichFields;
  whichFields.push_back(sid);
  whichFields.push_back(age);

  /* 1 every 5 fields indexed in PM */
  pm::CSVPlugin *pg =
      new pm::CSVPlugin(&ctx, filename, rec1, whichFields, 10, 2);
  catalog.registerPlugin(filename, pg);
  Scan scan = Scan(&ctx, *pg);

  /**
   * ROOT
   */
  Root rootOp = Root(&scan);
  scan.setParent(&rootOp);
  rootOp.produce();

  // Run function
  ctx.prepareFunction(ctx.getGlobalFunction());

  // Close all open files & clear
  pg->finish();
  catalog.clear();
}

TEST(CSV, ScanCsvWideBuildPM) {
  Context &ctx = *prepareContext("ScanCsvWidePM");
  Catalog &catalog = Catalog::getInstance();

  /**
   * SCAN
   */
  string filename = string("inputs/csv/30cols.csv");
  PrimitiveType *intType = new IntType();
  list<RecordAttribute *> attrList;
  RecordAttribute *attr6, *attr10, *attr19, *attr21, *attr26;

  for (int i = 1; i <= 30; i++) {
    RecordAttribute *attr = new RecordAttribute(i, filename, "field", intType);
    attrList.push_back(attr);

    if (i == 6) {
      attr6 = attr;
    }

    if (i == 10) {
      attr10 = attr;
    }

    if (i == 19) {
      attr19 = attr;
    }

    if (i == 21) {
      attr21 = attr;
    }

    if (i == 26) {
      attr26 = attr;
    }
  }

  RecordType rec1 = RecordType(attrList);

  vector<RecordAttribute *> whichFields;
  whichFields.push_back(attr6);
  whichFields.push_back(attr10);
  whichFields.push_back(attr19);
  whichFields.push_back(attr21);
  whichFields.push_back(attr26);

  /* 1 every 5 fields indexed in PM */
  pm::CSVPlugin *pg =
      new pm::CSVPlugin(&ctx, filename, rec1, whichFields, 10, 6);
  catalog.registerPlugin(filename, pg);
  Scan scan = Scan(&ctx, *pg);

  /**
   * ROOT
   */
  Root rootOp = Root(&scan);
  scan.setParent(&rootOp);
  rootOp.produce();

  // Run function
  ctx.prepareFunction(ctx.getGlobalFunction());

  // Close all open files & clear
  pg->finish();
  catalog.clear();
}

void scanCsvWideUsePM_(size_t *newline, short **offsets) {
  Context &ctx = *prepareContext("ScanCsvWideUsePM");
  Catalog &catalog = Catalog::getInstance();

  /**
   * SCAN
   */
  string filename = string("inputs/csv/30cols.csv");
  PrimitiveType *intType = new IntType();
  list<RecordAttribute *> attrList;
  RecordAttribute *attr6, *attr10, *attr19, *attr21, *attr26;

  for (int i = 1; i <= 30; i++) {
    RecordAttribute *attr = new RecordAttribute(i, filename, "field", intType);
    attrList.push_back(attr);

    if (i == 6) {
      attr6 = attr;
    }

    if (i == 10) {
      attr10 = attr;
    }

    if (i == 19) {
      attr19 = attr;
    }

    if (i == 21) {
      attr21 = attr;
    }

    if (i == 26) {
      attr26 = attr;
    }
  }

  RecordType rec1 = RecordType(attrList);

  vector<RecordAttribute *> whichFields;
  whichFields.push_back(attr6);
  whichFields.push_back(attr10);
  whichFields.push_back(attr19);
  whichFields.push_back(attr21);
  whichFields.push_back(attr26);

  /* 1 every 5 fields indexed in PM */
  pm::CSVPlugin *pg = new pm::CSVPlugin(&ctx, filename, rec1, whichFields, ';',
                                        10, 6, newline, offsets);
  catalog.registerPlugin(filename, pg);
  Scan scan = Scan(&ctx, *pg);

  /**
   * ROOT
   */
  Root rootOp = Root(&scan);
  scan.setParent(&rootOp);
  rootOp.produce();

  // Run function
  ctx.prepareFunction(ctx.getGlobalFunction());

  // Close all open files & clear
  pg->finish();
  catalog.clear();
}

TEST(CSV, scanCsvWideUsePM) {
  Context &ctx = *prepareContext("ScanCsvWideBuildPM");
  Catalog &catalog = Catalog::getInstance();

  /**
   * SCAN
   */
  string filename = string("inputs/csv/30cols.csv");
  PrimitiveType *intType = new IntType();
  list<RecordAttribute *> attrList;
  RecordAttribute *attr6, *attr10, *attr19, *attr21, *attr26;

  for (int i = 1; i <= 30; i++) {
    RecordAttribute *attr = new RecordAttribute(i, filename, "field", intType);
    attrList.push_back(attr);

    if (i == 6) {
      attr6 = attr;
    }

    if (i == 10) {
      attr10 = attr;
    }

    if (i == 19) {
      attr19 = attr;
    }

    if (i == 21) {
      attr21 = attr;
    }

    if (i == 26) {
      attr26 = attr;
    }
  }

  RecordType rec1 = RecordType(attrList);

  vector<RecordAttribute *> whichFields;
  whichFields.push_back(attr6);
  whichFields.push_back(attr10);
  whichFields.push_back(attr19);
  whichFields.push_back(attr21);
  whichFields.push_back(attr26);

  /* 1 every 5 fields indexed in PM */
  pm::CSVPlugin *pg =
      new pm::CSVPlugin(&ctx, filename, rec1, whichFields, 10, 6);
  catalog.registerPlugin(filename, pg);
  Scan scan = Scan(&ctx, *pg);

  /**
   * ROOT
   */
  Root rootOp = Root(&scan);
  scan.setParent(&rootOp);
  rootOp.produce();

  // Run function
  ctx.prepareFunction(ctx.getGlobalFunction());

  /*
   * Use PM in subsequent scan
   */
  scanCsvWideUsePM_(pg->getNewlinesPM(), pg->getOffsetsPM());

  // Close all open files & clear
  pg->finish();
  catalog.clear();
}

TEST(CSV, atoiCSV) {
  Context &ctx = *prepareContext("AtoiCSV");
  Catalog &catalog = Catalog::getInstance();

  /**
   * SCAN
   */
  string filename = string("inputs/csv/small.csv");
  PrimitiveType *intType = new IntType();
  PrimitiveType *floatType = new FloatType();
  PrimitiveType *stringType = new StringType();
  RecordAttribute *f1 = new RecordAttribute(1, filename, string("f1"), intType);
  RecordAttribute *f2 = new RecordAttribute(2, filename, string("f2"), intType);
  RecordAttribute *f3 = new RecordAttribute(3, filename, string("f3"), intType);
  RecordAttribute *f4 = new RecordAttribute(4, filename, string("f4"), intType);
  RecordAttribute *f5 = new RecordAttribute(5, filename, string("f5"), intType);

  list<RecordAttribute *> attrList;
  attrList.push_back(f1);
  attrList.push_back(f2);
  attrList.push_back(f3);
  attrList.push_back(f4);
  attrList.push_back(f5);

  RecordType rec1 = RecordType(attrList);

  vector<RecordAttribute *> whichFields;
  whichFields.push_back(f1);
  whichFields.push_back(f2);

  CSVPlugin *pg = new CSVPlugin(&ctx, filename, rec1, whichFields);
  catalog.registerPlugin(filename, pg);
  Scan scan = Scan(&ctx, *pg);

  /**
   * ROOT
   */
  Root rootOp = Root(&scan);
  scan.setParent(&rootOp);
  rootOp.produce();

  // Run function
  ctx.prepareFunction(ctx.getGlobalFunction());

  // Close all open files & clear
  pg->finish();
  catalog.clear();
}
