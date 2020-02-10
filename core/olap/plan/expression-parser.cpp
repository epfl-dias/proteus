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

#include "expression-parser.hpp"

#include <util/parallel-context.hpp>

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

expressions::extract_unit ExpressionParser::parseUnitRange(
    std::string range, ParallelContext *ctx) {
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
                                               ParallelContext *ctx) {
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
    const rapidjson::Value &val, ParallelContext *ctx) {
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
      if (val["v"].IsInt64()) {
        retValue = new expressions::DateConstant(val["v"].GetInt64());
      } else {
        assert(val["v"].IsString());
        retValue = new expressions::DateConstant(val["v"].GetString());
      }
    }
  } else if (strcmp(valExpression, "datetime") == 0) {
    if (isNull) {
      retValue = createNull(new DateType());
    } else {
      assert(val.HasMember("v"));
      if (val["v"].IsInt64()) {
        retValue = new expressions::DateConstant(val["v"].GetInt64());
      } else {
        assert(val["v"].IsString());
        retValue = new expressions::DateConstant(val["v"].GetString());
      }
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
    //    assert(!isNull);
    //    /* exprType */
    //    assert(val.HasMember(keyExprType));
    //    assert(val[keyExprType].IsObject());
    //    ExpressionType *exprType = parseExpressionType(val[keyExprType]);
    //
    //    /* argNo */
    //    assert(val.HasMember(keyArgNo));
    //    assert(val[keyArgNo].IsInt());
    //    int argNo = val[keyArgNo].GetInt();
    //
    //    /* 'projections' / attributes */
    //    assert(val.HasMember(keyAtts));
    //    assert(val[keyAtts].IsArray());
    //
    //    list<RecordAttribute> atts;
    //    for (const auto &v : val[keyAtts].GetArray()) {
    //      atts.emplace_back(*parseRecordAttr(v));
    //    }

    return arg;
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

    return expr[*recAttr];
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

      expressions::AttributeConstruction newAttr(newAttrName, newAttrExpr);
      newAtts.push_back(newAttr);
    }
    return expressions::RecordConstruction{newAtts};
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
        return new RecordAttribute(*arg.getExpressionType()->getArg(attrName));
        //        assert(false && "Attribute not found");
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
