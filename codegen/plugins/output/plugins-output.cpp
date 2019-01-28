/*
    RAW -- High-performance querying over raw, never-seen-before data.

                            Copyright (c) 2014
        Data Intensive Applications and Systems Labaratory (DIAS)
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

#include "plugins/output/plugins-output.hpp"
#include "expressions/expressions-generator.hpp"

Materializer::Materializer(
    vector<RecordAttribute *> wantedFields,
    //        const vector<expression_t>& wantedExpressions,
    vector<materialization_mode> outputMode)
    :  //        wantedExpressions(wantedExpressions),
      wantedFields(wantedFields),
      outputMode(outputMode) {
  oidsProvided = false;
}

Materializer::Materializer(vector<RecordAttribute *> whichFields,
                           vector<expression_t> wantedExpressions,
                           vector<RecordAttribute *> whichOIDs,
                           vector<materialization_mode> outputMode)
    : wantedExpressions(wantedExpressions),
      wantedOIDs(whichOIDs),
      wantedFields(whichFields),
      outputMode(outputMode) {
  oidsProvided = true;
}

Materializer::Materializer(vector<expression_t> wantedExpressions) {
  oidsProvided = true;
  vector<expression_t>::const_iterator it = wantedExpressions.begin();
  for (; it != wantedExpressions.end(); it++) {
    auto proj = dynamic_cast<const expressions::RecordProjection *>(
        it->getUnderlyingExpression());
    if (proj != NULL) {
      RecordAttribute *recAttr = new RecordAttribute(proj->getAttribute());
      if (recAttr->getAttrName() == activeLoop) {
        wantedOIDs.push_back(recAttr);
      } else {
        wantedFields.push_back(recAttr);
      }
    } else {
      string error_msg = "[Materializer: ] Unexpected value to materialize";
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
    outputMode.push_back(EAGER);
  }
}

OutputPlugin::OutputPlugin(RawContext *const context,
                           Materializer &materializer,
                           const map<RecordAttribute, RawValueMemory> *bindings)
    : context(context), materializer(materializer), currentBindings(bindings) {
  /**
   * TODO
   * PAYLOAD SIZE IS NOT A DECISION TO BE MADE BY THE PLUGIN
   * THE OUTPUT PLUGIN SHOULD BE THE ONE ENFORCING THE DECISION
   */
  const vector<RecordAttribute *> &wantedFields =
      materializer.getWantedFields();
  // Declare our result (value) type
  materializedTypes = new vector<Type *>();
  int payload_type_size = 0;
  identifiersTypeSize = 0;
  int tupleIdentifiers = 0;
  isComplex = false;
  // XXX Always materializing the 'active tuples' pointer
  {
    // FIXME use 'find' instead of looping
    for (const auto &oid : materializer.getWantedOIDs()) {
      // RecordAttribute currAttr = memSearch->first;
      //            cout << "HINT: " << currAttr.getOriginalRelationName() << "
      //            -- "
      //                    << currAttr.getRelationName() << "_" <<
      //                    currAttr.getName()
      //                    << endl;
      // if (currAttr.getAttrName() == activeLoop) {
      cout << "HINT - OID MAT'D: " << oid->getOriginalRelationName() << " -- "
           << oid->getRelationName() << "_" << oid->getName() << endl;
      Type *currType = oid->getOriginalType()->getLLVMType(
          context
              ->getLLVMContext());  // currentBindings.find(*oid)->second.mem->getAllocatedType();
      currType->dump();
      materializedTypes->push_back(currType);
      payload_type_size += (currType->getPrimitiveSizeInBits() / 8);
      // cout << "Active Tuple Size "<< (currType->getPrimitiveSizeInBits() / 8)
      // << endl;
      tupleIdentifiers++;
      materializer.addTupleIdentifier(*oid);
      // }
    }
  }
  identifiersTypeSize = payload_type_size;
  int attrNo = 0;
  /**
   * XXX
   * What if fields not materialized (yet)? This will crash w/o cause
   * Atm, that's always the case when dealing with e.g. JSON that is lazily
   * materialized
   */
  for (vector<RecordAttribute *>::const_iterator it = wantedFields.begin();
       it != wantedFields.end(); it++) {
    // map<RecordAttribute, RawValueMemory>::const_iterator itSearch =
    // currentBindings.find(*(*it));

    CachingService &cache = CachingService::getInstance();
    /* expr does not participate in caching search, so don't need it explicitly
     * => mock */
    list<RecordAttribute *> mockAtts = list<RecordAttribute *>();
    mockAtts.push_back(*it);
    list<RecordAttribute> mockProjections;
    RecordType mockRec = RecordType(mockAtts);
    expressions::InputArgument mockExpr{&mockRec, 0, mockProjections};
    expressions::RecordProjection e = expression_t{mockExpr}[*(*it)];
    CacheInfo info = cache.getCache(&e);
    bool isCached = false;
    if (info.structFieldNo != -1) {
      if (!cache.getCacheIsFull(&e)) {
      } else {
        isCached = true;
        cout << "[OUTPUT PG: ] *Cached* Expression found for "
             << e.getOriginalRelationName() << "."
             << e.getAttribute().getAttrName() << "!" << endl;
      }
    }

    // Field needed
    if (isCached == true) {
      LOG(INFO) << "[MATERIALIZER: ] *CACHED* PART OF PAYLOAD: "
                << (*it)->getAttrName();
      materialization_mode mode = (materializer.getOutputMode()).at(attrNo++);
      // gather datatypes: caches can only have int32 or float!!!
      Type *requestedType = (*it)->getLLVMType(context->getLLVMContext());
      isComplex = false;
      materializedTypes->push_back(requestedType);
      int fieldSize = requestedType->getPrimitiveSizeInBits() / 8;
      fieldSizes.push_back(fieldSize);
      payload_type_size += fieldSize;
    } else {
      LOG(INFO) << "[MATERIALIZER: ] PART OF PAYLOAD: " << (*it)->getAttrName();
      cout << "[MATERIALIZER: ] PART OF PAYLOAD: " << (*it)->getAttrName()
           << endl;

      materialization_mode mode = (materializer.getOutputMode()).at(attrNo++);
      // gather datatypes
      Type *currType;
      // if (itSearch != bindings.end()){
      //     currType =
      //     itSearch->second.mem->getType()->getPointerElementType();//(*it)->getLLVMType(context->getLLVMContext());//->getAllocatedType();
      // } else {
      currType = (*it)->getLLVMType(context->getLLVMContext());
      // }
      Type *requestedType =
          chooseType((*it)->getOriginalType(), currType, mode);
      requestedType->dump();
      materializedTypes->push_back(requestedType);
      int fieldSize = requestedType->getPrimitiveSizeInBits() / 8;
      fieldSizes.push_back(fieldSize);
      payload_type_size += fieldSize;
      // cout << "Field Size "<< fieldSize << endl;
      typeID id = (*it)->getOriginalType()->getTypeID();
      switch (id) {
        case BOOL:
        case INT:
        case FLOAT:
          isComplex = false;
          break;
        default:
          isComplex = true;
      }
    }
    // else {
    //     string error_msg = string("[OUTPUT PG: ] INPUT ERROR AT OPERATOR -
    //     UNKNOWN WANTED FIELD ") + (*it)->getAttrName(); map<RecordAttribute,
    //     RawValueMemory>::const_iterator memSearch;

    //     for (memSearch = currentBindings.begin();
    //             memSearch != currentBindings.end(); memSearch++) {

    //         RecordAttribute currAttr = memSearch->first;
    //         cout << "HINT: " << currAttr.getOriginalRelationName() << " -- "
    //         << currAttr.getRelationName() << "_" << currAttr.getName()
    //         << endl;
    //         if (currAttr.getAttrName() == activeLoop) {
    //             Type* currType = (memSearch->second).mem->getAllocatedType();
    //             materializedTypes->push_back(currType);
    //             payload_type_size += (currType->getPrimitiveSizeInBits() /
    //             8);
    //             //cout << "Active Tuple Size "<<
    //             (currType->getPrimitiveSizeInBits() / 8) << endl;
    //             tupleIdentifiers++;
    //             materializer.addTupleIdentifier(currAttr);
    //         }
    //     }
    //     LOG(ERROR) << error_msg;
    //     throw runtime_error(error_msg);
    // }
  }
  LOG(INFO) << "[OUTPUT PLUGIN: ] Size of tuple (payload): "
            << payload_type_size;

  // Result type specified
  payloadType =
      llvm::StructType::get(context->getLLVMContext(), *materializedTypes);
  payloadTypeSize = payload_type_size;
  //    cout << "Payload Size "<< payloadTypeSize << endl;
}

Value *OutputPlugin::getRuntimePayloadTypeSize() {
  RawCatalog &catalog = RawCatalog::getInstance();
  Value *val_size = context->createInt32(0);
  IRBuilder<> *Builder = context->getBuilder();
  RawValueMemory mem_activeTuple;

  // Pre-computed 'activeLoop' variables
  val_size = context->createInt32(identifiersTypeSize);

  int attrNo = 0;
  const vector<RecordAttribute *> &wantedFields =
      materializer.getWantedFields();
  vector<RecordAttribute *>::const_iterator it = wantedFields.begin();
  for (; it != wantedFields.end(); it++) {
    map<RecordAttribute, RawValueMemory>::const_iterator itSearch =
        currentBindings->find(*(*it));
    // Field needed
    if (itSearch != currentBindings->end()) {
      materialization_mode mode = (materializer.getOutputMode()).at(attrNo);
      Value *val_attr_size = NULL;
      if (mode == EAGER) {
        RecordAttribute currAttr = itSearch->first;
        Plugin *inputPg = catalog.getPlugin(currAttr.getOriginalRelationName());
        val_attr_size =
            inputPg->getValueSize(itSearch->second, currAttr.getOriginalType());
        val_size = Builder->CreateAdd(val_size, val_attr_size);
      } else {
        // Pre-computed
        val_attr_size = context->createInt32(fieldSizes.at(attrNo));
      }
      val_size = Builder->CreateAdd(val_size, val_attr_size);
      attrNo++;
    } else {
      string error_msg =
          string("[OUTPUT PG: ] UNKNOWN WANTED FIELD ") + (*it)->getAttrName();
      LOG(ERROR) << error_msg;
      throw runtime_error(error_msg);
    }
  }
  return val_size;
}

// TODO Many more cases to cater for
// Current most relevant example: JSONObject
Type *OutputPlugin::chooseType(const ExpressionType *exprType, Type *currType,
                               materialization_mode mode) {
  switch (exprType->getTypeID()) {
    case BAG:
    case SET:
    case LIST:
      LOG(ERROR)
          << "[OUTPUT PG: ] DEALING WITH COLLECTION TYPE - NOT SUPPORTED YET";
      throw runtime_error(string(
          "[OUTPUT PG: ] DEALING WITH COLLECTION TYPE - NOT SUPPORTED YET"));
      break;
    case RECORD:  // I am not sure if this is case can occur
      LOG(ERROR)
          << "[OUTPUT PG: ] DEALING WITH RECORD TYPE - NOT SUPPORTED YET";
      throw runtime_error(
          string("[OUTPUT PG: ] DEALING WITH RECORD TYPE - NOT SUPPORTED YET"));
    case BOOL:
    case DSTRING:
    case STRING:
    case DATE:
    case FLOAT:
    case INT64:
    case INT:
      switch (mode) {
        case EAGER:
          return currType;
        case LAZY:
        case INTERMEDIATE:
          LOG(ERROR)
              << "[OUTPUT PG: ] DEALING WITH PRIMITIVE TYPE - ONLY 'EAGER' "
                 "CONVERSION AVAILABLE";
          throw runtime_error(
              string("[OUTPUT PG: ] DEALING WITH PRIMITIVE TYPE - "
                     "ONLY 'EAGER' CONVERSION AVAILABLE"));
        case BINARY:
          LOG(WARNING)
              << "[OUTPUT PG: ] DEALING WITH PRIMITIVE TYPE - 'BINARY' "
                 "MODE ALREADY ACTIVE";
          break;
        case CSV:
        case JSON:
          LOG(ERROR)
              << "[OUTPUT PG: ] CONVERSION TO 'RAW' FORM NOT SUPPORTED (YET)";
          throw runtime_error(string(
              "[OUTPUT PG: ] CONVERSION TO 'RAW' FORM NOT SUPPORTED (YET)"));
      }
      break;
    default: {
      string error_msg = "[OUTPUT PG: ] TYPE - NOT SUPPORTED YET: ";
      LOG(ERROR) << error_msg << exprType->getTypeID();
      throw runtime_error(error_msg);
    }
  }

  return currType;
}

// TODO Must add functionality - currently only thin shell exists
Value *OutputPlugin::convert(Type *currType, Type *materializedType,
                             Value *val) {
  if (currType == materializedType) {
    return val;
  }
  // TODO Many more cases to cater for here
  return val;
}
