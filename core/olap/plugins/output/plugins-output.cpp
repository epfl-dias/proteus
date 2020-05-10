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

#include "plugins/output/plugins-output.hpp"

#include "expressions/expressions-generator.hpp"
#include "util/context.hpp"

Materializer::Materializer(
    std::vector<RecordAttribute *> wantedFields,
    //        const std::vector<expression_t>& wantedExpressions,
    std::vector<materialization_mode> outputMode)
    :  //        wantedExpressions(wantedExpressions),
      wantedFields(wantedFields),
      outputMode(outputMode) {
  oidsProvided = false;
}

Materializer::Materializer(std::vector<RecordAttribute *> whichFields,
                           std::vector<expression_t> wantedExpressions,
                           std::vector<RecordAttribute *> whichOIDs,
                           std::vector<materialization_mode> outputMode)
    : wantedExpressions(wantedExpressions),
      wantedOIDs(whichOIDs),
      wantedFields(whichFields),
      outputMode(outputMode) {
  oidsProvided = true;
}

Materializer::Materializer(const std::vector<expression_t> &wantedExpressions)
    : wantedExpressions(wantedExpressions) {
  oidsProvided = true;
  for (const auto &e : wantedExpressions) {
    auto recAttr = e.getRegisteredAs();
    if (recAttr.getAttrName() == activeLoop) {
      wantedOIDs.push_back(new RecordAttribute(recAttr));
    } else {
      wantedFields.push_back(new RecordAttribute(recAttr));
    }
    outputMode.push_back(EAGER);
  }
}

OutputPlugin::OutputPlugin(
    Context *const context, Materializer &materializer,
    const map<RecordAttribute, ProteusValueMemory> *bindings)
    : context(context), materializer(materializer), currentBindings(bindings) {
  /**
   * TODO
   * PAYLOAD SIZE IS NOT A DECISION TO BE MADE BY THE PLUGIN
   * THE OUTPUT PLUGIN SHOULD BE THE ONE ENFORCING THE DECISION
   */
  const std::vector<RecordAttribute *> &wantedFields =
      materializer.getWantedFields();
  // Declare our result (value) type
  materializedTypes = new std::vector<llvm::Type *>();
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
      llvm::Type *currType = oid->getOriginalType()->getLLVMType(
          context
              ->getLLVMContext());  // currentBindings.find(*oid)->second.mem->getAllocatedType();
      // currType->dump();
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
  for (std::vector<RecordAttribute *>::const_iterator it = wantedFields.begin();
       it != wantedFields.end(); it++) {
    // map<RecordAttribute, ProteusValueMemory>::const_iterator itSearch =
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
      llvm::Type *requestedType = (*it)->getLLVMType(context->getLLVMContext());
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
      llvm::Type *currType;
      // if (itSearch != bindings.end()){
      //     currType =
      //     itSearch->second.mem->getType()->getPointerElementType();//(*it)->getLLVMType(context->getLLVMContext());//->getAllocatedType();
      // } else {
      currType = (*it)->getLLVMType(context->getLLVMContext());
      // }
      llvm::Type *requestedType =
          chooseType((*it)->getOriginalType(), currType, mode);
      // requestedType->dump();
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
    //     ProteusValueMemory>::const_iterator memSearch;

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

llvm::Value *OutputPlugin::getRuntimePayloadTypeSize() {
  Catalog &catalog = Catalog::getInstance();
  llvm::Value *val_size = context->createInt32(0);
  auto Builder = context->getBuilder();

  // Pre-computed 'activeLoop' variables
  val_size = context->createInt32(identifiersTypeSize);

  int attrNo = 0;
  const std::vector<RecordAttribute *> &wantedFields =
      materializer.getWantedFields();
  std::vector<RecordAttribute *>::const_iterator it = wantedFields.begin();
  for (; it != wantedFields.end(); it++) {
    map<RecordAttribute, ProteusValueMemory>::const_iterator itSearch =
        currentBindings->find(*(*it));
    // Field needed
    if (itSearch != currentBindings->end()) {
      materialization_mode mode = (materializer.getOutputMode()).at(attrNo);
      llvm::Value *val_attr_size = nullptr;
      if (mode == EAGER) {
        RecordAttribute currAttr = itSearch->first;
        Plugin *inputPg = catalog.getPlugin(currAttr.getOriginalRelationName());
        assert(dynamic_cast<ParallelContext *>(context));
        val_attr_size =
            inputPg->getValueSize(itSearch->second, currAttr.getOriginalType(),
                                  dynamic_cast<ParallelContext *>(context));
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
llvm::Type *OutputPlugin::chooseType(const ExpressionType *exprType,
                                     llvm::Type *currType,
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
llvm::Value *OutputPlugin::convert(llvm::Type *currType,
                                   llvm::Type *materializedType,
                                   llvm::Value *val) {
  if (currType == materializedType) {
    return val;
  }
  // TODO Many more cases to cater for here
  return val;
}
