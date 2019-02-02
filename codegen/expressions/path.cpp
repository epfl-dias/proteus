/*
    Proteus -- High-performance query processing on heterogeneous hardware.

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

#include "expressions/path.hpp"

string Path::toString() const {
  stringstream ss;
  ss << desugarizedPath->getRelationName();
  expression_t currExpr = desugarizedPath;
  list<string> projNames = list<string>();
  while (currExpr.getTypeID() == expressions::RECORD_PROJECTION) {
    auto currProj = dynamic_cast<const expressions::RecordProjection *>(
        currExpr.getUnderlyingExpression());
    assert(currProj);
    projNames.insert(projNames.begin(), currProj->getProjectionName());
    currExpr = currProj->getExpr();
  }

  for (const auto &str : projNames) {
    ss << ".";
    ss << str;
  }

  LOG(INFO) << "[Unnest: ] path.toString = " << ss.str();
  return ss.str();
}
