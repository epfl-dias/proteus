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

#ifndef PATH_HPP_
#define PATH_HPP_

#include "common/common.hpp"
#include "expressions/expressions.hpp"
#include "plugins/plugins.hpp"
#include "util/catalog.hpp"

class Path {
 public:
  Path(string nestedName, const expressions::RecordProjection *desugarizedPath)
      : desugarizedPath(desugarizedPath), nestedName(nestedName) {
    assert(desugarizedPath && "Projection should be non-null");
    Catalog &catalog = Catalog::getInstance();
    string originalRelation = desugarizedPath->getOriginalRelationName();
    pg = catalog.getPlugin(originalRelation);
  }

  const expressions::RecordProjection *get() const { return desugarizedPath; }
  Plugin *getRelevantPlugin() const { return pg; }
  string getNestedName() const { return nestedName; }
  string toString() const;

 private:
  const expressions::RecordProjection *const desugarizedPath;
  string nestedName;
  Plugin *pg;
};

#endif /* PATH_HPP_ */
