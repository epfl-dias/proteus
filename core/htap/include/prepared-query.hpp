/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
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

#ifndef HARMONIA_PREPARED_QUERY_HPP_
#define HARMONIA_PREPARED_QUERY_HPP_

#include "plan/prepared-statement.hpp"

// std::vector<std::string relName, std::vector<std::string> relAttrs>
typedef std::vector<std::pair<std::string, std::vector<std::string>>>
    query_rel_t;

class PreparedQuery : public PreparedStatement {
 public:
  query_rel_t query_relations;

  PreparedQuery(PreparedStatement q) : PreparedStatement(std::move(q)) {}

  PreparedQuery(PreparedStatement q, query_rel_t relations)
      : PreparedStatement(std::move(q)), query_relations(relations) {}

  void setQueryRelations(query_rel_t r) { query_relations = std::move(r); }

  query_rel_t &getQueryRelations() { return query_relations; }
};

#endif /* HARMONIA_PREPARED_QUERY_HPP_ */
