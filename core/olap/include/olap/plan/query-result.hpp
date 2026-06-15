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

#ifndef QUERY_RESULT_HPP_
#define QUERY_RESULT_HPP_

#include <iostream>
#include <span>
#include <string>

class QueryResult {
 private:
  size_t fsize;
  char *resultBuf;

 public:
  const std::string outputfile;
  /**
   * @return a span of bytes representing the result. The span is valid until
   the QueryResult is destroyed.
   */
  std::span<const char> asSpan() const {
    return std::span<const char>(resultBuf, fsize);
  }
  explicit QueryResult(const std::string &query_name);
  QueryResult(const QueryResult &) = delete;
  QueryResult(QueryResult &&) noexcept;
  QueryResult &operator=(const QueryResult &) = delete;
  QueryResult &operator=(QueryResult &&) = delete;
  ~QueryResult();

  friend std::ostream &operator<<(std::ostream &out, const QueryResult &qr);
};

std::ostream &operator<<(std::ostream &out, const QueryResult &qr);

#endif /* QUERY_RESULT_HPP_ */
