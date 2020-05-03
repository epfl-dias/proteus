/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2020
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

#include <operators/unionall.hpp>
#include <util/flush-operator-tree.hpp>

#include "demangle.hpp"

class [[nodiscard]] spacer {
 public:
  const Operator &v;
  size_t space;

 private:
  spacer(const Operator &v, size_t space) : v(v), space(space) {}

 public:
  explicit spacer(const Operator &v) : spacer(v, 0) {}

  [[nodiscard]] spacer step(const Operator &child, size_t indent = 2) const {
    return {child, space + indent};
  }
};

std::ostream &operator<<(std::ostream &out, const spacer &s) {
  for (size_t i = 0; i < s.space; ++i) out << ' ';
  out << demangle(typeid(s.v).name());
  out << '(' << s.v.getRowType() << ")\n";
  if (auto u = dynamic_cast<const UnionAll *>(&s.v)) {
    for (const auto c : u->getChildren()) out << s.step(*c);
  } else if (auto c = dynamic_cast<const UnaryOperator *>(&s.v)) {
    out << s.step(*(c->getChild()));
  } else if (auto b = dynamic_cast<const BinaryOperator *>(&s.v)) {
    out << s.step(*(b->getLeftChild()));
    out << s.step(*(b->getRightChild()));
  }
  return out;
}

std::ostream &operator<<(std::ostream &out, const Operator &op) {
  out << spacer{op};
  return out;
}
