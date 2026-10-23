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

#include <lib/operators/scan.hpp>
#include <lib/operators/unionall.hpp>
#include <lib/util/flush-operator-tree.hpp>
#include <platform/util/demangle.hpp>

class [[nodiscard]] spacer {
 public:
  const Operator &op;
  size_t space;

 private:
  spacer(const Operator &op, size_t space) : op(op), space(space) {}

 public:
  explicit spacer(const Operator &op) : spacer(op, 0) {}

  [[nodiscard]] spacer step(const Operator &child, size_t indent = 2) const {
    return {child, space + indent};
  }
};

std::ostream &operator<<(std::ostream &out, const spacer &s) {
  for (size_t i = 0; i < s.space; ++i) out << ' ';
  out << demangle(typeid(s.op).name());
  out << '(' << s.op.getRowType() << ") [" << s.op.getUUID() << "]\n";
  if (auto u = dynamic_cast<const UnionAll *>(&s.op)) {
    for (const auto c : u->getChildren()) out << s.step(*c);
  } else if (dynamic_cast<const Scan *>(&s.op)) {
  } else if (auto c = dynamic_cast<const UnaryOperator *>(&s.op)) {
    out << s.step(*(c->getChild()));
  } else if (auto b = dynamic_cast<const BinaryOperator *>(&s.op)) {
    out << s.step(*(b->getLeftChild()));
    out << s.step(*(b->getRightChild()));
  }
  return out;
}

std::ostream &operator<<(std::ostream &out, const Operator &op) {
  out << spacer{op};
  return out;
}
