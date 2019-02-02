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

#ifndef BINARYOPERATORS_HPP_
#define BINARYOPERATORS_HPP_

namespace expressions {

class BinaryOperator {
 public:
  enum opID { EQ, NEQ, GE, GT, LE, LT, ADD, SUB, MULT, DIV, AND, OR, MAX, MIN };
  virtual opID getID() = 0;
  virtual ~BinaryOperator() = 0;
};

class Eq : public BinaryOperator {
  opID getID() { return EQ; }
};
class Neq : public BinaryOperator {
  opID getID() { return NEQ; }
};
class Ge : public BinaryOperator {
  opID getID() { return GE; }
};
class Gt : public BinaryOperator {
  opID getID() { return GT; }
};
class Le : public BinaryOperator {
  opID getID() { return LE; }
};
class Lt : public BinaryOperator {
  opID getID() { return LT; }
};
class Add : public BinaryOperator {
  opID getID() { return ADD; }
};
class Sub : public BinaryOperator {
  opID getID() { return SUB; }
};
class Mult : public BinaryOperator {
  opID getID() { return MULT; }
};
class Div : public BinaryOperator {
  opID getID() { return DIV; }
};
class And : public BinaryOperator {
  opID getID() { return AND; }
};
class Or : public BinaryOperator {
  opID getID() { return OR; }
};
class Max : public BinaryOperator {
  opID getID() { return MAX; }
};
class Min : public BinaryOperator {
  opID getID() { return MIN; }
};
}  // namespace expressions

#endif /* BINARYOPERATORS_HPP_ */
