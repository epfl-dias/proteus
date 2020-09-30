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

#ifndef BINARYOPERATORS_HPP_
#define BINARYOPERATORS_HPP_

namespace expressions {

class BinaryOperator {
 public:
  enum opID {
    EQ,
    NEQ,
    GE,
    GT,
    LE,
    LT,
    ADD,
    SUB,
    MULT,
    DIV,
    AND,
    OR,
    MAX,
    MIN,
    MOD,
    SHL,
    LSHR,
    ASHR,
    XOR,
  };
  virtual opID getID() = 0;
  virtual ~BinaryOperator();
};

class Eq : public BinaryOperator {
  opID getID() override { return EQ; }
};
class Neq : public BinaryOperator {
  opID getID() override { return NEQ; }
};
class Ge : public BinaryOperator {
  opID getID() override { return GE; }
};
class Gt : public BinaryOperator {
  opID getID() override { return GT; }
};
class Le : public BinaryOperator {
  opID getID() override { return LE; }
};
class Lt : public BinaryOperator {
  opID getID() override { return LT; }
};
class Add : public BinaryOperator {
  opID getID() override { return ADD; }
};
class Sub : public BinaryOperator {
  opID getID() override { return SUB; }
};
class Mult : public BinaryOperator {
  opID getID() override { return MULT; }
};
class Div : public BinaryOperator {
  opID getID() override { return DIV; }
};
class Mod : public BinaryOperator {
  opID getID() override { return MOD; }
};
class And : public BinaryOperator {
  opID getID() override { return AND; }
};
class Or : public BinaryOperator {
  opID getID() override { return OR; }
};
class Max : public BinaryOperator {
  opID getID() override { return MAX; }
};
class Min : public BinaryOperator {
  opID getID() override { return MIN; }
};
class Shl : public BinaryOperator {
  opID getID() override { return SHL; }
};
class Lshr : public BinaryOperator {
  opID getID() override { return LSHR; }
};
class Ashr : public BinaryOperator {
  opID getID() override { return ASHR; }
};
class Xor : public BinaryOperator {
  opID getID() override { return XOR; }
};
}  // namespace expressions

#endif /* BINARYOPERATORS_HPP_ */
