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

#ifndef BINARYOPERATORS_HPP_
#define BINARYOPERATORS_HPP_

namespace expressions	{

class BinaryOperator	{
public:
	virtual ~BinaryOperator() = 0;
};

class Eq   : public BinaryOperator {};
class Neq  : public BinaryOperator {};
class Ge   : public BinaryOperator {};
class Gt   : public BinaryOperator {};
class Le   : public BinaryOperator {};
class Lt   : public BinaryOperator {};
class Add  : public BinaryOperator {};
class Sub  : public BinaryOperator {};
class Mult : public BinaryOperator {};
class Div  : public BinaryOperator {};
class And  : public BinaryOperator {};
class Or  : public BinaryOperator {};
}

#endif /* BINARYOPERATORS_HPP_ */
