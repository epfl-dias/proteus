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

/*
 * atois.hpp
 *
 *  Created on: Apr 1, 2015
 *      Author: manolee
 */

#ifndef ATOIS_HPP_
#define ATOIS_HPP_

#include "common/common.hpp"
#include "util/context.hpp"

//#define DEBUGATOIS

void atois(llvm::Value *buf, llvm::Value *len, llvm::AllocaInst *mem_result,
           Context *context);
void atoi1(llvm::Value *buf, llvm::AllocaInst *mem_result, Context *context);
void atoi2(llvm::Value *buf, llvm::AllocaInst *mem_result, Context *context);
void atoi3(llvm::Value *buf, llvm::AllocaInst *mem_result, Context *context);
void atoi4(llvm::Value *buf, llvm::AllocaInst *mem_result, Context *context);
void atoi5(llvm::Value *buf, llvm::AllocaInst *mem_result, Context *context);
void atoi6(llvm::Value *buf, llvm::AllocaInst *mem_result, Context *context);
void atoi7(llvm::Value *buf, llvm::AllocaInst *mem_result, Context *context);
void atoi8(llvm::Value *buf, llvm::AllocaInst *mem_result, Context *context);
void atoi9(llvm::Value *buf, llvm::AllocaInst *mem_result, Context *context);
void atoi10(llvm::Value *buf, llvm::AllocaInst *mem_result, Context *context);

#endif /* ATOIS_HPP_ */
