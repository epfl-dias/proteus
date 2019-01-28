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

/*
 * atois.hpp
 *
 *  Created on: Apr 1, 2015
 *      Author: manolee
 */

#ifndef ATOIS_HPP_
#define ATOIS_HPP_

#include "common/common.hpp"
#include "util/raw-context.hpp"

//#define DEBUGATOIS

void atois(Value *buf, Value *len, AllocaInst *mem_result,
           RawContext *const context);
void atoi1(Value *buf, AllocaInst *mem_result, RawContext *const context);
void atoi2(Value *buf, AllocaInst *mem_result, RawContext *const context);
void atoi3(Value *buf, AllocaInst *mem_result, RawContext *const context);
void atoi4(Value *buf, AllocaInst *mem_result, RawContext *const context);
void atoi5(Value *buf, AllocaInst *mem_result, RawContext *const context);
void atoi6(Value *buf, AllocaInst *mem_result, RawContext *const context);
void atoi7(Value *buf, AllocaInst *mem_result, RawContext *const context);
void atoi8(Value *buf, AllocaInst *mem_result, RawContext *const context);
void atoi9(Value *buf, AllocaInst *mem_result, RawContext *const context);
void atoi10(Value *buf, AllocaInst *mem_result, RawContext *const context);

#endif /* ATOIS_HPP_ */
