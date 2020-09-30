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

#ifndef GPU_INTRINSICS_HPP_
#define GPU_INTRINSICS_HPP_

#include "olap/util/parallel-context.hpp"

namespace gpu_intrinsic {

llvm::Value *activemask(ParallelContext *context);

llvm::Value *load_ca(ParallelContext *context, llvm::Value *address);

llvm::Value *load_ca16(ParallelContext *const context, llvm::Value *address);

llvm::Value *load_cs(ParallelContext *const context, llvm::Value *address);

void store_wb32(ParallelContext *const context, llvm::Value *address,
                llvm::Value *value);

void store_wb16(ParallelContext *const context, llvm::Value *address,
                llvm::Value *value);

/**
 * @brief Creates a vote.all(i1) call
 *
 * Generates the call to vote.all with @p val_in as the parameter.
 *
 * @return a Value pointing to the result of the call
 **/
llvm::Value *all(ParallelContext *const context, llvm::Value *val_in);

/**
 * @brief Creates a vote.any(i1) call
 *
 * Generates the call to vote.any with @p val_in as the parameter.
 *
 * @return a Value pointing to the result of the call
 **/
llvm::Value *any(ParallelContext *const context, llvm::Value *val_in);

llvm::Value *shfl_bfly(ParallelContext *const context, llvm::Value *val_in,
                       uint32_t vxor, llvm::Value *mask = nullptr);

llvm::Value *shfl_bfly(ParallelContext *const context, llvm::Value *val_in,
                       llvm::Value *vxor, llvm::Value *mask = nullptr);

llvm::Value *ballot(ParallelContext *const context, llvm::Value *val_in);

}  // namespace gpu_intrinsic

#endif /* GPU_INTRINSICS_HPP_ */
