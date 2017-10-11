/*
    RAW -- High-performance querying over raw, never-seen-before data.

                            Copyright (c) 2017
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

#ifndef GPU_INTRINSICS_HPP_
#define GPU_INTRINSICS_HPP_

#include "util/gpu/gpu-raw-context.hpp"

namespace gpu_intrinsic{

/**
 * @brief Creates a vote.all(i1) call
 *
 * Generates the call to vote.all with @p val_in as the parameter.
 *
 * @return a Value pointing to the result of the call
 **/
Value * all(GpuRawContext * const context, Value * val_in);

/**
 * @brief Creates a vote.any(i1) call
 *
 * Generates the call to vote.any with @p val_in as the parameter.
 *
 * @return a Value pointing to the result of the call
 **/
Value * any(GpuRawContext * const context, Value * val_in);


Value * shfl_bfly(GpuRawContext * const context, 
                    Value *             val_in, 
                    uint32_t            vxor, 
                    Value *             mask = NULL);

Value * shfl_bfly(GpuRawContext * const context, 
                    Value *             val_in, 
                    Value *             vxor, 
                    Value *             mask = NULL);

Value * ballot(GpuRawContext * const context, Value * val_in);

}

#endif /* GPU_INTRINSICS_HPP_ */