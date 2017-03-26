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

#ifndef GPU_RAW_CONTEXT_HPP_
#define GPU_RAW_CONTEXT_HPP_

#include "util/raw-context.hpp"
#include "cuda.h"
#include "cuda_runtime_api.h"

class GpuRawContext: public RawContext {
public:

    GpuRawContext(const string& moduleName);
    ~GpuRawContext();

    virtual void setGlobalFunction(Function *F);
    virtual void prepareFunction(Function *F);

    virtual Value * threadId ();
    virtual Value * threadNum();
    virtual Value * laneId   ();

    string emitPTX();

    void compileAndLoad();
    CUfunction getKernel(std::string kernelName);

protected:
    virtual void createJITEngine();

    std::unique_ptr<TargetMachine> TheTargetMachine;

    CUmodule cudaModule;
};

#endif /* GPU_RAW_CONTEXT_HPP_ */
