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

    virtual int appendParameter(llvm::Type * ptype, bool noalias = false, bool readonly = false);
    virtual Argument * getArgument(int id) const;

    virtual void setGlobalFunction(Function *F = nullptr);
    virtual void prepareFunction(Function *F);

    virtual llvm::Value * threadId ();
    virtual llvm::Value * threadNum();
    virtual llvm::Value * laneId   ();

    string emitPTX();

    void compileAndLoad();
    CUfunction getKernel();

protected:
    virtual void createJITEngine();

    std::unique_ptr<TargetMachine> TheTargetMachine;

    CUmodule cudaModule;

    std::vector<llvm::Type *> inputs;
    std::vector<bool     > inputs_noalias;
    std::vector<bool     > inputs_readonly;

    std::vector<Argument *> args;

    string               kernelName;
};

#endif /* GPU_RAW_CONTEXT_HPP_ */
