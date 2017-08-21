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

#include "util/raw-pipeline.hpp"

class GpuRawContext: public RawContext {
public:

    GpuRawContext(const string& moduleName);
    ~GpuRawContext();

    virtual size_t appendParameter(llvm::Type * ptype, bool noalias  = false, bool readonly = false);
    virtual size_t appendStateVar (llvm::Type * ptype);

    virtual llvm::Argument * getArgument(size_t id) const;
    virtual llvm::Value    * getStateVar(size_t id) const;

    void registerOpen (std::function<void (RawPipeline * pip)> open );
    void registerClose(std::function<void (RawPipeline * pip)> close);

    void pushNewPipeline();
    void popNewPipeline();

    virtual void setGlobalFunction(Function *F = nullptr);
    virtual void prepareFunction(Function *F);

    virtual llvm::Value * threadId ();
    virtual llvm::Value * threadNum();
    virtual llvm::Value * laneId   ();
    virtual void          createMembar_gl();

    string emitPTX();

    void compileAndLoad();

    std::vector<CUfunction> getKernel();
    std::vector<RawPipeline *> getPipelines();

protected:
    virtual void createJITEngine();

    std::unique_ptr<TargetMachine> TheTargetMachine;

    CUmodule cudaModule;

    string                  kernelName;
    std::vector<RawPipelineGen> pipelines ;

    std::vector<RawPipelineGen> generators;
};

#endif /* GPU_RAW_CONTEXT_HPP_ */
