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

#ifndef RAW_CPU_PIPELINE_HPP_
#define RAW_CPU_PIPELINE_HPP_

#include "util/raw-pipeline.hpp"
#include "util/jit/raw-cpu-module.hpp"

class RawCpuPipelineGen: public RawPipelineGen {
protected:
    RawCpuModule                                        module;

private:
    RawCpuPipelineGen(  RawContext        * context                 , 
                        std::string         pipName         = "pip" , 
                        RawPipelineGen    * copyStateFrom   = NULL  );

    friend class RawCpuPipelineGenFactory;
public:
    virtual void compileAndLoad();

    virtual Module * getModule () const {return module.getModule();}

public:
    virtual void * getCompiledFunction(Function * f);
};

class RawCpuPipelineGenFactory: public RawPipelineGenFactory {
protected:
    RawCpuPipelineGenFactory(){}
public:
    static RawPipelineGenFactory &getInstance(){
        static RawCpuPipelineGenFactory instance;
        return instance;
    }


    RawPipelineGen * create(RawContext * context, std::string pipName, RawPipelineGen * copyStateFrom){
        return new RawCpuPipelineGen(context, pipName, copyStateFrom);
    }
};

#endif /* RAW_CPU_PIPELINE_HPP_ */