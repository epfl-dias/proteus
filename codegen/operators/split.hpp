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
#ifndef SPLIT_HPP_
#define SPLIT_HPP_

#include "operators/exchange.hpp"

class Split : public Exchange {
public:
    Split(  RawOperator  * const            child                   ,
            GpuRawContext * const           context                 ,
            int                             numOfParents            ,
            const vector<RecordAttribute*> &wantedFields            ,
            int                             slack                   ,
            expressions::Expression        *hash            = NULL  ,
            bool                            numa_local      = true  ,
            bool                            rand_local_cpu  = false ) :
                    Exchange            (
                                            child,
                                            context,
                                            numOfParents,
                                            wantedFields,
                                            slack,
                                            hash,
                                            numa_local,
                                            rand_local_cpu,
                                            1
                                        ),
                    produce_calls       (0){
        assert((!hash || !numa_local) && "Just to make it more clear that hash has precedence over numa_local");
    }

    virtual ~Split(){ LOG(INFO)<<"Collapsing Split operator";}

    virtual void produce();

    virtual void setParent(RawOperator * parent){
        UnaryRawOperator::setParent(parent);

        this->parent.emplace_back  (parent);
    }

protected:
    void open (RawPipeline * pip);

private:
    size_t                          produce_calls;
    std::vector<RawPipelineGen *>   catch_pip;
    std::vector<RawOperator    *>   parent;
};

#endif /* SPLIT_HPP_ */


