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
#ifndef HASH_REARRANGE_HPP_
#define HASH_REARRANGE_HPP_

#include "operators/operators.hpp"
#include "util/gpu/gpu-raw-context.hpp"

class HashRearrange : public UnaryRawOperator {
public:
    HashRearrange(  RawOperator * const             child,
                    GpuRawContext * const           context,
                    int                             numOfBuckets,
                    const vector<RecordAttribute*> &wantedFields,
                    expressions::Expression        *hashExpr,
                    RecordAttribute                *hashProject = NULL) :
                        UnaryRawOperator(child), 
                        context(context), 
                        numOfBuckets(numOfBuckets),
                        wantedFields(wantedFields),
                        hashExpr(hashExpr),
                        hashProject(hashProject),
                        blockSize(h_vector_size){
    }//FIMXE: default blocksize...

    virtual ~HashRearrange()                                             { LOG(INFO)<<"Collapsing HashRearrange operator";}

    virtual void produce();
    virtual void consume(RawContext* const context, const OperatorState& childState);
    virtual bool isFiltering() const {return false;}

protected:
    virtual void consume_flush();

    virtual void open (RawPipeline * pip);
    virtual void close(RawPipeline * pip);

    const vector<RecordAttribute *> wantedFields;
    const int                       numOfBuckets;
    RecordAttribute               * hashProject ;

    expressions::Expression *       hashExpr    ;

    void *                          flushFunc   ;

    size_t                          blkVar_id   ;
    size_t                          cntVar_id   ;
    size_t                          oidVar_id   ;

    size_t                          blockSize   ;

    GpuRawContext * const           context     ;
};

#endif /* HASH_REARRANGE_HPP_ */
