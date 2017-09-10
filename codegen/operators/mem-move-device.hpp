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
#ifndef MEM_MOVE_DEVICE_HPP_
#define MEM_MOVE_DEVICE_HPP_

#include "operators/operators.hpp"
#include "util/gpu/gpu-raw-context.hpp"

// void * make_mem_move_device(char * src, size_t bytes, int target_device, cudaStream_t strm);

class MemMoveDevice : public UnaryRawOperator {
public:
    MemMoveDevice(  RawOperator * const             child,
                    GpuRawContext * const           context,
                    const vector<RecordAttribute*> &wantedFields) :
                        UnaryRawOperator(child), 
                        context(context), 
                        wantedFields(wantedFields){}

    virtual ~MemMoveDevice()                                             { LOG(INFO)<<"Collapsing MemMoveDevice operator";}

    virtual void produce();
    virtual void consume(RawContext* const context, const OperatorState& childState);
    virtual bool isFiltering() const {return false;}

private:
    const vector<RecordAttribute *> wantedFields ;
    int                             device_id_var;
    int                             cu_stream_var;

    GpuRawContext * const context;

    void open(RawPipeline * pip);
};

#endif /* MEM_MOVE_DEVICE_HPP_ */
