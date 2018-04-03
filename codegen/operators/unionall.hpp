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
#ifndef UNIONALL_HPP_
#define UNIONALL_HPP_

#include "operators/exchange.hpp"
#include "util/gpu/gpu-raw-context.hpp"

class UnionAll : public Exchange {
public:
    UnionAll(   vector<RawOperator *          >   & children    ,
                GpuRawContext * const               context     ,
                const vector<RecordAttribute *>   & wantedFields):
                    Exchange(   NULL,
                                context,
                                1,
                                wantedFields,
                                8,
                                NULL,
                                false,
                                false,
                                children.size()),
                    children(children){
    }

    virtual ~UnionAll(){ LOG(INFO)<<"Collapsing UnionAll operator";}

    virtual void produce();
//     virtual void consume(   RawContext * const context, const OperatorState& childState);
//     virtual void consume(GpuRawContext * const context, const OperatorState& childState);
//     virtual bool isFiltering() const {return false;}

// protected:
//     virtual void generate_catch();

private:
    vector<RawOperator *> children;
};

#endif /* UNIONALL_HPP_ */


