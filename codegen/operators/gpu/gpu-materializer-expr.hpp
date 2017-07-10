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

#ifndef GPU_MATERIALIZER_EXPR_HPP_
#define GPU_MATERIALIZER_EXPR_HPP_

#include "operators/operators.hpp"
#include "operators/materializer-expr.hpp"
#include "expressions/expressions-generator.hpp"
#include "util/raw-functions.hpp"
#include "util/raw-caching.hpp"
#include "util/gpu/gpu-raw-context.hpp"


struct GpuMatExpr{
public:
    expressions::Expression *   expr     ;
    size_t                      packet   ;
    size_t                      bitoffset;
    size_t                      packind  ;

    constexpr GpuMatExpr(expressions::Expression *expr, size_t packet, size_t bitoffset):
                            expr(expr), packet(packet), bitoffset(bitoffset), packind(-1){}
};

/**
 * Issue when attempting realloc() on server
 */
class GpuExprMaterializer: public UnaryRawOperator {
public:
    // GpuExprMaterializer(expressions::Expression* expr, RawOperator* const child,
    //         GpuRawContext* const context, string opLabel);

    GpuExprMaterializer(const std::vector<GpuMatExpr> &exprs, 
            const std::vector<size_t> &packet_widths, RawOperator* const child,
            GpuRawContext* const context, string opLabel="out");
    // GpuExprMaterializer(expressions::Expression* expr, int linehint, RawOperator* const child,
    //             RawContext* const context, char* opLabel, Value * out_ptr, Value * out_cnt);
    virtual ~GpuExprMaterializer();
    virtual void produce();
    virtual void consume(RawContext* const context,
            const OperatorState& childState);
    virtual bool isFiltering() const {return false;}
private:
    // void freeArenas() const;
    // void updateRelationPointers() const;

    // StructType *toMatType;
    // struct matBuf opBuffer;
    // char *rawBuffer;
    // char **ptr_rawBuffer;

    // GpuRawContext* const    context;
    std::vector<GpuMatExpr> matExpr;
    string                  opLabel;
    
    std::vector<size_t>     packet_widths;

    std::vector<int>        out_param_ids;
    int                     cnt_param_id ;
};

#endif /* GPU_MATERIALIZER_EXPR_HPP_ */
