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

#ifndef HASH_GROUP_BY_CHAINED_HPP_
#define HASH_GROUP_BY_CHAINED_HPP_

#include "util/gpu/gpu-raw-context.hpp"
#include "operators/monoids.hpp"
#include "operators/operators.hpp"
#include "expressions/expressions.hpp"
#include "util/raw-pipeline.hpp"

struct GpuAggrMatExpr{
public:
    expressions::Expression *   expr     ;
    size_t                      packet   ;
    size_t                      bitoffset;
    size_t                      packind  ;
    Monoid                      m        ;
    bool                        is_m     ;

    constexpr GpuAggrMatExpr(expressions::Expression *expr, size_t packet, size_t bitoffset, Monoid m):
                            expr(expr), packet(packet), bitoffset(bitoffset), packind(-1), m(m), is_m(true){}

    constexpr GpuAggrMatExpr(expressions::Expression *expr, size_t packet, size_t bitoffset):
                            expr(expr), packet(packet), bitoffset(bitoffset), packind(-1), m(SUM), is_m(false){}

    bool is_aggregation(){
        return is_m;
    }
};


class HashGroupByChained : public UnaryRawOperator {
public:
    HashGroupByChained(
        const std::vector<GpuAggrMatExpr>              &agg_exprs, 
        // const std::vector<size_t>                      &packet_widths,
        const std::vector<expressions::Expression *>    key_expr,
        RawOperator * const                             child,

        int                                             hash_bits,

        GpuRawContext *                                 context,
        size_t                                          maxInputSize,
        string                                          opLabel = "gb_chained");
    virtual ~HashGroupByChained() { LOG(INFO)<< "Collapsing HashGroupByChained operator";}

    virtual void produce();
    virtual void consume(RawContext* const context, const OperatorState& childState);

    virtual bool isFiltering() const{
        return true;
    }

    virtual void open (RawPipeline * pip);
    virtual void close(RawPipeline * pip);

private:
    void prepareDescription();
    void generate_build(RawContext* const context, const OperatorState& childState);
    void generate_scan();
    void buildHashTableFormat();
    llvm::Value * hash(llvm::Value * key);
    llvm::Value * hash(llvm::Value * old_seed, llvm::Value * key);
    llvm::Value * hash(const std::vector<expressions::Expression *> &exprs, RawContext* const context, const OperatorState& childState);

    string                                  opLabel         ;

    std::vector<GpuAggrMatExpr>             agg_exprs       ;
    std::vector<size_t>                     packet_widths   ;
    std::vector<expressions::Expression *>  key_expr        ;
    std::vector<llvm::Type *>               ptr_types       ;

    int                                     head_param_id   ;
    std::vector<int>                        out_param_ids   ;
    int                                     cnt_param_id    ;

    int                                     hash_bits       ;
    size_t                                  maxInputSize    ;

    GpuRawContext *                         context         ;

    RawPipelineGen *                        probe_gen       ;
};

#endif /* HASH_GROUP_BY_CHAINED_HPP_ */
