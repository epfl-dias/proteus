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

#ifndef HASH_JOIN_CHAINED_HPP_
#define HASH_JOIN_CHAINED_HPP_

#include "operators/gpu/gpu-materializer-expr.hpp"
#include "util/gpu/gpu-raw-context.hpp"
#include <unordered_map>

class HashJoinChained : public BinaryRawOperator {
public:
    HashJoinChained(
            const std::vector<GpuMatExpr>      &build_mat_exprs, 
            const std::vector<size_t>          &build_packet_widths,
            expressions::Expression *           build_keyexpr,
            RawOperator * const                 build_child,
            const std::vector<GpuMatExpr>      &probe_mat_exprs, 
            const std::vector<size_t>          &probe_mat_packet_widths,
            expressions::Expression *           probe_keyexpr,
            RawOperator * const                 probe_child,
            int                                 hash_bits,
            GpuRawContext *                     context,
            size_t                              maxBuildInputSize,
            string                              opLabel = "hj_chained");
    virtual ~HashJoinChained() { LOG(INFO)<< "Collapsing HashJoinChained operator";}

    virtual void produce();
    virtual void consume(RawContext* const context, const OperatorState& childState);

    void open_probe (RawPipeline * pip);
    void open_build (RawPipeline * pip);
    void close_probe(RawPipeline * pip);
    void close_build(RawPipeline * pip);

    virtual bool isFiltering() const{
        return true;
    }
private:
    void generate_build(RawContext* const context, const OperatorState& childState);
    void generate_probe(RawContext* const context, const OperatorState& childState);
    void buildHashTableFormat();
    void probeHashTableFormat();
    llvm::Value * hash(llvm::Value * key);

    string                  opLabel;

    std::vector<GpuMatExpr> build_mat_exprs;
    std::vector<GpuMatExpr> probe_mat_exprs;
    std::vector<size_t>     build_packet_widths;
    expressions::Expression *build_keyexpr;

    expressions::Expression *probe_keyexpr;
    
    std::vector<size_t>     packet_widths;

    int                     head_param_id;
    std::vector<int>        out_param_ids;
    std::vector<int>        in_param_ids ;
    int                     cnt_param_id ;

    int                     probe_head_param_id;

    int                     hash_bits  ;
    size_t                  maxBuildInputSize;

    // GpuExprMaterializer *   build_mat  ;
    // GpuExprMaterializer *   probe_mat  ;
    GpuRawContext *         context;

    // std::unordered_map<int32_t, std::vector<void *>> confs;
    std::vector<void *> confs[128];
};

#endif /* HASH_JOIN_CHAINED_HPP_ */
