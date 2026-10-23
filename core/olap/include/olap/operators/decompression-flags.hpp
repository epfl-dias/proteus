/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2026
        Data Intensive Applications and Systems Laboratory (DIAS)
                Ecole Polytechnique Federale de Lausanne

                            All Rights Reserved.
*/

#ifndef PROTEUS_OLAP_DECOMPRESSION_FLAGS_HPP
#define PROTEUS_OLAP_DECOMPRESSION_FLAGS_HPP

#include <gflags/gflags_declare.h>

DECLARE_bool(decom_shared);
DECLARE_bool(decom_register);
DECLARE_bool(decom_global_stage);
DECLARE_uint32(global_stage_grid_rows);
DECLARE_bool(precompute_rl);
DECLARE_bool(pre_decom);
DECLARE_bool(use_metadata_in_block);
DECLARE_bool(precomputed_offsets);
DECLARE_bool(progress_trace);
DECLARE_uint32(max_grid_blocks);
DECLARE_int32(threads_per_block);
DECLARE_int32(items_per_thread);
DECLARE_bool(rfor_scalar_last_field);
DECLARE_bool(emit_tuple_oid_bindings);
DECLARE_bool(ghr_dedicated_buffers);
DECLARE_string(compression_type);
DECLARE_int32(tile_size);

#endif  // PROTEUS_OLAP_DECOMPRESSION_FLAGS_HPP
