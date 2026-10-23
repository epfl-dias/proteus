/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2026
        Data Intensive Applications and Systems Laboratory (DIAS)
                École Polytechnique Fédérale de Lausanne

                            All Rights Reserved.
*/

#include <olap/operators/decompression-flags.hpp>

#include <gflags/gflags.h>

DEFINE_bool(decom_shared, false,
            "If true, defer decompression to late operators when scan emits "
            "compressed blocks; otherwise keep MemMove-side decompression");

DEFINE_bool(decom_register, false,
            "If true, use CompressedBlockToTuplesRegisters instead of "
            "CompressedBlockToTuplesShared");

DEFINE_bool(decom_global_stage, false,
            "If true, decode tiles on demand into a bounded, recycled GPU "
            "global-memory staging pool and consume them from that pool");

DEFINE_uint32(global_stage_grid_rows, 16,
              "Number of blockIdx.y row-group workers in the one-kernel "
              "Global-stage window");

DEFINE_bool(precompute_rl, false,
            "If true, use RFOR datasets where RL streams are already cumulative "
            "(for --decom_register), so runtime RL prefix-sum conversion is skipped");

DEFINE_bool(pre_decom, false,
            "If true, pre-decompress RFOR/FLBP blocks once in PreDecompress::open "
            "and route unpack through BlockToTuples");

DEFINE_bool(use_metadata_in_block, true,
            "If true, use RFOR/FLBP metadata embedded in compressed blocks "
            "(PBMD layout) and enable metadata-in-block decode paths; "
            "if false, keep legacy metadata-from-JSON decompression flow");

DEFINE_bool(precomputed_offsets, true,
            "If true, CompressedBlockToTuples uses precomputed RFOR tile "
            "offset/val-section maps when available");

DEFINE_bool(progress_trace, false,
            "If true, CompressedBlockToTuples emits sampled GPU tile progress "
            "markers via printi64");

DEFINE_uint32(max_grid_blocks, 0,
              "Maximum CUDA gridX used by CompressedBlockToTuples kernels; "
              "0 selects twice the active GPU's SM count");

DEFINE_int32(
    threads_per_block, 0,
    "If > 0, override the backend-selected CompressedBlockToTuples CTA size "
    "at query-compilation time. 0 keeps the codec default. Backends may "
    "reject unsupported values.");

DEFINE_int32(
    items_per_thread, 0,
    "If > 0, override the backend-selected CompressedBlockToTuples items per "
    "thread at query-compilation time. 0 keeps the codec default. Backends "
    "may reject unsupported values.");

DEFINE_bool(rfor_scalar_last_field, false,
            "If true, CompressedBlockToTuples uses scalar RFOR decode for the "
            "last block field when there are >=4 block fields");

DEFINE_bool(emit_tuple_oid_bindings, true,
            "If true, CompressedBlockToTuples emits activeLoop/OID tuple "
            "bindings per tuple. Set to false only for experiments on "
            "pipelines that do not require tuple/OID bindings downstream.");

DEFINE_string(compression_type, "NONE",
              "enum option: NONE, CASCADED, LZ4, GDEFLATE, RFOR, FLBP");

DEFINE_int32(tile_size, 512,
             "RFOR dataset tile size in elements. Supported values: 512, 1024");
