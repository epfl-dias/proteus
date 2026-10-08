/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2023
        Data Intensive Applications and Systems Laboratory (DIAS)
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
#include "compressed-block-to-tuples.hpp"

#include <array>
#include <codegen/jit/pipeline.hpp>
#include <cstdint>
#include <gflags/gflags.h>
#include <limits>
#include <memory>
#include <utility>
#include <olap/operators/decompression-config.hpp>
#include <olap/operators/decompression-flags.hpp>
#include <olap/plugins/binary-block-nvme-plugin.hpp>
#include <olap/plugins/plugins.hpp>
#include <olap/util/jit/control-flow/if-statement.hpp>
#include <platform/memory/block-manager.hpp>
#include <platform/memory/memory-manager.hpp>
#include <platform/topology/topology.hpp>
#include <platform/util/tracing.hpp>

#include "lib/util/catalog.hpp"

#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/InlineAsm.h"

DECLARE_bool(one_kernel);
DECLARE_uint32(streaming_grid_rows);

namespace
{
    // Fixed materialize window size now that ctt_materialize_tile_slots flag
    // is removed.
    static constexpr uint32_t kMaterializeTileSlotsDefault = 512u;
    // Hard cap for the bounded streaming placement. The actual allocation is
    // grid_x * --streaming_grid_rows slots and is normally much smaller.
    static constexpr uint32_t kMaxStreamingTileSlots = 4096u;
    static constexpr uint32_t kCb2tGridBlocksPerSm = 2u;

    static constexpr bool RFOR_COOPERATIVE_DECODE_SUPPORTED = true;
    static constexpr bool FLBP_COOPERATIVE_DECODE_SUPPORTED = true;
    static constexpr uint32_t FLBP_VECTOR_SIZE =
        proteus::flbp::cb2t::kVectorSize;
    static constexpr uint32_t FLBP_VECTOR_METADATA_BYTES =
        proteus::flbp::cb2t::kMetadataBytes;

    struct Cb2tRuntimeConfig
    {
        int block_threads;
        int items_per_thread;
        int tile_size;
    };

    constexpr Cb2tRuntimeConfig makeRuntimeConfig(
        const int block_threads, const int items_per_thread)
    {
        return {
            block_threads,
            items_per_thread,
            block_threads * items_per_thread
        };
    }

    Cb2tRuntimeConfig getDefaultRuntimeConfig(
        const proteus::decompression::Cb2tCodec codec)
    {
        const auto defaults =
            proteus::decompression::getCb2tRuntimeDefaults(codec);
        return makeRuntimeConfig(defaults.block_threads, defaults.items_per_thread);
    }

    Cb2tRuntimeConfig getRforDefaultRuntimeConfig()
    {
        return getDefaultRuntimeConfig(
            proteus::decompression::Cb2tCodec::kRfor);
    }

    Cb2tRuntimeConfig getFlbpDefaultRuntimeConfig()
    {
        return getDefaultRuntimeConfig(
            proteus::decompression::Cb2tCodec::kFlbp);
    }

    Cb2tRuntimeConfig applyCb2tRuntimeOverrides(const Cb2tRuntimeConfig defaults)
    {
        CHECK_GE(FLAGS_threads_per_block, 0)
            << "[CB2T] threads_per_block must be >= 0";
        CHECK_GE(FLAGS_items_per_thread, 0)
            << "[CB2T] items_per_thread must be >= 0";
        const int block_threads =
            FLAGS_threads_per_block > 0 ? FLAGS_threads_per_block
                                        : defaults.block_threads;
        const int items_per_thread =
            FLAGS_items_per_thread > 0 ? FLAGS_items_per_thread
                                       : defaults.items_per_thread;
        CHECK_GT(block_threads, 0)
            << "[CB2T] block_threads must be > 0";
        CHECK_GT(items_per_thread, 0)
            << "[CB2T] items_per_thread must be > 0";
        return makeRuntimeConfig(block_threads, items_per_thread);
    }

    enum class Cb2tExecutionMode
    {
        kShared,
        kLegacyGlobalTileWindow,
        kRegisters,
        kGlobalStage
    };

    template <bool LEGACY_GLOBAL_TILE_WINDOW, bool REGISTER_FUSED,
              bool GLOBAL_STAGE>
    constexpr Cb2tExecutionMode selectExecutionMode()
    {
        static_assert(!(GLOBAL_STAGE && LEGACY_GLOBAL_TILE_WINDOW));
        static_assert(!(GLOBAL_STAGE && REGISTER_FUSED));
        if constexpr (GLOBAL_STAGE)
        {
            return Cb2tExecutionMode::kGlobalStage;
        }
        if constexpr (LEGACY_GLOBAL_TILE_WINDOW)
        {
            return Cb2tExecutionMode::kLegacyGlobalTileWindow;
        }
        if constexpr (REGISTER_FUSED)
        {
            return Cb2tExecutionMode::kRegisters;
        }
        return Cb2tExecutionMode::kShared;
    }

    const char* cb2tExecutionModeName(const Cb2tExecutionMode mode)
    {
        switch (mode)
        {
        case Cb2tExecutionMode::kShared:
            return "shared";
        case Cb2tExecutionMode::kLegacyGlobalTileWindow:
            return "legacy_global_tile_window";
        case Cb2tExecutionMode::kRegisters:
            return "registers";
        case Cb2tExecutionMode::kGlobalStage:
            return "global_stage";
        }
        LOG(FATAL) << "[CB2T] unknown execution mode";
        __builtin_unreachable();
    }

    enum class Cb2tCodecKind
    {
        kRfor,
        kFlbp
    };

    enum class Cb2tDecodeAbiKind
    {
        kRforTileMetadata,
        kFlbpTileMetadataWords
    };

    struct Cb2tDecodeKernelSpec
    {
        Cb2tDecodeAbiKind abi_kind;
        const char* cooperative_symbol;
        const char* predecode_symbol;
        const char* predecode_cumulative_symbol;
        const char* predecode_element_symbol;
        uint32_t scratch_elems;
        uint32_t chunk_metadata_bytes;
    };

    class ICb2tDecodeBackend
    {
    public:
        virtual ~ICb2tDecodeBackend() = default;
        [[nodiscard]] virtual const char* codecName() const = 0;
        [[nodiscard]] virtual Cb2tCodecKind codecKind() const = 0;
        [[nodiscard]] virtual Cb2tRuntimeConfig defaultRuntimeConfig() const = 0;
        [[nodiscard]] virtual Cb2tRuntimeConfig resolveRuntimeConfig(
            Cb2tRuntimeConfig requested) const = 0;
        [[nodiscard]] virtual Cb2tDecodeKernelSpec kernelSpec(
            Cb2tRuntimeConfig runtime_config) const = 0;
        [[nodiscard]] virtual uint32_t logicalTilesPerChunk(uint32_t tile_elems) const = 0;
        [[nodiscard]] virtual bool supportsCooperativeDecode() const = 0;
        [[nodiscard]] virtual bool supportsRegisterPredecode() const = 0;
    };

    Cb2tRuntimeConfig getSelectedCb2tRuntimeConfig(const ICb2tDecodeBackend& backend)
    {
        return backend.resolveRuntimeConfig(
            applyCb2tRuntimeOverrides(backend.defaultRuntimeConfig()));
    }

    uint32_t rforDecodeScratchElems(const Cb2tRuntimeConfig runtime_config)
    {
        // Must match the RFOR cooperative decode helper selected for this
        // runtime shape in buffer-manager.cu. The LoadRBinPack_notemplate_overlap
        // body keeps TWO distinct tile-sized shared regions live at once: the RLE
        // working_area at [0, T) and the packed-stream staged_area at [T, 2T)
        // (its own static_assert reserves required_scratch_words == 2*ITEMS_PER_GROUP).
        // Reserving only 1*T here let staged_area alias out_base_elems (the column
        // output tiles): decoding column k>=1 overwrote column 0's decoded tile
        // with packed bytes, deterministically corrupting decom_op results for
        // every multi-column query while dropping no tuples.
        return static_cast<uint32_t>(
            2u * runtime_config.block_threads * runtime_config.items_per_thread);
    }

    constexpr uint32_t flbpDecodeScratchElems()
    {
        // FLBP decode reads packed words directly and emits to output tile.
        return 0u;
    }

    class RforLoadRBinPackBackend final : public ICb2tDecodeBackend
    {
    public:
        [[nodiscard]] const char* codecName() const override
        {
            return "RFOR";
        }

        [[nodiscard]] Cb2tCodecKind codecKind() const override
        {
            return Cb2tCodecKind::kRfor;
        }

        [[nodiscard]] Cb2tRuntimeConfig defaultRuntimeConfig() const override
        {
            return getRforDefaultRuntimeConfig();
        }

        [[nodiscard]] Cb2tRuntimeConfig resolveRuntimeConfig(
            Cb2tRuntimeConfig requested) const override
        {
            // RFOR CB2T is not fully runtime-generic. We currently compile a
            // small set of concrete cooperative decode shapes and dispatch
            // between them here.
            const bool is_64x8 =
                requested.block_threads == 64 && requested.items_per_thread == 8;
            const bool is_128x4 =
                requested.block_threads == 128 && requested.items_per_thread == 4;
            const bool is_256x2 =
                requested.block_threads == 256 && requested.items_per_thread == 2;
            const bool is_512x2 =
                requested.block_threads == 512 && requested.items_per_thread == 2;
            const bool is_128x8 =
                requested.block_threads == 128 && requested.items_per_thread == 8;
            const bool is_256x4 =
                requested.block_threads == 256 && requested.items_per_thread == 4;
            CHECK(is_64x8 || is_128x4 || is_256x2 || is_512x2 ||
                  is_128x8 || is_256x4)
                << "[CB2T][RFOR] runtime shape "
                << requested.block_threads << "x" << requested.items_per_thread
                << " is not supported by the compiled RFOR device decode symbols; "
                << "supported shapes are 64x8, 128x4, "
                << "256x2 (--tile_size=512), and "
                << "512x2, 256x4, or 128x8 (--tile_size=1024)";
            return makeRuntimeConfig(
                requested.block_threads, requested.items_per_thread);
        }

        [[nodiscard]] Cb2tDecodeKernelSpec kernelSpec(
            const Cb2tRuntimeConfig runtime_config) const override
        {
            const bool is_256x2 =
                runtime_config.block_threads == 256 &&
                runtime_config.items_per_thread == 2;
            const bool is_128x8 =
                runtime_config.block_threads == 128 &&
                runtime_config.items_per_thread == 8;
            const bool is_256x4 =
                runtime_config.block_threads == 256 &&
                runtime_config.items_per_thread == 4;
            const bool is_64x8 =
                runtime_config.block_threads == 64 &&
                runtime_config.items_per_thread == 8;
            const bool is_512x2 =
                runtime_config.block_threads == 512 &&
                runtime_config.items_per_thread == 2;
            return {
                Cb2tDecodeAbiKind::kRforTileMetadata,
                is_64x8 ? "LoadRBinPack_notemplate_overlap_64x8"
                : is_512x2 ? "LoadRBinPack_notemplate_overlap_512x2"
                : is_256x4 ? "LoadRBinPack_notemplate_overlap_256x4"
                : is_128x8 ? "LoadRBinPack_notemplate_overlap_128x8"
                : is_256x2 ? "LoadRBinPack_notemplate_overlap_256x2"
                           : "LoadRBinPack_notemplate_overlap",
                is_64x8 ? "LoadRBinPack_predecode_notemplate_64x8"
                : is_512x2 ? "LoadRBinPack_predecode_notemplate_512x2"
                : is_256x4 ? "LoadRBinPack_predecode_notemplate_256x4"
                : is_128x8 ? "LoadRBinPack_predecode_notemplate_128x8"
                : is_256x2 ? "LoadRBinPack_predecode_notemplate_256x2"
                           : "LoadRBinPack_predecode_notemplate",
                is_64x8
                    ? "LoadRBinPack_predecode_cumulative_notemplate_64x8"
                : is_512x2
                    ? "LoadRBinPack_predecode_cumulative_notemplate_512x2"
                : is_256x4
                    ? "LoadRBinPack_predecode_cumulative_notemplate_256x4"
                : is_128x8
                    ? "LoadRBinPack_predecode_cumulative_notemplate_128x8"
                : is_256x2
                    ? "LoadRBinPack_predecode_cumulative_notemplate_256x2"
                    : "LoadRBinPack_predecode_cumulative_notemplate",
                nullptr,
                rforDecodeScratchElems(runtime_config),
                static_cast<uint32_t>(sizeof(uint32_t) * 6)
            };
        }

        [[nodiscard]] uint32_t logicalTilesPerChunk(
            const uint32_t /*tile_elems*/) const override
        {
            return 1u;
        }

        [[nodiscard]] bool supportsCooperativeDecode() const override
        {
            return RFOR_COOPERATIVE_DECODE_SUPPORTED;
        }

        [[nodiscard]] bool supportsRegisterPredecode() const override
        {
            return true;
        }
    };

    class FlbpLoadBackend final : public ICb2tDecodeBackend
    {
    public:
        [[nodiscard]] const char* codecName() const override
        {
            return "FLBP";
        }

        [[nodiscard]] Cb2tCodecKind codecKind() const override
        {
            return Cb2tCodecKind::kFlbp;
        }

        [[nodiscard]] Cb2tRuntimeConfig defaultRuntimeConfig() const override
        {
            return getFlbpDefaultRuntimeConfig();
        }

        [[nodiscard]] Cb2tRuntimeConfig resolveRuntimeConfig(
            Cb2tRuntimeConfig requested) const override
        {
            CHECK_GT(requested.block_threads, 0)
                << "[CB2T][FLBP] block_threads must be > 0";
            CHECK_GT(requested.items_per_thread, 0)
                << "[CB2T][FLBP] items_per_thread must be > 0";
            CHECK_LE(requested.block_threads, 1024)
                << "[CB2T][FLBP] block_threads must be <= 1024";
            requested.tile_size = requested.block_threads * requested.items_per_thread;
            CHECK_GT(requested.tile_size, 0)
                << "[CB2T][FLBP] tile_size must be > 0";
            CHECK_EQ(FLBP_VECTOR_SIZE % static_cast<uint32_t>(requested.tile_size), 0u)
                << "[CB2T][FLBP] tile size must divide FLBP vector size";
            return requested;
        }

        [[nodiscard]] Cb2tDecodeKernelSpec kernelSpec(
            const Cb2tRuntimeConfig runtime_config) const override
        {
            const bool full_vector_tile =
                runtime_config.tile_size == static_cast<int>(FLBP_VECTOR_SIZE);
            return {
                Cb2tDecodeAbiKind::kFlbpTileMetadataWords,
                full_vector_tile ? "LoadFLBPFullVector_notemplate_overlap"
                                 : "LoadFLBP_notemplate_overlap",
                nullptr,
                nullptr,
                full_vector_tile ? "DecodeFLBPElement_fullvector_notemplate"
                                 : "DecodeFLBPElement_notemplate",
                flbpDecodeScratchElems(),
                FLBP_VECTOR_METADATA_BYTES
            };
        }

        [[nodiscard]] uint32_t logicalTilesPerChunk(
            const uint32_t tile_elems) const override
        {
            CHECK_GT(tile_elems, 0u);
            CHECK_EQ(FLBP_VECTOR_SIZE % tile_elems, 0u)
                << "[CB2T][FLBP] tile size must divide FLBP vector size";
            return FLBP_VECTOR_SIZE / tile_elems;
        }

        [[nodiscard]] bool supportsCooperativeDecode() const override
        {
            return FLBP_COOPERATIVE_DECODE_SUPPORTED;
        }

        [[nodiscard]] bool supportsRegisterPredecode() const override
        {
            return true;
        }
    };

    const ICb2tDecodeBackend& getCb2tDecodeBackend()
    {
        static const RforLoadRBinPackBackend kRforBackend{};
        static const FlbpLoadBackend kFlbpBackend{};
        if (FLAGS_compression_type == "RFOR" || FLAGS_compression_type == "rfor")
        {
            return kRforBackend;
        }
        if (FLAGS_compression_type == "FLBP" || FLAGS_compression_type == "flbp")
        {
            return kFlbpBackend;
        }
        LOG(FATAL) << "[CB2T] unsupported compression type for decode backend: "
                   << FLAGS_compression_type;
        __builtin_unreachable();
    }

    struct Cb2tSharedLayout
    {
        size_t decode_scratch_elems;
        size_t column_tile_elems;
        size_t out_base_elems;
        size_t run_values_base_elems;
        size_t run_ends_base_elems;
        size_t chunk_meta_elems;
        size_t tile_meta_elems;
        size_t broadcast_elems;
        size_t broadcast_base_elems;
        // Kernel-start block-uniform broadcasts (block OID, active count, one
        // compressed-block pointer per column) get dedicated write-once slots.
        // They must never share slots with each other or with the per-tile
        // staging area: a barrier orders writer-before-readers, but nothing
        // stops a lagging warp from reading a slot AFTER a later phase
        // overwrote it (read->overwrite WAR race, silent warp-sized tuple
        // drops under one-kernel execution).
        size_t block_uniform_base_elems;
        size_t total_elems;
    };

    Cb2tSharedLayout buildCb2tSharedLayout(const size_t block_field_count,
                                           const size_t tile_elems,
                                           const bool use_predecode_registers,
                                           const bool use_streaming_global,
                                           const bool predecode_uses_run_scratch,
                                           const bool compact_precomputed_fastpath,
                                           const uint32_t decode_scratch_elems)
    {
        const bool fused_predecode_tuple_execution = use_predecode_registers;
        Cb2tSharedLayout layout{};
        layout.decode_scratch_elems = use_predecode_registers ? 0u : decode_scratch_elems;
        layout.column_tile_elems = fused_predecode_tuple_execution
                                       ? 0u
                                       : (use_streaming_global
                                              ? tile_elems
                                              : block_field_count * tile_elems);
        layout.out_base_elems = layout.decode_scratch_elems;
        layout.run_values_base_elems = layout.out_base_elems + layout.column_tile_elems;
        layout.run_ends_base_elems = layout.run_values_base_elems +
                                     (predecode_uses_run_scratch ? tile_elems : 0u);
        layout.chunk_meta_elems = 2 * block_field_count;
        layout.tile_meta_elems = compact_precomputed_fastpath ? 0u : (6 * block_field_count);
        layout.broadcast_elems = layout.chunk_meta_elems + layout.tile_meta_elems;
        layout.broadcast_base_elems = use_predecode_registers
                                          ? (predecode_uses_run_scratch
                                                 ? (layout.run_ends_base_elems + tile_elems)
                                                 : (layout.out_base_elems +
                                                    layout.column_tile_elems))
                                          : (layout.out_base_elems + layout.column_tile_elems);
        layout.total_elems = layout.broadcast_base_elems + layout.broadcast_elems;
        // Write-once block-uniform region: [oid_lo, oid_hi, cnt_lo, cnt_hi,
        // col0_ptr_lo, col0_ptr_hi, col1_ptr_lo, ...].
        layout.block_uniform_base_elems = layout.total_elems;
        layout.total_elems += 4 + 2 * block_field_count;
        return layout;
    }

    struct Cb2tPrecomputedOffsetMaps
    {
        bool use_precomputed_tile_offsets = false;
        std::vector<llvm::Value*> chunk_offsets_oid_maps;
    };

    struct Cb2tPlan
    {
        Cb2tCodecKind codec_kind;
        Cb2tRuntimeConfig runtime_config;
        uint32_t logical_tiles_per_chunk;
        Cb2tSharedLayout shared_layout;
        bool compact_precomputed_fastpath;
    };

    struct Cb2tColumnMetadataState
    {
        llvm::Value* num_chunks_i64 = nullptr;
        llvm::Value* physical_num_chunks_i64 = nullptr;
        llvm::Value* chunk_data_base = nullptr;
        llvm::Value* precomputed_chunk_offsets_ptr = nullptr;
        llvm::Value* magic_ok = nullptr;
        llvm::Value* chunk_sizes_ptr = nullptr;
    };

    struct Cb2tEmitContext
    {
        OlapParallelContext* context = nullptr;
        llvm::IRBuilder<>* Builder = nullptr;
        llvm::Type* i8_ty = nullptr;
        llvm::Type* i32_ty = nullptr;
        llvm::Type* i64_ty = nullptr;
        llvm::StructType* tile_metadata_ty = nullptr;
        llvm::PointerType* tile_metadata_ptr_ty = nullptr;
        llvm::StructType* flbp_vector_metadata_ty = nullptr;
        llvm::Value* shared_buffer_generic_ptr = nullptr;
        llvm::Value* is_block_leader = nullptr;
        llvm::Value* tile_id = nullptr;
        llvm::Value* tile_id_i32 = nullptr;
        llvm::Value* null_i32_ptr = nullptr;
        llvm::Function* barrier = nullptr;
        llvm::Function* membar_cta = nullptr;
        llvm::Value* zero_i64 = nullptr;
        llvm::Value* one_i64 = nullptr;
        llvm::Value* chunk_metadata_bytes_i32 = nullptr;
        size_t shared_broadcast_base_elems = 0;
        size_t shared_chunk_meta_elems = 0;
    };

    struct Cb2tTileResolution
    {
        llvm::Value* can_decompress_block = nullptr;
        llvm::Value* chunk_offset_slot = nullptr;
        llvm::Value* val_section_size_slot = nullptr;
        llvm::Value* tile_metadata_ptr = nullptr;
    };

    // Names where decompress_tile() places the decoded tile for a column.
    // Today this is the per-column shared output tile; later migration steps
    // can extend it to describe register/global placement without changing the
    // hook signature.
    struct Cb2tDecodeTarget
    {
        llvm::Value* out_ptr_i32 = nullptr;
    };

    // Loop-invariant machinery decompress_tile() needs: the decode device
    // symbols (selected per runtime shape from the codec's kernelSpec), the
    // cooperative shared scratch, and the register/fused placement state. Bundled
    // so the codec's emit_decompress_tile hook owns the decode dispatch instead
    // of an inline lambda in consume().
    struct Cb2tDecodeContext
    {
        bool use_predecode_registers = false;
        bool fused_predecode_tuple_execution = false;
        uint32_t block_threads = 0;
        uint32_t items_per_thread = 0;
        llvm::Value* tid_in_block = nullptr;
        llvm::Value* llvm0 = nullptr;
        llvm::Value* shared_buffer_shared_ptr = nullptr;
        llvm::Value* predecode_run_values_scratch_ptr = nullptr;
        llvm::Value* predecode_run_ends_scratch_ptr = nullptr;
        llvm::FunctionCallee decode_cooperative_func{};
        llvm::FunctionCallee decode_predecode_setup_func{};
        llvm::FunctionCallee decode_predecode_element_func{};
        llvm::StructType* rle_tile_metadata_ty = nullptr;
        llvm::ArrayType* fused_tuple_items_ty = nullptr;
        const std::vector<llvm::Value*>* fused_tuple_values_ptrs = nullptr;
        std::function<llvm::Value*(llvm::Value*, llvm::Value*, llvm::Value*,
                                   llvm::Value*)>
            rfor_predecode_element_inline{};
        llvm::Type* i32_ptr_ty = nullptr;
    };

    llvm::Value* makeSharedI32Slot(llvm::IRBuilder<>* Builder,
                                   llvm::Type* i32_ty,
                                   llvm::Value* shared_buffer_generic_ptr,
                                   const uint64_t idx)
    {
        return Builder->CreateInBoundsGEP(
            i32_ty,
            shared_buffer_generic_ptr,
            llvm::ConstantInt::get(i32_ty, idx));
    }

    llvm::StoreInst* createVolatileSharedI32Store(llvm::IRBuilder<>* Builder,
                                                  llvm::Value* value,
                                                  llvm::Value* slot)
    {
        auto* st = Builder->CreateStore(value, slot);
        st->setVolatile(true);
        return st;
    }

    llvm::LoadInst* createVolatileSharedI32Load(llvm::IRBuilder<>* Builder,
                                                llvm::Type* i32_ty,
                                                llvm::Value* slot)
    {
        auto* ld = Builder->CreateLoad(i32_ty, slot);
        ld->setVolatile(true);
        return ld;
    }

    using Cb2tTileMetadataWords = std::array<llvm::Value*, 6>;

    Cb2tTileMetadataWords loadTileMetadataWords(
        const Cb2tEmitContext& resolve_ctx, llvm::Value* tile_metadata_ptr)
    {
        Cb2tTileMetadataWords metadata_words{};
        for (unsigned meta_idx = 0; meta_idx < metadata_words.size(); ++meta_idx)
        {
            llvm::Value* meta_src_ptr = resolve_ctx.Builder->CreateStructGEP(
                resolve_ctx.tile_metadata_ty, tile_metadata_ptr, meta_idx);
            metadata_words[meta_idx] =
                resolve_ctx.Builder->CreateLoad(resolve_ctx.i32_ty, meta_src_ptr);
        }
        return metadata_words;
    }

    void stageTileMetadataWordsToShared(
        const Cb2tEmitContext& resolve_ctx,
        const Cb2tTileMetadataWords& metadata_words,
        const uint64_t staged_tile_meta_base)
    {
        for (unsigned meta_idx = 0; meta_idx < metadata_words.size(); ++meta_idx)
        {
            llvm::Value* meta_dst_ptr = makeSharedI32Slot(
                resolve_ctx.Builder,
                resolve_ctx.i32_ty,
                resolve_ctx.shared_buffer_generic_ptr,
                staged_tile_meta_base + meta_idx);
            createVolatileSharedI32Store(
                resolve_ctx.Builder, metadata_words[meta_idx], meta_dst_ptr);
        }
    }

    void stageFlbpTileMetadataToShared(const Cb2tEmitContext& resolve_ctx,
                                       const uint64_t staged_tile_meta_base,
                                       llvm::Value* base_value_i32,
                                       llvm::Value* bitwidth_i32,
                                       llvm::Value* element_count_i32,
                                       llvm::Value* subtile_offset_i32,
                                       llvm::Value* subtile_size_i32)
    {
        llvm::Value* base_slot = makeSharedI32Slot(
            resolve_ctx.Builder,
            resolve_ctx.i32_ty,
            resolve_ctx.shared_buffer_generic_ptr,
            staged_tile_meta_base + 0);
        llvm::Value* bitwidth_slot = makeSharedI32Slot(
            resolve_ctx.Builder,
            resolve_ctx.i32_ty,
            resolve_ctx.shared_buffer_generic_ptr,
            staged_tile_meta_base + 1);
        llvm::Value* element_count_slot = makeSharedI32Slot(
            resolve_ctx.Builder,
            resolve_ctx.i32_ty,
            resolve_ctx.shared_buffer_generic_ptr,
            staged_tile_meta_base + 2);
        llvm::Value* subtile_offset_slot = makeSharedI32Slot(
            resolve_ctx.Builder,
            resolve_ctx.i32_ty,
            resolve_ctx.shared_buffer_generic_ptr,
            staged_tile_meta_base + 3);
        llvm::Value* subtile_size_slot = makeSharedI32Slot(
            resolve_ctx.Builder,
            resolve_ctx.i32_ty,
            resolve_ctx.shared_buffer_generic_ptr,
            staged_tile_meta_base + 4);
        llvm::Value* reserved_slot = makeSharedI32Slot(
            resolve_ctx.Builder,
            resolve_ctx.i32_ty,
            resolve_ctx.shared_buffer_generic_ptr,
            staged_tile_meta_base + 5);
        createVolatileSharedI32Store(resolve_ctx.Builder, base_value_i32, base_slot);
        createVolatileSharedI32Store(resolve_ctx.Builder, bitwidth_i32, bitwidth_slot);
        createVolatileSharedI32Store(
            resolve_ctx.Builder, element_count_i32, element_count_slot);
        createVolatileSharedI32Store(
            resolve_ctx.Builder, subtile_offset_i32, subtile_offset_slot);
        createVolatileSharedI32Store(
            resolve_ctx.Builder, subtile_size_i32, subtile_size_slot);
        createVolatileSharedI32Store(
            resolve_ctx.Builder,
            llvm::ConstantInt::get(resolve_ctx.i32_ty, 0), reserved_slot);
    }

    // ---------------------------------------------------------------------
    // Tile location strategy (metadata-source axis, orthogonal to the codec).
    // The engine chooses how a tile's compressed bytes are located within the
    // column block independently of which codec decodes them: a self-describing
    // in-block PBMD header (paper "self-describing blocks") or a precomputed
    // chunk-offset map (engine-side optimization). Both yield the same byte
    // offset; the codec backend then parses its own header from there.
    // ---------------------------------------------------------------------
    class ICb2tTileLocator
    {
    public:
        virtual ~ICb2tTileLocator() = default;
        // Thread-uniform predicate: is the tile locatable from this source?
        [[nodiscard]] virtual llvm::Value* emitSourceOk(
            const Cb2tEmitContext& ctx,
            const Cb2tColumnMetadataState& state) const = 0;
        // Leader-only, inside the can-decompress scope: compute and return the
        // tile's byte offset into chunk_data_base for the given chunk index
        // (tile_id for RFOR; vector_id for FLBP).
        [[nodiscard]] virtual llvm::Value* emitChunkOffset(
            const Cb2tEmitContext& ctx, const Cb2tColumnMetadataState& state,
            llvm::Value* chunk_index_i64, llvm::Value* chunk_index_i32,
            size_t block_idx) const = 0;
    };

    // Precomputed chunk-offset map (plugin-owned device array, indexed by chunk).
    class Cb2tPrecomputedLocator final : public ICb2tTileLocator
    {
    public:
        [[nodiscard]] llvm::Value* emitSourceOk(
            const Cb2tEmitContext& ctx,
            const Cb2tColumnMetadataState& state) const override
        {
            return ctx.Builder->CreateICmpNE(
                state.precomputed_chunk_offsets_ptr, ctx.null_i32_ptr);
        }

        [[nodiscard]] llvm::Value* emitChunkOffset(
            const Cb2tEmitContext& ctx, const Cb2tColumnMetadataState& state,
            llvm::Value* /*chunk_index_i64*/, llvm::Value* chunk_index_i32,
            size_t /*block_idx*/) const override
        {
            llvm::Value* chunk_offset_ptr = ctx.Builder->CreateInBoundsGEP(
                ctx.i32_ty, state.precomputed_chunk_offsets_ptr, chunk_index_i32);
            return ctx.Builder->CreateLoad(ctx.i32_ty, chunk_offset_ptr);
        }
    };

    // Self-describing in-block PBMD header: either a direct per-chunk offset
    // table (first entry == 0) or a cumulative chunk-size prefix to be summed.
    class Cb2tInBlockLocator final : public ICb2tTileLocator
    {
    public:
        [[nodiscard]] llvm::Value* emitSourceOk(
            const Cb2tEmitContext& ctx,
            const Cb2tColumnMetadataState& state) const override
        {
            (void)ctx;
            return state.magic_ok;
        }

        [[nodiscard]] llvm::Value* emitChunkOffset(
            const Cb2tEmitContext& ctx, const Cb2tColumnMetadataState& state,
            llvm::Value* chunk_index_i64, llvm::Value* chunk_index_i32,
            size_t block_idx) const override
        {
            auto off_name = "loc_chunk_off_" + std::to_string(block_idx);
            llvm::Value* result_off_ptr =
                ctx.context->CreateEntryBlockAlloca(off_name, ctx.i32_ty);
            ctx.Builder->CreateStore(
                llvm::ConstantInt::get(ctx.i32_ty, 0), result_off_ptr);

            llvm::Value* first_chunk_index_i32 =
                ctx.Builder->CreateLoad(ctx.i32_ty, state.chunk_sizes_ptr);
            llvm::Value* pbmd_stores_offsets = ctx.Builder->CreateICmpEQ(
                first_chunk_index_i32, llvm::ConstantInt::get(ctx.i32_ty, 0));

            ctx.context->gen_if(
                {ProteusValue{pbmd_stores_offsets,
                              ctx.context->createFalse()}})([&]()
            {
                llvm::Value* chunk_offset_ptr = ctx.Builder->CreateInBoundsGEP(
                    ctx.i32_ty, state.chunk_sizes_ptr, chunk_index_i32);
                llvm::Value* chunk_offset_i32 =
                    ctx.Builder->CreateLoad(ctx.i32_ty, chunk_offset_ptr);
                ctx.Builder->CreateStore(chunk_offset_i32, result_off_ptr);
            });

            ctx.context->gen_if(
                {ProteusValue{ctx.Builder->CreateNot(pbmd_stores_offsets),
                              ctx.context->createFalse()}})([&]()
            {
                auto idx_name = "loc_chunk_idx_" + std::to_string(block_idx);
                auto acc_name = "loc_chunk_acc_" + std::to_string(block_idx);
                llvm::Value* chunk_idx_ptr =
                    ctx.context->CreateEntryBlockAlloca(idx_name, ctx.i64_ty);
                llvm::Value* chunk_acc_ptr =
                    ctx.context->CreateEntryBlockAlloca(acc_name, ctx.i64_ty);
                ctx.Builder->CreateStore(ctx.zero_i64, chunk_idx_ptr);
                ctx.Builder->CreateStore(ctx.zero_i64, chunk_acc_ptr);

                ctx.context->gen_while([&]()
                {
                    llvm::Value* idx =
                        ctx.Builder->CreateLoad(ctx.i64_ty, chunk_idx_ptr);
                    return ProteusValue{
                        ctx.Builder->CreateICmpULT(idx, chunk_index_i64),
                        ctx.context->createFalse()};
                })([&](llvm::BranchInst* /*loop_cond*/)
                {
                    llvm::Value* idx =
                        ctx.Builder->CreateLoad(ctx.i64_ty, chunk_idx_ptr);
                    llvm::Value* idx_i32 =
                        ctx.Builder->CreateTrunc(idx, ctx.i32_ty);
                    llvm::Value* size_ptr = ctx.Builder->CreateInBoundsGEP(
                        ctx.i32_ty, state.chunk_sizes_ptr, idx_i32);
                    llvm::Value* size_i32 =
                        ctx.Builder->CreateLoad(ctx.i32_ty, size_ptr);
                    llvm::Value* size_i64 =
                        ctx.Builder->CreateZExt(size_i32, ctx.i64_ty);
                    llvm::Value* acc =
                        ctx.Builder->CreateLoad(ctx.i64_ty, chunk_acc_ptr);
                    ctx.Builder->CreateStore(
                        ctx.Builder->CreateAdd(acc, size_i64), chunk_acc_ptr);
                    ctx.Builder->CreateStore(
                        ctx.Builder->CreateAdd(idx, ctx.one_i64), chunk_idx_ptr);
                });

                llvm::Value* chunk_offset_i32 = ctx.Builder->CreateTrunc(
                    ctx.Builder->CreateLoad(ctx.i64_ty, chunk_acc_ptr),
                    ctx.i32_ty);
                ctx.Builder->CreateStore(chunk_offset_i32, result_off_ptr);
            });

            return ctx.Builder->CreateLoad(ctx.i32_ty, result_off_ptr);
        }
    };

    // ---------------------------------------------------------------------
    // Codec backend hook #1: resolve_metadata (paper Sec. "Decompression
    // encapsulation"). One implementation per codec; the metadata source is
    // injected as an ICb2tTileLocator so the codec stays source-agnostic.
    // ---------------------------------------------------------------------
    class ICb2tCodecResolver
    {
    public:
        virtual ~ICb2tCodecResolver() = default;
        // Backend hook #1: resolve_metadata.
        [[nodiscard]] virtual Cb2tTileResolution emit_resolve_metadata(
            size_t block_idx, const Cb2tColumnMetadataState& state,
            const Cb2tEmitContext& ctx,
            const ICb2tTileLocator& locator) const = 0;
        // Backend hook #2: decompress_tile. Decodes one tile of the column into
        // the decode target (shared tile / fused registers), reading the
        // resolved metadata and the located val/rl streams.
        virtual void emit_decompress_tile(
            size_t block_idx, const Cb2tTileResolution& tile_resolution,
            const Cb2tDecodeTarget& decode_target, llvm::Value* val_data_ptr,
            llvm::Value* rl_data_ptr, const Cb2tEmitContext& ctx,
            const Cb2tDecodeContext& dctx) const = 0;
    };

    class Cb2tRforResolver final : public ICb2tCodecResolver
    {
    public:
        explicit Cb2tRforResolver(const bool compact_fastpath)
            : compact_fastpath_(compact_fastpath)
        {
        }

        [[nodiscard]] Cb2tTileResolution emit_resolve_metadata(
            size_t block_idx, const Cb2tColumnMetadataState& state,
            const Cb2tEmitContext& ctx,
            const ICb2tTileLocator& locator) const override
        {
            const uint64_t broadcast_base = static_cast<uint64_t>(
                ctx.shared_broadcast_base_elems + (2 * block_idx));
            llvm::Value* chunk_offset_slot = makeSharedI32Slot(
                ctx.Builder, ctx.i32_ty, ctx.shared_buffer_generic_ptr,
                broadcast_base);
            llvm::Value* val_section_size_slot = makeSharedI32Slot(
                ctx.Builder, ctx.i32_ty, ctx.shared_buffer_generic_ptr,
                broadcast_base + 1);
            const uint64_t staged_tile_meta_base = static_cast<uint64_t>(
                ctx.shared_broadcast_base_elems + ctx.shared_chunk_meta_elems +
                (6 * block_idx));

            llvm::Value* in_range = ctx.Builder->CreateICmpULT(
                ctx.tile_id, state.num_chunks_i64);
            llvm::Value* can_decompress_thread = ctx.Builder->CreateAnd(
                in_range, locator.emitSourceOk(ctx, state));

            ctx.context->gen_if(
                {ProteusValue{ctx.is_block_leader,
                              ctx.context->createFalse()}})([&]()
            {
                createVolatileSharedI32Store(
                    ctx.Builder, llvm::ConstantInt::get(ctx.i32_ty, 0),
                    chunk_offset_slot);
                createVolatileSharedI32Store(
                    ctx.Builder, llvm::ConstantInt::get(ctx.i32_ty, 0),
                    val_section_size_slot);

                ctx.context->gen_if(
                    {ProteusValue{can_decompress_thread,
                                  ctx.context->createFalse()}})([&]()
                {
                    llvm::Value* chunk_offset_i32 = locator.emitChunkOffset(
                        ctx, state, ctx.tile_id, ctx.tile_id_i32, block_idx);
                    createVolatileSharedI32Store(
                        ctx.Builder, chunk_offset_i32, chunk_offset_slot);

                    llvm::Value* chunk_offset_i64 =
                        ctx.Builder->CreateZExt(chunk_offset_i32, ctx.i64_ty);
                    llvm::Value* chunk_base = ctx.Builder->CreateInBoundsGEP(
                        ctx.i8_ty, state.chunk_data_base, chunk_offset_i64);
                    llvm::Value* tile_hdr = ctx.Builder->CreateBitCast(
                        chunk_base, ctx.tile_metadata_ptr_ty);
                    Cb2tTileMetadataWords tile_metadata_words =
                        loadTileMetadataWords(ctx, tile_hdr);

                    // RFOR val_section_size =
                    // chunk_metadata_bytes + ceil(val_count*val_bitwidth/32)*4
                    llvm::Value* vbw = tile_metadata_words[1];
                    llvm::Value* vcnt = tile_metadata_words[2];
                    llvm::Value* val_packed_words = ctx.Builder->CreateLShr(
                        ctx.Builder->CreateAdd(
                            ctx.Builder->CreateMul(vcnt, vbw),
                            llvm::ConstantInt::get(ctx.i32_ty, 31)),
                        llvm::ConstantInt::get(ctx.i32_ty, 5));
                    llvm::Value* val_section_size_i32 = ctx.Builder->CreateAdd(
                        ctx.chunk_metadata_bytes_i32,
                        ctx.Builder->CreateMul(
                            val_packed_words,
                            llvm::ConstantInt::get(ctx.i32_ty, 4)));
                    createVolatileSharedI32Store(
                        ctx.Builder, val_section_size_i32, val_section_size_slot);

                    if (!compact_fastpath_)
                    {
                        stageTileMetadataWordsToShared(
                            ctx, tile_metadata_words, staged_tile_meta_base);
                    }
                });
            });
            ctx.Builder->CreateCall(ctx.membar_cta);
            ctx.Builder->CreateCall(ctx.barrier);
            llvm::Value* can_decompress_block = ctx.Builder->CreateICmpNE(
                createVolatileSharedI32Load(
                    ctx.Builder, ctx.i32_ty, val_section_size_slot),
                llvm::ConstantInt::get(ctx.i32_ty, 0));

            llvm::Value* tile_metadata_ptr = nullptr;
            if (compact_fastpath_)
            {
                llvm::Value* chunk_offset_shared = createVolatileSharedI32Load(
                    ctx.Builder, ctx.i32_ty, chunk_offset_slot);
                llvm::Value* chunk_offset_shared_i64 =
                    ctx.Builder->CreateZExt(chunk_offset_shared, ctx.i64_ty);
                llvm::Value* chunk_base = ctx.Builder->CreateInBoundsGEP(
                    ctx.i8_ty, state.chunk_data_base, chunk_offset_shared_i64);
                tile_metadata_ptr = ctx.Builder->CreateBitCast(
                    chunk_base, ctx.tile_metadata_ptr_ty);
            }
            else
            {
                llvm::Value* tile_metadata_i32_ptr = makeSharedI32Slot(
                    ctx.Builder, ctx.i32_ty, ctx.shared_buffer_generic_ptr,
                    staged_tile_meta_base);
                tile_metadata_ptr = ctx.Builder->CreateBitCast(
                    tile_metadata_i32_ptr, ctx.tile_metadata_ptr_ty);
            }
            return {can_decompress_block, chunk_offset_slot,
                    val_section_size_slot, tile_metadata_ptr};
        }

        void emit_decompress_tile(
            size_t block_idx, const Cb2tTileResolution& tile_resolution,
            const Cb2tDecodeTarget& decode_target, llvm::Value* val_data_ptr,
            llvm::Value* rl_data_ptr, const Cb2tEmitContext& ctx,
            const Cb2tDecodeContext& dctx) const override
        {
            auto* Builder = ctx.Builder;
            llvm::Type* i32_ty = ctx.i32_ty;
            llvm::Type* i64_ty = ctx.i64_ty;
            llvm::Value* out_ptr_i32 = decode_target.out_ptr_i32;

            if (dctx.use_predecode_registers)
            {
                std::vector<llvm::Value*> decode_args{
                    tile_resolution.tile_metadata_ptr,
                    val_data_ptr,
                    rl_data_ptr,
                    dctx.predecode_run_values_scratch_ptr,
                    dctx.predecode_run_ends_scratch_ptr};
                Builder->CreateCall(dctx.decode_predecode_setup_func, decode_args);
                llvm::Value* run_count_ptr = Builder->CreateStructGEP(
                    dctx.rle_tile_metadata_ty, tile_resolution.tile_metadata_ptr, 2);
                llvm::Value* run_count_i32 =
                    Builder->CreateLoad(i32_ty, run_count_ptr);
                for (uint32_t item = 0; item < dctx.items_per_thread; ++item)
                {
                    llvm::Value* elem_idx = Builder->CreateAdd(
                        dctx.tid_in_block,
                        llvm::ConstantInt::get(
                            i64_ty,
                            static_cast<uint64_t>(item) * dctx.block_threads));
                    llvm::Value* elem_idx_i32 =
                        Builder->CreateTrunc(elem_idx, i32_ty);
                    llvm::Value* decoded_i32 = dctx.rfor_predecode_element_inline(
                        elem_idx_i32,
                        dctx.predecode_run_values_scratch_ptr,
                        dctx.predecode_run_ends_scratch_ptr,
                        run_count_i32);
                    if (dctx.fused_predecode_tuple_execution)
                    {
                        llvm::Value* fused_item_ptr = Builder->CreateInBoundsGEP(
                            dctx.fused_tuple_items_ty,
                            (*dctx.fused_tuple_values_ptrs)[block_idx],
                            {dctx.llvm0, llvm::ConstantInt::get(i32_ty, item)});
                        Builder->CreateStore(decoded_i32, fused_item_ptr);
                    }
                    else
                    {
                        llvm::Value* out_elem_ptr = Builder->CreateInBoundsGEP(
                            i32_ty, out_ptr_i32, elem_idx);
                        Builder->CreateStore(decoded_i32, out_elem_ptr);
                    }
                }
                return;
            }

            std::vector<llvm::Value*> decode_args{
                tile_resolution.tile_metadata_ptr,
                val_data_ptr,
                rl_data_ptr,
                dctx.shared_buffer_shared_ptr,
                out_ptr_i32};
            Builder->CreateCall(dctx.decode_cooperative_func, decode_args);
        }

    private:
        bool compact_fastpath_;
    };

    class Cb2tFlbpResolver final : public ICb2tCodecResolver
    {
    public:
        Cb2tFlbpResolver(const uint32_t logical_tiles_per_chunk,
                         const uint32_t tile_elems)
            : logical_tiles_per_chunk_(logical_tiles_per_chunk),
              tile_elems_(tile_elems)
        {
        }

        [[nodiscard]] Cb2tTileResolution emit_resolve_metadata(
            size_t block_idx, const Cb2tColumnMetadataState& state,
            const Cb2tEmitContext& ctx,
            const ICb2tTileLocator& locator) const override
        {
            const uint64_t broadcast_base = static_cast<uint64_t>(
                ctx.shared_broadcast_base_elems + (2 * block_idx));
            llvm::Value* chunk_offset_slot = makeSharedI32Slot(
                ctx.Builder, ctx.i32_ty, ctx.shared_buffer_generic_ptr,
                broadcast_base);
            llvm::Value* val_section_size_slot = makeSharedI32Slot(
                ctx.Builder, ctx.i32_ty, ctx.shared_buffer_generic_ptr,
                broadcast_base + 1);
            const uint64_t staged_tile_meta_base = static_cast<uint64_t>(
                ctx.shared_broadcast_base_elems + ctx.shared_chunk_meta_elems +
                (6 * block_idx));

            llvm::Value* logical_tiles_per_chunk_i64 =
                llvm::ConstantInt::get(ctx.i64_ty, logical_tiles_per_chunk_);
            llvm::Value* tile_elems_i32 =
                llvm::ConstantInt::get(ctx.i32_ty, tile_elems_);
            llvm::Value* vector_id = ctx.Builder->CreateUDiv(
                ctx.tile_id, logical_tiles_per_chunk_i64);
            llvm::Value* vector_id_i32 =
                ctx.Builder->CreateTrunc(vector_id, ctx.i32_ty);
            llvm::Value* subtile_idx_i32 = ctx.Builder->CreateTrunc(
                ctx.Builder->CreateURem(ctx.tile_id, logical_tiles_per_chunk_i64),
                ctx.i32_ty);

            llvm::Value* in_range = ctx.Builder->CreateICmpULT(
                vector_id, state.physical_num_chunks_i64);
            llvm::Value* can_decompress_thread = ctx.Builder->CreateAnd(
                in_range, locator.emitSourceOk(ctx, state));

            ctx.context->gen_if(
                {ProteusValue{ctx.is_block_leader,
                              ctx.context->createFalse()}})([&]()
            {
                createVolatileSharedI32Store(
                    ctx.Builder, llvm::ConstantInt::get(ctx.i32_ty, 0),
                    chunk_offset_slot);
                createVolatileSharedI32Store(
                    ctx.Builder, llvm::ConstantInt::get(ctx.i32_ty, 0),
                    val_section_size_slot);

                ctx.context->gen_if(
                    {ProteusValue{can_decompress_thread,
                                  ctx.context->createFalse()}})([&]()
                {
                    llvm::Value* chunk_offset_i32 = locator.emitChunkOffset(
                        ctx, state, vector_id, vector_id_i32, block_idx);
                    createVolatileSharedI32Store(
                        ctx.Builder, chunk_offset_i32, chunk_offset_slot);
                    // FLBP val_section_size == sizeof(FLBPVectorMetadata)
                    createVolatileSharedI32Store(
                        ctx.Builder, ctx.chunk_metadata_bytes_i32,
                        val_section_size_slot);

                    llvm::Value* chunk_offset_i64 =
                        ctx.Builder->CreateZExt(chunk_offset_i32, ctx.i64_ty);
                    llvm::Value* chunk_base = ctx.Builder->CreateInBoundsGEP(
                        ctx.i8_ty, state.chunk_data_base, chunk_offset_i64);
                    llvm::Value* vector_metadata_ptr =
                        ctx.Builder->CreateBitCast(
                            chunk_base,
                            ctx.flbp_vector_metadata_ty->getPointerTo());
                    llvm::Value* base_value_ptr = ctx.Builder->CreateStructGEP(
                        ctx.flbp_vector_metadata_ty, vector_metadata_ptr, 0);
                    llvm::Value* bitwidth_ptr = ctx.Builder->CreateStructGEP(
                        ctx.flbp_vector_metadata_ty, vector_metadata_ptr, 1);
                    llvm::Value* element_count_ptr = ctx.Builder->CreateStructGEP(
                        ctx.flbp_vector_metadata_ty, vector_metadata_ptr, 2);
                    llvm::Value* base_value_i32 = ctx.Builder->CreateLoad(
                        ctx.i32_ty, base_value_ptr);
                    llvm::Value* bitwidth_i8 = ctx.Builder->CreateLoad(
                        llvm::Type::getInt8Ty(ctx.Builder->getContext()),
                        bitwidth_ptr);
                    llvm::Value* bitwidth_i32 =
                        ctx.Builder->CreateZExt(bitwidth_i8, ctx.i32_ty);
                    llvm::Value* element_count_i32 = ctx.Builder->CreateLoad(
                        ctx.i32_ty, element_count_ptr);
                    llvm::Value* subtile_offset_i32 = ctx.Builder->CreateMul(
                        subtile_idx_i32, tile_elems_i32);
                    stageFlbpTileMetadataToShared(
                        ctx, staged_tile_meta_base, base_value_i32, bitwidth_i32,
                        element_count_i32, subtile_offset_i32, tile_elems_i32);
                });
            });
            ctx.Builder->CreateCall(ctx.membar_cta);
            ctx.Builder->CreateCall(ctx.barrier);

            llvm::Value* can_decompress_block = ctx.Builder->CreateICmpNE(
                createVolatileSharedI32Load(
                    ctx.Builder, ctx.i32_ty, val_section_size_slot),
                llvm::ConstantInt::get(ctx.i32_ty, 0));
            llvm::Value* tile_metadata_i32_ptr = makeSharedI32Slot(
                ctx.Builder, ctx.i32_ty, ctx.shared_buffer_generic_ptr,
                staged_tile_meta_base);
            llvm::Value* tile_metadata_ptr = ctx.Builder->CreateBitCast(
                tile_metadata_i32_ptr, ctx.tile_metadata_ptr_ty);
            return {can_decompress_block, chunk_offset_slot,
                    val_section_size_slot, tile_metadata_ptr};
        }

        void emit_decompress_tile(
            size_t block_idx, const Cb2tTileResolution& tile_resolution,
            const Cb2tDecodeTarget& decode_target, llvm::Value* val_data_ptr,
            llvm::Value* rl_data_ptr, const Cb2tEmitContext& ctx,
            const Cb2tDecodeContext& dctx) const override
        {
            auto* Builder = ctx.Builder;
            llvm::Type* i32_ty = ctx.i32_ty;
            llvm::Type* i64_ty = ctx.i64_ty;
            llvm::Type* i32_ptr_ty = dctx.i32_ptr_ty;
            llvm::Value* out_ptr_i32 = decode_target.out_ptr_i32;

            if (dctx.use_predecode_registers)
            {
                llvm::Value* flbp_tile_metadata_words = Builder->CreateBitCast(
                    tile_resolution.tile_metadata_ptr, i32_ptr_ty);
                for (uint32_t item = 0; item < dctx.items_per_thread; ++item)
                {
                    llvm::Value* elem_idx = Builder->CreateAdd(
                        dctx.tid_in_block,
                        llvm::ConstantInt::get(
                            i64_ty,
                            static_cast<uint64_t>(item) * dctx.block_threads));
                    llvm::Value* elem_idx_i32 =
                        Builder->CreateTrunc(elem_idx, i32_ty);
                    llvm::Value* decoded_i32 = Builder->CreateCall(
                        dctx.decode_predecode_element_func,
                        {flbp_tile_metadata_words, val_data_ptr, elem_idx_i32});
                    if (dctx.fused_predecode_tuple_execution)
                    {
                        llvm::Value* fused_item_ptr = Builder->CreateInBoundsGEP(
                            dctx.fused_tuple_items_ty,
                            (*dctx.fused_tuple_values_ptrs)[block_idx],
                            {dctx.llvm0, llvm::ConstantInt::get(i32_ty, item)});
                        Builder->CreateStore(decoded_i32, fused_item_ptr);
                    }
                    else
                    {
                        llvm::Value* out_elem_ptr = Builder->CreateInBoundsGEP(
                            i32_ty, out_ptr_i32, elem_idx);
                        Builder->CreateStore(decoded_i32, out_elem_ptr);
                    }
                }
                return;
            }

            std::vector<llvm::Value*> decode_args{
                Builder->CreateBitCast(tile_resolution.tile_metadata_ptr,
                                       i32_ptr_ty),
                val_data_ptr,
                rl_data_ptr,
                dctx.shared_buffer_shared_ptr,
                out_ptr_i32};
            Builder->CreateCall(dctx.decode_cooperative_func, decode_args);
        }

    private:
        uint32_t logical_tiles_per_chunk_;
        uint32_t tile_elems_;
    };

    std::unique_ptr<ICb2tTileLocator> makeTileLocator(
        const bool use_precomputed_tile_offsets)
    {
        if (use_precomputed_tile_offsets)
        {
            return std::make_unique<Cb2tPrecomputedLocator>();
        }
        return std::make_unique<Cb2tInBlockLocator>();
    }

    std::unique_ptr<ICb2tCodecResolver> makeCodecResolver(
        const Cb2tPlan& cb2t_plan, const bool use_precomputed_tile_offsets,
        const size_t tile_elems)
    {
        if (cb2t_plan.codec_kind == Cb2tCodecKind::kRfor)
        {
            const bool compact = use_precomputed_tile_offsets &&
                                 cb2t_plan.compact_precomputed_fastpath;
            return std::make_unique<Cb2tRforResolver>(compact);
        }

        CHECK(cb2t_plan.codec_kind == Cb2tCodecKind::kFlbp)
            << "[CB2T] expected FLBP codec path";
        return std::make_unique<Cb2tFlbpResolver>(
            cb2t_plan.logical_tiles_per_chunk,
            static_cast<uint32_t>(tile_elems));
    }

    Cb2tPrecomputedOffsetMaps buildPrecomputedOffsetMapsPlan(
        OlapParallelContext* context, llvm::IRBuilder<>* Builder, Plugin* pg,
        const std::vector<expression_t>& wantedFields, const size_t block_field_count,
        llvm::PointerType* i32_ptr_ptr_ty)
    {
        Cb2tPrecomputedOffsetMaps plan{};
        if (auto* nvme_pg = dynamic_cast<NvmePlugin*>(pg);
            nvme_pg && nvme_pg->scanEmitsCompressedBlocks() &&
            FLAGS_precomputed_offsets)
        {
            plan.use_precomputed_tile_offsets = true;
            uint8_t block_attr_no = 0;
            for (const auto& field : wantedFields)
            {
                if (!dynamic_cast<const BlockType*>(field.getExpressionType()))
                {
                    continue;
                }

                void* chunk_offsets_map =
                    nvme_pg->getPreloadedCompressedChunkOffsetsOidMapPtr(block_attr_no);
                if (!chunk_offsets_map)
                {
                    plan.use_precomputed_tile_offsets = false;
                    break;
                }

                auto chunk_offsets_addr = llvm::ConstantInt::get(
                    context->createSizeType(),
                    reinterpret_cast<uintptr_t>(chunk_offsets_map));
                plan.chunk_offsets_oid_maps.push_back(
                    Builder->CreateIntToPtr(chunk_offsets_addr, i32_ptr_ptr_ty));
                ++block_attr_no;
            }

            if (plan.chunk_offsets_oid_maps.size() != block_field_count)
            {
                plan.use_precomputed_tile_offsets = false;
            }
        }
        if (!plan.use_precomputed_tile_offsets)
        {
            plan.chunk_offsets_oid_maps.clear();
        }
        return plan;
    }

    unsigned int ctt_grid_cap()
    {
        if (FLAGS_max_grid_blocks != 0u)
        {
            return FLAGS_max_grid_blocks;
        }
        const int sm_count =
            topology::getInstance().getActiveGpu().properties.multiProcessorCount;
        CHECK_GT(sm_count, 0) << "active GPU reports no SMs";
        CHECK_LE(static_cast<uint64_t>(sm_count) * kCb2tGridBlocksPerSm,
                 std::numeric_limits<unsigned int>::max());
        return static_cast<unsigned int>(sm_count) * kCb2tGridBlocksPerSm;
    }
} // namespace

using namespace llvm;

// This operator is the GPU-side "decompression + unpack" path for late decode
// modes (--decom_op / --decom_reg). It consumes compressed blocks, decodes
// codec tiles/vectors (RFOR or FLBP), and exposes per-tuple values to the rest
// of the query.
//
// High-level flow (aligned with the student report):
// - A block corresponds to a 2MB row group (uncompressed).
// - A tile is backend block_threads * items_per_thread elements.
// - RFOR uses LoadRBinPack_notemplate_overlap.
// - FLBP uses LoadFLBP_notemplate_overlap and maps one 1024-vector to one or
//   more logical CB2T tiles depending on runtime tile size.
// - Decompressed tiles live in shared memory and are read immediately by the
//   following operators, or optionally materialized to global memory in the
//   LEGACY_GLOBAL_TILE_WINDOW specialization.
//
// NOTE: We build the shared memory buffer as a static addrspace(3) global and
// cast it to a generic pointer only at call boundaries, which mirrors NVVM
// semantics and keeps the IR easier to reason about.

template <bool LEGACY_GLOBAL_TILE_WINDOW, bool REGISTER_FUSED,
          bool GLOBAL_STAGE>
void CompressedBlockToTuples<LEGACY_GLOBAL_TILE_WINDOW, REGISTER_FUSED,
                             GLOBAL_STAGE>::produce_(
    OlapParallelContext* context)
{
    if (gpu)
    {
        const auto& decode_backend = getCb2tDecodeBackend();
        const auto runtime_config = getSelectedCb2tRuntimeConfig(decode_backend);
        (*context)->setMaxWorkerSize(
            static_cast<size_t>(runtime_config.block_threads),
            ctt_grid_cap());
    }
    context->registerOpen(this, [this](Pipeline* pip) { this->open(pip); });
    context->registerClose(this, [this](Pipeline* pip) { this->close(pip); });

    for (const auto& wantedField : wantedFields)
    {
        if (dynamic_cast<const BlockType*>(wantedField.getExpressionType()))
        {
            old_buffs.push_back(context->appendStateVar(llvm::PointerType::getUnqual(
                RecordAttribute{wantedField.getRegisteredAs()}.getLLVMType(
                    context->getLLVMContext()))));
        }
    }

    CHECK(!old_buffs.empty())
      << "There should be at least one BlockType'd argument in wantedFields";

    release_input_buffers_ = true;
    for (const auto& wantedField : wantedFields)
    {
        if (!dynamic_cast<const BlockType*>(wantedField.getExpressionType()))
        {
            continue;
        }
        auto* pg =
            Catalog::getInstance().getPlugin(wantedField.getRegisteredRelName());
        if (auto* nvme_pg = dynamic_cast<NvmePlugin*>(pg))
        {
            release_input_buffers_ = nvme_pg->unpackShouldReleaseInputBuffers();
        }
        break;
    }

    getChild()->produce(context);
}

template <bool LEGACY_GLOBAL_TILE_WINDOW, bool REGISTER_FUSED,
          bool GLOBAL_STAGE>
ProteusBareValue
CompressedBlockToTuples<LEGACY_GLOBAL_TILE_WINDOW, REGISTER_FUSED,
                        GLOBAL_STAGE>::step(
    OlapParallelContext* context, llvm::IntegerType* type)
{
    auto Builder = context->getBuilder();
    Value* inc;
    if (gpu && granularity == gran_t::GRID)
    {
        inc = Builder->CreateIntCast(context->threadNum(), type, false);
    }
    else
    {
        inc = ConstantInt::get(type, 1);
    }
    return {inc};
}

template <bool LEGACY_GLOBAL_TILE_WINDOW, bool REGISTER_FUSED,
          bool GLOBAL_STAGE>
void CompressedBlockToTuples<LEGACY_GLOBAL_TILE_WINDOW, REGISTER_FUSED,
                             GLOBAL_STAGE>::nextEntry(
    llvm::Value* mem_itemCtr, OlapParallelContext* context)
{
    // Prepare
    auto Builder = context->getBuilder();

    // Increment and store back

    Value* val_curr_itemCtr = Builder->CreateLoad(
        mem_itemCtr->getType()->getNonOpaquePointerElementType(), mem_itemCtr);

    Value* inc = step(context, (IntegerType*)val_curr_itemCtr->getType()).value;

    Value* val_new_itemCtr = Builder->CreateAdd(val_curr_itemCtr, inc);
    Builder->CreateStore(val_new_itemCtr, mem_itemCtr);
}

template <bool LEGACY_GLOBAL_TILE_WINDOW, bool REGISTER_FUSED,
          bool GLOBAL_STAGE>
// consume() emits the fused decompression+execution kernel body. All
// codec-specific logic lives behind the two backend hooks (emit_resolve_metadata,
// emit_decompress_tile on ICb2tCodecResolver); what remains here is
// codec-agnostic scaffolding, in phases:
//   1. ABI/type + shared-memory-layout setup and decode-symbol binding;
//   2. per-column static state hoisted out of the tile loop;
//   3. emit + decode contexts and the codec backend / tile locator selection;
//   4. the fused two-level loop (resolve_metadata -> decompress_tile -> barrier
//      -> inner tuple loop -> barrier) -- the paper's operator realization.
void CompressedBlockToTuples<LEGACY_GLOBAL_TILE_WINDOW, REGISTER_FUSED,
                             GLOBAL_STAGE>::consume(
    OlapParallelContext* context, const OperatorState& childState)
{
    const auto& decode_backend = getCb2tDecodeBackend();
    const auto runtime_config = getSelectedCb2tRuntimeConfig(decode_backend);
    (*context)->setMaxWorkerSize(
        static_cast<size_t>(runtime_config.block_threads), ctt_grid_cap());
    // Prepare

    LLVMContext& llvmContext = context->getLLVMContext();
    Module* module = context->getModule();
    IRBuilder<>* Builder = context->getBuilder();

    // TODO move into context
    expressions::ProteusValueExpression tid{
        new Int64Type(), {context->threadId(), context->createFalse()}
    };

    auto is_leader = eq(tid, INT64_C(0));

    // FIXME: do we need that?
    // Value *activemask;
    // if (dynamic_cast<GpuPipelineGen *>(context->getCurrentPipeline())) {
    //   activemask = gpu_intrinsic::activemask(context);
    // }

    // FIXME: assumes thread 0 gets to execute block2tuples
    gen_if(is_leader, childState, context)([&]()
    {
        bool printed_magic = true;
        size_t j = 0;
        for (const auto& wantedField : wantedFields)
        {
            if (!dynamic_cast<const BlockType*>(wantedField.getExpressionType()))
            {
                continue;
            }
            RecordAttribute attr{wantedField.getRegisteredAs()};
            Value* arg = Builder->CreateLoad(
                childState[attr].mem->getType()->getNonOpaquePointerElementType(),
                childState[attr].mem);


            if (!printed_magic)
            {
                // Read first 4 bytes of the block and print as int (magic).
                auto i32_ty = Type::getInt32Ty(llvmContext);
                auto i32_ptr_ty = PointerType::get(i32_ty, 0);
                Value* magic_ptr = Builder->CreateBitCast(arg, i32_ptr_ty);
                Value* magic_val = Builder->CreateLoad(i32_ty, magic_ptr);
                Function* printi = context->getFunction("printi");
                Builder->CreateCall(printi, {magic_val});
                printed_magic = true;
            }


            Value* old = Builder->CreateLoad(context->getStateVar(old_buffs[j])
                                                    ->getType()
                                                    ->getNonOpaquePointerElementType(),
                                             context->getStateVar(old_buffs[j]));
            if (release_input_buffers_)
            {
                Type* charPtrType = Type::getInt8PtrTy(llvmContext);
                old = Builder->CreateBitCast(old, charPtrType);

                // Release previous GPU buffers for this column. With --decom_op the
                // compressed block stays live until now, then we advance to the new one.
                Function* f = context->getFunction(
                    "release_buffers"); // FIXME: Assumes grid launch +
                // Assumes 1 block per kernel!
                Builder->CreateCall(f, {old});
            }

            Builder->CreateStore(arg, context->getStateVar(old_buffs[j]));

            ++j;
        }
        assert(j == old_buffs.size());
    });

    // FIXME: do we need that?
    // if (dynamic_cast<GpuPipelineGen *>(context->getCurrentPipeline())) {
    //   auto warpsync = context->getFunction("llvm.nvvm.bar.warp.sync");
    //   Builder->CreateCall(warpsync, {activemask});
    // }

    auto relName = [&]()
    {
        if (wantedFields.empty())
        {
            for (const auto& t : childState.getBindings())
            {
                if (t.first.getAttrName() == "activeCnt")
                {
                    return t.first.getRelationName();
                }
            }
            LOG(FATAL) << "cannot determine relation name";
        }
        else
        {
            return wantedFields[0].getRegisteredRelName();
        }
    }();

    Plugin* pg = Catalog::getInstance().getPlugin(relName);

    RecordAttribute tupleCnt{
        relName, "activeCnt",
        pg->getOIDType()
    }; // FIXME: OID type for blocks ?
    Value* cnt_raw = Builder->CreateLoad(
        childState[tupleCnt].mem->getType()->getNonOpaquePointerElementType(),
        childState[tupleCnt].mem, "cnt");

    size_t block_field_count = 0;
    for (const auto& field : wantedFields)
    {
        if (dynamic_cast<const BlockType*>(field.getExpressionType()))
        {
            ++block_field_count;
        }
    }
    unsigned int ctt_grid_blocks = ctt_grid_cap();
    constexpr auto execution_mode =
        selectExecutionMode<LEGACY_GLOBAL_TILE_WINDOW, REGISTER_FUSED,
                            GLOBAL_STAGE>();
    constexpr bool materialize_tiles_to_global =
        execution_mode == Cb2tExecutionMode::kLegacyGlobalTileWindow;
    constexpr bool use_predecode_registers =
        execution_mode == Cb2tExecutionMode::kRegisters;
    constexpr bool use_streaming_global =
        execution_mode == Cb2tExecutionMode::kGlobalStage;
    const uint32_t streaming_grid_rows =
        use_streaming_global ? std::max(1u, FLAGS_streaming_grid_rows) : 0u;
    const uint64_t streaming_tile_slots_u64 =
        static_cast<uint64_t>(ctt_grid_blocks) * streaming_grid_rows;
    CHECK_LE(streaming_tile_slots_u64,
             static_cast<uint64_t>(kMaxStreamingTileSlots))
        << "--streaming_grid_rows * CB2T grid_x exceeds the bounded staging "
           "slot limit";
    const uint32_t streaming_tile_slots =
        static_cast<uint32_t>(streaming_tile_slots_u64);
    const uint32_t materialize_tile_slots_requested = kMaterializeTileSlotsDefault;
    // Use a slot count that is at least grid size and a multiple of grid size.
    // This keeps materialize slots block-disjoint under grid-stride tile_id.
    uint32_t materialize_tile_slots =
        materialize_tile_slots_requested < ctt_grid_blocks
            ? ctt_grid_blocks
            : materialize_tile_slots_requested;
    if (materialize_tile_slots % ctt_grid_blocks != 0u)
    {
        materialize_tile_slots +=
            (ctt_grid_blocks - (materialize_tile_slots % ctt_grid_blocks));
    }
    CHECK_GT(runtime_config.block_threads, 0);
    CHECK_GT(runtime_config.items_per_thread, 0);
    CHECK_EQ(
        runtime_config.tile_size,
        runtime_config.block_threads * runtime_config.items_per_thread);
    // Decom_reg fused mode:
    // decode runs into reusable shared scratch and keep decoded tuple values in
    // per-thread slots consumed immediately by parent operators.
    constexpr bool fused_predecode_tuple_execution = use_predecode_registers;
    const bool predecode_uses_run_scratch =
        use_predecode_registers && decode_backend.codecKind() == Cb2tCodecKind::kRfor;
    const auto decode_kernel_spec = decode_backend.kernelSpec(runtime_config);
    if (use_predecode_registers && !decode_backend.supportsRegisterPredecode())
    {
        LOG(FATAL) << "[CB2T] selected decode backend does not support "
                      "register predecode mode";
    }
    const bool use_precomputed_cumulative_rl =
        use_predecode_registers && FLAGS_precompute_rl;
    // Compact precomputed fastpath:
    // - requested for decom_reg + precomputed cumulative RL inputs
    // - bypasses shared broadcast metadata staging in the hot decode path
    //
    // NOTE:
    // The compact RFOR path is currently unstable under one-kernel launches
    // (default --one_kernel=true), where it can trigger device illegal-memory
    // faults. In that configuration we fall back to the stable fused-register
    // path, which is still the high-performance decom_reg mode.
    const bool compact_precomputed_fastpath_requested =
        use_predecode_registers && use_precomputed_cumulative_rl &&
        FLAGS_precomputed_offsets;
    const bool compact_precomputed_fastpath =
        compact_precomputed_fastpath_requested && !FLAGS_one_kernel;
    if (compact_precomputed_fastpath_requested && !compact_precomputed_fastpath)
    {
        LOG_FIRST_N(WARNING, 1)
            << "[CB2T] compact decom_reg precomputed fastpath is disabled when "
               "--one_kernel=true due a known correctness issue; using stable "
               "fused-register decode path instead.";
    }
    const size_t tile_elems = static_cast<size_t>(runtime_config.tile_size);
    const uint32_t logical_tiles_per_chunk =
        decode_backend.logicalTilesPerChunk(static_cast<uint32_t>(tile_elems));
    CHECK_GT(logical_tiles_per_chunk, 0u);
    const Cb2tPlan cb2t_plan{
        decode_backend.codecKind(),
        runtime_config,
        logical_tiles_per_chunk,
        buildCb2tSharedLayout(
            block_field_count,
            tile_elems,
            use_predecode_registers,
            use_streaming_global,
            predecode_uses_run_scratch,
            compact_precomputed_fastpath,
            decode_kernel_spec.scratch_elems),
        compact_precomputed_fastpath
    };
    (*context)->setMaxWorkerSize(
        static_cast<size_t>(cb2t_plan.runtime_config.block_threads), ctt_grid_blocks);
    // One tile = runtime block_threads * runtime items_per_thread elements.
    CHECK_EQ(
        tile_elems,
        static_cast<size_t>(
            cb2t_plan.runtime_config.block_threads *
            cb2t_plan.runtime_config.items_per_thread));
    const uint32_t runtime_block_threads =
        static_cast<uint32_t>(cb2t_plan.runtime_config.block_threads);
    const uint32_t runtime_items_per_thread =
        static_cast<uint32_t>(cb2t_plan.runtime_config.items_per_thread);
    // Shared memory layout:
    // Default:
    // [decoder scratch | col0 tile | col1 tile | ... | broadcast metadata]
    //
    // Streaming-global mode:
    // [decoder scratch | one reusable decoded tile | broadcast metadata]
    //
    // Predecode-registers mode:
    // [col0 decoded tile | ... | colN decoded tile
    //  | predecode run_values scratch (1 tile)
    //  | predecode run_ends scratch   (1 tile)
    //  | broadcast metadata]
    // Broadcast metadata stores:
    //  - (chunk_offset, val_section_size) per block field
    //  - staged RFOR tile metadata (6x i32) per block field
    // Decom_reg predecode setup does not use cooperative decode scratch.
    const size_t shared_decode_scratch_elems =
        cb2t_plan.shared_layout.decode_scratch_elems;
    const size_t shared_column_tile_elems =
        cb2t_plan.shared_layout.column_tile_elems;
    const size_t shared_out_base_elems = cb2t_plan.shared_layout.out_base_elems;
    // In decom_reg mode, reuse a single run-value/run-end scratch pair for all
    // columns and materialize decoded tuples into shared out tiles.
    const size_t shared_run_values_base_elems =
        cb2t_plan.shared_layout.run_values_base_elems;
    const size_t shared_run_ends_base_elems =
        cb2t_plan.shared_layout.run_ends_base_elems;
    // Keep per-column broadcast slots for compressed-block pointer publication
    // block-uniform, even in compact precomputed mode.
    const size_t shared_chunk_meta_elems = cb2t_plan.shared_layout.chunk_meta_elems;
    const size_t shared_tile_meta_elems =
        cb2t_plan.shared_layout.tile_meta_elems;
    // Decode validity is encoded by val_section_size_slot != 0, so no extra
    // decode-flag slot is needed.
    const size_t shared_broadcast_elems = cb2t_plan.shared_layout.broadcast_elems;
    const size_t shared_broadcast_base_elems =
        cb2t_plan.shared_layout.broadcast_base_elems;
    const size_t shared_block_uniform_base_elems =
        cb2t_plan.shared_layout.block_uniform_base_elems;
    const size_t shared_total_elems = cb2t_plan.shared_layout.total_elems;

    // Create decode ABI types.
    auto void_ty = Type::getVoidTy(llvmContext);
    auto i32_ty = Type::getInt32Ty(llvmContext);
    auto i8_ty = Type::getInt8Ty(llvmContext);
    auto i32_ptr_ty = PointerType::get(i32_ty, 0);
    auto i32_global_ptr_ty = PointerType::get(i32_ty, 1);
    auto i32_shared_ptr_ty = PointerType::get(i32_ty, 3);
    auto i32_ptr_ptr_ty = PointerType::get(i32_ptr_ty, 0);
    auto i32_scratch_array_ty = ArrayType::get(i32_ty, shared_total_elems);
    auto RLETileMetadata_ty = StructType::create(
        llvmContext,
        {i32_ty, i32_ty, i32_ty, i32_ty, i32_ty, i32_ty}
    );
    auto RLETileMetadata_ptr_ty = PointerType::get(RLETileMetadata_ty, 0);
    auto FLBPTileMetadata_ty = StructType::create(
        llvmContext,
        {i32_ty, i32_ty, i32_ty, i32_ty, i32_ty, i32_ty}
    );
    auto FLBPTileMetadata_ptr_ty = PointerType::get(FLBPTileMetadata_ty, 0);
    auto FLBPVectorMetadata_ty = StructType::create(
        llvmContext,
        {i32_ty, i8_ty, i32_ty}
    );
    auto FLBPVectorMetadata_ptr_ty = PointerType::get(FLBPVectorMetadata_ty, 0);
    auto streaming_load_cg = InlineAsm::get(
        FunctionType::get(i32_ty, {i32_global_ptr_ty}, false),
        "ld.global.cg.u32 $0, [$1];", "=r,l", true);
    auto streaming_store_cg = InlineAsm::get(
        FunctionType::get(void_ty, {i32_global_ptr_ty, i32_ty}, false),
        "st.global.cg.u32 [$0], $1;", "l,r", true);
    const bool use_rfor_decode_abi =
        decode_kernel_spec.abi_kind == Cb2tDecodeAbiKind::kRforTileMetadata;
    llvm::StructType* active_tile_metadata_ty =
        use_rfor_decode_abi ? RLETileMetadata_ty : FLBPTileMetadata_ty;
    llvm::PointerType* active_tile_metadata_ptr_ty =
        use_rfor_decode_abi ? RLETileMetadata_ptr_ty : FLBPTileMetadata_ptr_ty;
    auto RforDecodeFn_ty = FunctionType::get(
        void_ty,
        {RLETileMetadata_ptr_ty, i32_ptr_ty, i32_ptr_ty, i32_shared_ptr_ty,
         i32_shared_ptr_ty},
        false
    );
    auto FlbpDecodeFn_ty = FunctionType::get(
        void_ty,
        {i32_ptr_ty, i32_ptr_ty, i32_ptr_ty, i32_shared_ptr_ty, i32_shared_ptr_ty},
        false
    );
    auto FlbpPredecodeElementFn_ty =
        FunctionType::get(i32_ty, {i32_ptr_ty, i32_ptr_ty, i32_ty}, false);

    // Decoder kernels are compiled from C++ (__device__). Keep output
    // tile pointer in shared address space to avoid generic-pointer ambiguity
    // on decode stores.
    FunctionCallee decode_cooperative_func;
    if (decode_kernel_spec.abi_kind == Cb2tDecodeAbiKind::kRforTileMetadata)
    {
        decode_cooperative_func = module->getOrInsertFunction(
            decode_kernel_spec.cooperative_symbol, RforDecodeFn_ty);
    }
    else
    {
        CHECK(
            decode_kernel_spec.abi_kind ==
            Cb2tDecodeAbiKind::kFlbpTileMetadataWords)
            << "[CB2T] expected FLBP decode ABI for non-RFOR backend";
        decode_cooperative_func = module->getOrInsertFunction(
            decode_kernel_spec.cooperative_symbol, FlbpDecodeFn_ty);
    }
    FunctionCallee decode_predecode_setup_func;
    FunctionCallee decode_predecode_element_func;
    if (use_predecode_registers)
    {
        if (decode_kernel_spec.abi_kind == Cb2tDecodeAbiKind::kRforTileMetadata)
        {
            CHECK(decode_kernel_spec.predecode_symbol != nullptr);
            CHECK(decode_kernel_spec.predecode_cumulative_symbol != nullptr);
            decode_predecode_setup_func = module->getOrInsertFunction(
                use_precomputed_cumulative_rl
                    ? decode_kernel_spec.predecode_cumulative_symbol
                    : decode_kernel_spec.predecode_symbol,
                RforDecodeFn_ty);
        }
        else
        {
            CHECK(
                decode_kernel_spec.abi_kind ==
                Cb2tDecodeAbiKind::kFlbpTileMetadataWords)
                << "[CB2T] expected FLBP decode ABI for FLBP register mode";
            CHECK(decode_kernel_spec.predecode_element_symbol != nullptr);
            decode_predecode_element_func = module->getOrInsertFunction(
                decode_kernel_spec.predecode_element_symbol,
                FlbpPredecodeElementFn_ty);
        }
    }

    auto precomputed_offset_maps = buildPrecomputedOffsetMapsPlan(
        context, Builder, pg, wantedFields, block_field_count, i32_ptr_ptr_ty);
    auto chunk_offsets_oid_maps =
        std::move(precomputed_offset_maps.chunk_offsets_oid_maps);
    const bool use_precomputed_tile_offsets =
        precomputed_offset_maps.use_precomputed_tile_offsets;
    if (compact_precomputed_fastpath && !use_precomputed_tile_offsets)
    {
        LOG(FATAL)
            << "[CB2T] compact decom_reg precomputed fastpath requires "
               "precomputed tile offsets/maps, but they were not available.";
    }
    LOG(INFO) << "[CB2T] rel=" << relName
              << " block_fields=" << block_field_count
              << " shared_chunk_meta_elems=" << shared_chunk_meta_elems
              << " shared_tile_meta_elems=" << shared_tile_meta_elems
              << " shared_broadcast_base_elems=" << shared_broadcast_base_elems
              << " shared_broadcast_elems=" << shared_broadcast_elems
              << " chunk_metadata_bytes=" << decode_kernel_spec.chunk_metadata_bytes
              << " mode=" << cb2tExecutionModeName(execution_mode)
              << " codec=" << decode_backend.codecName()
              << " threads_per_block_override=" << FLAGS_threads_per_block
              << " items_per_thread_override=" << FLAGS_items_per_thread
              << " logical_tiles_per_chunk=" << cb2t_plan.logical_tiles_per_chunk
              << " cb2t_block_threads=" << cb2t_plan.runtime_config.block_threads
              << " cb2t_items_per_thread=" << cb2t_plan.runtime_config.items_per_thread
              << " cb2t_tile_size=" << cb2t_plan.runtime_config.tile_size
              << " cb2t_grid_cap=" << ctt_grid_blocks
              << " cb2t_cooperative_decode="
              << (decode_backend.supportsCooperativeDecode() ? "true" : "false")
              << " materialize_tile_slots_requested="
              << materialize_tile_slots_requested
              << " materialize_tile_slots=" << materialize_tile_slots
              << " streaming_tile_slots="
              << streaming_tile_slots
              << " streaming_grid_rows=" << streaming_grid_rows
              << " streaming_staging_bytes="
              << (use_streaming_global
                      ? static_cast<size_t>(streaming_tile_slots) *
                            block_field_count * tile_elems * sizeof(uint32_t)
                      : 0u)
              << " cb2t_predecode_registers="
              << (use_predecode_registers ? "true" : "false")
              << " cb2t_register_decode_mode="
              << (use_predecode_registers ? (compact_precomputed_fastpath
                                                 ? "predecoded_runs_reuse_scratch_compact"
                                                 : (fused_predecode_tuple_execution
                                                        ? "predecoded_runs_fused_tuple"
                                                        : "predecoded_runs_reuse_scratch"))
                                          : "shared_tile")
              << " cb2t_runs_precomputed_cumulative="
              << (use_precomputed_cumulative_rl ? "true" : "false")
              << " emit_tuple_oid_bindings="
              << (FLAGS_emit_tuple_oid_bindings ? "true" : "false")
              << " use_precomputed_tile_offsets="
              << (use_precomputed_tile_offsets ? "true" : "false");

    // Create values for LoadRBinPack arguments
    auto undefined_init = UndefValue::get(i32_scratch_array_ty);
    auto llvm0 = ConstantInt::get(i32_ty, 0);
    // Allocate a statically-sized shared memory array (addrspace(3)):
    //   @shared_buffer = internal addrspace(3) global [N x i32] undef
    // TODO: dynamic shared memory would use an external addrspace(3) global
    // of size 0 and pass the required shared bytes at kernel launch.
    auto shared_buffer_GV = new GlobalVariable(
        *module,
        i32_scratch_array_ty,
        false,
        GlobalValue::InternalLinkage,
        undefined_init,
        "shared_buffer_GV",
        nullptr,
        GlobalValue::NotThreadLocal,
        3 //addrspace(3)
    );
    auto barrier = Intrinsic::getDeclaration(module, Intrinsic::nvvm_barrier0);
    auto membar_cta =
        Intrinsic::getDeclaration(module, Intrinsic::nvvm_membar_cta);
    auto storeSharedBroadcastI32 = [&](Value* value, Value* slot) {
        return createVolatileSharedI32Store(Builder, value, slot);
    };
    auto loadSharedBroadcastI32 = [&](Value* slot) {
        return createVolatileSharedI32Load(Builder, i32_ty, slot);
    };

    auto shared_buffer_shared_ptr = Builder->CreateInBoundsGEP(
        shared_buffer_GV->getValueType(),
        shared_buffer_GV,
        {llvm0, llvm0},
        "shared_buffer_shared"
    );
    // Keep a generic alias for existing helper metadata/broadcast accesses and
    // decoder scratch argument.
    auto shared_buffer_generic_ptr = Builder->CreateAddrSpaceCast(
        shared_buffer_shared_ptr, i32_ptr_ty, "shared_buffer_generic"
    );
    // Base of the per-column output tiles in shared memory.
    auto out_base_ptr = Builder->CreateInBoundsGEP(
        i32_ty, shared_buffer_shared_ptr,
        ConstantInt::get(i32_ty, shared_out_base_elems), "shared_out_base");
    Value* predecode_run_values_scratch_ptr = nullptr;
    Value* predecode_run_ends_scratch_ptr = nullptr;
    if (predecode_uses_run_scratch)
    {
        predecode_run_values_scratch_ptr = Builder->CreateInBoundsGEP(
            i32_ty, shared_buffer_shared_ptr,
            ConstantInt::get(i32_ty, shared_run_values_base_elems),
            "shared_predecode_run_values");
        predecode_run_ends_scratch_ptr = Builder->CreateInBoundsGEP(
            i32_ty, shared_buffer_shared_ptr,
            ConstantInt::get(i32_ty, shared_run_ends_base_elems),
            "shared_predecode_run_ends");
    }
    // Optional per-CTA global output tiles (addrspace(1)) used by the
    // LEGACY_GLOBAL_TILE_WINDOW specialization.
    Value* global_buffer_generic_ptr = nullptr;
    if (materialize_tiles_to_global)
    {
        const size_t global_tile_elems =
            static_cast<size_t>(materialize_tile_slots) * block_field_count * tile_elems;
        auto i32_global_array_ty = ArrayType::get(i32_ty, global_tile_elems);
        auto i32_global_array_ptr_ty = PointerType::get(i32_global_array_ty, 0);
        auto global_buffer_GV = new GlobalVariable(
            *module, i32_global_array_ty, false, GlobalValue::InternalLinkage,
            UndefValue::get(i32_global_array_ty), "ctt_global_buffer_GV", nullptr,
            GlobalValue::NotThreadLocal,
            1 /* addrspace(1): global memory */);
        auto global_buffer_generic_arr = Builder->CreateAddrSpaceCast(
            global_buffer_GV, i32_global_array_ptr_ty, "ctt_global_buffer_arr");
        global_buffer_generic_ptr = Builder->CreateInBoundsGEP(
            global_buffer_GV->getValueType(), global_buffer_generic_arr,
            {llvm0, llvm0}, "ctt_global_buffer");
    }

    Value* streaming_buffer_global_ptr = nullptr;
    if (use_streaming_global)
    {
        const size_t streaming_tile_elems =
            static_cast<size_t>(streaming_tile_slots) * block_field_count * tile_elems;
        auto streaming_array_ty = ArrayType::get(i32_ty, streaming_tile_elems);
        auto streaming_buffer_GV = new GlobalVariable(
            *module, streaming_array_ty, false, GlobalValue::InternalLinkage,
            UndefValue::get(streaming_array_ty), "ctt_streaming_buffer_GV", nullptr,
            GlobalValue::NotThreadLocal,
            1 /* addrspace(1): global memory */);
        streaming_buffer_global_ptr = Builder->CreateInBoundsGEP(
            streaming_buffer_GV->getValueType(), streaming_buffer_GV,
            {llvm0, llvm0}, "ctt_streaming_buffer");

    }

    // Populate shared_buffer
    // Equivalent to shared_buffer[i] = i
    // gen_if(is_leader, childState, context)([&]()
    // {
    //   for (int i = 0; i < shared_total_elems; ++i)
    //   {
    //     auto idx = ConstantInt::get(i32_ty, i);
    //
    //     // From shared_buffer_GV, creating a shared pointer for loads and stores
    //     auto shared_elem_ptr = Builder->CreateInBoundsGEP(
    //       shared_buffer_GV->getValueType(),
    //       shared_buffer_GV,
    //       { llvm0, idx },
    //       "shared_elem"
    //     );
    //
    //     Builder->CreateStore(idx, shared_elem_ptr);
    //   }
    // });
    Builder->CreateCall(barrier);

    auto i8_ptr_ty = Type::getInt8PtrTy(llvmContext);
    auto i8_shared_ptr_ty = PointerType::get(i8_ty, 3);
    auto i64_ty = Type::getInt64Ty(llvmContext);
    auto zero_i64 = ConstantInt::get(i64_ty, 0);
    auto one_i64 = ConstantInt::get(i64_ty, 1);
    auto u32_bytes = ConstantInt::get(i64_ty, sizeof(uint32_t));
    auto header_fixed_bytes = ConstantInt::get(i64_ty, 3 * sizeof(uint32_t));
    auto chunk_metadata_bytes =
        ConstantInt::get(i64_ty, decode_kernel_spec.chunk_metadata_bytes);
    auto null_i32_ptr = ConstantPointerNull::get(i32_ptr_ty);
    std::function<Value*(Value*, Value*, Value*, Value*)> decode_predecode_element_inline;
    if (use_predecode_registers && cb2t_plan.codec_kind == Cb2tCodecKind::kRfor)
    {
        decode_predecode_element_inline =
            [&](Value* local_idx_i32, Value* run_values_ptr_i32,
                Value* run_ends_ptr_i32, Value* run_count_i32) -> Value*
            {
                // Inline RFOR_PREDECODE run lookup into the generated kernel to
                // avoid a hot per-tuple device-function call.
                Value* decoded_slot =
                    context->CreateEntryBlockAlloca("ctt_reg_decoded", i32_ty);
                Builder->CreateStore(llvm0, decoded_slot);
                Value* has_runs =
                    Builder->CreateICmpSGT(run_count_i32, ConstantInt::get(i32_ty, 0));
                context->gen_if(
                    {ProteusValue{has_runs, context->createFalse()}})([&]()
                {
                    Value* left_slot =
                        context->CreateEntryBlockAlloca("ctt_reg_left", i32_ty);
                    Value* right_slot =
                        context->CreateEntryBlockAlloca("ctt_reg_right", i32_ty);
                    Builder->CreateStore(ConstantInt::get(i32_ty, 0), left_slot);
                    Builder->CreateStore(
                        Builder->CreateSub(run_count_i32, ConstantInt::get(i32_ty, 1)),
                        right_slot);

                    context->gen_while([&]()
                    {
                        Value* left = Builder->CreateLoad(i32_ty, left_slot);
                        Value* right = Builder->CreateLoad(i32_ty, right_slot);
                        return ProteusValue{
                            Builder->CreateICmpSLT(left, right), context->createFalse()
                        };
                    })([&](BranchInst* /*loop_cond*/)
                    {
                        Value* left = Builder->CreateLoad(i32_ty, left_slot);
                        Value* right = Builder->CreateLoad(i32_ty, right_slot);
                        Value* mid = Builder->CreateAdd(
                            left,
                            Builder->CreateLShr(
                                Builder->CreateSub(right, left),
                                ConstantInt::get(i32_ty, 1)));
                        Value* run_end_ptr =
                            Builder->CreateInBoundsGEP(i32_ty, run_ends_ptr_i32, mid);
                        Value* run_end = Builder->CreateLoad(i32_ty, run_end_ptr);
                        Value* move_right =
                            Builder->CreateICmpSGT(run_end, local_idx_i32);
                        Value* next_left = Builder->CreateSelect(
                            move_right, left,
                            Builder->CreateAdd(mid, ConstantInt::get(i32_ty, 1)));
                        Value* next_right =
                            Builder->CreateSelect(move_right, mid, right);
                        Builder->CreateStore(next_left, left_slot);
                        Builder->CreateStore(next_right, right_slot);
                    });

                    Value* run_idx = Builder->CreateLoad(i32_ty, left_slot);
                    Value* run_value_ptr =
                        Builder->CreateInBoundsGEP(i32_ty, run_values_ptr_i32, run_idx);
                    Value* decoded = Builder->CreateLoad(i32_ty, run_value_ptr);
                    Builder->CreateStore(decoded, decoded_slot);
                });
                return Builder->CreateLoad(i32_ty, decoded_slot);
            };
    }
    auto fused_tuple_items_ty = ArrayType::get(i32_ty, runtime_items_per_thread);
    std::vector<Value*> fused_tuple_values_ptrs;
    if (fused_predecode_tuple_execution)
    {
        fused_tuple_values_ptrs.reserve(block_field_count);
        for (size_t fused_block_idx = 0; fused_block_idx < block_field_count;
             ++fused_block_idx)
        {
            auto alloc_name =
                "ctt_fused_tuple_values_" + std::to_string(fused_block_idx);
            fused_tuple_values_ptrs.push_back(
                context->CreateEntryBlockAlloca(alloc_name, fused_tuple_items_ty));
        }
    }

    RecordAttribute block_oid_identifier{relName, activeLoop, pg->getOIDType()};
    Value* block_oid_raw = Builder->CreateLoad(
        childState[block_oid_identifier]
        .mem->getType()
        ->getNonOpaquePointerElementType(),
        childState[block_oid_identifier].mem, "block_oid_raw");
    Value* block_oid_i64_raw =
        Builder->CreateIntCast(block_oid_raw, i64_ty, false);

    // Each CUDA block starts from blockId() and grid-strides over tiles.
    auto tid_in_block = context->threadIdInBlock();
    auto tid_in_block_i32 = Builder->CreateTrunc(tid_in_block, i32_ty);
    auto is_block_leader = Builder->CreateICmpEQ(tid_in_block_i32, llvm0);
    auto block_dim_i64 = context->blockDim();
    auto grid_dim_i64 = context->gridDim();
    const bool emit_tuple_oid_bindings = FLAGS_emit_tuple_oid_bindings;
    Value* tuple_oid_slot_shared_ptr = nullptr;
    if (emit_tuple_oid_bindings)
    {
        auto tuple_oid_scratch_array_ty =
            ArrayType::get(cnt_raw->getType(), runtime_block_threads);
        auto tuple_oid_scratch_GV = new GlobalVariable(
            *module,
            tuple_oid_scratch_array_ty,
            false,
            GlobalValue::InternalLinkage,
            UndefValue::get(tuple_oid_scratch_array_ty),
            "ctt_tuple_oid_scratch_GV",
            nullptr,
            GlobalValue::NotThreadLocal,
            3  // addrspace(3)
        );
        tuple_oid_slot_shared_ptr = Builder->CreateInBoundsGEP(
            tuple_oid_scratch_GV->getValueType(),
            tuple_oid_scratch_GV,
            {llvm0, tid_in_block_i32},
            "ctt_tuple_oid_slot_shared");
    }

    // CompressedBlockToTuples is block-cooperative; normalize block OID to a
    // single value per CUDA block before indexing precomputed maps.
    //
    // Kernel-start broadcasts use the dedicated write-once block-uniform
    // region. They previously reused one slot pair (oid -> cnt -> col0 block
    // pointer) and aliased the per-tile staging area: a barrier orders the
    // leader's write before the readers, but nothing stops a lagging warp
    // from reading a slot AFTER the next phase overwrote it. That WAR race
    // dropped warp-sized tuple groups silently under one-kernel execution.
    Value* block_oid_lo_slot = Builder->CreateInBoundsGEP(
        i32_ty, shared_buffer_generic_ptr,
        ConstantInt::get(i32_ty, shared_block_uniform_base_elems));
    Value* block_oid_hi_slot = Builder->CreateInBoundsGEP(
        i32_ty, shared_buffer_generic_ptr,
        ConstantInt::get(i32_ty, shared_block_uniform_base_elems + 1));
    context->gen_if({ProteusValue{is_block_leader, context->createFalse()}})(
        [&]()
        {
            Value* oid_lo = Builder->CreateTrunc(block_oid_i64_raw, i32_ty);
            Value* oid_hi = Builder->CreateTrunc(
                Builder->CreateLShr(block_oid_i64_raw, ConstantInt::get(i64_ty, 32)),
                i32_ty);
            storeSharedBroadcastI32(oid_lo, block_oid_lo_slot);
            storeSharedBroadcastI32(oid_hi, block_oid_hi_slot);
        });
    Builder->CreateCall(membar_cta);
    Builder->CreateCall(barrier);
    Value* block_oid_lo = Builder->CreateZExt(
        loadSharedBroadcastI32(block_oid_lo_slot), i64_ty);
    Value* block_oid_hi = Builder->CreateZExt(
        loadSharedBroadcastI32(block_oid_hi_slot), i64_ty);
    Value* block_oid_i64 = Builder->CreateOr(
        block_oid_lo, Builder->CreateShl(block_oid_hi, ConstantInt::get(i64_ty, 32)));

    // Normalize active count to one value per CUDA block.
    // Dedicated slots: never reuse the block-OID pair (see comment above).
    Value* cnt_lo_slot = Builder->CreateInBoundsGEP(
        i32_ty, shared_buffer_generic_ptr,
        ConstantInt::get(i32_ty, shared_block_uniform_base_elems + 2));
    Value* cnt_hi_slot = Builder->CreateInBoundsGEP(
        i32_ty, shared_buffer_generic_ptr,
        ConstantInt::get(i32_ty, shared_block_uniform_base_elems + 3));
    Value* cnt_i64_raw = Builder->CreateIntCast(cnt_raw, i64_ty, false);
    context->gen_if({ProteusValue{is_block_leader, context->createFalse()}})(
        [&]()
        {
            Value* cnt_lo = Builder->CreateTrunc(cnt_i64_raw, i32_ty);
            Value* cnt_hi = Builder->CreateTrunc(
                Builder->CreateLShr(cnt_i64_raw, ConstantInt::get(i64_ty, 32)),
                i32_ty);
            storeSharedBroadcastI32(cnt_lo, cnt_lo_slot);
            storeSharedBroadcastI32(cnt_hi, cnt_hi_slot);
        });
    Builder->CreateCall(membar_cta);
    Builder->CreateCall(barrier);
    Value* cnt_lo = Builder->CreateZExt(
        loadSharedBroadcastI32(cnt_lo_slot), i64_ty);
    Value* cnt_hi = Builder->CreateZExt(
        loadSharedBroadcastI32(cnt_hi_slot), i64_ty);
    Value* cnt_uniform_i64 =
        Builder->CreateOr(cnt_lo, Builder->CreateShl(cnt_hi, ConstantInt::get(i64_ty, 32)));
    Value* cnt = Builder->CreateIntCast(cnt_uniform_i64, cnt_raw->getType(), false);

    if (FLAGS_progress_trace)
    {
        Value* is_block0 = Builder->CreateICmpEQ(context->blockId(), zero_i64);
        Value* emit_dims = Builder->CreateAnd(is_block_leader, is_block0);
        context->gen_if({ProteusValue{emit_dims, context->createFalse()}})([&]()
        {
            Function* printi64 = context->getFunction("printi64");
            Value* tagged_block_dim =
                Builder->CreateAdd(ConstantInt::get(i64_ty, 9300000000ULL),
                                   block_dim_i64);
            Value* tagged_grid_dim =
                Builder->CreateAdd(ConstantInt::get(i64_ty, 9400000000ULL),
                                   grid_dim_i64);
            Builder->CreateCall(printi64, {tagged_block_dim});
            Builder->CreateCall(printi64, {tagged_grid_dim});
        });
    }

    auto cnt_type = cnt->getType();
    auto block_dim_cnt = Builder->CreateIntCast(block_dim_i64, cnt_type, false);
    auto tid_in_block_cnt = Builder->CreateIntCast(tid_in_block, cnt_type, false);
    auto items_per_thread_cnt =
        ConstantInt::get(cnt_type, runtime_items_per_thread);
    auto tile_size_cnt = Builder->CreateMul(block_dim_cnt, items_per_thread_cnt);
    auto one_cnt = ConstantInt::get(cnt_type, 1);

    // Hoist per-column metadata and pointers out of the tile loop.
    std::vector<Value*> static_out_ptr_i32s;
    std::vector<Value*> static_out_ptr_i8s;
    std::vector<Cb2tColumnMetadataState> static_column_states;
    static_out_ptr_i32s.reserve(block_field_count);
    static_out_ptr_i8s.reserve(block_field_count);
    static_column_states.reserve(block_field_count);

    size_t static_block_idx = 0;
    for (const auto& field : wantedFields)
    {
        if (!dynamic_cast<const BlockType*>(field.getExpressionType()))
        {
            continue;
        }

        if (!fused_predecode_tuple_execution)
        {
            auto out_offset = ConstantInt::get(
                i64_ty,
                use_streaming_global
                    ? 0u
                    : static_cast<uint64_t>(static_block_idx * tile_elems));
            Value* out_ptr_i32 =
                Builder->CreateInBoundsGEP(i32_ty, out_base_ptr, out_offset);
            Value* out_ptr_i8_shared =
                Builder->CreateBitCast(out_ptr_i32, i8_shared_ptr_ty);
            Value* out_ptr_i8 =
                Builder->CreateAddrSpaceCast(out_ptr_i8_shared, i8_ptr_ty);
            static_out_ptr_i32s.push_back(out_ptr_i32);
            static_out_ptr_i8s.push_back(out_ptr_i8);
        }

        Cb2tColumnMetadataState static_state{};
        RecordAttribute attr{field.getRegisteredAs()};
        Value* block_ptr_raw = Builder->CreateLoad(
            childState[attr].mem->getType()->getNonOpaquePointerElementType(),
            childState[attr].mem);
        Value* block_base = nullptr;
        // Broadcast one compressed-block pointer per CUDA block. The decode path
        // calls block-cooperative device code and must consume a CTA-uniform
        // block pointer. Dedicated write-once slots per column; these used to
        // alias the per-tile chunk-metadata staging slots (and column 0 also
        // aliased the block-OID slots), which raced lagging readers.
        const uint64_t block_ptr_slot_base = static_cast<uint64_t>(
            shared_block_uniform_base_elems + 4 + (2 * static_block_idx));
        Value* block_ptr_lo_slot = makeSharedI32Slot(
            Builder, i32_ty, shared_buffer_generic_ptr, block_ptr_slot_base);
        Value* block_ptr_hi_slot = makeSharedI32Slot(
            Builder, i32_ty, shared_buffer_generic_ptr, block_ptr_slot_base + 1);
        Value* block_ptr_i64_raw = Builder->CreatePtrToInt(
            Builder->CreateBitCast(block_ptr_raw, i8_ptr_ty), i64_ty);
        context->gen_if({ProteusValue{is_block_leader, context->createFalse()}})(
            [&]()
            {
                Value* ptr_lo = Builder->CreateTrunc(block_ptr_i64_raw, i32_ty);
                Value* ptr_hi = Builder->CreateTrunc(
                    Builder->CreateLShr(block_ptr_i64_raw, ConstantInt::get(i64_ty, 32)),
                    i32_ty);
                storeSharedBroadcastI32(ptr_lo, block_ptr_lo_slot);
                storeSharedBroadcastI32(ptr_hi, block_ptr_hi_slot);
            });
        Builder->CreateCall(membar_cta);
        Builder->CreateCall(barrier);
        Value* block_ptr_lo = Builder->CreateZExt(
            loadSharedBroadcastI32(block_ptr_lo_slot), i64_ty);
        Value* block_ptr_hi = Builder->CreateZExt(
            loadSharedBroadcastI32(block_ptr_hi_slot), i64_ty);
        Value* block_ptr_i64 = Builder->CreateOr(
            block_ptr_lo,
            Builder->CreateShl(block_ptr_hi, ConstantInt::get(i64_ty, 32)));
        block_base = Builder->CreateIntToPtr(block_ptr_i64, i8_ptr_ty);
        Value* header_i32 = Builder->CreateBitCast(block_base, i32_ptr_ty);
        if (!use_precomputed_tile_offsets)
        {
            auto magic_const = ConstantInt::get(i32_ty, 0x50424D44);
            Value* magic = Builder->CreateLoad(i32_ty, header_i32);
            static_state.magic_ok = Builder->CreateICmpEQ(magic, magic_const);
        }

        Value* num_chunks_i32 = Builder->CreateLoad(
            i32_ty, Builder->CreateInBoundsGEP(i32_ty, header_i32,
                                               ConstantInt::get(i32_ty, 2)));
        Value* physical_num_chunks_i64 = Builder->CreateZExt(num_chunks_i32, i64_ty);
        static_state.physical_num_chunks_i64 = physical_num_chunks_i64;
        Value* logical_num_chunks_i64 = physical_num_chunks_i64;
        if (cb2t_plan.logical_tiles_per_chunk > 1u)
        {
            logical_num_chunks_i64 = Builder->CreateMul(
                physical_num_chunks_i64,
                ConstantInt::get(i64_ty, cb2t_plan.logical_tiles_per_chunk));
        }
        static_state.num_chunks_i64 = logical_num_chunks_i64;
        if (!use_precomputed_tile_offsets)
        {
            Value* chunk_sizes_ptr = Builder->CreateInBoundsGEP(
                i32_ty, header_i32, ConstantInt::get(i32_ty, 3));
            static_state.chunk_sizes_ptr = chunk_sizes_ptr;
        }

        Value* num_chunks_bytes = Builder->CreateMul(physical_num_chunks_i64, u32_bytes);
        Value* chunk_data_offset = Builder->CreateAdd(header_fixed_bytes, num_chunks_bytes);
        Value* chunk_data_base =
            Builder->CreateInBoundsGEP(i8_ty, block_base, chunk_data_offset);
        static_state.chunk_data_base = chunk_data_base;

        if (use_precomputed_tile_offsets)
        {
            Value* chunk_offsets_oid_map = chunk_offsets_oid_maps[static_block_idx];

            Value* chunk_offsets_ptr_ptr = Builder->CreateInBoundsGEP(
                i32_ptr_ty, chunk_offsets_oid_map, block_oid_i64);
            static_state.precomputed_chunk_offsets_ptr =
                Builder->CreateLoad(i32_ptr_ty, chunk_offsets_ptr_ptr);
        }
        else
        {
            static_state.precomputed_chunk_offsets_ptr = null_i32_ptr;
        }

        static_column_states.push_back(static_state);
        ++static_block_idx;
    }
    CHECK_EQ(static_block_idx, block_field_count);
    if (predecode_uses_run_scratch)
    {
        CHECK(predecode_run_values_scratch_ptr != nullptr);
        CHECK(predecode_run_ends_scratch_ptr != nullptr);
    }
    if (!fused_predecode_tuple_execution)
    {
        CHECK_EQ(static_out_ptr_i32s.size(), block_field_count);
        CHECK_EQ(static_out_ptr_i8s.size(), block_field_count);
    }
    else
    {
        CHECK_EQ(fused_tuple_values_ptrs.size(), block_field_count);
    }
    CHECK(!static_column_states.empty());

    auto tile_locator = makeTileLocator(use_precomputed_tile_offsets);
    auto codec_resolver = makeCodecResolver(
        cb2t_plan, use_precomputed_tile_offsets, tile_elems);

    // Defensive bound: never process more tuples than what compressed chunk
    // headers can provide for this block.
    Value* min_num_chunks_i64 = static_column_states.front().num_chunks_i64;
    for (size_t i = 1; i < static_column_states.size(); ++i)
    {
        Value* candidate = static_column_states[i].num_chunks_i64;
        min_num_chunks_i64 = Builder->CreateSelect(
            Builder->CreateICmpULT(candidate, min_num_chunks_i64), candidate,
            min_num_chunks_i64);
    }
    Value* tile_size_i64 = Builder->CreateIntCast(tile_size_cnt, i64_ty, false);
    Value* max_rows_from_chunks_i64 =
        Builder->CreateMul(min_num_chunks_i64, tile_size_i64);
    Value* cnt_i64 = Builder->CreateIntCast(cnt, i64_ty, false);
    Value* effective_cnt_i64 = Builder->CreateSelect(
        Builder->CreateICmpULT(cnt_i64, max_rows_from_chunks_i64), cnt_i64,
        max_rows_from_chunks_i64);
    Value* effective_cnt =
        Builder->CreateIntCast(effective_cnt_i64, cnt_type, false);
    auto num_tiles_cnt = Builder->CreateUDiv(
        Builder->CreateAdd(effective_cnt,
                           Builder->CreateSub(tile_size_cnt, one_cnt)),
        tile_size_cnt);

    // Backend hook #1 (paper Sec. "Decompression encapsulation", resolve_metadata):
    // interprets the per-tile header; dispatched to the codec's
    // ICb2tCodecResolver, parameterized by the chosen ICb2tTileLocator
    // (metadata source: precomputed offset map vs in-block PBMD header).
    const auto resolve_metadata =
        [&](const size_t block_idx, const Cb2tColumnMetadataState& static_state,
            const Cb2tEmitContext& tile_resolve_ctx) -> Cb2tTileResolution
    {
        return codec_resolver->emit_resolve_metadata(
            block_idx, static_state, tile_resolve_ctx, *tile_locator);
    };

    // Backend hook #2 (paper Sec. "Decompression encapsulation", decompress_tile):
    // tile_decoder builds the codec-specific cooperative/register decode closure;
    // decompress_tile (below) resolves the val/rl pointers and invokes it.
    // Backend hook #2 (paper Sec. "Decompression encapsulation", decompress_tile):
    // loop-invariant decode machinery, consumed by the codec resolver's
    // emit_decompress_tile. Symbol selection (which device decoder for this
    // runtime shape) lives in the codec backend's kernelSpec; here we bind the
    // resulting callees plus the register/fused placement state.
    const Cb2tDecodeContext decode_ctx{
        use_predecode_registers,
        fused_predecode_tuple_execution,
        runtime_block_threads,
        runtime_items_per_thread,
        tid_in_block,
        llvm0,
        shared_buffer_shared_ptr,
        predecode_run_values_scratch_ptr,
        predecode_run_ends_scratch_ptr,
        decode_cooperative_func,
        decode_predecode_setup_func,
        decode_predecode_element_func,
        RLETileMetadata_ty,
        fused_tuple_items_ty,
        &fused_tuple_values_ptrs,
        decode_predecode_element_inline,
        i32_ptr_ty
    };

    const auto decompress_tile =
        [&](const size_t block_idx, const Cb2tColumnMetadataState& static_state,
            const Cb2tTileResolution& tile_resolution,
            const Cb2tDecodeTarget& decode_target, const Cb2tEmitContext& ctx)
    {
        context->gen_if(
            {ProteusValue{
                tile_resolution.can_decompress_block,
                context->createFalse()}})([&]()
        {
            Value* val_section_size_i32 = loadSharedBroadcastI32(
                tile_resolution.val_section_size_slot);
            Value* chunk_offset_i64 = Builder->CreateZExt(
                loadSharedBroadcastI32(tile_resolution.chunk_offset_slot), i64_ty);
            Value* val_section_size_i64 =
                Builder->CreateZExt(val_section_size_i32, i64_ty);
            Value* chunk_base = Builder->CreateInBoundsGEP(
                i8_ty, static_state.chunk_data_base, chunk_offset_i64);
            Value* val_data_i8 =
                Builder->CreateInBoundsGEP(i8_ty, chunk_base, chunk_metadata_bytes);
            Value* val_data_ptr = Builder->CreateBitCast(val_data_i8, i32_ptr_ty);
            Value* rl_data_i8 = Builder->CreateInBoundsGEP(
                i8_ty, chunk_base, val_section_size_i64);
            Value* rl_data_ptr = Builder->CreateBitCast(rl_data_i8, i32_ptr_ty);
            codec_resolver->emit_decompress_tile(
                block_idx, tile_resolution, decode_target, val_data_ptr,
                rl_data_ptr, ctx, decode_ctx);
        });
    };

    std::vector<Value*> streaming_global_out_ptr_i32s;
    if (use_streaming_global)
    {
        CHECK(streaming_buffer_global_ptr != nullptr);
        Function* ctaid_y_fn =
            context->getFunction("llvm.nvvm.read.ptx.sreg.ctaid.y");
        Value* ctaid_y = Builder->CreateZExt(
            Builder->CreateCall(ctaid_y_fn, {}, "ctt_streaming_ctaid_y"),
            i64_ty);
        Value* streaming_slot_i64 = Builder->CreateAdd(
            Builder->CreateMul(ctaid_y, grid_dim_i64), context->blockId(),
            "ctt_streaming_slot");
        Value* slot_stride_elems = ConstantInt::get(
            i64_ty, static_cast<uint64_t>(block_field_count * tile_elems));
        Value* slot_base_elems =
            Builder->CreateMul(streaming_slot_i64, slot_stride_elems);

        streaming_global_out_ptr_i32s.reserve(block_field_count);
        for (size_t block_idx = 0; block_idx < block_field_count; ++block_idx)
        {
            Value* column_offset = ConstantInt::get(
                i64_ty, static_cast<uint64_t>(block_idx * tile_elems));
            Value* out_ptr_i32 = Builder->CreateInBoundsGEP(
                i32_ty, streaming_buffer_global_ptr,
                Builder->CreateAdd(slot_base_elems, column_offset));
            streaming_global_out_ptr_i32s.push_back(out_ptr_i32);
        }
    }

    // =====================================================================
    // Two-level decode/consume loop. Fused modes bridge through shared memory
    // or registers; streaming mode bridges through a bounded global slot.
    // Outer loop: grid-stride over compressed tiles. For each column it calls
    // the two codec backend hooks -- resolve_metadata() then decompress_tile()
    // -- writing one decoded tile per column into the shared scratchpad, then a
    // __syncthreads() barrier. Inner loop (emitDecodedTuples): each thread reads
    // its assigned values from the scratchpad, builds the tuple environment, and
    // inlines the downstream operators' consume(); a second barrier brackets it
    // before the scratchpad is reused for the next tile. Everything below the
    // hook calls is codec-agnostic operator scaffolding.
    // =====================================================================
    // Keep tile_id in SSA form (PHI) instead of per-thread alloca+stores.
    // Local-memory-backed loop counters can surface as DRAM traffic in NCU.
    BasicBlock* tile_loop_preheader = Builder->GetInsertBlock();
    Value* tile_id_init = context->blockId();
    PHINode* tile_id_phi = nullptr;

    context->gen_while([&]()
    {
        if (!tile_id_phi)
        {
            tile_id_phi = Builder->CreatePHI(i64_ty, 2, "tile_id");
            tile_id_phi->addIncoming(tile_id_init, tile_loop_preheader);
        }
        Value* tile_id = tile_id_phi;
        Value* tile_id_cnt = Builder->CreateIntCast(tile_id, cnt_type, false);
        return ProteusValue{
            Builder->CreateICmpULT(tile_id_cnt, num_tiles_cnt),
            context->createFalse()
        };
    })([&](BranchInst* /*tile_loop_cond*/)
    {
        Value* tile_id = tile_id_phi;
        Value* tile_id_i32 = Builder->CreateTrunc(tile_id, i32_ty);
        Value* tile_id_cnt = Builder->CreateIntCast(tile_id, cnt_type, false);

        if (FLAGS_progress_trace)
        {
            Value* is_block0 = Builder->CreateICmpEQ(context->blockId(), zero_i64);
            Value* sample_stride = ConstantInt::get(i64_ty, 64);
            Value* is_sample_tile = Builder->CreateICmpEQ(
                Builder->CreateURem(tile_id, sample_stride), zero_i64);
            Value* emit_progress = Builder->CreateAnd(
                is_block_leader, Builder->CreateAnd(is_block0, is_sample_tile));
            context->gen_if({ProteusValue{emit_progress, context->createFalse()}})(
                [&]()
                {
                    Function* printi64 = context->getFunction("printi64");
                    Value* tagged_tile =
                        Builder->CreateAdd(ConstantInt::get(i64_ty, 9100000000ULL),
                                           tile_id);
                    Builder->CreateCall(printi64, {tagged_tile});
                });
        }

        // Per-column pointers used by the tuple loop.
        // - Shared mode: consume from shared tiles.
        // - Legacy global-tile-window mode: consume from a
        //   global-memory tile window indexed by tile_id, so the working set can
        //   exceed L2 and show explicit GMEM roundtrip traffic in NCU.
        std::vector<Value*> tile_global_out_ptr_i32s;
        std::vector<Value*> tile_global_out_ptr_i8s;
        if (materialize_tiles_to_global)
        {
            tile_global_out_ptr_i32s.reserve(block_field_count);
            tile_global_out_ptr_i8s.reserve(block_field_count);
            Value* slots_total = ConstantInt::get(i64_ty, materialize_tile_slots);
            // Slot mapping for materialized tiles:
            // use the *runtime* gridDim.x so different blockIdx.x CTAs remain
            // disjoint even when launch grid differs from ctt_grid_cap().
            Value* slot_epochs = Builder->CreateUDiv(slots_total, grid_dim_i64);
            Value* has_slot_epochs =
                Builder->CreateICmpUGE(slot_epochs, one_i64);
            slot_epochs = Builder->CreateSelect(
                has_slot_epochs, slot_epochs, one_i64, "ctt_slot_epochs");

            // tile_id = blockIdx.x + k * gridDim.x, so floor(tile_id / gridDim.x)
            // is the per-CTA tile epoch k.
            Value* tile_epoch = Builder->CreateUDiv(tile_id, grid_dim_i64);
            Value* slot_epoch = Builder->CreateURem(tile_epoch, slot_epochs);
            Value* slot = Builder->CreateAdd(
                Builder->CreateMul(slot_epoch, grid_dim_i64), tile_id_init);
            // Defensive clamp for any runtime mismatch where gridDim.x > slots_total.
            slot = Builder->CreateURem(slot, slots_total);
            Value* slot_stride_elems = ConstantInt::get(
                i64_ty, static_cast<uint64_t>(block_field_count * tile_elems));
            Value* slot_base_elems = Builder->CreateMul(slot, slot_stride_elems);
            for (size_t block_idx = 0; block_idx < block_field_count; ++block_idx)
            {
                Value* global_col_offset = ConstantInt::get(
                    i64_ty, static_cast<uint64_t>(block_idx * tile_elems));
                Value* global_out_ptr_i32 = Builder->CreateInBoundsGEP(
                    i32_ty, global_buffer_generic_ptr,
                    Builder->CreateAdd(slot_base_elems, global_col_offset));
                tile_global_out_ptr_i32s.push_back(global_out_ptr_i32);
                tile_global_out_ptr_i8s.push_back(
                    Builder->CreateBitCast(global_out_ptr_i32, i8_ptr_ty));
            }
        }
        // In one-kernel mode the launch is 2D (grid.y = number of blocks), and
        // materialize slots are shared across CTA rows. To avoid cross-row
        // slot races, consume from shared tiles when one-kernel is enabled.
        Value* use_global_materialized_readback =
            ConstantInt::get(Type::getInt1Ty(llvmContext), !FLAGS_one_kernel);

        auto chunk_metadata_bytes_i32 = ConstantInt::get(
            i32_ty, decode_kernel_spec.chunk_metadata_bytes);
        const Cb2tEmitContext tile_resolve_ctx{
            context,
            Builder,
            i8_ty,
            i32_ty,
            i64_ty,
            active_tile_metadata_ty,
            active_tile_metadata_ptr_ty,
            FLBPVectorMetadata_ty,
            shared_buffer_generic_ptr,
            is_block_leader,
            tile_id,
            tile_id_i32,
            null_i32_ptr,
            barrier,
            membar_cta,
            zero_i64,
            one_i64,
            chunk_metadata_bytes_i32,
            shared_broadcast_base_elems,
            shared_chunk_meta_elems
        };

        for (size_t block_idx = 0; block_idx < block_field_count; ++block_idx)
        {
            Value* out_ptr_i32 = nullptr;
            if (!fused_predecode_tuple_execution)
            {
                out_ptr_i32 = static_out_ptr_i32s[block_idx];
            }
            const auto& static_state = static_column_states[block_idx];
            Cb2tTileResolution tile_resolution = resolve_metadata(
                block_idx, static_state, tile_resolve_ctx);
            decompress_tile(block_idx, static_state, tile_resolution,
                            Cb2tDecodeTarget{out_ptr_i32}, tile_resolve_ctx);
            if (use_streaming_global)
            {
                // Complete this column's cooperative decode, publish it to the
                // CTA-owned global slot, then reuse the single shared output
                // tile for the next projected column.
                Builder->CreateCall(barrier);
                Value* global_out_ptr_i32 =
                    streaming_global_out_ptr_i32s[block_idx];
                for (uint32_t item = 0; item < runtime_items_per_thread; ++item)
                {
                    Value* elem_idx = Builder->CreateAdd(
                        tid_in_block,
                        ConstantInt::get(
                            i64_ty, static_cast<uint64_t>(item) *
                                        runtime_block_threads));
                    Value* shared_elem_ptr = Builder->CreateInBoundsGEP(
                        i32_ty, out_ptr_i32, elem_idx);
                    Value* global_elem_ptr = Builder->CreateInBoundsGEP(
                        i32_ty, global_out_ptr_i32, elem_idx);
                    Value* elem_val =
                        Builder->CreateLoad(i32_ty, shared_elem_ptr);
                    Builder->CreateCall(
                        streaming_store_cg, {global_elem_ptr, elem_val});
                }
                Builder->CreateCall(membar_cta);
                Builder->CreateCall(barrier);
            }
            else if (predecode_uses_run_scratch)
            {
                // Reuse a single predecode scratch pair across columns.
                Builder->CreateCall(barrier);
            }
        }

        // In non-fused paths we need all column tiles materialized before tuple
        // consumption starts.
        if (!fused_predecode_tuple_execution && !use_streaming_global)
        {
            Builder->CreateCall(barrier);
        }
        if (materialize_tiles_to_global)
        {
            for (size_t block_idx = 0; block_idx < block_field_count; ++block_idx)
            {
                Value* shared_out_ptr_i32 = static_out_ptr_i32s[block_idx];
                Value* global_out_ptr_i32 = tile_global_out_ptr_i32s[block_idx];
                for (uint32_t item = 0; item < runtime_items_per_thread; ++item)
                {
                    Value* elem_idx = Builder->CreateAdd(
                        tid_in_block, ConstantInt::get(
                            i64_ty, static_cast<uint64_t>(item) *
                            runtime_block_threads));
                    Value* shared_elem_ptr = Builder->CreateInBoundsGEP(
                        i32_ty, shared_out_ptr_i32, elem_idx);
                    Value* global_elem_ptr = Builder->CreateInBoundsGEP(
                        i32_ty, global_out_ptr_i32, elem_idx);
                    Value* elem_val = Builder->CreateLoad(i32_ty, shared_elem_ptr);
                    StoreInst* materialize_store =
                        Builder->CreateStore(elem_val, global_elem_ptr);
                    // Keep global writeback observable even though tuple consumption now
                    // reads from shared memory.
                    materialize_store->setVolatile(true);
                }
            }
            // Publish global writeback before tuple loads.
            Builder->CreateCall(barrier);
        }

        // This CUDA block owns one tile, so compute [tile_base, tile_end).
        auto tile_base = Builder->CreateMul(tile_id_cnt, tile_size_cnt);
        auto tile_end = Builder->CreateAdd(tile_base, tile_size_cnt);
        tile_end = Builder->CreateSelect(
            Builder->CreateICmpULT(effective_cnt, tile_end), effective_cnt,
            tile_end);
        const std::function<void(Value * lhs, Value * lane_item_idx, MDNode * LoopID)>
            & emitDecodedTuples = [&](Value* lhs, Value* lane_item_idx, MDNode* LoopID)
            {
                std::map<RecordAttribute, ProteusValueMemory> variableBindings;
                if (emit_tuple_oid_bindings)
                {
                    // Some join pipelines still expect activeLoop/OID tuple
                    // bindings. Skip them in the common no-join case to keep
                    // the fused tuple loop register-light.
                    Builder->CreateStore(lhs, tuple_oid_slot_shared_ptr);
                    RecordAttribute tupleIdentifier{
                        relName, activeLoop, pg->getOIDType()
                    };
                    ProteusValueMemory mem_posWrapper{
                        tuple_oid_slot_shared_ptr, context->createFalse()
                    };
                    variableBindings[tupleIdentifier] = mem_posWrapper;
                    for (const auto& w : wantedFields)
                    {
                        RecordAttribute tIdentifier{
                            w.getRegisteredRelName(), activeLoop,
                            pg->getOIDType()
                        };
                        variableBindings[tIdentifier] = mem_posWrapper;
                    }
                }

                // Actual Work (Loop through attributes etc.)
                // Source pointer per block-field follows the selected placement:
                // shared/register fused, historical materialize, or the
                // collision-free bounded global streaming pool.
                size_t decomp_idx = 0;
                for (const auto& field : wantedFields)
                {
                    if (!dynamic_cast<const BlockType*>(field.getExpressionType()))
                    {
                        Value* arg = Builder->CreateLoad(
                            childState[field.getRegisteredAs()]
                                .mem->getType()
                                ->getNonOpaquePointerElementType(),
                            childState[field.getRegisteredAs()].mem);
                        variableBindings[field.getRegisteredAs()] =
                            context->toMem(arg, context->createFalse());
                        continue;
                    }
                    RecordAttribute attr{field.getRegisteredAs()};

                    auto arg_ptr_ty =
                        childState[attr].mem->getType()->getNonOpaquePointerElementType();
                    if (fused_predecode_tuple_execution)
                    {
                        Value* fused_item_ptr = Builder->CreateInBoundsGEP(
                            fused_tuple_items_ty, fused_tuple_values_ptrs[decomp_idx],
                            {llvm0, lane_item_idx});
                        Value* fused_arg_ptr =
                            Builder->CreateBitCast(fused_item_ptr, arg_ptr_ty);
                        ++decomp_idx;
                        variableBindings[field.getRegisteredAs()] =
                            ProteusValueMemory{fused_arg_ptr, context->createFalse()};
                        continue;
                    }
                    Value* local_idx = Builder->CreateSub(lhs, tile_base);
                    if constexpr (GLOBAL_STAGE)
                    {
                        Value* global_elem_ptr = Builder->CreateInBoundsGEP(
                            i32_ty,
                            streaming_global_out_ptr_i32s[decomp_idx],
                            local_idx);
                        ++decomp_idx;
                        Value* parsed = Builder->CreateCall(
                            streaming_load_cg, {global_elem_ptr},
                            "ctt_streaming_load");
                        if (auto* instruction = dyn_cast<Instruction>(parsed))
                        {
                            instruction->setMetadata(
                                LLVMContext::MD_mem_parallel_loop_access,
                                LoopID);
                        }
                        variableBindings[field.getRegisteredAs()] =
                            context->toMem(parsed, context->createFalse());
                        continue;
                    }
                    Value* decomp_src_ptr = static_out_ptr_i8s[decomp_idx];
                    if (materialize_tiles_to_global)
                    {
                        decomp_src_ptr = Builder->CreateSelect(
                            use_global_materialized_readback,
                            tile_global_out_ptr_i8s[decomp_idx],
                            static_out_ptr_i8s[decomp_idx]);
                    }
                    Value* arg = Builder->CreateBitCast(decomp_src_ptr, arg_ptr_ty);
                    ++decomp_idx;

                    // Translate global tuple id to this tile's local index.
                    Value* ptr = Builder->CreateGEP(
                        arg->getType()->getNonOpaquePointerElementType(), arg, local_idx);

                    if constexpr (!LEGACY_GLOBAL_TILE_WINDOW && !GLOBAL_STAGE)
                    {
                        // Deferred path: bind the decoded value location directly.
                        variableBindings[field.getRegisteredAs()] =
                            ProteusValueMemory{ptr, context->createFalse()};
                    }
                    else
                    {
                        // Global placement paths force an explicit load from
                        // the staged tile before invoking downstream operators.
                        Value* parsed = nullptr;
                        auto* materialized_load = Builder->CreateLoad(
                            ptr->getType()->getNonOpaquePointerElementType(),
                            ptr);
                        materialized_load->setVolatile(true);
                        materialized_load->setMetadata(
                            LLVMContext::MD_mem_parallel_loop_access,
                            LoopID);
                        parsed = materialized_load;
                        ptr = parsed;
                        variableBindings[field.getRegisteredAs()] =
                            context->toMem(ptr, context->createFalse());
                    }
                }

                // Triggering parent
                OperatorState state{*this, variableBindings};
                getParent()->consume(context, state);
            };

        // Build loop metadata once and reuse for fixed per-lane iterations.
        MDNode* LoopID;
        {
            MDString* vec_st =
                MDString::get(llvmContext, "llvm.loop.vectorize.enable");
            Type* int1Type = Type::getInt1Ty(llvmContext);
            Metadata* one = ConstantAsMetadata::get(ConstantInt::get(int1Type, 1));
            Metadata* vec_en[] = {vec_st, one};
            MDNode* vectorize_enable = MDNode::get(llvmContext, vec_en);

            MDString* itr_st =
                MDString::get(llvmContext, "llvm.loop.interleave.count");
            Metadata* count = ConstantAsMetadata::get(ConstantInt::get(i32_ty, 4));
            Metadata* itr_en[] = {itr_st, count};
            MDNode* interleave_count = MDNode::get(llvmContext, itr_en);

            Metadata* Args[] = {nullptr, vectorize_enable, interleave_count};
            LoopID = MDNode::get(llvmContext, Args);
            LoopID->replaceOperandWith(0, LoopID);
        }

        // Fixed lane-local tuple iteration: avoid memory-backed loop-counter
        // load/store traffic in the consume hot path.
        for (uint32_t item = 0; item < runtime_items_per_thread; ++item)
        {
            Value* lane_idx = Builder->CreateAdd(
                tid_in_block_cnt,
                ConstantInt::get(
                    cnt_type, static_cast<uint64_t>(item) * runtime_block_threads));
            Value* lhs = Builder->CreateAdd(tile_base, lane_idx);
            Value* in_range = Builder->CreateICmpSLT(lhs, tile_end);
            context->gen_if({ProteusValue{in_range, context->createFalse()}})(
                [&]()
                {
                    emitDecodedTuples(lhs, ConstantInt::get(i32_ty, item), LoopID);
                });
        }

        // Reuse shared memory for the next grid-stride tile only after all threads
        // in this block finished consuming the current tile.
        Builder->CreateCall(barrier);
        if (FLAGS_progress_trace)
        {
            Value* is_block0 = Builder->CreateICmpEQ(context->blockId(), zero_i64);
            Value* sample_stride = ConstantInt::get(i64_ty, 64);
            Value* is_sample_tile = Builder->CreateICmpEQ(
                Builder->CreateURem(tile_id, sample_stride), zero_i64);
            Value* emit_progress = Builder->CreateAnd(
                is_block_leader, Builder->CreateAnd(is_block0, is_sample_tile));
            context->gen_if({ProteusValue{emit_progress, context->createFalse()}})(
                [&]()
                {
                    Function* printi64 = context->getFunction("printi64");
                    Value* tagged_tile =
                        Builder->CreateAdd(ConstantInt::get(i64_ty, 9200000000ULL),
                                           tile_id);
                    Builder->CreateCall(printi64, {tagged_tile});
                });
        }
        Value* next_tile_id = Builder->CreateAdd(tile_id, grid_dim_i64);
        tile_id_phi->addIncoming(next_tile_id, Builder->GetInsertBlock());
    });
}

template <bool LEGACY_GLOBAL_TILE_WINDOW, bool REGISTER_FUSED,
          bool GLOBAL_STAGE>
void CompressedBlockToTuples<LEGACY_GLOBAL_TILE_WINDOW, REGISTER_FUSED,
                             GLOBAL_STAGE>::open(Pipeline* pip)
{
    event_range<range_log_op::UNPACK_OPEN> er{
        id, pip->getGeneratorUUID(),
        pip->getGroup()
    };
    void** buffs;

    if (gpu)
    {
        const size_t ptr_bytes = sizeof(void*) * old_buffs.size();
        buffs = (void**)MemoryManager::mallocGpu(ptr_bytes);

        // Only zero-init when we may release previous buffers. In the common
        // release_input_buffers_=false path this avoids an unnecessary GPU sync.
        if (release_input_buffers_)
        {
            cudaStream_t strm = createNonBlockingStream();
            gpu_run(cudaMemsetAsync(buffs, 0, ptr_bytes, strm));
            syncAndDestroyStream(strm);
        }
    }
    else
    {
        buffs =
            (void**)MemoryManager::mallocPinned(sizeof(void*) * old_buffs.size());
        memset(buffs, 0, sizeof(void*) * old_buffs.size());
    }

    for (size_t i = 0; i < old_buffs.size(); ++i)
    {
        pip->setStateVar<void*>(old_buffs[i], buffs + i);
    }
}

template <bool LEGACY_GLOBAL_TILE_WINDOW, bool REGISTER_FUSED,
          bool GLOBAL_STAGE>
void CompressedBlockToTuples<LEGACY_GLOBAL_TILE_WINDOW, REGISTER_FUSED,
                             GLOBAL_STAGE>::close(Pipeline* pip)
{
    event_range<range_log_op::UNPACK_CLOSE> er{
        id, pip->getGeneratorUUID(),
        pip->getGroup()
    };
    void** h_buffs = nullptr;
    void** buffs = pip->getStateVar<void**>(old_buffs.at(0));

    if (gpu)
    {
        if (release_input_buffers_)
        {
            h_buffs = (void**)malloc(sizeof(void*) * old_buffs.size());
            assert(h_buffs);
            cudaStream_t strm = createNonBlockingStream();
            gpu_run(cudaMemcpyAsync(h_buffs, buffs, sizeof(void *) * old_buffs.size(),
                cudaMemcpyDefault, strm));
            syncAndDestroyStream(strm);
        }
    }
    else
    {
        h_buffs = buffs;
    }

    if (release_input_buffers_)
    {
        for (size_t i = 0; i < old_buffs.size(); ++i)
        {
            BlockManager::release_buffer((int32_t*)h_buffs[i]);
        }
    }

    if (gpu)
        MemoryManager::freeGpu(buffs);
    else
        MemoryManager::freePinned(buffs);

    if (gpu && h_buffs) free(h_buffs);
}

template class CompressedBlockToTuples<false, false>;
template class CompressedBlockToTuples<true, false>;
template class CompressedBlockToTuples<false, true>;
template class CompressedBlockToTuples<false, false, true>;
