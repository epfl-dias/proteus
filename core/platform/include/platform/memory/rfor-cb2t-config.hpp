/*
    Proteus -- High-performance query processing on heterogeneous hardware.
*/

#ifndef PROTEUS_RFOR_CB2T_CONFIG_HPP
#define PROTEUS_RFOR_CB2T_CONFIG_HPP

namespace proteus
{
    namespace rfor
    {
        // Decode strategy used by CompressedBlockToTuples (fused in-query decode).
        namespace cb2t
        {
            static constexpr int kBlockThreads = 128;
            static constexpr int kItemsPerThread = 4;
            static constexpr int kTileSize = kBlockThreads * kItemsPerThread;
            static_assert(
                kBlockThreads * kItemsPerThread == kTileSize,
                "RFOR CB2T defaults must satisfy block_threads * items_per_thread == tile_size");
        } // namespace cb2t

        // Decode strategy used by pre-decompression (MemMove/GpuDecompressor path).
        // This may use a different thread decomposition than CB2T. Dataset layout
        // compatibility is checked at runtime via --tile_size and RFOR metadata.
        namespace predecompress
        {
            // Here dataset with tile_size 1024 is faster
             static constexpr int kBlockThreads = 256;
            static constexpr int kItemsPerThread = 4;
            // static constexpr int kBlockThreads = 128;
            // static constexpr int kItemsPerThread = 4;
            static constexpr int kTileSize = kBlockThreads * kItemsPerThread;
            static_assert(
                kBlockThreads * kItemsPerThread == kTileSize,
                "RFOR predecompress defaults must satisfy block_threads * items_per_thread == tile_size");
        } // namespace predecompress

        // Backward-compatible aliases (prefer namespaced constants above).
        static constexpr int kCb2tBlockThreads = cb2t::kBlockThreads;
        static constexpr int kCb2tItemsPerThread = cb2t::kItemsPerThread;
        static constexpr int kCb2tTileSize = cb2t::kTileSize;
    } // namespace rfor
} // namespace proteus

#endif  // PROTEUS_RFOR_CB2T_CONFIG_HPP
