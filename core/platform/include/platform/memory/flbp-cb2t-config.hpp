/*
    Proteus -- High-performance query processing on heterogeneous hardware.
*/

#ifndef PROTEUS_FLBP_CB2T_CONFIG_HPP
#define PROTEUS_FLBP_CB2T_CONFIG_HPP

#include <cstdint>

namespace proteus
{
    namespace flbp
    {
        // Immutable FLBP CB2T format contract.
        // Execution defaults (threads/items) are centralized in
        // core/olap/lib/operators/decompression-flags.cpp.
        namespace cb2t
        {
            static constexpr uint32_t kVectorSize = 1024;
            static constexpr uint32_t kMetadataBytes = 12;
        } // namespace cb2t

        // Decode contract used by pre-decompression (MemMove/GpuDecompressor path).
        namespace predecompress
        {
            static constexpr uint32_t kVectorSize = 1024;
            static constexpr int kBlockThreads = 32;
            static constexpr uint32_t kMetadataBytes = 12;
        } // namespace predecompress

        // Backward-compatible aliases (prefer namespaced constants above).
        static constexpr uint32_t kCb2tVectorSize = cb2t::kVectorSize;
        static constexpr uint32_t kCb2tMetadataBytes = cb2t::kMetadataBytes;
        static constexpr uint32_t kPredecompressVectorSize = predecompress::kVectorSize;
        static constexpr int kPredecompressBlockThreads = predecompress::kBlockThreads;
        static constexpr uint32_t kPredecompressMetadataBytes =
            predecompress::kMetadataBytes;
    } // namespace flbp
} // namespace proteus

#endif  // PROTEUS_FLBP_CB2T_CONFIG_HPP
