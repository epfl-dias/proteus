#ifndef TRANSPOSE_HPP
#define TRANSPOSE_HPP

#include <cstdint>
#include <cstring>

namespace generated::transpose::fallback::scalar {
void        transpose_i(const double* __restrict in, double* __restrict out);
void        transpose_i(const uint64_t* __restrict in, uint64_t* __restrict out);
void        transpose_i(const uint32_t* __restrict in, uint32_t* __restrict out);
void        transpose_i(const uint16_t* __restrict in, uint16_t* __restrict out);
void        transpose_i(const uint8_t* __restrict in, uint8_t* __restrict out);
void        transpose_o(const double* __restrict in, double* __restrict out);
void        transpose_o(const uint64_t* __restrict in, uint64_t* __restrict out);
void        transpose_o(const uint32_t* __restrict in, uint32_t* __restrict out);
void        transpose_o(const uint16_t* __restrict in, uint16_t* __restrict out);
void        transpose_o(const uint8_t* __restrict in, uint8_t* __restrict out);
inline void transpose_i(const int32_t* __restrict in, int32_t* __restrict out) {
	transpose_i(reinterpret_cast<const uint32_t*>(in), (reinterpret_cast<uint32_t*>(out)));
}

} // namespace generated::transpose::fallback::scalar

#endif
