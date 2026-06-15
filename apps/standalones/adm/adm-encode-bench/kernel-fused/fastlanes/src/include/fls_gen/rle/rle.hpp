#ifndef RLE_HPP
#define RLE_HPP

#include "debug.hpp"

namespace fastlanes {

template <int MINI_VEC_SIZE = 256, int MINI_VEC_N = (1024 / MINI_VEC_SIZE)>
uint32_t RLE(const int32_t* const vec_arr, uint8_t* const idx_arr, int32_t* const dict_rle_val) {

	auto cur_val {vec_arr[0]};
	auto cur_idx {0};
	idx_arr[0]      = 0;
	dict_rle_val[0] = cur_val;
	for (size_t i {1}; i < MINI_VEC_SIZE; ++i) {
		const auto nex_val = vec_arr[i];
		if (cur_val != nex_val) {
			cur_idx               = cur_idx + 1;
			dict_rle_val[cur_idx] = nex_val;
			cur_val               = nex_val;
		}
		idx_arr[i] = cur_idx;
	}

	return cur_idx + 1;
}

template <int MINI_VEC_SIZE = 256, int MINI_VEC_N = (1024 / MINI_VEC_SIZE)>
uint16_t VECTOR_RLE(const int32_t* const vec_p, uint8_t* const idx_p, int32_t* const dict_rle_val, uint16_t* dict_4_p) {

	dict_4_p[0]       = 0;
	dict_4_p[1]       = dict_4_p[0] + RLE(vec_p + 0, idx_p + 0, dict_rle_val + dict_4_p[0]);
	dict_4_p[2]       = dict_4_p[1] + RLE(vec_p + 256, idx_p + 256, dict_rle_val + dict_4_p[1]);
	dict_4_p[3]       = dict_4_p[2] + RLE(vec_p + 512, idx_p + 512, dict_rle_val + dict_4_p[2]);
	const auto reuslt = RLE(vec_p + 768, idx_p + 768, dict_rle_val + dict_4_p[3]);

	PRINT(dict_rle_val, "dict_rle_val");
	PRINT(dict_4_p, "dict_4_p");

	return reuslt;
}

} // namespace fastlanes

#endif // RLE_HPP
