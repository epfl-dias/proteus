/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2020
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

#ifndef PROTEUS_BITWISE_OPS_HPP
#define PROTEUS_BITWISE_OPS_HPP

template <typename T>
constexpr T next_power_of_2(T v);

template <typename T, size_t bits>
class next_power_of_2_impl {
 private:
  constexpr T operator()(T v) {
    return next_power_of_2_impl<T, bits * 2>{}(v | (v >> bits));
  }

  friend constexpr T next_power_of_2<>(T v);
  friend class next_power_of_2_impl<T, bits / 2>;
};

template <typename T>
class next_power_of_2_impl<T, sizeof(T) * 8> {
 private:
  constexpr T operator()(T v) { return v; }

  friend class next_power_of_2_impl<T, sizeof(T) * 8 / 2>;
};

template <typename T>
constexpr T next_power_of_2(T v) {
  return next_power_of_2_impl<T, 1>{}(v - 1) + 1;
}

static_assert(next_power_of_2(1) == 1);
static_assert(next_power_of_2(2) == 2);
static_assert(next_power_of_2(3) == 4);
static_assert(next_power_of_2(4) == 4);
static_assert(next_power_of_2(5) == 8);
static_assert(next_power_of_2(6) == 8);
static_assert(next_power_of_2(7) == 8);
static_assert(next_power_of_2(8) == 8);
static_assert(next_power_of_2(9) == 16);
static_assert(next_power_of_2(UINT64_C(0xFFFF'FFFF)) ==
              UINT64_C(0x1'0000'0000));
static_assert(next_power_of_2(UINT64_C(0xFFFF'FF00)) ==
              UINT64_C(0x1'0000'0000));
static_assert(next_power_of_2(UINT64_C(0x1'0000'0000)) ==
              UINT64_C(0x1'0000'0000));
static_assert(next_power_of_2(0) == 0);
static_assert(next_power_of_2(UINT64_C(0x0)) == UINT64_C(0x0));

#endif  // PROTEUS_BITWISE_OPS_HPP
