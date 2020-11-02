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

#ifndef PROTEUS_RDTSC_HPP
#define PROTEUS_RDTSC_HPP

#include <cstdint>

#if defined(__powerpc64__) || defined(__ppc64__)
inline uint64_t rdtsc() {
  uint64_t c;
  asm volatile("mfspr %0, 268" : "=r"(c));
  return c;
}
#else
#include <x86intrin.h>
inline uint64_t rdtsc() { return __rdtsc(); }
#endif

#endif  // PROTEUS_RDTSC_HPP
