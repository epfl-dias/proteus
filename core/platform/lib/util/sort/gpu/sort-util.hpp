/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
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

#ifndef SORT_UTIL_HPP_
#define SORT_UTIL_HPP_

#include <cstdlib>

extern "C" void qsort_i(void *ptr, size_t N);
extern "C" void qsort_l(void *ptr, size_t N);
extern "C" void qsort_ii(void *ptr, size_t N);
extern "C" void qsort_il(void *ptr, size_t N);
extern "C" void qsort_li(void *ptr, size_t N);
extern "C" void qsort_ll(void *ptr, size_t N);
extern "C" void qsort_iii(void *ptr, size_t N);
extern "C" void qsort_iil(void *ptr, size_t N);
extern "C" void qsort_ili(void *ptr, size_t N);
extern "C" void qsort_ill(void *ptr, size_t N);
extern "C" void qsort_lii(void *ptr, size_t N);
extern "C" void qsort_lil(void *ptr, size_t N);
extern "C" void qsort_lli(void *ptr, size_t N);
extern "C" void qsort_lll(void *ptr, size_t N);
extern "C" void qsort_iiii(void *ptr, size_t N);
extern "C" void qsort_iiil(void *ptr, size_t N);
extern "C" void qsort_iili(void *ptr, size_t N);
extern "C" void qsort_iill(void *ptr, size_t N);
extern "C" void qsort_ilii(void *ptr, size_t N);
extern "C" void qsort_ilil(void *ptr, size_t N);
extern "C" void qsort_illi(void *ptr, size_t N);
extern "C" void qsort_illl(void *ptr, size_t N);
extern "C" void qsort_liii(void *ptr, size_t N);
extern "C" void qsort_liil(void *ptr, size_t N);
extern "C" void qsort_lili(void *ptr, size_t N);
extern "C" void qsort_lill(void *ptr, size_t N);
extern "C" void qsort_llii(void *ptr, size_t N);
extern "C" void qsort_llil(void *ptr, size_t N);
extern "C" void qsort_llli(void *ptr, size_t N);
extern "C" void qsort_llll(void *ptr, size_t N);

#endif /* SORT_UTIL_HPP_ */
