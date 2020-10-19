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

#include "util/sort/gpu/sort-util.hpp"

// NOTE: Clang fails to compile thrust, but that is fine if we call this part
// only from generated code
#define _CubLog(format, ...) printf(format, __VA_ARGS__)
#define THRUST_CPP_DIALECT 2017
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

#include "memory/block-manager-conf.hpp"
#include "util/timing.hpp"

template <typename T, typename... Trest>
struct minalignof {
  static constexpr size_t get() {
    // pre c++14
    return alignof(T) >= minalignof<Trest...>::get()
               ? minalignof<Trest...>::get()
               : alignof(T);
    // requires c++14
    // return min(minalignof<Trest...>::get(), alignof(T));
  }
};

template <typename T>
struct minalignof<T> {
  static constexpr size_t get() { return alignof(T); }
};

// template<typename T, typename... Trest>
// struct maxalignof{
//     static constexpr size_t get(){
//         // pre c++14
//         return alignof(T) <= maxalignof<Trest...>::get() ?
//         maxalignof<Trest...>::get() : alignof(T);
//         // requires c++14
//         // return max(minalignof<Trest...>::get(), alignof(T));
//     }
// };

// template<typename T>
// struct maxalignof<T>{
//     static constexpr size_t get(){
//         return alignof(T);
//     }
// };

template <typename T, typename... Trest>
struct qsort_helper_t;  // to get over gcc not believing constexpr is
                        // constexpr...

/**
 * Bacerafull!
 * It expects the template arguments to be the reverse order of the wanted!
 */
template <size_t alignment, /* alignment is needed because of a bug in gcc... */
          typename T, typename... Trest>
struct alignas(alignment) qsort_packed_t {
  qsort_helper_t<Trest...> r;
  char pad[sizeof(qsort_helper_t<Trest...>) % alignof(T)];
  T a;

  __host__ __device__ bool operator==(
      const qsort_packed_t<alignment, T, Trest...> &other) const {
    return (r == other.r) && (a == other.a);
  }

  __host__ __device__ bool operator<(
      const qsort_packed_t<alignment, T, Trest...> &other) const {
    return (r == other.r) ? (a < other.a) : (r < other.r);
  }
} __attribute__((packed));

template <typename T, typename... Trest>
struct qsort_helper_t {
  qsort_packed_t<minalignof<T, Trest...>::get(), T, Trest...> r;

  __host__ __device__ bool operator==(
      const qsort_helper_t<T, Trest...> &other) const {
    return (r == other.r);
  }
};

template <size_t alignment, typename T>
struct qsort_packed_t<alignment, T> {
  T a;

  __host__ __device__ bool operator==(
      const qsort_packed_t<alignment, T> &other) const {
    return a == other.a;
  }

  __host__ __device__ bool operator<(
      const qsort_packed_t<alignment, T> &other) const {
    return a < other.a;
  }
};

template <typename T, typename... Trest>
struct qsort_unaligned_t {
  // if there is space at the end of the nested struct that fits the current
  // element, then we should place it there, in order to be equivalent to
  // the LLVM structs.
  // The space inside the nested struct is generated for alignemnt reasons.
  // There is the non-standard __attribute__((packed)) but somethings breaks
  // thrust's sort. (maybe an alignment issue?)
  union {
    qsort_unaligned_t<Trest...> r;
    T a[((sizeof(qsort_helper_t<Trest...>) + alignof(T) - 1) / alignof(T)) + 1];
  };

  __host__ __device__ bool operator==(
      const qsort_unaligned_t<T, Trest...> &other) const {
    return (r == other.r) && (curr() == other.curr());
  }

  __host__ __device__ bool operator<(
      const qsort_unaligned_t<T, Trest...> &other) const {
    return (r == other.r) ? (curr() < other.curr()) : (r < other.r);
  }

  __host__ __device__ T curr() const {
    return a[(sizeof(qsort_helper_t<Trest...>) / sizeof(T))];
  }
};

template <typename T>
struct qsort_unaligned_t<T> {
  T a;

  __host__ __device__ bool operator==(const qsort_unaligned_t<T> &other) const {
    return a == other.a;
  }

  __host__ __device__ bool operator<(const qsort_unaligned_t<T> &other) const {
    return a < other.a;
  }

  __host__ __device__ T curr() const { return a; }
};

template <typename... T>
struct qsort_t {
 public:
  qsort_unaligned_t<T...> a;

 public:
  __host__ __device__ bool operator==(const qsort_t<T...> &other) const {
    return a == other.a;
  }

  __host__ __device__ bool operator<(const qsort_t<T...> &other) const {
    return a < other.a;
  }
};

template <typename... T>
void gpu_qsort(void *ptr, size_t N) {
  time_block t{"Tsort: "};
  typedef qsort_t<T...> to_sort_t;
  thrust::device_ptr<to_sort_t> mem{(to_sort_t *)ptr};
  assert(N * sizeof(to_sort_t) <= (DEFAULT_BUFF_CAP * sizeof(int32_t)) &&
         "Overflow in GPUSort's buffer");
  std::cout << "Sorting started..." << sizeof...(T) << " " << sizeof(to_sort_t)
            << " " << N << std::endl;
  thrust::sort(thrust::system::cuda::par, mem, mem + N);
  std::cout << "Sorting finished" << std::endl;
}

/**
 * WARNING: TEMPLATES should be in the REVERSE order wrt to the layout!!!!!!!!!!
 */

// 1 attribute:
extern "C" void qsort_i(void *ptr, size_t N) { gpu_qsort<int32_t>(ptr, N); }

extern "C" void qsort_l(void *ptr, size_t N) { gpu_qsort<int64_t>(ptr, N); }

// 2 attributes:
extern "C" void qsort_ii(void *ptr, size_t N) {
  gpu_qsort<int32_t, int32_t>(ptr, N);
}

extern "C" void qsort_il(void *ptr, size_t N) {
  gpu_qsort<int64_t, int32_t>(ptr, N);
}

extern "C" void qsort_li(void *ptr, size_t N) {
  gpu_qsort<int32_t, int64_t>(ptr, N);
}

extern "C" void qsort_ll(void *ptr, size_t N) {
  gpu_qsort<int64_t, int64_t>(ptr, N);
}

// 3 attributes:
extern "C" void qsort_iii(void *ptr, size_t N) {
  gpu_qsort<int32_t, int32_t, int32_t>(ptr, N);
}

extern "C" void qsort_iil(void *ptr, size_t N) {
  // this assertion is false! very unexpected...
  // static_assert(sizeof(int32_t, int32_t, int64_t>) == sizeof({int32_t,
  // int32_t, int64_t}), "!!!");
  gpu_qsort<int64_t, int32_t, int32_t>(ptr, N);
}

extern "C" void qsort_ili(void *ptr, size_t N) {
  gpu_qsort<int32_t, int64_t, int32_t>(ptr, N);
}

extern "C" void qsort_ill(void *ptr, size_t N) {
  gpu_qsort<int64_t, int64_t, int32_t>(ptr, N);
}

extern "C" void qsort_lii(void *ptr, size_t N) {
  gpu_qsort<int32_t, int32_t, int64_t>(ptr, N);
}

extern "C" void qsort_lil(void *ptr, size_t N) {
  gpu_qsort<int64_t, int32_t, int64_t>(ptr, N);
}

extern "C" void qsort_lli(void *ptr, size_t N) {
  gpu_qsort<int32_t, int64_t, int64_t>(ptr, N);
}

extern "C" void qsort_lll(void *ptr, size_t N) {
  gpu_qsort<int64_t, int64_t, int64_t>(ptr, N);
}

// 4 attributes:
extern "C" void qsort_iiii(void *ptr, size_t N) {
  gpu_qsort<int32_t, int32_t, int32_t, int32_t>(ptr, N);
}

extern "C" void qsort_iiil(void *ptr, size_t N) {
  gpu_qsort<int64_t, int32_t, int32_t, int32_t>(ptr, N);
}

extern "C" void qsort_iili(void *ptr, size_t N) {
  gpu_qsort<int32_t, int64_t, int32_t, int32_t>(ptr, N);
}

extern "C" void qsort_iill(void *ptr, size_t N) {
  gpu_qsort<int64_t, int64_t, int32_t, int32_t>(ptr, N);
}

extern "C" void qsort_ilii(void *ptr, size_t N) {
  gpu_qsort<int32_t, int32_t, int64_t, int32_t>(ptr, N);
}

extern "C" void qsort_ilil(void *ptr, size_t N) {
  gpu_qsort<int64_t, int32_t, int64_t, int32_t>(ptr, N);
}

extern "C" void qsort_illi(void *ptr, size_t N) {
  gpu_qsort<int32_t, int64_t, int64_t, int32_t>(ptr, N);
}

extern "C" void qsort_illl(void *ptr, size_t N) {
  gpu_qsort<int64_t, int64_t, int64_t, int32_t>(ptr, N);
}

extern "C" void qsort_liii(void *ptr, size_t N) {
  gpu_qsort<int32_t, int32_t, int32_t, int64_t>(ptr, N);
}

extern "C" void qsort_liil(void *ptr, size_t N) {
  gpu_qsort<int64_t, int32_t, int32_t, int64_t>(ptr, N);
}

extern "C" void qsort_lili(void *ptr, size_t N) {
  gpu_qsort<int32_t, int64_t, int32_t, int64_t>(ptr, N);
}

extern "C" void qsort_lill(void *ptr, size_t N) {
  gpu_qsort<int64_t, int64_t, int32_t, int64_t>(ptr, N);
}

extern "C" void qsort_llii(void *ptr, size_t N) {
  gpu_qsort<int32_t, int32_t, int64_t, int64_t>(ptr, N);
}

extern "C" void qsort_llil(void *ptr, size_t N) {
  gpu_qsort<int64_t, int32_t, int64_t, int64_t>(ptr, N);
}

extern "C" void qsort_llli(void *ptr, size_t N) {
  gpu_qsort<int32_t, int64_t, int64_t, int64_t>(ptr, N);
}

extern "C" void qsort_llll(void *ptr, size_t N) {
  gpu_qsort<int64_t, int64_t, int64_t, int64_t>(ptr, N);
}

extern "C" void qsort_lliiil(void *ptr, size_t N) {
  gpu_qsort<int64_t, int32_t, int32_t, int32_t, int64_t, int64_t>(ptr, N);
}

extern "C" void qsort_iillllllll(void *ptr, size_t N) {
  gpu_qsort<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
            int64_t, int32_t, int32_t>(ptr, N);
}

/**
 * WARNING: TEMPLATES should be in the REVERSE order wrt to the layout!!!!!!!!!!
 */
