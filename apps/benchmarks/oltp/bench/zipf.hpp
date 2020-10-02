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

#ifndef PROTEUS_ZIPF_HPP
#define PROTEUS_ZIPF_HPP

#define USE_DRAND false

#include <cassert>
#include <cmath>
#include <random>
#include <type_traits>

#include "memory/memory-manager.hpp"
#include "storage/table.hpp"
#include "topology/affinity_manager.hpp"
#include "topology/topology.hpp"

namespace bench_utils {

template <typename T>
class ZipfianGenerator {
 public:
  static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>,
                "ZipfianGenerator must be instantiated with unsigned integral "
                "type template only.");



  ZipfianGenerator(size_t n_records, double theta, size_t n_workers = 1,
                   size_t n_partitions = 1, bool partition_local = false,
                   bool worker_local = false)
      : _n_partitions(n_partitions),
        _n_workers(n_workers),
        _n_workers_per_partition(std::ceil((double)n_workers / n_partitions)),
        _n(n_records - 1),
        _seed_initializer(0),
        _partition_local(partition_local),
        _worker_local(worker_local) {
    _n_per_part = (n_records / n_partitions) - 1;
    _n_per_worker = (n_records / _n_workers_per_partition) - 1;
    _theta = theta;

    auto n_to_use = _n;
    if (_worker_local)
      n_to_use = _n_per_worker;
    else if (_partition_local)
      n_to_use = _n_per_part;
    else
      n_to_use = _n;

    _alpha = 1.0 / (1.0 - theta);
    _zetan = zeta(n_to_use, theta);
    _eta = (1.0 - std::pow(2.0 / n_to_use, 1.0 - theta)) /
           (1.0 - zeta(2, theta) / (zeta(n_to_use, theta)));

#if USE_DRAND

    this->rand_buffer = (struct drand48_data **)MemoryManager::mallocPinned(
        n_partitions * sizeof(struct drand48_data *));
    for (uint i = 0; i < n_partitions; i++) {
      exec_location{
          topology::getInstance()
              .getCpuNumaNodes()[storage::NUMAPartitionPolicy::getInstance()
                                     .getPartitionInfo(i)
                                     .numa_idx]}
          .activate();

      rand_buffer[i] = (struct drand48_data *)MemoryManager::mallocPinned(
          std::ceil(n_workers / n_partitions) * sizeof(struct drand48_data));
    }

    int c = 0;
    for (ushort i = 0; i < n_partitions; i++) {
      for (int j = 0; j < (n_workers / n_partitions); j++) {
        srand48_r(c++, &rand_buffer[i][j]);
      }
    }
#endif
  }

  ~ZipfianGenerator() {
#if USE_DRAND
    for (ushort i = 0; i < this->_n_partitions; i++) {
      MemoryManager::freePinned(rand_buffer[i]);
    }
    MemoryManager::freePinned(rand_buffer);
#endif
  }

  inline auto nextval(uint partition_id = 0, uint worker_id = 0) {
    return (*this)(partition_id, worker_id);
  }

  auto inline operator()(uint partition_id = 1, uint worker_id = 1) const {
    double u;

#if USE_DRAND
    drand48_r(&rand_buffer[partition_id][worker_id % _n_workers_per_partition],
              &u);
#else
    // FIXME: I am not sure about having thread-local random generator,
    //  would this be correct distribution generator or it will cause problems?
    static thread_local std::mt19937 engine(std::random_device{}());
    static thread_local std::uniform_real_distribution<double> dist{0.0, 1.0};
    u = dist(engine);
#endif

    if (_partition_local)
      return (partition_id * _n_per_part) + this->val(u);
    else if (_worker_local)
      return (worker_id * _n_per_worker) + this->val(u);
    else
      return this->val(u);
  }

 private:
  inline double zeta(size_t n, double theta_z) {
    double sum = 0;
    for (uint64_t i = 1; i <= n; i++) sum += std::pow(1.0 / i, theta_z);
    return sum;
  }

  inline double val(double u) const {
    static thread_local double alpha_half_pow = 1 + pow(0.5, _theta);
    static thread_local auto _thread_local_zetan = _zetan;
    static thread_local auto _thread_local_eta = _eta;
    static thread_local auto _thread_local_alpha = _alpha;
    static thread_local auto _n_rec =
        (_worker_local ? _n_per_worker : (_partition_local ? _n_per_part : _n));

    double uz = u * _thread_local_zetan;
    int v;
    if (uz < 1) {
      v = 0;
    } else if (uz < 1 + alpha_half_pow) {
      v = 1;
    } else {
      v = static_cast<T>(_n_rec *
                         std::pow(_thread_local_eta * u - _thread_local_eta + 1,
                                  _thread_local_alpha));
    }
    assert(v >= 0 && v < _n_rec);
    return v;
  }

  const size_t _n;
  size_t _n_per_part;
  size_t _n_per_worker;
  double _theta;
  double _alpha;
  double _zetan;
  double _eta;

  const std::atomic<size_t> _seed_initializer{};

  const ushort _n_partitions;
  const uint _n_workers;
  const uint _n_workers_per_partition;
  const bool _partition_local;
  const bool _worker_local;

#if USE_DRAND
  struct drand48_data **rand_buffer;
#endif
};

};  // namespace utils

#endif  // PROTEUS_ZIPF_HPP
