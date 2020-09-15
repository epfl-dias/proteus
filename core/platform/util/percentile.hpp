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

#ifndef PROTEUS_PERCENTILE_HPP
#define PROTEUS_PERCENTILE_HPP

#include <vector>
#include <cassert>
#include <fstream>

namespace proteus::utils {

template <class T> class Percentile {
 public:


  Percentile(){}
  Percentile(size_t reserved_capacity){
    points.reserve(reserved_capacity);
  }

  inline void reserve(size_t quantity){
    points.reserve(quantity);
  }

  inline void add(const T &value) {
    points.push_back(value);
  }

  inline void add(const Percentile &p){
    std::copy(p.points.begin(), p.points.end(), std::back_inserter(points));
  }

  inline void add(const std::vector<T> &v) {
    std::copy(v.begin(), v.end(), std::back_inserter(points));
  }



  T nth(double n) {
    assert(n > 0 && n <= 100);

    if (points.size() == 0) {
      return 0;
    }

    // Sort the data points
    std::sort(points.begin(), points.end());

    auto sz = points.size();
    auto i = static_cast<decltype(sz)>(ceil(n / 100 * sz)) - 1;

    assert(i >= 0 && i < points.size());

    return points[i];
  }


  // shouldn't be on critical path.
  void save_cdf(const std::string &path, size_t step = 1000) {
    if (points.size() == 0) {
      return;
    }

    if (path.empty()) {
      return;
    }

    // Sort the data points
    std::sort(points.begin(), points.end());

    std::ofstream cdf;
    cdf.open(path);

    cdf << "value\tcdf" << std::endl;
    auto step_size = std::max(1, int(points.size() * 0.99 / step));

    std::vector<T> cdf_result;

    for (auto i = 0u; i < 0.99 * points.size(); i += step_size) {
      cdf_result.push_back(points[i]);
    }

    for (auto i = 0u; i < cdf_result.size(); i++) {
      cdf << cdf_result[i] << "\t" << 1.0 * (i + 1) / cdf_result.size()
          << std::endl;
    }

    cdf.close();
  }


  Percentile operator+(const Percentile& p) {
    Percentile tmp;
    tmp.add(this->points);
    tmp.add(p.points);
    return tmp;
  }

  T operator[](double n) {
    assert(n > 0 && n <= 100);
    return nth(n);
  }

 private:
  std::vector<T> points;
};
}

#endif  // PROTEUS_PERCENTILE_HPP
