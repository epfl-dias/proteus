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

#include "percentile.hpp"

namespace proteus::utils {

std::map<std::string, Percentile*> PercentileRegistry::global_registry;
std::mutex PercentileRegistry::g_lock;

Percentile::Percentile(std::string key) : Percentile() {
  PercentileRegistry::register_global(key, this);
}

size_t Percentile::nth(double n) {
  assert(n > 0 && n <= 100);

  if (points.empty()) {
    return 0;
  }

  // Sort the data points
  std::sort(points.begin(), points.end());

  auto sz = points.size();
  auto i = static_cast<decltype(sz)>(ceil(n / 100 * sz)) - 1;

  assert(i >= 0 && i < points.size());

  return points[i];
}

void Percentile::save_cdf(const std::string& out_path, size_t step) {
  if (points.empty()) {
    return;
  }

  if (out_path.empty()) {
    assert(false && "empty save path");
  }

  // Sort the data points
  std::sort(points.begin(), points.end());

  std::ofstream cdf;
  cdf.open(out_path);

  cdf << "value\tcdf" << std::endl;
  auto step_size = std::max(1, int(points.size() * 0.99 / step));

  std::deque<size_t> cdf_result;

  for (auto i = 0u; i < 0.99 * points.size(); i += step_size) {
    cdf_result.push_back(points[i]);
  }

  for (auto i = 0u; i < cdf_result.size(); i++) {
    cdf << cdf_result[i] << "\t" << 1.0 * (i + 1) / cdf_result.size()
        << std::endl;
  }

  cdf.close();
}

}  // namespace proteus::utils
