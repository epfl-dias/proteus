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

#include <algorithm>
#include <cmath>
#include <platform/util/percentile.hpp>

namespace proteus::utils {

std::map<std::string, Percentile*> PercentileRegistry::global_registry;
std::mutex PercentileRegistry::g_lock;

Percentile::Percentile(bool thread_safe) : m_threadsafe(thread_safe) {}

Percentile::Percentile(const std::string& key)
    : m_points{}, m_threadsafe(true) {
  PercentileRegistry::register_global(key, this);
}

size_t Percentile::nth(double n) {
  assert(n > 0 && n <= 100);

  if (m_points.empty()) {
    return 0;
  }

  // Sort the data m_points
  std::sort(m_points.begin(), m_points.end());

  auto sz = m_points.size();
  auto i = static_cast<decltype(sz)>(std::ceil(n / 100 * sz)) - 1;

  assert(i >= 0 && i < m_points.size());

  return m_points[i];
}

double Percentile::mean() const {
  if (m_points.size() == 0) {
    return -1;
  }
  double sum = 0;
  for (const auto& point : m_points) {
    sum += point;
  }

  return sum / static_cast<double>(m_points.size());
}

void Percentile::save_cdf(const std::string& out_path, size_t step) {
  if (m_points.empty()) {
    return;
  }

  if (out_path.empty()) {
    assert(false && "empty save path");
  }

  // Sort the data m_points
  std::sort(m_points.begin(), m_points.end());

  std::ofstream cdf;
  cdf.open(out_path);

  cdf << "value\tcdf" << std::endl;
  auto step_size = std::max(1, int(m_points.size() * 0.99 / step));

  std::deque<size_t> cdf_result;

  for (auto i = 0u; i < 0.99 * m_points.size(); i += step_size) {
    cdf_result.push_back(m_points[i]);
  }

  for (auto i = 0u; i < cdf_result.size(); i++) {
    cdf << cdf_result[i] << "\t" << 1.0 * (i + 1) / cdf_result.size()
        << std::endl;
  }

  cdf.close();
}

}  // namespace proteus::utils
