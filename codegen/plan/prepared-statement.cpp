/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2019
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

#include "plan/prepared-statement.hpp"
#include "topology/affinity_manager.hpp"
#include "topology/topology.hpp"

#if __has_include("ittnotify.h")
#include <ittnotify.h>
#else
#define __itt_resume() ((void)0)
#define __itt_pause() ((void)0)
#endif

void PreparedStatement::execute() {
  // just to be sure...
  for (const auto &gpu : topology::getInstance().getGpus()) {
    set_device_on_scope d{gpu};
    gpu_run(cudaDeviceSynchronize());
  }

  __itt_resume();

  {
    time_block t("Texecute w sync: ");

    {
      time_block t("Texecute       : ");

      for (Pipeline *p : pipelines) {
        nvtxRangePushA("pip");
        {
          time_block t("T: ");

          p->open();
          p->consume(0);
          p->close();

          std::cout << std::dec;
        }
        nvtxRangePop();
      }

      std::cout << std::dec;
    }

    // just to be sure...
    for (const auto &gpu : topology::getInstance().getGpus()) {
      set_device_on_scope d{gpu};
      gpu_run(cudaDeviceSynchronize());
    }
  }

  __itt_pause();
  for (const auto &gpu : topology::getInstance().getGpus()) {
    set_device_on_scope d{gpu};
    gpu_run(cudaProfilerStop());
  }
}
