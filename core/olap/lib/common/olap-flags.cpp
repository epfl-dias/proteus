/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2026
        Data Intensive Applications and Systems Laboratory (DIAS)
                École Polytechnique Fédérale de Lausanne

                            All Rights Reserved.
*/

#include <gflags/gflags.h>

DECLARE_bool(in_gpu);
DEFINE_bool(in_gpu, false, "Use Shaper::INGPU instead of Shaper::NVMEGPU");

DECLARE_bool(one_kernel);
DEFINE_bool(one_kernel, true,
            "If true, use the one-kernel-over-entire-dataset CpuToGpu path. "
            "Effective only when --in_gpu is also true.");
