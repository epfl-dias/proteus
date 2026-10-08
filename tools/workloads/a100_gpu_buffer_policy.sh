#!/usr/bin/env bash
# Validated A100 memory-pool fractions for in-GPU revision queries. Keep this
# policy query-specific but codec/placement-independent so comparisons within a
# query use the same pool reservation.

declare -Ar A100_GPU_BUFFER_FRACTIONS=(
  [ssb:Q1_1]=0.8
  [ssb:Q2_1]=0.8
  [ssb:Q3_1]=0.8
  [ssb:Q4_1]=0.6
  [tpch:Q1]=0.6
  [tpch:Q3]=0.6
  [tpch:Q5]=0.6
  [tpch:Q6]=0.6
  [tpch:Q12]=0.6
  [tpch:Q14]=0.6
  [tpch:Q19]=0.6
  [stack:Q1]=0.8
  [stack:Q4]=0.8
  [stack:Q6]=0.8
  [stack:Q11]=0.8
  [stack:Q12]=0.8
  [stack:Q14]=0.8
)

a100_gpu_buffers_for_query() {
  local key="${1:?workload required}:${2:?query required}"
  echo "${A100_GPU_BUFFER_FRACTIONS[$key]:-0.6}"
}
