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

class cuda_cor_const_arenas_t : public cor_const_arenas_t {
 public:
  static void init(size_t size_bytes) {
    cor_const_arenas_t::init(size_bytes);
    // gpu_run(cudaHostRegister(cor_const_arenas_t::olap_arena, size_bytes, 0));
    // gpu_run(cudaHostRegister(cor_const_arenas_t::oltp_arena, size_bytes, 0));
  }
};