#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import os
import sys
import argparse
import datetime

parser = argparse.ArgumentParser(description='Check licenses.')
parser.add_argument('--print-license', action='store_true',
                   help='print suggested license')
parser.add_argument('file', type=str, nargs='*',
                   help='files to check for correct license')

args = parser.parse_args()


projectline = r"""[^\n]+"""
year = r"""(\d+)"""
escchar = "\\"
optstart = r"""("""
optend = r""")?"""

if args.print_license:
    projectline = r"""Proteus -- High-performance query processing on heterogeneous hardware."""
    year = str(datetime.datetime.now().year)
    escchar = ""
    optstart = ""
    optend = ""

header = r"""/""" + escchar + r"""*""" + optstart + r"""
    """ + projectline + r"""
""" + optend + r"""
                            Copyright """ + escchar + r"""(c""" + escchar + r""") """ + year + r"""
        Data Intensive Applications and Systems Laboratory """ + escchar + r"""(DIAS""" + escchar + r""")
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
""" + escchar + r"""*/
"""

exts = [".cpp", ".hpp", ".cu", ".cuh", ".c", ".h"]

# Files that should not contain the header (usually files from external projects)
external_files = [
    "core/platform/util/radix/aggregations/prj_params.h",
    "core/platform/util/radix/aggregations/radix-aggr.cpp",
    "core/platform/util/radix/aggregations/radix-aggr.hpp",
    "core/platform/util/radix/aggregations/types.h",
    "core/platform/util/radix/joins/prj_params.h",
    "core/platform/util/radix/joins/radix-join.cpp",
    "core/platform/util/radix/joins/radix-join.hpp",
    "core/platform/util/radix/joins/types.h",
    "core/platform/util/radix/prj_params.h",
    "core/platform/util/radix/types.h",
    "jsmn/jsmn.c",
    "jsmn/jsmn.h"
    
    # TODO: fix licensing on following files:
    "apps/benchmarks/oltp/bench-runner.cpp",
    "apps/benchmarks/oltp/bench/micro_ssb.cpp",
    "apps/benchmarks/oltp/bench/tpcc.hpp",
    "apps/benchmarks/oltp/bench/ycsb.hpp",
    "apps/benchmarks/oltp/bench/tpcc_64.cpp",
    "apps/benchmarks/oltp/bench/micro_ssb.hpp",
    "apps/benchmarks/oltp/bench/tpcc.cpp",
    "core/htap/rm/resource_manager.hpp",
    "core/oltp/server.cpp",
    "core/oltp/adaptors/aeolus-plugin.cpp",
    "core/oltp/adaptors/include/aeolus-plugin.hpp",
    "core/oltp/engine/glo.cpp",
    "core/oltp/engine/snapshot/cor_arena.cpp",
    "core/oltp/engine/snapshot/cor_arena.hpp",
    "core/oltp/engine/snapshot/cor_const_arena.cpp",
    "core/oltp/engine/snapshot/cow_arena.cpp",
    "core/oltp/engine/include/glo.hpp",
    "core/oltp/engine/include/utils/spinlock.h",
    "core/oltp/engine/include/utils/lock.hpp",
    "core/oltp/engine/include/utils/utils.hpp",
    "core/oltp/engine/include/utils/atomic_bit_set.hpp",
    "core/oltp/engine/include/snapshot/cow_arena.hpp",
    "core/oltp/engine/include/snapshot/cor_const_arena.hpp",
    "core/oltp/engine/include/snapshot/circular_master_arena.hpp",
    "core/oltp/engine/include/snapshot/snapshot_manager.hpp",
    "core/oltp/engine/include/snapshot/arena.hpp",
    "core/oltp/engine/include/scheduler/affinity_manager.hpp",
    "core/oltp/engine/include/scheduler/comm_manager.hpp",
    "core/oltp/engine/include/scheduler/topology.hpp",
    "core/oltp/engine/include/scheduler/worker.hpp",
    "core/oltp/engine/include/storage/table.hpp",
    "core/oltp/engine/include/storage/memory_manager.hpp",
    "core/oltp/engine/include/transactions/cc.hpp",
    "core/oltp/engine/include/transactions/txn_utils.hpp",
    "core/oltp/engine/include/transactions/transaction_manager.hpp",
    "core/oltp/engine/include/interfaces/bench.hpp",
    "core/oltp/engine/include/indexes/hash_index.hpp",
    "core/oltp/engine/include/indexes/hash_array.hpp",
    "core/oltp/engine/scheduler/topology.cpp",
    "core/oltp/engine/scheduler/worker.cpp",
    "core/oltp/engine/scheduler/affinity_manager.cpp",
    "core/oltp/engine/scheduler/comm_manager.cpp",
    "core/oltp/engine/scheduler/threadpool.hpp",
    "core/oltp/engine/storage/memory_manager.cpp",
    "core/oltp/engine/storage/row_store.hpp",
    "core/oltp/engine/storage/column_store.cpp",
    "core/oltp/engine/storage/data_types.hpp",
    "core/oltp/engine/storage/delta_storage.cpp",
    "core/oltp/engine/storage/row_store.cpp",
    "core/oltp/engine/transactions/cc.cpp",
    "core/oltp/engine/transactions/transaction_manager.cpp",
    "core/oltp/engine/indexes/index.hpp",
    "core/oltp/engine/indexes/hash_array.cpp",
]

exclude_dirs = [
    "cmake-build-debug",
    ".idea/fileTemplates"
]

root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
found_bad = False

def check_files(files, d):
    global found_bad
    for f in files:
        for ext in exts:
            if (f.endswith(ext)):
                p = os.path.join(d, f)
                relpath = os.path.relpath(p, root)
                with open(p) as file:
                    if relpath in external_files:
                        # files in external_files are from external projects
                        # do not use our header
                        if (re.match(header, file.read())):
                            print("Remove our license: " + str(relpath))
                            found_bad = True
                    else:
                        if (not re.match(header, file.read())):
                            print("Missing license: " + str(relpath))
                            found_bad = True

if args.print_license:
    if len(args.file) > 0:
        print("WARNING: Ignoring passed files")
    sys.stdout.write(header)
elif len(args.file) > 0:
    check_files(args.file, root)
else:
    for d, subdirs, files in os.walk(root, followlinks=False):
        reldir = os.path.relpath(d, root)
        if reldir in exclude_dirs: continue
        if any([reldir.startswith(exclude + os.path.sep) for exclude in exclude_dirs]): continue
        check_files(files, d)

if found_bad:
    sys.exit(-1)
