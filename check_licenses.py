#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import os
import sys

header = r"""/\*(
    [^\n]+
)?
                            Copyright \(c\) (\d+)
        Data Intensive Applications and Systems Laboratory \(DIAS\)
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
\*/
"""

exts = [".cpp", ".hpp", ".cu", ".cuh", ".c", ".h"]

# Files that should not contain the header (usually files from external projects)
external_files = [
    "codegen/util/radix/prj_params.h",
    "codegen/util/radix/types.h",
    "codegen/util/radix/joins/radix-join.cpp",
    "codegen/util/radix/joins/prj_params.h",
    "codegen/util/radix/joins/types.h",
    "codegen/util/radix/joins/radix-join.hpp",
    "codegen/util/radix/aggregations/prj_params.h",
    "codegen/util/radix/aggregations/types.h",
    "codegen/util/radix/aggregations/radix-aggr.cpp",
    "codegen/util/radix/aggregations/radix-aggr.hpp",
    "jsmn/jsmn.c",
    "jsmn/jsmn.h",
]

root = os.path.dirname(os.path.realpath(__file__))
found_bad = False

for d, subdirs, files in os.walk(root, followlinks=False):
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

if found_bad:
    sys.exit(-1)
