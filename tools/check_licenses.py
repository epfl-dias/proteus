#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import os
import sys
import argparse
import datetime
import subprocess

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
ending = optstart + r""" """ + optend

if args.print_license:
    projectline = r"""Proteus -- High-performance query processing on heterogeneous hardware."""
    year = str(datetime.datetime.now().year)
    escchar = ""
    optstart = ""
    optend = ""
    ending = ""

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
""" + ending + escchar + r"""*/
"""

exts = [".cpp", ".hpp", ".cu", ".cuh", ".c", ".h", ".scala", ".java"]

# Files that should not contain the header (usually files from external projects)
external_files = [
    "core/platform/include/platform/util/radix/aggregations/prj_params.h",
    "core/platform/include/platform/util/radix/aggregations/radix-aggr.hpp",
    "core/platform/include/platform/util/radix/aggregations/types.h",
    "core/platform/include/platform/util/radix/joins/prj_params.h",
    "core/platform/include/platform/util/radix/joins/radix-join.hpp",
    "core/platform/include/platform/util/radix/joins/types.h",
    "core/platform/include/platform/util/radix/prj_params.h",
    "core/platform/include/platform/util/radix/types.h",
    "core/platform/lib/util/radix/aggregations/radix-aggr.cpp",
    "core/platform/lib/util/radix/joins/radix-join.cpp",
    "external/jsmn/jsmn.c",
    "external/jsmn/include/jsmn.h",
    "core/planner/src/main/scala/ch/epfl/dias/emitter/PlanToJSON.scala",
    "core/planner/src/main/scala/ch/epfl/dias/emitter/PlanConversionException.scala",
    "core/planner/src/main/scala/ch/epfl/dias/repl/Repl.scala",
    "core/planner/src/main/scala/ch/epfl/dias/calcite/adapter/pelago/costs/RelCost.scala",
    "core/planner/src/main/scala/ch/epfl/dias/calcite/adapter/pelago/costs/CostModel.scala",
    "core/planner/src/main/scala/ch/epfl/dias/calcite/adapter/pelago/ddl/PelagoDdlExecutor.java",
    "core/planner/src/main/scala/ch/epfl/dias/calcite/adapter/pelago/rules/LikeToJoinRule.scala",
    "core/planner/src/main/scala/ch/epfl/dias/calcite/adapter/pelago/schema/PelagoTableFactory.java",
    "core/planner/src/main/scala/ch/epfl/dias/calcite/adapter/pelago/schema/PelagoSchemaFactory.java",
    "core/planner/src/main/scala/ch/epfl/dias/calcite/adapter/pelago/schema/PelagoResultTable.java",
    "core/planner/src/main/scala/ch/epfl/dias/calcite/adapter/pelago/schema/MalformedPlugin.java",
    "core/planner/src/main/scala/ch/epfl/dias/calcite/adapter/pelago/schema/PelagoSchema.java",
    "core/planner/src/main/scala/ch/epfl/dias/calcite/adapter/pelago/executor/PelagoExecutor.scala",
    "core/planner/src/main/scala/ch/epfl/dias/calcite/adapter/pelago/executor/PelagoEnumerator.java",
    "core/planner/src/main/scala/ch/epfl/dias/calcite/adapter/pelago/rel/PelagoResultScan.java",
    "core/planner/src/main/scala/ch/epfl/dias/calcite/adapter/pelago/rel/PelagoFilter.scala",
    "core/planner/src/main/scala/ch/epfl/dias/calcite/adapter/pelago/rel/PelagoDeviceCross.scala",
    "core/planner/src/main/scala/ch/epfl/dias/calcite/adapter/pelago/rel/PelagoLogicalJoin.scala",
    "core/planner/src/main/scala/ch/epfl/dias/calcite/adapter/pelago/rel/PelagoValues.scala",
    "core/planner/src/main/scala/ch/epfl/dias/calcite/adapter/pelago/rel/PelagoUnion.scala",
    "core/planner/src/main/scala/ch/epfl/dias/calcite/adapter/pelago/rel/PelagoUnnest.scala",
    "core/planner/src/main/scala/ch/epfl/dias/calcite/adapter/pelago/rel/PelagoSort.scala",
    "core/planner/src/main/scala/ch/epfl/dias/calcite/adapter/pelago/rel/PelagoProject.scala",
    "core/planner/src/main/scala/ch/epfl/dias/calcite/adapter/pelago/rel/PelagoRel.scala",
    "core/planner/src/main/scala/ch/epfl/dias/calcite/adapter/pelago/rel/PelagoRouter.scala",
    "core/planner/src/main/scala/ch/epfl/dias/calcite/adapter/pelago/rel/PelagoDictTableScan.scala",
    "core/planner/src/main/scala/ch/epfl/dias/calcite/adapter/pelago/rel/PelagoJoin.scala",
    "core/planner/src/main/scala/ch/epfl/dias/calcite/adapter/pelago/rel/PelagoAggregate.scala",
    "core/planner/src/main/scala/ch/epfl/dias/calcite/adapter/pelago/rel/PelagoPack.scala",
    "core/planner/src/main/scala/ch/epfl/dias/calcite/adapter/pelago/rel/PelagoUnpack.scala",
    "core/planner/src/main/scala/ch/epfl/dias/calcite/adapter/pelago/rel/PelagoTableModify.scala",
    "core/planner/src/main/scala/ch/epfl/dias/calcite/adapter/pelago/rel/PelagoTableScan.scala",
    "core/planner/src/main/scala/ch/epfl/dias/calcite/adapter/pelago/jdbc/Driver.java",
    "core/planner/src/main/scala/org/apache/calcite/plan/volcano/PelagoCostFactory.java",
    "core/planner/src/main/scala/org/apache/calcite/plan/volcano/PelagoCost.java",
    "core/planner/src/main/scala/org/apache/calcite/plan/volcano/StealSetId.scala",
    "core/planner/src/main/scala/org/apache/calcite/sql/ddl/SqlCreatePelagoTable.java",
    "core/planner/src/main/scala/org/apache/calcite/sql/ddl/SqlDdlPelagoNodes.java",
    "core/planner/src/main/scala/org/apache/calcite/sql/ddl/package-info.java",
    "core/planner/src/main/scala/org/apache/calcite/prepare/PelagoPrepareImpl.java",
    "core/planner/src/main/scala/org/apache/calcite/prepare/PelagoPreparingStmt.java",
    "core/planner/src/main/scala/org/apache/calcite/jdbc/PelagoMetaImpl.java",
    "core/planner/src/main/scala/org/apache/calcite/server/PelagoMutableArrayTable.java",
    "core/planner/src/test/scala/ch/epfl/dias/calcite/adapter/pelago/PelagoPlannerTest.java",
    "core/planner/src/test/scala/ch/epfl/dias/calcite/adapter/pelago/PelagoTestConnectionFactory.java",
    "core/planner/src/test/scala/ch/epfl/dias/calcite/adapter/pelago/PelagoQueryTest.java",
]

exclude_dirs = [
    "cmake-build-debug",
    ".idea/fileTemplates"
]

root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
found_bad = False

def check_files(files, d, ignore_gitignored_files):
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
                            if (not ignore_gitignored_files) or subprocess.Popen(["git", "check-ignore", "-q", relpath]).wait() != 0:
                                print("Missing license: " + str(relpath))
                                found_bad = True

if args.print_license:
    if len(args.file) > 0:
        print("WARNING: Ignoring passed files")
    sys.stdout.write(header)
elif len(args.file) > 0:
    check_files(args.file, root, False)
else:
    for d, subdirs, files in os.walk(root, followlinks=False):
        reldir = os.path.relpath(d, root)
        if reldir in exclude_dirs: continue
        if any([reldir.startswith(exclude + os.path.sep) for exclude in exclude_dirs]): continue
        check_files(files, d, True)

if found_bad:
    sys.exit(-1)
