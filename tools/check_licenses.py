#!/usr/bin/env python

import getopt
import os
import re
import sys


def usage(progname, retval=0):
    print("%s [-r <folder>]" % progname)
    print("\t-r <folder>           \tBase folder for which to check the licenses,"
          "\r\t                    \tby default the current working "
          "directory.")
    print("\t--print-license       \tPrint suggested license"
          "directory.")
    print("")

    sys.exit(retval)


#############################################################################
# Manage parameters & global symbols

def check(root):
    with open("LICENSE") as file_name:
        header = file_name.read()

    # Replace years with regex
    copyright_match = r"""Copyright \(c\) \d+-\d+"""
    copyright_insert = r"""Copyright (c) \d+-\d+"""
    header = re.sub(copyright_match, copyright_insert, header)
    header = re.sub(r"([()])", r"\\\g<1>", header)

    # Reconstruct C++ header
    header = r"""^/\*\n(.*\n\n)?""" + header + r"""\*/\n"""

    header = re.compile(header, re.MULTILINE)
    exts = [".cpp", ".hpp", ".cu", ".cuh", ".c", ".h"]

    # Files that should not contain the header (usually files from external
    # projects)
    external_files = [
    "lib/cxxopts.hpp"
    ]

    found_bad = False

    for d, subdirs, files in os.walk(root, followlinks=False):
        for file_name in files:
            for ext in exts:
                if file_name.endswith(ext):
                    p = os.path.join(d, file_name)
                    relpath = os.path.relpath(p, root)
                    with open(p) as file_handle:
                        if relpath in external_files:
                            # files in external_files are from external projects
                            # do not use our header
                            if header.search(file_handle.read()):
                                print("Remove our license: " + str(relpath))
                                found_bad = True
                        else:
                            if not header.search(file_handle.read()):
                                print("Missing license: " + str(relpath))
                                found_bad = True

    if found_bad:
        sys.exit(-1)

def print_license():
    print(r"""/*
                  AEOLUS - In-Memory HTAP-Ready OLTP Engine
""")
    with open("LICENSE") as file_name:
        header = file_name.read()
    print(header + r"""*/""")

def main(argv):
    progname = argv[0]
    root = os.getcwd()

    try:
        opts, args = getopt.getopt(argv[1:], 'r:', ['print-license'])
    except getopt.GetoptError:
        usage(progname, 1)

    for opt, arg in opts:
        if opt == '-r':
            root = arg
        elif opt == '-h':
            usage(progname)
        elif opt == '--print-license':
            print_license()
        else:
            usage(progname, 1)

    root = os.path.realpath(root)
    assert(os.path.isdir(root))

    check(root)


if __name__ == "__main__":
    main(sys.argv)
