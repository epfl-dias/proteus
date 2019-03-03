/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2014
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

#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"

// The lines below are copied from llvm/lib/IR/AsmWriter.cpp and modified
// to use weak linking.
// (Specifically from commit f38b0f184c9b94d59f9eec1a3c4d086d2746a5a7
// of https://github.com/llvm-mirror/llvm)
// The release builds of LLVM undefine dump(), but as we are constantly in
// development mode, we use them very often. So, we define them here to
// allow building the project with a pre-packaged LLVM and maintaining the
// full functionality.

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
// Value::dump - allow easy printing of Values from the debugger.
LLVM_DUMP_METHOD __attribute__((weak)) void llvm::Value::dump() const {
  print(dbgs(), /*IsForDebug=*/true);
  dbgs() << '\n';
}

// Type::dump - allow easy printing of Types from the debugger.
LLVM_DUMP_METHOD __attribute__((weak)) void llvm::Type::dump() const {
  print(dbgs(), /*IsForDebug=*/true);
  dbgs() << '\n';
}

// Module::dump() - Allow printing of Modules from the debugger.
LLVM_DUMP_METHOD __attribute__((weak)) void llvm::Module::dump() const {
  print(dbgs(), nullptr,
        /*ShouldPreserveUseListOrder=*/false, /*IsForDebug=*/true);
}

// Allow printing of Comdats from the debugger.
LLVM_DUMP_METHOD __attribute__((weak)) void llvm::Comdat::dump() const {
  print(dbgs(), /*IsForDebug=*/true);
}

// NamedMDNode::dump() - Allow printing of NamedMDNodes from the debugger.
LLVM_DUMP_METHOD __attribute__((weak)) void llvm::NamedMDNode::dump() const {
  print(dbgs(), /*IsForDebug=*/true);
}

LLVM_DUMP_METHOD __attribute__((weak)) void llvm::Metadata::dump() const {
  dump(nullptr);
}

LLVM_DUMP_METHOD __attribute__((weak)) void llvm::Metadata::dump(
    const llvm::Module *M) const {
  print(dbgs(), M, /*IsForDebug=*/true);
  dbgs() << '\n';
}

// Allow printing of ModuleSummaryIndex from the debugger.
LLVM_DUMP_METHOD __attribute__((weak)) void llvm::ModuleSummaryIndex::dump()
    const {
  print(dbgs(), /*IsForDebug=*/true);
}
#endif
