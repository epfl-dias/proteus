/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2023
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

#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/ExecutorProcessControl.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/Mangling.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_os_ostream.h>

#include "codegen/test/context-test-utlis.hpp"

namespace codegen {

/**
 * @class DummyJITet
 * is used as a replacement for JITer_impl ONLY for Context unit tests.
 *
 * This class has only basic functionalities to support minimal working examples
 * of the code generation using Context. There is no optimizations for IR, so it
 * can be printed out.
 * @see JITer, JITer_impl
 */
class DummyJITer {
 public:
  DummyJITer(llvm::orc::JITTargetMachineBuilder JTMB,
             const llvm::DataLayout& DL_)
      : objectLayer(
            ES,
            []() { return std::make_unique<llvm::SectionMemoryManager>(); }),
        compileLayer(
            ES, objectLayer,
            std::make_unique<llvm::orc::ConcurrentIRCompiler>(std::move(JTMB))),
        DL(DL_),
        mangle(ES, DL),
        ctx(std::make_unique<llvm::LLVMContext>()),
        mainJD(cantFail(ES.createJITDylib("main"))) {}

  ~DummyJITer() {
    if (auto Err = ES.endSession()) ES.reportError(std::move(Err));
  }

  static std::unique_ptr<DummyJITer> Create() {
    auto JTMB = cantFail(llvm::orc::JITTargetMachineBuilder::detectHost());
    auto DL = cantFail(JTMB.getDefaultDataLayoutForTarget());

    return std::make_unique<DummyJITer>(std::move(JTMB), std::move(DL));
  }

  llvm::LLVMContext& getContext() { return *ctx.getContext(); }

  const llvm::DataLayout& getDataLayout() const { return DL; }

  void addModule(std::unique_ptr<llvm::Module> module) {
    llvm::cantFail(compileLayer.add(
        mainJD, llvm::orc::ThreadSafeModule(std::move(module), ctx)));
  }

  llvm::JITEvaluatedSymbol lookup(llvm::StringRef name) {
    return llvm::cantFail(ES.lookup({&mainJD}, mangle(name.str())));
  }

 private:
#if LLVM_VERSION_MAJOR >= 13
  llvm::orc::ExecutionSession ES{
      llvm::cantFail(llvm::orc::SelfExecutorProcessControl::Create())};
#else
  ExecutionSession ES;
#endif

  llvm::orc::RTDyldObjectLinkingLayer objectLayer;
  llvm::orc::IRCompileLayer compileLayer;
  llvm::DataLayout DL;
  llvm::orc::MangleAndInterner mangle;
  llvm::orc::ThreadSafeContext ctx;
  llvm::orc::JITDylib& mainJD;
};

/**
 * @class DummyCPUModule
 * is used as a replacement for CPUModule ONLY for Context unit tests.
 *
 * Does not contain compileAndLoad function, as this functionality was extracted
 * to the DummyTestContext for simplicity
 * @see CPUModule, DummyTestContext
 */
class DummyCPUModule {
 public:
  explicit DummyCPUModule(const std::string& moduleName)
      : jiter(DummyJITer::Create()),
        llvmCtx(jiter->getContext()),
        builder(llvm::IRBuilder<>(llvmCtx)),
        module(std::make_unique<llvm::Module>(moduleName, llvmCtx)) {
    module->setDataLayout(jiter->getDataLayout());
  }

  ~DummyCPUModule() = default;

  [[nodiscard]] llvm::Module* getModule() const { return module.get(); }

  [[nodiscard]] llvm::IRBuilder<>* getBuilder() { return &builder; }

  [[nodiscard]] llvm::LLVMContext& getContext() { return llvmCtx; }

  void addModule() { jiter->addModule(std::move(module)); }

  llvm::JITEvaluatedSymbol lookup(llvm::StringRef name) {
    return jiter->lookup(name);
  }

 private:
  std::unique_ptr<DummyJITer> jiter;

  llvm::LLVMContext& llvmCtx;
  llvm::IRBuilder<> builder;
  std::unique_ptr<llvm::Module> module;
};

DummyTestContext::DummyTestContext(const std::string& moduleName)
    : Context(moduleName),
      module(std::make_unique<DummyCPUModule>(moduleName)),
      voidType(llvm::Type::getVoidTy(module->getContext())),
      i32_type(llvm::Type::getInt32Ty(module->getContext())) {}

DummyTestContext::~DummyTestContext() = default;

void DummyTestContext::prepareFunction(llvm::Function* F) {}

llvm::Module* DummyTestContext::getModule() const {
  return module->getModule();
}

llvm::IRBuilder<>* DummyTestContext::getBuilder() const {
  return module->getBuilder();
}

void DummyTestContext::InitJIT() {
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();
}

llvm::Function* DummyTestContext::prepareTestFunction(
    llvm::StringRef name, llvm::Type* result,
    const std::vector<llvm::Type*>& arguments) {
  auto* fType = llvm::FunctionType::get(result, arguments, false);
  auto* function = llvm::Function::Create(
      fType, llvm::Function::ExternalLinkage, name, getModule());

  auto* bb = llvm::BasicBlock::Create(module->getContext(), "entry", function);
  getBuilder()->SetInsertPoint(bb);

  return function;
}

void DummyTestContext::finishTestFunction(bool dump) {
  llvm::verifyModule(*module->getModule());

  if (dump) {
    module->getModule()->print(llvm::errs(), nullptr);
  }

  module->addModule();
}

uintptr_t DummyTestContext::getCompiledRaw(llvm::StringRef name) {
  return static_cast<uintptr_t>(module->lookup(name).getAddress());
}

}  // namespace codegen
