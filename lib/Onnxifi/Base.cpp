/*
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "Base.h"

#include "glow/Importer/ONNXIFIModelLoader.h"

#include "llvm/Support/Format.h"

namespace glow {
namespace onnxifi {
namespace {
const char *compatibilityFunctionName = "check";
} // namespace

onnxStatus BackendId::checkGraphCompatibility(const void *onnxModel,
                                              size_t onnxModelSize) {
  Module module;

  auto function = module.createFunction(compatibilityFunctionName);

  std::unique_ptr<ONNXIFIModelLoader> loader;
  auto loaderOrErr = ONNXIFIModelLoader::parse(
      onnxModel, onnxModelSize, 0 /*weightCount*/,
      nullptr /*weightDescriptors*/, *function,
      false /*loadInputsAsPlaceholders*/, getUseOnnx());
  if (loaderOrErr) {
    loader = std::move(*loaderOrErr);
  } else {
    // TODO: Use a more specific ONNXIFI error code here to denote what about
    // this operator is not supported (shape, type, etc).
    llvm::errs() << "Error when loading protobuf: "
                 << llvm::toString(loaderOrErr.takeError()) << "\n";
    return ONNXIFI_STATUS_UNSUPPORTED_OPERATOR;
  }

  if (!glowBackend_) {
    return ONNXIFI_STATUS_INTERNAL_ERROR;
  }

  glow::lower(function, /* loweredMap */ nullptr, glowBackend_.get());

  // Call the backend's transformPostLowering to match the normal compilation
  // pipeline then DCE any nodes that are no longer needed.
  CompilationOptions opts;
  opts.mode = CompilationMode::Infer;
  if (glowBackend_->transformPostLowering(function, opts)) {
    glow::DCE(function);
  }

  const auto &nodes = function->getNodes();

  for (const auto &node : nodes) {
    if (!glowBackend_->isOpSupported(node)) {
      // TODO: Use a more specific ONNXIFI error code here to denote what about
      // this operator is not supported (shape, type, etc).
      return ONNXIFI_STATUS_UNSUPPORTED_OPERATOR;
    }
  }

  return ONNXIFI_STATUS_SUCCESS;
}

// static
std::unique_ptr<runtime::HostManager>
BackendId::createHostManager(glow::BackendKind kind) {
  std::vector<runtime::DeviceManagerConfig> configs;
  auto config = runtime::DeviceManagerConfig();
  config.deviceConfig = nullptr;
  config.backendKind = kind;
  configs.push_back(std::move(config));
  return llvm::make_unique<runtime::HostManager>(configs);
}

bool Event::signal() {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (fired_) {
      return false;
    }
    fired_ = true;
  }
  cond_.notify_all();
  return true;
}

void Event::wait() {
  std::unique_lock<std::mutex> guard(mutex_);
  cond_.wait(guard, [this] { return fired_ == true; });
}

onnxStatus Graph::initGraph(const void *onnxModel, size_t onnxModelSize,
                            uint32_t weightCount,
                            const onnxTensorDescriptorV1 *weightDescriptors) {

  auto id = makeUniqueGraphId();
  netName_ = llvm::formatv("inference_function_%d", id);

  Function *function = m_.createFunction(netName_);

  // TODO: make better error reporting.
  std::unique_ptr<ONNXIFIModelLoader> loader =
      TEMP_EXIT_ON_ERR(ONNXIFIModelLoader::parse(
          onnxModel, onnxModelSize, weightCount, weightDescriptors, *function,
          true /*loadInputsAsPlaceholders*/, backendPtr_->getUseOnnx()));

  onnxInputToPlaceholder_ = loader->getInputVarsMapping();
  onnxOutputToPlaceholder_ = loader->getOutputVarsMapping();

  auto res = backendPtr_->getHostManager().addNetwork(&m_);

  // TODO: add higher resolution error reporting
  if (res != runtime::ResultCode::Ready) {
    return ONNXIFI_STATUS_INTERNAL_ERROR;
  }

  return ONNXIFI_STATUS_SUCCESS;
}

// static
size_t Graph::makeUniqueGraphId() {
  static std::atomic<size_t> nextId{0};
  return nextId++;
}

void Graph::run(
    const llvm::DenseMap<Placeholder *, onnxPointer> &inputPlaceholderToBuffer,
    const llvm::DenseMap<Placeholder *, onnxPointer>
        &outputPlaceholderToBuffer) {
  // Copy tensors from the input addresses to the Glow tensors.
  llvm::SmallVector<Tensor *, 4> tensors;
  llvm::SmallVector<Placeholder *, 4> phs;
  for (auto inputVar : inputPlaceholderToBuffer) {
    auto *var = inputVar.first;
    auto *type = var->getType();
    void *inputBuffer = reinterpret_cast<void *>(inputVar.second);
    tensors.push_back(new Tensor(inputBuffer, type));
    phs.push_back(var);
  }

  auto bindings = llvm::make_unique<PlaceholderBindings>();

  // Run inference.
  auto &mod = executionEngine_.getModule();
  bindings->allocate(mod.getPlaceholders());
  updateInputPlaceholders(*bindings, phs, tensors);

  // Lambda capturing work to do after the graph has finished running.
  auto afterRun = [tensors = std::move(tensors), outputPlaceholderToBuffer](
                      std::unique_ptr<glow::PlaceholderBindings> bindings) {
    // Tensors do not own underlying memory for input buffer,
    // just delete memory allocated for the tensor object itself.
    for (size_t i = 0; i < tensors.size(); ++i) {
      delete tensors[i];
    }

    // Copy output data from the graph to the onnxifi outputs.
    for (auto &outputVar : outputPlaceholderToBuffer) {
      void *outputAddress = reinterpret_cast<void *>(outputVar.second);
      Tensor *res = bindings->get(outputVar.first);
      memcpy(outputAddress, res->getUnsafePtr(),
             res->size() * res->getType().getElementSize());
    }
  };

  if (backendPtr_->getUseHostManager()) {
    backendPtr_->runOnHostManager(
        inferenceFunctionName, std::move(bindings),
        [afterRun = std::move(afterRun)](
            int runIdentifier, int resultCode,
            std::unique_ptr<glow::PlaceholderBindings> bindings) {
          afterRun(std::move(bindings));
        });
  } else {
    executionEngine_.run(*bindings);
    afterRun(std::move(bindings));
  }
}

Graph::Graph(BackendPtr backendPtr) : backendPtr_(backendPtr) {}

Graph::~Graph() {
  // Remove network from hostmanager
  // backendPtr_->getHostManager().removeNetwork(netName_);
}

onnxStatus Graph::setIOAndRunAsync(
    uint32_t inputsCount, const onnxTensorDescriptorV1 *inputDescriptors,
    uint32_t outputsCount, const onnxTensorDescriptorV1 *outputDescriptors,
    EventPtr outputEvent) {

  auto ctx = llvm::make_unique<Context>();

  // Create tensors for input placeholders
  for (unsigned i = 0; i < inputsCount; ++i) {
    const auto &inOnnxTensor = inputDescriptors[i];
    auto *inOnnxBuffer = reinterpret_cast<void *>(inOnnxTensor.buffer);

    auto inPhIt = onnxInputToPlaceholder_.find(inOnnxTensor.name);
    if (inPhIt == onnxInputToPlaceholder_.end()) {
      return ONNXIFI_STATUS_UNIDENTIFIED_NAME;
    }

    auto &inPhPtr = inPhIt->getValue();

    Tensor t(inOnnxBuffer, inPhPtr->getType());

    ctx->insert(inPhPtr, std::move(t));
  }

  // Create tensors for output placeholders
  for (unsigned i = 0; i < outputsCount; ++i) {
    const auto &outOnnxTensor = outputDescriptors[i];
    auto *outOnnxBuffer = reinterpret_cast<void *>(outOnnxTensor.buffer);

    auto outPhIt = onnxOutputToPlaceholder_.find(outOnnxTensor.name);
    if (outPhIt == onnxOutputToPlaceholder_.end()) {
      return ONNXIFI_STATUS_UNIDENTIFIED_NAME;
    }

    auto &outPhPtr = outPhIt->getValue();

    Tensor t(outOnnxBuffer, outPhPtr->getType());

    ctx->insert(outPhPtr, std::move(t));
  }

  // Run
  getHostManager().runNetwork(
      netName_, std::move(ctx),
      [outputEvent](runtime::RunIdentifierTy runId, runtime::ResultCode result,
                    std::unique_ptr<Context> ctx) { outputEvent->signal(); });

  return ONNXIFI_STATUS_SUCCESS;
}

} // namespace onnxifi
} // namespace glow
