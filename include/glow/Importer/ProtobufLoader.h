/**
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

#ifndef GLOW_IMPORTER_PROTOBUFLOADER_H
#define GLOW_IMPORTER_PROTOBUFLOADER_H

#include "glow/Base/Tensor.h"
#include "glow/Graph/Graph.h"

#include "glow/Support/Error.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <google/protobuf/text_format.h>

#include <memory>
#include <string>
#include <vector>

/// This is the maximum allowed protobuf size (2GB).
#define MAX_PROTO_SIZE 0x7FFFFFFF

namespace glow {

/// Returns true iff all elements of \p a are the same.
bool isArrayConstant(const llvm::ArrayRef<size_t> a);

/// Prints a single serialized protocol buffer node. This method is useful for
/// debugging the network and printing errors.
template <typename T> std::string nodeToString(const T &node) {
  std::string str;
  google::protobuf::TextFormat::PrintToString(node, &str);
  return str;
}

/// Reads a single integer.
template <typename T> static llvm::Expected<int> loadInt(const T *arg) {
  RETURN_ERR_IF_NOT(arg->has_i(), "Node has no Int value");
  return arg->i();
}

/// Reads a single float.
template <typename T> static llvm::Expected<float> loadFloat(const T *arg) {
  RETURN_ERR_IF_NOT(arg->has_f(), "Node has no float value");
  return arg->f();
}

/// Reads a single string.
template <typename T>
static llvm::Expected<const std::string &> loadStr(const T *arg) {
  RETURN_ERR_IF_NOT(arg->has_s(), "Node has no str value");
  return arg->s();
}

/// Load the 'shape' record into a vector of sizes.
template <typename ElemTy = size_t, typename AttrType>
std::vector<ElemTy> getShape(const AttrType *arg) {
  std::vector<ElemTy> dim;
  for (auto i : arg->ints()) {
    dim.push_back(i);
  }
  return dim;
}

/// Returns canonical name for a given operator: either \p name() from proto,
/// or its first output's name.
template <typename T> std::string loadOperatorName(const T &op) {
  return op.name().length() ? op.name() : op.output(0);
}

/// Loads model: graph and weights.
class ProtobufLoader {
protected:
  /// The graph that we are constructing.
  Function &G_;
  /// Saves network nodes by name.
  llvm::StringMap<NodeValue> nodeValueByName_;
  /// A list of weight tensors indexed by name.
  llvm::StringMap<std::unique_ptr<Tensor>> tensors_;
  /// A map from names of the external outputs of the network to Variables.
  llvm::StringMap<Placeholder *> outputVarsByName_;

  /// \returns the tensor that was registered under the name \p name.
  llvm::Expected<Tensor *> getTensorByName(llvm::StringRef name);

  /// Create a new constant that's initialized with \p tensor, and register it
  /// under the name \p name. \returns The newly created constant.
  llvm::Expected<Constant *> createAndRegisterConstant(llvm::StringRef name,
                                                       const Tensor &tensor);

  /// Create a new Placeholder of type \p T, and register it
  /// under the name \p name. \returns The newly created placeholder.
  llvm::Expected<Placeholder *>
  createAndRegisterPlaceholder(llvm::StringRef name, TypeRef T);

  /// \returns the NodeValue that was registered with the name \p name or
  /// a nullptr wrapped in a NodeValue if no node has been registered with this
  /// name.
  NodeValue getNodeValueByNameOrNullNodeValue(llvm::StringRef name) const;

public:
  /// \returns the NodeValue that was registered with the name \p name.
  /// \pre hasNodeByName(name)
  llvm::Expected<NodeValue> getNodeValueByName(llvm::StringRef name) const;

  /// \returns the NodeValue that was registered with the name \p name or create
  /// a new Constant for a tensor with this name. In case a new constant is
  /// created, this method registers it under \p name.
  llvm::Expected<NodeValue>
  getNodeValueOrCreateConstantByName(llvm::StringRef name);

  /// \returns True if the node that's registered using \p name exists.
  bool hasNodeByName(llvm::StringRef name) const;

  /// Constructs new ProtobufLoader object. It will populate the network into \p
  /// F. The list \p types and \p names are used to initialized the inputs and
  /// outputs with specific names and types.
  /// If \p errPtr is not null then if an error occurs it will get assigned
  /// there otherwise if an error occurs it will abort.
  ProtobufLoader(llvm::ArrayRef<const char *> tensorNames,
                 llvm::ArrayRef<TypeRef> types, Function &F,
                 llvm::Error *errPtr = nullptr);

  ProtobufLoader(const ProtobufLoader &other) = delete;
  ProtobufLoader &operator=(const ProtobufLoader &) = delete;
  virtual ~ProtobufLoader() = default;

  /// \returns mapping between ONNX names and actual Glow output nodes.
  const llvm::StringMap<Placeholder *> &getOutputVarsMapping() const {
    return outputVarsByName_;
  }

  /// \returns the single final output of the network. The function assumes that
  /// there is only one output, returns Error otherwise. For image
  /// classification, this single final output is usually the result of the last
  /// softmax or regression layer.
  llvm::Expected<Placeholder *> getSingleOutput() {
    RETURN_ERR_IF_NOT(outputVarsByName_.size() == 1,
                      "There must be only one output.");
    return outputVarsByName_.begin()->second;
  }

  /// \returns the Placeholder for the external output with \p name.
  /// \pre outputVarsByName_.find(name) != outputVarsByName_.end()
  llvm::Expected<Placeholder *> getOutputByName(llvm::StringRef name) const;
};

} // namespace glow

#endif // GLOW_IMPORTER_PROTOBUFLOADER_H
