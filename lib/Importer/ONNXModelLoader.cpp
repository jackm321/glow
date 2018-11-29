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

#include "glow/Importer/ONNXModelLoader.h"
#include "glow/Base/Tensor.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "onnx/onnx_pb.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace glow;
using llvm::cast;

using ArgumentDictionaryTy =
    std::unordered_map<std::string, const ONNX_NAMESPACE::AttributeProto *>;

/// Translates the protocol buffer node \p op into a random access map.
static ArgumentDictionaryTy
loadArgumentMap(const ONNX_NAMESPACE::NodeProto &op) {
  ArgumentDictionaryTy dict;
  for (auto i = 0, e = op.attribute_size(); i < e; i++) {
    const ONNX_NAMESPACE::AttributeProto &arg = op.attribute(i);
    dict[arg.name()] = &arg;
  }
  return dict;
}

llvm::Expected<bool>
ONNXModelLoader::getBroadcast(const ArgumentDictionaryTy &dict) {
  if (opsetVersion_ > 6) {
    return true;
  }
  if (!dict.count("broadcast")) {
    return false;
  }

  int broadcast;
  ASSIGN_VALUE_OR_RETURN_ERR(broadcast, loadInt(dict.at("broadcast")));
  return broadcast == 1;
}

llvm::Error ONNXModelLoader::setVersion(ONNX_NAMESPACE::ModelProto MP) {
  irVersion_ = MP.ir_version();
  opsetVersion_ = 0;
  RETURN_ERR_IF_NOT(
      irVersion_ >= 3,
      "This ONNX model with ir_version < 3 is too old to be supported.");
  for (const auto &imp : MP.opset_import()) {
    if (!imp.has_domain() || imp.domain() == "") {
      opsetVersion_ = imp.version();
      break;
    }
  }
  RETURN_ERR_IF_NOT(opsetVersion_ > 0,
                    "The opset of this ONNX model is not supported.");
  RETURN_SUCCESS();
}

llvm::Expected<ONNX_NAMESPACE::ModelProto>
ONNXModelLoader::loadProto(google::protobuf::io::ZeroCopyInputStream &iStream) {
  // Construct and configure a Coded Input Stream
  google::protobuf::io::CodedInputStream codedStream(&iStream);

  // Don't warn about large file sizes.
  codedStream.SetTotalBytesLimit(MAX_PROTO_SIZE, MAX_PROTO_SIZE);
  ONNX_NAMESPACE::ModelProto MP;
  bool parseNet = MP.ParseFromCodedStream(&codedStream);
  RETURN_ERR_IF_NOT(parseNet, "Failed to parse ModelProto");
  return MP;
}

llvm::Expected<ONNX_NAMESPACE::ModelProto>
ONNXModelLoader::loadProto(const void *onnxModel, size_t onnxModelSize) {
  google::protobuf::io::ArrayInputStream arrayStream(onnxModel, onnxModelSize);
  return loadProto(arrayStream);
}

llvm::Expected<ONNX_NAMESPACE::ModelProto>
ONNXModelLoader::loadProto(const std::string &filename) {
  std::ifstream ff(filename, std::ios::in | std::ios::binary);
  RETURN_ERR_IF_NOT(ff, "Can't find the model or network files.");

  // TODO: intend to find a way to reuse the following function later
  // for the text format onnx model:
  // bool ONNXModelLoader::loadProto(ONNX_NAMESPACE::GraphProto &net,
  //  google::protobuf::io::ZeroCopyInputStream &iStream)
  if (filename.find(".onnxtxt") != std::string::npos) {
    std::string str((std::istreambuf_iterator<char>(ff)),
                    std::istreambuf_iterator<char>());
    ONNX_NAMESPACE::ModelProto MP;
    bool parseNet = google::protobuf::TextFormat::ParseFromString(str, &MP);

    RETURN_ERR_IF_NOT(parseNet, "Failed to parse ModelProto");
    return MP;
  }

  google::protobuf::io::IstreamInputStream fileStream(&ff);
  return loadProto(fileStream);
}

namespace {
/// Helper type for pads.
using Pads = std::vector<unsigned_t>;
} // namespace

llvm::Expected<Pads> getPads(const ArgumentDictionaryTy &dict) {
  if (dict.count("pads")) {
    return getShape<unsigned_t>(dict.at("pads"));
  }
  if (dict.count("auto_pad")) {
    std::string padStr;
    ASSIGN_VALUE_OR_RETURN_ERR(padStr, loadStr(dict.at("auto_pad")));
    if (padStr == "VALID") {
      // Return default value 0 for pads.
      return Pads({0, 0, 0, 0});
    }
    RETURN_ERR("only auto_pad==VALID is supported");
  }
  // Return default value 0 for pads.
  return Pads({0, 0, 0, 0});
}

/// Loads tensor \p T from the input \p in.
static llvm::Error loadTensor(const ONNX_NAMESPACE::TensorProto &in,
                              Tensor *T) {
  std::vector<size_t> dim;
  for (auto d : in.dims()) {
    dim.push_back(d);
  }

  if (in.data_type() == ONNX_NAMESPACE::TensorProto::FLOAT) {
    T->reset(ElemKind::FloatTy, dim);

    if (in.float_data_size() > 0) {
      auto TH = T->getHandle<>();
      size_t i = 0;
      for (auto f : in.float_data()) {
        TH.raw(i++) = f;
      }
    } else if (in.has_raw_data()) {
      std::istringstream inStream(in.raw_data(), std::stringstream::binary);
      inStream.read(T->getUnsafePtr(), T->size() * sizeof(float));
    } else {
      RETURN_ERR("Unsupported Tensor format.");
    }
  } else if (in.data_type() == ONNX_NAMESPACE::TensorProto::INT64) {
    T->reset(ElemKind::Int64ITy, dim);

    if (in.int64_data_size() > 0) {
      auto TH = T->getHandle<int64_t>();
      size_t i = 0;
      for (auto f : in.int64_data()) {
        TH.raw(i++) = f;
      }
    } else if (in.has_raw_data()) {
      std::istringstream inStream(in.raw_data(), std::stringstream::binary);
      inStream.read(T->getUnsafePtr(), T->size() * sizeof(int64_t));
    } else {
      RETURN_ERR("Unsupported Tensor format.");
    }
  } else if (in.data_type() == ONNX_NAMESPACE::TensorProto::INT32) {
    // There are few cases when we will have int32 tensors. For example, the
    // second output of Concat from Caffe2 concat op is int32
    T->reset(ElemKind::Int32ITy, dim);

    if (in.int32_data_size() > 0) {
      auto TH = T->getHandle<int32_t>();
      size_t i = 0;
      for (auto f : in.int32_data()) {
        TH.raw(i++) = f;
      }
    } else if (in.has_raw_data()) {
      std::istringstream inStream(in.raw_data(), std::stringstream::binary);
      inStream.read(T->getUnsafePtr(), T->size() * sizeof(int32_t));
    } else {
      RETURN_ERR("Unsupported Tensor format.");
    }
  } else {
    RETURN_ERR("Only float and index tensors are supported");
  }
  RETURN_SUCCESS();
}

llvm::Error ONNXModelLoader::loadOperator(const ONNX_NAMESPACE::NodeProto &op) {
  ArgumentDictionaryTy dict = loadArgumentMap(op);
  const std::string &typeName = op.op_type();

  // Check if operator is supported in parent class, CommonOperatorLoader.
  bool tryLoadCommonOperatorResult;
  ASSIGN_VALUE_OR_RETURN_ERR(tryLoadCommonOperatorResult,
                             tryLoadCommonOperator(typeName, op, dict));
  if (tryLoadCommonOperatorResult) {
    RETURN_SUCCESS();
  }

  const std::string &opName = loadOperatorName(op);

  // Load tensors with values:
  if (typeName == "Constant") {
    /*
      output: "Parameter6"
      name: "Parameter6"
      op_type: "Constant"
      attribute {
        name: "value"
        t {
          dims: 8
          data_type: FLOAT
          float_data: -0.161539719
          float_data: -0.433835655
          float_data: 0.091641359
          float_data: -0.0168522168
          float_data: -0.0650264397
          float_data: -0.131737873
          float_data: 0.0204175506
          float_data: -0.121110231
        }
        type: TENSOR
      }
      doc_string: ""
      domain: ""
    */

    const auto &name = op.output(0);
    // If the tensor is pre-populated by the user of this class then we don't
    // need to allocate a new tensor.
    if (tensors_.count(name)) {
      RETURN_SUCCESS();
    }

    RETURN_ERR_IF_NOT(dict["value"]->type() ==
                          ONNX_NAMESPACE::AttributeProto::TENSOR,
                      "Only Tensor type constants are supported.");

    auto *T = new Tensor();
    RETURN_IF_ERR(loadTensor(dict["value"]->t(), T));
    tensors_[name] = T;
    RETURN_SUCCESS();
  }

  if (typeName == "Conv") {
    // Load the attributes
    std::vector<unsigned_t> strides(2, 1);
    if (dict.count("strides")) {
      strides = getShape<unsigned_t>(dict.at("strides"));
    }
    unsigned_t group = 1;
    if (dict.count("group")) {
      ASSIGN_VALUE_OR_RETURN_ERR(group, loadInt(dict["group"]));
    }

    // Pads : {pad_top, pad_left, pad_bottom, pad_right}
    Pads pads;
    ASSIGN_VALUE_OR_RETURN_ERR(pads, getPads(dict));

    // Load the inputs
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    NodeValue filterValue;
    ASSIGN_VALUE_OR_RETURN_ERR(filterValue,
                               getNodeValueOrCreateConstantByName(op.input(1)));

    // Transpose the filter to the right format. Glow expects to read the
    // weights in the format CRSK. ONNX stores the operators as KCRS.
    // C - output_depth, R - filter_height, S - filter_width, K - input_depth.
    TransposeNode *filterTransposeNode =
        G_.createTranspose(opName, filterValue, NCHW2NHWC);

    // The structure of the conv weigts is: NHWC. We take the C, which is the
    // number of filters. We use this value to calculate the size of the bias
    // if it is not specified.
    const NodeValue filterTransposedValue = filterTransposeNode->getResult();
    size_t depth = filterTransposedValue.dims()[0];

    // Get the kernel shape from the input.
    std::vector<unsigned_t> kernelShape(2);
    kernelShape[0] = filterTransposedValue.dims()[1];
    kernelShape[1] = filterTransposedValue.dims()[2];

    // Extra check when the 'kernel_shape' attribute exists.
    // The 'kernel_shape' attribute is redundant not mandatory.
    if (dict.count("kernel_shape")) {
      std::vector<unsigned_t> kernelShapeAttribute =
          getShape<unsigned_t>(dict.at("kernel_shape"));
      RETURN_ERR_IF_NOT(
          (kernelShape[0] == kernelShapeAttribute[0] &&
           kernelShape[1] == kernelShapeAttribute[1]),
          "The 'kernel_shape' attribute is not consistent with the actual "
          "convolution kernel shape.");
      (void)kernelShapeAttribute; // Avoids compilation warning in release mode.
    }

    // Construct the Bias field.
    Tensor biasTensor(ElemKind::FloatTy, {depth});
    biasTensor.zero();

    // Check if we have a serialized bias vector.
    if (op.input_size() > 2) {
      auto &biasTensorName = op.input(2);
      if (tensors_.count(biasTensorName)) {
        // Load the serialized bias vector.
        Tensor *b;
        ASSIGN_VALUE_OR_RETURN_ERR(b, getTensorByName(biasTensorName));
        biasTensor.assign(b);
      }
    }
    auto *bias = G_.getParent()->createConstant("conv.bias", biasTensor);

    // ONNX passes the input as NCHW, and we expect the input to be NHWC.
    auto *tr = G_.createTranspose(opName, in, NCHW2NHWC);

    // Calculate the size and allocate the output buffer.
    ShapeNHWC idim = ShapeNHWC(tr->getResult().dims());
    auto outSz =
        calculateConvPoolOutputDims(idim.h, idim.w, kernelShape, strides, pads);
    std::array<size_t, 4> outDims = {
        {idim.n, outSz.first, outSz.second, depth}};
    auto outTy = G_.getParent()->uniqueType(ElemKind::FloatTy, outDims);

    auto *node = G_.createConv(opName, tr, filterTransposeNode, bias, outTy,
                               kernelShape, strides, pads, group);

    // Transpose the output back.
    auto *N = G_.createTranspose(opName, node, NHWC2NCHW);
    addNodeAsOutput(op, N);

    RETURN_SUCCESS();
  }

  if (typeName == "MaxPool" || typeName == "AveragePool") {
    // Glow doesn't support argmax output yet.
    if (op.output_size() > 1) {
      RETURN_ERR("Glow doesn't support argmax output yet.");
    }
    // Load the inputs:
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    std::vector<unsigned_t> strides(2, 1);
    if (dict.count("strides")) {
      strides = getShape<unsigned_t>(dict.at("strides"));
    }
    std::vector<unsigned_t> kernels =
        getShape<unsigned_t>(dict.at("kernel_shape"));

    Pads pads;
    ASSIGN_VALUE_OR_RETURN_ERR(pads, getPads(dict));

    if (in.dims().size() != 4 || kernels.size() != 2) {
      // Glow only handles 2D pooling currently.
      RETURN_ERR("Glow only handles 2D pooling currently.");
    }

    auto *tr = G_.createTranspose(opName, in, NCHW2NHWC);

    // If 'global_pooling' is set then the operation will pool over the size of
    // the input by doing: kernel = height/width.
    if (dict.count("global_pooling")) {
      auto Ty = in.getType();
      kernels[0] = Ty->dims()[2];
      kernels[1] = Ty->dims()[3];
    }

    Node *node = nullptr;
    if (typeName == "MaxPool") {
      node = G_.createMaxPool(opName, tr, kernels, strides, pads);
    } else {
      node = G_.createAvgPool(opName, tr, kernels, strides, pads);
    }
    auto *N = G_.createTranspose(opName, node, NHWC2NCHW);
    addNodeAsOutput(op, N);
    RETURN_SUCCESS();
  }

  if (typeName == "GlobalAveragePool") {
    // Load the inputs:
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    std::vector<unsigned_t> strides(2, 1);
    if (dict.count("strides")) {
      strides = getShape<unsigned_t>(dict.at("strides"));
    }

    std::vector<unsigned_t> kernels(2);
    kernels[0] = in.dims()[2];
    kernels[1] = in.dims()[3];

    Pads pads;
    ASSIGN_VALUE_OR_RETURN_ERR(pads, getPads(dict));

    auto *tr = G_.createTranspose(opName, in, NCHW2NHWC);
    Node *node = G_.createAvgPool(opName, tr, kernels, strides, pads);
    auto *N = G_.createTranspose(opName, node, NHWC2NCHW);
    addNodeAsOutput(op, N);
    RETURN_SUCCESS();
  }

  if (typeName == "Squeeze") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    auto axes = getShape(dict["axes"]);
    Node *node = G_.createSqueeze(opName, in, axes);
    addNodeAsOutput(op, node);
    RETURN_SUCCESS();
  }

  if (typeName == "Unsqueeze") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    auto axes = getShape(dict["axes"]);
    Node *node = G_.createExpandDims(opName, in, axes);
    addNodeAsOutput(op, node);
    RETURN_SUCCESS();
  }

  if (typeName == "BatchNormalization") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    Tensor *scale;
    ASSIGN_VALUE_OR_RETURN_ERR(scale, getTensorByName(op.input(1)));
    Tensor *bias;
    ASSIGN_VALUE_OR_RETURN_ERR(bias, getTensorByName(op.input(2)));
    Tensor *mean;
    ASSIGN_VALUE_OR_RETURN_ERR(mean, getTensorByName(op.input(3)));
    Tensor *var;
    ASSIGN_VALUE_OR_RETURN_ERR(var, getTensorByName(op.input(4)));
    float epsilon = 1e-5f; // default
    auto epsilonIt = dict.find("epsilon");
    if (epsilonIt != dict.end()) {
      ASSIGN_VALUE_OR_RETURN_ERR(epsilon, loadFloat(epsilonIt->second));
    }

    auto *scaleV = G_.getParent()->createConstant("scale", *scale);
    auto *biasV = G_.getParent()->createConstant("bias", *bias);
    auto *meanV = G_.getParent()->createConstant("mean", *mean);
    auto *varV = G_.getParent()->createConstant("var", *var);
    auto *node = G_.createBatchNormalization(opName, in, biasV, scaleV, meanV,
                                             varV, 1, epsilon);

    addNodeAsOutput(op, node);
    RETURN_SUCCESS();
  }

  if (typeName == "Concat") {
    const unsigned numInputs = op.input_size();
    llvm::SmallVector<NodeValue, 4> inputs;
    inputs.reserve(numInputs);
    for (unsigned i = 0; i < numInputs; i++) {
      NodeValue in;
      ASSIGN_VALUE_OR_RETURN_ERR(
          in, getNodeValueOrCreateConstantByName(op.input(i)));
      inputs.push_back(in);
    }

    int axis;
    ASSIGN_VALUE_OR_RETURN_ERR(axis, loadInt(dict["axis"]));
    Node *node = G_.createConcat(opName, inputs, axis);

    addNodeAsOutput(op, node);
    RETURN_SUCCESS();
  }

  if (typeName == "FCTransposed") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    if (in.getType()->dims().size() > 2) {
      size_t axis = 1;
      if (dict.count("axis")) {
        ASSIGN_VALUE_OR_RETURN_ERR(axis, loadInt(dict["axis"]));
      }
      in = G_.createFlatten("fc.in", in, axis);
    }

    Tensor *w;
    ASSIGN_VALUE_OR_RETURN_ERR(w, getTensorByName(op.input(1)));
    Tensor *b;
    ASSIGN_VALUE_OR_RETURN_ERR(b, getTensorByName(op.input(2)));
    unsigned_t axis_w = 1;
    if (dict.count("axis_w")) {
      ASSIGN_VALUE_OR_RETURN_ERR(axis_w, loadInt(dict["axis_w"]));
    }

    // w is stored already transposed. No need to additionally transpose it.
    Tensor tmp;
    if (w->dims().size() > 2) {
      auto wDims = flattenCdr(w->dims(), axis_w);
      tmp.reset(ElemKind::FloatTy, {wDims.first, wDims.second});
      tmp.copyRawFrom(w);
      w = &tmp;
    }

    auto W =
        G_.getParent()->addConstant(new Constant("weights", std::move(*w)));
    auto B = G_.getParent()->addConstant(new Constant("biases", std::move(*b)));
    auto *node = G_.createFullyConnected(opName, in, W, B);

    addNodeAsOutput(op, node);
    RETURN_SUCCESS();
  }

  if (typeName == "Gemm") {
    NodeValue A;
    ASSIGN_VALUE_OR_RETURN_ERR(A,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    NodeValue B;
    ASSIGN_VALUE_OR_RETURN_ERR(B,
                               getNodeValueOrCreateConstantByName(op.input(1)));
    NodeValue C;
    ASSIGN_VALUE_OR_RETURN_ERR(C,
                               getNodeValueOrCreateConstantByName(op.input(2)));

    bool broadcastC;
    ASSIGN_VALUE_OR_RETURN_ERR(broadcastC, getBroadcast(dict));
    bool transA = false;
    if (dict.count("transA")) {
      ASSIGN_VALUE_OR_RETURN_ERR(transA, loadInt(dict["transA"]));
    }
    bool transB = false;
    if (dict.count("transB")) {
      ASSIGN_VALUE_OR_RETURN_ERR(transB, loadInt(dict["transB"]));
    }
    // TODO: support alpha * A * B + beta * C

    if (transA)
      A = G_.createTranspose(opName, A, {1, 0});
    if (transB)
      B = G_.createTranspose(opName, B, {1, 0});

    MatMulNode *mul = G_.createMatMul(opName, A, B);
    if (broadcastC) {
      int axis = mul->getResult().dims().size() - C.dims().size();
      C = G_.createBroadcast(opName, C, mul->getResult().dims(), axis);
    }

    Node *node = G_.createAdd(opName, mul, C);
    addNodeAsOutput(op, node);
    RETURN_SUCCESS();
  }

  if (typeName == "Transpose") {
    RETURN_IF_ERR(loadTranspose(op, dict, "perm"));
    RETURN_SUCCESS();
  }

  if (typeName == "MatMul") {
    NodeValue LHS;
    ASSIGN_VALUE_OR_RETURN_ERR(LHS,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    NodeValue RHS;
    ASSIGN_VALUE_OR_RETURN_ERR(RHS,
                               getNodeValueOrCreateConstantByName(op.input(1)));

    Node *node = G_.createMatMul(opName, LHS, RHS);
    addNodeAsOutput(op, node);
    RETURN_SUCCESS();
  }

  RETURN_ERR("Failed to load operator.");
}

llvm::Error ONNXModelLoader::loadInitializers(ONNX_NAMESPACE::GraphProto &net) {
  // Load the network initializaers:
  for (const auto &in : net.initializer()) {
    Tensor *T = new Tensor();
    RETURN_IF_ERR(loadTensor(in, T));
    tensors_[in.name()] = T;
  }
  RETURN_SUCCESS();
}

llvm::Error ONNXModelLoader::setOutputNodes(ONNX_NAMESPACE::GraphProto &net) {
  if (net.output_size() == 0) {
    RETURN_ERR("Net output size must be greater than 0");
  }

  for (int i = 0; i < net.output_size(); i++) {
    const auto &outputName = net.output(i).name();
    NodeValue r;
    ASSIGN_VALUE_OR_RETURN_ERR(r,
                               getNodeValueOrCreateConstantByName(outputName));
    SaveNode *SN = G_.createSave("save_" + outputName, r);
    outputVarsByName_[outputName] = SN->getPlaceholder();
  }

  RETURN_SUCCESS();
}

llvm::Error ONNXModelLoader::loadNetwork(ONNX_NAMESPACE::GraphProto &net) {
  /// Load the network operators:
  for (int i = 0; i < net.node_size(); i++) {
    auto &op = net.node(i);
    RETURN_IF_ERR(loadOperator(op));
  }

  RETURN_SUCCESS();
}

ONNXModelLoader::ONNXModelLoader(Function &F) : ONNXModelLoader(nullptr, F) {}

ONNXModelLoader::ONNXModelLoader(llvm::Error *errPtr, Function &F)
    : CommonOperatorLoader(errPtr, {}, {}, F) {}

llvm::Error
ONNXModelLoader::checkInputs(ONNX_NAMESPACE::GraphProto &net,
                             llvm::ArrayRef<const char *> tensorNames,
                             llvm::ArrayRef<TypeRef> types) {
  for (size_t i = 0; i < tensorNames.size(); i++) {
    // Look if a corresponding input exists.
    for (int j = 0; j < net.input_size(); j++) {
      const ONNX_NAMESPACE::ValueInfoProto &valueInfo = net.input(j);
      const std::string &inputName = valueInfo.name();

      if (inputName != tensorNames[i]) {
        continue;
      }

      llvm::ArrayRef<size_t> dims = types[i]->dims();
      const ONNX_NAMESPACE::TensorShapeProto &shape =
          valueInfo.type().tensor_type().shape();
      (void)shape;

      // Check if the number of dimensions is consistent.
      RETURN_ERR_IF_NOT(dims.size() == (size_t)shape.dim_size(),
                        "Mismatch between input image and ONNX input shape");
      // Allow batch dimensions to be different.
      for (size_t k = 1; k < dims.size(); k++) {
        RETURN_ERR_IF_NOT(dims[k] == (size_t)shape.dim(k).dim_value(),
                          "Mismatch between input image and ONNX input shape");
      }
    }
  }
  RETURN_SUCCESS();
}

llvm::Error ONNXModelLoader::construct(const std::string &modelDescFilename,
                                       llvm::ArrayRef<const char *> tensorNames,
                                       llvm::ArrayRef<TypeRef> types) {
  // The ONNX model that we are deserializing.
  ONNX_NAMESPACE::ModelProto modelDef;
  ASSIGN_VALUE_OR_RETURN_ERR(modelDef, loadProto(modelDescFilename));

  RETURN_IF_ERR(setVersion(modelDef));

  ONNX_NAMESPACE::GraphProto graphDef = modelDef.graph();
  RETURN_IF_ERR(checkInputs(graphDef, tensorNames, types));

  RETURN_IF_ERR(loadInitializers(graphDef));
  RETURN_IF_ERR(loadNetwork(graphDef));

  RETURN_IF_ERR(setOutputNodes(graphDef));

  RETURN_SUCCESS();
}

ONNXModelLoader::ONNXModelLoader(const std::string &modelDescFilename,
                                 llvm::ArrayRef<const char *> tensorNames,
                                 llvm::ArrayRef<TypeRef> types, Function &F)
    : ONNXModelLoader(nullptr, modelDescFilename, tensorNames, types, F) {}

ONNXModelLoader::ONNXModelLoader(llvm::Error *errPtr,
                                 const std::string &modelDescFilename,
                                 llvm::ArrayRef<const char *> tensorNames,
                                 llvm::ArrayRef<TypeRef> types, Function &F)
    : CommonOperatorLoader(errPtr, tensorNames, types, F) {
  auto err = construct(modelDescFilename, tensorNames, types);
  if (errPtr) {
    *errPtr = std::move(err);
  } else {
    UNWRAP(std::move(err));
  }
}
