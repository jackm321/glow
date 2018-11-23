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

#include "glow/Importer/Caffe2ModelLoader.h"
#include "glow/Base/Tensor.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/Support/Error.h"

#include "llvm/Support/Casting.h"

#include "caffe2/proto/caffe2.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace glow;
using llvm::cast;

using ArgumentDictionaryTy =
    std::unordered_map<std::string, const caffe2::Argument *>;

/// Translates the protocol buffer node \p op into a random access map.
static ArgumentDictionaryTy loadArgumentMap(const caffe2::OperatorDef &op) {
  ArgumentDictionaryTy dict;
  for (auto i = 0, e = op.arg_size(); i < e; i++) {
    const caffe2::Argument &arg = op.arg(i);
    dict[arg.name()] = &arg;
  }
  return dict;
}

static llvm::Expected<std::vector<unsigned_t>>
getPads(const ArgumentDictionaryTy &dict) {
  if (dict.count("pad")) {
    int pad;
    ASSIGN_VALUE_OR_RETURN_ERR(pad, loadInt(dict.at("pad")));
    std::vector<unsigned_t> pads(4, pad);
    return pads;
  }
  if (dict.count("pad_t")) {
    std::vector<unsigned_t> pads(4);
    ASSIGN_VALUE_OR_RETURN_ERR(pads[0], loadInt(dict.at("pad_t")));
    RETURN_ERR_IF_NOT(dict.count("pad_l"), "missing pad_l");
    ASSIGN_VALUE_OR_RETURN_ERR(pads[1], loadInt(dict.at("pad_l")));
    RETURN_ERR_IF_NOT(dict.count("pad_b"), "missing pad_b");
    ASSIGN_VALUE_OR_RETURN_ERR(pads[2], loadInt(dict.at("pad_b")));
    RETURN_ERR_IF_NOT(dict.count("pad_r"), "missing pad_r");
    ASSIGN_VALUE_OR_RETURN_ERR(pads[3], loadInt(dict.at("pad_r")));
    return pads;
  }
  if (dict.count("pads")) {
    return getShape<unsigned_t>(dict.at("pads"));
  }
  // Return default value 0 for pads.
  return std::vector<unsigned_t>{0, 0, 0, 0};
}

/// Translates the "order" field of dictionary \p dict into a channel number.
static llvm::Expected<unsigned_t> getChannel(const ArgumentDictionaryTy &dict) {
  std::string order = "NCHW"; // default
  auto orderIt = dict.find("order");
  if (orderIt != dict.end()) {
    ASSIGN_VALUE_OR_RETURN_ERR(order, loadStr(orderIt->second));
  }
  if (order == "NHWC") {
    return 3;
  } else if (order == "NCHW") {
    return 1;
  }
  RETURN_ERR("Invalid order field");
}

static llvm::Expected<std::vector<unsigned_t>>
getSizeHW(ArgumentDictionaryTy &dict, const std::string &name,
          unsigned_t defaultValue) {
  if (dict.count(name)) {
    int value;
    ASSIGN_VALUE_OR_RETURN_ERR(value, loadInt(dict[name]));
    std::vector<unsigned_t> result(2, value);
    return result;
  }
  if (dict.count(name + "_h") && dict.count(name + "_w")) {
    std::vector<unsigned_t> result(2);
    ASSIGN_VALUE_OR_RETURN_ERR(result[0], loadInt(dict[name + "_h"]));
    ASSIGN_VALUE_OR_RETURN_ERR(result[1], loadInt(dict[name + "_w"]));
    return result;
  }
  if (dict.count(name + "s")) {
    return getShape<unsigned_t>(dict.at(name + "s"));
  }
  return std::vector<unsigned_t>{defaultValue, defaultValue};
}

llvm::Expected<caffe2::NetDef>
Caffe2ModelLoader::loadProtoFile(const std::string &filename) {
  std::ifstream ff(filename, std::ios::in | std::ios::binary);
  RETURN_ERR_IF_NOT(ff, "Can't find the model or network files.");

  caffe2::NetDef net;

  bool parseNet = false;
  if (filename.find(".pbtxt") != std::string::npos) {
    std::string str((std::istreambuf_iterator<char>(ff)),
                    std::istreambuf_iterator<char>());
    parseNet = google::protobuf::TextFormat::ParseFromString(str, &net);
  } else {
    // Construct and configure a Coded Input Stream
    google::protobuf::io::IstreamInputStream filestr(&ff);
    google::protobuf::io::CodedInputStream codedstr(&filestr);
    // Don't warn about large file sizes.
    codedstr.SetTotalBytesLimit(MAX_PROTO_SIZE, MAX_PROTO_SIZE);
    parseNet = net.ParseFromCodedStream(&codedstr);
  }

  RETURN_ERR_IF_NOT(parseNet, "Failed to parse the network descriptor.");
  return net;
}

llvm::Expected<bool>
Caffe2ModelLoader::getBroadcast(const ArgumentDictionaryTy &dict) {
  if (!dict.count("broadcast")) {
    return false;
  }
  int broadcast;
  ASSIGN_VALUE_OR_RETURN_ERR(broadcast, loadInt(dict.at("broadcast")));
  return broadcast == 1;
}

llvm::Error Caffe2ModelLoader::loadOperator(const caffe2::OperatorDef &op) {
  ArgumentDictionaryTy dict = loadArgumentMap(op);
  const std::string &typeName = op.type();

  // Check if operator is supported in parent class, CommonOperatorLoader.
  bool loadCommonOperatorSuccess;
  ASSIGN_VALUE_OR_RETURN_ERR(loadCommonOperatorSuccess,
                             tryLoadCommonOperator(typeName, op, dict));
  if (loadCommonOperatorSuccess) {
    RETURN_SUCCESS();
  }
  const std::string &opName = loadOperatorName(op);

  if (typeName == "Conv") {
    // Load the inputs:
    std::vector<unsigned_t> strides;
    ASSIGN_VALUE_OR_RETURN_ERR(strides, getSizeHW(dict, "stride", 1));
    std::vector<unsigned_t> pads;
    ASSIGN_VALUE_OR_RETURN_ERR(pads, getPads(dict));
    std::vector<unsigned_t> kernels;
    ASSIGN_VALUE_OR_RETURN_ERR(kernels, getSizeHW(dict, "kernel", 0));
    unsigned_t group = 1;
    if (dict.count("group")) {
      ASSIGN_VALUE_OR_RETURN_ERR(group, loadInt(dict["group"]));
    }
    std::string order = "NCHW";
    if (dict.count("order")) {
      ASSIGN_VALUE_OR_RETURN_ERR(order, loadStr(dict["order"]));
    }

    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    Tensor *w;
    ASSIGN_VALUE_OR_RETURN_ERR(w, getTensorByName(op.input(1)));

    // Transpose the weights to the right format. Glow expects to read the
    // weights in the format CRSK. Caffe2 stores the operators as KCRS.
    // C - output_depth, R - filter_height, S - filter_width, K - input_depth.
    Tensor wtag;
    w->transpose(&wtag, NCHW2NHWC);

    // The structure of the conv weigts is: NHWC. We take the C, which is the
    // number of filters. We use this value to calculate the size of the bias
    // if it is not specified.
    size_t depth = wtag.dims()[0];

    // Construct the Filter field.
    auto *filter = G_.getParent()->createConstant("conv.filter", wtag);

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

    // We expect the input to be NHWC.
    Node *tr;
    if (order == "NCHW") {
      tr = G_.createTranspose(opName, in, NCHW2NHWC);
    } else {
      tr = in;
    }

    // Calculate the size and allocate the output buffer.
    ShapeNHWC idim = ShapeNHWC(tr->getType(0)->dims());
    auto outSz =
        calculateConvPoolOutputDims(idim.h, idim.w, kernels, strides, pads);
    std::array<size_t, 4> outDims = {
        {idim.n, outSz.first, outSz.second, depth}};
    auto outTy = G_.getParent()->uniqueType(ElemKind::FloatTy, outDims);

    Node *node = G_.createConv(opName, tr, filter, bias, outTy, kernels,
                               strides, pads, group);

    if (order == "NCHW") {
      // Transpose the output back.
      node = G_.createTranspose(opName, node, NHWC2NCHW);
    }
    addNodeAsOutput(op, node);
    RETURN_SUCCESS();
  }

  if (typeName == "MaxPool" || typeName == "AveragePool") {
    // Load the inputs:
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    std::vector<unsigned_t> strides;
    ASSIGN_VALUE_OR_RETURN_ERR(strides, getSizeHW(dict, "stride", 1));
    std::vector<unsigned_t> kernels;
    ASSIGN_VALUE_OR_RETURN_ERR(kernels, getSizeHW(dict, "kernel", 0));
    std::vector<unsigned_t> pads;
    ASSIGN_VALUE_OR_RETURN_ERR(pads, getPads(dict));
    std::string order = "NCHW";
    if (dict.count("order")) {
      ASSIGN_VALUE_OR_RETURN_ERR(order, loadStr(dict["order"]));
    }
    // We expect the input to be NHWC.
    Node *tr;
    if (order == "NCHW") {
      tr = G_.createTranspose(opName, in, NCHW2NHWC);
    } else {
      tr = in;
    }

    // If 'global_pooling' is set then the operation will pool over the size of
    // the input by doing: kernels = {height, width}.
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
    if (order == "NCHW") {
      // Transpose the output back.
      node = G_.createTranspose(opName, node, NHWC2NCHW);
    }
    addNodeAsOutput(op, node);
    RETURN_SUCCESS();
  }

  if (typeName == "SpatialBN") {
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

    unsigned_t channel;
    ASSIGN_VALUE_OR_RETURN_ERR(channel, getChannel(dict));
    auto *scaleV = G_.getParent()->createConstant("scale", *scale);
    auto *biasV = G_.getParent()->createConstant("bias", *bias);
    auto *meanV = G_.getParent()->createConstant("mean", *mean);
    auto *varV = G_.getParent()->createConstant("var", *var);
    auto *node = G_.createBatchNormalization(opName, in, biasV, scaleV, meanV,
                                             varV, channel, epsilon);

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

    // If axis exists it takes priority over channel.
    unsigned_t channel;
    if (dict.count("axis")) {
      ASSIGN_VALUE_OR_RETURN_ERR(channel, loadInt(dict["axis"]));
    } else {
      ASSIGN_VALUE_OR_RETURN_ERR(channel, getChannel(dict));
    }

    Node *node = G_.createConcat(opName, inputs, channel);

    unsigned_t addAxis = 0;
    if (dict.count("add_axis")) {
      ASSIGN_VALUE_OR_RETURN_ERR(addAxis, loadInt(dict["add_axis"]));
    }

    if (addAxis) {
      // When add axis is used, this means we have to add a new dimension before
      // the axis, instead of merging on the axis.
      std::vector<size_t> outputDims = inputs[0].dims();
      for (const auto &input : inputs) {
        RETURN_ERR_IF_NOT(
            outputDims[channel] == input.dims()[channel],
            "inputs need all to have the same dims for concat with add_axis");
      }
      outputDims.insert(outputDims.begin() + channel, numInputs);
      node = G_.createReshape(opName, node, outputDims);
    }
    // Concat has multiple outputs in Caffe2, but I believe the other output
    // (split_info) is not used for inference.
    nodeValueByName_[op.output(0)] = NodeValue(node, 0);
    RETURN_SUCCESS();
  }

  if (typeName == "FC" || typeName == "FCTransposed") {
    // Load the inputs:
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

    // Load weights.
    Tensor *w;
    ASSIGN_VALUE_OR_RETURN_ERR(w, getTensorByName(op.input(1)));
    Tensor *b;
    ASSIGN_VALUE_OR_RETURN_ERR(b, getTensorByName(op.input(2)));
    unsigned_t axis_w = 1;
    if (dict.count("axis_w")) {
      ASSIGN_VALUE_OR_RETURN_ERR(axis_w, loadInt(dict["axis_w"]));
    }

    // Caffe2 stores the transposed W matrix. In here we first coerce W to a 2D
    // matrix size if necessay and then transpose it back.
    Tensor tmp;
    if (w->dims().size() > 2) {
      auto wDims = flattenCdr(w->dims(), axis_w);
      tmp.reset(ElemKind::FloatTy, {wDims.first, wDims.second});
      tmp.copyRawFrom(w);
      w = &tmp;
    }
    Tensor wtag;
    if (typeName == "FC") {
      w->transpose(&wtag, {1, 0});
    } else {
      wtag.assign(w);
    }

    auto W =
        G_.getParent()->addConstant(new Constant("weights", std::move(wtag)));
    auto B = G_.getParent()->addConstant(new Constant("biases", std::move(*b)));
    auto *node = G_.createFullyConnected(opName, in, W, B);

    // Save the outputs:
    addNodeAsOutput(op, node);
    RETURN_SUCCESS();
  }

  if (typeName == "ChannelShuffle") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));

    size_t group;
    ASSIGN_VALUE_OR_RETURN_ERR(group, loadInt(dict["group"]));
    size_t kernel;
    ASSIGN_VALUE_OR_RETURN_ERR(kernel, loadInt(dict["kernel"]));

    Node *node = G_.createChannelShuffle(opName, in, group, kernel);
    addNodeAsOutput(op, node);
    RETURN_SUCCESS();
  }

  if (typeName == "Squeeze") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    auto dims = getShape(dict["dims"]);
    Node *node = G_.createSqueeze(opName, in, dims);
    addNodeAsOutput(op, node);
    RETURN_SUCCESS();
  }

  if (typeName == "Log") {
    // Load the inputs:
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    // Create the log:
    auto *R = G_.createLog(opName, in);
    addNodeAsOutput(op, R);
    RETURN_SUCCESS();
  }

  if (typeName == "Logit") {
    // Load the input and (optional) epsilon clamping value:
    NodeValue input;
    ASSIGN_VALUE_OR_RETURN_ERR(input,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    auto epsIt = dict.find("eps");
    // default: 1e-6 (as in Caffe2)
    float eps = 1E-6f;
    if (epsIt != dict.end()) {
      ASSIGN_VALUE_OR_RETURN_ERR(eps, loadFloat(epsIt->second));
    }

    auto *node = G_.createLogit(opName, input, eps);
    // Save the outputs:
    addNodeAsOutput(op, node);
    RETURN_SUCCESS();
  }

  if (typeName == "EQ") {
    NodeValue in0;
    ASSIGN_VALUE_OR_RETURN_ERR(in0,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    NodeValue in1;
    ASSIGN_VALUE_OR_RETURN_ERR(in1,
                               getNodeValueOrCreateConstantByName(op.input(1)));
    auto *node = G_.createCmpEQ(opName, in0, in1);
    addNodeAsOutput(op, node);
    RETURN_SUCCESS();
  }

  if (typeName == "Tile") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    unsigned_t tiles;
    ASSIGN_VALUE_OR_RETURN_ERR(tiles, loadInt(dict["tiles"]));
    unsigned_t axis;
    ASSIGN_VALUE_OR_RETURN_ERR(axis, loadInt(dict["axis"]));

    auto *node = G_.createTile(opName, in, tiles, axis);
    addNodeAsOutput(op, node);
    RETURN_SUCCESS();
  }

  if (typeName == "Free") {
    // Glow frees memory automatically.
    RETURN_SUCCESS();
  }
  if (typeName == "StopGradient") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    // Currently Caffe2 importer only supports inference.
    addNodeAsOutput(op, in);
    RETURN_SUCCESS();
  }

  if (typeName == "Transpose") {
    RETURN_IF_ERR(loadTranspose(op, dict, "axes"));
    RETURN_SUCCESS();
  }

  if (typeName == "NCHW2NHWC") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    auto *node = G_.createTranspose(opName, in, NCHW2NHWC);
    addNodeAsOutput(op, node);
    RETURN_SUCCESS();
  }

  if (typeName == "CopyCPUToMKL" || typeName == "CopyMKLToCPU" ||
      typeName == "Copy" || typeName == "EnsureCPUOutput" ||
      typeName == "EnsureDense") {
    // Glow does not support any of these ops now, so implement them as no-ops.
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    addNodeAsOutput(op, in);
    RETURN_SUCCESS();
  }

  if (typeName == "Slice") {
    NodeValue data;
    ASSIGN_VALUE_OR_RETURN_ERR(data,
                               getNodeValueOrCreateConstantByName(op.input(0)));

    auto starts = getShape<ssize_t>(dict["starts"]);
    auto ends = getShape<ssize_t>(dict["ends"]);

    std::vector<size_t> newStarts, newEnds;
    RETURN_ERR_IF_NOT(starts.size() == ends.size(),
                      "Slice starts and ends must be the same size.");
    for (size_t i = 0; i < starts.size(); i++) {
      ssize_t newStart = starts[i];
      if (newStart == -1) {
        newStart = data.dims()[i];
      }
      RETURN_ERR_IF_NOT(newStart >= 0, "Indices should never be negative.");
      newStarts.push_back(newStart);

      ssize_t newEnd = ends[i];
      if (newEnd == -1) {
        newEnd = data.dims()[i];
      }
      RETURN_ERR_IF_NOT(newEnd >= 0, "Indices should never be negative.");
      newEnds.push_back(newEnd);
    }

    Node *SN = G_.createSlice(opName, data, newStarts, newEnds);
    addNodeAsOutput(op, SN);
    RETURN_SUCCESS();
  }

  if (typeName == "MatMul") {
    RETURN_IF_ERR(loadBatchMatMul(op, dict, false));
    RETURN_SUCCESS();
  }

  if (typeName == "Cast") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    int to;
    ASSIGN_VALUE_OR_RETURN_ERR(to, loadInt(dict["to"]));

    switch (to) {
    case caffe2::TensorProto_DataType_FLOAT: {
      RETURN_ERR_IF_NOT(in.getElementType() == ElemKind::FloatTy,
                        "Can only cast float to float.");
      break;
    }
    case caffe2::TensorProto_DataType_INT32:
    case caffe2::TensorProto_DataType_INT64: {
      RETURN_ERR_IF_NOT(in.getElementType() == ElemKind::Int64ITy,
                        "Can only cast int to int.");
      break;
    }
    default:
      llvm_unreachable("Unsupported Cast type.");
    }

    addNodeAsOutput(op, in);
    RETURN_SUCCESS();
  }

  if (typeName == "ScatterAssign") {
    NodeValue data;
    ASSIGN_VALUE_OR_RETURN_ERR(data,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    NodeValue indices;
    ASSIGN_VALUE_OR_RETURN_ERR(indices,
                               getNodeValueOrCreateConstantByName(op.input(1)));
    NodeValue slices;
    ASSIGN_VALUE_OR_RETURN_ERR(slices,
                               getNodeValueOrCreateConstantByName(op.input(2)));

    Node *SAN = G_.createScatterAssign(opName, data, indices, slices);
    addNodeAsOutput(op, SAN);
    RETURN_SUCCESS();
  }

  if (typeName == "ConstantFill" || typeName == "GivenTensorIntFill" ||
      typeName == "GivenTensorInt64Fill") {
    RETURN_IF_ERR(loadWeight(op));
    RETURN_SUCCESS();
  }

  if (typeName == "SigmoidCrossEntropyWithLogits") {
    NodeValue logits;
    ASSIGN_VALUE_OR_RETURN_ERR(logits,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    NodeValue targets;
    ASSIGN_VALUE_OR_RETURN_ERR(targets,
                               getNodeValueOrCreateConstantByName(op.input(1)));
    Node *SCEL =
        G_.createSigmoidCrossEntropyWithLogits(opName, logits, targets);
    addNodeAsOutput(op, SCEL);
    RETURN_SUCCESS();
  }

  if (typeName == "AveragedLoss") {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    auto *node = G_.createBatchedReduceMean(opName, in, 0);
    addNodeAsOutput(op, node);
    RETURN_SUCCESS();
  }

  RETURN_ERR(unexpectedNodeErrorMessage(op, "Unsupported operator."));
}

llvm::Error Caffe2ModelLoader::loadNetwork(caffe2::NetDef &net) {
  /// Load the network operators:
  for (int i = 0; i < net.op_size(); i++) {
    auto &op = net.op(i);
    RETURN_IF_ERR(loadOperator(op));
  }

  RETURN_ERR_IF_NOT(net.external_output_size(),
                    "Network needs external outputs defined.");

  for (int i = 0; i < net.external_output_size(); i++) {
    auto &outputName = net.external_output(i);
    NodeValue r;
    ASSIGN_VALUE_OR_RETURN_ERR(r, getNodeValueByName(outputName));
    auto *SN = G_.createSave("save_" + outputName, r);
    outputVarsByName_[outputName] = SN->getPlaceholder();
  }
  RETURN_SUCCESS();
}

llvm::Error Caffe2ModelLoader::loadWeight(const caffe2::OperatorDef &op) {
  ArgumentDictionaryTy dict = loadArgumentMap(op);
  const std::string &typeName = op.type();

  /// Load tensors with values:
  if (typeName == "GivenTensorFill" || typeName == "GivenTensorIntFill" ||
      typeName == "GivenTensorInt64Fill") {
    /*
     output: "conv1_w"
     name: ""
     type: "GivenTensorFill"
     arg {
     name: "shape"
     ints: 96
     ints: 3
     ints: 11
     ints: 11
     }
     arg {
     name: "values"
     floats: -0.028315347
     */

    auto *T = new Tensor();
    for (auto &o : op.output()) {
      tensors_[o] = T;
    }

    auto dim = getShape(dict["shape"]);

    size_t i = 0;
    if (dict["values"]->floats_size()) {
      RETURN_ERR_IF_NOT(
          (typeName != "GivenTensorIntFill" &&
           typeName != "GivenTensorInt64Fill"),
          "typeName must be GivenTensorIntFill or GivenTensorInt64Fill.");
      T->reset(ElemKind::FloatTy, dim);
      auto TH = T->getHandle<>();
      for (auto num : dict["values"]->floats()) {
        TH.raw(i++) = num;
      }
    } else if (dict["values"]->ints_size()) {
      T->reset(ElemKind::Int64ITy, dim);
      auto TH = T->getHandle<int64_t>();
      for (auto num : dict["values"]->ints()) {
        TH.raw(i++) = num;
      }
    } else {
      RETURN_ERR(unexpectedNodeErrorMessage(
          op, "Unsupported data type for GivenTensorFill."));
    }

    RETURN_ERR_IF_NOT(i == T->size(),
                      "The number of serialized values does not "
                      "match the size of the tensor.");
    RETURN_SUCCESS();
  }

  // Load tensors with constant fill:
  if (typeName == "ConstantFill") {
    /*
     output: "data"
     name: ""
     type: "ConstantFill"
     arg {
     name: "shape"
     ints: 1
     }
     */

    const auto &name = op.output(0);
    // If the tensor is pre-populated by the user of this class then we don't
    // need to allocate a new tensor.
    if (tensors_.count(name)) {
      RETURN_SUCCESS();
    }

    auto *T = new Tensor();
    tensors_[name] = T;

    // The shape is set either the shape argument, or from another input
    // tensor. Shape takes priority over input.
    std::vector<size_t> dims;
    if (dict.count("shape")) {
      dims = getShape(dict["shape"]);
    } else {
      RETURN_ERR_IF_NOT(op.input_size() > 0,
                        "If no shape provided, must have input shape.");
      // It must be registered as a tensor because it must be statically set
      // already, as shapes must be statically known.
      Tensor *in;
      ASSIGN_VALUE_OR_RETURN_ERR(in, getTensorByName(op.input(0)));
      dims = in->dims();
    }

    int to = caffe2::TensorProto_DataType_FLOAT;
    if (dict.count("dtype")) {
      ASSIGN_VALUE_OR_RETURN_ERR(to, loadInt(dict["dtype"]));
    }

    switch (to) {
    case caffe2::TensorProto_DataType_FLOAT: {
      T->reset(ElemKind::FloatTy, dims);
      auto TH = T->getHandle<float>();
      float f = 0.0f;
      if ((dict.count("value") && dict["value"]->has_f())) {
        ASSIGN_VALUE_OR_RETURN_ERR(f, loadFloat(dict["value"]));
      }
      TH.clear(f);
      break;
    }
    case caffe2::TensorProto_DataType_INT32:
    case caffe2::TensorProto_DataType_INT64:
    case caffe2::TensorProto_DataType_BOOL: {
      T->reset(ElemKind::Int64ITy, dims);
      auto TH = T->getHandle<int64_t>();
      int i = 0;
      if ((dict.count("value") && dict["value"]->has_i())) {
        ASSIGN_VALUE_OR_RETURN_ERR(i, loadInt(dict["value"]));
      }
      TH.clear(i);
      break;
    }
    default:
      RETURN_ERR("Unsupported datatype for ConstantFill.");
    }

    RETURN_SUCCESS();
  }

  if (typeName == "UniformFill") {
    /*
     output: "fc/w"
     name: ""
     type: "UniformFill"
     arg {
       name: "max"
       f: 0.25
     }
     arg {
       name: "shape"
       ints: 1
       ints: 16
     }
     arg {
       name: "min"
       f: -0.25
     }
    */
    const auto &name = op.output(0);
    auto *T = new Tensor();
    tensors_[name] = T;
    auto dim = getShape(dict["shape"]);
    T->reset(ElemKind::FloatTy, dim);
    auto TH = T->getHandle<>();
    float tensorMin;
    ASSIGN_VALUE_OR_RETURN_ERR(tensorMin, loadFloat(dict["min"]));
    float tensorMax;
    ASSIGN_VALUE_OR_RETURN_ERR(tensorMax, loadFloat(dict["max"]));

#ifndef NDEBUG
    llvm::outs() << "The model contains UniformFill operator, which generates"
                 << " random numbers. This could be source of discrepancy.\n";
#endif // NDEBUG
    // Uniformly generate random numbers in [tensorMin; tensorMax).
    for (size_t i = 0, e = T->size(); i != e; i++) {
      TH.raw(i) = G_.getParent()->getPRNG().nextRandReal(tensorMin, tensorMax);
    }
    RETURN_SUCCESS();
  }

  RETURN_ERR(unexpectedNodeErrorMessage(op, "Unsupported weight kind"));
}

llvm::Error Caffe2ModelLoader::loadWeights(caffe2::NetDef &net) {
  for (auto &op : net.op()) {
    RETURN_IF_ERR(loadWeight(op));
  }
  RETURN_SUCCESS();
}

Caffe2ModelLoader::Caffe2ModelLoader(const std::string &netDescFilename,
                                     const std::string &netWeightFilename,
                                     llvm::ArrayRef<const char *> names,
                                     llvm::ArrayRef<TypeRef> types, Function &F)
    : CommonOperatorLoader(names, types, F) {
  // The caffe2 network descriptor that we are deserializing.
  caffe2::NetDef networkDef = UNWRAP(loadProtoFile(netDescFilename));

  // The caffe2 weights that we are deserializing.
  caffe2::NetDef weightsDef = UNWRAP(loadProtoFile(netWeightFilename));

  TEMP_UNWRAP(loadWeights(weightsDef));
  TEMP_UNWRAP(loadNetwork(networkDef));
}
