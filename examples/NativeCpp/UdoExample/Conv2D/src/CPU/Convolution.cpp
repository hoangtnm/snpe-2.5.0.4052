//==============================================================================
// Auto Generated Code for ConvolutionOpPackage
//==============================================================================
#include <iostream>
#include <string>
#include <string.h>
#include <cmath>
#include <algorithm>

#include "CpuBackendUtils.hpp"
#include "CustomOpPackage.hpp"

using namespace qnn::custom;
using namespace qnn::custom::utils;

namespace convolution {

Qnn_ErrorHandle_t execute(CustomOp* operation) {

    int32_t numFilters = 0;
    int32_t kernelSize = 0;
    int32_t biasTerm = 1;
    int32_t pad = 0;
    int32_t stride = 1;
    int32_t padH = 0;
    int32_t padW = 0;
    int32_t strideH = 1;
    int32_t strideW = 1;
    int32_t kernelH = 0;
    int32_t kernelW = 0;
    int32_t groups = 1;

    auto m_Inputs = operation->getInput(0);
    auto m_Outputs = operation->getOutput(0);

    const float* in  = (float*)m_Inputs->data;
    float* out  = (float*)m_Outputs->data;

    float* filter  = (float*)(operation->getInput(1))->data;
    float* bias  = (float*)(operation->getInput(2))->data;

    biasTerm = (int32_t)(operation->getParam("bias_term")->scalarParam);
    groups = (int32_t)(operation->getParam("group")->scalarParam);
    kernelH = (int32_t)(operation->getParam("kernel_h")->scalarParam);
    kernelW = (int32_t)(operation->getParam("kernel_w")->scalarParam);
    padH = (int32_t)(operation->getParam("pad_h")->scalarParam);
    padW = (int32_t)(operation->getParam("pad_w")->scalarParam);
    strideH = (int32_t)(operation->getParam("stride_h")->scalarParam);
    numFilters = (int32_t)(operation->getParam("num_output")->scalarParam);
    pad = *((int32_t*)(operation->getParam("pad")->tensorParam->data));
    kernelSize = *((int32_t*)(operation->getParam("kernel_size")->tensorParam->data));
    stride = *((int32_t*)(operation->getParam("stride")->tensorParam->data));

    if (kernelSize != 0)
    {
        kernelH = kernelSize;
        kernelW = kernelSize;
    }
    if (pad != 0)
    {
        padH = pad;
        padW = pad;
    }
    if (stride != 0)
    {
        strideH = stride;
        strideW = stride;
    }

    //Input height, width and depth.
    const size_t inputHeight = m_Inputs->currentDimensions[1];
    const size_t inputWidth = m_Inputs->currentDimensions[2];
    const size_t inputDepth = m_Inputs->currentDimensions[3];

    //Output height, width and depth
    const size_t outputHeight = m_Outputs->currentDimensions[1];
    const size_t outputWidth = m_Outputs->currentDimensions[2];
    const size_t outputDepth = m_Outputs->currentDimensions[3];

    //Filter height, width and depth
    size_t filterHeight  = (operation->getInput(1))->currentDimensions[2];
    size_t filterWidth = (operation->getInput(1))->currentDimensions[3];
    size_t filterDepth = (operation->getInput(1))->currentDimensions[1];

    // set the depth for each group of filters
    uint32_t outputGroupDepth = numFilters / groups;

    float outputActivationMin = std::numeric_limits<float>::lowest();
    float outputActivationMax = std::numeric_limits<float>::max();

    const float* filterbase = nullptr;

    for (int32_t oh = 0; oh < outputHeight; oh++)
    {
        for (int32_t ow = 0; ow < outputWidth; ow++)
        {
            filterbase = filter;
            int32_t inputOriginH = oh * strideH - padH;
            int32_t inputOriginW = ow * strideW - padW;
            for (int32_t g = 0; g < groups; g++)
            {
                for (int32_t d = 0; d < outputGroupDepth; d++)
                {
                    float sum = 0.0f;
                    for (int32_t fd = 0; fd < filterDepth; fd++)
                    {
                        for (int32_t fh = 0; fh < filterHeight; fh++)
                        {
                            int32_t inputH = inputOriginH + fh;
                            if (inputH < 0 || inputH >= static_cast<int32_t>(inputHeight))
                                continue;
                            for (int32_t fw = 0; fw < filterWidth; fw++)
                            {
                                int32_t inputW = inputOriginW + fw;
                                if (inputW < 0 || inputW >= static_cast<int32_t>(inputWidth))
                                    continue;
                                uint32_t inputD = filterDepth * g + fd;
                                uint32_t filterIndex = fd * filterHeight * filterWidth + fh * filterWidth + fw;
                                uint32_t inputIndex = inputDepth * (inputH * inputWidth + inputW) + inputD;
                                sum += filterbase[filterIndex] * in[inputIndex];
                            }
                        }
                    }
                    if (biasTerm)
                    {
                        sum += bias[g * outputGroupDepth + d];
                    }
                    sum = std::max(std::min(sum, outputActivationMax), outputActivationMin);
                    out[d] = sum;
                    filterbase += (filterHeight * filterWidth * filterDepth);
                }
                out += outputGroupDepth;
            }
        }
    }
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t finalize(const CustomOp* operation) {
  QNN_CUSTOM_BE_ENSURE_EQ(operation->numInput(), 3, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)
  QNN_CUSTOM_BE_ENSURE_EQ(operation->numOutput(), 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)

  /**
   * Add code here
   **/

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t free(CustomOp& operation) {

    /**
    * Add code here
    **/

    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t populateFromNode(const QnnOpPackage_Node_t node,
                                   QnnOpPackage_GraphInfrastructure_t graphInfrastructure,
                                   CustomOp* operation) {
  // Add input
  for (uint32_t i = 0; i < numInputs(node); i++) {
    operation->addInput(getInput(node, i));
  }

  for (uint32_t i = 0; i < numOutputs(node); i++) {
    operation->addOutput(getOutput(node, i));
  }

  // Add params
   // The getParam function returns a pair -> hasParam, paramValue
   // Check that parameter has be retrieved. Pair.first is false if it was not found and the paramValue is nullptr

   auto kernel_sizePair = getParam(node, "kernel_size");


    QNN_CUSTOM_BE_ENSURE(kernel_sizePair.first, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)

    operation->addParam("kernel_size", kernel_sizePair.second);


   auto stridePair = getParam(node, "stride");

   QNN_CUSTOM_BE_ENSURE(stridePair.first, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)
   operation->addParam("stride", stridePair.second);


   auto padPair = getParam(node, "pad");

   QNN_CUSTOM_BE_ENSURE(padPair.first, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)
   operation->addParam("pad", padPair.second);

   auto bias_termPair = getParam(node, "bias_term");

   QNN_CUSTOM_BE_ENSURE(bias_termPair.first, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)
   operation->addParam("bias_term", bias_termPair.second);

   auto groupPair = getParam(node, "group");

   QNN_CUSTOM_BE_ENSURE(groupPair.first, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)
   operation->addParam("group", groupPair.second);

   auto kernel_hPair = getParam(node, "kernel_h");

   QNN_CUSTOM_BE_ENSURE(kernel_hPair.first, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)
   operation->addParam("kernel_h", kernel_hPair.second);

   auto kernel_wPair = getParam(node, "kernel_w");

   QNN_CUSTOM_BE_ENSURE(kernel_wPair.first, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)
   operation->addParam("kernel_w", kernel_wPair.second);


   auto pad_hPair = getParam(node, "pad_h");

   QNN_CUSTOM_BE_ENSURE(pad_hPair.first, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)
   operation->addParam("pad_h", pad_hPair.second);


   auto pad_wPair = getParam(node, "pad_w");

   QNN_CUSTOM_BE_ENSURE(pad_wPair.first, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)
   operation->addParam("pad_w", pad_wPair.second);


   auto stride_hPair = getParam(node, "stride_h");

   QNN_CUSTOM_BE_ENSURE(stride_hPair.first, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)
   operation->addParam("stride_h", stride_hPair.second);


   auto stride_wPair = getParam(node, "stride_w");

   QNN_CUSTOM_BE_ENSURE(stride_wPair.first, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)
   operation->addParam("stride_w", stride_wPair.second);


   auto num_outputPair = getParam(node, "num_output");

   QNN_CUSTOM_BE_ENSURE(num_outputPair.first, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)
   operation->addParam("num_output", num_outputPair.second);

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t validateOpConfig(Qnn_OpConfig_t opConfig) {
  QNN_CUSTOM_BE_ENSURE_EQ(
      strcmp(opConfig.v1.typeName, "Convolution"), 0, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)

  QNN_CUSTOM_BE_ENSURE_EQ(opConfig.v1.numOfInputs, 3, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)
  QNN_CUSTOM_BE_ENSURE_EQ(opConfig.v1.numOfOutputs, 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)

  return QNN_SUCCESS;
}
}  // namespace convolution

CustomOpRegistration_t* register_ConvolutionCustomOp() {
  using namespace convolution;
  static CustomOpRegistration_t ConvolutionRegister = {execute, finalize, free, validateOpConfig, populateFromNode};
  return &ConvolutionRegister;
}

REGISTER_OP(Convolution, register_ConvolutionCustomOp);