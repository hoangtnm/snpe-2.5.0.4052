//==============================================================================
// Auto Generated Code for SoftmaxOpPackage
//==============================================================================
#include <iostream>
#include <string>
#include <cmath>
#include <algorithm>

#include "CpuBackendUtils.hpp"
#include "CustomOpPackage.hpp"

using namespace qnn::custom;
using namespace qnn::custom::utils;

namespace softmax {

Qnn_ErrorHandle_t execute(CustomOp* operation) {

    auto m_Input  = operation->getInput(0);
    auto m_Outputs = operation->getOutput(0);

    const uint32_t rank = m_Outputs->rank;
    const size_t depth = m_Outputs->currentDimensions[rank - 1];
    uint32_t tensorLength = 1;
    for(uint32_t j = 0; j < rank; ++j)
    {
        tensorLength *= (uint32_t)(m_Outputs->currentDimensions[j]);

    }
    const size_t numPixels = tensorLength/depth;
    for( size_t pix = 0; pix < numPixels; ++pix )
    {
        const float* in = (float*)m_Input->data+pix*depth;
        float* out = (float*)m_Outputs->data+pix*depth;

        // find the max element for max subtraction
        float maxElt = std::numeric_limits<float>::lowest();
        for( size_t i = 0; i < depth; ++i )
        {
            maxElt = std::max( maxElt, in[i] );
        }

        // compute exponentiations
        float expSum = 0.0;
        for( size_t i = 0; i < depth; ++i )
        {
            const float ei = expf( in[i] - maxElt );
            out[i] = ei;
            expSum += ei;
        }
        // normalize
        for( size_t i = 0; i < depth; ++i )
        {
            out[i] = out[i] / expSum;
        }
    }
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t finalize(const CustomOp* operation) {
  QNN_CUSTOM_BE_ENSURE_EQ(operation->numInput(), 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)
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

  // Add output
  for (uint32_t i = 0; i < numOutputs(node); i++) {
    operation->addOutput(getOutput(node, i));
  }


  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t validateOpConfig(Qnn_OpConfig_t opConfig) {
  QNN_CUSTOM_BE_ENSURE_EQ(
      strcmp(opConfig.v1.typeName, "Softmax"), 0, QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT)

  QNN_CUSTOM_BE_ENSURE_EQ(opConfig.v1.numOfInputs, 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)
  QNN_CUSTOM_BE_ENSURE_EQ(opConfig.v1.numOfOutputs, 1, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)

  return QNN_SUCCESS;
}
}  // namespace softmax

CustomOpRegistration_t* register_SoftmaxCustomOp() {
  using namespace softmax;
  static CustomOpRegistration_t SoftmaxRegister = {execute, finalize, free, validateOpConfig, populateFromNode};
  return &SoftmaxRegister;
}

REGISTER_OP(Softmax, register_SoftmaxCustomOp);