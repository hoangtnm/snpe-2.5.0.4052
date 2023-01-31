# ==============================================================================
#
#  Copyright (c) 2019-2020,2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import caffe
from .caffe_base_translation import CaffeTranslationBase, CaffeTranslations
from qti.aisw.converters.common.converter_ir import op_adapter
from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.utils.converter_utils import *


# ------------------------------------------------------------------------------
#   Dropout, and other identity ops
# ------------------------------------------------------------------------------
class CaffeIdentityTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, converter_context):
        if len(layer.top) != len(layer.bottom):
            raise RuntimeError(code_to_message.get_error_message('ERROR_CAFFE_NUM_BOTTOM_NOT_EQ_TO_NUM_TOP'))
        return op_adapter.IdentityOp(layer.name)


CaffeTranslations.register_translation(CaffeIdentityTranslation(),
                                       converter_type('dropout', 'caffe'),
                                       converter_type(caffe.proto.caffe_pb2.V1LayerParameter.DROPOUT, 'caffe'),
                                       op_adapter.IdentityOp.TRANSLATION_KEY)


class CaffeStaticTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, converter_context):
        return None

    def extract_input_names(self, src_op, converter_context):
        # return no input names since it is a static op
        return []

    def extract_output_names(self, src_op, converter_context):
        # return no output names since it is a static op
        return []


CaffeTranslations.register_translation(CaffeStaticTranslation(),
                                       converter_type('silence', 'caffe'),
                                       converter_type('accuracy', 'caffe'),
                                       converter_type('softmaxwithloss', 'caffe'))
