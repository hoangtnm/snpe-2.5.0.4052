# ==============================================================================
#
#  Copyright (c) 2019-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from qti.aisw.converters.common.converter_ir import translation

# ------------------------------------------------------------------------------
#   CaffeTranslation
# ------------------------------------------------------------------------------
CaffeTranslations = translation.TranslationBank()


class CaffeTranslationBase(translation.ConversionTranslationBase):

    def __init__(self):
        translation.ConversionTranslationBase.__init__(self)

    def add_src_op_info(self, node_name, src_op, graph):
        # Create a mapping of all layers and their inputs/outputs
        graph.add_src_op_info(node_name, [str(i) for i in src_op.bottom],
                                         [str(o) for o in src_op.top])

    def extract_parameters(self, src_op, converter_context):
        raise NotImplementedError("extract_parameters() for {} not implemented ".format(str(self.__class__.__name__)))

    def extract_input_names(self, src_op, converter_context):
        return list(map(str, src_op.bottom))

    def extract_output_names(self, src_op, converter_context):
        return list(map(str, src_op.top))



