# =============================================================================
#
#  Copyright (c) 2018-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.converters.common.converter_ir.op_adapter import DepthToSpaceOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence
)
from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.utils import converter_utils


class DepthToSpaceLayerResolver(LayerResolver, object):
    TF_ATTRIBUTE_BLOCK_SIZE = 'block_size'
    TF_ATTRIBUTE_DATA_FORMAT = 'data_format'
    TF_ATTRIBUTE_SUPPORTED_DATA_FORMAT = ['NHWC']

    class Descriptor(LayerDescriptor):
        def __init__(self, layer_type, name, nodes, block_size):
            super(DepthToSpaceLayerResolver.Descriptor, self).__init__(layer_type, name, nodes)
            self.block_size = block_size

        def is_input_tensor(self, op, tensor):
            return tensor == op.inputs[0]

        @property
        def output_names(self):
            return [str(self.child_ops[0].outputs[0].name)]

        def is_output_op(self, op):
            return op in self.child_ops

        def get_output_names_for(self, input_tensors):
            return self.output_names

    def __init__(self):
        self.sequence = GraphSequence([ConverterSequenceNode('root', ['DepthToSpace'])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)

        # Nothing matched
        if len(matches) == 0:
            return []

        potential_descriptors = []
        for match in matches:
            depth_to_space_op = match['root']
            block_size = [depth_to_space_op.get_attr(self.TF_ATTRIBUTE_BLOCK_SIZE)] * 2
            data_format = depth_to_space_op.get_attr(self.TF_ATTRIBUTE_DATA_FORMAT).decode('utf-8')
            converter_utils.log_assert(data_format in self.TF_ATTRIBUTE_SUPPORTED_DATA_FORMAT,
                                       code_to_message.get_error_message("ERROR_TF_DEPTH_TO_SPACE_DATA_FORMAT")
                                       (self.TF_ATTRIBUTE_SUPPORTED_DATA_FORMAT, data_format))
            consumed_nodes = match.consumed_nodes

            potential_descriptors.append(
                DepthToSpaceLayerResolver.Descriptor('DepthToSpace',
                                                     str(depth_to_space_op.name),
                                                     consumed_nodes,
                                                     block_size))

        return potential_descriptors


class DepthToSpaceLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: DepthToSpaceLayerResolver.Descriptor
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return ir_graph.add(DepthToSpaceOp(name=descriptor.layer_name,
                                           block_size=descriptor.block_size),
                            input_names=[input_name],
                            output_names=[output_name])
