# =============================================================================
#
#  Copyright (c) 2015-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.converters.common import ir_graph as c_ir_graph
from qti.aisw.converters.common.converter_ir.op_adapter import LrnOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence
)


class LrnLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, operations, window_size, alpha, beta, bias):
            super(LrnLayerResolver.Descriptor, self).__init__('LRN', name, operations)
            self.window_size = window_size
            self.alpha = alpha
            self.beta = beta
            self.bias = bias

    def __init__(self):
        self.sequence = GraphSequence([ConverterSequenceNode('root', ['LRN'])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        if len(matches) == 0:
            return []
        potential_descriptors = []
        for match in matches:
            lrn_op = match['root']
            window_size = 1 + lrn_op.get_attr('depth_radius') * 2
            alpha = lrn_op.get_attr('alpha')
            beta = lrn_op.get_attr('beta')
            bias = lrn_op.get_attr('bias')
            consumed_nodes = match.consumed_nodes
            potential_descriptors.append(
                LrnLayerResolver.Descriptor(str(lrn_op.name), consumed_nodes, window_size, alpha, beta, bias))
        return potential_descriptors


class LrnLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: LrnLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return ir_graph.add(LrnOp(name=descriptor.layer_name,
                                  alpha=float(descriptor.alpha),
                                  beta=descriptor.beta,
                                  bias=descriptor.bias,
                                  radius=int((descriptor.window_size-1)/2),
                                  region=c_ir_graph.QNN_OP_LRN_REGION_ACROSS_CHANNEL),
                            input_names=input_name,
                            output_names=output_name)
