# ==============================================================================
#
#  Copyright (c) 2019-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import collections
import numpy

import caffe
import caffe.proto.caffe_pb2 as caffe_pb2

from .caffe_base_translation import CaffeTranslationBase, CaffeTranslations

from qti.aisw.converters.common import ir_graph
from qti.aisw.converters.common.converter_ir import op_adapter
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisTracker
from qti.aisw.converters.common.converter_ir.op_adapter import IRPaddingStrategies
from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.utils.converter_utils import *


# helpers
def get_conv_params(conv_param):
    parms_type = collections.namedtuple("ConvParams", ["pad_x", "pad_y", "stridex", "stridey"])
    pad_x, pad_y = 0, 0
    if conv_param.pad_h or conv_param.pad_w:
        pad_x = conv_param.pad_w
        pad_y = conv_param.pad_h
    elif isinstance(conv_param.pad, int):
        # Segnet version of caffe.proto has defined  pad optional (not repeated).
        # It implies that it is scalar rather vector
        pad_x = conv_param.pad
        pad_y = conv_param.pad
    else:
        if len(conv_param.pad) > 0:
            pad_x = conv_param.pad[0]
            pad_y = conv_param.pad[0]
        if len(conv_param.pad) > 1:
            pad_x = conv_param.pad[1]

    stride_x, stride_y = 1, 1
    if conv_param.stride_h or conv_param.stride_w:
        stride_x = conv_param.stride_w
        stride_y = conv_param.stride_h
    elif isinstance(conv_param.stride, int):
        # Segnet version of caffe.proto has defined  stride optional (not repeated).
        # It implies that it is scalar rather vector
        stride_x = conv_param.stride
        stride_y = conv_param.stride
    else:
        if len(conv_param.stride) > 0:
            stride_x = conv_param.stride[0]
            stride_y = conv_param.stride[0]
        if len(conv_param.stride) > 1:
            stride_x = conv_param.stride[1]

    return parms_type(pad_x, pad_y, stride_x, stride_y)


def align_weight_and_bias_as_input(name, layer, graph, weights, bias):
    weights_name = layer.name + f"_{name}_w"
    bias_name = layer.name + f"_{name}_b"
    weights_constant_op = op_adapter.ConstantOp(weights_name, tensor=weights)
    bias_constant_op = op_adapter.ConstantOp(bias_name, tensor=bias)
    weight_node = graph.add(weights_constant_op, [], [weights_name], axis_formats=[AxisTracker.AxisFormat.ANY])
    bias_node = graph.add(bias_constant_op, [], [bias_name], axis_formats=[AxisTracker.AxisFormat.ANY])
    graph.add_src_op_info(weights_name, None, weight_node.output_names[0])
    graph.add_src_op_info(bias_name, None, bias_node.output_names[0])

    return weights_name, bias_name


# -----------------------------------------------------------------
# Converter translations
# -----------------------------------------------------------------
class CaffeBatchNormalizationTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)
        self.weights_name = None
        self.bias_name = None

    def extract_parameters(self, layer, converter_context):
        graph = converter_context.ir_graph
        # from the batch_norm layer we get weights W1 and bias B1:
        # y  = W1.x + B1
        input_name = graph.naming_policy.get_input_names(layer, layer.bottom)[0]
        prev_bias = None
        if hasattr(graph.get_buffer(input_name).producer.op, 'bias'):
            prev_bias = len(graph.get_buffer(input_name).producer.op.bias)
        weights, bias = converter_context.weights.get_batch_norm_weights(layer, prev_bias)

        if any(numpy.isnan(weights)) or any(numpy.isinf(weights)):
            log_warning("Got NaN/Inf weights in {} layer, initializing with 1's".format(layer.name))
            weights.fill(1)
        if any(numpy.isnan(bias)) or any(numpy.isinf(bias)):
            bias.fill(0)
            log_warning("Got NaN/Inf bias in {} layer, initializing with 0's".format(layer.name))

        # If use_global_stats is False (Caffe training mode) treat this as instance normalization
        compute_statistics = False
        epsilon = layer.batch_norm_param.eps
        if layer.batch_norm_param.HasField("use_global_stats") and not layer.batch_norm_param.use_global_stats:
            # Reset weights and biases to 1s and 0s
            weights.fill(1)
            bias.fill(0)
            compute_statistics = True

        self.weights_name, self.bias_name = align_weight_and_bias_as_input('bn', layer, graph, weights, bias)

        if compute_statistics:
            return op_adapter.InstanceNormOp(layer.name,
                                             mode=ir_graph.QNN_OP_INSTANCE_NORM_MODE_MU_SIGMA,
                                             region=ir_graph.QNN_OP_INSTANCE_NORM_REGION_ACROSS_SPATIAL,
                                             epsilon=epsilon
                                             )

        return op_adapter.BatchnormOp(layer.name)

    def extract_input_names(self, layer, converter_context):
        graph = converter_context.ir_graph
        input_name = graph.naming_policy.get_input_names(layer, layer.bottom)[0]
        return [input_name, self.weights_name, self.bias_name]


CaffeTranslations.register_translation(CaffeBatchNormalizationTranslation(),
                                       converter_type('batchnorm', 'caffe'),
                                       op_adapter.BatchnormOp.TRANSLATION_KEY)


# This is placeholder in-case needed in the future(no use-case identified so far),
# hence not registered as part of CaffeTranslations atm and no sdk-tests exercising this layer
class CaffeBNTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)
        self.weights_name = None
        self.bias_name = None

    def extract_parameters(self, layer, converter_context):
        graph = converter_context.ir_graph
        weights, bias = converter_context.weights.get_bn_weights(layer)
        self.weights_name, self.bias_name = align_weight_and_bias_as_input('bn', layer, graph, weights, bias)

        return op_adapter.BatchnormOp(layer.name)

    def extract_input_names(self, layer, converter_context):
        graph = converter_context.ir_graph
        input_name = graph.naming_policy.get_input_names(layer, layer.bottom)[0]
        return [input_name, self.weights_name, self.bias_name]

# CaffeTranslations.register_translation(CaffeBNTranslation(),
#                                        converter_type('bn', 'caffe'))


class CaffeConvTranslation(CaffeTranslationBase):

    _WEIGHT = "_weight"
    _BIAS = "_bias"

    def __init__(self):
        CaffeTranslationBase.__init__(self)
        self.weight_node_name = ""
        self.bias_node_name = ""

    def extract_parameters(self, layer, converter_context):
        graph = converter_context.ir_graph
        conv_param = layer.convolution_param
        self.weight_node_name = layer.name + self._WEIGHT
        self.bias_node_name = layer.name + self._BIAS

        # Extract and add static weights/biases in Caffe to IR graph as ConstantOp inputs
        bias_term = getattr(conv_param, "bias_term", True)
        c_weights, c_bias = converter_context.weights.get_conv_weights(layer, bias_term)

        weights_op = op_adapter.ConstantOp(self.weight_node_name, tensor=c_weights)
        log_debug(code_to_message.get_debugging_message('DEBUG_EXTRACT_WEIGHTS')(layer.name, c_weights.shape))
        graph.add(weights_op, [], [self.weight_node_name], axis_formats=[AxisTracker.AxisFormat.OIHW])

        bias_op = op_adapter.ConstantOp(self.bias_node_name, tensor=c_bias)
        log_debug(code_to_message.get_debugging_message('DEBUG_EXTRACT_BIAS')(layer.name, c_bias.shape))
        graph.add(bias_op, [], [self.bias_node_name], axis_formats=[AxisTracker.AxisFormat.ANY])

        # Extract convolution parameters
        groups = getattr(conv_param, "group", 1)
        pad_x, pad_y, stride_x, stride_y = get_conv_params(conv_param)

        dilation_x, dilation_y = 1, 1
        if len(conv_param.dilation) > 0:
            dilation_x = conv_param.dilation[0]
            dilation_y = conv_param.dilation[0]
        if len(conv_param.dilation) > 1:
            dilation_x = conv_param.dilation[1]

        # Determine whether this is a regular Conv2dOp or DepthwiseConv2dOp
        input_name = self.extract_input_names(layer, converter_context)[0]
        num_input_channels = graph.src_axis_order.extract_2d_spatial_dims(graph.get_buffer(input_name).shape)[-1]
        weights_shape = graph.get_buffer(self.weight_node_name).shape
        num_output_channels = graph.src_axis_order.extract_conv2d_weights_dims(weights_shape)[-1]

        convolution_class = op_adapter.Conv2dOp
        # Criteria is that groups == num_input_channels (from input) == num_output_channels
        if groups == num_input_channels and num_input_channels == num_output_channels:
            convolution_class = op_adapter.DepthwiseConv2dOp

        return convolution_class(layer.name,
                                 padx_before=pad_x,
                                 padx_after=pad_x,
                                 pady_before=pad_y,
                                 pady_after=pad_y,
                                 stridex=stride_x,
                                 stridey=stride_y,
                                 dilationx=dilation_x,
                                 dilationy=dilation_y,
                                 groups=groups)

    def extract_input_names(self, layer, converter_context):
        # Extend source input names with constant nodes added for weights/bias
        src_input_names = list(map(str, layer.bottom))
        src_input_names.extend([self.weight_node_name, self.bias_node_name])
        return src_input_names


CaffeTranslations.register_translation(CaffeConvTranslation(),
                                       converter_type('convolution', 'caffe'),
                                       converter_type(caffe.proto.caffe_pb2.V1LayerParameter.CONVOLUTION, 'caffe'),
                                       op_adapter.Conv2dOp.TRANSLATION_KEY)


class CaffeDeConvTranslation(CaffeConvTranslation):

    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, converter_context):
        graph = converter_context.ir_graph
        conv_param = layer.convolution_param
        self.weight_node_name = layer.name + self._WEIGHT
        self.bias_node_name = layer.name + self._BIAS

        # Extract and add static weights/biases in Caffe to IR graph as ConstantOp inputs
        bias_term = getattr(conv_param, "bias_term", True)
        c_weights, c_bias = converter_context.weights.get_deconv_weights(layer, bias_term)
        weights_op = op_adapter.ConstantOp(self.weight_node_name, tensor=c_weights)
        log_debug(code_to_message.get_debugging_message('DEBUG_EXTRACT_WEIGHTS')(layer.name, c_weights.shape))
        graph.add(weights_op, [], [self.weight_node_name], axis_formats=[AxisTracker.AxisFormat.IOHW])

        bias_op = op_adapter.ConstantOp(self.bias_node_name, tensor=c_bias)
        log_debug(code_to_message.get_debugging_message('DEBUG_EXTRACT_BIAS')(layer.name, c_bias.shape))
        graph.add(bias_op, [], [self.bias_node_name], axis_formats=[AxisTracker.AxisFormat.ANY])

        # Extract deconvolution parameters
        pad_x, pad_y, stride_x, stride_y = get_conv_params(conv_param)
        groups = conv_param.group if hasattr(conv_param, "group") else 1

        return op_adapter.TransposeConv2dOp(name=layer.name,
                                            bias_op_name=self.bias_node_name,
                                            stridex=stride_x,
                                            stridey=stride_y,
                                            padx_before=pad_x,
                                            padx_after=pad_x,
                                            pady_before=pad_y,
                                            pady_after=pad_y,
                                            padding_size_strategy=ir_graph.PADDING_SIZE_EXPLICIT,
                                            output_height=0,
                                            output_width=0,
                                            groups=groups)


CaffeTranslations.register_translation(CaffeDeConvTranslation(),
                                       converter_type('deconvolution', 'caffe'),
                                       converter_type(caffe.proto.caffe_pb2.V1LayerParameter.DECONVOLUTION, 'caffe'),
                                       op_adapter.TransposeConv2dOp.TRANSLATION_KEY)


class CaffeFullyConnectedTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, converter_context):
        graph = converter_context.ir_graph
        # Compute parameters for fc layer
        c_input_names = graph.naming_policy.get_input_names(layer, layer.bottom)
        if len(c_input_names) != 1:
          raise ValueError("FullyConnected expects only 1 input, got {}".format(len(c_input_names)))
        input_channel = graph.get_buffer(c_input_names[0]).get_buf_dims()[1]
        fc_param = layer.inner_product_param
        bias_term = getattr(fc_param, "bias_term", True)
        c_weights, c_bias = converter_context.weights.get_fc_weights(layer, input_channel, bias_term)
        self.weights_name, self.bias_name = align_weight_and_bias_as_input('fc', layer, graph, c_weights, c_bias)

        return op_adapter.FullyConnectedOp(name=layer.name)

    def extract_input_names(self, layer, converter_context):
        graph = converter_context.ir_graph
        input_name = graph.naming_policy.get_input_names(layer, layer.bottom)[0]
        return [input_name, self.weights_name, self.bias_name]


CaffeTranslations.register_translation(CaffeFullyConnectedTranslation(),
                                       converter_type('innerproduct', 'caffe'),
                                       converter_type(caffe.proto.caffe_pb2.V1LayerParameter.INNER_PRODUCT, 'caffe'),
                                       op_adapter.FullyConnectedOp.TRANSLATION_KEY)


class CaffeMVNTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)
        self.weights_name = None
        self.bias_name = None

    def extract_parameters(self, layer, converter_context):
        graph = converter_context.ir_graph
        prev_layer = graph.naming_policy.get_input_names(layer, layer.bottom)[0]
        layer_channel = graph.get_buffer(prev_layer).get_buf_dims()[1]

        # Generate unit weights and zero bias. Makes it compatible with Batchnorm
        weights = numpy.full(layer_channel,1.0,dtype=numpy.float32)
        bias = numpy.full(layer_channel,0.0,dtype=numpy.float32)
        self.weights_name, self.bias_name = align_weight_and_bias_as_input('in', layer, graph, weights, bias)

        region = ir_graph.QNN_OP_INSTANCE_NORM_REGION_ACROSS_SPATIAL
        if layer.mvn_param.across_channels:
            region = ir_graph.QNN_OP_INSTANCE_NORM_REGION_ACROSS_CHANNEL

        return op_adapter.InstanceNormOp(layer.name,
                                         mode=ir_graph.QNN_OP_INSTANCE_NORM_MODE_MU_SIGMA,
                                         across_spatial=region,
                                         epsilon=layer.mvn_param.eps,
                                         normalize_variance=layer.mvn_param.normalize_variance)

    def extract_input_names(self, layer, converter_context):
        graph = converter_context.ir_graph
        input_name = graph.naming_policy.get_input_names(layer, layer.bottom)[0]
        return [input_name, self.weights_name, self.bias_name]


CaffeTranslations.register_translation(CaffeMVNTranslation(),
                                       converter_type('mvn', 'caffe'))


class CaffeNormalizeTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)
        self.weights_name = None
        self.bias_name = None

    def extract_parameters(self, layer, converter_context):
        graph = converter_context.ir_graph
        weights = converter_context.weights.get_normalize_weights(layer).flatten(order='C')

        # from the normalize layer we get weights W:
        # if channel_shared is true, there is only a single weight which we will
        # replicate across the input channels.
        if layer.norm_param.channel_shared:
            input_name = graph.naming_policy.get_input_names(layer, layer.bottom)[0]
            input_channel = graph.get_buffer(input_name).get_buf_dims()[1]
            weights = weights[0] * numpy.ones([input_channel], dtype=numpy.float32)
        # this layer does not support bias values. construct an array of zeros.
        bias = numpy.zeros(shape=[len(weights)], dtype=numpy.float32)
        self.weights_name, self.bias_name = align_weight_and_bias_as_input('in', layer, graph, weights, bias)

        region = ir_graph.QNN_OP_INSTANCE_NORM_REGION_ACROSS_SPATIAL
        if not layer.norm_param.across_spatial:
            region = ir_graph.QNN_OP_INSTANCE_NORM_REGION_ACROSS_CHANNEL
        return op_adapter.InstanceNormOp(layer.name,
                                         mode=ir_graph.QNN_OP_INSTANCE_NORM_MODE_RMS,
                                         region=region)

    def extract_input_names(self, layer, converter_context):
        graph = converter_context.ir_graph
        input_name = graph.naming_policy.get_input_names(layer, layer.bottom)[0]
        return [input_name, self.weights_name, self.bias_name]


CaffeTranslations.register_translation(CaffeNormalizeTranslation(),
                                       converter_type('normalize', 'caffe'))


class CaffePoolTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, converter_context):
        graph = converter_context.ir_graph
        pool_param = layer.pooling_param

        c_pool_type = ir_graph.QNN_OP_POOL_MAX_2D
        if pool_param.pool:
            c_pool_type = ir_graph.QNN_OP_POOL_AVG_2D

        size_x = pool_param.kernel_size
        size_y = size_x
        if pool_param.kernel_h or pool_param.kernel_w:
            size_x = pool_param.kernel_w
            size_y = pool_param.kernel_h

        stride_x = pool_param.stride
        stride_y = stride_x
        if pool_param.stride_h or pool_param.stride_w:
            stride_x = pool_param.stride_w
            stride_y = pool_param.stride_h

        pad_x = pool_param.pad
        pad_y = pad_x
        if pool_param.pad_h or pool_param.pad_w:
            pad_x = pool_param.pad_w
            pad_y = pool_param.pad_h

        include_padding = True
        input_name = graph.naming_policy.get_input_names(layer, layer.bottom)[0]
        input_dim = graph.get_buffer(input_name).get_buf_dims()
        if pool_param.global_pooling:
            size_y = input_dim[2]
            size_x = input_dim[3]
            stride_x, stride_y = 1, 1
            pad_x, pad_y = 0, 0
            include_padding = False

        # if there is a second top, this will be upsampled later.
        if len(layer.top) > 1:
            if size_x != size_y or stride_x != stride_y or pad_x != pad_y:
                raise ValueError(
                    code_to_message.get_error_message('ERROR_CAFFE_INDEX_BASED_UPSAMPLING_DOES_NOT_SUPPORT_RECT_POOL')
                    (str(layer.name)))  # TODO: should this go inside ir-to-dlc?

        return op_adapter.Pool2dOp(layer.name,
                                   pool_type=c_pool_type,
                                   size_x=size_x,
                                   size_y=size_y,
                                   stride_x=stride_x,
                                   stride_y=stride_y,
                                   padx_before=pad_x,
                                   padx_after=pad_x,
                                   pady_before=pad_y,
                                   pady_after=pad_y,
                                   padding_size_strategy=IRPaddingStrategies.PADDING_SIZE_EXPLICIT,
                                   count_pad_for_edges=include_padding)


CaffeTranslations.register_translation(CaffePoolTranslation(),
                                       converter_type('pooling', 'caffe'),
                                       converter_type(caffe.proto.caffe_pb2.V1LayerParameter.POOLING, 'caffe'),
                                       op_adapter.Pool2dOp.TRANSLATION_KEY)


class CaffeRoiPoolTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, converter_context):
        graph = converter_context.ir_graph
        roi_pool_param = layer.roi_pooling_param

        pooled_size_h = roi_pool_param.pooled_h
        pooled_size_w = roi_pool_param.pooled_w
        spatial_scale = roi_pool_param.spatial_scale

        # The output channel is equal to the input feature map channel. We are assuming that input[0] is the feature map.
        input_name = graph.naming_policy.get_input_names(layer, layer.bottom)[0]
        input_dims = graph.get_buffer(input_name).get_buf_dims()
        output_dim = [input_dims[0], input_dims[1], pooled_size_h, pooled_size_w]
        log_debug(code_to_message.get_debugging_message("DEBUG_INFERRED_SHAPE")(layer.name, output_dim))

        return op_adapter.RoiPoolingOp(layer.name,
                                       output_shape=output_dim,
                                       pooled_size_h=pooled_size_h,
                                       pooled_size_w=pooled_size_w,
                                       spatial_scale=spatial_scale)

    def infer_output_shapes(self, op, input_shapes):
        return [op.output_shape]


CaffeTranslations.register_translation(CaffeRoiPoolTranslation(),
                                       converter_type('roipooling', 'caffe'),
                                       op_adapter.RoiPoolingOp.TRANSLATION_KEY)


class CaffeScaleTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def add_op(self, src_op, converter_context):
        # Caffe ScaleLayer is equivalent to input * weights + bias. ScaleLayer computes the
        # elementwise product of the input and weights, with the dimensions of weights "broadcast"
        # to match the dimensions of input based on the axis and num_axes parameters. ScaleLayer can
        # additionally perform a "broadcast" sum between the result of input * weights and bias when
        # bias_term is set to true. The weights input may be omitted, in which case it's learned as
        # parameter of the layer. The bias is always learned as parameter of the layer if it exists.
        # Here translates ScaleLayer to ElementWiseAdd(ElementWiseMultiply(input, weights), bias).
        graph = converter_context.ir_graph
        weights, bias, axis, num_axes = self.extract_parameters(src_op, converter_context)
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)

        # If only one input is given, inputs[0] is the input, and the weights is a learned parameter
        # that can be extracted from layer parameters. If two inputs are given, inputs[0] is the
        # input, and inputs[1] is the weights.
        input_dim = graph.get_buffer(input_names[0]).shape
        if len(input_names) == 2:
            weight_dim = graph.get_buffer(input_names[1]).get_buf_dims()
        else:
            weight_dim = weights.shape

        # Note: the broadcast rules of QNN ElementwiseBinaryOp and Caffe ScaleLayer are different.
        # For ElementwiseBinaryOp, dimensions are right alignment, and the 1-extending starts with
        # the trailing dimensions of weights until the ranks are equal. For ScaleLayer, only
        # dimensions with indices [0:axis] and [axis+num_axes:] need to be 1-extended. For example,
        # the dimensions of input and weights are [100,3,40,60] and [3,40] respectively, and axis=1,
        # num_axes=2. For ElementwiseBinaryOp, the dimensions of weights are 1-extended to
        # [1,1,3,40] from right. However, for ScaleLayer, the dimensions of weights should be
        # 1-extended to [1,3,40,1] based on the values of axis and num_axes. To be consistent with
        # the definition of ScaleLayer, the dimensions of weights are 1-extended in advance here.
        # Caculates 1-extended dimensions for weights
        weight_extended_dim = []
        weight_extended_dim.extend([1] * axis)
        weight_extended_dim.extend(weight_dim)
        weight_extended_dim.extend([1] * (len(input_dim) - (axis + num_axes)))

        # Reshapes dimensions of weights to 1-extended dimensions
        if weight_dim != weight_extended_dim:
            # If weights is provided as an input, uses ReshapeOp to change its dimensions.
            # Otherwise, uses numpy.ndarray.reshape to change its dimensions.
            if len(input_names) == 2:
                reshape_op_name = input_names[1] + "_reshape"
                reshape_op = op_adapter.ReshapeOp(reshape_op_name, shape=weight_extended_dim)
                graph.add(reshape_op, [input_names[1]], [reshape_op_name], axis_formats=[
                          graph.get_buffer(input_names[0]).get_axis_format()])
                input_names[1] = reshape_op_name
            else:
                weights = weights.reshape(weight_extended_dim)

        # If weights is a learned parameter, adds ConstantOp which holds weights value
        if len(input_names) == 1:
            weights_const_op_name = src_op.name + "_weights_const"
            weights_const_op = op_adapter.ConstantOp(weights_const_op_name, tensor=weights)
            graph.add(weights_const_op, [], weights_const_op_name,
                      axis_formats=[graph.get_buffer(input_names[0]).get_axis_format()])
            input_names.append(weights_const_op_name)

        # Adds ElementWiseMultiplyOp which computes the elementwise product of the input and weights
        mul_op_name = src_op.name + "_mul"
        mul_op = op_adapter.ElementwiseBinaryOp(
            mul_op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_MULTIPLY)
        if bias is None:
            # Adds ElementWiseMultiplyOp node with ScaleLayer output name
            return graph.add(mul_op, input_names, output_names)
        else:
            # Adds ElementWiseMultiplyOp node with intermediate output name
            graph.add(mul_op, input_names, [mul_op_name])

            # Caculates 1-extended dimensions for bias
            bias_dim = bias.shape
            bias_extended_dim = []
            bias_extended_dim.extend([1] * axis)
            bias_extended_dim.extend(bias_dim)
            bias_extended_dim.extend([1] * (len(input_dim) - (axis + num_axes)))

            # Reshapes dimensions of bias to 1-extended dimensions
            if bias_dim != bias_extended_dim:
                bias = bias.reshape(bias_extended_dim)

            # Adds ConstantOp which holds bias value
            bias_const_op_name = src_op.name + "_bias_const"
            bias_const_op = op_adapter.ConstantOp(bias_const_op_name, tensor=bias)
            graph.add(bias_const_op, [], bias_const_op_name, axis_formats=[
                      graph.get_buffer(input_names[0]).get_axis_format()])

            # Adds ElementWiseAddOp which computes the elementwise sum between the result of
            # input * weights and bias
            add_op_name = src_op.name + "_add"
            add_op = op_adapter.ElementwiseBinaryOp(
                add_op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_ADD)
            return graph.add(add_op, [mul_op_name, bias_const_op_name], output_names)

    def extract_parameters(self, layer, converter_context):
        graph = converter_context.ir_graph
        scale_param = layer.scale_param
        bias_term = getattr(scale_param, "bias_term", False)
        input_name = graph.naming_policy.get_input_names(layer, layer.bottom)[0]
        input_dim = graph.get_buffer(input_name).get_buf_dims()

        input_channel = input_dim[1]
        s_weights, s_bias = converter_context.weights.get_scale_weights(
            layer, bias_term, input_channel)

        # Because the get_scale_weights() function returns a numpy.ndarray for s_bias whether
        # bias_term is True or False, here reassigns s_bias to None if bias_term is False
        if bias_term is False:
            s_bias = None

        axis = scale_param.axis
        if axis < 0:
            axis = len(input_dim) + axis

        if len(layer.bottom) == 2:
            weight_input_name = graph.naming_policy.get_input_names(layer, layer.bottom)[1]
            weight_input_dim = graph.get_buffer(weight_input_name).get_buf_dims()
        else:
            weight_input_dim = s_weights.shape

        if len(layer.bottom) == 2:
            # When two bottoms are given, num_axes is determined by the rank of bottom[1]
            num_axes = len(weight_input_dim)
        else:
            num_axes = scale_param.num_axes
            if num_axes < 0:
                num_axes = len(input_dim) - axis

        for i in range(len(weight_input_dim)):
            if input_dim[axis + i] != weight_input_dim[i]:
                raise ValueError("Dimension at index {} of bottom[0] does not match dimension at "\
                                 "index {} of weight: {} vs {}.".format(axis + i, i,
                                 input_dim[axis + i], weight_input_dim[i]))

        return [s_weights, s_bias, axis, num_axes]


CaffeTranslations.register_translation(CaffeScaleTranslation(), converter_type('scale', 'caffe'))


class CaffeArgMaxTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, converter_context):
        argmax_param = layer.argmax_param
        return op_adapter.ArgOp(layer.name,
                                arg_type=ir_graph.QNN_OP_ARGMAX,
                                axis=argmax_param.axis,
                                keep_dims=True)


CaffeTranslations.register_translation(CaffeArgMaxTranslation(),
                                       converter_type('argmax', 'caffe'),
                                       op_adapter.ArgOp.TRANSLATION_KEY)
