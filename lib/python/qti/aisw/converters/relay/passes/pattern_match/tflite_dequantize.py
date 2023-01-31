# ==============================================================================
#
#  Copyright (c) 2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import numpy as np
import tvm
from tvm.relay.dataflow_pattern import *

DataTypeBitwidths = {
    'float16': 16,
    'float32': 32,
    'int8': 8,
    'int16': 16,
    'int32': 32,
    'uint8': 8,
    'uint16': 16,
    'uint32': 32,
    None: 8
}


def dequantize(data, scale, zp):
    return (data.astype(scale.dtype)-zp.astype(scale.dtype))*scale


def get_key_from_expr(expr: tvm.relay.expr):
    return hash(expr)


class DequantizeQnnPattern(tvm.relay.ExprMutator):

    def __init__(self, dtype_dict, expr_to_quantization_params_dict, output_names_dict):
        super().__init__()
        self.dtype_dict = dtype_dict
        self.pattern = self.get_pattern()

        # quantization info dict for relay expr, will be used in translation
        self.expr_to_quantization_params_dict = expr_to_quantization_params_dict

        # we need to reset output names dict for dequantize relay
        # e.g., origin is dict[hash(qnn.conv2d)]: 'Conv2d', need to update to dict[hash(conv2d)]: 'Conv2d'
        self.output_names_dict = output_names_dict

        # use span to keep output name for old expr, will be used to set output names for new expr
        self.span_output_names_dict = {expr.span: name for _, (expr, name) in self.output_names_dict.items() if expr.span is not None}

        # clear output names dict, it will be filled after visit_function
        self.output_names_dict.clear()

        # type count for dequantize pass to generate new span
        self.type_count = {}

    def get_pattern(self):
        """all dequantize pattern need to implement this function"""
        raise NotImplementedError()

    def dequantize_qnn_expr(self, expr, args):
        """all dequantize pattern need to implement this function"""
        raise NotImplementedError()

    def visit_call(self, call):
        # if match quantized pattern, dequantize them
        # else recursive visit the args and create a new call since their args may change in dequantize pass
        if self.pattern.match(call):
            args = [self.visit(arg) for arg in call.args]
            # after vistiing, the args may be updated, so we need to pass new args in this way
            new_call = self.dequantize_qnn_expr(call, args)
        else:
            new_fn = self.visit(call.op)
            new_args = [self.visit(arg) for arg in call.args]
            new_call = tvm.relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)

        # set output names for call, basically only relay.Call will have output names
        # refer https://github.com/apache/tvm/blob/993a8ea094575f2823aebf2b8eb37e9f4ac44d7d/python/tvm/relay/frontend/tflite.py#L280
        self.set_output_names(call, new_call)
        return new_call

    def visit_function(self, fn):
        new_body = self.visit(fn.body)
        # not like ExprMutator, we need to get free_vars after visit body since new relay.Var may be added
        new_params = tvm.relay.analysis.free_vars(new_body)
        return tvm.relay.Function(list(new_params), new_body, fn.ret_type, fn.type_params, fn.attrs)

    def populate_quantization_info(self, old_expr, new_expr, offsets, scales, is_symmetric=False, dtype=None):
        """
        this function should be overrided for those op having multiple output, e.g. relay.split
        since they need to populate each encoding for each output
        """
        # set span before populating quantization info, so later we can use expr to get quantization info
        new_expr = self.set_span(old_expr, new_expr)

        # need to broadcast offset for per channel op
        offsets = np.broadcast_to(offsets, scales.shape)

        # get bitwidth from dtype
        bw = DataTypeBitwidths[dtype]

        if scales.size > 1:
            # per channel quantization
            q_info = []
            for offset, scale in zip(offsets, scales):
                q_info.append({
                    "bitwidth": bw,
                    "offset": offset,
                    "scale": scale,
                    "is_symmetric": "True", # symmetric is True for per channel quantization
                })
        else:
            q_info = {
                "bitwidth": bw,
                "offset": offsets,
                "scale": scales,
                "is_symmetric": str(is_symmetric),
            }
        key = get_key_from_expr(new_expr)
        # the q_info is contained in a list to sequentially map output names
        # generated in relay translation
        self.expr_to_quantization_params_dict[key] = (new_expr, [q_info])
        return new_expr

    def set_span(self, old_expr:tvm.relay.expr, new_expr:tvm.relay.expr):
        # set span for quantization info
        if old_expr.span is None:
            relay_type = type(new_expr)
            count = self.type_count.get(relay_type, 0)
            self.type_count[relay_type] = count + 1
            span_name = '{}_{}_{}'.format(self.__class__.__name__, new_expr.__class__.__name__, count)
            span = tvm.relay.Span(tvm.relay.SourceName(span_name), 0, 0, 0, 0)
        else:
            # if there already one, reuse it
            span = old_expr.span

        # currently we only set span for these three type of relay expr, add more if needed
        if isinstance(new_expr, tvm.relay.Call):
            new_args = [arg for arg in new_expr.args]
            new_expr = tvm.relay.Call(
                new_expr.op, new_args, new_expr.attrs, new_expr.type_args, span
            )
        elif isinstance(new_expr, tvm.relay.Var):
            new_expr = tvm.relay.var(new_expr.name_hint, shape=new_expr.type_annotation.shape, dtype='float32', span=span)
        elif isinstance(new_expr, tvm.relay.Constant):
            new_expr = tvm.relay.const(new_expr.data, span=span)
        return new_expr

    def set_output_names(self, old_expr:tvm.relay.expr, new_expr:tvm.relay.expr):
        # update output names for new expr
        new_key = get_key_from_expr(new_expr)
        if old_expr.span in self.span_output_names_dict and old_expr.span is not None:
            self.output_names_dict[new_key] = (new_expr, self.span_output_names_dict[old_expr.span])
            del self.span_output_names_dict[old_expr.span]

    def dequantize_constant_expr(self, constant_expr, constant_scale, constant_zero_point):
        constant_array = constant_expr.data.asnumpy()
        dequantized_constant_array = dequantize(constant_array, constant_scale, constant_zero_point)
        new_constant_expr = tvm.relay.const(dequantized_constant_array)
        new_constant_expr = self.populate_quantization_info(constant_expr, new_constant_expr,
                                                            constant_zero_point, constant_scale,
                                                            dtype=constant_expr.data.dtype)
        return new_constant_expr

    def dequantize_var_expr(self, var_expr, var_scale, var_zero_point, op_expr):
        self.dtype_dict[var_expr.name_hint] = 'float32'
        new_var_expr = tvm.relay.var(var_expr.name_hint, shape=var_expr.type_annotation.shape, dtype='float32')
        new_var_expr = self.populate_quantization_info(var_expr, new_var_expr, var_zero_point, var_scale,
                                                       dtype=op_expr.args[0].type_annotation.dtype)
        return new_var_expr


class DequantizeQnnAddPattern(DequantizeQnnPattern):
    """
    dequantize pattern from
    https://github.com/apache/tvm/blob/993a8ea094575f2823aebf2b8eb37e9f4ac44d7d/python/tvm/relay/frontend/tflite.py#L1361
    """
    def get_pattern(self):
        self._add = is_op('qnn.add')(wildcard(), wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant())
        self._clip = is_op('clip')(self._add)
        return self._add

    def visit_call(self, call):
        if self._clip.match(call):
            args = [self.visit(arg) for arg in call.args]
            return args[0]
        else:
            return super().visit_call(call)

    def dequantize_qnn_expr(self, add_expr, args):
        # Input expressions
        lhs_expr = args[0]
        rhs_expr = args[1]
        output_scale = args[6].data.asnumpy()
        output_zero_point = args[7].data.asnumpy()

        # Extract quantized attributes
        lhs_scale = args[2].data.asnumpy()
        lhs_zero_point = args[3].data.asnumpy()
        rhs_scale = args[4].data.asnumpy()
        rhs_zero_point = args[5].data.asnumpy()

        # Dequantize input expressions if they are Constant
        if isinstance(lhs_expr, tvm.relay.Constant):
            new_lhs_expr = self.dequantize_constant_expr(lhs_expr, lhs_scale, lhs_zero_point)
        elif isinstance(lhs_expr, tvm.relay.Var):
            new_lhs_expr = self.dequantize_var_expr(lhs_expr, lhs_scale, lhs_zero_point, add_expr)
        else:
            new_lhs_expr = lhs_expr

        if isinstance(rhs_expr, tvm.relay.Constant):
            new_rhs_expr = self.dequantize_constant_expr(rhs_expr, rhs_scale, rhs_zero_point)
        elif isinstance(rhs_expr, tvm.relay.Var):
            new_rhs_expr = self.dequantize_var_expr(rhs_expr, rhs_scale, rhs_zero_point, add_expr)
        else:
            new_rhs_expr = rhs_expr

        new_add_expr = tvm.relay.add(new_lhs_expr, new_rhs_expr)

        # Search data type from source nodes
        src_expr = lhs_expr
        while hasattr(src_expr, 'op') and src_expr.op.name not in ['qnn.requantize', 'qnn.quantize'] \
              and not isinstance(src_expr, tvm.relay.Constant) and not isinstance(src_expr, tvm.relay.Var):
            src_expr = src_expr.args[0]

        if src_expr.op.name in ['qnn.requantize', 'qnn.quantize']:
            out_dtype = src_expr.attrs['out_dtype']
        elif isinstance(src_expr, tvm.relay.Constant):
            out_dtype = src_expr.data.dtype
        elif isinstance(src_expr, tvm.relay.Var):
            out_dtype = src_expr.type_annotation.dtype
        else:
            raise ValueError("Data type is not avaliable in source expression")

        # after converting to tflite model, some activations are squashed(e.g., relu), the following clip try to recover origin range for floating model
        fmin = (float(tvm.tir.op.min_value(out_dtype).value)-output_zero_point)*output_scale
        fmax = (float(tvm.tir.op.max_value(out_dtype).value)-output_zero_point)*output_scale
        new_clip_expr = tvm.relay.clip(new_add_expr, fmin, fmax)

        # populate quantization info for new clip
        new_clip_expr = self.populate_quantization_info(add_expr, new_clip_expr, output_zero_point, output_scale, out_dtype)

        return new_clip_expr


class DequantizeQnnAvgPool2dPattern(DequantizeQnnPattern):
    """
    dequantize pattern from
    https://github.com/apache/tvm/blob/993a8ea094575f2823aebf2b8eb37e9f4ac44d7d/python/tvm/relay/frontend/tflite.py#L2531
    """
    def get_pattern(self):
        _cast = is_op('cast')(wildcard()).has_attr({'dtype': 'int32'})
        _avg_pool2d = is_op('nn.avg_pool2d')(_cast)
        _cast = is_op('cast')(_avg_pool2d)
        return _cast

    def dequantize_qnn_expr(self, cast_expr, args):
        avg_pool2d_expr = args[0]
        new_avg_pool2d_expr = tvm.relay.nn.avg_pool2d(avg_pool2d_expr.args[0].args[0], **avg_pool2d_expr.attrs)
        return new_avg_pool2d_expr


class DequantizeQnnConcatPattern(DequantizeQnnPattern):
    """
    dequantize pattern from
    https://github.com/apache/tvm/blob/993a8ea094575f2823aebf2b8eb37e9f4ac44d7d/python/tvm/relay/frontend/tflite.py#L1146
    """
    def get_pattern(self):
        _concatenate = is_op('qnn.concatenate')(wildcard(), wildcard(), wildcard(), wildcard(), wildcard())
        return _concatenate

    def dequantize_qnn_expr(self, concatenate_expr, args):
        # rewrite qnn.concatenate to concatenate
        # args[0] is relay expr to concatenate
        # refer https://github.com/apache/tvm/blob/993a8ea094575f2823aebf2b8eb37e9f4ac44d7d/python/tvm/relay/qnn/op/qnn.py#L328
        tuple = tvm.relay.Tuple([arg for arg in args[0].fields])
        new_concatenate_expr = tvm.relay.concatenate(tuple, **concatenate_expr.attrs)
        return new_concatenate_expr


class DequantizeQnnConvPattern(DequantizeQnnPattern):
    """
    dequantize pattern from
    https://github.com/apache/tvm/blob/993a8ea094575f2823aebf2b8eb37e9f4ac44d7d/python/tvm/relay/frontend/tflite.py#L2200
    """
    def get_pattern(self):
        _conv2d = is_op('qnn.conv2d')(wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant())
        _bias_add = is_op('nn.bias_add')(_conv2d, is_constant())
        self._requantize = is_op('qnn.requantize')(_bias_add, is_constant(), is_constant(), is_constant(), is_constant())
        self._clip = is_op('clip')(self._requantize)
        return self._requantize

    def visit_call(self, call):
        if self._clip.match(call):
            args = [self.visit(arg) for arg in call.args]
            return args[0]
        else:
            return super().visit_call(call)

    def dequantize_qnn_expr(self, requantize_expr, args):

        # requantize
        bias_add_expr = args[0]
        bias_scale = args[1].data.asnumpy()
        bias_zero_point = args[2].data.asnumpy()
        output_scale = args[3].data.asnumpy()
        output_zero_point = args[4].data.asnumpy()

        # bias add
        conv2d_expr = bias_add_expr.args[0]
        bias_expr = bias_add_expr.args[1]
        bias = bias_expr.data.asnumpy()

        # conv
        data_expr = conv2d_expr.args[0]
        kernel_expr = conv2d_expr.args[1]
        kernel = kernel_expr.data.asnumpy()
        kernel_zero_point = conv2d_expr.args[3].data.asnumpy()
        kernel_scale = conv2d_expr.args[5].data.asnumpy()
        input_zero_point = conv2d_expr.args[2].data.asnumpy()
        input_scale = conv2d_expr.args[4].data.asnumpy()

        # Expand dimension for depthwise convolution so they can broadcast later
        if conv2d_expr.attrs['kernel_layout'] == 'HWOI':
            kernel_scale = np.expand_dims(kernel_scale, axis=-1)

        # dequantize and populate quantization info for kernel and bias
        new_kernel_expr = self.dequantize_constant_expr(kernel_expr, kernel_scale, kernel_zero_point)
        new_bias_expr = self.dequantize_constant_expr(bias_expr, bias_scale, bias_zero_point)

        # if data is relay.Var, change its dtype
        if isinstance(data_expr, tvm.relay.Var):
            new_data_expr = self.dequantize_var_expr(data_expr, input_scale, input_zero_point, conv2d_expr)
        else:
            new_data_expr = data_expr

        # create dequantized relay expr
        conv2d_attrs = dict(conv2d_expr.attrs)
        conv2d_attrs['out_dtype'] = ''
        new_conv2d_expr = tvm.relay.nn.conv2d(new_data_expr, new_kernel_expr, **conv2d_attrs)
        new_bias_add_expr = tvm.relay.nn.bias_add(new_conv2d_expr, new_bias_expr, **bias_add_expr.attrs)

        # after converting to tflite model, some activations are squashed(e.g., relu), the following clip try to recover origin range for floating model
        fmin = (float(tvm.tir.op.min_value(requantize_expr.attrs['out_dtype']).value)-output_zero_point)*output_scale
        fmax = (float(tvm.tir.op.max_value(requantize_expr.attrs['out_dtype']).value)-output_zero_point)*output_scale
        new_clip_expr = tvm.relay.clip(new_bias_add_expr, fmin, fmax)

        # populate quantization info for new clip
        new_clip_expr = self.populate_quantization_info(requantize_expr, new_clip_expr, output_zero_point, output_scale, dtype=requantize_expr.attrs['out_dtype'])

        return new_clip_expr


class DequantizeQnnPadPattern(DequantizeQnnPattern):
    """
    dequantize pattern from
    https://github.com/apache/tvm/blob/993a8ea094575f2823aebf2b8eb37e9f4ac44d7d/python/tvm/relay/frontend/tflite.py#L2616
    """
    def get_pattern(self):
        self._add = is_op('qnn.add')(wildcard(), wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant())

        # use pattern from conv pattern
        conv_pattern = DequantizeQnnConvPattern({}, {}, {})
        self._conv_pattern = conv_pattern.get_pattern()
        self._conv_requantize = conv_pattern._requantize
        self._conv_clip = conv_pattern._clip

        _in_expr = self._add | self._conv_pattern | self._conv_clip
        _pad = is_op('nn.pad')(_in_expr, wildcard())
        return _pad

    def dequantize_qnn_expr(self, pad_expr, args):
        in_expr = args[0]
        pad_value = args[1].data.asnumpy()

        if self._add.match(in_expr):
            output_offset = in_expr.args[7].data.asnumpy()
        elif self._conv_requantize.match(in_expr):
            output_offset = in_expr.args[4].data.asnumpy()
        elif self._conv_clip.match(in_expr):
            output_offset = in_expr.args[0].args[4].data.asnumpy()
        else:
            # if there is no pattern matched, should not enter here
            raise ValueError()

        pad_value_expr = tvm.relay.const(pad_value-output_offset)
        return tvm.relay.nn.pad(in_expr, pad_value=pad_value_expr, **pad_expr.attrs)


class DequantizeQnnReluPattern(DequantizeQnnPattern):
    """
    dequantize pattern from
    https://github.com/apache/tvm/blob/993a8ea094575f2823aebf2b8eb37e9f4ac44d7d/python/tvm/relay/frontend/tflite.py#L910
    """
    def get_pattern(self):
        _clip = is_op('clip')(wildcard())
        _requantize = is_op('qnn.requantize')(_clip, is_constant(), is_constant(), is_constant(), is_constant())
        return _requantize

    def dequantize_qnn_expr(self, requantize_expr, args):
        clip = args[0]
        output_scale = args[3].data.asnumpy()
        output_zero_point = args[4].data.asnumpy()
        fmin = (float(tvm.tir.op.min_value(requantize_expr.attrs['out_dtype']).value)-output_zero_point)*output_scale
        fmax = (float(tvm.tir.op.max_value(requantize_expr.attrs['out_dtype']).value)-output_zero_point)*output_scale
        new_clip_expr = tvm.relay.clip(clip.args[0], fmin, fmax)
        # new_clip_expr = self.populate_quantization_info(requantize_expr, new_clip_expr, output_zero_point, output_scale, dtype=requantize_expr.attrs['out_dtype'])
        return new_clip_expr


class DequantizeQnnConv2dTransposePattern(DequantizeQnnPattern):
    """
    dequantize pattern from
    https://github.com/apache/tvm/blob/993a8ea094575f2823aebf2b8eb37e9f4ac44d7d/python/tvm/relay/frontend/tflite.py#L3130
    """
    def get_pattern(self):
        _conv2d_transpose = is_op('qnn.conv2d_transpose')(wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant())
        _bias_add = is_op('nn.bias_add')(_conv2d_transpose, is_constant())
        _requantize = is_op('qnn.requantize')(_bias_add, is_constant(), is_constant(), is_constant(), is_constant())

        return _requantize

    def dequantize_qnn_expr(self, requantize_expr, args):
        # requantize
        bias_add_expr = args[0]
        bias_scale = args[1].data.asnumpy()
        bias_zero_point = args[2].data.asnumpy()
        output_scale = args[3].data.asnumpy()
        output_zero_point = args[4].data.asnumpy()

        # bias add
        conv2d_transpose_expr = bias_add_expr.args[0]
        bias_expr = bias_add_expr.args[1]

        # conv
        data_expr = conv2d_transpose_expr.args[0]
        kernel_expr = conv2d_transpose_expr.args[1]
        input_zero_point = conv2d_transpose_expr.args[2].data.asnumpy()
        kernel_zero_point = conv2d_transpose_expr.args[3].data.asnumpy()
        input_scale = conv2d_transpose_expr.args[4].data.asnumpy()
        kernel_scale = conv2d_transpose_expr.args[5].data.asnumpy()

        if conv2d_transpose_expr.attrs['kernel_layout'] == 'OIHW':
            kernel_scale = np.expand_dims(kernel_scale, axis=1)
            kernel_scale = np.expand_dims(kernel_scale, axis=2)

        # dequantize and populate quantization info for kernel and bias
        new_kernel_expr = self.dequantize_constant_expr(kernel_expr, kernel_scale, kernel_zero_point)
        new_bias_expr = self.dequantize_constant_expr(bias_expr, bias_scale, bias_zero_point)

        # if data is input of model, change its dtype
        if isinstance(data_expr, tvm.relay.Var):
            new_kernel_expr = self.dequantize_var_expr(data_expr, input_scale, input_zero_point, conv2d_transpose_expr)
        else:
            new_data_expr = data_expr

        # create dequantized relay expr
        conv2d_transpose_attrs = dict(conv2d_transpose_expr.attrs)
        conv2d_transpose_attrs['out_dtype'] = 'float32'
        new_conv2d_transpose_expr = tvm.relay.op.nn.conv2d_transpose(new_data_expr, new_kernel_expr, **conv2d_transpose_attrs)
        new_bias_add_expr = tvm.relay.nn.bias_add(new_conv2d_transpose_expr, new_bias_expr, **bias_add_expr.attrs)

        # after converting to tflite model, some activations are squashed(e.g., relu), the following clip try to recover origin range for floating model
        fmin = (float(tvm.tir.op.min_value(requantize_expr.attrs['out_dtype']).value)-output_zero_point)*output_scale
        fmax = (float(tvm.tir.op.max_value(requantize_expr.attrs['out_dtype']).value)-output_zero_point)*output_scale
        new_clip_expr = tvm.relay.clip(new_bias_add_expr, fmin, fmax)

        # populate quantization info for new clip
        new_clip_expr = self.populate_quantization_info(requantize_expr, new_clip_expr, output_zero_point, output_scale, dtype=requantize_expr.attrs['out_dtype'])

        return new_clip_expr


class DequantizeQnnDensePattern(DequantizeQnnPattern):
    """
    dequantize pattern from
    https://github.com/apache/tvm/blob/993a8ea094575f2823aebf2b8eb37e9f4ac44d7d/python/tvm/relay/frontend/tflite.py#L1888
    """
    def get_pattern(self):
        _dense = is_op('qnn.dense')(wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant())
        _bias_add = is_op('nn.bias_add')(_dense, is_constant())
        _requantize = is_op('qnn.requantize')(_bias_add, is_constant(), is_constant(), is_constant(), is_constant())
        return _requantize

    def dequantize_qnn_expr(self, requantize_expr, args):

        # requantize
        bias_add_expr = args[0]
        bias_scale = args[1].data.asnumpy()
        bias_zero_point = args[2].data.asnumpy()
        output_scale = args[3].data.asnumpy()
        output_zero_point = args[4].data.asnumpy()

        # bias add
        dense_expr = bias_add_expr.args[0]
        bias_expr = bias_add_expr.args[1]

        # conv
        data_expr = dense_expr.args[0]
        weight_expr = dense_expr.args[1]
        weight_zero_point = dense_expr.args[3].data.asnumpy()
        weight_scale = dense_expr.args[5].data.asnumpy()
        input_zero_point = dense_expr.args[2].data.asnumpy()
        input_scale = dense_expr.args[4].data.asnumpy()

        # dequantize and populate quantization info for weight and bias
        new_weight_expr = self.dequantize_constant_expr(weight_expr, weight_scale, weight_zero_point)
        new_bias_expr = self.dequantize_constant_expr(bias_expr, bias_scale, bias_zero_point)

        # if data is relay.Var, change its dtype
        if isinstance(data_expr, tvm.relay.Var):
            new_data_expr = self.dequantize_var_expr(data_expr, input_scale, input_zero_point, dense_expr)
        else:
            new_data_expr = data_expr

        # create dequantized relay expr
        dense_attrs = dict(dense_expr.attrs)
        dense_attrs['out_dtype'] = ''
        new_dense_expr = tvm.relay.nn.dense(new_data_expr, new_weight_expr, **dense_attrs)
        new_bias_add_expr = tvm.relay.nn.bias_add(new_dense_expr, new_bias_expr, **bias_add_expr.attrs)

        # after converting to tflite model, some activations are squashed(e.g., relu), the following clip try to recover origin range for floating model
        fmin = (float(tvm.tir.op.min_value(requantize_expr.attrs['out_dtype']).value)-output_zero_point)*output_scale
        fmax = (float(tvm.tir.op.max_value(requantize_expr.attrs['out_dtype']).value)-output_zero_point)*output_scale
        new_clip_expr = tvm.relay.clip(new_bias_add_expr, fmin, fmax)

        # populate quantization info for new clip
        new_clip_expr = self.populate_quantization_info(requantize_expr, new_clip_expr, output_zero_point, output_scale, dtype=requantize_expr.attrs['out_dtype'])

        return new_clip_expr


class DequantizeQnnDequantizePattern(DequantizeQnnPattern):
    """
    remove rest of dequantize op
    """
    def get_pattern(self):
        self._dequantize = is_op('qnn.dequantize')(wildcard(), is_constant(), is_constant())
        return self._dequantize

    def dequantize_qnn_expr(self, dequantize_expr, args):
        input_scale = args[1].data.asnumpy()
        input_zero_point = args[2].data.asnumpy()
        new_expr = self.populate_quantization_info(args[0], args[0], input_scale, input_zero_point)
        return new_expr


class DequantizeQnnResizePattern(DequantizeQnnPattern):
    """
    dequantize pattern from
    https://github.com/apache/tvm/blob/993a8ea094575f2823aebf2b8eb37e9f4ac44d7d/python/tvm/relay/frontend/tflite.py#L705
    """
    def get_pattern(self):
        _dequantize = is_op('qnn.dequantize')(wildcard(), is_constant(), is_constant())
        _resize = is_op('image.resize')(_dequantize)
        _quantize = is_op('qnn.quantize')(_resize, is_constant(), is_constant())
        return _quantize

    def dequantize_qnn_expr(self, resize_expr, args):
        resize = args[0]
        dequantize = resize.args[0]
        new_resize = tvm.relay.image.resize(dequantize.args[0], **resize.attrs)
        return new_resize


class DequantizeQnnSigmomidPattern(DequantizeQnnPattern):
    """
    dequantize pattern from
    https://github.com/apache/tvm/blob/993a8ea094575f2823aebf2b8eb37e9f4ac44d7d/python/tvm/relay/frontend/tflite.py#L810
    """
    def get_pattern(self):
        _dequantize = is_op('qnn.dequantize')(wildcard(), is_constant(), is_constant())
        _sigmoid = is_op('sigmoid')(_dequantize)
        _quantize = is_op('qnn.quantize')(_sigmoid, is_constant(), is_constant())
        return _quantize

    def dequantize_qnn_expr(self, quantize_expr, args):
        sigmoid = args[0]
        dequantize = sigmoid.args[0]
        new_sigmoid = tvm.relay.sigmoid(dequantize.args[0])
        return new_sigmoid


class DequantizeQnnMulPattern(DequantizeQnnPattern):
    """
    dequantize pattern from
    https://github.com/apache/tvm/blob/993a8ea094575f2823aebf2b8eb37e9f4ac44d7d/python/tvm/relay/frontend/tflite.py#L1389
    """
    def get_pattern(self):
        _mul = is_op('qnn.mul')(wildcard(), wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant())
        return _mul

    def dequantize_qnn_expr(self, mul_expr, args):
        # Input expressions
        lhs_expr = args[0]
        rhs_expr = args[1]

        # Extract quantized attributes
        lhs_scale = args[2].data.asnumpy()
        lhs_zero_point = args[3].data.asnumpy()
        rhs_scale = args[4].data.asnumpy()
        rhs_zero_point = args[5].data.asnumpy()

        # Dequantize input expressions if they are Constant or Var
        if isinstance(lhs_expr, tvm.relay.Constant):
            new_lhs_expr = self.dequantize_constant_expr(lhs_expr, lhs_scale, lhs_zero_point)
        elif isinstance(lhs_expr, tvm.relay.Var):
            new_lhs_expr = self.dequantize_var_expr(lhs_expr, lhs_scale, lhs_zero_point, mul_expr)
        else:
            new_lhs_expr = lhs_expr

        if isinstance(rhs_expr, tvm.relay.Constant):
            new_rhs_expr = self.dequantize_constant_expr(rhs_expr, rhs_scale, rhs_zero_point)
        elif isinstance(lhs_expr, tvm.relay.Var):
            new_lhs_expr = self.dequantize_var_expr(lhs_expr, lhs_scale, lhs_zero_point, mul_expr)
        else:
            new_rhs_expr = rhs_expr

        new_mul = tvm.relay.multiply(new_lhs_expr, new_rhs_expr)

        return new_mul


class DequantizeQnnQuantizePattern(DequantizeQnnPattern):
    """
    remove rest of quantize/requantize op
    """
    def get_pattern(self):
        self._quantize = is_op('qnn.quantize')(wildcard(), is_constant(), is_constant())
        self._requantize = is_op('qnn.requantize')(wildcard(), is_constant(), is_constant(), is_constant(), is_constant())
        return self._quantize | self._requantize

    def dequantize_qnn_expr(self, expr, args):
        output_scale = args[1].data.asnumpy()
        output_zero_point = args[2].data.asnumpy()
        new_expr = self.populate_quantization_info(args[0], args[0], output_zero_point, output_scale, dtype=expr.attrs['out_dtype'])
        return new_expr


@tvm.ir.transform.module_pass(opt_level=3)
class DequantizePass:

    def __init__(self, dtype_dict, expr_to_quantization_params_dict, output_names_dict):
        self.dtype_dict = dtype_dict
        self.expr_to_quantization_params_dict = expr_to_quantization_params_dict
        self.output_names_dict = output_names_dict

    # This function can define a pass.
    def transform_module(self, mod, ctx):
        # Pad/Add pattern must before others to grab output offset/out_dtype
        mod.update_func(mod.get_global_var("main"), DequantizeQnnPadPattern(self.dtype_dict, self.expr_to_quantization_params_dict, self.output_names_dict).visit(mod['main']))
        mod.update_func(mod.get_global_var("main"), DequantizeQnnAddPattern(self.dtype_dict, self.expr_to_quantization_params_dict, self.output_names_dict).visit(mod['main']))

        mod.update_func(mod.get_global_var("main"), DequantizeQnnAvgPool2dPattern(self.dtype_dict, self.expr_to_quantization_params_dict, self.output_names_dict).visit(mod['main']))
        mod.update_func(mod.get_global_var("main"), DequantizeQnnConcatPattern(self.dtype_dict, self.expr_to_quantization_params_dict, self.output_names_dict).visit(mod['main']))
        mod.update_func(mod.get_global_var("main"), DequantizeQnnConvPattern(self.dtype_dict, self.expr_to_quantization_params_dict, self.output_names_dict).visit(mod['main']))
        mod.update_func(mod.get_global_var("main"), DequantizeQnnConv2dTransposePattern(self.dtype_dict, self.expr_to_quantization_params_dict, self.output_names_dict).visit(mod['main']))
        mod.update_func(mod.get_global_var("main"), DequantizeQnnDensePattern(self.dtype_dict, self.expr_to_quantization_params_dict, self.output_names_dict).visit(mod['main']))
        mod.update_func(mod.get_global_var("main"), DequantizeQnnMulPattern(self.dtype_dict, self.expr_to_quantization_params_dict, self.output_names_dict).visit(mod['main']))
        mod.update_func(mod.get_global_var("main"), DequantizeQnnReluPattern(self.dtype_dict, self.expr_to_quantization_params_dict, self.output_names_dict).visit(mod['main']))
        mod.update_func(mod.get_global_var("main"), DequantizeQnnResizePattern(self.dtype_dict, self.expr_to_quantization_params_dict, self.output_names_dict).visit(mod['main']))
        mod.update_func(mod.get_global_var("main"), DequantizeQnnSigmomidPattern(self.dtype_dict, self.expr_to_quantization_params_dict, self.output_names_dict).visit(mod['main']))

        # remove rest of quantize/requantize/dequantize op
        mod.update_func(mod.get_global_var("main"), DequantizeQnnQuantizePattern(self.dtype_dict, self.expr_to_quantization_params_dict, self.output_names_dict).visit(mod['main']))
        mod.update_func(mod.get_global_var("main"), DequantizeQnnDequantizePattern(self.dtype_dict, self.expr_to_quantization_params_dict, self.output_names_dict).visit(mod['main']))
        return mod
