# ==============================================================================
#
#  Copyright (c) 2021-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from abc import ABC
from qti.aisw.converters.common import ir_graph
from qti.aisw.converters.common.utils.converter_utils import log_debug2, log_debug3, converter_type, log_assert
from qti.aisw.converters.common.converter_ir.op_graph import IROpGraph
from qti.aisw.converters.common.converter_ir.op_adapter import (
    ArgOp,
    ConstantOp,
    CumSumOp,
    ElementwiseBinaryOp,
    ElementwiseTernaryOp,
    ElementwiseUnaryOp,
    NeuronOp,
    ReduceOp,
    TopKOp
)

from qti.aisw.converters.relay.translations.relay_translations import RelayTranslationBase
from qti.aisw.converters.relay.translations import RelayTranslations

import tvm
from tvm import relay

import numpy as np


# ------------------------------------------------------------------------------
#   ArgMax
# ------------------------------------------------------------------------------
class RelayArgMaxTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayArgMaxTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        axis = relay_expr.attrs.axis
        keepdims = relay_expr.attrs.keepdims
        exclude = relay_expr.attrs.exclude

        if isinstance(axis, tvm.ir.container.Array):
            axis = [int(i) for i in axis]
        elif isinstance(axis, tvm.tir.expr.IntImm):
            axis = int(axis)
        else:
            TypeError("Argmax axis is of Unsupported datatype {}".format(type(axis)))

        if keepdims:
            keepdims = True
        else:
            keepdims = False

        attr_dict["axis"] = axis
        attr_dict["keepdims"] = keepdims
        attr_dict["exclude"] = exclude

        log_debug3("\taxis {} {}", type(axis), axis)
        log_debug3("\tkeepdims {} {}", type(keepdims), keepdims)
        log_debug3("\texclude {} {}", type(exclude), exclude)

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, ArgOp.TRANSLATION_KEY, ArgOp.LEGACY_TRANSLATION_KEY)
        input_shapes = converter_context.get_input_shapes(relay_expr)

        axis = attr_dict["axis"]
        keepdims = attr_dict["keepdims"]
        exclude = attr_dict["exclude"]

        if exclude:
            axis = [i for i in range(len(input_shapes[0])) if i not in axis]

        if isinstance(axis, list):
            if len(axis) > 1:
                raise ValueError("Argmax axis only supported as scalar, got list {}".format(axis))
            axis = axis[0]

        ir_op = ArgOp(op_name,
                      arg_type = ir_graph.QNN_OP_ARGMAX,
                      axis=axis,
                      keep_dims=keepdims)

        return ir_op


RelayTranslations.register_translation(RelayArgMaxTranslation(),
                                       converter_type('argmax', 'relay'))


# ------------------------------------------------------------------------------
#   CumulativeSum
# ------------------------------------------------------------------------------
class RelayCumulativeSumTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayCumulativeSumTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        attrs = relay_expr.attrs
        attr_dict['axis'] = attrs.axis
        if attr_dict['axis'] is None:
            attr_dict['axis'] = 0
        attr_dict['exclusive'] = attrs.exclusive
        if attr_dict['exclusive'] is None:
            attr_dict['exclusive'] = False

        log_debug3("\axis {}", attr_dict['axis'])
        log_debug3("\exclusive {}", attr_dict['exclusive'])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, CumSumOp.TRANSLATION_KEY, CumSumOp.LEGACY_TRANSLATION_KEY)
        axis_attrs = attr_dict.get('axis')
        if isinstance(axis_attrs, tvm.tir.IntImm):
            axis_attrs = int(axis_attrs)
        ir_op = CumSumOp(name=op_name,
                         axis=axis_attrs,
                         exclusive=attr_dict.get('exclusive'))
        return ir_op


RelayTranslations.register_translation(RelayCumulativeSumTranslation(),
                                       converter_type('cumsum', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseTernary
# ------------------------------------------------------------------------------
class RelayElementwiseTernaryBaseTranslation(RelayTranslationBase):
    def __init__(self, ir_op_type):
        super(RelayElementwiseTernaryBaseTranslation, self).__init__()
        self.ir_op_type = ir_op_type

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        op_name = converter_context.get_op_name(relay_expr, self.ir_op_type,
                                                ElementwiseTernaryOp.ir_to_legacy_type[self.ir_op_type])

        input_0_name = input_names[0]
        input_1_name = input_names[1]
        input_2_name = input_names[2]

        if not quir_graph.has_buffer(input_0_name):
            # Op has 1st input as Constant tensor
            input_0_tensor = relay_params[input_0_name]
            log_debug3("\tconst tensor type {} shape {}", type(input_0_tensor), input_0_tensor.shape)
            if isinstance(input_0_tensor, tvm.runtime.ndarray.NDArray) or \
                    isinstance(input_0_tensor, tvm.runtime.NDArray):
                input_0_tensor = input_0_tensor.asnumpy()

            if not input_0_tensor.shape:
                input_0_tensor = np.reshape(input_0_tensor.data, (1,))
            log_debug3("\tconst tensor after type {} shape {}", type(input_0_tensor), input_0_tensor.shape)

            constant_output_name = op_name + "_const_0"
            input_names[0] = constant_output_name
            log_debug2("Adding Constant Op name:{} with output:{}", constant_output_name, constant_output_name)
            quir_graph.add(ConstantOp(constant_output_name, input_0_tensor), [], [constant_output_name])

        if not quir_graph.has_buffer(input_1_name):
            # Op has 2nd input as Constant tensor
            input_1_tensor = relay_params[input_names[1]]
            log_debug3("\tconst tensor type {} shape {}", type(input_1_tensor), input_1_tensor.shape)
            if isinstance(input_1_tensor, tvm.runtime.ndarray.NDArray) or \
                    isinstance(input_1_tensor, tvm.runtime.NDArray):
                input_1_tensor = input_1_tensor.asnumpy()

            if not input_1_tensor.shape:
                input_1_tensor = np.reshape(input_1_tensor.data, (1,))
            log_debug3("\tconst tensor after type {} shape {}", type(input_1_tensor), input_1_tensor.shape)

            constant_output_name = op_name + "_const_1"
            input_names[1] = constant_output_name
            log_debug2("Adding Constant Op name:{} with output:{}", constant_output_name, constant_output_name)
            quir_graph.add(ConstantOp(constant_output_name, input_1_tensor), [], [constant_output_name])

        if not quir_graph.has_buffer(input_2_name):
            # Op has 2nd input as Constant tensor
            input_2_tensor = relay_params[input_names[2]]
            log_debug3("\tconst tensor type {} shape {}", type(input_2_tensor), input_2_tensor.shape)
            if isinstance(input_2_tensor, tvm.runtime.ndarray.NDArray) or \
                    isinstance(input_2_tensor, tvm.runtime.NDArray):
                input_2_tensor = input_2_tensor.asnumpy()

            if not input_2_tensor.shape:
                input_2_tensor = np.reshape(input_2_tensor.data, (1,))
            log_debug3("\tconst tensor after type {} shape {}", type(input_2_tensor), input_2_tensor.shape)

            constant_output_name = op_name + "_const_2"
            input_names[2] = constant_output_name
            log_debug2("Adding Constant Op name:{} with output:{}", constant_output_name, constant_output_name)
            quir_graph.add(ConstantOp(constant_output_name, input_2_tensor), [], [constant_output_name])

        ir_op = ElementwiseTernaryOp(op_name, eltwise_type=self.ir_op_type)

        return ir_op


# ------------------------------------------------------------------------------
#   ElementwiseBinaryBase
# ------------------------------------------------------------------------------
class RelayElementwiseBinaryBaseTranslation(RelayTranslationBase):
    def __init__(self, ir_op_type):
        super(RelayElementwiseBinaryBaseTranslation, self).__init__()
        self.ir_op_type = ir_op_type

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        op_name = converter_context.get_op_name(relay_expr, self.ir_op_type,
                                                ElementwiseBinaryOp.ir_to_legacy_type[self.ir_op_type])

        input_0_name = input_names[0]
        input_1_name = input_names[1]

        if not quir_graph.has_buffer(input_0_name):
            # Op has 1st input as Constant tensor
            input_0_tensor = relay_params[input_0_name]
            log_debug3("\tconst tensor type {} shape {}", type(input_0_tensor), input_0_tensor.shape)
            if isinstance(input_0_tensor, tvm.runtime.ndarray.NDArray) or \
                    isinstance(input_0_tensor, tvm.runtime.NDArray):
                input_0_tensor = input_0_tensor.asnumpy()

            if not input_0_tensor.shape:
                input_0_tensor = np.reshape(input_0_tensor.data, (1,))
            log_debug3("\tconst tensor after type {} shape {}", type(input_0_tensor), input_0_tensor.shape)

            constant_output_name = op_name + "_const_0"
            input_names[0] = constant_output_name
            log_debug2("Adding Constant Op name:{} with output:{}", constant_output_name, constant_output_name)
            quir_graph.add(ConstantOp(constant_output_name, input_0_tensor), [], [constant_output_name])

        if not quir_graph.has_buffer(input_1_name):
            # Op has 2nd input as Constant tensor
            input_1_tensor = relay_params[input_names[1]]
            log_debug3("\tconst tensor type {} shape {}", type(input_1_tensor), input_1_tensor.shape)
            if isinstance(input_1_tensor, tvm.runtime.ndarray.NDArray) or\
                    isinstance(input_1_tensor, tvm.runtime.NDArray):
                input_1_tensor = input_1_tensor.asnumpy()

            if not input_1_tensor.shape:
                input_1_tensor = np.reshape(input_1_tensor.data, (1,))
            log_debug3("\tconst tensor after type {} shape {}", type(input_1_tensor), input_1_tensor.shape)

            constant_output_name = op_name + "_const_1"
            input_names[1] = constant_output_name
            log_debug2("Adding Constant Op name:{} with output:{}", constant_output_name, constant_output_name)
            quir_graph.add(ConstantOp(constant_output_name, input_1_tensor), [], [constant_output_name])

        ir_op = ElementwiseBinaryOp(op_name, eltwise_type=self.ir_op_type)

        return ir_op


# ------------------------------------------------------------------------------
#   ElementwiseDiv
# ------------------------------------------------------------------------------
class RelayElementwiseDivTranslation(RelayElementwiseBinaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseDivTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_DIVIDE)

RelayTranslations.register_translation(RelayElementwiseDivTranslation(),
                                       converter_type('divide', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseEqual
# ------------------------------------------------------------------------------
class RelayElementwiseEqualTranslation(RelayElementwiseBinaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseEqualTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_EQUAL)

RelayTranslations.register_translation(RelayElementwiseEqualTranslation(),
                                       converter_type('equal', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseFloorDiv
# ------------------------------------------------------------------------------
class RelayElementwiseFloorDivTranslation(RelayElementwiseBinaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseFloorDivTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_FLOOR_DIV)

RelayTranslations.register_translation(RelayElementwiseFloorDivTranslation(),
                                       converter_type('floor_divide', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseGreater
# ------------------------------------------------------------------------------
class RelayElementwiseGreaterTranslation(RelayElementwiseBinaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseGreaterTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_GREATER)

RelayTranslations.register_translation(RelayElementwiseGreaterTranslation(),
                                       converter_type('greater', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseLess
# ------------------------------------------------------------------------------
class RelayElementwiseLessTranslation(RelayElementwiseBinaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseLessTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_LESS)

RelayTranslations.register_translation(RelayElementwiseLessTranslation(),
                                       converter_type('less', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseSum
# ------------------------------------------------------------------------------
class RelayElementwiseSumTranslation(RelayElementwiseBinaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseSumTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_ADD)

RelayTranslations.register_translation(RelayElementwiseSumTranslation(),
                                       converter_type('add', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseSub
# ------------------------------------------------------------------------------
class RelayElementwiseSubTranslation(RelayElementwiseBinaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseSubTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_SUBTRACT)

RelayTranslations.register_translation(RelayElementwiseSubTranslation(),
                                       converter_type('subtract', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseMax
# ------------------------------------------------------------------------------
class RelayElementwiseMaxTranslation(RelayElementwiseBinaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseMaxTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_MAXIMUM)

RelayTranslations.register_translation(RelayElementwiseMaxTranslation(),
                                       converter_type('maximum', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseMin
# ------------------------------------------------------------------------------
class RelayElementwiseMinTranslation(RelayElementwiseBinaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseMinTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_MINIMUM)

RelayTranslations.register_translation(RelayElementwiseMinTranslation(),
                                       converter_type('minimum', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwisePower
# ------------------------------------------------------------------------------
class RelayElementwisePowerTranslation(RelayElementwiseBinaryBaseTranslation):
    def __init__(self):
        super(RelayElementwisePowerTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_POWER)

RelayTranslations.register_translation(RelayElementwisePowerTranslation(),
                                       converter_type('power', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseProd
# ------------------------------------------------------------------------------
class RelayElementwiseProdTranslation(RelayElementwiseBinaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseProdTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_MULTIPLY)

RelayTranslations.register_translation(RelayElementwiseProdTranslation(),
                                       converter_type('multiply', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseSelect
# ------------------------------------------------------------------------------
class RelayElementwiseSelectTranslation(RelayElementwiseTernaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseSelectTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_SELECT)

RelayTranslations.register_translation(RelayElementwiseSelectTranslation(),
                                       converter_type('where', 'relay'))


# ------------------------------------------------------------------------------
#   Tanh
# ------------------------------------------------------------------------------
class RelayTanhTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayTanhTranslation, self).__init__()

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, NeuronOp.TRANSLATION_KEY,
                                                NeuronOp.LEGACY_TRANSLATION_KEY)

        ir_op = NeuronOp(op_name,
                         ir_graph.QNN_OP_TANH,
                         alpha=1.0,
                         beta=1.0)
        return ir_op


RelayTranslations.register_translation(RelayTanhTranslation(),
                                       converter_type('tanh', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseUnaryBase
# ------------------------------------------------------------------------------
class RelayElementwiseUnaryBaseTranslation(RelayTranslationBase):
    def __init__(self, eltwise_type):
        super(RelayElementwiseUnaryBaseTranslation, self).__init__()
        self.eltwise_type = eltwise_type

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):

        op_name = converter_context.get_op_name(relay_expr,
                                                ElementwiseUnaryOp.ir_to_legacy_type[self.eltwise_type],
                                                ElementwiseUnaryOp.ir_to_legacy_type[self.eltwise_type])
        ir_op = ElementwiseUnaryOp(op_name, eltwise_type=self.eltwise_type)

        return ir_op


# ------------------------------------------------------------------------------
#   ElementwiseUnaryAbs
# ------------------------------------------------------------------------------
class RelayElementwiseUnaryAbsTranslation(RelayElementwiseUnaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseUnaryAbsTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_ABS)

RelayTranslations.register_translation(RelayElementwiseUnaryAbsTranslation(),
                                       converter_type('abs', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseUnaryExp
# ------------------------------------------------------------------------------
class RelayElementwiseUnaryExpTranslation(RelayElementwiseUnaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseUnaryExpTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_EXP)

RelayTranslations.register_translation(RelayElementwiseUnaryExpTranslation(),
                                       converter_type('exp', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseUnaryFloor
# ------------------------------------------------------------------------------
class RelayElementwiseUnaryFloorTranslation(RelayElementwiseUnaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseUnaryFloorTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_FLOOR)

RelayTranslations.register_translation(RelayElementwiseUnaryFloorTranslation(),
                                       converter_type('floor', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseUnaryLog
# ------------------------------------------------------------------------------
class RelayElementwiseUnaryLogTranslation(RelayElementwiseUnaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseUnaryLogTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_LOG)

RelayTranslations.register_translation(RelayElementwiseUnaryLogTranslation(),
                                       converter_type('log', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseUnaryRsqrt
# ------------------------------------------------------------------------------
class RelayElementwiseUnaryRsqrtTranslation(RelayElementwiseUnaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseUnaryRsqrtTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_RSQRT)

RelayTranslations.register_translation(RelayElementwiseUnaryRsqrtTranslation(),
                                       converter_type('rsqrt', 'relay'))


# ------------------------------------------------------------------------------
#   ElementwiseUnarySqrt
# ------------------------------------------------------------------------------
class RelayElementwiseUnarySqrtTranslation(RelayElementwiseUnaryBaseTranslation):
    def __init__(self):
        super(RelayElementwiseUnarySqrtTranslation, self).__init__(ir_graph.QNN_OP_ELEMENT_WISE_SQUARE_ROOT)

RelayTranslations.register_translation(RelayElementwiseUnarySqrtTranslation(),
                                       converter_type('sqrt', 'relay'))


# ------------------------------------------------------------------------------
#   ReduceOp
# ------------------------------------------------------------------------------
class RelayReduceBaseTranslation(RelayTranslationBase, ABC):
    def __init__(self, reduce_type):
        super(RelayReduceBaseTranslation, self).__init__()
        self.reduce_type = reduce_type

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        attr_dict['keep_dims'] = relay_expr.attrs.keepdims
        axis_attrs = relay_expr.attrs.axis
        if isinstance(axis_attrs, tvm.ir.container.Array):
            attr_dict['axis'] = [int(i) for i in axis_attrs]
        elif isinstance(axis_attrs, tvm.tir.IntImm):
            attr_dict['axis'] = [int(axis_attrs)]
        else:
            attr_dict['axis'] = list()
        attr_dict['exclude'] = relay_expr.attrs.exclude

        log_debug3("\taxis {}", attr_dict['axis'])
        log_debug3("\tkeep_dims {}", attr_dict['keep_dims'])
        log_debug3("\texclude {}", attr_dict['exclude'])

        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, self.reduce_type,
                                                ReduceOp.ir_to_legacy_type[self.reduce_type])
        axis = attr_dict['axis']
        exclude = attr_dict['exclude']
        input_shape = converter_context.get_input_shapes(relay_expr)[0]
        input_dim = len(input_shape)
        if len(axis) == 0:
            axis = [i for i in range(input_dim)]

        if exclude:
            axis = [i for i in range(input_dim) if i not in axis]

        keep_dims = attr_dict['keep_dims']
        ir_op = ReduceOp(op_name,
                         reduce_type=self.reduce_type,
                         axes=axis,
                         keep_dims=keep_dims)

        return ir_op


# ------------------------------------------------------------------------------
#   ReduceMaxOp
# ------------------------------------------------------------------------------
class RelayReduceMaxTranslation(RelayReduceBaseTranslation):
    def __init__(self):
        super(RelayReduceMaxTranslation, self).__init__(ir_graph.QNN_OP_REDUCE_MAX)

RelayTranslations.register_translation(RelayReduceMaxTranslation(),
                                       converter_type('max', 'relay'))


# ------------------------------------------------------------------------------
#   ReduceMinOp
# ------------------------------------------------------------------------------
class RelayReduceMinTranslation(RelayReduceBaseTranslation):
    def __init__(self):
        super(RelayReduceMinTranslation, self).__init__(ir_graph.QNN_OP_REDUCE_MIN)

RelayTranslations.register_translation(RelayReduceMinTranslation(),
                                       converter_type('min', 'relay'))


# ------------------------------------------------------------------------------
#   ReduceMeanOp
# ------------------------------------------------------------------------------
class RelayReduceMeanTranslation(RelayReduceBaseTranslation):
    def __init__(self):
        super(RelayReduceMeanTranslation, self).__init__(ir_graph.QNN_OP_REDUCE_MEAN)

RelayTranslations.register_translation(RelayReduceMeanTranslation(),
                                       converter_type('mean', 'relay'))


# ------------------------------------------------------------------------------
#   ReduceProdOp
# ------------------------------------------------------------------------------
class RelayReduceProdTranslation(RelayReduceBaseTranslation):
    def __init__(self):
        super(RelayReduceProdTranslation, self).__init__(ir_graph.QNN_OP_REDUCE_PROD)

RelayTranslations.register_translation(RelayReduceProdTranslation(),
                                       converter_type('prod', 'relay'))


# ------------------------------------------------------------------------------
#   ReduceSumOp
# ------------------------------------------------------------------------------
class RelayReduceSumTranslation(RelayReduceBaseTranslation):
    def __init__(self):
        super(RelayReduceSumTranslation, self).__init__(ir_graph.QNN_OP_REDUCE_SUM)

RelayTranslations.register_translation(RelayReduceSumTranslation(),
                                       converter_type('sum', 'relay'))


# ------------------------------------------------------------------------------
#   TopKOp
# ------------------------------------------------------------------------------
class RelayTopKTranslation(RelayTranslationBase):
    def __init__(self):
        super(RelayTopKTranslation, self).__init__()

    def extract_attributes(self,
                           relay_expr: relay.expr.Call,
                           relay_params: dict,
                           **kwargs):
        attr_dict = {}
        k = int(relay_expr.attrs.k)
        axis = int(relay_expr.attrs.axis)
        is_ascend = relay_expr.attrs.is_ascend
        attr_dict["k"] = k
        attr_dict["axis"] = axis
        attr_dict["is_ascend"] = is_ascend
        attr_dict["sorted"] = True
        log_debug3("\tk {} {}", type(k), k)
        log_debug3("\taxis {} {}", type(axis), axis)
        log_debug3("\tis_ascend {} {}", type(is_ascend), is_ascend)
        return attr_dict

    def translate_op(self,
                     relay_expr: relay.expr.Call,
                     relay_params: dict,
                     converter_context,
                     quir_graph: IROpGraph,
                     attr_dict: dict,
                     input_names: list):
        op_name = converter_context.get_op_name(relay_expr, TopKOp.TRANSLATION_KEY, TopKOp.LEGACY_TRANSLATION_KEY)
        k = attr_dict['k']
        input_shapes = converter_context.get_input_shapes(relay_expr)
        axis = attr_dict['axis']
        input_rank = len(input_shapes[0])
        if axis < 0:
            axis += input_rank
        if axis != input_rank - 1:
            raise ValueError("TopK axis only supported as input_rank-1, got axis {}".format(axis))
        if k < 1:
        #TVM topk define:k (int or relay.Expr, optional) – Number of top elements to select. Return all elements if k < 1.
            k = input_shapes[0][axis]
        if k > input_shapes[0][axis]:
            raise ValueError("ERROR_TOPK_K_INVALID:{}".format(k))
        is_ascend = attr_dict['is_ascend']
        is_sorted = attr_dict['sorted']
        largest = not is_ascend
        log_assert(largest, "TopK Op {}, attribute largest must be True", op_name)
        log_assert(is_sorted, "TopK Op {}, attribute sorted must be True", op_name)
        log_assert(input_rank >= 1, "ERROR_TOPK_INPUT_TENSOR_RANK:input_rank should >= 1,but got {}", input_rank)

        ir_op = TopKOp(op_name, k=k)

        return ir_op

RelayTranslations.register_translation(RelayTopKTranslation(),
                                       converter_type('topk', 'relay'))
