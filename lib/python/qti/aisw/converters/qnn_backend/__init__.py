# ==============================================================================
#
#  Copyright (c) 2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from qti.aisw.converters.common import libPyIrGraph as ir_graph
try:
    from . import libPyQnnDefinitions as qnn_definitions
except ImportError as e:
    try:
        import libPyQnnDefines as qnn_definitions
    except ImportError:
        raise ImportError(e)
