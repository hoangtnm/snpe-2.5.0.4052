#=============================================================================
#
#  Copyright (c) 2021-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================
import os
import sys

TENSORFLOW = 'TENSORFLOW'
CAFFE = 'CAFFE'
CAFFE_SSD = 'CAFFE_SSD'
ONNX = 'ONNX'
TFLITE = 'TFLITE'

def getEnvironment(configParams, sdkDir, mlFramework=None, pythonPath=None):
    environ = dict()
    environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    environ['PYTHONPATH'] = os.path.join(sdkDir, 'lib/python')
    if pythonPath:
        environ['PYTHONPATH'] = pythonPath + ':' + environ['PYTHONPATH']
    environ['SNPE_SDK_ROOT'] = sdkDir
    if "ANDROID_NDK_PATH" not in configParams:
        print("ERROR: Please provide ANDROID_NDK PATH to config_file.", flush=True)
        exit(-1)
    environ['PATH'] = configParams["ANDROID_NDK_PATH"]
    if "CLANG_PATH" not in configParams:
        print("ERROR: Please provide CLANG PATH to config_file.", flush=True)
        exit(-1)
    environ['PATH'] = configParams["CLANG_PATH"] + ':' + environ['PATH']
    if "BASH_PATH" not in configParams:
        print("ERROR: Please provide BASH PATH to config_file.", flush=True)
        exit(-1)
    environ['PATH'] = configParams["BASH_PATH"] + ':' + environ['PATH']
    if "PY3_PATH" not in configParams:
        print("ERROR: Please provide Python3 environment bin PATH to config_file.", flush=True)
        exit(-1)
    if "BIN_PATH" not in configParams:
        print("ERROR: Please provide BIN PATH to config_file.", flush=True)
        exit(-1)
    environ['PATH'] = configParams['PY3_PATH'] + ':' + environ['PATH'] + ':' + configParams["BIN_PATH"]
    environ['PATH'] = os.path.join(sdkDir, 'bin/x86_64-linux-clang') + ':' + environ['PATH']
    environ['LD_LIBRARY_PATH'] = os.path.join(sdkDir, 'lib/x86_64-linux-clang')
    environ['SNPE_UDO_ROOT'] = os.path.join(sdkDir, 'share/SnpeUdo/')
    if mlFramework == TENSORFLOW:
        if 'TENSORFLOW_HOME' not in configParams:
            print('ERROR: Please provide TENSORFLOW PATH to config_file.', flush=True)
            exit(-1)
        environ['TENSORFLOW_HOME'] = configParams['TENSORFLOW_HOME']
        environ['PYTHONPATH'] = environ['PYTHONPATH'] + ':' + environ['TENSORFLOW_HOME']
        environ['PYTHONPATH'] = environ['PYTHONPATH'] + ':' + os.path.join(environ['TENSORFLOW_HOME'], 'distribute')
        environ['PYTHONPATH'] = environ['PYTHONPATH'] + ':' + os.path.join(environ['TENSORFLOW_HOME'], 'dependencies/python')
    elif mlFramework == TFLITE:
        if 'TFLITE_HOME' not in configParams:
            print('ERROR: Please provide TFLITE PATH to config_file.', flush=True)
            exit(-1)
        environ['TFLITE_HOME'] = configParams['TFLITE_HOME']
        environ['PYTHONPATH'] = environ['PYTHONPATH'] + ':' + environ['TFLITE_HOME']
        environ['PYTHONPATH'] = environ['PYTHONPATH'] + ':' + os.path.join(environ['TFLITE_HOME'], 'distribute')
        environ['PYTHONPATH'] = environ['PYTHONPATH'] + ':' + os.path.join(environ['TFLITE_HOME'], 'dependencies/python')
    elif mlFramework == CAFFE:
        if 'CAFFE_HOME' not in configParams:
            print('ERROR: Please provide CAFFE PATH to config_file.', flush=True)
            exit(-1)
        environ['CAFFE_HOME'] = configParams['CAFFE_HOME']
        environ['PATH'] = os.path.join(environ['CAFFE_HOME'], 'build/install/bin') + ':' + environ['PATH']
        environ['PATH'] = os.path.join(environ['CAFFE_HOME'], 'distribute/bin') + ':' + environ['PATH']
        environ['LD_LIBRARY_PATH'] = os.path.join(environ['CAFFE_HOME'], 'build/install/lib') + ':' + environ['LD_LIBRARY_PATH']
        environ['LD_LIBRARY_PATH'] = environ['LD_LIBRARY_PATH'] + ':' + os.path.join(environ['CAFFE_HOME'], 'distribute/caffe/distribute/lib')
        environ['LD_LIBRARY_PATH'] = environ['LD_LIBRARY_PATH'] + ':' + os.path.join(environ['CAFFE_HOME'], 'distribute/lib')
        environ['PYTHONPATH'] = environ['PYTHONPATH'] + ':' + os.path.join(environ['CAFFE_HOME'], 'distribute/caffe/distribute/python')
        environ['PYTHONPATH'] = environ['PYTHONPATH'] + ':' + os.path.join(environ['CAFFE_HOME'], 'dependencies/python')
        environ['PYTHONPATH'] = environ['PYTHONPATH'] + ':' + os.path.join(environ['CAFFE_HOME'], 'build/install/python')
        environ['PYTHONPATH'] = environ['PYTHONPATH'] + ':' + os.path.join(environ['CAFFE_HOME'], 'distribute/python')
    elif mlFramework == CAFFE_SSD:
        if 'CAFFE_SSD_HOME' not in configParams:
            print('ERROR: Please provide CAFFE_SSD PATH to config_file.', flush=True)
            exit(-1)
        environ['CAFFE_HOME'] = configParams['CAFFE_SSD_HOME']
        environ['PATH'] = os.path.join(environ['CAFFE_HOME'], 'build/install/bin') + ':' + environ['PATH']
        environ['PATH'] = os.path.join(environ['CAFFE_HOME'], 'distribute/bin') + ':' + environ['PATH']
        environ['LD_LIBRARY_PATH'] = os.path.join(environ['CAFFE_HOME'], 'build/install/lib') + ':' + environ['LD_LIBRARY_PATH']
        environ['LD_LIBRARY_PATH'] = environ['LD_LIBRARY_PATH'] + ':' + os.path.join(environ['CAFFE_HOME'], 'distribute/ssd/distribute/lib')
        environ['LD_LIBRARY_PATH'] = environ['LD_LIBRARY_PATH'] + ':' + os.path.join(environ['CAFFE_HOME'], 'distribute/lib')
        environ['PYTHONPATH'] = environ['PYTHONPATH'] + ':' + os.path.join(environ['CAFFE_HOME'], 'distribute/ssd/distribute/python')
        environ['PYTHONPATH'] = environ['PYTHONPATH'] + ':' + os.path.join(environ['CAFFE_HOME'], 'dependencies/python')
        environ['PYTHONPATH'] = environ['PYTHONPATH'] + ':' + os.path.join(environ['CAFFE_HOME'], 'build/install/python')
        environ['PYTHONPATH'] = environ['PYTHONPATH'] + ':' + os.path.join(environ['CAFFE_HOME'], 'distribute/python')
    elif mlFramework == ONNX:
        if 'ONNX_HOME' not in configParams:
            print('ERROR: Please provide ONNX path.', flush=True)
            exit(-1)
        environ['ONNX_HOME'] = configParams['ONNX_HOME']
        environ['PYTHONPATH'] = environ['PYTHONPATH'] + ':' + environ['ONNX_HOME']
        environ['PYTHONPATH'] = environ['PYTHONPATH'] + ':' + os.path.join(environ['ONNX_HOME'], 'distribute')
        environ['PYTHONPATH'] = environ['PYTHONPATH'] + ':' + os.path.join(environ['ONNX_HOME'], 'dependencies/python')

    return environ

def setEnvironment(configParams, sdkDir, mlFramework=None):
    if mlFramework == TENSORFLOW:
        if 'TENSORFLOW_HOME' not in configParams:
            print('ERROR: Please provide TENSORFLOW PATH to config_file.', flush=True)
            exit(-1)
        curr_sys_path = []
        for path in sys.path:
            curr_sys_path.append(path)
        sys.path.clear()
        sys.path.append(os.path.join(sdkDir, 'lib/python'))
        sys.path.append(configParams['TENSORFLOW_HOME'])
        sys.path.append(os.path.join(configParams['TENSORFLOW_HOME'], 'distribute'))
        sys.path.append(os.path.join(configParams['TENSORFLOW_HOME'], 'dependencies/python'))
        for path in curr_sys_path:
            if path not in sys.path:
                sys.path.append(path)
    elif mlFramework == TFLITE:
        if 'TFLITE_HOME' not in configParams:
            print('ERROR: Please provide TFLITE PATH to config_file.', flush=True)
            exit(-1)
        curr_sys_path = []
        for path in sys.path:
            curr_sys_path.append(path)
        sys.path.clear()
        sys.path.append(os.path.join(sdkDir, 'lib/python'))
        sys.path.append(configParams['TFLITE_HOME'])
        sys.path.append(os.path.join(configParams['TFLITE_HOME'], 'distribute'))
        sys.path.append(os.path.join(configParams['TFLITE_HOME'], 'dependencies/python'))
        for path in curr_sys_path:
            if path not in sys.path:
                sys.path.append(path)
    elif mlFramework == CAFFE:
        if 'CAFFE_HOME' not in configParams:
            print('ERROR: Please provide CAFFE PATH to config_file.', flush=True)
            exit(-1)
        curr_sys_path = []
        for path in sys.path:
            curr_sys_path.append(path)
        sys.path.clear()
        sys.path.append(os.path.join(sdkDir, 'lib/python'))
        sys.path.append(os.path.join(configParams['CAFFE_HOME'], 'distribute/caffe/distribute/python'))
        sys.path.append(os.path.join(configParams['CAFFE_HOME'], 'dependencies/python'))
        sys.path.append(os.path.join(configParams['CAFFE_HOME'], 'build/install/python'))
        sys.path.append(os.path.join(configParams['CAFFE_HOME'], 'distribute/python'))
        sys.path.append(os.path.join(configParams['CAFFE_HOME'], 'distribute/python'))
        for path in curr_sys_path:
            if path not in sys.path:
                sys.path.append(path)
    elif mlFramework == CAFFE_SSD:
        if 'CAFFE_SSD_HOME' not in configParams:
            print('ERROR: Please provide CAFFE_SSD PATH to config_file.', flush=True)
            exit(-1)
        curr_sys_path = []
        for path in sys.path:
            curr_sys_path.append(path)
        sys.path.clear()
        sys.path.append(os.path.join(sdkDir, 'lib/python'))
        sys.path.append(os.path.join(configParams['CAFFE_SSD_HOME'], 'distribute/caffe/distribute/python'))
        sys.path.append(os.path.join(configParams['CAFFE_SSD_HOME'], 'dependencies/python'))
        sys.path.append(os.path.join(configParams['CAFFE_SSD_HOME'], 'build/install/python'))
        sys.path.append(os.path.join(configParams['CAFFE_SSD_HOME'], 'distribute/python'))
        sys.path.append(os.path.join(configParams['CAFFE_SSD_HOME'], 'distribute/python'))
        for path in curr_sys_path:
            if path not in sys.path:
                sys.path.append(path)
    elif mlFramework == ONNX:
        if 'ONNX_HOME' not in configParams:
            print('ERROR: Please provide ONNX path.', flush=True)
            exit(-1)
        curr_sys_path = []
        for path in sys.path:
            curr_sys_path.append(path)
        sys.path.clear()
        sys.path.append(os.path.join(sdkDir, 'lib/python'))
        sys.path.append(configParams['ONNX_HOME'])
        sys.path.append(os.path.join(configParams['ONNX_HOME'], 'distribute'))
        sys.path.append(os.path.join(configParams['ONNX_HOME'], 'dependencies/python'))
        for path in curr_sys_path:
            if path not in sys.path:
                sys.path.append(path)
