#
# Copyright (c) 2020, 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#

import os
import subprocess
import shutil

SNPE_UDO_PATH                           = ''
if 'SNPE_ROOT' in os.environ:
    SNPE_UDO_PATH                       =  os.path.join(os.environ['SNPE_ROOT'], 'examples', 'NativeCpp', 'UdoExample', 'Softmax')
UDO_PACKAGE                             = 'SoftmaxUdoPackage'
INCEPTION_V3_UDO_DLC_FILENAME           = 'inception_v3_udo.dlc'
INCEPTION_V3_UDO_QUANTIZED_DLC_FILENAME = 'inception_v3_udo_quantized.dlc'
INCEPTION_V3_UDO_PLUGIN                 = 'Softmax.json'
INCEPTION_V3_UDO_PLUGIN_DSP             = 'Softmax_Quant.json'
INCEPTION_V3_UDO_PLUGIN_HTP             = 'Softmax_v68.json'

# UDO Setup
def setup_udo(udo_package_path, runtime, is_quantized, enable_htp = False):
    if not os.path.isdir(os.path.join(udo_package_path, UDO_PACKAGE)):
        create_udo_package(udo_package_path, is_quantized, enable_htp)
        set_udo_impl(udo_package_path, is_quantized, enable_htp)

    compile_udo_package(udo_package_path, runtime, enable_htp)

# Step 1: Create UDO Package
def create_udo_package(udo_package_path, is_quantized, enable_htp):
    try:
        import mako
    except ImportError as e:
        raise RuntimeError('Mako cannot be found. Please install Mako to use UDO Package Generator')
    if not os.path.isdir(SNPE_UDO_PATH):
        raise RuntimeError('UdoExample cannot be found. Please place UdoExample under ${SNPE_ROOT}/examples/NativeCpp')

    print('INFO: Creating UDO Package ' + UDO_PACKAGE)
    if enable_htp:
        config_path = os.path.join(SNPE_UDO_PATH, 'config', INCEPTION_V3_UDO_PLUGIN_HTP)
    elif is_quantized == True:
        config_path = os.path.join(SNPE_UDO_PATH, 'config', INCEPTION_V3_UDO_PLUGIN_DSP)
    else:
        config_path = os.path.join(SNPE_UDO_PATH, 'config', INCEPTION_V3_UDO_PLUGIN)
    cmd = ['snpe-udo-package-generator', '-c', '-p', config_path, '-o', udo_package_path]
    subprocess.call(cmd)

# Step 2: Set UDO Implementations
def set_udo_impl(udo_package_path, is_quantized, enable_htp):
    print('INFO: Populating UDO Package Implementations')
    impl_libs = [os.path.join('CPU', 'Softmax.cpp')]
    package_libs = [os.path.join('CPU', 'src', 'ops', 'Softmax.cpp')]

    gpu_impl_libs = os.path.join('GPU', 'Softmax.cpp')
    gpu_package_libs = os.path.join('GPU', 'src', 'ops', 'Softmax.cpp')
    impl_libs.append(gpu_impl_libs)
    package_libs.append(gpu_package_libs)

    if enable_htp:
        dsp_impl_lib = os.path.join('DSP_v68', 'SoftmaxImplLibDsp.cpp')
        impl_libs.append(dsp_impl_lib)
        package_libs.append(os.path.join('DSP_V68', 'SoftmaxImplLibDsp.cpp'))
    elif not is_quantized:
        dsp_impl_lib = os.path.join('DSP', 'SoftmaxFloatImpl', 'SoftmaxImplLibDsp.c')
        impl_libs.append(dsp_impl_lib)
        package_libs.append(os.path.join('DSP', 'SoftmaxImplLibDsp.c'))
        impl_lib_include_path = os.path.join(SNPE_UDO_PATH, 'src', 'DSP', 'SoftmaxFloatImpl', 'SoftmaxImplLibDsp.h')
        package_include_path = os.path.join(udo_package_path, UDO_PACKAGE, 'include', 'SoftmaxImplLibDsp.h')
    else:
        dsp_impl_lib = os.path.join('DSP', 'SoftmaxInt8Impl', 'SoftmaxImplLibDsp.c')
        impl_libs.append(dsp_impl_lib)
        package_libs.append(os.path.join('DSP', 'SoftmaxImplLibDsp.c'))
        impl_lib_include_path = os.path.join(SNPE_UDO_PATH, 'src', 'DSP', 'SoftmaxInt8Impl', 'SoftmaxImplLibDsp.h')
        package_include_path = os.path.join(udo_package_path, UDO_PACKAGE, 'include', 'SoftmaxImplLibDsp.h')

    for impl_lib, package_lib in zip(impl_libs, package_libs):
        impl_lib_path = os.path.join(SNPE_UDO_PATH, 'src', impl_lib)
        package_path = os.path.join(udo_package_path, UDO_PACKAGE, 'jni', 'src', package_lib)
        print("Implementation:" + impl_lib_path)
        print("Package Source:" + package_path)
        if not os.path.isfile(impl_lib_path):
            raise RuntimeError('SnpeUdo src cannot be found. Please place share directory under ${SNPE_ROOT}')
        shutil.copyfile(impl_lib_path, package_path)
        if impl_lib == dsp_impl_lib and not enable_htp:
            if (os.path.isfile(impl_lib_include_path)):
                shutil.copyfile(impl_lib_include_path, package_include_path)

# Step 3: Compile UDO Packages
def compile_udo_package(udo_package_path, runtime, enable_htp):
    if runtime == 'cpu':
        compile_x86_cpu(udo_package_path)
        compile_android_cpu(udo_package_path)
    elif runtime == 'gpu':
        compile_android_gpu(udo_package_path)
    elif runtime == 'dsp' or runtime == 'aip':
        compile_x86_cpu(udo_package_path)
        compile_dsp(udo_package_path, enable_htp)
    else:
        compile_all(udo_package_path, enable_htp)

def compile_all(udo_package_path, enable_htp):
    if 'ANDROID_NDK_ROOT' not in os.environ:
        raise RuntimeError('ANDROID_NDK_ROOT not setup.  Please run the SDK env setup script.')
    proc = subprocess.Popen(['make','all', 'PLATFORM=arm64-v8a'], cwd=os.path.join(udo_package_path, UDO_PACKAGE))
    proc.communicate()
    if enable_htp:
        proc = subprocess.Popen(['make', 'dsp_x86'], cwd=os.path.join(udo_package_path, UDO_PACKAGE))
        proc.communicate()

def compile_android_cpu(udo_package_path):
    if 'ANDROID_NDK_ROOT' not in os.environ:
        print('WARNING: ANDROID_NDK_ROOT not set. Skipping compilation for Android CPU.')
        return
    print('INFO: Compiling UDO Package for CPU runtime')
    proc = subprocess.Popen(['make','cpu_android', 'PLATFORM=arm64-v8a'], cwd=os.path.join(udo_package_path, UDO_PACKAGE))
    proc.communicate()

def compile_android_gpu(udo_package_path):
    if 'CL_INCLUDE_PATH' not in os.environ:
        raise RuntimeError('CL_INCLUDE_PATH not set. Please set CL_INCLUDE_PATH to compile GPU UDO Package')
    if 'CL_LIBRARY_PATH' not in os.environ:
        raise RuntimeError('CL_LIBRARY_PATH not set. Please set CL_LIBRARY_PATH to compile GPU UDO Package')
    if 'ANDROID_NDK_ROOT' not in os.environ:
        raise RuntimeError('ANDROID_NDK_ROOT not setup.  Please run the SDK env setup script.')
    print('INFO: Compiling UDO Package for GPU runtime')
    proc = subprocess.Popen(['make','gpu_android', 'PLATFORM=arm64-v8a'], cwd=os.path.join(udo_package_path, UDO_PACKAGE))
    proc.communicate()

def compile_dsp(udo_package_path, enable_htp):
    if 'HEXAGON_SDK_ROOT' not in os.environ:
        raise RuntimeError('HEXAGON_SDK_ROOT not set. Please set HEXAGON_SDK_ROOT to compile DSP UDO Package')
    if 'HEXAGON_TOOLS_ROOT' not in os.environ:
        raise RuntimeError('HEXAGON_TOOLS_ROOT not set. Please set HEXAGON_TOOLS_ROOT to HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/8.3.07')
    if 'SDK_SETUP_ENV' not in os.environ:
        raise RuntimeError('SDK_SETUP_ENV not set. Please set SDK_SETUP_ENV=Done to compile DSP UDO Package')
    if enable_htp:
        if 'HEXAGON_SDK4_ROOT' not in os.environ:
            raise RuntimeError('HEXAGON_SDK4_ROOT not set. Please set HEXAGON_SDK4_ROOT to compile DSP_V68 UDO Package')
        if 'QNN_SDK_ROOT' not in os.environ:
            raise RuntimeError('QNN_SDK_ROOT not set. Please set QNN_SDK_ROOT to compile DSP_V68 UDO Package')
        if 'X86_CXX' not in os.environ:
            raise RuntimeError('X86_CXX clang not set. Please set x86_64 clang path to compile DSP_V68 UDO Package')
    print('INFO: Compiling UDO Package for DSP runtime')
    if enable_htp:
        proc = subprocess.Popen(['make', 'dsp', 'dsp_x86'], cwd=os.path.join(udo_package_path, UDO_PACKAGE))
        proc.communicate()
    else:
        proc = subprocess.Popen(['make', 'dsp'], cwd=os.path.join(udo_package_path, UDO_PACKAGE))
        proc.communicate()


def compile_x86_cpu(udo_package_path):
    print('INFO: Compiling UDO Package for Linux CPU runtime')
    print(os.path.join(udo_package_path, UDO_PACKAGE))
    proc = subprocess.Popen(['make', 'cpu_x86'], cwd=os.path.join(udo_package_path, UDO_PACKAGE))
    proc.communicate()
