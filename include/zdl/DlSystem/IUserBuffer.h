//==============================================================================
//
// Copyright (c) 2022 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef _IUSER_BUFFER_H
#define _IUSER_BUFFER_H

#include <stddef.h>
#include <stdint.h>

#include "DlSystem/SnpeApiExportDefine.h"
#include "DlSystem/TensorShape.h"
#include "DlSystem/DlError.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @addtogroup c_apis C
@{ */

/** @name IUserBuffer
 *  Description of IUserBuffer.
 * @{ */

/**
 * A typedef to indicate a SNPE UserByfferEncoding handle
 */
typedef void* Snpe_UserBufferEncoding_Handle_t;

/**
 * @brief .
 *
 * An enum class of all supported element types in a IUserBuffer
 */
//enum class Snpe_UserBufferEncoding_ElementType_t
typedef enum
{
  /// Unknown element type.
  SNPE_USERBUFFERENCODING_ELEMENTTYPE_UNKNOWN         = 0,

  /// Each element is presented by float.
  SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT           = 1,

  /// Each element is presented by an unsigned int.
  SNPE_USERBUFFERENCODING_ELEMENTTYPE_UNSIGNED8BIT    = 2,

  /// Each element is presented by float16.
  SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT16         = 3,

  /// Each element is presented by an 8-bit quantized value.
  SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF8             = 10,

  /// Each element is presented by an 16-bit quantized value.
  SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF16            = 11,

  /// Each element is presented by Int32
  SNPE_USERBUFFERENCODING_ELEMENTTYPE_INT32           = 12,

  /// Each element is presented by UInt32
  SNPE_USERBUFFERENCODING_ELEMENTTYPE_UINT32          = 13,

  /// Each element is presented by Int8
  SNPE_USERBUFFERENCODING_ELEMENTTYPE_INT8            = 14,

  /// Each element is presented by UInt8
  SNPE_USERBUFFERENCODING_ELEMENTTYPE_UINT8           = 15,

}Snpe_UserBufferEncoding_ElementType_t;


/**
 * @brief Retrieves the element type
 *
 * @param[in] userBufferHandle : Handle to access userBufferEncoding
 *
 * @return Element type
 */
SNPE_API
Snpe_UserBufferEncoding_ElementType_t Snpe_UserBufferEncoding_GetElementType(Snpe_UserBufferEncoding_Handle_t userBufferHandle);


/**
 * @brief .
 *
 * A base class buffer source type
 *
 * @note User buffer from CPU support all kinds of runtimes;
 *       User buffer from GLBUFFER support only GPU runtime.
 */
typedef void* Snpe_UserBufferSource_Handle_t;

typedef enum
{
  /// Unknown buffer source type.
  SNPE_USERBUFFERSOURCE_SOURCETYPE_UNKNOWN = 0,

  /// The network inputs are from CPU buffer.
  SNPE_USERBUFFERSOURCE_SOURCETYPE_CPU = 1,

  /// The network inputs are from OpenGL buffer.
  SNPE_USERBUFFERSOURCE_SOURCETYPE_GLBUFFER = 2
}Snpe_UserBufferSource_SourceType_t;

/**
 * @brief Retrieves the source type
 *
 * @param[in] userBufferHandle : Handle to access userBufferSource
 *
 * @return Source type
 */
SNPE_API
Snpe_UserBufferSource_SourceType_t Snpe_UserBufferSource_GetSourceType(Snpe_UserBufferSource_Handle_t userBufferHandle);


/**
 * @brief .
 *
 * An source type where input data is delivered from OpenGL buffer
 */
SNPE_API
Snpe_UserBufferSource_Handle_t Snpe_UserBufferSourceGLBuffer_Create();

/**
 * @brief Destroys the userBuffer
 *
 * @param[in] userBufferHandle : Handle to access the UserBuffer
 *
 * @return Error code. Returns SNPE_SUCCESS if destruction successful
 */
SNPE_API
Snpe_ErrorCode_t Snpe_UserBufferSourceGLBuffer_Delete(Snpe_UserBufferSource_Handle_t userBufferHandle);

// Encoding 8 Bit
/**
 * @brief .
 *
 * An encoding type where each element is represented by an unsigned int
 */
SNPE_API
Snpe_UserBufferEncoding_Handle_t Snpe_UserBufferEncodingUnsigned8Bit_Create();

/**
 * @brief Destroys the encodingUnsigned8Bit
 *
 * @param[in] userBufferHandle : Handle to access the encodingUnsigned8Bit
 *
 * @return Error code. Returns SNPE_SUCCESS if destruction successful
 */
SNPE_API
Snpe_ErrorCode_t Snpe_UserBufferEncodingUnsigned8Bit_Delete(Snpe_UserBufferEncoding_Handle_t userBufferHandle);

/**
 * @brief Retrieves the size of the element, in bytes.
 *
 * @param[in] userBufferHandle : Handle to access the encoding
 *
 * @return Size of the element, in bytes.
 */
SNPE_API
size_t Snpe_UserBufferEncodingUnsigned8Bit_GetElementSize(Snpe_UserBufferEncoding_Handle_t userBufferHandle);


// Encoding Float
/**
 * @brief .
 *
 * An encoding type where each element is represented by a float
 */
SNPE_API
Snpe_UserBufferEncoding_Handle_t Snpe_UserBufferEncodingFloat_Create();

/**
 * @brief Destroys the encodingFloat
 *
 * @param[in] userBufferHandle : Handle to access the encoding
 *
 * @return Error code. Returns SNPE_SUCCESS if destruction successful
 */
SNPE_API
Snpe_ErrorCode_t Snpe_UserBufferEncodingFloat_Delete(Snpe_UserBufferEncoding_Handle_t userBufferHandle);

/**
 * @brief Retrieves the size of the element, in bytes.
 *
 * @param[in] userBufferHandle : Handle to access the encoding
 *
 * @return Size of the element, in bytes.
 */
SNPE_API
size_t Snpe_UserBufferEncodingFloat_GetElementSize(Snpe_UserBufferEncoding_Handle_t userBufferHandle);

// Encoding FloatN
/**
 * @brief .
 *
 * An encoding type where each element is represented by a float N
 */
SNPE_API
Snpe_UserBufferEncoding_Handle_t Snpe_UserBufferEncodingFloatN_Create(uint8_t bWidth);

/**
 * @brief Destroys the encodingFloatN
 *
 * @param[in] userBufferHandle : Handle to access the encoding
 *
 * @return Error code. Returns SNPE_SUCCESS if destruction successful
 */
SNPE_API
Snpe_ErrorCode_t Snpe_UserBufferEncodingFloatN_Delete(Snpe_UserBufferEncoding_Handle_t userBufferHandle);

/**
 * @brief Retrieves the size of the element, in bytes.
 *
 * @param[in] userBufferHandle : Handle to access the encoding
 *
 * @return Size of the element, in bytes.
 */
SNPE_API
size_t Snpe_UserBufferEncodingFloatN_GetElementSize(Snpe_UserBufferEncoding_Handle_t userBufferHandle);

/**
 * @brief .
 *
 * An encoding type where each element is represented by tfN, which is an
 * N-bit quantizd value, which has an exact representation of 0.0
 */
SNPE_API
Snpe_UserBufferEncoding_Handle_t Snpe_UserBufferEncodingTfN_Create(uint64_t stepFor0, float stepSize, uint8_t bWidth);

/**
 * @brief Destroys the encodingTfN
 *
 * @param[in] userBufferHandle : Handle to access the encoding
 *
 * @return Error code. Returns SNPE_SUCCESS if destruction successful
 */
SNPE_API
Snpe_ErrorCode_t Snpe_UserBufferEncodingTfN_Delete(Snpe_UserBufferEncoding_Handle_t userBufferHandle);

/**
 * @brief Retrieves the size of the element, in bytes.
 *
 * @param[in] userBufferHandle : Handle to access the encoding
 *
 * @return Size of the element, in bytes.
 */
SNPE_API
size_t Snpe_UserBufferEncodingTfN_GetElementSize(Snpe_UserBufferEncoding_Handle_t userBufferHandle);

/**
 * @brief Sets the step value that represents 0
 *
 * @param[in] userBufferHandle : Handle to access the encoding
 *
 * @param[in] stepExactly0 : The step value that represents 0
 *
 */
SNPE_API
void Snpe_UserBufferEncodingTfN_SetStepExactly0(Snpe_UserBufferEncoding_Handle_t userBufferHandle, uint64_t stepExactly0);

/**
 * @brief Sets the float value that each step represents
 *
 * @param[in] userBufferHandle : Handle to access the encoding
 *
 * @param[in] quantizedStepSize : The float value of each step size
 *
 */
SNPE_API
void Snpe_UserBufferEncodingTfN_SetQuantizedStepSize(Snpe_UserBufferEncoding_Handle_t userBufferHandle, float quantizedStepSize);

/**
 * @brief Retrieves the step that represents 0.0
 *
 * @param[in] userBufferHandle : Handle to access the encoding
 *
 * @return Step value
 */
SNPE_API
uint64_t Snpe_UserBufferEncodingTfN_GetStepExactly0(Snpe_UserBufferEncoding_Handle_t userBufferHandle);

/**
 * @brief Retrieves the step size
 *
 * @param[in] userBufferHandle : Handle to access the encoding
 *
 * @return Step size
 */
SNPE_API
float Snpe_UserBufferEncodingTfN_GetQuantizedStepSize(Snpe_UserBufferEncoding_Handle_t userBufferHandle);

/**
 * Calculates the minimum floating point value that
 * can be represented with this encoding.
 *
 * @param[in] userBufferHandle : Handle to access the encoding
 *
 * @return Minimum representable floating point value
 */
SNPE_API
float Snpe_UserBufferEncodingTfN_GetMin(Snpe_UserBufferEncoding_Handle_t userBufferHandle);

/**
 * Calculates the maximum floating point value that
 * can be represented with this encoding.
 *
 * @param[in] userBufferHandle : Handle to access the encoding
 *
 * @return Maximum representable floating point value
 */
SNPE_API
float Snpe_UserBufferEncodingTfN_GetMax(Snpe_UserBufferEncoding_Handle_t userBufferHandle);


SNPE_API
Snpe_UserBufferEncoding_ElementType_t Snpe_UserBufferEncodingTfN_GetTypeFromWidth(Snpe_UserBufferEncoding_Handle_t userBufferHandle, uint8_t width);

// Encoding Int N
/**
 * @brief .
 *
 * An encoding type where each element is represented by a Int
 */
SNPE_API
Snpe_UserBufferEncoding_Handle_t Snpe_UserBufferEncodingIntN_Create(uint8_t bWidth);

/**
 * @brief Destroys the encodingIntN
 *
 * @param[in] userBufferHandle : Handle to access the encoding
 *
 * @return Error code. Returns SNPE_SUCCESS if destruction successful
 */
SNPE_API
Snpe_ErrorCode_t Snpe_UserBufferEncodingIntN_Delete(Snpe_UserBufferEncoding_Handle_t userBufferHandle);

/**
 * @brief Retrieves the size of the element, in bytes.
 *
 * @param[in] userBufferHandle : Handle to access the encoding
 *
 * @return Size of the element, in bytes.
 */
SNPE_API
size_t Snpe_UserBufferEncodingIntN_GetElementSize(Snpe_UserBufferEncoding_Handle_t userBufferHandle);

// Encoding Uint N
/**
 * @brief .
 *
 * An encoding type where each element is represented by a Uint
 */
SNPE_API
Snpe_UserBufferEncoding_Handle_t Snpe_UserBufferEncodingUintN_Create(uint8_t bWidth);

/**
 * @brief Destroys the encodingUintN
 *
 * @param[in] userBufferHandle : Handle to access the encoding
 *
 * @return Error code. Returns SNPE_SUCCESS if destruction successful
 */
SNPE_API
Snpe_ErrorCode_t Snpe_UserBufferEncodingUintN_Delete(Snpe_UserBufferEncoding_Handle_t userBufferHandle);

/**
 * @brief Retrieves the size of the element, in bytes.
 *
 * @param[in] userBufferHandle : Handle to access the encoding
 *
 * @return Size of the element, in bytes.
 */
SNPE_API
size_t Snpe_UserBufferEncodingUintN_GetElementSize(Snpe_UserBufferEncoding_Handle_t userBufferHandle);

/**
 * A typedef to indicate a SNPE IUserBuffer handle
 * UserBuffer contains a pointer and info on how to walk it and interpret its content.
 */
typedef void* Snpe_IUserBuffer_Handle_t;

/**
 * Destroys/frees an IUserBuffer
 *
 * @param[in] userBufferHandle : Handle to access the IUserBuffer
 *
 * @return SNPE_SUCCESS if Delete operation successful.
 */
SNPE_API
Snpe_ErrorCode_t Snpe_IUserBuffer_Delete(Snpe_IUserBuffer_Handle_t userBufferHandle);


/**
 * @brief Retrieves the total number of bytes between elements in each dimension if
 * the buffer were to be interpreted as a multi-dimensional array.
 *
 * @param[in] userBufferHandle : Handle to access the user Buffer
 *
 * @warning Do not modify the TensorShape returned by reference. Treat it as a const reference.
 *
 * @return A const reference to the number of bytes between elements in each dimension.
 * e.g. A tightly packed tensor of floats with dimensions [4, 3, 2] would
 * return strides of [24, 8, 4].
 */
SNPE_API
Snpe_TensorShape_Handle_t Snpe_IUserBuffer_GetStrides_Ref(Snpe_IUserBuffer_Handle_t userBufferHandle);

/**
 * @brief Retrieves the size of the buffer, in bytes.
 *
 * @param[in] userBufferHandle : Handle to access the user Buffer
 *
 * @return Size of the underlying buffer, in bytes.
 */
SNPE_API
size_t Snpe_IUserBuffer_GetSize(Snpe_IUserBuffer_Handle_t userBufferHandle);

/**
 * @brief Retrieves the size of the inference data in the buffer, in bytes.
 *
 * The inference results from a dynamic-sized model may not be exactly the same size
 * as the UserBuffer provided to SNPE. This function can be used to get the amount
 * of output inference data, which may be less or greater than the size of the UserBuffer.
 *
 * If the inference results fit in the UserBuffer, getOutputSize() would be less than
 * or equal to getSize(). But if the inference results were more than the capacity of
 * the provided UserBuffer, the results would be truncated to fit the UserBuffer. But,
 * getOutputSize() would be greater than getSize(), which indicates a bigger buffer
 * needs to be provided to SNPE to hold all of the inference results.
 *
 * @param[in] userBufferHandle : Handle to access the user Buffer
 *
 * @return Size required for the buffer to hold all inference results, which can be less
 * or more than the size of the buffer, in bytes.
 */
SNPE_API
size_t Snpe_IUserBuffer_GetOutputSize(Snpe_IUserBuffer_Handle_t userBufferHandle);

/**
 * @brief Changes the underlying memory that backs the UserBuffer.
 *
 * This can be used to avoid creating multiple UserBuffer objects
 * when the only thing that differs is the memory location.
 *
 * @param[in] userBufferHandle : Handle to access the user Buffer
 *
 * @param[in] buffer : Pointer to the memory location
 *
 * @return Whether the set succeeds.
 */
SNPE_API
int Snpe_IUserBuffer_SetBufferAddress(Snpe_IUserBuffer_Handle_t userBufferHandle, void* buffer);

/**
 * @brief Gets a reference to the data encoding object of
 *        the underlying buffer
 *
 * This is necessary when the UserBuffer is re-used, and the encoding
 * parameters can change.  For example, each input can be quantized with
 * different step sizes.
 *
 * @param[in] userBufferHandle : Handle to access the user Buffer
 *
 * @return Data encoding meta-data
 */
SNPE_API
Snpe_UserBufferEncoding_Handle_t Snpe_IUserBuffer_GetEncoding_Ref(Snpe_IUserBuffer_Handle_t userBufferHandle);

/** @} */
/** @} */ /* end_addtogroup c_apis C */

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // _IUSER_BUFFER_H
