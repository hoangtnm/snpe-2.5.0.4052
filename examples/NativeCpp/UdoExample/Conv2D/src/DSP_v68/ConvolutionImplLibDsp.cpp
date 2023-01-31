//==============================================================================
// Auto Generated Code for Conv2DPackage
//==============================================================================
#include "optimize.h"
#include "op_register_ext.h"

static constexpr auto operatorName = "Convolution";

// op execute function declarations
template<typename TensorType>
GraphStatus convolutionImpl(TensorType& Output,
                            const TensorType& Input,
                            const TensorType& weight_filler,
                            const TensorType& bias_filler,
                            const Tensor& bias_term,
                            const Tensor& group,
                            const Tensor& kernel_h,
                            const Tensor& kernel_w,
                            const Tensor& pad_h,
                            const Tensor& pad_w,
                            const Tensor& stride_h,
                            const Tensor& stride_w,
                            const Tensor& num_output,
                            const Tensor& pad,
                            const Tensor& stride,
                            const Tensor& kernel_size);

//op definitions
DEF_PACKAGE_OP((convolutionImpl<Tensor>), operatorName)

/* execute functions for ops */

template<typename TensorType>
GraphStatus convolutionImpl(TensorType& Output,
                            const TensorType& Input,
                            const TensorType& weight_filler,
                            const TensorType& bias_filler,
                            const Tensor& bias_term,
                            const Tensor& group,
                            const Tensor& kernel_h,
                            const Tensor& kernel_w,
                            const Tensor& pad_h,
                            const Tensor& pad_w,
                            const Tensor& stride_h,
                            const Tensor& stride_w,
                            const Tensor& num_output,
                            const Tensor& pad,
                            const Tensor& stride,
                            const Tensor& kernel_size)

{

    //Initialise params
    int32_t numFilters = num_output(0,0,0,0);
    int32_t kernelSize = kernel_size(0,0,0,0);
    int32_t biasTerm = bias_term(0, 0, 0, 0);
    int32_t padding = pad(0, 0, 0, 0);
    int32_t strides = stride(0, 0, 0, 0);
    int32_t padH = pad_h(0, 0, 0, 0);
    int32_t padW = pad_w(0, 0, 0, 0);
    int32_t strideH = stride_h(0, 0, 0, 0);
    int32_t strideW = stride_w(0, 0, 0, 0);
    int32_t kernelH = kernel_h(0, 0, 0, 0);
    int32_t kernelW = kernel_w(0, 0, 0, 0);
    int32_t groups = group(0, 0, 0, 0);

    if (kernelSize != 0)
    {
        kernelH = kernelSize;
        kernelW = kernelSize;
    }
    if (padding != 0)
    {
        padH = padding;
        padW = padding;
    }
    if (strides != 0)
    {
        strideH = strides;
        strideW = strides;
    }

    auto [b_out, h_out, w_out, d_out] = Output.dims();

    int32_t d_filter = weight_filler.dim(1);
    int32_t h_filter = weight_filler.dim(2);
    int32_t w_filter = weight_filler.dim(3);

    int32_t h_in = Input.dim(1);
    int32_t w_in = Input.dim(2);

    Idx outputGroupDepth = numFilters / groups;

    for (int32_t ob = 0; ob < b_out; ob++)
    {
        for (int32_t oh = 0; oh < h_out; oh++)
        {
            for (int32_t ow = 0; ow < w_out; ow++)
            {
                for (int32_t g = 0; g < groups; g++)
                {
                    for (int32_t d = 0; d < outputGroupDepth; d++)
                    {
                        int32_t inputOriginH = (int32_t)oh * strideH - padH;
                        int32_t inputOriginW = (int32_t)ow * strideW - padW;
                        float sum = 0.0f;
                        int32_t depth = d + g*outputGroupDepth;
                        if(biasTerm)
                        {
                            sum += bias_filler(0,0,0,depth);
                        }
                        for (uint32_t fh = 0; fh < h_filter; fh++)
                        {
                            for (uint32_t fw = 0; fw < w_filter; fw++)
                            {
                                int32_t inputH  = inputOriginH + (int32_t)fh;
                                int32_t inputW  = inputOriginW + (int32_t)fw;
                                for (uint32_t fd = 0; fd < d_filter; fd++)
                                {
                                    if (inputH >= 0 && inputH < (int32_t)(h_in) && inputW >= 0 &&
                                        inputW < (int32_t)(w_in))
                                    {
                                        float inval = Input(ob,inputH,inputW,fd);
                                        float filtval = weight_filler(depth,fd,fh,fw);
                                        sum += inval * filtval;
                                    }

                                }
                            }
                        }

                        Output(ob,oh,ow,depth) = sum ;
                    }
                }
            }
        }
    }

    return GraphStatus::Success;
}



