/*
 * Copyright (c) 2021 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.networkEvaluation;

public class SuperResolutionResult extends Result {
    private double psnr;
    private double ssim;

    public SuperResolutionResult() {
        super();
        psnr = 0;
        ssim = 0;
    }

    public double getPnsr() {
        return psnr;
    }

    public void setPnsr(double pnsr) {
        this.psnr = pnsr;
    }

    public double getSsim() {
        return ssim;
    }

    public void setSsim(double ssim) {
        this.ssim = ssim;
    }

    @Override
    public void clear() {
        super.clear();
        psnr = 0;
        ssim = 0;
    }

    @Override
    public String toString() {
        String result = "";
        result = result + "FPS: " + super.getFPS()
                + "\nInference Time: " + super.getInferenceTime() + "s\n:"
                + "\nPSNR:" + getPnsr() + "\nSSIM:" + getSsim();
        return result;
    }
}
