/*
 * Copyright (c) 2021 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.utils;

import android.util.Log;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class MathUtils {
    public final static String TAG = MathUtils.class.getSimpleName();
    public static double min(ArrayList<Double> nums){
        double minNum = Double.MAX_VALUE;
        for (double num : nums){
            if (num <= minNum){
                minNum = num;
            }
        }
        return minNum;
    }

    public static double max(ArrayList<Double> nums){
        double maxNum = 0;
        for (double num : nums){
            if (num >= maxNum){
                maxNum = num;
            }
        }
        return maxNum;
    }

    public static int minMatrix(int[][] num) {
        int min = Integer.MAX_VALUE;
        for(int i = 0; i < num.length; i++) {
            for(int j = 0; j < num[0].length; j++) {
                if(num[i][j] < min) {
                    min = num[i][j];
                }
            }
        }
        return min;
    }

    public static int maxMatrix(int[][] num) {
        int max = Integer.MIN_VALUE;
        for(int i = 0; i < num.length; i++) {
            for(int j = 0; j < num[0].length; j++) {
                if(num[i][j] > max) {
                    max = num[i][j];
                }
            }
        }
        return max;
    }

    public static double getAverage(List<Double> num) {
        if(num.size() == 0) {
            throw new NullPointerException();
        }
        double average = 0.0;
        for(int i = 0; i < num.size(); i++) {
            average += num.get(i);
        }
        return (double) average/num.size();
    }
    public static int[][] matrixReshape(float[] matrix, int height, int width) {
        if(matrix.length == 0 || matrix.length != height * width) {
            Log.e(TAG, "matrix length=" + matrix.length + ", height=" + height + ", width=" + width);
            return null;
        }
        int [][] reshapedMatrix = new int[height][width];

        for(int i = 0; i < height; i++) {
            for(int j = 0; j < width; j++) {
                reshapedMatrix[i][j] = (int)matrix[i*width + j];
            }
        }
        return reshapedMatrix;
    }
    public static int[][] matrixReshapeAndCut(float[] matrix, int cut_height, int cut_width, int reshape_height, int reshape_width) {
        if(matrix.length == 0 || matrix.length != reshape_height * reshape_width) {
            Log.e(TAG, "matrix length=" + matrix.length + ", reshape height=" + reshape_height + ", reshape width=" + reshape_width);
            return null;
        }
        int [][] cutMatrix = new int[cut_height][cut_width];

        for(int i = 0; i < reshape_height; i++) {
            for(int j = 0; j < reshape_width; j++) {
                if(i >= cut_height || j >= cut_width)
                    continue;
                cutMatrix[i][j] = (int)matrix[i*reshape_width + j];
            }
        }
        return cutMatrix;
    }

    public static  int[][] getConfusionMat(int[][] anno, int[][] pred, int[] labels,String[] conf) {
        int[][] cf = new int[labels.length][3];
        for(int i = 0; i < labels.length; i++) {
            for(int j = 0; j < conf.length; j++) {
                cf[i][j] = 0;
            }
        }
        Set<String> resPos = new HashSet<>();
        Set<String> annoPos = new HashSet<>();
        Set<String> TP = new HashSet<>();
        Set<String> FN = new HashSet<>();
        Set<String> FP = new HashSet<>();

        for(int label: labels) {
            resPos.clear();
            annoPos.clear();
            for(int i = 0; i < pred.length; i++) {
                for(int j = 0; j < pred[0].length; j++) {
                    if(pred[i][j] == label) {
                        resPos.add("("+ i +","+ j +")");
                    }
                }
            }
            for(int i = 0; i < anno.length; i++) {
                for(int j = 0; j < anno[0].length; j++) {
                    if(anno[i][j] == label) {
                        annoPos.add("("+ i +","+ j +")");
                    }
                }
            }
            TP.clear();
            TP.addAll(resPos);
            TP.retainAll(annoPos);

            FN.clear();
            FN.addAll(annoPos);
            FN.removeAll(TP);

            FP.clear();
            FP.addAll(resPos);
            FP.removeAll(TP);

            cf[label][0] = TP.size();
            cf[label][1] = FP.size();
            cf[label][2] = FN.size();
        }
        return cf;
    }
    public static double[] calAverages(double[]precision, double[]recall, int[] labels) {
        double[] result = new double[2];
        List<Integer> precT = new ArrayList<Integer>();
        List<Integer> recallT = new ArrayList<Integer>();
        List<Double> precFinal = new ArrayList<Double>();
        List<Double> recallFinal = new ArrayList<Double>();
        for(int i = 0; i < labels.length; i++) {
            if(Double.isNaN(precision[i])) {
                precT.add(i);
            }
            if(Double.isNaN(recall[i])) {
                recallT.add(i);
            }
        }
        Set tagSet = new HashSet();
        for(int i = 0; i < labels.length; i++) {
            tagSet.add(i);
        }
        Set precTSet = new HashSet(precT);
        Set recallTSet = new HashSet(recallT);
        precTSet.addAll(recallTSet);
        tagSet.removeAll(precTSet);
        List<Object> finalList = Arrays.asList(tagSet.toArray());
        List<Integer> notCal = new ArrayList<Integer>();
        for(Object i:precTSet.toArray()) {
            notCal.add(labels[(Integer) i]);
        }
        for(int i = 0; i < finalList.size(); i++) {
            precFinal.add(precision[(Integer) finalList.get(i)]);
            recallFinal.add(recall[(Integer) finalList.get(i)]);
        }
        result[0] = getAverage(precFinal);
        result[1] = getAverage(recallFinal);
        return result;

    }
    public static  double[]  calSegIndex(int[][] confMatrix, int[] labels) {
        double[] GlobalACC = new double[confMatrix.length];
        double[] IOU = new double[confMatrix.length];
        double[] Recall = new double[confMatrix.length];
        double[] Prec = new double[confMatrix.length];
        double[] F1Score = new double[confMatrix.length];
        int[] TP = new int[confMatrix.length];
        int[] FP = new int[confMatrix.length];
        int[] FN = new int[confMatrix.length];
        for(int i = 0; i < TP.length; i++) {
            TP[i] = confMatrix[i][0];
        }
        for(int i = 0; i < FP.length; i++) {
            FP[i] = confMatrix[i][1];
        }
        for(int i = 0; i < FN.length; i++) {
            FN[i] = confMatrix[i][2];
        }
        for(int i = 0; i < confMatrix.length; i++) {
            if(TP[i]+FN[i] == 0) {
                GlobalACC[i] = Double.NaN;
            }
            else {
                GlobalACC[i] = (double)TP[i]/(TP[i]+FN[i]);
            }
            if(FP[i]+FN[i]+TP[i] == 0){
                IOU[i] = Double.NaN;
            }
            else{
                IOU[i] = (double)TP[i]/(FP[i]+FN[i]+TP[i]);
            }
            if(TP[i]+FN[i] == 0) {
                Recall[i] = Double.NaN;
            }
            else{
                Recall[i] = (double)TP[i]/(TP[i]+FN[i]);
            }
            if(TP[i]+FP[i] == 0){
                Prec[i] = Double.NaN;
            }
            else{
                Prec[i] = (double)TP[i]/(TP[i]+FP[i]);
            }
            if(Recall[i]+Prec[i] == 0) {
                F1Score[i] = Double.NaN;
            }
            else{
                F1Score[i] = (2*Prec[i]*Recall[i])/(Recall[i]+Prec[i]);
            }

        }
        double[] result;
        result= MathUtils.calAverages(Prec, Recall, labels);
        double averagePrec = result[0];
        result = MathUtils.calAverages(F1Score, Recall, labels);
        double averageF1Score = result[0];
        result = MathUtils.calAverages(IOU, Recall, labels);
        double averageIOU = result[0];
        result = MathUtils.calAverages(GlobalACC, Recall, labels);
        double averageGlobalAcc = result[0];
        double averageRecall = result[1];
        result= new double[]{averageGlobalAcc, averageIOU,averageRecall,averagePrec,averageF1Score};
        return result;
    }
}