# -*- coding: utf-8 -*-
'''
Date: 2021-04-29 17:09:34
Author: ZHR
Description: please input the description
LastEditTime: 2021-05-05 20:46:55
'''
import SVM_Functions as s

if __name__ == "__main__":
    X_lin, y_lin = s.loadData("task1/task1_linear.mat")
    s.plotData(X_lin, y_lin, title="linear图像")
    model1 = s.svmTrain_SMO(X_lin, y_lin, C=1, max_iter=20)
    s.visualizeBoundaryLinear(X_lin, y_lin, model1, title='linear_SVM边界图')

    X_gauss, y_gauss = s.loadData("task1/task1_gaussian.mat")
    s.plotData(X_gauss, y_gauss, title="gaussian图像")
    model2 = s.svmTrain_SMO(X_gauss, y_gauss, C=1, kernelFunction='gaussian', K_matrix=s.gaussianKernel(X_gauss, sigma=0.1))
    s.visualizeBoundaryGaussian(X_gauss, y_gauss, model2, sigma=0.1, title='gaussian_SVM边界图')
