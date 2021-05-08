# -*- coding: utf-8 -*-
'''
Date: 2021-04-30 16:22:30
Author: ZHR
Description: please input the description
LastEditTime: 2021-05-06 11:06:27
'''
import SVM_Functions as s
import numpy as np

def predict_rate(sigmaList, CList):
    # error_num = 0
    index = 0
    X_gauss, y_gauss = s.loadData("task2/task2.mat")
    rate_list = [[] for i in range(len(CList))]
    # s.plotData(X_gauss, y_gauss, title="gaussian图像")
    for c in CList:
        for Sigma in sigmaList:
            model = s.svmTrain_SMO(X_gauss, y_gauss, C=c, kernelFunction='gaussian', K_matrix=s.gaussianKernel(X_gauss, sigma=Sigma))
            pre = s.svmPredict(model, X_gauss, Sigma)
            rate = 1 - (np.sum(abs(np.array(pre) - np.array(y_gauss))) / float(len(y_gauss)))
            rate_list[index].append(('%.3f' % rate))
        # print(rate_list[index])
        index += 1
    return rate_list
    # s.visualizeBoundaryGaussian(X_gauss, y_gauss, model2, sigma=0.1, title='gaussian_SVM边界图')


def List_print(List):
    for i in range(len(List)):
        if i == (len(List) - 1):
            print(List[i])
        else:
            print(List[i], end='\t')


if __name__ == "__main__":
    # X_gauss, y_gauss = s.loadData("task2/task2.mat")
    # s.plotData(X_gauss, y_gauss, title="gaussian图像")
    # model2 = s.svmTrain_SMO(X_gauss, y_gauss, C=1, kernelFunction='gaussian', K_matrix=s.gaussianKernel(X_gauss, sigma=0.1))
    # s.visualizeBoundaryGaussian(X_gauss, y_gauss, model2, sigma=0.1, title='gaussian_SVM边界图')

    sigmaList = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    CList = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    rate_list = predict_rate(sigmaList, CList)
    print('Training Accuracy:\n' + 'sigma', end='\t')
    List_print(sigmaList)
    print('c')
    for i in range(len(rate_list)):
        print(CList[i], end='\t')
        List_print(rate_list[i])
