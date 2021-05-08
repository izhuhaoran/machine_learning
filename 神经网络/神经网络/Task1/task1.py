'''
Date: 2021-04-23 20:17:01
Author: ZHR
Description: 逻辑回归python实现
LastEditTime: 2021-04-27 09:18:06
'''
# import math
import numpy as np
# import pandas as pd


def sigmoid(z):
    '''
    @description: sigmoid函数，y=1/√(1+e^-x)
    @param {*} z
    '''
    return 1 / (1 + np.exp(-z))


def init_data(filepath):
    '''
    @description: 导入训练集数据，使用np.loat，分隔符为','
    @param {*}
    @return {*} dataMatIn特征数据, classLabels标签数据
    '''
    data = np.loadtxt(filepath, delimiter='\t')
    dataMatIn = data[:, 0:-1]
    classLabels = data[:, -1]
    dataMatIn = np.insert(dataMatIn, 0, 1, axis=1)  # 特征数据集，添加1是构造常数项x0
    return dataMatIn, classLabels


def grad_ascent(dataMatIn, classLabels):
    '''
    @description: 梯度上升法求权重
    @param {*} dataMatIn特征数据, classLabels标签数据
    @return {*} weights权重
    '''
    dataMatrix = np.mat(dataMatIn)  # (m,n)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    weights = np.ones((n, 1))  # 初始化回归系数（n, 1)
    alpha = 0.001  # 步长
    maxCycle = 500  # 最大循环次数

    for i in range(maxCycle):
        h = sigmoid(dataMatrix * weights)  # sigmoid 函数
        error = labelMat - h  # y-h, (m - 1)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


def stoc_grad_ascent(dataMatIn, classLabels):
    '''
    @description: 随机梯度上升法，思想：每次只使用一个数据样本点来更新回归系数。这样就大大减小计算开销
    @param {*} dataMatIn特征数据, classLabels标签数据
    @return {*} weights权重
    '''
    m, n = np.shape(dataMatIn)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatIn[i] * weights))  # 数值计算
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatIn[i]
    return weights


def stoc_grad_ascent_one(dataMatIn, classLabels, numIter=150):
    '''
    @description: 随机梯度上升的改进方法
    @param {*} dataMatIn特征数据, classLabels标签数据
    @return {*} weights权重
    '''
    m, n = np.shape(dataMatIn)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1 + i + j) + 0.0001  # 保证多次迭代后新数据仍然有影响力
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatIn[i] * weights))  # 数值计算
            error = classLabels[randIndex] - h
            weights = weights + alpha * dataMatIn[randIndex] * error
            del (dataIndex[randIndex])
    return weights


def classifyVector(inX, trainWeights):
    '''
    @description: 分类函数，prob>0.5，则分入1，否则分类0
    @param {*} inx传入的特征数据, trainWeights权重
    @return {*} 1，0
    '''
    prob = sigmoid(sum(inX * trainWeights))
    if prob > 0.5:
        return 1
    else:
        return 0


def Test_predict(weights, test_dataMatIn, test_classLabels):
    '''
    @description: 测试集预测函数
    @param {*}  weights权重
    '''
    error = 0
    for test_data, test_label in zip(test_dataMatIn, test_classLabels):
        label = classifyVector(test_data, weights)
        if label != test_label:
            error = error + 1
    error_rate = float(error) / len(test_classLabels)
    print('测试集预测结果错误率为：', error_rate)
    return error_rate


if __name__ == '__main__':
    train_dataMatIn, train_classLabels = init_data('Task1/horseColicTraining.txt')
    print("训练集数据导入完成")
    weights = stoc_grad_ascent_one(train_dataMatIn, train_classLabels)  # 计算权值
    print("回归系数W计算完成")
    test_dataMatIn, test_classLabels = init_data('Task1/horseColicTest.txt')
    print("训练集数据导入完成")
    Test_predict(weights, test_dataMatIn, test_classLabels)
    # rate_list = []
    # for i in range(10):
    #     rate_list.append(Test_predict(weights, test_dataMatIn, test_classLabels))  # 进行预测
    # rate_ave = np.mean(rate_list)
    # print('平均错误率为：', rate_ave)
