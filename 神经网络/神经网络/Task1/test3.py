'''
Date: 2021-04-25 19:48:36
Author: ZHR
Description: 使用Logistic回归估计马疝病的死亡率
LastEditTime: 2021-04-26 14:43:21
'''
import numpy as np


def sigmoid(z):
    '''
    @description: sigmoid函数，y=1/√(1+e^-x)
    @param {*} z
    '''
    return 1 / (1 + np.exp(-z))


# 定义分类函数，prob>0.5，则分入1，否则分类0
def classifyVector(inX, trainWeights):
    prob = sigmoid(sum(inX * trainWeights))
    if prob > 0.5:
        return 1
    else:
        return 0


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
            alpha = 4 / (1 + i + j) + 0.01  # 保证多次迭代后新数据仍然有影响力
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatIn[i] * weights))  # 数值计算
            error = classLabels[randIndex] - h
            weights = weights + alpha * dataMatIn[randIndex] * error
            del (dataIndex[randIndex])
    return weights


def colicTest():
    '''
    @description: 马疝病预测函数
    @param {*}
    @return {*} errorRate预测错误的概率
    '''
    traindata = open('Task1/horseColicTraining.txt')  # 训练数据
    testdata = open('Task1/horseColicTest.txt')  # 测试数据
    trainSet = []
    trainLabels = []
    for line in traindata.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainSet.append(lineArr)
        trainLabels.append(float(currLine[-1]))
    trainWeights = stoc_grad_ascent_one(np.array(trainSet), trainLabels, 500)  # 改进的随机梯度上升法

    errorCount = 0
    numTest = 0
    for line in testdata.readlines():
        numTest += 1
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if classifyVector(np.array(lineArr), trainWeights) != int(currLine[-1]):
            errorCount += 1
    errorRate = (float(errorCount) / numTest)
    print('预测错误概率为:%f' % errorRate)
    return errorRate


if __name__ == "__main__":
    colicTest()
