'''
Date: 2021-04-23 20:17:01
Author: ZHR
Description: please input the description
LastEditTime: 2021-04-25 19:34:38
'''
# import math
import numpy as np
import pandas as pd


def sigmoid(z):
    '''
    @description: sigmoid函数，y=1/√(1+e^-x)
    @param {*} z
    '''
    return 1 / (1 + np.exp(-z))


def init_data():
    '''
    @description: 导入训练集数据，使用np.loat，分隔符为','
    @param {*}
    @return {*} dataMatIn特征数据, classLabels标签数据
    '''
    data = np.loadtxt('钞票训练集.txt', delimiter=',')
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
            alpha = 4 / (1 + i + j) + 0.01  # 保证多次迭代后新数据仍然有影响力
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatIn[i] * weights))  # 数值计算
            error = classLabels[i] - h
            weights = weights + alpha * error * dataMatIn[i]
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


def Test_predict(weights):
    '''
    @description: 测试集预测函数
    @param {*}  weights权重
    '''
    testdata = []   # 用于存储测试集数据及结果保存为csv文件
    labelList = []  # 用于保存测试集预测结果
    frTest = open('钞票测试集.txt')  # 测试数据
    for line in frTest.readlines():
        currLine = line.strip().replace('\n', '').split(',')
        lineArr = []
        for i in range(5):
            if i == 0:
                lineArr.append(float(1))    # 第一位插入一个1，用于与W0相乘
            else:
                lineArr.append(float(currLine[i - 1]))
        label = classifyVector(np.array(lineArr), weights)
        labelList.append(label)
        lineArr.append(label)
        testdata.append(lineArr[1:])
    txtsave('test_label.txt', labelList)
    data_pd = pd.DataFrame(testdata, columns=['变量名1', '变量名2', '变量名3', '变量名4', '真钞or假钞'])
    data_pd.to_csv('测试集结果.csv', encoding='utf-8')


def txtsave(filename, data):
    '''
    @description: filename为写入txt文件的路径，data为写入的数据.
    @param {*} filename, data
    @return {*}
    '''
    file = open(filename, 'w')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '').replace(',', '\t')  # 去除[]
        s = s.replace("'", '').replace(',', '').replace(',', '\t') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print(filename, "保存文件成功")


def txtread(filepath):
    '''
    @description: txt文件读取，将文本按照空格分片成list，同时去除\n
    @param {*} filepath
    @return {*} file_data
    '''
    with open(filepath, 'r') as fr:
        file_data = [inst.strip().replace('\n', '').split(' ') for inst in fr.readlines()]
    print(filepath, '文件读取成功')
    return file_data


if __name__ == '__main__':
    dataMatIn, classLabels = init_data()
    print("训练集数据导入完成")
    weights = stoc_grad_ascent_one(dataMatIn, classLabels)  # 计算权值
    print("权值计算完成")
    Test_predict(weights)  # 进行预测
    # turelabel = txtread('true.txt')
    # test_label = txtread('test_label.txt')
    # i = 0 # 错误率计数
    # for m, n in zip(turelabel, test_label):
    #     if m != n:
    #         i += 1
    #     else:
    #         pass
    # rate = float(i) / len(turelabel)
    # print('错误率为：', rate)
