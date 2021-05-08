'''
Date: 2021-04-25 19:48:10
Author: ZHR
Description: please input the description
LastEditTime: 2021-04-25 20:53:41
'''
import numpy as np

def sigmoid(z):
    '''
    @description: sigmoid函数，y=1/√(1+e^-x)
    @param {*} z
    '''
    return 1 / (1 + np.exp(-z))


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


def stoc_grad_ascent(data_matrix, class_labels, iter_num=200):
    m, n = np.shape(data_matrix)
    alpha = 0.01
    weights = np.ones(n)
    for iteration in range(iter_num):
        # print("==> iteration %d <==" % iteration)
        for i in range(m):
            h = sigmoid(np.sum(data_matrix[i] * weights))
            error = class_labels[i] - h
            weights = weights + alpha * error * data_matrix[i]
        # print("weights: ", weights)
    return weights


def classify_vector(in_x, weights):
    prob = sigmoid(np.sum(in_x * weights))
    if prob > 0.5:
        return 1
    else:
        return 0


def colic_test():
    file_train = open('Task1\\horseColicTraining.txt')
    file_test = open('Task1\\horseColicTest.txt')
    training_set = []
    training_labels = []
    for line in file_train.readlines():
        curr_line = line.strip().split('\t')
        line_array = []
        for i in range(len(curr_line)):
            line_array.append(float(curr_line[i]))
        training_set.append(line_array)
        training_labels.append(float(curr_line[-1]))
    train_weights = stoc_grad_ascent_one(np.array(training_set), training_labels, 500)
    error_count = 0
    test_vec_num = 0.0
    for line in file_test.readlines():
        test_vec_num += 1.0
        curr_line = line.strip().split('\t')
        line_array = []
        for i in range(len(curr_line)):
            line_array.append(float(curr_line[i]))
        if int(classify_vector(np.array(line_array), train_weights)) != int(curr_line[-1]):
            error_count += 1
    error_rate = error_count / test_vec_num
    print("The error rate of this test is: %f" % error_rate)
    return error_rate


if __name__ == '__main__':
    colic_test()
