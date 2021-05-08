# -*- coding: utf-8 -*-
'''
Date: 2021-05-04 16:31:52
Author: ZHR
Description: please input the description
LastEditTime: 2021-05-06 23:59:18
'''
import SVM_Functions as s
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np


def txtsave(filename, data):  # filename为写入txt文件的路径，data为要写入数据列表.
    file = open(filename, 'w')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '').replace(',', '\t')  # 去除[]
        s = s.replace("'", '').replace(',', '').replace(',', '\t') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print(filename, "保存文件成功")


def predict_sklearn(x, y, x_test):
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.2)
    svm_clf = svm.LinearSVC()
    svm_clf.fit(x_train, y_train)
    acc = svm_clf.score(x_validation, y_validation)
    print('验证集精度为: %.4f' % acc)
    
    # svm_clf.fit(x, y)
    dec = svm_clf.predict(x_test)
    np.savetxt('task3.txt', dec, fmt="%d")


def predict_rate_valid(x, y, x_test):
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.2)
    model = s.svmTrain_SMO(x_train, y_train, C=1, tol=1e-3, max_iter=20)

    pre_train = s.svmPredict(model, x_train)
    pre_validation = s.svmPredict(model, x_validation)
    y_test = s.svmPredict(model, x_test)

    rate1 = 1 - (np.sum(abs(np.array(pre_train) - np.array(y_train))) / float(len(y_train)))
    rate2 = 1 - (np.sum(abs(np.array(pre_validation) - np.array(y_validation))) / float(len(y_validation)))
    # rate1 = ('%.4f' % rate1)
    # rate2 = ('%.4f' % rate2)
    print('用训练集20%%划分为验证集, 训练集模型准确度为：%.4f, 验证集准确度为：%.4f' % (rate1, rate2))

    txtsave('task3/task3_testlabels2.txt', y_test)
    return rate1, rate2


def predict_rate(x, y, x_test):
    model = s.svmTrain_SMO(x, y, C=1, tol=1e-3, max_iter=20)
    pre_train = s.svmPredict(model, x)
    y_test = s.svmPredict(model, x_test)

    rate = 1 - (np.sum(abs(np.array(pre_train) - np.array(y))) / float(len(y)))
    # rate = ('%.4f' % rate)
    print('训练集精度为：%.4f' % rate)

    txtsave('task3/task3_testlabels1.txt', y_test)
    return rate


if __name__ == '__main__':
    xTrain, yTrain = s.loadData('task3/task3_train.mat')
    xTest = s.loadData('task3/task3_test.mat')
    # train_len = len(xTrain)
    # train_feat_len = len(xTrain[0])
    print('训练集样本数为：%d, 特征维度为：%d' % (len(xTrain), len(xTrain[0])))
    print('测试集样本数为：%d, 特征维度为：%d' % (len(xTest), len(xTest[0])))
    # rate = predict_rate(xTrain, yTrain, xTest)
    # rate1, rate2 = predict_rate_valid(xTrain, yTrain, xTest)
    predict_sklearn(xTrain, yTrain, xTest)
