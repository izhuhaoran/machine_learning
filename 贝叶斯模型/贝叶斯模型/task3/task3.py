# coding: utf-8
'''
Date: 2021-04-15 17:07:27
Author: ZHR
Description: 朴素贝叶斯分类电影评论
LastEditTime: 2021-04-19 22:37:40
'''

# from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
# from sklearn.feature_extraction.text import TfidfVectorizer
# import numpy as np


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


def dataget(file_path1):
    '''
    @description: 处理读取的txt数据转化为数据向量，采用one-hot编码
    @param {*} file_path1
    @return {*} data_return
    '''
    data_read = txtread(file_path1)
    # 评论总共25000行，单词词集为10000，初始化训练集数据二维列表
    data_return = [[0 for i in range(10000)] for i in range(len(data_read))]
    for i in range(len(data_read)):
        for num in data_read[i]:
            data_return[i][int(num) - 1] = 1
    print(file_path1, "数据处理完毕")
    return data_return


def prior_probability(label_list):
    '''
    @description: 计算先验概率
    @param {*} label_list
    @return {*} prior
    '''
    # label_list = txtread(filepath)
    # 标签1为正面评论，0位负面评论，对标签列表求和/列表长度 = 先验概率(正面评论)
    prior = float(sum(label_list)) / len(label_list)
    return prior


def TextClassifier(train_feature_list, test_feature_list, train_class_list):
    '''
    @description: 使用sklearn中的MultinomailNB()生成朴素贝叶斯分类器,并预测测试集标签结果保存为txt
    @param {*} train_feature_list, test_feature_list, train_class_list, file_path
    @return {*}
    '''
    # prior = prior_probability(train_class_list)
    classifier = MultinomialNB().fit(train_feature_list, train_class_list)
    # class sklearn.naive_bayes.MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
    # alpha：浮点型可选参数，默认为1.0;  fit_prior：布尔型可选参数，默认为True。布尔参数fit_prior表示是否要考虑先验概率
    # class_prior：先验概率，可选参数，默认为None。
    train_accuracy = classifier.score(train_feature_list, train_class_list)
    print("分类器训练集拟合精度为：", train_accuracy)

    test_class_list = classifier.predict(test_feature_list)
    print("分类器测试集拟合精度为：", test_accuracy)
    return test_class_list


def main():
    file_path_traindata = "task3\\train\\train_data.txt"
    file_path_testdata = "task3\\test\\test_data.txt"
    file_path_trainlabel = "task3\\train\\train_labels.txt"
    file_path_result = "task3\\result\\"
    traindata_list = dataget(file_path_traindata)
    testdatas_list = dataget(file_path_testdata)
    trainlabel_list = txtread(file_path_trainlabel)

    test_class_list = TextClassifier(traindata_list, testdatas_list, trainlabel_list)

    txtsave(file_path_result + 'test_labels.txt', test_class_list)
    print('测试集测试结果保存完毕')


if __name__ == "__main__":
    # main()
    test_classlist = txtread('task3\\result\\labels.txt')
    test_list = txtread('task3\\result\\test_labels.txt')
    i = 0
    for m, n in zip(test_classlist, test_list):
        if(m != n):
            i += 1
    test_accuracy = float(i) / 25000
    print(test_accuracy)
