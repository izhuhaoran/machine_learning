# coding: utf-8
'''
Date: 2021-04-13 15:23:16
Author: ZHR
Description: 利用sklearn的朴素贝叶斯分类搜狗新闻文本
LastEditTime: 2021-04-19 17:19:37
'''

import os
# import time
import re
import random
import jieba
# import sklearn
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# import numpy as np


def MakeWordsSet(words_file):
    """
    函数说明:读取文件里的内容，并去重
    Parameters:
        words_file - 文件路径
    Returns:
        words_set - 读取的内容的set集合
    """
    words_set = set()
    with open(words_file, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            word = line.strip().encode('utf-8').decode("utf-8")
            if len(word) > 0 and word not in words_set:  # 去重
                words_set.add(word)
    return words_set


def TextProcessing(folder_path, test_size=0.2):
    """
    函数说明:中文文本处理
    Parameters:
        folder_path - 文本存放的路径
        test_size - 测试集占比，默认占所有数据集的百分之20
    Returns:
        all_words_list - 按词频降序排序的训练集列表
        train_data_list - 训练集列表
        test_data_list - 测试集列表
        train_class_list - 训练集标签列表
        test_class_list - 测试集标签列表
    """
    folder_list = os.listdir(folder_path)
    text_list = []
    class_list = []

    # 类间循环
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path, folder)
        files = os.listdir(new_folder_path)
        # 类内循环
        j = 1
        for file in files:
            if j > 100:  # 每类text样本数最多100
                break
            with open(os.path.join(new_folder_path, file), 'r', encoding='utf-8') as fp:
                raw = fp.read()
                text = re.sub(r'\W*', '', raw)  # 去除标点符号以及特殊字符

            word_cut = jieba.cut(text, cut_all=False)  # 精确模式，返回的结构是一个可迭代的genertor
            # 精确模式，试图将句子最精确地切开，适合文本分析，占用内存小
            # word_list = list(word_cut)  # genertor转化为list，每个词unicode格式
            data_text = ''
            for word in word_cut:
                data_text += word + " "
            text_list.append(data_text)
            # data_list.append(word_list)
            class_list.append(folder.encode('utf-8').decode('utf-8'))  # 标签也即每个文件夹的名称
            j += 1

    # 划分训练集和测试集
    text_class_list = zip(text_list, class_list)
    text_class_list = list(text_class_list)
    random.shuffle(text_class_list)  # 随机打乱数据来划分训练集和测试集

    index = int(len(text_class_list) * test_size) + 1
    train_list = text_class_list[index:]
    test_list = text_class_list[:index]
    train_data_list, train_class_list = zip(*train_list)  # 拆分train_list为特征与标签
    test_data_list, test_class_list = zip(*test_list)

    return train_data_list, test_data_list, train_class_list, test_class_list


def TextFeatures_tf(train_data_list, test_data_list, stopwords):
    # 使用tf-idf把文本转为向量
    tfidf_transformer = TfidfVectorizer(encoding='utf-8', stop_words=stopwords, max_features=1000, lowercase=False, max_df=0.5, sublinear_tf=True, smooth_idf=True)
    # tfidf_transformer = TfidfVectorizer(max_features=1000, lowercase=False)
    train_feature_list = tfidf_transformer.fit_transform(train_data_list)

    test_feature_list = tfidf_transformer.transform(test_data_list)

    return train_feature_list, test_feature_list


def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list):
    """
    函数说明:分类器
    Parameters:
        train_feature_list - 训练集向量化的特征文本
        test_feature_list - 测试集向量化的特征文本
        train_class_list - 训练集分类标签
        test_class_list - 测试集分类标签
    Returns:
        test_accuracy - 分类器精度
    """
    classifier = MultinomialNB().fit(train_feature_list, train_class_list)
    # class sklearn.naive_bayes.MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
    # alpha：浮点型可选参数，默认为1.0;  fit_prior：布尔型可选参数，默认为True。布尔参数fit_prior表示是否要考虑先验概率
    # class_prior：先验概率，可选参数，默认为None。

    test_pre = classifier.predict(test_feature_list)
    test_accuracy = metrics.accuracy_score(test_pre, test_class_list)

    train_accuracy = classifier.score(train_feature_list, train_class_list)
    # test_accuracy = classifier.score(test_feature_list, test_class_list)

    print('测试集测试结果为：', classifier.predict(test_feature_list))
    print('\n测试集原始标签为：', test_class_list)

    return train_accuracy, test_accuracy


def main():
    # 文本预处理
    folder_path = './task2/Database/SogouC/Sample'
    train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path, test_size=0.2)

    # 生成stopwords_set
    stopwords_file = './task2/stopwords_cn.txt'
    stopwords_set = MakeWordsSet(stopwords_file)

    # 文本特征提取和分类
    train_feature_list, test_feature_list = TextFeatures_tf(train_data_list, test_data_list, list(stopwords_set))

    # 生成朴素贝叶斯分类器并得出精度
    train_accuracy, test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list)
    print('\n朴素贝叶斯分类器分类新闻文本的精度为：训练集精度：%f  测试集精度：%f' % (train_accuracy, test_accuracy))
    return train_accuracy, test_accuracy


if __name__ == '__main__':
    main()
    # rate = [main() for i in range(10)]
    # rate_average = np.mean(rate)
    # print("共100次测试，朴素贝叶斯分类平均精度为：%f" % rate_average)
