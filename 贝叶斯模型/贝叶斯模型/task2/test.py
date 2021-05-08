'''
Date: 2021-04-13 15:23:16
Author: ZHR
Description: 利用sklearn的朴素贝叶斯分类搜狗新闻文本
LastEditTime: 2021-04-19 17:25:03
'''
# coding: utf-8
import os
# import time
import random
import jieba
# import sklearn
from sklearn.naive_bayes import MultinomialNB
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


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
    data_list = []
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

            word_cut = jieba.cut(raw, cut_all=False)  # 精确模式，返回的结构是一个可迭代的genertor
            # 精确模式，试图将句子最精确地切开，适合文本分析，占用内存小
            word_list = list(word_cut)  # genertor转化为list，每个词unicode格式
            data_list.append(word_list)
            class_list.append(folder.encode('utf-8').decode('utf-8'))  # 标签也即每个文件夹的名称
            j += 1

    # 划分训练集和测试集
    data_class_list = zip(data_list, class_list)
    data_class_list = list(data_class_list)
    random.shuffle(data_class_list)  # 随机打乱数据来划分训练集和测试集

    index = int(len(data_class_list) * test_size) + 1
    train_list = data_class_list[index:]
    test_list = data_class_list[:index]
    train_data_list, train_class_list = zip(*train_list)  # 拆分train_list为特征与标签
    test_data_list, test_class_list = zip(*test_list)

    # 统计词频放入all_words_dict
    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            # 判断是否存在于词袋中，如果没有，则出现次数为1，否则+1
            if word in all_words_dict:
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1

    # 根据键的值倒序排序
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f: f[1], reverse=True)  # 内建函数sorted参数需为list
    # 从排序后的结果中取出单词作为特征列表
    all_words_list = list(list(zip(*all_words_tuple_list))[0])

    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list


def words_dict(all_words_list, deleteN, stopwords_set=set()):
    """
    函数说明:文本特征选取
    Parameters:
        all_words_list - 训练集所有文本列表
        deleteN - 删除词频最高的deleteN个词
        stopwords_set - 指定的结束语
    Returns:
        feature_words - 特征集
    """
    # 选取特征词
    feature_words = []
    n = 1
    for t in range(deleteN, len(all_words_list), 1):
        if n > 1000:  # feature_words词袋的维度1000
            break
        # 如果这个词不是数字，并且不是指定的结束语，并且单词长度大于1小于5，那么这个词就可以作为feature_word
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
            n += 1
    return feature_words


def TextFeatures(train_data_list, test_data_list, feature_words):
    """
    函数说明:根据feature_words将文本向量化，采用one-hot编码，生成向量集
    Parameters:
        train_data_list - 训练集
        test_data_list - 测试集
        feature_words - 特征集
    Returns:
        train_feature_list - 训练集向量化列表
        test_feature_list - 测试集向量化列表
    """
    def text_features(text, feature_words):
        text_words = set(text)
        # 若词存在于词袋，则为1，否则为0
        features = [1 if word in text_words else 0 for word in feature_words]
        return features

    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    return train_feature_list, test_feature_list


# def TextFeatures_tf(train_data_list, test_data_list, stopwords):
#     # cv = CountVectorizer()
#     # vecot_matrix = count_vector.fit_transform(train_data)
#     # 使用tf-idf把文本转为向量
#     tv = TfidfVectorizer(stop_words=stopwords, max_features=5000, lowercase=False)
#     train_feature_list = tv.fit(train_data_list)
#     test_feature_list = tv.fit(test_data_list)

#     return train_feature_list, test_feature_list


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
    test_accuracy = classifier.score(test_feature_list, test_class_list)
    # print('测试集测试结果为：', classifier.predict(test_feature_list))
    # print('\n测试集原始标签为：', test_class_list)

    return test_accuracy


def main():
    # 文本预处理
    folder_path = './task2/Database/SogouC/Sample'
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path, test_size=0.2)

    # 生成stopwords_set
    stopwords_file = './task2/stopwords_cn.txt'
    stopwords_set = MakeWordsSet(stopwords_file)

    # 文本特征提取和分类
    deleteN = 450
    feature_words = words_dict(all_words_list, deleteN, stopwords_set)
    train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words)
    # train_feature_list, test_feature_list = TextFeatures_tf(list(train_data_list), list(test_data_list), list(stopwords_set))

    # 生成朴素贝叶斯分类器并得出精度
    accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list)
    # print('\n朴素贝叶斯分类器分类新闻文本的精度为：', accuracy)
    return accuracy


if __name__ == '__main__':
    rate = [main() for i in range(10)]
    rate_average = np.mean(rate)
    print("共10次测试，朴素贝叶斯分类平均精度为：%f" % rate_average)
