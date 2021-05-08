'''
Date: 2021-04-16 15:30:39
Author: ZHR
Description: please input the description
LastEditTime: 2021-04-16 16:14:58
'''
# -*- coding: utf-8 -*-
from file import txtread, txtsave
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from six import StringIO
from sklearn import tree
import pydotplus
import numpy as np

# def vectorize_sequences(sequences, dimension=10000):
#     results = np.zeros((len(sequences), dimension))
#     for i, sequence in enumerate(sequences):
#         results[i, sequence] = 1.  # 索引results矩阵中的位置，赋值为1，全部都是从第0行0列开始的
#     return results


def dataget(file_path1, file_path2):
    train_data = txtread(file_path1)
    label_data = txtread(file_path2)
    traindata = [[0 for i in range(10000)] for i in range(len(train_data))]
    for i in range(len(train_data)):
        for train_num in train_data[i]:
            traindata[i][int(train_num) - 1] = 1
    print("训练数据处理完毕")
    return traindata, label_data


def datatest(file_path):
    test_data = txtread(file_path)
    testdata = [[0 for i in range(10000)] for i in range(len(test_data))]
    for i in range(len(test_data)):
        for test_num in test_data[i]:
            testdata[i][int(test_num) - 1] = 1
    print("测试数据处理完毕")
    return testdata


def tree_func(traindata, label_data, testdata):
    count = 0
    clf = tree.DecisionTreeClassifier()
    x_train, x_test, y_train, y_test = train_test_split(list(traindata), label_data, test_size=0.2)
    clf = clf.fit(x_train, y_train)
    dot_data = StringIO()
    tree.export_graphviz(
        clf,
        out_file=dot_data,  # 绘制决策树
        class_names=clf.classes_,
        filled=True,
        rounded=True,
        special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("task3/result/task3_tree.pdf")
    print('决策树生成完毕')
    txtsave('task3/result/train_result.txt', clf.predict(list(traindata)))
    # print(clf.score(list(traindata), label_data))  # 决策树本身精度
    labels = clf.predict(x_test)
    for ture_data, predicted_data in zip(y_test, labels):
        if ture_data == predicted_data:
            count += 1
    print(count / len(y_test))
    txtsave('task3/result/test_result.txt', clf.predict(list(testdata)))
    print('测试集结果预测完毕')


if __name__ == "__main__":
    file_path1 = 'task3/train_data.txt'
    file_path2 = 'task3/train_labels.txt'
    file_path3 = 'task3/test_data.txt'
    # label_data = txtread(file_path2)
    # train_data = txtread(file_path1)
    # test_data = txtread(file_path3)
    # traindata = vectorize_sequences(train_data)
    # testdata = vectorize_sequences(test_data)

    traindata, label_data = dataget(file_path1, file_path2)
    testdata = datatest(file_path3)

    tree_func(traindata, label_data, testdata)
    print(1)
