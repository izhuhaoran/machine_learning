'''
Date: 2021-04-25 19:48:53
Author: ZHR
Description: please input the description
LastEditTime: 2021-04-26 20:06:13
'''
import pickle
# import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from scipy.sparse import csr_matrix

from sklearn.model_selection import train_test_split


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
        file_data = [inst.strip() for inst in fr.readlines()]
    print(filepath, '文件读取成功')
    return file_data


def DataRead(filename):
    with open(filename, 'rb') as f:
        data_texts = pickle.load(f)
    print(filename, '读取完成')
    return data_texts


def DataProcess():
    train_texts = DataRead('Task2/train/train_texts.dat')
    test_texts = DataRead('Task2/test/test_texts.dat')
    train_labels = txtread('Task2/train/train_labels.txt')
    print('数据读取完毕')
    return train_texts, test_texts, train_labels


def network_clf(train_texts, test_texts, train_labels):
    Vectorizer = TfidfVectorizer(max_features=10000)
    Vectors_train = Vectorizer.fit_transform(train_texts)
    train_features = csr_matrix(Vectors_train).toarray()
    print("训练集TF_IDF特征提取完毕")

    Vectors_test = Vectorizer.transform(test_texts)
    test_features = csr_matrix(Vectors_test).toarray()
    print("测试集TF_IDF特征提取完毕")

    X_train, X_train_test, Y_train, Y_train_test = train_test_split(train_features, train_labels, test_size=0.2,)
    print('训练集和验证集划分完毕')
    # clf = MLPClassifier()
    clf = MLPClassifier(hidden_layer_sizes=(100,),
                        activation="relu",
                        solver='adam',
                        alpha=0.0001,
                        batch_size='auto',
                        learning_rate="constant",
                        learning_rate_init=0.001, )
    clf.fit(X_train, Y_train)
    print('模型训练完毕')

    train_accuracy = clf.score(X_train_test, Y_train_test)
    print('分类器在训练集的精度为：', train_accuracy)

    test_pre = clf.predict(test_features)
    print('测试集预测标签完毕')
    txtsave('Task2/test/test_labels.txt', test_pre)

    return train_accuracy


def main():
    train_texts, test_texts, train_labels = DataProcess()
    network_clf(train_texts, test_texts, train_labels)


if __name__ == "__main__":
    main()
