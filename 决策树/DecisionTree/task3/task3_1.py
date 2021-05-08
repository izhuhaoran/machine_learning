import numpy as np
from sklearn import tree
import pydotplus
from six import StringIO
from sklearn.model_selection import train_test_split


def clearData(datafilename):
    data = []
    f = open(datafilename + ".txt", "r", encoding='utf-8')
    for line in f.readlines():
        line = line.split()
        for i, v in enumerate(line):
            line[i] = int(v)
        data.append(line)
    return data


def clearlabel(labelfilname):
    f = open(labelfilname + ".txt", "r", encoding='utf-8')
    label = []
    for line in f.readlines():
        label.append(line[:-1])
    return label


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))  # 数据集长度，每个评论维度10000
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1  # one-hot
    return results


def list2txt(testdata):
    with open('test_labels.txt', 'w') as txtfile:
        for temp in testdata:
            txtfile.write(str(temp) + '\n')


if __name__ == '__main__':
    train_data = clearData("task3/train_data")
    test_data = clearData("task3/test_data")
    train_label = clearlabel("task3/train_labels")
    train = vectorize_sequences(train_data)
    test = vectorize_sequences(test_data)
    clf = tree.DecisionTreeClassifier(max_depth=19)
    X_train, X_test, Y_train, Y_test = train_test_split(list(train), train_label, test_size=0.15)
    clf = clf.fit(X_train, Y_train)
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,  # 绘制决策树
                         class_names=clf.classes_,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("task3_tree.pdf")
    test_lable = clf.predict(list(test))
    list2txt(test_lable)
    print('file successfully')
    new_label = clf.predict(X_test)
    count = 0
    for ture_data, predicted_data in zip(Y_test, new_label):
        if ture_data == predicted_data:
            count += 1
    print(count / len(Y_test))
