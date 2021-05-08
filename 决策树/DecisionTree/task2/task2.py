# -*- coding: UTF-8 -*-
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from six import StringIO
from sklearn import tree
import pandas as pd
import numpy as np
import pydotplus
import sys
import csv
from math import log


def csv_sort(file_path):
    df = pd.read_csv(file_path, usecols=['BSSIDLabel', 'RSSLabel', 'RoomLabel', 'finLabel'])
    df = df.sort_values('RSSLabel')
    df.to_csv('train_sorted.csv', encoding='utf-8-sig', index=None)

def BSSID_get(file_path):
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        usecolumn = [row[0] for row in reader]
        usecolumn = usecolumn[1:]
        BSSID_List = set(usecolumn)
        print(len(BSSID_List))
    return BSSID_List


def data_Get(dataset_Path):
    traincsv = pd.read_csv(
        dataset_Path,
        usecols=['BSSIDLabel', 'RSSLabel', 'RoomLabel', 'finLabel'])

    BSSID_List = traincsv['BSSIDLabel'].tolist()
    BSSID_List = set(BSSID_List)
    fin_List = traincsv['finLabel'].tolist()
    RSS_List = traincsv['RSSLabel'].tolist() 
    Room_List = traincsv['RoomLabel'].tolist() 
    fin_List = set(fin_List)
    RSS_List = set(RSS_List)
    # Room_List = set(Room_List)

    # print(traincsv['BSSIDLabel'][1])
    # print(len(fin_List))
    return traincsv, BSSID_List, fin_List


# 计算数据集的房间标签的香农熵
def calcShannonEnt(dataSet):
    i = 0
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet['RoomLabel']:     # 为所有可能分类创建字典
        currentLabel = featVec  # 取数据集的标签
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0           # 分类标签值初始化
        labelCounts[currentLabel] += 1  # 给标签赋值
        i += 1
    shannonEnt = 0.0                    # 熵初始化
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries       # 求得每个标签的概率     # L(Xi) = -log2P(Xi)
        shannonEnt -= prob * log(prob, 2)   # 以2为底求对数      # H = - Σi=1 n  P(Xi)*log2P(Xi)
        # 注意这里是-= 虽然是求和 但是求和值<0 所以这里-=
    return shannonEnt


# 计算数据集的房间标签的香农熵
def gain(dataSet, RSS_List):
    i = 0
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in RSS_List:     # 为所有可能分类创建字典
        currentLabel = featVec  # 取数据集的标签
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0           # 分类标签值初始化
        labelCounts[currentLabel] += 1  # 给标签赋值
        i += 1
    shannonEnt = 0.0                    # 熵初始化
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries       # 求得每个标签的概率     # L(Xi) = -log2P(Xi)
        shannonEnt -= prob * log(prob, 2)   # 以2为底求对数      # H = - Σi=1 n  P(Xi)*log2P(Xi)
        # 注意这里是-= 虽然是求和 但是求和值<0 所以这里-=
    return shannonEnt


def pd_data(dataset, BSSID_List, fin_List):
    i = 0
    dict_BSSID = {}
    List_Room = [0 for _ in range(len(fin_List))]
    for BSSID in BSSID_List:
        if(BSSID not in dict_BSSID):
            dict_BSSID[BSSID] = [0 for _ in range(len(fin_List))]
        else:
            pass
    for i in range(len(dataset)):
        times = dataset['finLabel'][i]
        bssid = dataset['BSSIDLabel'][i]
        room = dataset['RoomLabel'][i]
        if(bssid not in dict_BSSID):
            dict_BSSID[bssid] = [0 for _ in range(len(fin_List))]
            dict_BSSID[bssid][times-1] = 1

        else:
            dict_BSSID[bssid][times-1] = 1
            List_Room[times-1] = str(room)

    return dict_BSSID, List_Room
    



if __name__ == "__main__":
    file_path = 'task2\\TrainDT.csv'
    sortfile_path = 'train_sorted.csv'
    Labels = ['BSSIDLabel', 'RSSLabel', 'RoomLabel', 'tearRate']  #特征标签
    # csv_sort(file_path)
    # dataSet1, RSS_List1 = data_Get(sortfile_path)
    dataSet, BSSID_List, fin_List = data_Get(file_path)
    # a = calcShannonEnt(dataSet)
    # b = calcShannonEnt(dataSet)

    dict_BSSID, List_Room = pd_data(dataSet, BSSID_List, fin_List)

    lenses_pd = pd.DataFrame(dict_BSSID)  #生成pandas.DataFrame
    print(lenses_pd) 
    sys.stdout.flush()   

    le = LabelEncoder()  #创建LabelEncoder()对象，用于序列化
    for col in lenses_pd.columns:  #序列化
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    print(lenses_pd)                                                        #打印编码信息
    sys.stdout.flush()

    clf = tree.DecisionTreeClassifier(max_depth=4)  #创建DecisionTreeClassifier()类
    clf = clf.fit(lenses_pd.values.tolist(), List_Room)  #使用数据，构建决策树

    dot_data = StringIO()

    tree.export_graphviz(
        clf,
        out_file=dot_data,  #绘制决策树
        feature_names=lenses_pd.keys(),
        class_names=clf.classes_,
        filled=True,
        rounded=True,
        special_characters=True)
    # print(dot_data.getvalue())
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("task2\\result\\task2.pdf")  #保存绘制好的决策树，以PDF的形式存储。

    print('1')
