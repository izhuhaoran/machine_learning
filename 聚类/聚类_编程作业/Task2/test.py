'''
Date: 2021-05-13 08:17:53
Author: ZHR
Description: please input the description
LastEditTime: 2021-05-14 15:41:44
'''
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
# from tqdm import tqdm
# from tqdm._tqdm import trange


def data_Get(data_Path):
    '''
    @description: 使用pandas读取csv数据文件
    @param {*}
        data_Path: 数据文件路径
    @return {*}
        traincsv: csv整体数据  BSSID_List: BBSID信息去重list
        fin_List: 采集时间fin去重List  Room_List: 标签Room数据list
    '''
    traincsv = pd.read_csv(data_Path, usecols=['BSSIDLabel', 'RSSLabel', 'RoomLabel', 'finLabel'])

    BSSID_List = traincsv['BSSIDLabel'].tolist()
    fin_List = traincsv['finLabel'].tolist()
    # RSS_List = traincsv['RSSLabel'].tolist()
    Room_List = list(traincsv['RoomLabel'].tolist())
    BSSID_List = list(set(BSSID_List))
    fin_List = list(set(fin_List))
    # RSS_List = list(set(RSS_List))

    return traincsv, BSSID_List, fin_List, Room_List


def data_trans(data_Path):
    datacsv, BSSID_List, fin_List, Room_List = data_Get(data_Path)
    train_data = [[0 for i in range(len(BSSID_List))] for i in range(len(fin_List))]
    # for i in tqdm(range(len(datacsv))):
    for i in range(len(datacsv)):
        times = datacsv['finLabel'][i]
        bssid = datacsv['BSSIDLabel'][i]
        rss = datacsv['RSSLabel'][i]
        # room = datacsv['RoomLabel'][i]
        try:
            index = BSSID_List.index(bssid)
            train_data[times - 1][index] = rss
            # train_data[times - 1][-1] = room
        except Exception:
            print("ERROR: unexpected BSSID")
    print(data_Path, "数据处理完毕")
    return np.mat(train_data), Room_List


if __name__ == "__main__":
    datapath1 = "Task2/DataSetKMeans1.csv"
    datapath2 = "Task2/DataSetKMeans2.csv"
    X_train, Y_train = data_trans(datapath1)
    # X_train, Y_train = data_trans(datapath2)

    numClust = len(set(Y_train))
    Y_pred = KMeans(n_clusters=numClust, random_state=0).fit_predict(X_train)
    print("聚类结果Y_pred：\n", Y_pred)
    dbi_score = davies_bouldin_score(X_train, Y_pred)
    print("Kmeans聚类结果的DBI指数为：%f" % dbi_score)

    mds = MDS(n_components=2)
    X_transformed = np.mat(mds.fit_transform(X_train))

    # 定义画布，背景
    fig = plt.figure("wifi信息聚类可视化")
    rect = [0.0, 0.0, 1.0, 1.0]
    # 不同图形标识
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    # 采用不同图形标识不同簇
    for i in range(numClust):
        ptsInCurrCluster = X_transformed[np.nonzero(Y_pred[:] == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle, s=90)
    plt.show()

    # print(Y_pred)
