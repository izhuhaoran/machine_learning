# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=12)

Class = ['p', 'n', 'n', 'p', 'p', 'n', 'n', 'p', 'p', 'n']
# 真正标签 P表示正例；N表示反例
Score = [0.93, 0.85, 0.80, 0.7, 0.55, 0.50, 0.40, 0.3, 0.2, 0.1]
# 预测为正样本的概率
TP = [0] * 10
FP = [0] * 10
FN = [0] * 10
TN = [0] * 10
TPR = [0.] * 10  # TPR = TP / (TP+FN)
FPR = [0.] * 10  # FPR = FP / (TN+FP)
AUC = 0.0

for i in range(10):
    threshold = Score[i]  # 设置分类阈值为当前score
    for score, clas in zip(Score, Class):
        if (score > threshold):  # score > threshold 预测结果为正例
            if (clas == 'p'):
                TP[i] = TP[i] + 1  # 真实标签为正p，为真正例TP
            else:
                FP[i] = FP[i] + 1  # 真是标签为反f，为假正例FP
        else:  # score <= threshold 预测结果为反例
            if (clas == 'p'):
                FN[i] = FN[i] + 1  # 真实标签为正p，为假反例FN
            else:
                TN[i] = TN[i] + 1  # 真是标签为反f，为真反例TN

for i in range(10):
    TPR[i] = TP[i] / (TP[i] + FN[i])  # TPR = TP / (TP+FN)  y坐标
    FPR[i] = FP[i] / (TN[i] + FP[i])  # FPR = FP / (TN+FP)  x坐标
TPR.append(1.0)
FPR.append(1.0)
xy = list(zip(FPR, TPR))
sorted(xy, key=(lambda x: x[0]))  # xy数组按照FDR的值进行排序，方便计算AUC
for i in range(10):
    AUC = AUC + 0.5 * (xy[i + 1][0] - xy[i][0]) * (xy[i + 1][1] + xy[i][1])
    # 计算AUC
print('AUC的值为：', AUC)

plt.figure("ROC曲线")  # 定义一个图像窗口ROC曲线
plt.plot(FPR, TPR, color='r', markerfacecolor='blue', marker='o')  # 绘制ROC曲线
for locat in xy:  # 绘制每个点的坐标
    plt.text(locat[0], locat[1], (locat[0], locat[1]), ha='center', va='bottom', fontsize=10)
plt.title(u"ROC曲线", fontproperties=font)
plt.xlabel(u"FPR假正例率", fontproperties=font)
plt.ylabel(u"TPR真正例率", fontproperties=font)
plt.show()
print('程序结束')