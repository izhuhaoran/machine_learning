'''
Date: 2021-04-30 20:53:53
Author: ZHR
Description: please input the description
LastEditTime: 2021-05-06 14:37:41
'''
import numpy as np

def List_print(List):
    for i in range(len(List)):
        if i == (len(List) - 1):
            print(List[i])
        else:
            print(List[i], end='\t')

rate_list = [[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30] for i in range(8)]
sigmaList = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
c = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
# List_print(sigmaList)

print('Training Accuracy:\n' + 'sigma', end='\t')
List_print(sigmaList)
print('c')
for i in range(len(rate_list)):
    print(c[i], end='\t')
    List_print(rate_list[i])

a = [[1.0], [0.0], [1.0], [0.0]]
b = [[1.0], [0.0], [0.0], [0.0]]

print(np.sum(abs(np.array(a) - np.array(b))))

# c = np.insert(a, 0, b, axis=0)

# a = 1.0
# print(round(a, 3))
# a = ('%.2f' % a)
# print(a + 'ds')

# print(c)