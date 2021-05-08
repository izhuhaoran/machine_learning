'''
Date: 2021-04-29 13:46:57
Author: ZHR
Description: please input the description
LastEditTime: 2021-04-29 14:12:03
'''
import math


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


print(sigmoid(-0.037))