'''
Date: 2021-05-04 16:37:49
Author: ZHR
Description: please input the description
LastEditTime: 2021-05-05 23:46:33
'''
def txtread(filepath):
    with open(filepath, 'r') as fr:
        file_data = [inst.strip().replace('\n', '').split(' ') for inst in fr.readlines()]
            # print(file_data)
    print(filepath, '文件读取成功')
    return file_data


def txtsave(filename, data):  # filename为写入txt文件的路径，data为要写入数据列表.
    file = open(filename, 'w')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '').replace(',', '\t')  # 去除[]
        s = s.replace("'", '').replace(',', '').replace('.', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print(filename, "保存文件成功")


file_data = txtread('task3/task3_testlabels1.txt')
txtsave('task3/task3_test_labels1.txt', file_data)