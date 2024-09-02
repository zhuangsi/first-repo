import numpy as np
import re


def read_data(filename):
    with open(filename, 'r', encoding='GBK') as file:
        lines = file.readlines()

    for i in range(len(lines)):
        lines[i] = lines[i].replace('{', '[').replace('}', ']').replace('\n', '')

    coordinates_text = ''.join(lines[1: 18])
    coordinates = np.array(eval('[' + coordinates_text))
    num_nodes = len(coordinates)

    M_f = np.zeros((num_nodes, num_nodes))
    M_d = np.zeros((num_nodes, num_nodes))

    flows_text = ''.join(lines).replace('\\', '')
    flow_parts = re.split(r'"i=\d+', flows_text)[1:]
    i = 0
    for line in flow_parts:
        M_f[i, :] = np.array(eval(line))
        i += 1

    # 计算距离矩阵
    for i in range(num_nodes):
        for j in range(num_nodes):
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[j]
            M_d[i, j] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    return M_f, M_d, coordinates


# 使用示例
M_f, M_d, coordinates = read_data('data/data2.txt')

print(coordinates)
print(M_f)
print(M_d)

# 保存为numpy数组文件
np.save('data/coordinates2.npy', coordinates)
np.save('data/M_f2.npy', M_f)
np.save('data/M_d2.npy', M_d)
