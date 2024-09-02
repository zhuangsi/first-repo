import numpy as np


def read_data(filename):
    with open(filename, 'r', encoding='GBK') as file:
        lines = file.readlines()

    data = []
    for line in lines:
        line = ' '.join(line.split()).replace(';', ' ')
        seq = line.split()
        if len(seq) == 4:
            data.append(seq)

    data = np.array(data, dtype=float)
    max_node = int(np.max(data[:, :2]))

    M_f = np.zeros((max_node, max_node))
    M_d = np.zeros((max_node, max_node))

    for row in data:
        i, j, flow, dist = int(row[0])-1, int(row[1])-1, row[2], row[3]
        M_f[i, j] = flow
        M_d[i, j] = dist

    return M_f, M_d


M_f, M_d = read_data('data/data1.txt')

print(M_f)
print(M_d)

np.save('data/M_f1.npy', M_f)
np.save('data/M_d1.npy', M_d)
