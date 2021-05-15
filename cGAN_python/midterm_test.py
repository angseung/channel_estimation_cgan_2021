import numpy as np

M = np.array([
    [1, 3, 0, 4],
    [1, 5, 0, 1],
    [0, 8, 1, 5],
    [0, 2, 1, 3]
])

cat = np.array([
    [0, 4, 2],
    [1, 1, 1],
    [0, 1, 6]
])

mouse = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])

cow = np.array([
    [1, 3, 2],
    [3, 0, 0],
    [0, 2, 1]
])

snake = np.array([
    [0, 0, 1],
    [7, 0, 2],
    [1, 3, 1]
])

filt_list = [cat, mouse, cow, snake]
ans_list = [np.zeros([2, 2]), np.zeros([2, 2]), np.zeros([2, 2]), np.zeros([2, 2])]

for ind, filt in enumerate(filt_list):
    for i in [0, 1]: # row
        for j in [0, 1]: # col

            val = filt * M[i : i + 3, j : j + 3];
            val = val.sum(axis = None)
            ans_list[ind][i, j] = val
            print(val)