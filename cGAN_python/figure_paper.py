import numpy as np
from matplotlib import pyplot as plt

nm_list = np.load("nmse.npy")

fig = plt.figure(0, figsize = (5, 5))
ep = list(range(1, 25 + 1))
plt.plot(range(1, 25 + 1), nm_list[0], "^-")
plt.plot(range(1, 25 + 1), nm_list[1], "x--")

plt.grid(1)
plt.xlim([1, 25])
plt.xlabel("Epoch")
plt.ylabel("NMSE (dB)")
plt.legend(["3GPP 3D SCM, beta=0.8, eta=0.0011",
            "Proposed in [5], beta=0.5, eta=1e-10"])
plt.title("GAN Training NMSE Score Comparison in dB Scale")

for x, y in zip(ep, nm_list[0]):
    if (x > 9):
        plt.text(x, y + 0.5, "%.3f" % y,  # 좌표 (x축 = v, y축 = y[0]..y[1], 표시 = y[0]..y[1])
                 fontsize=9,
                 color='black',
                 horizontalalignment='center',  # horizontalalignment (left, center, right)
                 verticalalignment='bottom',
                 rotation=90)  # verticalalignment (top, center, bottom)
plt.show()

import pandas as pd
nm = np.array(nm_list).T
A = pd.DataFrame(nm)
A.to_excel("nmse.xlsx")
