
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("siamese_net-master/siamese-net/output.csv", header=None)

Original = df.loc[df[1] == 'OriginalPair', 2]
questioned = df.loc[df[1] == 'QuestionedPair', 2]
y = [(Original),(questioned)]
x =[1,2]

for xe, ye in zip(x, y):
    plt.scatter([xe] * len(ye), ye,s=1)

plt.xticks([1, 2])
plt.axes().set_xticklabels(["Original Pair","Questioned Pair"])

plt.show()