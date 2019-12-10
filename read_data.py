from scipy.io import arff
import pandas as pd
import numpy as np
import torch

# data[0] 数据
# data[1] 特征名

def readData(name):
    data = arff.loadarff(name)
    df = pd.DataFrame(data[0])

    x = df.drop(['result'], axis=1)
    x = np.array(x, dtype='float32')
    x = torch.from_numpy(x)

    y = df['result']
    y = np.array(y, dtype='int64')
    # y[y > 0] = 1
    y = torch.from_numpy(y)

    return x,y

