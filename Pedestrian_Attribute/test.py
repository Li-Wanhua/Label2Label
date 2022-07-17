import torch
import numpy as np
from tools.function import get_pedestrian_metrics
a = torch.load("/home/zhexuan_wh/model/result.pkl")
pt = a['pt']
gt = a['gt']
res,ia = get_pedestrian_metrics(gt, pt)
lpr = res['label_pos_recall']
pt = np.where(pt>0.5,1,0)
l = np.zeros(35)
gt = gt.sum(axis = 1)
for i in range(1,35):
    index = np.where(gt == i)
    if len(index[0]) > 0:
        l[i] = ia[index].mean()
print(l)
