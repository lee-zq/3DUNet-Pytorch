import pandas as pd
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import torch,random
import numpy as np
from collections import OrderedDict

class Logger():
    def __init__(self,save_name):
        self.log = None
        self.summary = None
        self.name = save_name

    def update(self,epoch,train_log,val_log):
        item = OrderedDict({'epoch':epoch})
        item.update(train_log)
        item.update(val_log)
        item = dict_round(item,4) # 保留小数点后四位有效数字
        print(item)
        self.update_csv(item)
        self.update_tensorboard(item)

    def update_csv(self,item):
        tmp = pd.DataFrame(item,index=[0])
        if self.log is not None:
            self.log = self.log.append(tmp, ignore_index=True)
        else:
            self.log = tmp
        self.log.to_csv('%s/log.csv' %self.name, index=False)

    def update_tensorboard(self,item):
        if self.summary is None:
            self.summary = SummaryWriter('%s/' % self.name)
        epoch = item['epoch']
        for key,value in item.items():
            if key != 'epoch': self.summary.add_scalar(key, value, epoch)

def setpu_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    random.seed(seed)

def dict_round(dic,num):
    for key,value in dic.items():
        dic[key] = round(value,num)
    return dic