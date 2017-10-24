import pprint, pickle
import pandas as pd
import numpy as np
class Data_saver():
    def __init__(self, obj, file_url):
        self.obj = obj
        self.file_url = file_url
    def save(self):
        output = open(self.file_url, 'wb')
        pickle.dump(self.obj, output)
        output.close()
    def load(self):
        pkl_file = open(self.file_url, 'rb')
        data = pickle.load(pkl_file)
        pkl_file.close()
        return data
file_url = '/Users/zhangzhihao/Documents/webbrain/data/pkl_saver/feature.pkl'
data = pd.read_csv('/Users/zhangzhihao/Documents/webbrain/data/features/feature_matrix.csv')
saver = Data_saver(data, file_url)
saver.save()
print(saver.load())


