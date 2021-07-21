import re
import pandas as pd

path = './data/sample_coco.txt'

lines = []
with open(path) as f:
    lines = f.readlines()

class my_dictionary(dict):
    def __init__(self):
        self = dict()
        
    def add(self, key, value):
        self[key] = value

def datarestructure(columns,dataList):
    ls = []
    dataDict = my_dictionary()
    for i in dataList:
        a = i.replace(',\n','')
        val = re.findall('[0-9]+', a)
        for cnt,dat in zip(columns,val):
            dataDict.add(cnt,dat)
        ls.append(dataDict.copy())
    return ls

columns = ['id', 'height', 'width', 'x', 'y', 'bbox_width', 'bbox_height']
data = pd.DataFrame(datarestructure(columns,dataList=lines))

data.to_csv('./data/coco.csv',index=False)