import os
import pandas as pd
import re

#匹配list中的所有非数字字母字符，即标点符号
file_path = '/data/Nutstore/Temp/test.txt'

f1 = open(file_path,encoding='utf-8')
lines = f1.readlines()

separator = list(set([re.search(r"\W",l.strip('\n').strip('.')).group(0)  for l in lines if re.search(r"\W",l.strip('\n').strip('.')) != None]))
separator

len(lines)
authors = []
for l in lines:
    l = l.strip('\n')
    for i in separator:
        l = l.replace(i,';')
    l = l.strip(';')
    l = l.replace(';;',';')
    authors.append(l)

df = pd.DataFrame({'authors':authors})
df.to_csv('authors_liu.csv',encoding='utf-8',index=False)
