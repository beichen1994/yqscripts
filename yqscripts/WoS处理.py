#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
import pandas as pd
test = pd.read_csv('test.txt',sep='\t',encoding='utf-16le',index_col=False,quoting=csv.QUOTE_NONE)
test

