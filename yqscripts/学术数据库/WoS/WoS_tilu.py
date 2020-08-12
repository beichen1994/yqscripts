#%%
import os
import pandas as pd

#%%
fields = []
with open('tilu_fields','r',encoding='utf-8') as f:
    while True:
        line = f.readline()
        if not line:
            break
        line = line.replace('\n','')
        fields.append(line)

# %%
fields_abbreviation=[]
fields_Chinese=[]
for i in range(len(fields)):
    if i%2==0:
        fields_abbreviation.append(fields[i])
    else:
        fields_Chinese.append(fields[i])

# %%
Chinese=[]
for col in cols:
    ind = fields_abbreviation.index(col)
    Chinese.append(fields_Chinese[ind])

# %%
tilu = []
tilu.append(Chinese)
cols=[]
for file in files:
    with open(os.path.join(data_path,file),'r',encoding='utf-8',errors='ignore') as f:
        lines = f.readlines()
        print(len(lines))
    cols=lines[0].strip('\n').split('\t')
    for i in range(1,len(lines)):
        if lines[i] != '\n':
            line = lines[i].split('\t')[:-1]
            tilu.append(line)
# %%
tilu_df = pd.DataFrame(tilu,columns=cols)


# %%
