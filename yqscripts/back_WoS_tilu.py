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

fields_Chinese
# %%
data_path = "/home/data/2industrial_sensor"
raw_data = os.path.join(data_path,'WoS_data')
files = os.listdir(raw_data)

# %%
tilu=[[i for i in fields_Chinese]]
tilu
#%%
#这里一定要看文件的编码是utf-8还是utf-16，要不然会出错
# 形成源数据表
cols=[]
for file in files:
    print(file)
    with open(os.path.join(raw_data,file),'r',encoding='utf-16',errors='ignore') as f:
        lines = f.readlines()
        #print(len(lines))
    cols=lines[0].strip('\n').split('	')
    for i in range(1,len(lines)):
        if lines[i] != '\n':
            line = lines[i].split('\t')[:-1]
            #print(len(line))
            tilu.append(line)

tilu_df = pd.DataFrame(tilu,columns=cols)

#%% 通讯作者地址
addr = list(tilu_df.RP.values)

countrys=[]
country = [i.split(', ')[-1] for i in addr]
for i in country:
    if 'USA' in i:
        countrys.append('USA')
    else:
        i = i.strip('."').strip('')
        countrys.append(i)
print(len(addr))
print(len(countrys))
tilu_df['country'] = countrys
#%%
# 出版年份
list(tilu_df.PY.values)

# %%
# 基金资助机构
[i.split(' [')[0] for i in tilu_df.loc[3122].FU.split('; ')]

fund = list(tilu_df.FU.values)
funds=[]
for i in fund:
    f_str=''
    for j in i.split('; '):
        f = j.split(' [')[0]
        f_str = f_str+f+';'
    funds.append(f_str)
funds
tilu_df['fund']=funds
# %%
# 关键词
tilu_df.DE
#%%
# 被引频次合计
tilu_df.Z9

#%%
#形成国家-年份-信息字典
my_country=['Peoples R China','USA','Russia','Japan','South Korea','Germany','England','France','India','Israel','Sweden','Finland','Switzerland']
my_year = [str(i) for i in range(2012,2020)]

country_year_pub={}
country_year_fund={}
country_year_cite={}
for co in my_country:
    country_df = tilu_df[tilu_df['country']==co]
    print("国家为")
    print(co)
    year_pub_list=[]
    year_fund_list=[]
    year_cite_list=[]
    for ye in my_year:
        country_year_df = country_df[country_df['PY'] == ye]   
        print("年份为")
        print(ye)
        print("文章数目为")
        print(len(country_year_df))
        year_pub_list.append(len(country_year_df))
 
        #基金
        funds_dict={}
        funds = country_year_df['fund']
        for fs in funds.values:
            fs_list = fs.split(';')
            for fs1 in fs_list:
                if fs1 not in funds_dict:
                    funds_dict[fs1] = 1
                else:
                    funds_dict[fs1]+=1
        print("基金种类有")
        print(len(funds_dict))
        year_fund_list.append(funds_dict)
    
        #被引
        cite_num=0
        cites=country_year_df['Z9']
        for ci in cites.values:
            cite_num+=int(ci)
        print("被引频次总共为")
        print(cite_num)
        year_cite_list.append(cite_num)
    country_year_pub[co] = year_pub_list
    country_year_fund[co] = year_fund_list
    country_year_cite[co] = year_cite_list
# %%
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] ='simhei'
#%matplotlib qt5 
# %% [markdown]
## 国家发文量
#%%
X =[i for i in range(len(my_country))]
pub_nums = []
for co in my_country:
    pub_num=0
    for pub in country_year_pub[co]:
        pub_num+=int(pub)
    pub_nums.append(pub_num)
print(pub_nums)

# 条图
plt.bar(X,pub_nums,label='Paper numbers',alpha=0.8)
for x,y in zip(X,pub_nums):
    plt.text(x,y+1,'%d' %y, ha='center',fontsize=10)

plt.legend()
plt.xticks([i for i in range(len(my_country))],labels=my_country,rotation=90)
plt.xlabel('Country')
plt.ylabel('paper numbers')
plt.show()
plt.savefig(os.path.join(data_path,'paper_numbers_of_country.png'))
# %% [markdown]
## 国家每年发文量变化
#%%
import matplotlib.pyplot as plt
#显示中文
from matplotlib import rcParams
rcParams['font.family'] ='simhei'
#%matplotlib qt5 

#折线图
plt.figure(figsize=(8,4)) #创建绘图对象  

plt.plot(my_year, country_year_pub[my_country[0]], label=my_country[0],color = 'black', linewidth = 1.0,marker = '.',linestyle = '--')
plt.plot(my_year, country_year_pub[my_country[1]], label=my_country[1],color = 'black', linewidth = 1.0,marker = '.',linestyle = '--')
plt.plot(my_year, country_year_pub[my_country[2]], label=my_country[2],color = 'black', linewidth = 1.0,marker = '.',linestyle = '--')
plt.plot(my_year, country_year_pub[my_country[3]], label=my_country[3],color = 'black', linewidth = 1.0,marker = '.',linestyle = '--')
plt.plot(my_year, country_year_pub[my_country[4]], label=my_country[4],color = 'black', linewidth = 1.0,marker = '.',linestyle = '--')
plt.plot(my_year, country_year_pub[my_country[5]], label=my_country[5],color = 'black', linewidth = 1.0,marker = '.',linestyle = '--')
plt.plot(my_year, country_year_pub[my_country[6]], label=my_country[6],color = 'black', linewidth = 1.0,marker = '.',linestyle = '--')
plt.plot(my_year, country_year_pub[my_country[7]], label=my_country[7],color = 'black', linewidth = 1.0,marker = '.',linestyle = '--')
plt.plot(my_year, country_year_pub[my_country[8]], label=my_country[8],color = 'black', linewidth = 1.0,marker = '.',linestyle = '--')
plt.plot(my_year, country_year_pub[my_country[9]], label=my_country[9],color = 'black', linewidth = 1.0,marker = '.',linestyle = '--')
plt.plot(my_year, country_year_pub[my_country[10]], label=my_country[10],color = 'black', linewidth = 1.0,marker = '.',linestyle = '--')
plt.plot(my_year, country_year_pub[my_country[11]], label=my_country[11],color = 'black', linewidth = 1.0,marker = '.',linestyle = '--')
plt.plot(my_year, country_year_pub[my_country[12]], label=my_country[12],color = 'black', linewidth = 1.0,marker = '.',linestyle = '--')

plt.legend(loc=1)

plt.title('国家发文数量变化')
plt.xlabel("年份") #X轴标签  
plt.ylabel("发文数量")  #Y轴标签 
plt.show() 
plt.savefig(os.path.join(data_path,'paper_numbers_of_country_year.png'))

# %%
