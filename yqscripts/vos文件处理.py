import os
import random
import math
import pandas as pd
import numpy as np


class RandomCircle(object):
    #设定圆的横纵坐标，半径，圆的点个数
    def __init__(self,x,y,r,num):
        self.x = x
        self.y = y
        self.r = r
        self.num = num
    # 根据圆的点的个数随机生成list,可以根据圆心角来规划圆形
    def random_list(self):
        n = self.num
        dp = [[[] for i in range(n+1)] for j in range(n+1)]
        for i in range(n+1):
            dp[i][i] = ["{}".format(i)]
        for i in range(1,n+1):
            for j in range(i-1, 0, -1):
                dp[i][j] = ["{},{}".format(j, xx) for xx in dp[i-j][j]] + dp[i][j+1]
        return random.choice(dp[n][1]).split(',')
    def corordinate(self):
        zuobiao_x = []
        zuobiao_y = []
        for N in self.random_list():
            N = int(N)
            radius = random.uniform(0.01,self.r)
            theta=2*math.pi*np.array([random.random() for _ in range(N)])
            x=radius*np.cos(theta)+self.x
            y=radius*np.sin(theta)+self.y
            for each in range(len(x)):
                zuobiao_x.append(str(x[each]))
                zuobiao_y.append(str(y[each]))
        f = open('random_circle.txt','w',encoding='utf-8')
        f.write('X'+'\t'+'Y'+'\n')
        for i in range(len(zuobiao_x)):
            f.write(str(zuobiao_x[i])+'\t'+str(zuobiao_y[i])+'\n')
        f.close()
    def main(self):
        self.corordinate()

if __name__ == '__main__':
    rc = RandomCircle(-0.03,0.03,0.03,15)
    rc.main()


#依据list中的每一组关键词、分隔符，修改分隔符为分号
def kw_list_separator(separator,keyword_list):
    return ['; '.join(list(set(str(i).split(separator)))).strip('; ') for i in keyword_list]

#依据list中的关键词组、年份，生成vos文件
def make_vos(years,keyword_list,path):
    rows = len(keyword_list)
    f = open(path,'w',encoding='utf-8')
    f.write('UT'+'\t'+'PY'+'\t'+'DE'+'\n')
    for l in range(rows):
        f.write(str(l)+'\t')
        f.write(str(years[l])+'\t')
        f.write(keyword_list[l]+'\n')
    f.close()


#依据list中的关键词组、分隔符（分号），统计词频
def kw_num(separator,keyword_list):
    kw_num = {}
    kw_list = []
    num_list = []
    for i in keyword_list:
        for j in str(i).split(separator):
            if j not in kw_num:
                kw_num[j] = 1
            else:
                kw_num[j] += 1
    for key,value in kw_num.items():
        kw_list.append(key)
        num_list.append(value)
    return kw_list,num_list

# 依据list中的关键词组、需要的关键词组，筛选vos文件中选取的词
def filter(need_keyword,keyword_list):
    new_keyword_list = []
    for k in keyword_list:
        ks = str(k).split('; ')
        new_ks = ''
        for kss in ks:
            if kss in need_keyword:
                new_ks = new_ks+kss+'; '
        new_ks = new_ks.strip('; ')
        new_keyword_list.append(new_ks)
    return new_keyword_list


#%%
folder_path = '/home/Nutstore/博士研究生/我的研究/2020-03颠覆性技术/data'
my_data_path = '/home/Nutstore/我的数据'
keywords_df = pd.read_excel(os.path.join(folder_path,'process_关键词提取.xlsx'),sheet_name='关键词提取')
keywords_df.columns
years = list(keywords_df['年份'].values)

ty = 'textblob'

keyword_ = list(keywords_df[ty+'关键词'].values)
#修改分隔符号
keyword_ = kw_list_separator('||',keyword_)
#生成vos文件
make_vos(years,keyword_,os.path.join(folder_path,'process_关键词分析',ty+'_vos.txt'))

data_df = pd.read_csv(os.path.join(folder_path,'process_关键词分析',ty+'_vos.txt'),sep = '\t')
data_df.columns
keyword_list = list(data_df['DE'].values)

#统计词频
kw_list,num_list = kw_num('; ',keyword_list)
cipin_df = pd.DataFrame({'关键词':kw_list,'词频':num_list})
cipin_df=cipin_df.sort_values(by=['词频'],ascending=False)
cipin_df.to_excel(os.path.join(folder_path,'process_关键词分析',ty+'_词频.xlsx'),sheet_name = '词频')


cipin_df = pd.read_excel(os.path.join(folder_path,'process_关键词分析',ty+'_词频1.xlsx'),sheet_name = '选取的词')
cipin_df.columns
need_keyword = list(cipin_df['关键词'].values)

#筛选
new_keyword_list = filter(need_keyword,keyword_list)
data_df['DE'] = new_keyword_list
data_df.to_csv(os.path.join(folder_path,'process_关键词分析',ty+'_vos1.txt'),encoding='utf-8',sep='\t',index=False)



