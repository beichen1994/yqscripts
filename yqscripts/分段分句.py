import os
import pandas as pd
import numpy as np
import re

def rule_para(text,n):
    p1 = '[a-z]\)'
    p1_ = re.compile(p1)
    p2 = '[a-z]\.'
    p2_ = re.compile(p2)
    p3 = '^[A-Z]'
    p3_ = re.compile(p3)
    p4 = '^\d'
    p4_ = re.compile(p4)
    comment_list = text.split('\n')
    para = 0
    comment_list2 = []
    for i in comment_list:
        if len(i) >= 3:
            if i[0] == '*':
                i = 'A'+i[1:]
            if i[0] == '-':
                i = 'A'+i[1:]
            else:
                m1 = re.match(p1,i)
                m2 = re.match(p2,i)
                m3 = re.match(p3,i)
                m4 = re.match(p4,i)
                if m1 != None:
                    i = re.sub(p1_,'A',i)
                if m2 != None:
                    i = re.sub(p2_,'A',i)
                if m3 != None:
                    i = re.sub(p3_,'A',i)
                if m4 != None:
                    i = re.sub(p4_,'A',i)
            comment_list2.append(i)
    max_len = max([len(i) for i in comment_list2])
    each_comment=""
    print(len(comment_list2))
    for k in range(len(comment_list2)-1):
        if (len(comment_list2[k]) != 0) and (len(comment_list2[k+1]) != 0):
            if (len(comment_list2[k]) < max_len*n):
                if (comment_list2[k][-1] == '.') or (comment_list2[k][-1] == '?'):
                    if (comment_list2[k+1][0] == 'A') or (comment_list2[k+1][0] == ' '):
                        para += 1
                        comment_list2[k] = comment_list2[k] + '|||||||'
    for c in comment_list2:
        each_comment = each_comment+c+'\n'
    para+=1
    return para,each_comment

def rule_sen(text):
    p1 = 'Dr\.'
    p1_ = re.compile(p1)
    p2 = 'M\.D\.'
    p2_ = re.compile(p2)
    p3 = 'vs\.'
    p3_ = re.compile(p3)
    p4 = 'et al\.'
    p4_ = re.compile(p4)
    p5 = 'e\.g\.'
    p5_ = re.compile(p5)
    p6 = 'e\.g'
    p6_ = re.compile(p6)
    p7 = '[A-Z]\.'
    p7_ = re.compile(p7)
    p8 = 'etc\.'
    p8_ = re.compile(p8)
    p9 = 'no\.'
    p9_ = re.compile(p9)
    p10 = 'ref\.'
    p10_ = re.compile(p10)
    p11 = 'i\.e\.'
    p11_ = re.compile(p11)
    p12 = 'Fig\.\d'
    p12_ = re.compile(p12)
    p13 = '\d\.\d'
    p13_ = re.compile(p13)
    p14= '[\.\.]'
    p14_ = re.compile(p14)
    p15= '\.\d\.'
    p15_ = re.compile(p15)
    comment_list = text.split('\n')
    comment_list2 = []
    for i in comment_list:
        m1 = re.search(p1,i)
        m2 = re.search(p2,i)
        m3 = re.search(p3,i)
        m4 = re.search(p4,i)
        m5 = re.search(p5,i)
        m6 = re.search(p6,i)
        m7 = re.search(p7,i)
        m8 = re.search(p8,i)
        m9 = re.search(p9,i)
        m10 = re.search(p10,i)
        m11 = re.search(p11,i)
        m12 = re.search(p12,i)
        m13 = re.search(p13,i)
        m14 = re.search(p14,i)
        m15 = re.search(p15,i)
        if m1 != None:
            i = re.sub(p1_,'',i)
        if m2 != None:
            i = re.sub(p2_,'',i)
        if m3 != None:
            i = re.sub(p3_,'',i)
        if m4 != None:
            i = re.sub(p4_,'',i)
        if m5 != None:
            i = re.sub(p5_,'',i)
        if m6 != None:
            i = re.sub(p6_,'',i)
        if m7 != None:
            i = re.sub(p7_,'',i)
        if m8 != None:
            i = re.sub(p8_,'',i)
        if m9 != None:
            i = re.sub(p9_,'',i)
        if m10 != None:
            i = re.sub(p10_,'',i)
        if m11 != None:
            i = re.sub(p11_,'',i)
        if m12 != None:
            i = re.sub(p12_,'',i)
        if m13 != None:
            i = re.sub(p13_,'',i)
        if m14 != None:
            i = re.sub(p14_,'',i)
        if m15 != None:
            i = re.sub(p15_,'',i)
        comment_list2.append(i)
    each_comment=""
    for c in comment_list2:
        each_comment = each_comment+c+'\n'
    return each_comment




folder_path = '/data/Nutstore/guangyao_yuqi科研组/gy_data/BMJ/split_paragh/'
file_path = os.path.join(folder_path,'new_para.xlsx')
file_df = pd.read_excel(file_path)
file_df.columns
file_df.shape[0]
comments = file_df.comments

for n in [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]:
    each_paras=[]
    each_comments=[]
    for f in range(len(file_df)):
        comment = comments.iloc[f]
        para,each_comment = rule_para(comment,n)
        each_paras.append(para)
        each_comments.append(each_comment)
    file_df['新段数'+str(n)] = each_paras
    file_df['新划分'+str(n)] = each_comments
file_df
#file_df.to_excel(os.path.join(folder_path,'new_para5.xlsx'),sheet_name = 'sheet2')


file_path = os.path.join(folder_path,'new_para5.xlsx')
file_df = pd.read_excel(file_path)
file_df.columns
file_df.shape[0]
comments = file_df['新划分1']
comments_sens=[]
for f in range(len(file_df)):
    comment = comments.iloc[f]
    each_comment = rule_sen(str(comment))
    print(each_comment)
    each_comment_sen = '%%%%%%'.join(re.split(r'[.?]',str(each_comment)))
    comments_sens.append(each_comment_sen)
 
file_df['分句'] = comments_sens
file_df['分句']

file_df.to_excel(os.path.join(folder_path,'new_para5_分句.xlsx'),sheet_name = 'sheet1')

"""
一、首先先替换

1. 把一下的内容直接换成空

Dr.
M.D.
vs. 
et al. 
e.g.
e.g

大写字母点，如
A.
……
Z.

etc.
no.
ref.
i.e.
Fig.1   Fig.+数字

诸如0.3，9.5  两个数字之间有点，换掉


2. 以下的换成一个点
..  超过两个的，应该换成成一个
.?
.2. 点数字点，换成一个点

二、
以句号(． ) /问号(?)作为句子的分隔符，切分出句子



具体思想来自胡老师的论文
(1)以句号(． ) /问号(?)作为句子的分隔符，切分出句子，并写入 sentence 数据表中。
由于句号除作为句子结束符外，还可能出现在人名(如“Iijima S． ”)、数字(如“0． 123”)或其他缩写中(如“etc．”“e． g．”“Fig．1”)中。对于这类情况，主要采取词表替换(主要针对缩写中的句号)和正则表达式替换(主要针对人名和数字中的句号)相结合的方法，将干扰句号首先替换掉其他特殊符号，切分之后再进行恢复

"""

