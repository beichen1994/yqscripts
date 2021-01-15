folder_path = 'F:\\google_drive'
import numpy as np
import pandas as pd
import os
import time
import re
import csv
import codecs
import operator
#----------------------------------------------------------
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.font_manager import FontProperties
#列举电脑中所有字体
#fonts=[font.name for font in fontManager.ttflist if os.path.exists(font.fname) and os.stat(font.fname).st_size>2e6]
fonts=[font.name for font in fontManager.ttflist if os.path.exists(font.fname)]
#解决负号'-'显示为方块的问题
matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib.font_manager import _rebuild
#----------------------------------------------------------
import seaborn as sns
sns.set() #加载seaborn默认格式设定
sns.set_style("whitegrid")
#----------------------------------------------------------
from collections import Counter
from collections import defaultdict
import statsmodels.api as sm
#----------------------------------------------------------
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
# 精确率
from sklearn.metrics import accuracy_score
# 混淆矩阵
from sklearn.metrics import confusion_matrix
# 精准率
from sklearn.metrics import precision_score
# 召回率
from sklearn.metrics import recall_score 
# F1 Score
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#----------------------------------------------------------
import nltk
from nltk import data
data.path.append(os.path.join(folder_path,"辅助文件","nltk_data"))
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
nltk_stop = nltk.corpus.stopwords.words("english")
nltk_stop_upper = [i.capitalize() for i in nltk_stop]
nltk_stop = nltk_stop+nltk_stop_upper
nltk_stop.append('—')
stop_words = open(os.path.join(folder_path,'辅助文件','停用词',"停用词公共.txt"),'r',encoding='utf-8').read().split('\n')+nltk_stop
#----------------------------------------------------------------------------
import jieba
import jieba.analyse as ana
#----------------------------------------------------------
import wordcloud
import gensim
from gensim.models import doc2vec
from textblob import TextBlob
#----------------------------------------------------------
from cnsenti import Emotion
from cnsenti import Sentiment
from snownlp import SnowNLP
#----------------------------------------------------------

#函数：选择文件夹
def folder_select(folder_path=folder_path):
    filenames = os.listdir(folder_path)
    selected_filename = filenames[1]
    return os.path.join(folder_path, selected_filename)

#函数：读取数据
def load_data(filename):
    if filename[-3:] == 'xls':
        data_df = pd.read_excel(filename)
    if filename[-4:] == 'xlsx':
        data_df = pd.read_excel(filename)
    if filename[-3:] == 'csv':
        data_df = pd.read_csv(filename)
    if filename[-3:] == 'txt':
        data_df = open(filename,'r',encoding='utf-8').read()
    return data_df

#函数：读取WoS数据
def load_WoS_data(filename):
    df = pd.read_csv(filename,sep='\t',encoding='utf-16le',index_col=False,quoting=csv.QUOTE_NONE)
    return df

#函数：Derwent纯文本转制表符分隔
#函数：利用空格匹配数据
def derwent_txt2df(file):
    txt_patent_list = []
    #匹配出一个专利
    pattern = re.compile('\nPT P(.*?)\nER',re.S)
    patent = pattern.findall(file)
    for single_patent in patent:
        patent_list = single_patent.split('\n')[1:]
        #按照小旗子为1，小旗子为1.0，组合字段
        flag_list=list()
        for p in patent_list:
            key = p[0:2]
            if key == '   ':
                flag_list.append(0)
            else:
                flag_list.append(1)
        combine_list = []
        for flag_index in range(len(flag_list)-2):
            small_combine=[]
            if flag_list[flag_index]==1 and flag_list[flag_index+1] ==1:
                small_combine.append(flag_index)
                combine_list.append(small_combine)
                continue
            if flag_list[flag_index]==1 and flag_list[flag_index+1] ==0:
                small_combine.append(flag_index)
                for flag_index2 in range(flag_index+1,len(flag_list)):
                    if flag_list[flag_index2] == 1:
                        break
                    small_combine.append(flag_index2)
                combine_list.append(small_combine)
                continue
        small_combine=[]
        if flag_list[len(flag_list)-2] == 1 and flag_list[len(flag_list)-1] ==1:
            combine_list.append([len(flag_list)-2])
            combine_list.append([len(flag_list)-1])
        elif flag_list[len(flag_list)-2] == 1 and flag_list[len(flag_list)-1] ==0:
            small_combine.append(len(flag_list)-2)
            small_combine.append(len(flag_list)-1)
            combine_list.append(small_combine)
        elif flag_list[len(flag_list)-2] == 0 and flag_list[len(flag_list)-1] ==1:
            combine_list.append([len(flag_list)-1])
        patent_dict=dict()
        for combine in combine_list:
            if len(combine) == 1:
                key = patent_list[combine[0]][0:2]
                value = patent_list[combine[0]][3:]
                patent_dict[key] = value
            else:
                key = patent_list[combine[0]][0:2]
                value = '###'.join([patent_list[single_combine][3:] for single_combine in combine])
                patent_dict[key] = value
        single_patent_df = pd.DataFrame(patent_dict,index=[0])
        txt_patent_list.append(single_patent_df)
    txt_patent_df = pd.concat(txt_patent_list)
    return txt_patent_df

#函数：利用正则表达式匹配数据
def reg_derwent_term1(file,term1,term2):
    terms = []
    #匹配出一个专利
    pattern = re.compile('\nPT P(.*?)\nER',re.S)
    patent = pattern.findall(file)
    for i in range(0,len(patent)):
        #提取特定字符串
        pattern1 = re.compile('\n'+term1+' (.*?)\n'+term2,re.S)
        save=pattern1.findall(patent[i])
        if len(save) == 1:
            single_term = ';'.join(save[0].split('\n'))
            terms.append(single_term)
    return terms

#函数：展示数据基本情况
def terms_basic(data_df):
    print("数据列的形状:",data_df.shape)
    print("数据列的基本描述:",data_df.describe())

#函数:字段特殊符号清洗
def clean_terms_df(terms_df): 
    # re.match尝试从字符串的起始位置匹配一个模式
    # re.search()会扫描整个string查找匹配,会扫描整个字符串并bai返回第一个成功的匹配
    # re.sub(pattern, repl, string[, count])  count用于指定最多替换次数，不指定时全部替换。

    # 微博正文文本清洗
    content = list(terms_df.values)
    content_1=[]
    for c in content:
        c1 = c.strip('   ~').strip(' ').strip('~').strip('\xa0\xa0\n')

        p1 = re.compile('//')
        m1 = re.search(p1,c1)
        if m1 != None:
            c1 = re.sub(p1,',',c1)
        
        p17 = re.compile('回复@.*?:')
        m17 = re.search(p17,c1)
        if m17 != None:
            c1 = re.sub(p17,'',c1)

        p2 = re.compile('"##.*?##')
        m2 = re.search(p2,c1)
        if m2 != None:
            c1 = re.sub(p2,'',c1)
        
        
        p3 = re.compile('O网页链接')
        m3 = re.search(p3,c1)
        if m3 != None:
            c1 = re.sub(p3,'',c1)

        p4 = re.compile('##.*?##')
        m4 = re.search(p4,c1)
        if m4 != None:
            c1 = re.sub(p4,'',c1)

    
        p5 = re.compile('转发微博')
        m5 = re.search(p5,c1)
        if m5 != None:
            c1 = re.sub(p5,'',c1)
        
        p6 = re.compile('@.*? ')
        m6 = re.search(p6,c1)
        if m6 != None:
            c1 = re.sub(p6,'',c1)

        p7 = re.compile('http //.*? ')
        m7 = re.search(p7,c1)
        if m7 != None:
            c1 = re.sub(p7,'',c1)
        
        p8 = re.compile('#.*?#')
        m8 = re.search(p8,c1)
        if m8 != None:
            c1 = re.sub(p8,'',c1)
        
        p9 = re.compile('http://.*? ')
        m9 = re.search(p9,c1)
        if m9 != None:
            c1 = re.sub(p9,'',c1)

        p10 = re.compile('@.*?：')
        m10 = re.search(p10,c1)
        if m10 != None:
            c1 = re.sub(p10,'',c1)

        p11 = re.compile('轉發微博')
        m11 = re.search(p11,c1)
        if m11 != None:
            c1 = re.sub(p11,'',c1)

        p12 = re.compile('转发')
        m12 = re.search(p12,c1)
        if m12 != None:
            c1 = re.sub(p12,'',c1)

        p13 = re.compile('\$.*?\$')
        m13 = re.search(p13,c1)
        if m13 != None:
            c1 = re.sub(p13,'',c1)
        
        p14 = re.compile('[组图共*张]')
        m14 = re.search(p14,c1)
        if m14 != None:
            c1 = re.sub(p14,'',c1)
        
        p15 = re.compile('\[.*?\]')
        m15 = re.search(p15,c1)
        if m15 != None:
            c1 = re.sub(p15,'',c1)
        

        p16 = re.compile('http ,t.cn.*? ')
        m16 = re.search(p16,c1)
        if m16 != None:
            c1 = re.sub(p16,'',c1)
        
        p18 = re.compile('\(.*?\)')
        m18 = re.search(p18,c1)
        if m18 != None:
            c1 = re.sub(p18,'',c1)
        
        p19 = re.compile('\（.*?\）')
        m19 = re.search(p19,c1)
        if m19 != None:
            c1 = re.sub(p19,'',c1)

        p20 = re.compile('http ,t.cn.*?')
        m20 = re.search(p20,c1)
        if m20 != None:
            c1 = re.sub(p20,'',c1)
        
        c1 =c1.replace(' →_→','').replace('【',' ').replace('】',',').replace('\xa0','').replace('…','').replace(': ','')

        c1 = c1.strip('   ~').strip(' ').strip('~').strip('\xa0\xa0\n').strip('!').strip('!!').strip('!!!').strip('！').strip('！！').strip('！！！').strip(',').strip('。').strip(", ").strip(' ').strip('　　').strip('。。。').strip('：').strip('，').strip('、').strip('， ').strip('*').strip('¡¾').strip('？？？ ').strip('？？ ').strip('？ ').strip(':').strip('原')
        content_1.append(c1)

        return content_1

#函数：统计某个字段最多的前10位¶
def count_rank_10(df,ziduan,sep):
    items = []
    for i in list(df[ziduan].values):
        for j in str(i).split(sep):
            items.append(j)
    obj = collections.Counter(items).most_common(10)
    obj = dict(obj)
    return obj

#函数：统计某个字段的n元组
def get_top_ngram(terms_df, n=None):
    vector = CountVectorizer(ngram_range=(n, n)).fit(terms_df)
    bag_of_words = vector.transform(terms_df)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) 
                  for word, idx in vector.vocabulary_.items()] #vec.vocabulary_矩阵化后，二元组词对应的列id
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq

#----------------------------------------------------------------------------------------------------------
# pandas 快捷操作
def pan_dup_False(A_df,fields):
    A_uni_df = A_df[A_df[fields].duplicated()==False] 
    return A_uni_df

def pan_dup_True(A_df,fields):
    A_uni_df = A_df[A_df[fields].duplicated()==True]
    return A_uni_df

def pan_nan_True(A_df,fields):
    field_df = A_df[A_df[fields].map(lambda x: x is np.nan) == True]
    return field_df

def pan_nan_False(A_df,fields):
    field_df = A_df[A_df[fields].map(lambda x: x is np.nan) == False]
    return field_df

#----------------------------------------------------------------------------------------------------------
#函数：某个字段分词并词形还原
def preprocess_terms(terms_df,stop_words):
    corpus=[]
    lem=WordNetLemmatizer()
    terms = list(terms_df.values)
    for t in terms:
        words=[w for w in nltk.word_tokenize(t) if (w not in stop_words)]
        words=[lem.lemmatize(w) for w in words if len(w)>2]
        corpus.append(words)
    return corpus 

#函数：RAKE算法分词
class Rake:
    def __init__(self, documents, stopwordsFilePath, minPhraseChar, maxPhraseLength):
        self.docs = documents
        self.minPhraseChar = minPhraseChar
        self.maxPhraseLength = maxPhraseLength
        self.stopwordsFilePath = stopwordsFilePath
        stopwords = []
        for word in codecs.open(stopwordsFilePath, 'r', 'utf-8'):
            stopwords.append(word.strip())
        stopwordsRegex = []
        for word in stopwords:
            regex = r'\b' + word + r'(?![\w-])'
            stopwordsRegex.append(regex)
        self.stopwordsPattern = re.compile('|'.join(stopwordsRegex), re.IGNORECASE)
        
    def isNumber(self,s):
        try:
            float(s) if '.' in s else int(s)
            return True
        except ValueError:
            return False
        
    def separateWords(self, text):
        splitter = re.compile('[^a-zA-Z0-9_\\+\\-/]')
        words = []
        for word in splitter.split(text):
            word = word.strip().lower()
            if len(word) > 0 and word != '' and not self.isNumber(word):
                words.append(word)
        return words 
    def calculatePhraseScore(self, phrases):
        wordFrequency = {}
        wordDegree = {}
        for phrase in phrases:
            wordList = self.separateWords(phrase)
            wordListLength = len(wordList)
            wordListDegree = wordListLength - 1
            for word in wordList:
                wordFrequency.setdefault(word, 0)
                wordFrequency[word] += 1
                wordDegree.setdefault(word, 0)
                wordDegree[word] += wordListDegree
        for item in wordFrequency:
            wordDegree[item] = wordDegree[item] + wordFrequency[item]
        wordScore = {}
        for item in wordFrequency:
            wordScore.setdefault(item, 0)
            wordScore[item] = wordDegree[item] * 1.0 / wordFrequency[item]
        phraseScore = {}
        for phrase in phrases:
            phraseScore.setdefault(phrase, 0)
            wordList = self.separateWords(phrase)
            candidateScore = 0
            for word in wordList:
                candidateScore += wordScore[word]
            phraseScore[phrase] = candidateScore
        return phraseScore
    def execute(self):
        result = [] 
        for document in self.docs:
            sentenceDelimiters = re.compile(u'[.!?,;:\t\\\\"\\(\\)\\\'\u2019\u2013]|\\s\\-\\s')
            sentences = sentenceDelimiters.split(str(document))
            phrases = []
            for s in sentences:
                tmp = re.sub(self.stopwordsPattern, '|', s.strip())
                phrasesOfSentence = tmp.split("|")
                for phrase in phrasesOfSentence:
                    phrase = phrase.strip().lower()
                    if phrase != "" and len(phrase) >= self.minPhraseChar and len(phrase.split()) <= self.maxPhraseLength:
                        phrases.append(phrase)
            phraseScore = self.calculatePhraseScore(phrases)
            keywords = sorted(phraseScore.items(), key = operator.itemgetter(1), reverse=True)
            result.append(str(keywords[0:int(len(keywords)/3)]))
        return result

#函数：jieba分词
def jieba_extract(terms_list,stop_words):
    terms_word=[]
    for i in range(len(terms_list)):
        single_terms_list =terms_list[i]
        #分词：精确模式切词
        word_list = jieba.cut(single_terms_list,cut_all = False)    
        #去除停用词
        word_list = [word for word in word_list if word not in stop_words]
        #去除空格
        word_list = [word for word in word_list if word not in ['',' ','  ']]
        word_str = '||'.join(word_list)
        terms_word.append(word_str)
    return terms_word

#函数：nltk算法分词
def nltk_fenci(terms_df,stop_words):
    mingci=['NN','NNS','NNP']
    xingrongci=['JJ','JJR','JJS']
    lianci = ['CC']
    jieci = ['IN']
    big_cixing_combine=[]
    for mc in mingci:
        for mc2 in mingci:
            big_cixing_combine.append((mc,mc2))
        for xrc in xingrongci:
            big_cixing_combine.append((xrc,mc)) 
    tri_cixing_combine=[]
    for mc1 in mingci:
        for mc2 in mingci:
            for mc3 in mingci:
                tri_cixing_combine.append((mc1,mc2,mc3))
        for lc in lianci:
            for mc2 in mingci:
                tri_cixing_combine.append((mc1,lc,mc2))
        for jc in jieci:
            for mc2 in mingci:
                tri_cixing_combine.append((mc1,jc,mc2))
    cixings=[]
    for bcc in big_cixing_combine:
        cixings.append(bcc)
    for tcc in tri_cixing_combine:
        cixings.append(tcc)
    for mc in mingci:
        cixings.append((mc))

    cixings = list(set(cixings))
    all_words = []
    for i in range(len(terms_df)):
        #获得一个摘要
        single_terms_df = terms_df.iloc[i]  
        houxuancizu=[] 
        #对该摘要分句
        sens = nltk.sent_tokenize(single_terms_df)
        for s1 in sens:
            # 分词
            tokens_ori=word_tokenize(s1) 
            #定义符号列表
            interpunctuations = [',', ' ','.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']  
            #删除标点符号
            tokens_ori_1 = [word for word in tokens_ori if word not in interpunctuations]  
            #删除终止词
            tokens = [i for i in tokens_ori_1 if i not in stop_words]
            # 获得所有二元词组
            bigram=nltk.bigrams(tokens)
            bigram_list = list(bigram)
            #获得所有三元词组
            trigram=nltk.trigrams(tokens)
            trigram_list = list(trigram)
            #对s1进行词性标注
            data=[]
            data=data+nltk.pos_tag(tokens)    
            #根据词性提取特定词性的关键词
            w=[] #单词列表
            p=[] #词性列表
            #单个的词
            for word in data:
                w.append(word[0])
                p.append(word[1])
                if word[1] in cixings:
                    houxuancizu.append(word[0])
            #二元的词组
            if len(w) == len(bigram_list)+1:
                for b in range(len(bigram_list)):
                    wor = tuple((w[b],w[b+1]))
                    #print(wor)
                    cixing = tuple((p[b],p[b+1]))
                    #print(cixing)
                    if cixing in cixings:
                        wo = wor[0]+" "+wor[1]
                        houxuancizu.append(wo)
            #三元的词组
            if len(w) == len(trigram_list)+2:
                for t in range(len(trigram_list)):
                    wor = tuple((w[t],w[t+1],w[t+2]))
                    cixing = tuple((p[t],p[t+1],p[t+2]))
                    if cixing in cixings:
                        wo = wor[0]+" "+wor[1]+" "+wor[2]
                        houxuancizu.append(wo)
            #最后得到的候选词组
        houxuancizu = '||'.join(houxuancizu)
        all_words.append(houxuancizu)
    return all_words

#分词权重计算
#函数：jieba的TF-IDF权重计算
def jieba_TF_IDF(terms_df,stop_words):
    all_words_tf_idf=[]
    for i in range(len(terms_df)):
        single_terms_df =terms_df[i]
        #分词：精确模式切词
        word_list = jieba.cut(single_terms_df,cut_all = False)    
        #去除停用词
        word_list = [word for word in word_list if word not in stop_words]
        #去除空格
        word_list = [word for word in word_list if word not in ['',' ','  ']]
        word_str = '||'.join(word_list)
        keywords_tf_idf = jieba.analyse.extract_tags(word_str,topK = 10, withWeight = True)
        #使用结巴默认的idf文件进行关键词提取，展示权重前十的关键词
        word_str_tf_idf = '||'.join([str(k[0]) for k in keywords_tf_idf])
        all_words_tf_idf.append(word_str_tf_idf)
    return all_words_tf_idf

#函数：jieba的TextRank权重计算
def jieba_TextRank(terms_df,stop_words):
    all_words_textrank=[]
    for i in range(len(terms_df)):
        single_terms_df =terms_df[i]
        #分词：精确模式切词
        word_list = jieba.cut(single_terms_df,cut_all = False)    
        #去除停用词
        word_list = [word for word in word_list if word not in stop_words]
        word_str = '||'.join(word_list)
        keywords_textrank = jieba.analyse.textrank(word_str,topK = 10, withWeight = True,allowPOS=('n', 'nr', 'ns')) 
        word_str_textrank = '||'.join([str(k[0]) for k in keywords_textrank])
        all_words_textrank.append(word_str_textrank)
    return all_words_textrank

#函数：将字段中的值处理为分号分隔，以方便生成vos文件
def terms_list_separator(separator,terms):
    return ['; '.join(list(set(str(i).split(separator)))).strip('; ') for i in terms]

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
    cipin_df = pd.DataFrame({'关键词':kw_list,'词频':num_list})
    cipin_df=cipin_df.sort_values(by=['词频'],ascending=False)
    cipin_df.to_excel(os.path.join(folder_path,'process_关键词分析',ty+'_词频.xlsx'),sheet_name = '词频')

# 通过设定圆心、半径、圆的点个数随机生成点坐标
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
#rc = RandomCircle(-0.03,0.03,0.03,15)
#rc.main()

#----------------------------------------------------------------------------------------------------------
#函数：Doc2Vec模型
def my_doc2vec_model(doclist):
    reslist = []
    for i,doc in enumerate(doclist):
        blob = TextBlob(doc)
        np = list(blob.noun_phrases)
        reslist.append(doc2vec.TaggedDocument(np, [i]))
    return reslist

#函数：根据机器学习算法选择机器学习参数
def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        #C = st.slider('C', 0.01, 10.0)
        C = 0.01
        params['C'] = C
    elif clf_name == 'KNN':
        #K = st.slider('K', 1, 15)
        K = 1
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

#函数：根据机器学习算法构造分类器
def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    elif clf_name == 'Logistic Regression':
        clf = LogisticRegression()
    elif clf_name == 'Decision Tree':
        clf = DecisionTreeClassifier()
    elif clf_name == 'Linear Regression':
        clf = LinearRegression()
    elif clf_name == 'Random Forest':
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], random_state=1234)
    return clf


