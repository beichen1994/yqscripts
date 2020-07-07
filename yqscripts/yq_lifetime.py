"""
思路：在2005-2006这个时间窗口下，有5篇被引文献(发表日期为CITED_PY),施引文献的发表时间为CITING_PY，共有6年时间间隔(Citing_Cited)
    0   1   2   3   4   5   6
A   2   3   5   7   8   8   7   
B   0   2   3   4   5   6   7
C   1   3   4   5   6   4   7
D   2   4   5   5   6   6   7
E   0   2   3   4   5   4   6

"""
import os
import pandas as pd

folder_path = 
file_path = os.path.join(folder_path,)
