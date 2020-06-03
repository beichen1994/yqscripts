import sys
import os
import csv
import numpy as np
import pandas as pd

def is_valid_file(file_name):
    """Ensure that the input_file exists."""
    if not os.path.exists(file_name):
        print("The file '{}' does not exist!".format(file_name))
        sys.exit(1)

#csv文件中必须有keywords这一列，并且里面的关键词必须是用空格分割的
#csv文件中必须有year这一列

def parse_file(input_file,output_path,output_file):
    
    is_valid_file(input_file)
    
    csv_df = pd.read_csv(input_file,encoding='utf-8') 
    keywords = list(csv_df['keywords'].values) 
    year_ = list(csv_df['year'].values) 
    keywords_list=[]
    for i in keywords:
        keywords_str=''
        for j in str(i).split(';'):
            if j != ' ':
                keywords_str = keywords_str+j+'; '
        keywords_str = keywords_str.strip('; ; ')
        keywords_list.append(keywords_str)
    rows = len(keywords_list)
    domain_dfs = pd.DataFrame({'DE':keywords_list})
    domain_dfs['UT']=[i for i in range(rows)]
    domain_dfs['PY'] = year_ 
    order=['UT','PY','DE']
    domain_dfs=domain_dfs[order]
    UT = list(domain_dfs.UT.values)
    PY = list(domain_dfs.PY.values)
    DE = list(domain_dfs.DE.values)
    current_output = os.path.join(output_path,"{}".format(output_file))
    f = open(current_output,'w',encoding='utf-8')
    f.write('UT	PY	DE')
    f.write('\n')
    for i in range(domain_dfs.shape[0]):
        f.write(str(UT[i]))
        f.write('	')
        f.write(str(PY[i]))
        f.write('	')
        f.write(str(DE[i]))
        f.write('\n')
    f.close()


