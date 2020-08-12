import os
import time
import pandas as pd
import numpy as np

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
    

