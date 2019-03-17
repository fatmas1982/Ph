#! /usr/bin/python3.6

from collections import Counter
import pandas as pd
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot') 
import numpy as np
import scipy.stats.stats as st

from string import digits
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import csv

import re

import string
#!pip install gensim
import gensim.models as md
from gensim.models.phrases import Phrases, Phraser
import os
from sklearn.datasets import fetch_20newsgroups
import nltk




def read_cvs_by_pands(path_database,file_databbase,index_col, header):
    return pd.read_csv(path_database+file_databbase,index_col=index_col,header=header,sep=',')




def save_file_to_database(data_rows,path_database,file_databbase,header_list):#header_list=['index','text']
    outfile = open(path_database+file_databbase,'w')
    writer=csv.writer(outfile)
    #header_list=['uuid','paragraph','doc_id']
    i=0
    for line in data_rows:
        row=[i,line]#,'paragraph no.'+str(i)]
        if i==0:
            
            writer.writerow(header_list)
            writer.writerow(row)
        else:
            #print('ff')
            writer.writerow(row)
        i+= 1
        #outfile.close()





import csv   
def add_row(row,path_database,file_name):
    #fields=['first','second','third']
    with open(path_database+file_name, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(row)
        print("printed")


# In[9]:


def autolabel(rects,ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 3+height,
                '%d' % int(height+.5),
                ha='center', va='bottom')
    




#path_database='./drive_PH/Colab Notebooks/' 
path_database='/media/fsg/74C86089C8604C04/Futuure-paper/results/'#'./results/'
path_stop_word=path_database+'input/stopwords/'

pragraph_index='pragraph_index_20.csv'
Sentences='Sentences_20.csv'
Sentences_not_stops='Sentences_not_stops_20.csv'
lesk='lesk_20.csv'

pragraph_index_reuters='pragraph_index_reuters.csv'
Sentences_reuters='Sentences_reuters.csv'
Sentences_not_stops_reuters='Sentences_not_stops_reuters.csv'
lesk_reuters='lesk_reuters.csv'

####
BrownDataset='BrownDataset.csv'
BrownDataset_lesk='BrownDataset_lesk.csv'
BrownDataset_lesk_ifidf='BrownDataset_lesk_ifidf.csv'
BrownDataset_ifidf_top_n='BrownDataset_lesk_ifidf_top_n.csv' 
H1_dataset='Brown_H1.csv'
Brown_H2='Brown_H2.csv'
Brown_H3='Brown_H3.csv'
Brown_H4='Brown_H4.csv'
Brown_W_H4='Brown_W_H4.csv'

Brown_H4_inv='Brown_H4_inv.csv'


Brown_H_S='Brown_H_S.csv'
Brown_W_S='Brown_W_S.csv'


######33
NewsGroupDataset='NewsGroupDataset.csv'
NewsGroupDataset_lesk='NewsGroupDataset_lesk.csv'
NewsGroupDataset_lesk_ifidf='NewsGroupDataset_lesk_ifidf.csv'
NewsGroupDataset_ifidf_top_n='NewsGroupDataset_lesk_ifidf_top_n.csv' 
NewsGroupDataset_H1='NewsGroup_H1.csv'
NewsGroup_H2='NewsGroup_H2.csv'
NewsGroup_H3='NewsGroup_H3.csv'
NewsGroup_H4='NewsGroup_H4.csv'
NewsGroup_W_H4='NewsGroup_W_H4.csv'

NewsGroup_H4_inv='NewsGroup_H4_inv.csv'

NewsGroup_H_S='NewsGroup_H_S.csv'
NewsGroup_W_S='NewsGroup_W_S.csv'

######33
ReuterDataset='ReuterDataset.csv'
ReuterDataset_lesk='ReuterDataset_lesk.csv'
ReuterDataset_lesk_ifidf='ReuterDataset_lesk_ifidf.csv'
ReuterDataset_ifidf_top_n='ReuterDataset_lesk_ifidf_top_n.csv'
ReuterDataset_H1='Reuter_H1.csv'
Reuter_H2='Reuter_H2.csv'
Reuter_H3='Reuter_H3.csv'
Reuter_H4='Reuter_H4.csv'
Reuter_W_H4='Reuter_W_H4.csv'

Reuter_H4_inv='Reuter_H4_inv.csv'

Reuter_H_S='Reuter_H_S.csv'
Reuter_W_S='Reuter_W_S.csv'

############333333333
OhsumedDataset='OhsumedDataset.csv'
OhsumedDataset_lesk='OhsumedDataset_lesk.csv'
OhsumedDataset_lesk_ifidf='OhsumedDataset_lesk_ifidf.csv'
OhsumedDataset_ifidf_top_n='OhsumedDataset_ifidf_top_n.csv'
OhsumedDataset_H1='OhsumedDataset_H1.csv'
Ohsumed_H2='Ohsumed_H2.csv'
Ohsumed_H3='Ohsumed_H3.csv'
Ohsumed_H4='Ohsumed_H4.csv'
Ohsumed_W_H4='Ohsumed_W_H4.csv'

Ohsumed_H4_inv='Ohsumed_H4_inv.csv'
Ohsumed_H_S='Ohsumed_H_S.csv'
Ohsumed_W_S='Ohsumed_W_S.csv'

###########3




##########
BBCDataset='BBCDataset.csv'


BBCDataset_lesk='BBCDataset_lesk.csv'
BBCDataset_lesk_ifidf='BBCDataset_lesk_ifidf.csv'
BBCDataset_ifidf_top_n='BBCDataset_ifidf_top_n.csv'
BBCDataset_H1='BBCDataset_H1.csv'
BBC_H2='BBC_H2.csv'
BBC_H3='BBC_H3.csv'
BBC_H4='BBC_H4.csv'
BBC_W_H4='BBC_W_H4.csv'

BBC_H4_inv='BBC_H4_inv.csv'
BBC_H_S='BBC_H_S.csv'
BBC_W_S='BBC_W_S.csv'
   


def isNotBlank (myString):
    if myString and myString.strip():
        #myString is not None AND myString is not empty or blank
        return True
    #myString is None OR myString is empty or blank
    return False

#isNotBlank(' ')


# In[ ]:







# In[21]:


#threshold_correlation_coefficient(read_cvs_by_pands(path_database,'H1_news_group_reuters_n.csv',index_col=0,header=0),path_database,'H2_news_group_reuters_n.csv')


# In[ ]:


#threshold_correlation_coefficient(read_cvs_by_pands(path_database,'H1_news_group_n.csv',index_col=0,header=0),path_database,'H2_news_group_n.csv')


# # Matrix2:subset_correlation_coefficient:
# 
#     sort (assending=false) matrix1 by value of current cluster(topic=column name).
# 
# remove matrix1 rows have zero value in currnet cluster. remove column matrix1 to become rows = column (square matrix).
# 

# In[32]:



#df_res_cor_coeff_subSet_cols['ace.n.03']


# # Matrix3:permutaion_subset_correlation_coefficient:
# 
# check value in other column(excpet topic column) if this value =0 check value in topic colmn for row and column keep the gretest value and remove small
# 

# In[33]:



# In[77]:




# In[78]:


#permutaion_correlation_coefficient(path_database,'H2_news_group_reuters_n.csv')


# In[37]:


#permutaion_correlation_coefficient(path_database,'H2_news_group_n.csv')


# In[79]:


#s=read_cvs_by_pands(path_database,'H4_news_group_reuters_n.csv',index_col=0,header=0)
#s


# In[39]:




# In[40]:


# In[552]:


'''
Get W for H res_sem_sim_threshold_correlation_coefficient_table 
'''


#nmf_W_res_sem_sim_permutaion_correlation_coefficient('tf_idf_lesk_table_20_100_100_n.csv','H2_news_group_n.csv',path_database,'W2_news_group_n.csv','H2_inv_news_group_n.csv')



# In[554]:



#nmf_W_res_sem_sim_permutaion_correlation_coefficient('tf_idf_lesk_table_reuters_100_100_n.csv','H2_news_group_reuters_n.csv',path_database,'W2_news_group_reuters_n.csv','H2_news_group_reuters_n.csv')


# # Reports 

# In[51]:


""" NOTES:
      - requires Python 2.4 or greater
      - elements of the lists must be hashable
      - order of the original lists is not preserved
"""
def unique(a):
    """ return the list with duplicate elements removed """
    return list(set(a))

def intersect(a, b):
    """ return the intersection of two lists """
    return list(set(a) & set(b))

def union(a, b):
    """ return the union of two lists """
    return list(set(a) | set(b))


# In[76]:


#microphone.n.01 'H4_news_group_n.csv' 5 10 path_database report_topic_term



def intersect_df(df1,df2):
    H4_20=read_cvs_by_pands(path_database,df1,index_col=0,header=0)
    H4_column=H4_20.columns
    Hs_20=read_cvs_by_pands(path_database,df2,index_col=0,header=0) 
    Hs_column=Hs_20.columns
    for i in range(5):
        a=H4_20[H4_column[i]]
        b=Hs_20[Hs_column[i]]
        c=intersect(a, b)
        print(c)
    



def counter_conditions_tables(path_database,main_table,counting_table,counting_column_name,condition,num_after_cond):
    df_Thresholding=read_cvs_by_pands(path_database,main_table,index_col=0,header=0)
    if condition =='!=':
        r=(df_Thresholding != num_after_cond).sum(axis=0)#columns
    if condition =='==':
        r=(df_Thresholding == num_after_cond).sum(axis=0)
    if condition =='>=':
        r=(df_Thresholding >= num_after_cond).sum(axis=0)
    if condition =='<=':
        r=(df_Thresholding <= num_after_cond).sum(axis=0)
    if condition =='<':
        r=(df_Thresholding < num_after_cond).sum(axis=0)
    if condition =='>':
        r=(df_Thresholding > num_after_cond).sum(axis=0)
    #print(type(r))
    df=pd.DataFrame(r,columns=[counting_column_name])
    df.T.to_csv(path_database+counting_table)#columns=df_Thresholding.columns)
    return df
     






