#! /usr/bin/python3.6


import pandas as pd

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

###Evaluation_Files###########

def read_cvs_by_pands(path_database,file_databbase,index_col, header):
    return pd.read_csv(path_database+file_databbase,index_col=index_col,header=header,sep=',')



path_evaluation='/media/fsg/74C86089C8604C04/Futuure-paper/evaluation/'

path_tkm='/media/fsg/74C86089C8604C04/download/tkm-master/output/AllWords/'




#path_database='./drive_PH/Colab Notebooks/' 
path_database='/media/fsg/74C86089C8604C04/Futuure-paper/results/'#'./results/'


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
    




   


def isNotBlank (myString):
    if myString and myString.strip():
        #myString is not None AND myString is not empty or blank
        return True
    #myString is None OR myString is empty or blank
    return False




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

'''
Return matrix terms per top -topic
''' 
def top_term_top_topic(keyword,H_df,top_topic,top_term,path_database):
    H4_20=u.read_cvs_by_pands(path_database,H_df,index_col=0,header=0)
    H4_20_max5=H4_20.loc[keyword].sort_values(ascending=False)[:top_topic]
    print(H4_20_max5)
    df_H4_20_max5 = pd.DataFrame(H4_20_max5)#
    df_H4_20_max5
    d={}
    for i in range(len(df_H4_20_max5.index)):

        list_topic_word=u.read_cvs_by_pands(path_database,H_df,index_col=0,header=0)[df_H4_20_max5.index[i]].sort_values(ascending=False)[:top_term]
        #pd.DataFrame(df)
        d[df_H4_20_max5.index[i]]=list_topic_word.index
    #print(d)
    final_df=pd.DataFrame(d,columns=df_H4_20_max5.index)
    name_file=keyword.split('.')[0]
    final_df.to_csv(path_evaluation+name_file+'.csv')
    #final_df.to_csv(path_database+report_topic_term)
    return final_df

'''
Return matrix top -topic only not term in rows
''' 
def top_term_top_topic_all(keyword,H_df,top_topic,top_term,path_database):
  
    H4_20=u.read_cvs_by_pands(path_database,H_df,index_col=0,header=0)
    H4_20_max5=H4_20.loc[keyword].sort_values(ascending=False)[:top_topic]
    df_H4_20_max5 = pd.DataFrame(H4_20_max5)#
    df_H4_20_max5
    d={}
    for i in range(len(df_H4_20_max5.index)):

        list_topic_word=u.read_cvs_by_pands(path_database,H_df,index_col=0,header=0)[df_H4_20_max5.index[i]].sort_values(ascending=False)[:top_term]
        #pd.DataFrame(df)
        d[df_H4_20_max5.index[i]]=list_topic_word#.index
    #print(d)
    final_df=pd.DataFrame(d,columns=df_H4_20_max5.index)
    name_file=keyword.split('.')[0]
    final_df.to_csv(path_evaluation+name_file+'.csv')
    return final_df

#top_term_top_topic_all('stage_set.n.01',u.Brown_H4,5, 10, u.path_database)




def clean_csv_for_topic_evaluation(text):
    import re
    from string import punctuation
    # Clean the text, with the option to remove stop_words and to stem words.
    
    # Clean the text
    
    text = re.sub(r"\n\n", " ", text)
    text = re.sub(r"\t", " ", text)
    
    text = re.sub(r"'", "", text)
    text = re.sub(r",", " ", text)
    #text = re.sub(r"]", " ", text)
    #text = re.sub(r"[", " ", text)
    text = re.sub(r'"', " ", text)
    #text = re.sub(r"^", " ", text)
    #text = re.sub(r"%", " ", text)
    
    
    # Remove punctuation from text
    #text = ''.join([c for c in text if c not in punctuation])
    #remove number
    #text=''.join(c if c not in map(str,range(0,10)) else "" for c in text)
    
   
    return(text)

def clean_CSV_fortopic_Evaluation(pathcsv,pathdestnation):
    
    w=open(pathdestnation,'a')#'/media/fsg/74C86089C8604C04/download/topic_interpretability-master/ref_corpus/wiki/ReuterDataset_lesk.txt'
    i=0
    #'/media/fsg/74C86089C8604C04/download/topic_interpretability-master/ref_corpus/wiki/ReuterDataset_lesk.csv'
    with(open(pathcsv,'r')) as f:
        line=f.read()
        new=clean_csv_for_topic_evaluation(line)
        #w.write(str(i))
        #i+=1
        w.write(new)
        #w.write("===============")
    w.close()
    print("End")

#clean_CSV_fortopic_Evaluation('/media/fsg/74C86089C8604C04/Futuure-paper/evaluation/csv/OhsumedAll_top5_terms_topics.csv','/media/fsg/74C86089C8604C04/Futuure-paper/evaluation/Topics/OhsumedAll_top5_terms_topics.txt')



def Myjaccard(first_document, second_document):
    #https://github.com/sknepal/DocSim/blob/master/Jaccard%20and%20Cosine%20Similarity.ipynb
    #calculate jaccard similarity
    intersection = set(first_document).intersection(set(second_document))
    union = set(first_document).union(set(second_document))
    return round(len(intersection)/len(union), 2)


def Jaccard_All(tkmf,SNNMFf,JC):
    tkm=open(tkmf,'r')#/media/fsg/74C86089C8604C04/download/topic_interpretability-master/data/Topics_20newsgroup.txt','r')
    tkm_topics=tkm.readlines()
    SNNMF=open(SNNMFf,'r')#'/media/fsg/74C86089C8604C04/download/topic_interpretability-master/data/NewsGroupAll_top5_terms_topics.txt','r')
    SNNMF_topics=SNNMF.readlines()
    JC_files=open(JC,'a')#'/media/fsg/74C86089C8604C04/Futuure-paper/evaluation/Jaccard/Jaccard.csv','a')
    len(tkm_topics)
    header=''
    header=header.join('SNNMF')#+','+'TKM'+'\n'
    header=header.join(',')
    header=header.join('TKM')
    header=header.join('\n')
    JC_files.write(header)
    for i in range(21):
        All_topics_Jaccard=''
        #topics_Jaccard=[]
        J=Myjaccard(SNNMF_topics[i],tkm_topics[i])
        #print(J)
        All_topics_Jaccard=str(i)+","+SNNMF_topics[i].replace('\n',',')+tkm_topics[i].replace('\n',',')+str(J)+"\n"
        #print(All_topics_Jaccard)
        #topics_Jaccard.append(SNNMF_topics[i])
        #topics_Jaccard.append(tkm_topics[i])
        #topics_Jaccard.append(J)
        #All_topics_Jaccard.append(topics_Jaccard)
        JC_files.write(All_topics_Jaccard)
    #print(All_topics_Jaccard)
    JC_files.close()
#tkmf='/media/fsg/74C86089C8604C04/download/topic_interpretability-master/data/Topics_Reouter.txt'
#SNNMFf='/media/fsg/74C86089C8604C04/download/topic_interpretability-master/data/ReuterAll_top5_terms_topics.txt'
#jc='/media/fsg/74C86089C8604C04/Futuure-paper/evaluation/Jaccard/Reouter.csv'
#Jaccard_All(tkmf,SNNMFf,jc)

# selsect max5 term per topic from tkmlgenerated topic to test in interprobabilty
def tkml_topic_to_intropapilty_format(sourcefile,destnation):
    #''/media/fsg/74C86089C8604C04/download/tkm-master/output/Topics_20newsgroup.txt'
    #'/media/fsg/74C86089C8604C04/download/tkm-master/output/Topics_20newsgroup.csv'
    n=open(destnation,'a')
    with(open(sourcefile,'r')) as f:
            #line=f.readlines()

            lines=f.readlines()
            l=0

            for line in lines:
                #if l<100:
                terms_value_list=line.split(',')
                i=0
                text=''
                for t in terms_value_list:
                    if i<5:
                        #print(t)
                        term=t.split(" ")[-2]
                        #print(term)
                        text=text+term+" "
                        i+=1
                text=text+"\n"
                #print(text)
                n.write(text)
                #print("==============================")
                l+=1
            print(l)
            print(len(lines))
    n.close()


#tkml_topic_to_intropapilty_format('/media/fsg/74C86089C8604C04/download/tkm-master/output/Topic_generated/Topics_BBC.csv','/media/fsg/74C86089C8604C04/download/tkm-master/output/Topic_test/Topics_BBC.txt')


from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic as wnic
from nltk.tokenize import word_tokenize

def similarity_by_infocontent(sense1, sense2, option):
    #sense1="Synset('"+sense1+"')"
    #sense2="Synset('"+sense2+"')"
    sense1 = wn.synset(sense1)
    sense2 = wn.synset(sense2)
    #print(sense1,sense2)
    """ Returns similarity scores by information content. """
    #if sense1.pos != sense2.pos: # infocontent sim can't do diff POS.
        #return 0

    info_contents = ['ic-bnc-add1.dat', 'ic-bnc-resnik-add1.dat', 
                     'ic-bnc-resnik.dat', 'ic-bnc.dat', 

                     'ic-brown-add1.dat', 'ic-brown-resnik-add1.dat', 
                     'ic-brown-resnik.dat', 'ic-brown.dat', 

                     'ic-semcor-add1.dat', 'ic-semcor.dat',

                     'ic-semcorraw-add1.dat', 'ic-semcorraw-resnik-add1.dat', 
                     'ic-semcorraw-resnik.dat', 'ic-semcorraw.dat', 

                     'ic-shaks-add1.dat', 'ic-shaks-resnik.dat', 
                     'ic-shaks-resnink-add1.dat', 'ic-shaks.dat', 

                     'ic-treebank-add1.dat', 'ic-treebank-resnik-add1.dat', 
                     'ic-treebank-resnik.dat', 'ic-treebank.dat']

    if option in ['res', 'resnik']:
        #return wn.res_similarity(sense1, sense2, wnic.ic('ic-bnc-resnik-add1.dat'))
        #print('simRe snik (c1,c2) = -log p(lso(c1,c2)) = IC(lso(c1,c2)')
        return wn.res_similarity(sense1, sense2, wnic.ic('ic-treebank-resnik-add1.dat'))
    #return min(wn.res_similarity(sense1, sense2, wnic.ic(ic)) \
    #             for ic in info_contents)

    elif option in ['jcn', "jiang-conrath"]:
        #return wn.jcn_similarity(sense1, sense2, wnic.ic('ic-bnc-add1.dat'))
        #print('sim(jcn) (c1,c2 )= (IC(c1) + IC(c2 )) - 2IC(lso(c1,c2 ))')
        return wn.jcn_similarity(sense1, sense2, wnic.ic('ic-treebank.dat'))

    elif option in ['lin']:
        #return wn.lin_similarity(sense1, sense2, wnic.ic('ic-bnc-add1.dat'))
        #print('sim(lin) (c1,c2)=(2IC(lso(c1,c2 )))/(IC(c1)+IC(c2))')
        return wn.lin_similarity(sense1, sense2, wnic.ic('ic-treebank.dat'))

def sim(sense1, sense2, option="path"):
    """ Calculates similarity based on user's choice. """
    option = option.lower()
    if option.lower() in ["path", "path_similarity", 
                        "wup", "wupa", "wu-palmer", "wu-palmer",
                        'lch', "leacock-chordorow"]:
        return similarity_by_path(sense1, sense2, option) 
    elif option.lower() in ["res", "resnik",
                          "jcn","jiang-conrath",
                          "lin"]:
        return similarity_by_infocontent(sense1, sense2, option)

def max_similarity(context_sentence, ambiguous_word, option="path", 
                   pos=None, best=True):
    """
    Perform WSD by maximizing the sum of maximum similarity between possible 
    synsets of all words in the context sentence and the possible synsets of the 
    ambiguous words (see http://goo.gl/XMq2BI):
    {argmax}_{synset(a)}(\sum_{i}^{n}{{max}_{synset(i)}(sim(i,a))}
    """
    result = {}
    for i in wn.synsets(ambiguous_word):
        try:
            if pos and pos != str(i.pos()):
                continue
        except:
            if pos and pos != str(i.pos):
                continue
        result[i] = sum(max([sim(i,k,option) for k in wn.synsets(j)]+[0]) \
                        for j in word_tokenize(context_sentence))

    if option in ["res","resnik"]: # lower score = more similar
        result = sorted([(v,k) for k,v in result.items()])
    else: # higher score = more similar
        result = sorted([(v,k) for k,v in result.items()],reverse=True)
    #print (result)
    if best: return result[0][1];
    return result

def semantic_network_less_than_1(list_terms,imagename):
    import networkx as nx
    import numpy as np
    import matplotlib.pyplot as plt
    import pylab
    #%matplotlib inline  
    #list_terms=['microphone.n.01','range.n.04','cam.n.02','volt.n.01','mistake.n.01','mode.n.06','fiberglass.n.01','million.n.01','turk.n.01','keyboard.n.01']
    #list_terms=['m','r','c','v']
    G = nx.DiGraph()
    #G.add_edges_from([('A', 'A')], weight=5)
    for i in range(len(list_terms)):
        for x in range(i,len(list_terms)):
            if i !=x:

                a = (list_terms[i],)
                #print(a)
                l = list(a)
                #if x <len(list_terms)-1:
                l.append(list_terms[x])
                b=tuple(l)
                w=similarity_by_infocontent(list_terms[i],list_terms[x],'res')
                #w=i**x
                #print(b,w)
                G.add_edges_from([b], weight=w)


    val_map = {'microphone.n.01': 100.0,
               'range.n.04': 75,
               'cam.n.02': 50,
              'volt.n.01':52}

    values = [val_map.get(node, 0.45) for node in G.nodes()]
    edge_labels=dict([((u,v,),d['weight']) for u,v,d in G.edges(data=True)])
    #print(edge_labels.values())
    #red_edges = [('C','D'),('D','A')]
    #edge_colors = ['black' if not edge in red_edges else 'red' for edge in G.edges()]
    edge_colors = ['black' if edge >=1 else 'red' for edge in edge_labels.values()]

    pos=nx.layout.shell_layout(G)#spring_layout(G)

    pylab.figure(3,figsize=(20,20))

    nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
    s=nx.draw(G,pos, node_color = values, node_size=7550,edge_color=edge_colors,edge_cmap=plt.cm.Reds,with_labels=True)
    #G = nx.DiGraph(directed=True)
    #import matplotlib.pyplot as plt
    #plt.rcParams["figure.figsize"] = (2000,3000)
    #pylab.plot(.1,.2)
    #plt.figure(figsize=(300, 300))
    #pylab.size(300*300)
    plt.savefig(imagename, format="PNG", dpi=300)
    #pylab.savefig(imagename, format="PNG")
    #plt.close()
    pylab.show()
list_terms=['information_technology.n.01','there.n.01','associate_in_nursing.n.01','second.n.01','nobelium.n.01']
img='/media/fsg/74C86089C8604C04/Futuure-paper/evaluation/Jaccard/TKM_20Nwesgroups.png'
semantic_network_less_than_1(list_terms,img)
