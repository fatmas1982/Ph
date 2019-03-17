#! /usr/bin/python3.6

import Utils as u
import pandas as pd

def count_number_topic_forword_TKM(path_evaluation,file_dataset,word):
    #'/media/fsg/74C86089C8604C04/download/tkm-master/output/AllWords/BBCTopics.csv'
    with open(path_evaluation+file_dataset,'r') as f:
        #word='deoxyadenosine_monophosphate.n.01'
        listlinse=f.readlines()
        counter=0
        for i in range(len(listlinse)):
            if word in listlinse[i]:
                sublist=listlinse[i].split(',')
                for s in range (len(sublist)):
                    if word in sublist[s]:
                        #print(i)
                        value=sublist[s][len(word):].split(' ')[1].replace(' ','')
                        #print(value)
                        if float(value) > 0:
                            #print(value)
                            counter+=1



    return counter


def count_number_topic_term_forword_TKM(path_evaluation,file_dataset,word):
    #'/media/fsg/74C86089C8604C04/download/tkm-master/output/AllWords/BBCTopics.csv'
    with open(path_evaluation+file_dataset,'r') as f:
        #word='deoxyadenosine_monophosphate.n.01'
        listlinse=f.readlines()
        counter=0
        val={}
        for i in range(len(listlinse)):
            if word in listlinse[i]:
                sublist=listlinse[i].split(',')
                for s in range (len(sublist)):
                    if word in sublist[s]:
                        #print(i)
                        value=sublist[s][len(word):].split(' ')[1].replace(' ','')
                        #print(value)
                        if float(value) > 0:
                            #print(value)
                            val[i]=float(value)
                            counter+=1
        #print(val)
        max_n_topics= sorted([(value,key) for (key,value) in val.items()])[:5]
        #print(d[0])#[1])
        
        sub_dict={}
        for g in range(len(max_n_topics)):
            index=max_n_topics[g][1]
            #print(index)
            sublist=listlinse[index].split(',')[1:]
            #print(sublist)
            #print("=================")
            sub_count=0
            for s in range (len(sublist)):
                value=sublist[s].split(' ')[-1].replace(' ','')
                #print(value)
                try:
                    if float(value) > 0:
                        sub_count+=1
                               
                except ValueError:
                    value=0
                    
            sub_dict[g]=sub_count
            

    sorted_sub_dict= sorted(sub_dict.values(), reverse=True)#sorted([(value) for (key,value) in sub_dict.items()])
    #ub_dict.sorted()
    
    #print(sub_dict.values())
    return sorted_sub_dict


def count_top_topic(keyword,H1_df,H2_df,H4_df,H_tkm,path_database,path_tkm,DatasetName):
    #header=['Topic 1','Topic 2','Topic 3','Topic 4','Topic 5']
    header=['SNNMF_H1','SNNMF_H2','SNNMF_H4','TKM']
    row=[]
    
    H1_20=u.read_cvs_by_pands(path_database,H1_df,index_col=0,header=0)
    H1_20_max=H1_20.loc[keyword].sort_values(ascending=False)
    df_H1=pd.DataFrame(H1_20_max)
    r1=(df_H1 != 0).sum(axis=0)#columns
    #print(r1)
    row.append(r1.values[0])
    
    
    H2_20=u.read_cvs_by_pands(path_database,H2_df,index_col=0,header=0)
    H2_20_max=H2_20.loc[keyword].sort_values(ascending=False)
    df_H2=pd.DataFrame(H2_20_max)
    r2=(df_H2 != 0).sum(axis=0)#columns
    #print(r2)
    row.append(r2.values[0])
    
    H4_20=u.read_cvs_by_pands(path_database,H4_df,index_col=0,header=0)
    H4_20_max=H4_20.loc[keyword].sort_values(ascending=False)
    df_H4=pd.DataFrame(H4_20_max)
    r4=(df_H4 != 0).sum(axis=0)#columns
    #print(r4)
    row.append(r4.values[0])
    
    
    row.append(count_number_topic_forword_TKM(path_tkm,H_tkm,keyword))
    
    rows=[]
    rows.append(row)
    #print(rows)
    df = pd.DataFrame(rows,columns=header,index=[DatasetName])#'''
    return df#r#H4_20[H4_20_max5.index]

def autolabel(rects,ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 3+height,
                '%d' % int(height+.5),
                ha='center', va='bottom')
    


def plot_reuls_topic(jobname,dataset,path_database):#,cpu_cs,cpu_spliit,gpu_cs,gpu_split):

    import numpy as np
    import matplotlib.pyplot as plt
    #%matplotlib inline 
    # data to plot
    n_groups = 1
    job_name=jobname
    job_path=job_name+'/'
   
    F1 =dataset['SNNMF_H1'].values
    #print(tuple(F1))
    F2 =dataset['SNNMF_H2'].values
    F3 =dataset['SNNMF_H4'].values
    F4 =dataset['TKM'].values
    

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.1
    opacity = 1
    patterns = [ "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]

    rects_F1 = plt.bar(index, F1, bar_width,
                     alpha=opacity,
                     color='W',
                     label='SNNMF_H1',edgecolor='black', hatch=patterns[3])

    rects_F2 = plt.bar(.1+index + bar_width, F2, bar_width,
                     alpha=opacity,
                     color='W',
                     label='SNNMF_H2',edgecolor='black', hatch=patterns[5])

    rects_F3 = plt.bar(.1+index + bar_width*2+.1, F3, bar_width,
                     alpha=opacity,
                     color='W',
                     label='SNNMF_H4',edgecolor='black', hatch=patterns[9])
    
    rects_F4 = plt.bar(.1+index + bar_width*2+.3, F4, bar_width,
                     alpha=opacity,
                     color='W',
                     label='TKM',edgecolor='black', hatch=patterns[1])


    plt.xlabel('Methods')
    plt.ylabel('Topics Number Not Equal Zero')
    plt.title(" ")
    plt.xticks([index,index+.2,index+.4,index+.6] , ('SNNMF_H1', 'SNNMF_H 2', 'SNNMF_H 4','TKM'))#,rotation=90)
    plt.legend(loc='upper center', bbox_to_anchor=(1.15, 1), shadow=True, ncol=1)
    

    plt.tight_layout()
    autolabel(rects_F1,ax)
    autolabel(rects_F2,ax)
    autolabel(rects_F3,ax)
    autolabel(rects_F4,ax)

    #plt.show()
    plt.savefig(path_database+job_name+'.png', dpi=1200, format='png', bbox_inches='tight') 
    # use format='svg' or 'pdf' for vectorial pictures







def plot_reuls(jobname,dataset,path_evaluation):#,cpu_cs,cpu_spliit,gpu_cs,gpu_split):

    import numpy as np
    import matplotlib.pyplot as plt
    #%matplotlib inline 
    # data to plot
    n_groups = 5
    job_name=jobname
    job_path=job_name+'/'
   
    F1 =dataset.loc['H1'].values
    #print(tuple(F1))
    F2 =dataset.loc['H2'].values
    F3 =dataset.loc['H4'].values
    F4 =dataset.loc['TKM'].values
    

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.1
    opacity = 1
    patterns = [ "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]

    rects_F1 = plt.bar(index, F1, bar_width,
                     alpha=opacity,
                     color='W',
                     label='H1',edgecolor='black', hatch=patterns[3])

    rects_F2 = plt.bar(.1+index + bar_width, F2, bar_width,
                     alpha=opacity,
                     color='W',
                     label='H2',edgecolor='black', hatch=patterns[5])

    rects_F3 = plt.bar(.1+index + bar_width*2+.1, F3, bar_width,
                     alpha=opacity,
                     color='W',
                     label='H4',edgecolor='black', hatch=patterns[9])
    
    rects_F4 = plt.bar(.1+index + bar_width*2+.3, F4, bar_width,
                     alpha=opacity,
                     color='W',
                     label='TKM',edgecolor='black', hatch=patterns[1])


    plt.xlabel('Topics')
    plt.ylabel('Number of Terms Not Equal Zero')
    plt.title(" ")
    plt.xticks(index + bar_width, ('Topic 1', 'Topic 2', 'Topic 3','Topic 4','Topic 5'))
    plt.legend(loc='upper center', bbox_to_anchor=(1.15, 1), shadow=True, ncol=1)#frameon=False, loc='best', ncol=1)
    

    plt.tight_layout()
    autolabel(rects_F1,ax)
    autolabel(rects_F2,ax)
    autolabel(rects_F3,ax)
    autolabel(rects_F4,ax)

    #plt.show()
    plt.savefig(path_evaluation+job_name+'.png', dpi=1200, format='png', bbox_inches='tight') 
    # use format='svg' or 'pdf' for vectorial pictures


def count_top_term_top_topic(keyword,H1_df,H2_df,H4_df,tkm_H,top_topic,path_database,path_tkm):
    header=['Topic 1','Topic 2','Topic 3','Topic 4','Topic 5']
    index=['H1','H2','H4','TKM']
    rows=[]
    H1_20=u.read_cvs_by_pands(path_database,H1_df,index_col=0,header=0)
    H1_20_max=H1_20.loc[keyword].sort_values(ascending=False)
    H1_20_max5=H1_20_max[:top_topic]  
    print("H1_20_max5.index",H1_20_max5.index)
    r1=(H1_20[H1_20_max5.index] != 0).sum(axis=0)#columns
    rows.append(r1.values)
    print(r1)
    #print("----------------------------")
    
    H2_20=u.read_cvs_by_pands(path_database,H2_df,index_col=0,header=0)
    H2_20_max=H2_20.loc[keyword].sort_values(ascending=False)
    H2_20_max5=H2_20_max[:top_topic]   
    print("H2_20_max5.index",H2_20_max5.index)
    r2=(H2_20[H2_20_max5.index] != 0).sum(axis=0)#columns
    rows.append(r2.values)
    #print(H2_20_max)
    #print("----------------------------")
    H4_20=u.read_cvs_by_pands(path_database,H4_df,index_col=0,header=0)
    H4_20_max=H4_20.loc[keyword].sort_values(ascending=False)
    H4_20_max5=H4_20_max[:top_topic]   
    print("H4_20_max5.index",H4_20_max5.index)
    r4=(H4_20[H4_20_max5.index] != 0).sum(axis=0)#columns
    rows.append(r4.values)
    #print(H4_20_max)
    #print("----------------------------")
   
    rows.append(count_number_topic_term_forword_TKM(path_tkm,tkm_H,keyword))
    #print(H_s_20_max5.index)
    #rows.append(r_s.valuse())
   
    df = pd.DataFrame(rows,columns=header,index=index)#
    return df,H1_20_max5,H2_20_max5,H4_20_max5#r#H4_20[H4_20_max5.index]

def count_number_topic_term_forword_TKM(path_evaluation,file_dataset,word):
    #'/media/fsg/74C86089C8604C04/download/tkm-master/output/AllWords/BBCTopics.csv'
    with open(path_evaluation+file_dataset,'r') as f:
        #word='deoxyadenosine_monophosphate.n.01'
        listlinse=f.readlines()
        counter=0
        val={}
        for i in range(len(listlinse)):
            if word in listlinse[i]:
                sublist=listlinse[i].split(',')
                for s in range (len(sublist)):
                    if word in sublist[s]:
                        #print(i)
                        value=sublist[s][len(word):].split(' ')[1].replace(' ','')
                        #print(value)
                        if float(value) > 0:
                            #print(value)
                            val[i]=float(value)
                            counter+=1
        #print(val)
        max_n_topics= sorted([(value,key) for (key,value) in val.items()])[:5]
        #print(d[0])#[1])
        
        sub_dict={}
        for g in range(len(max_n_topics)):
            index=max_n_topics[g][1]
            #print(index)
            sublist=listlinse[index].split(',')[1:]
            #print(sublist)
            #print("=================")
            sub_count=0
            for s in range (len(sublist)):
                value=sublist[s].split(' ')[-1].replace(' ','')
                #print(value)
                try:
                    if float(value) > 0:
                        sub_count+=1
                               
                except ValueError:
                    value=0
                    
            sub_dict[g]=sub_count
            

    sorted_sub_dict= sorted(sub_dict.values(), reverse=True)#sorted([(value) for (key,value) in sub_dict.items()])
    #ub_dict.sorted()
    
    #print(sub_dict.values())
    return sorted_sub_dict
'''
Return matrix terms per top -topic
''' 
def top_term_top_topic(keyword,H_df,top_topic,top_term,path_database,path_evaluation):
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



#top_term_top_topic('stage_set.n.01',u.Brown_H4,5, 10, u.path_database,u.path_evaluation)
'''
Return matrix top -topic only not term in rows
''' 
def top_term_top_topic_all(keyword,H_df,top_topic,top_term,path_database,path_evaluation):
  
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

#top_term_top_topic_all('stage_set.n.01',u.Brown_H4,5, 10, u.path_database,u.path_evaluation)


'''
get max 5 terms per each topic in curps 
'''

def top5_terms_per_Alltopic(path_database,H4,path_evaluation):
    df_h4=u.read_cvs_by_pands(path_database,H4,index_col=0,header=0)
    count_nonzero=df_h4.astype(bool).sum(axis=1)
    count_nonzero.sort_values(inplace=True,ascending=False)
    #count_nonzero
    max_selected_colums=count_nonzero.nlargest(100).index
    #print("ssss",max_selected_colums)
    df_h4_max=df_h4[max_selected_colums]
    #print("ffffffff",df_h4_max)
    d={}
    for i in range(len(df_h4_max.index)):

        list_topic_word=df_h4_max[df_h4_max.index[i]].sort_values(ascending=False)[:5]
        #print(list_topic_word)
        #print("************************")
        #pd.DataFrame(df)
        d[df_h4_max.index[i]]=list_topic_word.index
    #print(d)
    final_df=pd.DataFrame(d,columns=df_h4_max.index)
    df_topics=final_df.T
    name_file=H4.split('_')[0]+"All_top5_terms_topics.csv"
    df_topics.to_csv(path_evaluation+name_file)
    print("End")
    return df_topics


#top5_terms_per_Alltopic(u.path_database,u.Ohsumed_H4,u.path_evaluation)


