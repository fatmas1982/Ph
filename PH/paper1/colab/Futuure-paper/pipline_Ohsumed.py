#! /usr/bin/python3.6
import text_processing as tp
import Utils as u

def pipeline_Ohsumed_dataset():
    #to read files from foldersand convert it to CSV each row contain file
    #tp.convert_courpsfile_to_csv('/media/fsg/74C86089C8604C04/Futuure-paper/data_source/','ohsumed-all')
    #Conver docs list to lesk list
    #tp.docs_list_to_lesk_list(u.path_database,u.OhsumedDataset,u.OhsumedDataset_lesk)
    # TF_IDF
    #tp.tfidf(u.path_database,u.OhsumedDataset_lesk,u.OhsumedDataset_lesk_ifidf)
    # to get top 100 topic and doc 
    #tp.sub_max_n_topic_tfidf_n_doc(u.path_database,u.OhsumedDataset_lesk_ifidf,u.OhsumedDataset_ifidf_top_n,100)
    #calculate H1
    #f=open(u.path_database+u.OhsumedDataset_ifidf_top_n) 
    #terms=f.readline().split(',')[1:]
    #terms=[term.replace("'",'') for term in terms]
    #print(terms[0:3])
    #tp.sim_docs_lesk(terms,'res',u.OhsumedDataset_H1,u.path_database)
    #Compute H2
     #tp.threshold_correlation_coefficient(u.read_cvs_by_pands(u.path_database,u.OhsumedDataset_H1,index_col=0,header=0),u.path_database,u.Ohsumed_H2)
    #Compute H3,H4
    #tp.permutaion_correlation_coefficient(u.path_database,u.Ohsumed_H2,u.Ohsumed_H4)
    #Compute W      
    #tp.nmf_W_res_sem_sim_permutaion_correlation_coefficient\
    #(u.OhsumedDataset_ifidf_top_n,u.Ohsumed_H2,u.path_database,u.Ohsumed_W_H4,u.Ohsumed_H4_inv)
    #Comute Traditional NMF 
    #tp.traditional_NMF(u.path_database,u.OhsumedDataset_ifidf_top_n,u.Ohsumed_H_S,u.Ohsumed_W_S)
    

pipeline_Ohsumed_dataset()
