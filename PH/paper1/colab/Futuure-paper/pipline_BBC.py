#! /usr/bin/python3.6
import text_processing as tp
import Utils as u

def pipeline_BBC_dataset():
    #docs=tp.read20newsgroups()
    #Save Docs
    #header_list=['index','Doc']
    #u.save_file_to_database(docs,u.path_database,u.NewsGroupDataset,header_list)
    #Conver docs list to lesk list
    #tp.docs_list_to_lesk_list(u.path_database,u.BBCDataset,u.BBCDataset_lesk)
    # TF_IDF
    #tp.tfidf(u.path_database,u.BBCDataset_lesk,u.BBCDataset_lesk_ifidf)
    # to get top 100 topic and doc 
    #tp.sub_max_n_topic_tfidf_n_doc(u.path_database,u.BBCDataset_lesk_ifidf,u.BBCDataset_ifidf_top_n,100)
    #calculate H1
    #f=open(u.path_database+u.BBCDataset_ifidf_top_n) 
    #terms=f.readline().split(',')[1:]
    #terms=[term.replace("'",'') for term in terms]
    #print(terms[0:3])
    #tp.sim_docs_lesk(terms,'res',u.BBCDataset_H1,u.path_database)
    #Compute H2
    #tp.threshold_correlation_coefficient\
    #(u.read_cvs_by_pands(u.path_database,u.BBCDataset_H1,index_col=0,header=0),u.path_database,u.BBC_H2)
    #Compute H3,H4
    #tp.permutaion_correlation_coefficient(u.path_database,u.BBC_H2,u.BBC_H4)
    #Compute W
    #tp.nmf_W_res_sem_sim_permutaion_correlation_coefficient\
    #(u.BBCDataset_ifidf_top_n,u.BBC_H4,u.path_database,u.BBC_W_H4,u.BBC_H4_inv)
    #Comute Traditional NMF 
    tp.traditional_NMF(u.path_database,u.BBCDataset_ifidf_top_n,u.BBC_H_S,u.BBC_W_S)
    

pipeline_BBC_dataset()
