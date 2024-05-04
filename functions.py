# -*- coding: utf-8 -*-
import tqdm
import numpy as np 
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import functools
import seaborn as sns
import math
import itertools
import re
from re import search
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score,classification_report, accuracy_score,precision_score,recall_score



def add_important_articles(score_df,score_article,model_w2v,n=3):
    list_top = []
    for word in tqdm(list(score_df.word.values)):
        top_3 = score_article[score_article["abstract_cleaned"].str.contains(word)].sort_values("score",ascending=False)[:n]
        fi_dict = dict(zip(top_3.doi,top_3.score))
        list_top.append(fi_dict)
    score_df["top_important_articles"] = list_top
    return score_df


def add_similar_words(score_df,score_article,model_w2v,n=5):
    top_words = []
    for word in tqdm(list(score_df.word.values)):
        top_words.append(model_w2v.wv.most_similar(positive=[str(word)], topn=n))
    score_df["top_sim_words"] = top_words
    return score_df



def resulted_matrics_table(matrix_pred_real,frac_articles = 1):
    res_list_of_lists = []
    for name, pred, real in matrix_pred_real: 
        list_results = []
        predictions = [1 if i >=0.5 else 0 for i in pred]
        
        if frac_articles==1:
            name = name
        else:
            name = name
            
        if search("train",name):
            list_results.append(name)
        else:
            list_results.append(name)      
        
        list_results.append(roc_auc_score(real,pred))
        list_results.append(accuracy_score(real,predictions))
        list_results.append(precision_score(real,predictions))
        list_results.append(recall_score(real,predictions))
        res_list_of_lists.append(list_results)
    df_res = pd.DataFrame(res_list_of_lists,columns = ["dataset_of_predictions","AUC","Accuracy","Precision","Recall"])
    return df_res


def add_target_opencitatins_marginal(target_year,df,target_col_name,get_cum=False): 
    """
    Create target flag for dataset of doi (df) for selected target year.
    Is automatically derived from all doi of each year of publication in df -> 
    -> median of citations of articles published in year of publication of the articles. 
    Target = higly and lowly cited articles based on median of the year of publication. Derived from 
             marginal citations of the target_year which express relevance of articles that year.
    Needed columns: doi, Year (of publication)
    
    """    
    df_cit_year = pd.read_csv("sources/citationcounts_oci_revised_year.csv",on_bad_lines="skip",encoding="utf-8")
    df_cit_year = df_cit_year.rename(columns={'count_opencitations': 'Year_of_citations',"year;;;;;;":"OpenCitations","doi_x":"doi"})
    df_cit_year['OpenCitations'] =  df_cit_year['OpenCitations'].str.extract(r'(\d+)', expand=False)
    df_cit_year = df_cit_year[df_cit_year['Year_of_citations']!="?"]
    df_cit_year = df_cit_year[df_cit_year.pmc.notnull()]
    df_cit_year = df_cit_year[['doi',"Year_of_citations",'OpenCitations']].dropna()
    df_cit_year['OpenCitations'] = df_cit_year['OpenCitations'].astype(int)
    df_cit_year_pivot = pd.pivot_table(df_cit_year, values='OpenCitations', index=['doi'], columns=['Year_of_citations'], aggfunc=np.sum)

     # filter citations for target year
    df_cited_in_target_year = df_cit_year_pivot[[num for num in list(df_cit_year_pivot.columns) if int(num)==target_year]] 
    
    # for target - needed to have not null citations in target year (citations starting with 1 citations)
    df_cited_in_target_year = df_cited_in_target_year[pd.notnull(df_cited_in_target_year[str(target_year)])] 
    df_cited_in_target_year.columns = ["OpenCitations"] # rename column of citations to target column
   
    if get_cum == True:
        # filter citations for target years
        df_cited_in_target_year = df_cit_year_pivot
        df_cit_year_pivot["OpenCitations"] = df_cited_in_target_year.sum(axis=1)
 
    
    # derive target based on median of year of publication 
    df_joined = df.merge(df_cited_in_target_year,on="doi",how="inner") # joined with data to have there additional information

    median_of_year = df_joined[["Year","OpenCitations"]].groupby("Year"
                      ).median().reset_index().rename(columns={"OpenCitations": 'median_cit'}) 

    df_with_med =  df_joined.merge(median_of_year, on="Year", how="left")
    df_with_med[target_col_name] =  np.where( df_with_med["OpenCitations"]> df_with_med["median_cit"],1, 0)
    print(df_with_med[target_col_name].value_counts())
    
    return df_with_med



def old_target_cumulative_citations_to_2021(df):
    """
    Create target flag for dataset of doi (df).Is automatically derived from all doi of each year of publication in df -> 
    -> median of citations of articles published in year of publication of the articles. 
    Target = higly and lowly cited articles based on median of the year of publication. Derived from 
             cumulative citations (2021).
    Needed columns: doi, Year (of publication)
    
    """    
    df_cit = pd.read_csv("sources/citationcounts_oci_revised.csv",error_bad_lines=False,encoding="utf-8")
    df_cit = df_cit.rename(columns={'count_opencitations;;;;;;': 'OpenCitations'})
    df_cit['OpenCitations'] =  df_cit['OpenCitations'].str.extract(r'(\d+)', expand=False)
    df_cit = df_cit[['doi','OpenCitations']].dropna()
    df_cit['OpenCitations'] = df_cit['OpenCitations'].astype(int)

    df_merged = df_cit.merge(df, on="doi",how="inner") # want to have articles from Kaggle with opencitations
    df_merged = df_merged[pd.notnull(df_merged['Year'])] # for target - needed to have not null year
    median_of_year = df_merged[["Year","OpenCitations"]].groupby("Year").median().reset_index().rename(columns={
        'OpenCitations': 'median_cit'})
    df_with_med = df_merged.merge(median_of_year, on="Year", how="left")
    df_merged["target"] =  np.where(df_with_med['OpenCitations']> df_with_med["median_cit"],1, 0)
    return df_merged



def indiv_word_target_info_per_target(word, years, df_all, df_target_all_years):
    
    df_with_targets = df_target_all_years.join(df_all[["doi","abstract_cleaned"]].set_index("doi"),how="left")
    df_with_targets["is_in_article"] = list(df_with_targets.abstract_cleaned.str.contains(word).values)
    df_with_targets = df_with_targets.reset_index()
    
    dfs = []
    cols_l = []
    
    for year_cit in years:
        
        if year_cit in [2017,2018,2019]:
            score_of_word = pd.read_csv('3.Classifiers_outputs/train_'+str(year_cit)+'/'+"FI_LR_W2V.csv")[["word","score"]]
            score_of_word_bow = pd.read_csv('3.Classifiers_outputs/train_'+str(year_cit)+'/'+"FI_LR_BOW.csv")[["word","score"]]     
            score_of_word_f = score_of_word.merge(score_of_word_bow,on="word",how="left")
            score_of_word_f = score_of_word_f[["score_x","score_y","word"]].rename({"score_x": "score_w2v", "score_y": "score_bow"},axis='columns')
            
            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(score_of_word_f[["score_bow","score_w2v"]]) 
            scaled_data = pd.DataFrame(scaled_data,index = score_of_word_f.word, columns = ["score_bow_sc","score_w2v_sc"])
            scaled_data = scaled_data.reset_index()
            score_of_word_f = score_of_word_f.merge(scaled_data,on="word")
            
            score_of_word_f = score_of_word_f[score_of_word_f["word"]==word]
    
       
        if not year_cit in [2017,2018,2019]:
            score_of_word_f = pd.DataFrame(np.nan, index=[0], columns=['score_w2v',"score_bow","word","score_bow_sc","score_w2v_sc"])
            score_of_word_f["word"] = word
        
 
            
        # pocet clanku kde je slovo a citace pro dany rok jsou nenulove
        score_of_word_f["cnt_with_targ_"+str(year_cit)] = df_with_targets[["doi","OpenCitations_"+str(year_cit),
                             "target_"+str(year_cit),"is_in_article"]].dropna()["is_in_article"].fillna(0).astype(int).sum()
           

        # pocet clanku kde je slovo a citace pro dany rok jsou nenulove a zaroven clanek je highly cited
        score_of_word_f['%_of_high_'+str(year_cit)] = (df_with_targets[(df_with_targets["target_"+str(year_cit)]==1 )][["doi","OpenCitations_"+str(year_cit),
                                                    "target_"+str(year_cit),"is_in_article"
                                        ]].dropna()["is_in_article"].fillna(0).astype(int).sum())/df_with_targets[["doi","OpenCitations_"+str(year_cit),
                                                    "target_"+str(year_cit),"is_in_article"
                                        ]].dropna()["is_in_article"].fillna(0).astype(int).sum()
                                                       
        cols = list(score_of_word_f.columns)
        score_of_word_f["year_cit"] = year_cit
        score_of_word_f = score_of_word_f.drop(["Unnamed: 0"], axis=1, errors='ignore')
        
        if not score_of_word_f.empty:
            cols_l.append(cols)   
            dfs.append(score_of_word_f)
            
    cols_l =  list(itertools.chain(*cols_l))
    cols_l = list( dict.fromkeys(cols_l) )
    fin = pd.DataFrame(np.vstack(dfs),columns = ["score_w2v","score_bow","word","score_bow_sc","score_w2v_sc","cnt","perc","year_cit"] )
    
    return fin




def roc(y_test, probs):
    
    """
    y_test_df has to have same column names as probs_df.
    """          
    if isinstance(probs,np.ndarray)==False:
        for probs_col in probs.columns:
            probs_df_f = probs[[probs_col]].values.ravel()
            y_test_df_f = y_test[[probs_col]].values.ravel()
            print('Model '+str(probs_col)+': ROC AUC=%.3f'  % (roc_auc_score(y_test_df_f, probs_df_f)))
            lr_fpr, lr_tpr, _ = roc_curve(y_test_df_f, probs_df_f)
            plt.plot(lr_fpr, lr_tpr, marker='.', label='Model '+str(probs_col))
        
        y_test_rand = y_test.iloc[:, 0]
        ns_probs = [0 for _ in range(len(y_test_rand))]
        ns_auc = roc_auc_score(y_test_rand, ns_probs)
        ns_fpr, ns_tpr, _ = roc_curve(y_test_rand, ns_probs)
        plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        print('No Skill: ROC AUC=%.3f' % (ns_auc))
    
    if isinstance(probs,np.ndarray)==True:
        print('Model '+': ROC AUC=%.3f'  % (roc_auc_score(y_test, probs)))
        lr_fpr, lr_tpr, _ = roc_curve(y_test, probs)
        plt.plot(lr_fpr, lr_tpr, marker='.', label='Model ')
        ns_probs = [0 for _ in range(len(y_test))]
        ns_auc = roc_auc_score(y_test, ns_probs)
        ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
        plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        print('No Skill: ROC AUC=%.3f' % (ns_auc))

    
    plt.xlabel('False Positive Rate')
    plt.legend()
    return plt.show()


def x_first_last_val(results,perc):
    vysledky_asc = results.sort_values("OpenCitations",ascending=False)
    delka = round(len(vysledky_asc)*perc)
    vysledky_asc = vysledky_asc.iloc[:delka]
    vysledky_des = results[results["OpenCitations"]<=min(results["OpenCitations"])].sample(len(vysledky_asc),random_state=0)
    final_vysledky = pd.concat([vysledky_des, vysledky_asc], ignore_index=True)
    return final_vysledky

def prob_plots(probs, y_test): 
    results = pd.DataFrame(zip(list(probs),list(y_test)),columns=["y_pred","real"])
    plt.figure(figsize=(15,3))
    plt.subplot(131)
    sns.boxplot(x="real",y="y_pred",data=results)
    plt.subplot(132)
    sns.violinplot(x="real",y="y_pred",data=results, showfliers=False)
    plt.subplot(133)
    sns.kdeplot(results[results["real"]==0].y_pred)
    sns.kdeplot(results[results["real"]==1].y_pred)
    return plt.show()


def transform_to_document_vector(text_col_tokenized,
                                 model,
                                 index_col_list,
                                 agg_func = "avg" # or sum
                                ):
    doc_vectors_final = []
    for doc in tqdm(text_col_tokenized):
        doc_vectors = []
        word_names = []
   
    
        for word in doc:
    #        if word in model.wv.vocab:
             if word in model.wv.index_to_key:
    #           doc_vectors.append(model[word])
                doc_vectors.append(model.wv[word])
                word_names.append(word)
                
        doc_vectors_pd = pd.DataFrame(doc_vectors)
        doc_vectors_pd.index = word_names
        
        if agg_func == "avg":
            doc_vector = list(doc_vectors_pd.mean())
        
        if agg_func == "sum":
            doc_vector = list(doc_vectors_pd.sum())
            
        doc_vectors_final.append(doc_vector)
    
    doc_vectors_final_pd = pd.DataFrame(doc_vectors_final)
    doc_vectors_final_pd.index = index_col_list
    
    return doc_vectors_final_pd


def tokenized_column(string_col
                    ):  
    tokenized_list=[]
    for i in list(string_col):
        i = str(i)
        li = list(i.split(' '))
        tokenized_list.append(li)
    return tokenized_list

def remove_punc_lower(text):   
    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    # I dont want to remove dash, because dash is connection between words!
    remove = string.punctuation
    remove = remove.replace("-", "")     
    remove = remove.replace("/", "")
    pattern = r"[{}]".format(remove)
    text = re.sub(pattern, "", str(text).lower().strip())    
    return text

def binned_true_positives_cumsum(y_true,
                                 y_probas,
                                 bin_size= 1000,  
                                 order_ascending = False,
                                 plot_bins= True
                      ):
    
    # converting formats: 
    if isinstance(y_probas, pd.Series):
        y_probas = y_probas.to_frame()
    
    if isinstance(y_probas, np.ndarray):
        y_probas = pd.DataFrame(y_probas)
        y_probas.columns = y_probas.columns.astype(str)
        
    if isinstance(y_true, pd.Series):
        y_true = y_true.values

    # Create aggregated dataframe with cumulative counts of TP:
    dfs = []
    for probs in list(y_probas.columns):
        probs_df_with_y_true = pd.DataFrame({'y_true':y_true, probs:y_probas[probs].values}).sort_values(probs,ascending = order_ascending)
        probs_df_with_y_true["bin_index"] = np.arange(len(probs_df_with_y_true)) // bin_size 
        df_agg = probs_df_with_y_true.groupby("bin_index",as_index=False).agg({'y_true':['count','sum']})
        df_agg.columns = [col[1] for col in df_agg.columns]
        df_agg = df_agg.rename(columns={"count": "Total positive ", "sum": probs}).reset_index()
        df_agg[probs+"_cum"] = df_agg[probs].cumsum()
        globals()[probs] =df_agg
        dfs.append(globals()[probs])

    df_all = functools.reduce(lambda left,right: pd.merge(left,right,on='index',how='outer'),dfs)
    
    if plot_bins:
        plt.figure(figsize=(20, 10))
        plt.title("Graph of cumulative sum of highly cited articles ")
        for probs in list(y_probas.columns):
            plt.plot(df_all["index"], df_all[probs+"_cum"], label=probs+"_cum")
        plt.ylabel("Cumulative_highly_cited_articles_sum")
        plt.xlabel("bin")
        plt.legend()
    
    return df_all


def generate_target_per_sel_years(list_years ,df_all ):
    
    years_save = []
    for year in list_years:
        new_name_cit = "OpenCitations_"+str(year)
        new_name_targ = "target_"+str(year)
        df_all_add_t = df_all_add_t.rename({'OpenCitations': new_name_cit, 'target': new_name_targ}, axis=1)
        df_all_add_t = df_all_add_t.set_index("doi")
        years_save.append(df_all_add_t)
 
    df_fin = pd.concat(years_save, axis=0)
    cols = list(df_fin.columns)
    df_target_all_years = df_fin.reset_index().groupby("doi").sum(cols , min_count=1)
    
    return df_target_all_years



def indiv_word_statistics_per_years(word, df_all,df_with_target,score_of_word):
    
 
    arr = df_all.reset_index()[df_all.reset_index()['abstract_cleaned'].fillna("").str.contains(word)]["Year"].unique()
    years = list(arr)
    df_with_target["is_in_article"] = list(df_with_target.reset_index().abstract_cleaned.str.contains(word).values)
    df_all["is_in_article"] = list(df_all.reset_index().abstract_cleaned.str.contains(word).values)
    
    dfs = [] 
    for year in years:
        score_of_word_f = score_of_word[score_of_word["word"]==word]
        score_of_word_f["Year"] = year
        
        df_with_target_f = df_with_target[df_with_target["Year"]==year]
        score_of_word_f["cnt_of_art_with_targ"] = (df_with_target_f[df_with_target_f["Year"]
                                                ==year]["is_in_article"].fillna(0).astype(int).sum())
        score_of_word_f['%_of_high'] = (df_with_target_f[(df_with_target_f["target"]==1 ) 
        & (df_with_target_f["Year"]==year)]["is_in_article"].fillna(0).astype(int).sum())/df_with_target_f[df_with_target_f["Year"]==year]["is_in_article"].fillna(0).astype(int).sum()
                                                       
        
        df_all_f = df_all[df_all["Year"]==year]
        score_of_word_f['cnt_of_articles'] = (df_all_f[df_all_f["Year"]==year]["is_in_article"].fillna(0).astype(int).sum())
        
        df_f = df_with_target_f.reset_index()[df_with_target_f.reset_index()['abstract_cleaned'].fillna("").str.contains(word)]
        df_f = df_f[df_f['Year']==year]
        df_f_low = df_f[df_f['target']==0]
        df_f_high = df_f[df_f['target']==1]

        if df_f_low.groupby("Year").agg({'doi': '****'.join}).reset_index().empty ==True:
            score_of_word_f['articles_low'] = "doi is missing"    
        if df_f_high.groupby("Year").agg({'doi': '****'.join}).reset_index().empty ==True:
            score_of_word_f['articles_high'] = "doi is missing"
        
        if df_f_low.groupby("Year").agg({'doi': '****'.join}).reset_index().empty ==False:
            score_of_word_f['articles_low'] = df_f_low.groupby("Year").agg({'doi': '****'.join}).reset_index().iloc[0]['doi']
            
        if df_f_high.groupby("Year").agg({'doi': '****'.join}).reset_index().empty ==False:     
            score_of_word_f['articles_high'] =df_f_high.groupby("Year").agg({'doi': '****'.join}).reset_index().iloc[0]['doi'] 
        score_of_word_f = score_of_word_f[["word","score","Year","cnt_of_articles","cnt_of_art_with_targ",
                                           "%_of_high","articles_low","articles_high"]]
      
        dfs.append(score_of_word_f)
        
    specific_drug = pd.DataFrame(np.vstack(dfs),
       columns =["word","score","Year","cnt_of_articles","cnt_of_art_with_targ",
                                           "%_of_high","articles_low","articles_high"] )
    
    return specific_drug

def importance_lr_bow(tokens_bow,lreg_bow,n,odds_ratio = True):
    
    if n == 0:
        if odds_ratio == True:
            feature_importance = pd.DataFrame(tokens_bow, columns = ["word"])
            feature_importance["score"] = pow(math.e, lreg_bow.coef_[0])
        
        if odds_ratio == False:
            feature_importance = pd.DataFrame(tokens_bow, columns = ["word"])
            feature_importance["score"] =lreg_bow.coef_[0]
            
        feature_importance = feature_importance.sort_values(by = ["score"], ascending=False)
        
    if n != 0:
        feature_importance = pd.DataFrame(tokens_bow, columns = ["word"])
        if odds_ratio == True:
            feature_importance["score"] = pow(math.e, lreg_bow.coef_[0])
        if odds_ratio == False:    
            feature_importance["score"] = lreg_bow.coef_[0]
        feature_importance = feature_importance.sort_values(by = ["score"], ascending=False).iloc[[*range(n),*range(-n,0)]]
    
    return feature_importance



def score_of_word(model_w2v,lreg_w2v):
    
    words = model_w2v.wv.key_to_index.keys()
    we_dict = {word:model_w2v.wv[word] for word in words}
    df_of_words_vectors = pd.DataFrame(we_dict.items())
    df_of_words_vectors_ar = np.array(df_of_words_vectors[1].to_list())

    probs = lreg_w2v.predict_proba(df_of_words_vectors_ar)

    score_of_word = pd.DataFrame(list(probs[:,1]),columns =["score"] , index=df_of_words_vectors[0])
    score_of_word = score_of_word.sort_values("score",ascending=False).reset_index()
    score_of_word.columns = ["word","score"]
    
    return score_of_word

def word_score_info(words_drugs, score_of_word,df_with_target,df_all,add_dois = False):
    
    if isinstance(words_drugs, list) == True:
        words_drugs = words_drugs
    
    if isinstance(words_drugs, list) == False:
        import operator
        b = map(operator.itemgetter(0), words_drugs)
        words_drugs = list(b)
    
    final_results_of_select = score_of_word[score_of_word["word"].isin(words_drugs)]
    df_with_target = df_with_target.reset_index()
    df_all = df_all.reset_index()
    
    dfs = []
    list_of_words = list(final_results_of_select["word"].values)

    for word in tqdm(list_of_words):
        df_with_target["is_in_article"] = list(df_with_target.abstract_cleaned.str.contains(word).values)
        final_results_f = final_results_of_select[final_results_of_select["word"]==word]
        final_results_f['cnt_of_articles_with_targ'] = (df_with_target["is_in_article"].fillna(0).astype(int).sum())
        final_results_f['%_of_high_cit'] = (df_with_target[df_with_target["target"]==1]
        ["is_in_article"].fillna(0).astype(int).sum())/df_with_target["is_in_article"].fillna(0).astype(int).sum()
        
        df_all["is_in_article"] = list(df_all.abstract_cleaned.str.contains(word).values)
        final_results_f['cnt_of_articles'] = (df_all["is_in_article"].fillna(0).astype(int).sum())
        final_results_f["first_year"] = df_all[df_all['abstract_cleaned'].fillna("").str.contains(word)].agg({"Year":min})[0]
        df_f = df_with_target[df_with_target['abstract_cleaned'].fillna("").str.contains(word)]
        df_f['word'] = word
    
        df_f_low = df_f[df_f['target']==0]
        df_f_high = df_f[df_f['target']==1]
        
        if add_dois: 
            if df_f_low.groupby("word").agg({'doi': '****'.join}).empty ==True:
                final_results_f['articles_low'] = "doi is missing"
        
            if df_f_high.groupby("word").agg({'doi': '****'.join}).empty ==True:
                final_results_f['articles_high'] = "doi is missing"
        
            if df_f_low.groupby("word").agg({'doi': '****'.join}).empty ==False:
                final_results_f['articles_low'] = df_f_low.groupby("word").agg({'doi': '****'.join}).iloc[0]['doi']
        
            if df_f_high.groupby("word").agg({'doi': '****'.join}).empty ==False:     
                final_results_f['articles_high'] =df_f_high.groupby("word").agg({'doi': '****'.join}).iloc[0]['doi']    
            final_results_f = final_results_f[["word","score","cnt_of_articles_with_targ","%_of_high_cit",
                                           "cnt_of_articles","first_year","articles_low","articles_high"]]
        
        
        dfs.append(final_results_f)
    
    drugs_score_compared = pd.DataFrame(np.vstack(dfs),columns = ["word","score","cnt_of_articles_with_targ","%_of_high_cit","cnt_of_articles","first_year","articles_low","articles_high"] )
    return drugs_score_compared


def word_score_info_wo_target(words_drugs, score_of_word,df_all):
    
    if isinstance(words_drugs, list) == True:
        words_drugs = words_drugs
    
    if isinstance(words_drugs, list) == False:
        import operator
        b = map(operator.itemgetter(0), words_drugs)
        words_drugs = list(b)
    
    final_results_of_select = score_of_word[score_of_word["word"].isin(words_drugs)]
    df_all = df_all.reset_index()
    
    dfs = []
    list_of_words = list(final_results_of_select["word"].values)

    for word in tqdm(list_of_words):
        final_results_f = final_results_of_select[final_results_of_select["word"]==word]       
        df_all["is_in_article"] = list(df_all.abstract_cleaned.str.contains(word).values)
        final_results_f['cnt_of_articles'] = (df_all["is_in_article"].fillna(0).astype(int).sum())
        final_results_f["first_year"] = df_all[df_all['abstract_cleaned'].fillna("").str.contains(word)].agg({"Year":min})[0]       
        dfs.append(final_results_f)
    drugs_score_compared = pd.DataFrame(np.vstack(dfs),columns = ["word","score","cnt_of_articles","first_year"] )
    return drugs_score_compared

def word_score_info_just_perc(words_drugs, score_of_word,df_with_target,df_all):
    
    if isinstance(words_drugs, list) == True:
        words_drugs = words_drugs
    
    if isinstance(words_drugs, list) == False:
        import operator
        b = map(operator.itemgetter(0), words_drugs)
        words_drugs = list(b)
        print(words_drugs)
    
    final_results_of_select = score_of_word[score_of_word["word"].isin(words_drugs)]
    print(len(final_results_of_select))

    df_with_target = df_with_target.reset_index()
    df_all = df_all.reset_index()
    
    dfs = []
    for word in tqdm(list(final_results_of_select["word"].values)):
        escaped_word = re.escape(word)
        df_with_target["is_in_article"] = list(df_with_target.abstract_cleaned.str.contains(escaped_word).values)
        #df_with_target["is_in_article"] = list(df_with_target.abstract_cleaned.str.contains(word).values)
        final_results_f = final_results_of_select[final_results_of_select["word"]==word]
        final_results_f['cnt_of_articles_with_targ'] = (df_with_target["is_in_article"].fillna(0).astype(int).sum())
        final_results_f['%_of_high'] = (df_with_target[df_with_target["target"]==1]
        ["is_in_article"].fillna(0).astype(int).sum())/df_with_target["is_in_article"].fillna(0).astype(int).sum()
        dfs.append(final_results_f)
          
    drugs_score_compared = pd.DataFrame(np.vstack(dfs),columns = ["word","score","cnt_of_articles_with_targ","%_of_high_cit"] )
    return drugs_score_compared