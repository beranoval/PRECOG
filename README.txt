All jupyter notebooks are started sequentially according to the numbering in the title.

To run code prepare folders in folowing structure with input files: 
    
/sources
├── citationcounts_oci_revised_year.csv
├── lid.176.bin -> ke stažení na https://fasttext.cc/docs/en/language-identification.html
├── /cord19
    ├──/23_04_2022
       ├── metadata.csv -> ke stažení na https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge
/outputs
├──/classifier
   ├──/train_2017
   ├──/train_2018
   ├──/train_2019
   ├──/train_2020
   ├──/train_2021
├──/w2v
├──/corr

After running the jupyter notebooks, other created outputs will be saved automatically to prepared folders.

Description of notebooks and their outputs:

1.Preprocessing 
= Notebook for filtering articles and cleaning abstracts of CORD-19 corpus. 
Output: outputs/df_sw_tok_low_punc_lemm_v7.csv

2.Train_embeddings 
= Train W2V models for separate time intervals 
Outputs: 
    outputs/w2v/w2v_published_between_2016_2017.model
    outputs/w2v/w2v_published_between_2017_2018.model
    outputs/w2v/w2v_published_between_2018_2019.model
    outputs/w2v/w2v_published_between_2019_2020.model
    outputs/w2v/w2v_published_between_2020_2021.model

3.Articles_of_year_QuantVal
= Outputs for articles results in Table 3 for Experiment 1: Predictions of impactful articles.
Outputs:
---Pickle files of trained models on BOW and W2V for classic and regularized LR: 
    =/train_2017/lreg_bow_2017.sav
    =/train_2018/lreg_bow_2018.sav
    =/train_2019/lreg_bow_2019.sav
    =/train_2020/lreg_bow_2020.sav
    
    =/train_2017/lreg_reg_bow_2017.sav
    =/train_2018/lreg_reg_bow_2018.sav
    =/train_2019/lreg_reg_bow_2019.sav
    =/train_2020/lreg_reg_bow_2020.sav
    
    = /train_2017/lreg_w2v_avg_2017.sav
    =/train_2018/lreg_w2v_avg_2018.sav
    =/train_2019/lreg_w2v_avg_2019.sav
    =/train_2020/lreg_w2v_avg_2020.sav
    
    = /train_2017/lreg_reg_w2v_avg_2017.sav
    =/train_2018/lreg_reg_w2v_avg_2018.sav
    =/train_2019/lreg_reg_w2v_avg_2019.sav
    =/train_2020/lreg_reg_w2v_avg_2020.sav
    
    =train_2017/tokens_bow_2017.data
    =train_2018/tokens_bow_2018.data
    =train_2019/tokens_bow_2019.data
    =train_2020/tokens_bow_2020.data
    
---Validation_of_predictions of W2V and BOW:
    =/train_2017/res_all_bow_2017.csv
    =/train_2018/res_all_bow_2018.csv
    =/train_2019/res_all_bow_2019.csv
    =/train_2020/res_all_bow_2020.csv
    
    =/train_2017/res_all_w2v_avg_2017.csv
    =/train_2018/res_all_w2v_avg_2018.csv
    =/train_2019/res_all_w2v_avg_2019.csv
    =/train_2020/res_all_w2v_avg_2020.csv
    

4.Articles_of_year_QualVal
= Outputs for results in Experiment 5: Qualitative evaluation of top-predicted articles
Outputs:
    classifiers/train_2021/lreg_bow_2021.model
    classifiers/train_2021/lreg_w2v_avg_2021.model
    classifiers/train_2021/train_final_articles_score_table_bow.csv
    classifiers/train_2021/train_final_articles_score_table_w2v.csv
    classifiers/train_2021/tokens_bow_2021.data
    
    
5.Entities_of_year_QuantVal
= Outputs for results in Experiment 2: Correlation of feature importance with actual impact
Outputs:
-outputs/corr/lr_bow.pdf
-outputs/corr/lr_w2v.pdf
-outputs/corr/lr_bow_2021.pdf
-outputs/corr/lr_w2v_2021.pdf

6.Entities_of_year_QualVal
= Outputs for results in Experiment 4: Qualitative evaluation of top-predicted biological entities.





