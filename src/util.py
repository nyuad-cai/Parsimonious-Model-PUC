
from sklearn.metrics import accuracy_score, roc_auc_score,average_precision_score, roc_curve, f1_score,precision_recall_curve
from random import choice
import pandas as pd
import numpy as np


def evaluate_new(df):
    auroc = roc_auc_score(df['y_truth'], df['y_pred'])
    auprc = average_precision_score(df['y_truth'], df['y_pred'])
    return auprc, auroc

def bootstraping_eval(df, num_iter):
    """This function samples from the testing dataset to generate a list of performance metrics using bootstraping method"""
    auroc_list = []
    auprc_list = []
    for _ in range(num_iter):
        sample = df.sample(frac=1, replace=True)
        auprc, auroc = evaluate_new(sample)
        auroc_list.append(auroc)
        auprc_list.append(auprc)
    return auprc_list, auroc_list

def computing_confidence_intervals(list_,true_value):
    #https://github.com/nyuad-cai/COVID19Complications
    """This function calcualts the 95% Confidence Intervals"""
    delta = (true_value - list_)
    list(np.sort(delta))
    delta_lower = np.percentile(delta, 97.5)
    delta_upper = np.percentile(delta, 2.5)

    upper = true_value - delta_upper
    lower = true_value - delta_lower
    return (upper,lower)

def get_model_performance(df):
    test_auprc, test_auroc = evaluate_new(df)
    auprc_list, auroc_list = bootstraping_eval(df, num_iter=1000)
    upper_auprc, lower_auprc = computing_confidence_intervals(auprc_list, test_auprc)
    upper_auroc, lower_auroc = computing_confidence_intervals(auroc_list, test_auroc)
    print("--------------")
    text_a=str(f"AUROC {round(test_auroc, 3)} ( {round(lower_auroc, 3)} , {round(upper_auroc, 3)} ) CI 95%")
    text_b=str(f"AUPRC {round(test_auprc, 3)} ( {round(lower_auprc, 3)} , {round(upper_auprc, 3)}) CI 95% ")
    print(text_a)
    print(text_b)

    return (test_auprc, upper_auprc, lower_auprc), (test_auroc, upper_auroc, lower_auroc), (text_a,text_b)

def get_antibiotic_stats(full_df,test_df,new_label,test_index):
    """This function calcualts the impact on antibiotics section"""

    #samples from the test set
    full_df["test"]= (full_df.EncounterKey).isin(test_index)*1
    test= meds[meds["test"] == 1]
    print("test", len(test))

    test["new_label"]=test[new_label].astype("int")
      
    prescribed = test[test["new_prescribed"] ==1]
    not_prescribed = test [test["new_prescribed"] ==0]
    print("prescribed", len(prescribed)/len(test)*100)
    print("not_prescribed", len(not_prescribed)/len(test)*100)
    percent = (len(prescribed)/len(test))*100

    # prescription metrics 
    tn, fp, fn, tp = confusion_matrix(test[new_label], test["new_prescribed"]).ravel()
    p_base=tp/(tp+fp)
    s_base = tn/(tn+fp)
    sensitivty = tp/(tp+fn)
    
    print("fixed sensitivty:", tp/(tp+fn))
    print("specifcity:", s_base)
    print("PPV:",p_base)
    print("NPV:", tn/(fn+tn))
    
    # fix senstivity
    test["y_pred"] = test_df["y_pred"]
    print("nan", test_df["y_pred"].isna().sum(), test[new_label].isna().sum())
    
    threshold = get_threshold(test[new_label], test["y_pred"],sensitivty)
    test["adjusted_senstivity"]=np.where(test['y_pred']>=threshold, 1, 0)
    
    positive_pred = test[test["adjusted_senstivity"] ==1]
    negative_pred = test[test["adjusted_senstivity"] ==0]
    
    print("positive prediction ", len(positive_pred)/len(test)*100)
    print("negative prediction ", len(negative_pred)/len(test)*100)
    
    
    tn, fp, fn, tp = confusion_matrix(test[new_label], test["adjusted_senstivity"]).ravel()
    PPVs_m=tp/(tp+fp)
    sepc_m = tn/(tn+fp)
    print("specifcity:", sepc_m)
    print("PPV:", PPVs_m)
    print("NPV:", tn/(fn+tn))    
    
def get_threshold(y_test, y_pred,target_recall):
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    idx = (np.abs(recall - target_recall)).argmin()
    print('Setting threshold to %f the model achieves %f sensitivity' %(thresholds[idx], recall[idx]))
