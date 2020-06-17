#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
from gensim.models.doc2vec import Doc2Vec
import progressbar
import nltk
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

nltk.download('punkt')
data = json.load(open("/data/vijitvm/LNLP_sir/newf/splitter.json","r"))


# In[2]:


train_file_list = []
test_file_list = []
dev_file_list = []
for name in data['data'].keys():
    if(data['data'][name]['split']=="train"):
        train_file_list.append(name)
    if(data['data'][name]['split']=="test"):
        test_file_list.append(name)
    if(data['data'][name]['split']=="dev"):
        dev_file_list.append(name)


# In[3]:


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
model= Doc2Vec.load("/data/vijitvm/LNLP_sir/doc2vec1500train/d2v.model")


# In[ ]:


path = "/data/vijitvm/LNLP_sir/Processed_and_removed_decisions/"

def _return_embs(file_list, model):
    set_embs = []
    for i in progressbar.progressbar(range(len(file_list))):
        _ = file_list[i]
        text = open(path + _, "r").read()
        text = text.lower()
        text = text.replace("(","")
        text = text.replace(")","")
        text = text.replace("[","")
        text = text.replace("]","")
        text = text.replace(",","")
        text = text.replace("-","")
        toks = word_tokenize(text)
        v1 = model.infer_vector(toks)

        set_embs.append(v1)
    
    return set_embs
    
train_avg = _return_embs(train_file_list, model)
dev_avg = _return_embs(dev_file_list, model)
test_avg = _return_embs(test_file_list, model)


# In[ ]:


def numeric_label(verdict):
    if(verdict == "Accepted"):
        return 1
    else:
        return 0


# In[15]:


train_labels = [numeric_label(data['data'][f_n]['label']) for f_n in train_file_list]
dev_labels = [numeric_label(data['data'][f_n]['label']) for f_n in dev_file_list]
test_labels = [numeric_label(data['data'][f_n]['label']) for f_n in test_file_list]


# In[16]:


def metrics_calculator(preds, test_labels):
    cm = confusion_matrix(test_labels, preds)
    TP = []
    FP = []
    FN = []
    for i in range(0,2):
        summ = 0
        for j in range(0,2):
            if(i!=j):
                summ=summ+cm[i][j]

        FN.append(summ)
    for i in range(0,2):
        summ = 0
        for j in range(0,2):
            if(i!=j):
                summ=summ+cm[j][i]

        FP.append(summ)
    for i in range(0,2):
        TP.append(cm[i][i])
    precision = []
    recall = []
    for i in range(0,2):
        precision.append(TP[i]/(TP[i] + FP[i]))
        recall.append(TP[i]/(TP[i] + FN[i]))

    macro_precision = sum(precision)/2
    macro_recall = sum(recall)/2
    micro_precision = sum(TP)/(sum(TP) + sum(FP))
    micro_recall = sum(TP)/(sum(TP) + sum(FN))
    micro_f1 = (2*micro_precision*micro_recall)/(micro_precision + micro_recall)
    macro_f1 = (2*macro_precision*macro_recall)/(macro_precision + macro_recall)
    return macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1


# In[17]:


def LR_scores(train_avg, dev_avg, test_avg, train_labels, dev_labels, test_labels):
    f = open("LR_results.txt", "w+")
    f.write("Varying the max_iters from 50 to 500\n\n")
    for it in range(50,500,50):
        LR = LogisticRegression(C=1, max_iter =it)
        LR.fit(train_avg, train_labels)
        d_preds = LR.predict(dev_avg)
        Heading = "For max_iters: " + str(it) + "\n"
        d_res = "Accuracy -> " + str(accuracy_score(d_preds, dev_labels)*100) + "\n"
        macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1 = metrics_calculator(d_preds, dev_labels)
        d_metrics = "Micros : " + str(micro_precision) + " " + str(micro_recall) + " " + str(micro_f1) + " Macros: " + str(macro_precision) + " " + str(macro_recall) + " " + str(macro_f1) + "\n"
        f.write(Heading + "Dev set:\n"+ d_res + d_metrics)
        
        t_preds = LR.predict(test_avg)
        t_res = "Accuracy -> " + str(accuracy_score(t_preds, test_labels)*100) + "\n"
        macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1 = metrics_calculator(t_preds, test_labels)
        t_metrics = "Micros : " + str(micro_precision) + " " + str(micro_recall) + " " + str(micro_f1) + " Macros: " + str(macro_precision) + " " + str(macro_recall) + " " + str(macro_f1) + "\n"
        f.write("Test set:\n"+ t_res + t_metrics + "\n\n")
        
    f.close()


# In[25]:


def RF_scores(train_avg, dev_avg, test_avg, train_labels, dev_labels, test_labels):
    f = open("RF_results.txt", "w+")
    f.write("Varying the n_estimators from 50 to 500\n\n")
    for n_est in range(50,500,50):
        clf=RandomForestClassifier(n_estimators=n_est)
        clf.fit(train_avg,train_labels)
        d_preds = clf.predict(dev_avg)
        Heading = "For n_estimators: " + str(n_est) + "\n"
        d_res = "Accuracy -> " + str(accuracy_score(d_preds, dev_labels)*100) + "\n"
        macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1 = metrics_calculator(d_preds, dev_labels)
        d_metrics = "Micros : " + str(micro_precision) + " " + str(micro_recall) + " " + str(micro_f1) + " Macros: " + str(macro_precision) + " " + str(macro_recall) + " " + str(macro_f1) + "\n"
        f.write(Heading + "Dev set:\n"+ d_res + d_metrics)
        
        t_preds = clf.predict(test_avg)
        t_res = "Accuracy -> " + str(accuracy_score(t_preds, test_labels)*100) + "\n"
        macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1 = metrics_calculator(t_preds, test_labels)
        t_metrics = "Micros : " + str(micro_precision) + " " + str(micro_recall) + " " + str(micro_f1) + " Macros: " + str(macro_precision) + " " + str(macro_recall) + " " + str(macro_f1) + "\n"
        f.write("Test set:\n"+ t_res + t_metrics + "\n\n")
        
    f.close()


# In[26]:


def SVM_scores(train_avg, dev_avg, test_avg, train_labels, dev_labels, test_labels):
    f = open("SVM_results.txt", "w+")
    f.write("Varying the kernels: \n\n")
    kers = ["linear", "poly", "rbf"]
    for k in kers:
        print("Running for {0}".format(k))
        SVM = svm.SVC(C=1, kernel=k)
        SVM.fit(train_avg, train_labels)
        d_preds = SVM.predict(dev_avg)
        Heading = "For kernel: " + k + "\n"
        d_res = "Accuracy -> " + str(accuracy_score(d_preds, dev_labels)*100) + "\n"
        macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1 = metrics_calculator(d_preds, dev_labels)
        d_metrics = "Micros : " + str(micro_precision) + " " + str(micro_recall) + " " + str(micro_f1) + " Macros: " + str(macro_precision) + " " + str(macro_recall) + " " + str(macro_f1) + "\n"
        f.write(Heading + "Dev set:\n"+ d_res + d_metrics)
        print(Heading + "Dev set:\n"+ d_res + d_metrics)

        t_preds = SVM.predict(test_avg)
        t_res = "Accuracy -> " + str(accuracy_score(t_preds, test_labels)*100) + "\n"
        macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1 = metrics_calculator(t_preds, test_labels)
        t_metrics = "Micros : " + str(micro_precision) + " " + str(micro_recall) + " " + str(micro_f1) + " Macros: " + str(macro_precision) + " " + str(macro_recall) + " " + str(macro_f1) + "\n"
        f.write("Test set:\n"+ t_res + t_metrics + "\n\n")
        print("Test set:\n"+ t_res + t_metrics + "\n\n")
    f.close()


# In[ ]:


def Ada_scores(train_avg, dev_avg, test_avg, train_labels, dev_labels, test_labels):
    f = open("Ada_results.txt", "w+")
    f.write("Varying the n_estimators from 50 to 500\n\n")
    for n_est in range(50,500,50):
        model = AdaBoostClassifier(n_estimators=n_est)
        model.fit(train_avg,train_labels)
        d_preds = model.predict(dev_avg)
        Heading = "For n_estimators: " + str(n_est) + "\n"
        d_res = "Accuracy -> " + str(accuracy_score(d_preds, dev_labels)*100) + "\n"
        macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1 = metrics_calculator(d_preds, dev_labels)
        d_metrics = "Micros : " + str(micro_precision) + " " + str(micro_recall) + " " + str(micro_f1) + " Macros: " + str(macro_precision) + " " + str(macro_recall) + " " + str(macro_f1) + "\n"
        f.write(Heading + "Dev set:\n"+ d_res + d_metrics)
        
        t_preds = model.predict(test_avg)
        t_res = "Accuracy -> " + str(accuracy_score(t_preds, test_labels)*100) + "\n"
        macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1 = metrics_calculator(t_preds, test_labels)
        t_metrics = "Micros : " + str(micro_precision) + " " + str(micro_recall) + " " + str(micro_f1) + " Macros: " + str(macro_precision) + " " + str(macro_recall) + " " + str(macro_f1) + "\n"
        f.write("Test set:\n"+ t_res + t_metrics + "\n\n")
        
    f.close()


# In[ ]:


def GBoost_scores(train_avg, dev_avg, test_avg, train_labels, dev_labels, test_labels):
    f = open("GBoost_results.txt", "w+")
    f.write("Varying the n_estimators from 50 to 500\n\n")
    for n_est in range(50,500,50):
        model = GradientBoostingClassifier(n_estimators=n_est)
        model.fit(train_avg,train_labels)
        d_preds = model.predict(dev_avg)
        Heading = "For n_estimators: " + str(n_est) + "\n"
        d_res = "Accuracy -> " + str(accuracy_score(d_preds, dev_labels)*100) + "\n"
        macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1 = metrics_calculator(d_preds, dev_labels)
        d_metrics = "Micros : " + str(micro_precision) + " " + str(micro_recall) + " " + str(micro_f1) + " Macros: " + str(macro_precision) + " " + str(macro_recall) + " " + str(macro_f1) + "\n"
        f.write(Heading + "Dev set:\n"+ d_res + d_metrics)
        
        t_preds = model.predict(test_avg)
        t_res = "Accuracy -> " + str(accuracy_score(t_preds, test_labels)*100) + "\n"
        macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1 = metrics_calculator(t_preds, test_labels)
        t_metrics = "Micros : " + str(micro_precision) + " " + str(micro_recall) + " " + str(micro_f1) + " Macros: " + str(macro_precision) + " " + str(macro_recall) + " " + str(macro_f1) + "\n"
        f.write("Test set:\n"+ t_res + t_metrics + "\n\n")
        
    f.close()


# In[ ]:


LR_scores(train_avg, dev_avg, test_avg, train_labels, dev_labels, test_labels)
RF_scores(train_avg, dev_avg, test_avg, train_labels, dev_labels, test_labels)
SVM_scores(train_avg, dev_avg, test_avg, train_labels, dev_labels, test_labels)
Ada_scores(train_avg, dev_avg, test_avg, train_labels, dev_labels, test_labels)
GBoost_scores(train_avg, dev_avg, test_avg, train_labels, dev_labels, test_labels)

