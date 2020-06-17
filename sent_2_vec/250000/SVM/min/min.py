import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
x=np.load('/../data/legal_NLP/legalNLP_data/final_embd_files/sent2vec_embd_files/final_sv_mincount5_dim200_Vsize250000_Ngram2/train_embd_final_sv_mincount5_dim200_Vsize250000_Ngram2.npy',allow_pickle='true')
x_train_data=x[:,0]
y_train=x[:,1]
y_train=y_train.astype('int')
x_train=np.zeros([32305,200])
for i in range(0,32305):
    b=np.zeros([len(x_train_data[i]),200])
    for j in range(0,len(x_train_data[i])):
        b[j,:]=x_train_data[i][j]
    
    x_train[i,:]=np.amin(b,axis=0)
y=np.load('/../data/legal_NLP/legalNLP_data/final_embd_files/sent2vec_embd_files/final_sv_mincount5_dim200_Vsize250000_Ngram2/test_embd_final_sv_mincount5_dim200_Vsize250000_Ngram2.npy',allow_pickle='true')
x_test_data=y[:,0]
y_test=y[:,1]
y_test=y_test.astype('int')
x_test=np.zeros([1517,200])
for i in range(0,1517):
    b=np.zeros([len(x_test_data[i]),200])
    for j in range(0,len(x_test_data[i])):
        b[j,:]=x_test_data[i][j]
    
    x_test[i,:]=np.amin(b,axis=0)
z=np.load('/../data/legal_NLP/legalNLP_data/final_embd_files/sent2vec_embd_files/final_sv_mincount5_dim200_Vsize250000_Ngram2/validation_embd_final_sv_mincount5_dim200_Vsize250000_Ngram2.npy',allow_pickle='true')
x_dev_data=z[:,0]
y_dev=z[:,1]
y_dev=y_dev.astype('int')
x_dev=np.zeros([994,200])
for i in range(0,994):
    b=np.zeros([len(x_dev_data[i]),200])
    for j in range(0,len(x_dev_data[i])):
        b[j,:]=x_dev_data[i][j]
    
    x_dev[i,:]=np.amin(b,axis=0)
y_train1=np.zeros([32305,1])
for i in range(0,32305):
    y_train1[i][0]=y_train[i]
y_test1=np.zeros([1517,1])
for i in range(0,1517):
    y_test1[i][0]=y_test[i]
y_dev1=np.zeros([994,1])
for i in range(0,994):
    y_dev1[i][0]=y_dev[i]
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
def SVM_scores(train_avg, dev_avg, test_avg, train_labels, dev_labels, test_labels):
    f = open("SVM_min_results.txt", "w+")
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
SVM_scores(x_train,x_dev,x_test,y_train1,y_dev1,y_test1)