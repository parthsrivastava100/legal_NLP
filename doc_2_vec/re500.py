import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
x_train=np.load('train_embd_500.npy')
y_train=np.load('train_val.npy')
x_test=np.load('test_embd_500.npy')
y_test=np.load('test_val.npy')
x_dev=np.load('validation_embd_500.npy')
y_dev=np.load('dev_val.npy')


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




RF_scores(x_train,x_dev,x_test,y_train,y_dev,y_test)