
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import numpy as np 
import pandas as pd 
from sklearn import *
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import PassiveAggressiveClassifier

#from sklearn.tree import export_graphviz
#from IPython import display


main = tkinter.Tk()
main.title("Credit Card Fraud Detection") #designing main screen
main.geometry("1300x1200")

global filename
global cls
global X, Y, X_train, X_test, y_train, y_test
global random_acc # all global variables names define in above lines
global clean
global attack
global total


def traintest(train):     #method to generate test and train data from dataset for 6:4 ratio
    X = train.values[:, 0:29] 
    Y = train.values[:, 30]
    print(X)
    print(Y)
    X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size = 0.4, random_state = 0)
    return X, Y, X_train, X_test, y_train, y_test

def generateModel(): #method to read dataset values which contains all five features data for 6:4 ratio
    global X, Y, X_train, X_test, y_train, y_test
    train = pd.read_csv(filename)
    X, Y, X_train, X_test, y_train, y_test = traintest(train)
    text.insert(END,"Train & Test Model Generated\n\n")
    text.insert(END,"Total Dataset Size : "+str(len(train))+"\n")
    text.insert(END,"Split Training Size : "+str(len(X_train))+"\n")
    text.insert(END,"Split Test Size : "+str(len(X_test))+"\n")



def traintest1(train):     #method to generate test and train data from dataset for 7:3 ratio
    X = train.values[:, 0:29]
    Y = train.values[:, 30]
    print(X)
    print(Y)
    X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size = 0.3, random_state = 0)
    return X, Y, X_train, X_test, y_train, y_test

def generateModel1(): #method to read dataset values which contains all five features data for 7:3 ratio 
    global X, Y, X_train, X_test, y_train, y_test
    train = pd.read_csv(filename)
    X, Y, X_train, X_test, y_train, y_test = traintest1(train)
    text.insert(END,"Train & Test Model Generated\n\n")
    text.insert(END,"Total Dataset Size : "+str(len(train))+"\n")
    text.insert(END,"Split Training Size : "+str(len(X_train))+"\n")
    text.insert(END,"Split Test Size : "+str(len(X_test))+"\n")


def traintest2(train):     #method to generate test and train data from dataset for 8:2 ratio
    X = train.values[:, 0:29]
    Y = train.values[:, 30]
    print(X)
    print(Y)
    X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size = 0.2, random_state = 0)
    return X, Y, X_train, X_test, y_train, y_test

def generateModel2(): #method to read dataset values which contains all five features data for 8:2 ratio
    global X, Y, X_train, X_test, y_train, y_test
    train = pd.read_csv(filename)
    X, Y, X_train, X_test, y_train, y_test = traintest2(train)
    text.insert(END,"Train & Test Model Generated\n\n")
    text.insert(END,"Total Dataset Size : "+str(len(train))+"\n")
    text.insert(END,"Split Training Size : "+str(len(X_train))+"\n")
    text.insert(END,"Split Test Size : "+str(len(X_test))+"\n")


def upload(): #function to upload tweeter profile
    global filename
    filename = filedialog.askopenfilename(initialdir="dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");



def prediction(X_test, cls):  #prediction done here
    y_pred = cls.predict(X_test) 
    for i in range(50):
      print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred 
	
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred, details): 
    accuracy = accuracy_score(y_test,y_pred)*100
    text.insert(END,details+"\n\n")
    text.insert(END,"Accuracy : "+str(accuracy)+"\n\n")
    return accuracy


def runRandomForest():
    headers = ["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount","Class"]
    global random_acc
    global cls
    global X, Y, X_train, X_test, y_train, y_test
    cls = RandomForestClassifier(n_estimators=50,max_depth=2,random_state=0,class_weight='balanced')
    cls.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n\n")
    prediction_data = prediction(X_test, cls)
    random_acc = cal_accuracy(y_test, prediction_data,'Random Forest Accuracy')

def runLogisticRegression():
    headers = ["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount","Class"]
    global logistic_acc
    global cls
    global X, Y, X_train, X_test, y_train, y_test
    cls = LogisticRegression(solver='liblinear',max_iter=100,random_state=0,class_weight='balanced')
    cls.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n\n")
    prediction_data = prediction(X_test, cls)
    logistic_acc = cal_accuracy(y_test, prediction_data,'Logistic Regression Accuracy')

def runDecisionTree():
    headers = ["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount","Class"]
    global decision_acc
    global cls
    global X, Y, X_train, X_test, y_train, y_test
    cls = DecisionTreeClassifier(max_depth=2,random_state=0,class_weight='balanced')
    cls.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n\n")
    prediction_data = prediction(X_test, cls)
    decision_acc = cal_accuracy(y_test, prediction_data,'Decision tree Accuracy')

def runGaussianNB():
    headers = ["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount","Class"]
    global naive_acc
    global cls
    global X, Y, X_train, X_test, y_train, y_test
    cls = GaussianNB()
    cls.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n\n")
    prediction_data = prediction(X_test, cls)
    naive_acc = cal_accuracy(y_test, prediction_data,'Naive Bayes Accuracy')

def runPassiveAggressive():
    headers = ["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount","Class"]
    global passive_acc
    global cls
    global X, Y, X_train, X_test, y_train, y_test
    cls = PassiveAggressiveClassifier(C = 0.5, random_state = 5)
    cls.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n\n")
    prediction_data = prediction(X_test, cls)
    passive_acc = cal_accuracy(y_test, prediction_data,'Passive Aggreessive Accuracy')

def compGraph():
    data = {'Random Forest':random_acc , 'Logistic Regression':logistic_acc, 'Decision Tree':decision_acc, 'Naive_Bayes':naive_acc, 'Passive Aggressive':passive_acc}
    algorithms = list(data.keys())
    accuracy = list(data.values())
    acc_fig = plt.figure(figsize = (10, 5))
    plt.bar(algorithms, accuracy, color = "blue", width = 0.2)
    plt.xlabel("\nMachine Learning Algorithms")
    plt.ylabel("Accuracies")
    plt.title("Comparison graph of different algorithms")
    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='Credit Card Fraud Detection Using Machine learning Algorithms')
title.config(bg='greenyellow', fg='dodger blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Credit Card Dataset", command=upload)
uploadButton.place(x=0,y=550)
uploadButton.config(font=font1)  

modelButton = Button(main, text="Split 6:4", command=generateModel)
modelButton.place(x=250,y=550)
modelButton.config(font=font1) 

modelButton = Button(main, text="Split 7:3", command=generateModel1)
modelButton.place(x=400,y=550)
modelButton.config(font=font1)

modelButton = Button(main, text="Split 8:2", command=generateModel2)
modelButton.place(x=550,y=550)
modelButton.config(font=font1)

runrandomButton = Button(main, text="Run Random Forest Algorithm", command=runRandomForest)
runrandomButton.place(x=700,y=550)
runrandomButton.config(font=font1)

runLogisticButton = Button(main, text="Run Logistic Regression Algorithm", command=runLogisticRegression)
runLogisticButton.place(x=1000,y=550)
runLogisticButton.config(font=font1)

runDecButton = Button(main, text="Run Decision Tree Algorithm", command=runDecisionTree)
runDecButton.place(x=0,y=600)
runDecButton.config(font=font1)

runnavButton = Button(main, text="Run Naive Bayes Algorithm", command=runGaussianNB)
runnavButton.place(x=250,y=600)
runnavButton.config(font=font1)

runPasButton = Button(main, text="Run Passive Aggressive Algorithm", command=runPassiveAggressive)
runPasButton.place(x=500,y=600)
runPasButton.config(font=font1)

compgraphButton = Button(main, text="Accuracy Comparison Graph", command=compGraph)
compgraphButton.place(x=800,y=600)
compgraphButton.config(font=font1)

exitButton = Button(main, text="Exit", command=exit)
exitButton.place(x=1050,y=600)
exitButton.config(font=font1) 

main.config(bg='LightSkyBlue')
main.mainloop()
