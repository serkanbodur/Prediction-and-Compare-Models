# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 21:58:35 2019

@author: Mysia
"""
#%%
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.linear_model import LinearRegression
from keras.optimizers import SGD
warnings.filterwarnings('ignore')

df = pd.read_csv('online_shoppers_intention.csv')
head=df.info()

#Before going any further, there are a couple of null values in the data that we need to clean up.
df.isnull().sum()
df.dropna(inplace=True)


df = df.sample(frac=1).reset_index(drop=True) # Shuffle
print("\n Import Data Successfull")

# **Exploratory Data Analysis**
print("\n Exploratory Data Analysis")

obj_df = df.select_dtypes(include=['object']).copy()
a=obj_df.head()


#As a result no non values

month=obj_df["Month"].value_counts()

visitorType=obj_df["VisitorType"].value_counts()

#Then i create a dictionary to transform categorical value to int value
month_dict={"Month": {"May": 5,"Nov": 11,"Mar": 3, "Dec": 12,"Oct": 10,"Sep": 9,"Aug": 8,"Jul": 7,"June": 6,"Feb": 2},
            "VisitorType": {"Returning_Visitor": 1,"New_Visitor": 2,"Other": 3}}

#Convert the columns to numerical datas
obj_df.replace(month_dict, inplace=True)
obj_df.head()

#And create a new dataframe just numerical and boolean values
df2 = df.drop(['VisitorType','Month'], axis=1)
my_df=pd.concat([df2,obj_df],axis=1)


plt.figure(figsize=(10,10))
a=my_df.Revenue.groupby(my_df.VisitorType).sum()
plt.pie(a, labels=a.index);
plt.show()
#%%
#corrmat = df.corr()  
#f, ax = plt.subplots(figsize =(9, 8)) 
#sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1) 
#
##Mounth of buy
#plt.figure(figsize=(10,10));
#sums = df.Revenue.groupby(df.Month).sum()
#plt.pie(sums, labels=sums.index);
#plt.show()
#
##Visitor Type of buy
#plt.figure(figsize=(6,6));
#sums1 = df.Revenue.groupby(df.VisitorType).sum()
#plt.pie(sums1, labels=sums1.index);
#plt.show()
#
#sns.countplot(df['Revenue'])
#plt.show()
#
#
#sns.set(rc={'figure.figsize':(11.7,8.27)})
#sns.scatterplot(x='ProductRelated_Duration',y='BounceRates', data=df, hue='Revenue',palette='prism')
#plt.show()
#
#sns.scatterplot(x='PageValues',y='BounceRates', data=df, hue='Revenue', palette='prism')
#plt.show()
#
#sns.set(rc={'figure.figsize':(11.7,8.27)})
#sns.scatterplot(x='Informational_Duration',y='BounceRates', data=df, hue='Revenue',palette='prism')
#plt.show()
#
#sns.set(rc={'figure.figsize':(11.7,8.27)})
#sns.scatterplot(x='ProductRelated',y='ExitRates', data=df, hue='Revenue',palette='prism')
#plt.show()

# **Data Preprocessing**

print('\n Data Preprocessing')






#MLP-5 datas
#df2 = df.drop(['TrafficType','VisitorType','SpecialDay','OperatingSystems','Browser','Informational','Informational_Duration','ProductRelated','ProductRelated_Duration','BounceRates','Administrative_Duration','Revenue','Month'], axis=1) 

#LR-12
#df2 = df.drop(['TrafficType','Browser','SpecialDay','ProductRelated_Duration','BounceRates','Revenue'], axis=1) 

#LR-11

y = my_df['Revenue']
print('\n Done')

#Lr-17
my_df = my_df.drop(['Revenue'], axis=1)
X = pd.get_dummies(my_df,drop_first=True)

##SVM-10
#my_df=my_df.drop(['Administrative_Duration','Browser','TrafficType','ProductRelated_Duration','ProductRelated','Informational','Informational_Duration'],axis=1)
#X = pd.get_dummies(my_df,drop_first=True)

##RF-5
#my_df=my_df.drop(['ExitRates','OperatingSystems','Browser','Administrative_Duration','SpecialDay','ProductRelated','Administrative','Informational','Informational_Duration','Region','TrafficType','VisitorType'],axis=1)
#X = pd.get_dummies(my_df,drop_first=True)

##KNN-5
#my_df=my_df.drop(['ExitRates','BounceRates','Browser','Administrative_Duration','SpecialDay','ProductRelated','ProductRelated_Duration','Informational','Informational_Duration','Region','TrafficType','Weekend'],axis=1)
#X = pd.get_dummies(my_df,drop_first=True)

##Lr-16
#my_df=my_df.drop(['Month'],axis=1)
#X = pd.get_dummies(my_df,drop_first=True)
#
##Lr-15
#my_df=my_df.drop(['Month','ProductRelated_Duration'],axis=1)
#X = pd.get_dummies(my_df,drop_first=True)
#
##Lr-14
#my_df=my_df.drop(['Month','ProductRelated_Duration','ProductRelated'],axis=1)
#X = pd.get_dummies(my_df,drop_first=True)
##
##Lr-13
#my_df=my_df.drop(['Month','ProductRelated_Duration','ProductRelated','Administrative'],axis=1)
#X = pd.get_dummies(my_df,drop_first=True)
#
##Lr-12
#my_df=my_df.drop(['Month','ProductRelated_Duration','ProductRelated','Administrative','Informational'],axis=1)
#X = pd.get_dummies(my_df,drop_first=True)
#
##Lr-11
#my_df=my_df.drop(['Month','ProductRelated_Duration','ProductRelated','Administrative','Informational','Informational_Duration'],axis=1)
#X = pd.get_dummies(my_df,drop_first=True)
#
##Lr-10
#my_df=my_df.drop(['Region','Month','ProductRelated_Duration','ProductRelated','Administrative','Informational','Informational_Duration'],axis=1)
#X = pd.get_dummies(my_df,drop_first=True)
#
#Lr-9
#my_df=my_df.drop(['Administrative_Duration','SpecialDay','Month','ProductRelated_Duration','ProductRelated','Administrative','Informational','Informational_Duration'],axis=1)
#X = pd.get_dummies(my_df,drop_first=True)
#
##Lr-8-ES8
#my_df=my_df.drop(['Browser','Region','SpecialDay','Month','ProductRelated_Duration','ProductRelated','Administrative','Informational','Informational_Duration'],axis=1)
#X = pd.get_dummies(my_df,drop_first=True)
#
#Lr-8*ES
my_df=my_df.drop(['Browser','Region','SpecialDay','Month','ProductRelated_Duration','ProductRelated','Administrative','Informational','Informational_Duration'],axis=1)
X = pd.get_dummies(my_df,drop_first=True)

##Lr-7
#my_df=my_df.drop(['BounceRates','Administrative_Duration','OperatingSystems','SpecialDay','Month','ProductRelated_Duration','ProductRelated','Administrative','Informational','Informational_Duration'],axis=1)
#X = pd.get_dummies(my_df,drop_first=True)
##
##Lr-6
#my_df=my_df.drop(['Administrative_Duration','OperatingSystems','SpecialDay','Month','ProductRelated_Duration','ProductRelated','Administrative','Informational','Informational_Duration','Browser','TrafficType'],axis=1)
#X = pd.get_dummies(my_df,drop_first=True)
##
##Lr-5
#my_df=my_df.drop(['BounceRates','ExitRates','Administrative_Duration','SpecialDay','Month','ProductRelated_Duration','ProductRelated','Administrative','Informational','Informational_Duration','Region','TrafficType'],axis=1)
#X = pd.get_dummies(my_df,drop_first=True)

##Lr-4
#my_df=my_df.drop(['Browser','OperatingSystems','BounceRates','ExitRates','Administrative_Duration','SpecialDay','Month','ProductRelated_Duration','ProductRelated','Administrative','Informational','Informational_Duration','TrafficType'],axis=1)
#X = pd.get_dummies(my_df,drop_first=True)
#
##Lr-3
#my_df=my_df.drop(['Browser','Region','Weekend','BounceRates','ExitRates','Administrative_Duration','SpecialDay','Month','ProductRelated_Duration','ProductRelated','Administrative','Informational','Informational_Duration','TrafficType'],axis=1)
#X = pd.get_dummies(my_df,drop_first=True)
#
##Lr-2
#my_df=my_df.drop(['OperatingSystems','Browser','Region','Weekend','BounceRates','ExitRates','Administrative_Duration','SpecialDay','Month','ProductRelated_Duration','ProductRelated','Administrative','Informational','Informational_Duration','TrafficType'],axis=1)
#X = pd.get_dummies(my_df,drop_first=True)
#
##Lr-1
#my_df=my_df.drop(['VisitorType','OperatingSystems','Browser','Region','Weekend','BounceRates','ExitRates','Administrative_Duration','SpecialDay','Month','ProductRelated_Duration','ProductRelated','Administrative','Informational','Informational_Duration','TrafficType'],axis=1)
#X = pd.get_dummies(my_df,drop_first=True)


#my_df = my_df.drop(['Revenue','Informational','ProductRelated','ProductRelated_Duration','Month'], axis=1)
#X = pd.get_dummies(my_df,drop_first=True)

#X.Weekend = X.Weekend.astype(int)
#
#X.head()



from sklearn.preprocessing import StandardScaler
#scalar = StandardScaler()
#X = scalar.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Function for calculating accuracy
def accuracy(y_test,y_pred):
    from sklearn.metrics import confusion_matrix
    result = confusion_matrix(y_pred,y_test)
    acc = ((result[0][0]+result[1][1])/(len(y_test)))*100
    return acc

print("\n Standardising and train test split done Successfully")


from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
#**MODEL BUILDING**
 
# Support Vector Machine
print("\n MODEL BUILDING")

print("\n Building Support Vector Machine")
from sklearn.svm import SVC
svc = SVC(random_state=42)
###efsrandom = EFS(svc, 
###           min_features=3,
###           max_features=3,
###           scoring='accuracy',
###           print_progress=True,
###           n_jobs=-1,
###           cv=5) #cross validation fold
###efsrandom = efsrandom.fit(X, y)
###print('Best accuracy score: %.2f' % efsrandom.best_score_)
###print('Best subset (indices):', efsrandom.best_idx_)
###print('Best subset (corresponding names):', efsrandom.best_feature_names_)
##

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

## Sequential Forward Selection
#sfs = SFS(svc, 
#          k_features=17, 
#          forward=True, 
#          floating=False, 
#          scoring='accuracy',
#          cv=4,
#          n_jobs=-1)
#sfs = sfs.fit(X, y)
#
#print('\nSequential Forward Selection (k=17):')
#print(sfs.subsets_)


#fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')
#plt.ylim([0.8, 1])
#plt.title('Sequential Forward Selection')
#plt.grid()
#plt.show()

model = svc.fit(X_train,y_train)
y_pred_svc = model.predict(X_test)
print("\n Done")
print("\n Accuracy of SVM: ",accuracy(y_test,y_pred_svc))


#
#Logistic Regression
print("\n Building Logistic Regression")
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=42)


#efslr = EFS(lr, 
#           min_features=6,
#           max_features=6,
#           scoring='accuracy',
#           print_progress=True,
#           n_jobs=-1,
#           cv=5) #cross validation fold
#efslr = efslr.fit(X, y)
#print('Best accuracy score: %.2f' % efslr.best_score_)
#print('Best subset (indices):', efslr.best_idx_)
#print('Best subset (corresponding names):', efslr.best_feature_names_)
#

model_lr = lr.fit(X_train,y_train)
y_pred_lr = model_lr.predict(X_test)
print("\n Done")
print("\n Accuracy of Logistic Regression: ",accuracy(y_test,y_pred_lr))


print("\n Building Random Forest Model")
# Random Forest
from sklearn.ensemble import RandomForestClassifier
model_rf_classi = RandomForestClassifier(max_depth=2, random_state=10)
#
##efsrandom = EFS(model_rf_classi, 
##           min_features=9,
##           max_features=9,
##           scoring='accuracy',
##           print_progress=True,
##           n_jobs=-1,
##           cv=5) #cross validation fold
#
#
#
#
##efsrandom = efsrandom.fit(X, y)
##print('Best accuracy score: %.2f' % efsrandom.best_score_)
##print('Best subset (indices):', efsrandom.best_idx_)
##print('Best subset (corresponding names):', efsrandom.best_feature_names_)
#
#
#
model_rf = model_rf_classi.fit(X_train,y_train)
y_pred_enrf = model_rf.predict(X_test)
print("\n Done")
print("\n Accuracy of Random Forest: ",accuracy(y_test,y_pred_enrf))


#Multilayer Perceptron
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,200,200,100), max_iter=50, alpha=0.0001,solver='sgd', verbose=10,  random_state=42,tol=0.000000001)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
print("\n Done")
print("\n Accuracy of Multilayer Perceptron: ",accuracy(y_test,y_pred_mlp))




#KNN Classifier
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=40, weights='distance',metric='manhattan', n_jobs=-1)

#efs1 = EFS(knn, 
#           min_features=2,
#           max_features=2,
#           scoring='accuracy',
#           print_progress=True,
#           cv=5) #cross validation fold
##
#
#efs1 = efs1.fit(X, y)
#
#print('Best accuracy score: %.2f' % efs1.best_score_)
#print('Best subset (indices):', efs1.best_idx_)
#print('Best subset (corresponding names):', efs1.best_feature_names_)
#



knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("\n Done")
print("\n Accuracy of KNN: ",accuracy(y_test,y_pred_knn))

# Neural Network Approach
print("\n Building Neural Network Model")
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation,Dense
from livelossplot.keras import PlotLossesCallback

print("\n Training the model")
classifier = Sequential()
classifier.add(Dense(units = 128, activation = 'relu', input_dim = 8))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 128, activation = 'sigmoid'))
classifier.add(Dense(units = 128, activation = 'sigmoid'))
classifier.add(Dense(units = 128, activation="relu"))
classifier.add(Dense(units = 256, activation="relu"))
classifier.add(Dropout(0.1))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compile model
epochs = 100
learning_rate = 0.01
decay_rate = learning_rate / epochs
momentum = 0.8
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train,batch_size = 128,epochs = epochs, verbose=2)




## Setting the figure size and resolution
#fig = plt.figure(figsize=(10, 6), dpi=300)


#history=classifier.fit(X_train, y_train,validation_data=(X_test, y_test), batch_size = 32, epochs = 60, verbose=2)
#
### evaluate the model
##train_acc = classifier.evaluate(X_train, y_train, verbose=0)
##test_acc = classifier.evaluate(X_test, y_test, verbose=0)
##print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
## plot loss during training
#plt.subplot(211)
#plt.title('Loss')
#plt.plot(history.history['loss'], label='train')
#plt.plot(history.history['val_loss'], label='test')
#plt.legend()
## plot accuracy during training
#plt.subplot(212)
#plt.title('Accuracy')
#plt.plot(history.history['acc'], label='train')
#plt.plot(history.history['val_acc'], label='test')
#plt.legend()
#plt.show()



print("\n Done")

#

y_pred_nn = classifier.predict_classes(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
print("SVM")
SVM = classification_report(y_test, y_pred_svc)
print(SVM)
cm_SVM = confusion_matrix(y_test, y_pred_svc)
print("Confusion Matrix of SVM \n",cm_SVM)

print("Logistic Regression")
LR = classification_report(y_test, y_pred_lr)
print(LR)
cm_LR = confusion_matrix(y_test, y_pred_lr)
print("Confusion Matrix of Logistic Regression \n",cm_LR)

print("RF")
RF = classification_report(y_test, y_pred_enrf)
print(RF)
cm_RF = confusion_matrix(y_test, y_pred_enrf)
print("Confusion Matrix of Random Forest \n",cm_RF)
#
#print("Neural Network")
#NN = classification_report(y_test, y_pred_nn)
#print(NN)
#cm_NN = confusion_matrix(y_test, y_pred_nn)
#print(cm_NN)
#print("Confusion Matrix of Neural Network \n",cm_NN)

print("Multilayer Perceptron")
MLP = classification_report(y_test, y_pred_mlp)
print(MLP)
cm_MLP = confusion_matrix(y_test, y_pred_mlp)
print("Confusion Matrix of Multilayer Perceptron \n",cm_MLP)

print("KNN")
KNN = classification_report(y_test, y_pred_knn)
print(KNN)
cm_KNN = confusion_matrix(y_test, y_pred_knn)
print("Confusion Matrix of KNN \n",cm_KNN)

print("\n Accuracy of Neural Network: ",accuracy(y_test,y_pred_nn))
data= [['SVM',accuracy(y_test,y_pred_svc)],['LogisticRegression',accuracy(y_test,y_pred_lr)],['DecisionTree',accuracy(y_test,y_pred_enrf)],['LinearDiscriminant',accuracy(y_test,y_pred_nn)],['Gaussian Naive Bayes',accuracy(y_test,y_pred_mlp)],['KNN',accuracy(y_test,y_pred_knn)]]
accuracy_compare = pd.DataFrame(data, columns = ['Method', 'Accuracy'])

sns.barplot(x=accuracy_compare['Method'],y=accuracy_compare['Accuracy'])
plt.show()