# Grey Files
# Muhammad Alahmadi
# Md Mirajul Islam

import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
import csv
import pandas as pd
import sklearn.metrics
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier


from sampling import one_sided_selection, random_undersampling, smote

def get_raw_data(n,noTest= False,test_size = 0.2):
    """
    Read the raw data from the source files located in the ECONet directory

    Input: None

    Output:
        train: panda containing all of the training data including target values
        test: panda containing all of the testing features without target values
    """
    file1 = open('ECONet/train.csv')
    file2 = open('ECONet/test.csv')
    csvreader1 = csv.reader(file1)
    csvreader2 = csv.reader(file2)
    header1 = []
    header2 = []
    header1 = next(csvreader1)
    header2 = next(csvreader2)
    rows1 = []
    rows2 = []
    i = 0
    for row in csvreader1:
        if i >= n:
            break
        i += 1
        rows1.append(row)
    
    '''if noTest:
        i = 0
        test_index = int(n * (test_size + 1))
        for row in csvreader2:
            if i < n:
                continue
            elif i >= test_index:
                break
            rows2.append(row)
            i += 1
    else:

        for row in csvreader2:
            rows2.append(row)'''
            
    train = pd.DataFrame(data=rows1, columns=header1)
    test = pd.DataFrame(data=rows2, columns=header2)
    
    return train, test

def get_data(n,noTest=False):
    """
    Get the data from the ECONet dataset in useable format

    Input: None

    Output:
        train_x: panda containing all of the attributes of the training data set
        train_y: panda containing all of the target values of the training data set
        test_x: panda containing all of the attributes of the test data set
    """
    train, test_x = get_raw_data(n,noTest)
    train.drop(['Station', 'Ob', 'value', 'measure'], inplace=True, axis=1)
    test_x.drop(['Station', 'Ob', 'value', 'measure'], inplace=True, axis=1)
    train_y = train['target']
    train_x = train.drop('target', axis=1)
    return train_x, train_y, test_x


def train_adaboost_classifier(x_data, y_data, n_estimators, random_state=0):
    """
    Train an AdaBoost classifier on the given data

    Input:
        x_data: attributes used to train the classifier
        y_data: target values used to train the classifier
        n_estimators: max number of estimators used in the AdaBoost classifier
        random_state: random seed for the classifier
    
    Output:
        trained AdaBoost classifier
    """
    return AdaBoostClassifier(n_estimators=n_estimators, random_state=random_state).fit(x_data, y_data)

def train_svm_classifier(x_data, y_data, C=1.0, random_state=0):
    """
    Train an SVM classifier on the given data

    Input:
        x_data: attributes used to train the classifier
        y_data: target values used to train the classifier
        C: regularization parameter for the classifier
        random_state: random seed for the classifier
    
    Output:
        trained SVM classifier
    """
    return SVC(C=C, random_state=random_state).fit(x_data, y_data)

    
def cross_validation(X,Y,test_size,k,model,random_seed):

    scores = cross_val_score(model, X, Y, cv=KFold(n_splits=k), scoring='f1_macro')

    # Manual Cross Validation

    '''kf = KFold(n_splits=k)
    scores = []
    
    for train_index, test_index in kf.split(X):
        
        model.fit(X.iloc[train_index], Y.iloc[train_index])
        pred = model.predict(X.iloc[test_index])
        scores.append(accuracy_score(Y.iloc[test_index],pred))'''
            
    return scores

sv = False
ada = False
rf = False
# start = time.process_time()


numberOfRows = float('inf')
X, Y, test_x = get_data(n= numberOfRows, noTest= True)

# spiltting the data
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size = 0.2, random_state = 0,shuffle=False)

# sampling
res_x_train,res_y_train = one_sided_selection(X_train,Y_train)
res2_x_train,res2_y_train = random_undersampling(X_train,Y_train)
res3_x_train,res3_y_train = smote(X_train,Y_train)

# random forest

if rf:
    # Not done
    # To do

    model = train_svm_classifier(X_train,Y_train)
    print(f1_score(Y_test, model.predict(X_test), average='macro'))

    model2 = train_svm_classifier(res_x_train,res_y_train)
    print(f1_score(Y_test, model2.predict(X_test), average='macro'))

    model3 = train_svm_classifier(res2_x_train,res2_y_train)
    print(f1_score(Y_test, model3.predict(X_test), average='macro'))

    model4 = train_svm_classifier(res3_x_train,res3_y_train)
    print(f1_score(Y_test, model4.predict(X_test), average='macro'))


# SVM
if sv:

    model = train_svm_classifier(X_train,Y_train)
    print(f1_score(Y_test, model.predict(X_test), average='macro'))

    model2 = train_svm_classifier(res_x_train,res_y_train)
    print(f1_score(Y_test, model2.predict(X_test), average='macro'))

    model3 = train_svm_classifier(res2_x_train,res2_y_train)
    print(f1_score(Y_test, model3.predict(X_test), average='macro'))

    model4 = train_svm_classifier(res3_x_train,res3_y_train)
    print(f1_score(Y_test, model4.predict(X_test), average='macro'))



#ada boost 
if ada:
    n_estimators = 20
    model = train_adaboost_classifier(X_train,Y_train,n_estimators=n_estimators)
    print(f1_score(Y_test, model.predict(X_test), average='macro'))

    model2 = train_adaboost_classifier(res_x_train,res_y_train,n_estimators=n_estimators)
    print(f1_score(Y_test, model2.predict(X_test), average='macro'))

    model3 = train_adaboost_classifier(res2_x_train,res2_y_train,n_estimators=n_estimators)
    print(f1_score(Y_test, model3.predict(X_test), average='macro'))

    model4 = train_adaboost_classifier(res3_x_train,res3_y_train,n_estimators=n_estimators)
    print(f1_score(Y_test, model4.predict(X_test), average='macro'))


#print(time.process_time() - start)



# code for cross validation - to do later

'''scores = cross_validation(train_x,Y_train,test_size= 0.2 ,k= 7,random_seed= 0,model = model)
print(np.mean(scores))
scores = cross_validation(res_x_train,res_y_train,test_size= 0.2 ,k= 7,random_seed= 0,model = model)
print(np.mean(scores))
scores = cross_validation(res2_x_train,res2_y_train,test_size= 0.2 ,k= 7,random_seed= 0,model = model)
print(np.mean(scores))
scores = cross_validation(res3_x_train,res3_y_train,test_size= 0.2 ,k= 7,random_seed= 0,model = model)
print(np.mean(scores))'''
