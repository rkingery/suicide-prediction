import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score
from scipy import sparse
np.random.seed(42)

path = '/Users/ryankingery/desktop/suicides/data/'

def get_data(path):
    df = pd.read_csv(path+'std_format_raw_data.csv',index_col=0)    
    text = df.text
    labels = df.labels
    # shuffle text and labels together    
    idx = np.random.permutation(len(labels))
    labels = labels[idx]
    text = text[idx]    
    return text,labels

def get_subsample(text,labels,ratio):
    idx = np.random.permutation(len(labels))
    labels = labels[idx]
    text = text[idx]
    
    num_labels_1 = np.sum(labels==1)
    text_label_1 = list(text[labels==1])
    labels_label_1 = list(labels[labels==1])
    text_label_0 = list(text[labels==0][:ratio*(num_labels_1+1)])
    labels_label_0 = list(labels[labels==0][:ratio*(num_labels_1+1)])
    text_balanced = text_label_1+text_label_0
    labels_balanced = np.array(labels_label_1+labels_label_0)
    return text_balanced,labels_balanced

def upsample(X_train,y_train):
    is_0 = np.where(y_train == 0)[0]
    is_1 = np.where(y_train == 1)[0]
    is_1_up = np.random.choice(is_1, size=len(is_0), replace=True)
    #X_train_up = np.concatenate((X_train[is_1_up], X_train[is_0]))
    X_train_up = sparse.vstack([X_train[is_1_up], X_train[is_0]])
    y_train_up = np.concatenate((y_train[is_1_up], y_train[is_0]))
    # reshuffle training data
    idx = np.random.permutation(len(y_train_up))
    X_train_up = X_train_up[idx]
    y_train_up = y_train_up[idx]   
    return X_train_up,y_train_up

def train_subsamples(text,labels,runs=10,ratio=1):
    # need to readjust so that test set has correct distribution
    print('Training subsamples with ratio of '+str(ratio)+':1')
    for i in range(runs):
        text_sub,labels_sub = get_subsample(text,labels,ratio)
        count_vect = CountVectorizer()
        X_counts = count_vect.fit_transform(text_sub)
        tfidf_transformer = TfidfTransformer()
        X_tfidf = tfidf_transformer.fit_transform(X_counts)
        y = labels_sub
        X_train,X_test,y_train,y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42+i)
        if ratio>1:
            X_train,y_train = upsample(X_train,y_train)
        
        model = MultinomialNB()
        model.fit(X_train, y_train)
        
        print('--- Model Evaluation: Iteration '+str(i)+' ---')
        print('training accuracy: ',model.score(X_train,y_train))
        print('test accuracy: ',model.score(X_test,y_test))
        print('AUC score: ',roc_auc_score(y_test,model.predict_proba(X_test)[:,1]))
        print('F score: ',f1_score(y_test,model.predict(X_test)))
        print('Kappa: ',cohen_kappa_score(y_test,model.predict(X_test)))
        print('Precision: ',precision_score(y_test,model.predict(X_test)))
        print('Recall: ',recall_score(y_test,model.predict(X_test)))
        print('Confusion Matrix:')
        print(confusion_matrix(y_test,model.predict(X_test)))

def train_upsample(text,labels):
    count_vect = CountVectorizer()
    count_vect.fit(text)
    X_counts_all = count_vect.fit_transform(text)
    tfidf_transformer = TfidfTransformer()
    tfidf_transformer.fit(X_counts_all)
    X_tfidf_all = tfidf_transformer.fit_transform(X_counts_all)
    y_all = labels.values
    
    X_train,X_test,y_train,y_test = train_test_split(X_tfidf_all, y_all, test_size=0.1, random_state=41)
    X_train_up,y_train_up = upsample(X_train,y_train)
    
    model = MultinomialNB()
    #model = LogisticRegression(C=1e-5)
    model.fit(X_train_up, y_train_up)
    
    print('--- Model Evaluation ---')
    #print('training accuracy: ',model.score(X_train_up,y_train_up))
    print('test accuracy: ',model.score(X_test,y_test))
    print('AUC score: ',roc_auc_score(y_test,model.predict_proba(X_test)[:,1]))
    print('F score: ',f1_score(y_test,model.predict(X_test)))
    print('Kappa: ',cohen_kappa_score(y_test,model.predict(X_test)))
    print('Precision: ',precision_score(y_test,model.predict(X_test)))
    print('Recall: ',recall_score(y_test,model.predict(X_test)))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test,model.predict(X_test)))
    

if __name__ == '__main__':
    text,labels = get_data(path)
    train_subsamples(text,labels,runs=1,ratio=30)


