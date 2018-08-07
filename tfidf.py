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
from scipy import sparse
np.random.seed(42)

def get_subsample(text,labels):
    idx = np.random.permutation(len(labels))
    labels = labels[idx]
    text = text[idx]
    
    num_labels_1 = np.sum(labels==1)
    ratio = 1 # ratio of 0 labels (non-suicide) to 1 labels (suicide)
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

def train_subsamples(text,labels):
    for i in range(10):
        text_balanced,labels_balanced = get_subsample(text,labels)
        X_counts = count_vect.transform(text_balanced)
        X_tfidf = tfidf_transformer.transform(X_counts)
        y = labels_balanced
        X_train,X_test,y_train,y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=41)
        
        model = MultinomialNB()
        model.fit(X_train, y_train)
        
        print('--- Model Evaluation on Subsample ---')
        print('training accuracy: ',model.score(X_train,y_train))
        print('test accuracy: ',model.score(X_test,y_test))
        print('AUC score: ',roc_auc_score(y_test,model.predict_proba(X_test)[:,1]))
        print('F score: ',f1_score(y_test,model.predict(X_test)))
        print(confusion_matrix(y_test,model.predict(X_test)))


# read data
path = '/Users/ryankingery/desktop/suicides/data/'
df = pd.read_csv(path+'std_format_raw_data.csv',index_col=0)

text = df.text
labels = df.labels

idx = np.random.permutation(len(labels))
labels = labels[idx]
text = text[idx]

count_vect = CountVectorizer()
count_vect.fit(text)
X_counts_all = count_vect.transform(text)

#tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
#X_train_tf = tf_transformer.transform(X_train_counts)
#X_train_tf.shape

tfidf_transformer = TfidfTransformer()
tfidf_transformer.fit(X_counts_all)
X_tfidf_all = tfidf_transformer.transform(X_counts_all)
y_all = df.labels.values

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
print(confusion_matrix(y_test,model.predict(X_test)))



