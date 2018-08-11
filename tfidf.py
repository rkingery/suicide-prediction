import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE, ADASYN
from scipy import sparse
np.random.seed(42)

path = '/Users/ryankingery/desktop/suicides/data/'

def get_data(path):
    #df = pd.read_csv(path+'std_format_raw_data.csv',index_col=0) 
    df = pd.read_csv(path+'text_with_labels.csv')
    text = df.text
    labels = df.labels
    # shuffle text and labels together    
    idx = np.random.permutation(len(labels))
    labels = labels[idx]
    text = text[idx]    
    return text,labels

def numericalize_data(text,labels):
    count_vect = CountVectorizer()
    count_vect.fit(text)
    X_counts_all = count_vect.fit_transform(text)
    tfidf_transformer = TfidfTransformer()
    tfidf_transformer.fit(X_counts_all)
    X_tfidf_all = tfidf_transformer.fit_transform(X_counts_all)
    y_all = labels.values
    return X_tfidf_all,y_all

def subsample(X_train,y_train,ratio):
    idx = np.random.permutation(len(y_train))
    y_train = y_train[idx]
    X_train = X_train[idx]
    
    num_labels_1 = np.sum(y_train==1)
    X_label_1 = X_train[y_train==1]
    y_label_1 = y_train[y_train==1]
    X_label_0 = X_train[y_train==0][:ratio*(num_labels_1+1)]
    y_label_0 = y_train[y_train==0][:ratio*(num_labels_1+1)]
    X_balanced = sparse.vstack([X_label_1,X_label_0])
    y_balanced = np.concatenate([y_label_1,y_label_0])
    
    # shuffle before returning   
    idx = np.random.permutation(len(y_balanced))
    y_balanced = y_balanced[idx]
    X_balanced = X_balanced[idx] 
    return X_balanced,y_balanced

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
    
def train_subsamples(X_tfidf_all,y_all,runs=10,ratio=1):
    X_train,X_test,y_train,y_test = train_test_split(X_tfidf_all, y_all, 
                                                     test_size=0.1, random_state=41)   
    print('Training subsamples with ratio of '+str(ratio)+':1')
    for i in range(runs):
        X_train,y_train = subsample(X_train,y_train,ratio)
        if ratio>1:
            #X_train,y_train = upsample(X_train,y_train)
            X_train,y_train = SMOTE().fit_sample(X_train, y_train)
            #X_train,y_train = ADASYN().fit_sample(X_train, y_train)
        
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

def train_upsample(X_tfidf_all,y_all):
    
    X_train,X_test,y_train,y_test = train_test_split(X_tfidf_all, y_all, 
                                                     test_size=0.2, random_state=41)
    X_train_up,y_train_up = upsample(X_train,y_train)
    #X_train,y_train = SMOTE().fit_sample(X_train, y_train)
    #X_train,y_train = ADASYN().fit_sample(X_train, y_train)
    
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
    
def visualize_tfidf(X_tfidf_all,y_all):
    
    svd = TruncatedSVD(2)
    X_pca = svd.fit_transform(X_tfidf_all)
    
    labels = ['non-suicides','suicides']
    alphas = [0.2,1.]
    sizes = [.5,1.]
    colors = ['blue','orange']
    for i in [0,1]:
        plt.scatter(X_pca[y_all == i, 0], X_pca[y_all == i, 1], s=sizes[i], c=colors[i],
                    label=labels[i], alpha=alphas[i])
    plt.title('PCA on tf-idf of clinical notes')
    plt.legend(loc='upper left')
    plt.show()
    
    for i in [0,1]:
        for j in [0,1]:
            plt.hist(X_pca[y_all==j,i],density=True,label=labels[j])
        plt.title('Histogram: PCA dim '+str(i+1))
        plt.legend(loc='upper left')
        plt.show()
        
    for i in [0,1]:
        plt.hist2d(X_pca[y_all == i, 0], X_pca[y_all == i, 1])
        plt.title('2D histogram of label '+str(i))
        plt.show()


if __name__ == '__main__':
    text,labels = get_data(path)
    X_tfidf_all,y_all = numericalize_data(text,labels)
    train_subsamples(X_tfidf_all,y_all,runs=1,ratio=95)
    #train_upsample(X_tfidf_all,y_all)
    #visualize_tfidf(X_tfidf_all,y_all)


