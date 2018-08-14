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
from sklearn.externals import joblib
from imblearn.over_sampling import SMOTE, ADASYN
from scipy import sparse
import xgboost as xgb
from pathlib import Path
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
    count_vect = CountVectorizer(max_features=100000)
    count_vect.fit(text)
    X_counts = count_vect.fit_transform(text)
    tfidf_tr = TfidfTransformer()
    tfidf_tr.fit(X_counts)
    X_tfidf = tfidf_tr.fit_transform(X_counts)
    y = labels.values
    return X_tfidf,y,count_vect,tfidf_tr

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
    
def train(X_tfidf,y,ratio=1):
    X_train,X_test,y_train,y_test = train_test_split(X_tfidf, y, 
                                                     test_size=0.1, random_state=41) 
    if not Path(path+'model.pkl').exists():
        print('Training subsamples with ratio of '+str(ratio)+':1')
        if ratio is not 'all':   
            X_train,y_train = subsample(X_train,y_train,ratio)
        #X_train,y_train = upsample(X_train,y_train)
        X_train,y_train = SMOTE().fit_sample(X_train, y_train)
        #X_train,y_train = ADASYN().fit_sample(X_train, y_train)
    
        #model = MultinomialNB()
        #model = RandomForestClassifier(n_estimators=60,min_samples_leaf=13,
        #                               random_state=2,max_features=.5,oob_score=True,
        #                               n_jobs=-1)
        model = xgb.XGBClassifier(n_estimators=100,seed=2)
        model.fit(X_train, y_train)
        joblib.dump(model, path+'model.pkl')
    else:
        model = joblib.load(path+'model.pkl')
    
    print('--- XGBoost Model Evaluation ---')
    print('training accuracy: ',model.score(X_train,y_train))
    print('test accuracy: ',model.score(X_test,y_test))
    print('AUC score: ',roc_auc_score(y_test,model.predict_proba(X_test)[:,1]))
    print('F score: ',f1_score(y_test,model.predict(X_test)))
    print('Kappa: ',cohen_kappa_score(y_test,model.predict(X_test)))
    print('Precision: ',precision_score(y_test,model.predict(X_test)))
    print('Recall: ',recall_score(y_test,model.predict(X_test)))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test,model.predict(X_test)))
    
    return model
    
def visualize_tfidf(X_tfidf,y):
    
    svd = TruncatedSVD(2)
    X_pca = svd.fit_transform(X_tfidf)
    
    labels = ['non-suicides','suicides']
    alphas = [0.2,1.]
    sizes = [.5,1.]
    colors = ['blue','orange']
    for i in [0,1]:
        plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], s=sizes[i], c=colors[i],
                    label=labels[i], alpha=alphas[i])
    plt.title('PCA on tf-idf of clinical notes')
    plt.legend(loc='upper left')
    plt.show()
    
    for i in [0,1]:
        for j in [0,1]:
            plt.hist(X_pca[y==j,i],density=True,label=labels[j])
        plt.title('Histogram: PCA dim '+str(i+1))
        plt.legend(loc='upper left')
        plt.show()
        
    for i in [0,1]:
        plt.hist2d(X_pca[y == i, 0], X_pca[y == i, 1])
        plt.title('2D histogram of label '+str(i))
        plt.show()

def plot_feature_importances(model,int_to_str,max_num=10):
    top_importances = -np.sort(-model.feature_importances_)[:max_num]
    top_features = np.argsort(-model.feature_importances_)[:max_num]
    top_words = [int_to_str[i] for i in top_features]
    
    plt.figure()
    plt.title('Feature importances')
    plt.barh(range(max_num), top_importances,
             color='r', align='center')
    plt.yticks(range(max_num), top_words)
    plt.ylim([-1, max_num])
    plt.show()

if __name__ == '__main__':
    text,labels = get_data(path)
    X_tfidf,y,count_vect,tfidf_tr = numericalize_data(text,labels)
    model = train(X_tfidf,y,ratio='all')
    str_to_int = count_vect.vocabulary_
    int_to_str = {val:key for key,val in str_to_int.items()}
    plot_feature_importances(model,int_to_str)
    #visualize_tfidf(X_tfidf_all,y_all)


