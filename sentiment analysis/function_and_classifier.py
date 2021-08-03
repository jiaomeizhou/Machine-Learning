# -*- coding: utf-8 -*-
import jieba
import jieba.analyse
import time
from sklearn.metrics import f1_score
import sklearn
import time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def text_to_tfidf_matrix(corpus):
    '''
        This function transforms a corpus into a tf-idf weighted matrix
        corpus: a list of sentences (word segmented)
        word_tfidf_matrix: a tf-idf weighted document-word matrix
    '''

    vectorizer = CountVectorizer(ngram_range=(1, 1))  # 抽取器，ngram参数可设置
    word_counts = vectorizer.fit_transform(corpus)  # 稀疏向量的压缩矩阵 <class 'scipy.sparse.csr.csr_matrix'>
    print('counted words in training data...')
    #     features = vectorizer.get_feature_names()
    #     for i, feature in enumerate(features):
    #         print(i, feature)
    # print(word_counts)

    tfidf_transformer = TfidfTransformer()
    word_tfidf_matrix = tfidf_transformer.fit_transform(
        word_counts)  # 稀疏向量*tfidf权重后的压缩矩阵 <class 'scipy.sparse.csr.csr_matrix'>
    print('create tfidf matrix on training data...')

    return word_tfidf_matrix


classifiers = {'BernoulliNB': BernoulliNB(), 'MultinomialNB': MultinomialNB(),
               'linearSVC': LinearSVC(), 'KNN': KNeighborsClassifier(),
               'DecisionTree': DecisionTreeClassifier(), 'RandomForest': RandomForestClassifier(),
               'LogisticRegression': LogisticRegression()}


def model_training_testing(name, model, X_train, X_test, Y_train, Y_test):
    print('==========={}==========='.format(name))
    t1 = time.time()
    # training
    clf = model.fit(X_train, Y_train)
    t2 = time.time()
    print( '{} training finished, and used {:.4f} seconds'.format(str(name), t2-t1))
    # testing
    predicted = clf.predict(X_test)
    score = sklearn.metrics.f1_score(Y_test, predicted, labels=None, pos_label=1, average='binary', sample_weight=None)
    print(score)
    print(Y_test)
    #accuracy = np.mean(cross_val_score(model, X_test, Y_test, cv=10, scoring='roc_auc'))#设置10折交叉验证
    #print ('training time: {}, accuracy: {}\n'.format(t2-t1, accuracy))



# 去除标点的函数
def remove_punctuation(line):
    line = str(line)
    if line.strip()=='':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('',line)
    return line