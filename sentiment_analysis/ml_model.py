import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import json
import re
import pickle
import jieba as jb
import demjson
from prep_and_classifier import *
from prep import text_to_tfidf_matrix, text_to_tfidf_matrix2
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer




# step 1. build a corpus: text word list

# 加载停用词
stopwords = stopwordslist("./chineseStopWords.txt")

# 加载数据
df_usual_train = pd.read_excel('./SMP测试集/train/usual_train.xls')
df_usual_eval = pd.read_excel('./SMP测试集/eval/usual_eval.xls')
df_virus_train = pd.read_excel('./SMP测试集/train/virus_train.xls')
df_virus_eval = pd.read_excel('./SMP测试集/eval/virus_eval.xls')

# 将情绪标签转化为数字
df_usual_train['lable_id'] = df_usual_train['情绪标签'].factorize()[0]
lable_id_df = df_usual_train[['情绪标签', 'lable_id']].drop_duplicates().sort_values('lable_id').reset_index(drop=True)
df_virus_train['lable_id'] = df_virus_train['情绪标签'].factorize()[0]
lable_id_df2 = df_virus_train[['情绪标签', 'lable_id']].drop_duplicates().sort_values('lable_id').reset_index(drop=True)

#删除除字母,数字，汉字以外的所有符号
df_usual_train['clean_content'] = df_usual_train['文本'].apply(remove_punctuation)
df_usual_eval['clean_content'] = df_usual_eval['文本'].apply(remove_punctuation)
df_virus_train['clean_content'] = df_virus_train['文本'].apply(remove_punctuation)
df_virus_eval['clean_content'] = df_virus_eval['文本'].apply(remove_punctuation)
#df_usual_eval.sample(10)
df_usual_train['clean_content'] = df_usual_train['clean_content'].str.cat(df_usual_train['情绪标签']) # 将情感标签也加入训练语料

#分词，并过滤停用词
df_usual_train['cut_content'] = df_usual_train['clean_content'].apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))
df_usual_eval['cut_content'] = df_usual_eval['clean_content'].apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))
df_virus_train['cut_content'] = df_virus_train['clean_content'].apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))
df_virus_eval['cut_content'] = df_virus_eval['clean_content'].apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))
# df_virus_train['cut_content'] = df_virus_train['clean_content'].apply(lambda x: " ".join(jb.cut(x))) # 不去除停用词，效果略差
# df_virus_eval['cut_content'] = df_virus_eval['clean_content'].apply(lambda x: " ".join(jb.cut(x)))

usual_train_data = df_usual_train['cut_content'].values
usual_eval_data = df_usual_eval['cut_content'].values
virus_train_data = df_virus_train['cut_content'].values
virus_eval_data = df_virus_eval['cut_content'].values


# step 2. preprocessing: feature encoding
targets = df_virus_train['lable_id'].values
target = np.array(targets)  # answer
word_tfidf_train = text_to_tfidf_matrix(virus_train_data)  # text feature

# 保存经过fit的vectorizer 与 经过fit的tfidftransformer,预测时使用
vectorizer = CountVectorizer(decode_error="replace")
tfidftransformer = TfidfTransformer()
vec_train = vectorizer.fit_transform(virus_train_data)
train_tfidf = tfidftransformer.fit_transform(vec_train)
feature_path = 'models/feature.pkl'
with open(feature_path, 'wb') as fw:
    pickle.dump(vectorizer.vocabulary_, fw)

tfidftransformer_path = 'models/tfidftransformer.pkl'
with open(tfidftransformer_path, 'wb') as fw:
    pickle.dump(tfidftransformer, fw)

# 加载特征
feature_path = 'models/feature.pkl'
loaded_vec = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open(feature_path, "rb")))

# 加载TfidfTransformer
tfidftransformer_path = 'models/tfidftransformer.pkl'
tfidftransformer = pickle.load(open(tfidftransformer_path, "rb"))

#测试用transform
test_tfidf = tfidftransformer.transform(loaded_vec.transform(virus_eval_data))



# step 3. training and testing

####分割训练集与测试集####

X_train, X_test, Y_train, Y_test = train_test_split(train_tfidf, target, test_size=0.2, shuffle=True)


#######训练与测试#######

print('start to train and test...\n')

# for name, model in classifiers.items():
#     model_training_testing(name, model, X_train, X_test, Y_train,Y_test)
# exit()
# 选择表现最好的模型训练
x_test = test_tfidf
#clf = OneVsOneClassifier(LinearSVC())
clf = OneVsOneClassifier(MultinomialNB())
clf = clf.fit(train_tfidf, target)
predicted = clf.predict(x_test)
#print(clf.score(Y_test, predicted)) #accuracy: 0.7401872524306806; 训练文本拼接标签后，0.9897371263953907


#使用集成分类器：VotingClassifier
voting_clf = VotingClassifier(estimators=[
    ('log_clf', OneVsOneClassifier(LogisticRegression())),
    ('svm_clf', OneVsOneClassifier(SVC(kernel='linear'))),
    ('NB_clf', OneVsOneClassifier(MultinomialNB()))
], voting='hard')
voting_clf.fit(X_train, Y_train)
#voting_clf.score(X_test, Y_test)
predicted = voting_clf.predict(x_test)
predicted = predicted.tolist()
print(len(predicted))

# 将标签从数字恢复成文本标签
sentiments = []
for item in predicted:
    if item == 0:
        sentiments.append('angry')
    if item == 1:
        sentiments.append('happy')
    if item == 2:
        sentiments.append('neural')
    if item == 3:
        sentiments.append('surprise')
    if item == 4:
        sentiments.append('sad')
    if item == 5:
        sentiments.append('fear')
    else:
        # print('Error', item)
        continue
print(sentiments)


#######写入文件########
submission_df = pd.DataFrame(data={'id': df_virus_eval['数据编号'].values, 'label': sentiments})
l = submission_df.to_dict(orient='records')
with open('virus_result.txt','w',encoding='utf-8') as f:
    f.write(json.dumps(l, indent=4))
