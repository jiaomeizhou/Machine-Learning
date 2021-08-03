import pandas as pd
import re
from prep_and_classifier import *

# =======数据预处理=======

df = pd.read_csv('./data/db_book.txt', header=None, sep='\t', lineterminator='\n') # 读取文件
df["review"] = df[0] + df[2] # 将书名作为特征
df["review"] = df["review"].apply(remove_punctuation) # 去除标点
df_train_all = df.sample(frac=0.8,random_state=0,axis=0) # 按6：2：2的比例划分训练集、测试集和验证集
df_test = df[~df.index.isin(df_train_all.index)] # 测试集
df_train = df_train_all.sample(frac=0.75,random_state=0,axis=0)# 将df_test分成训练集和验证集
df_dev = df_train_all[~df_train_all.index.isin(df_train.index)]
# df_dev = df.sample(frac=0.2,random_state=0,axis=0)
# df_test = df.sample(frac=0.2,random_state=0,axis=0)
df_train.iloc[:, [-1,1]].to_csv('./data/train_data.txt', index=False, sep='\t',header=None) # 23858
df_dev.iloc[:, [-1,1]].to_csv('./data/dev_data.txt', index=False, sep='\t',header=None) # 7953
df_test.iloc[:, [-1,1]].to_csv('./data/test_data.txt', index=False, sep='\t',header=None) #7953
df_train_all.iloc[:, [-1,1]].to_csv('./data/train_data_all.txt', index=False, sep='\t',header=None)


# 创建停用词list
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


# 对句子进行分词
def seg_sentence(sentence):
    sentence_seged = jieba.cut(sentence.strip())
    stopwords = stopwordslist('./data/chineseStopWords.txt')  # 这里加载停用词的路径
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr

# 分词
df_test['cut'] = df_test['review'].apply(seg_sentence)
df_train_all["cut"] = df_train_all["review"].apply(seg_sentence)
df_test.iloc[:, [-1,1]].to_csv('./data/test_data_cut.txt', index=False, sep='\t',header=None)

# =========训练除CNN外的其他模型===========


# 划分训练集和测试集
train_data_cut = df_train_all['cut'].tolist()# 31811
test_data_cut = df_test['cut'].tolist() # 7953
data_all = train_data_cut + test_data_cut # 39764
# 处理标签
train_label = df_train_all[1].tolist()
test_label = df_test[1].tolist()
Y_train = []
Y_test = []
for i in train_label:
    if i == 'N':
        Y_train.append(0)
    elif i == 'P':
        Y_train.append(1)
    else:
        print('error', i)
for i in test_label:
    if i == 'N':
        Y_test.append(0)
    elif i == 'P':
        Y_test.append(1)
    else:
        print('error', i)

# 将文本表示为TFIDF，得出各个模型的结果
word_tfidf = text_to_tfidf_matrix(data_all)
X_train = word_tfidf[:31811]
X_test = word_tfidf[31811:]
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)
model_training_testing(name, model, X_train, X_test, Y_train, Y_test)