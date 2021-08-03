import pandas as pd
from prep_and_classifier import *


# 加载停用词
stopwords = stopwordslist("./chineseStopWords.txt")

# 加载数据
df_usual_train = pd.read_excel('./data/train/usual_train.xls')

# 去掉标点
df_virus_train = pd.read_excel('./data/train/virus_train.xls')
df_virus_eval = pd.read_excel('./data/eval/virus_eval.xls')
df_usual_train["文本"] = df_usual_train["文本"].apply(remove_punctuation)
df_virus_train["文本"] = df_virus_train["文本"].apply(remove_punctuation)
df_virus_eval["文本"] = df_virus_eval["文本"].apply(remove_punctuation)

# 将训练数据按比例分割成三部分，训练集-验证集-测试集
virus_train_data = df_virus_train.sample(frac=0.6,random_state=0,axis=0)
virus_other_data = df_virus_train[~df_virus_train.index.isin(virus_train_data.index)]
virus_dev_data = virus_other_data.sample(frac=0.6,random_state=0,axis=0)
virus_test_data = virus_other_data[~virus_other_data.index.isin(virus_dev_data.index)]
# print(usual_train_data.shape, usual_dev_data.shape, usual_test_data.shape)
# 将分割好的数据写入txt
virus_train_data.iloc[:, [1, 2]].to_csv('./data/virus_train.txt', index=False, sep='\t',header=None)
virus_dev_data.iloc[:, [1, 2]].to_csv('./data/virus_dev.txt', index=False, sep='\t',header=None)
virus_test_data.iloc[:, [1, 2]].to_csv('./data/virus_test.txt', index=False, sep='\t',header=None)
df_virus_eval.iloc[:, 1].to_csv('./data/virus_realtest.txt', index=False, sep='\t',header=None)

df_virus_eval.iloc[:, 1].to_csv('./data/virus_test.txt', index=False, sep='\t',header=None)
df_usual_eval = pd.read_excel('./data/eval/usual_eval.xls')
df_usual_eval["文本"] = df_usual_eval["文本"].apply(remove_punctuation)
df_usual_eval.iloc[:, 1].to_csv('./data/usual_test.txt', index=False, sep='\t',header=None)


