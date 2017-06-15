import pandas as pd
import numpy as np
stopword = open('stopword_chinese.txt','r',encoding='UTF-8').read().split()
punctuation = open('punctuation.txt','r',encoding='UTF-8').read().split()
train_data = open('TrainingData.txt','r',encoding='UTF-8').readlines()
test_data = open('TestData.txt','r',encoding='UTF-8').readlines()
train_data_id = []
train_data_class = []
train_data_article = []
test_data_id = []
test_data_article = []
#建立標點符號和停用字的集合
punctuation.extend(stopword)
punctuationStopword = punctuation
for item in train_data :
    train_data_id.append(item.split("\t")[0])
    train_data_class.append(item.split("\t")[1])  
    train_data_article1 = item.split("\t")[2].strip("\n")
    train_data_article_re = ''.join([r for r in train_data_article1 if r not in punctuationStopword])
    train_data_article.append(train_data_article_re)
for item in test_data :
    test_data_id.append(item.split("\t")[0]) 
    test_data_article1 = item.split("\t")[1].strip("\n")
    test_data_article_re = ''.join([r for r in test_data_article1 if r not in punctuationStopword])
    test_data_article.append(test_data_article_re)
#將訓練集與測試集放入pandas的dataframe
trainSet={"ID":train_data_id, "class":train_data_class, "article":train_data_article}
trainSet_df = pd.DataFrame(trainSet)
testSet={"ID":test_data_id, "article":test_data_article}
testSet_df = pd.DataFrame(testSet)
#將訓練集分類給與0,1的label
class_mapping = {label:idx for idx,label in enumerate(np.unique(trainSet_df["class"]))}
trainSet_df["class"] = trainSet_df["class"].map(class_mapping)
X_train = trainSet_df.loc[:,'article'];
y_train = trainSet_df.loc[:, 'class'];
X_test =  testSet_df.loc[:,'article'];

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

#使用tfidf計算feature weighting
vectorizer = TfidfVectorizer(min_df=0.05,
                             max_df = 0.7,
                             sublinear_tf=True,
                             stop_words= punctuationStopword,
                             use_idf=True)
train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)
#使用Multinomial Naive Bayes分類器，alpha為模型平滑參數
clf = MultinomialNB(alpha=0.01, fit_prior=True, class_prior=None)
fit_MultinomialNB = clf.fit(train_vectors, y_train)
pred = clf.predict(test_vectors)
testSet_df["predict"] = pred
#將1,0的分類對回sport與politics
inv_class_mapping = {v: k for k, v in class_mapping.items()}
testSet_df["Similarity"] = testSet_df["predict"].map(inv_class_mapping)
testSet_df.to_csv('testSet_df.csv')
