#TF
import re
stopword = open('stopword_chinese.txt','r',encoding='UTF-8').read()
punctuation = open('punctuation.txt','r',encoding='UTF-8').read()
text_lines = open('TMBD_news_files_2000.txt','r',encoding='UTF-8').readlines()
text = open('TMBD_news_files_2000.txt','r',encoding='UTF-8').read()
text_re = re.sub('[Ｏ○●@※+:．;,…－─&#《》〈〉『』↑ＡＢＣＤＥＦＧＩＫＬＭＮＰＱＲＳＴＵＶ十■\ufeff\u3000]','', text)
text_ps = ''
for r in text_re:
    if r not in punctuation:
        if r not in stopword:
            text_ps += r
content = text_ps.split()

#TF
def calcTF(doc):
    word_count = dict()
    for word in doc:
        if word in word_count.keys():
            word_count[word] += 1
        else:
            word_count[word] = 1
    for w in word_count:
        word_count[w] = word_count[w]/len(word_count)
    return word_count

tf = calcTF(content)

#DF
def calcDF(doc):
    all = []
    for l in doc:
        all.append(l)

    doc_count = dict()
    for word in tf:
        for d in range(0,len(all)):
            if word in all[d]:
                if word in doc_count.keys():
                    doc_count[word] += 1
                else:
                    doc_count[word] = 1
    return doc_count

df = calcDF(text_lines)

#IDF
def calcIDF(dflist):
    idf_dict = {}
    import math
    for word ,count in dflist.items():
        idf_dict[word] = math.log(len(text_lines)/(1+count))
    return idf_dict

idf = calcIDF(df)

#TF-IDF
def calTFIDF(tfs, idfs):
    tfidf_dict = {}
    for word, value in tfs.items():
        if word in idfs.keys():
            tfidf_dict[word] = value*idfs[word]
    return tfidf_dict

tfidf = calTFIDF(tf, idf)

#doc-tfidf
def docVec(doc):
    arr = []
    for word, value in tfidf.items():
        if word in doc and tfidf.items():
            arr.append(value)
        else:
            arr.append(0)
    return arr

#arrange doc#
for i in range(0, len(text_lines)):
    exec("doc%d = %s" % (i + 1, repr(text_lines[i])))

#doc-vec
import numpy as np
doc_vec = np.array(docVec(text_lines[0]))
for j in range(1, len(text_lines)):
    doc_vec = np.vstack((doc_vec,np.array(docVec(text_lines[j]))))

#Similarity
def simDoc(a, b):
    import numpy as np
    import numpy.linalg as LA
    sim = np.inner(a, b)/(LA.norm(a)*LA.norm(b))
    return sim

#Similarity-compare
cmp_dict = {}
for c in range(0, len(text_lines)):
    cmp_dict[c+1] = (simDoc(doc_vec[55], doc_vec[c]))

#sorting
import operator
sorted_cmp_dict = sorted(cmp_dict.items(), key=operator.itemgetter(1), reverse=True)
for k,y in sorted_cmp_dict:
    if k != 56:
        print(k,y)
