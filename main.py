import math
from collections import OrderedDict
import numpy as np
import pandas as pd
import sys
from heapq import nlargest
docSize=100
pagesize=5
championlist_r=math.ceil(pagesize*1.5)
elimination_idf_ts=2
doc_eliminate_ts=0.8

 #extracting the tokens
champion=OrderedDict()
dict= {}
term_freq={} # total frequency(in all docs)
suffix=["یم","ید","ند","م","ی",'د']
#step1 & 2
persian=['\u200c','آ','ا','ب','پ','ت','ث','ج','چ','ح','خ','د','ذ','ر','ز','ژ','س','ش','ص','ض','ط','ظ','ع','غ','ف','ق','ک','گ','ل','م','ن','و' ,'ه','ی']
persian_dict={}
tf={}

for i in range(len(persian)):
    persian_dict[persian[i]]=i


#defining tf: a dictionary: keys = docID , value = (a dictionary: key=term , value=term freq )
# {docID,{term,term_freq_in_docID}}

for i in range(docSize):
    f=  open("docs/"+str(i+1)+".txt", "r",encoding="utf-8")
    wordandfreq={}
    for line in f:
        for word in line.split():
            word2=''.join(ch for ch in word if ch.isalpha() or ch=="\u200c")
            dict.setdefault(word2,[]).append(i+1)
            term_freq.setdefault(word2,[]).append(1)
            if word2 not in wordandfreq.keys():
                wordandfreq[word2]=1
            else:
                wordandfreq[word2]=wordandfreq[word2]+1

    tf[str(i+1)]={}
    tf[str(i+1)]=wordandfreq


#print(tf['1'])
#removing duplicates from docIDs
for i in dict:
    temp=[]
    for j in range(len(dict[i])):
        if dict[i][j] not in temp:
            temp.append(dict[i][j])
    dict[i]=temp

#adding frequency
for word in dict.keys():
    list=dict[word]
    list.insert(0,len(list))
    dict[word]=list
def add_freq(word,dict):
    list=dict[word]
    list.insert(0,len(list))
    dict[word]=list
    return dict[word]

#dict["آمار"]=[7,20,39,40,68,87,90,99]
#step3
#3.1 removing 'می' from the beginning and 3.2 removing suffixes
#includes step4
def get_suffix(word):
    for s in suffix:
        if word.endswith(s):
            word2=word.replace(s,"")
            break
        else:
            word2=word
    return word2

def remove_pre_suf(dict):
    dict2={}
    key_list=()
    key_list=dict.keys()
    for word in key_list:
        if "می‌" in word:
            #print(word,dict[word])
            word2=word.replace("می\u200c","")
            word2=get_suffix(word2)
            if word2 in dict.keys():
                tmp=dict[word2][1:]
                dict2[word2]=tmp
                for i in dict[word][1:]:
                    if i not in dict2[word2]:
                        dict2[word2].append(i)
                dict2[word2]=add_freq(word2,dict2)
            elif word2 != "\u200c":
                list=dict[word]
                dict2[word2]=list
            #dict.pop(word)
            #print(word2,dict2[word2])
        else:
            dict2[word]=dict[word]
    return dict2
dict=remove_pre_suf(dict)

#3.3
#removing "؛"ها and "های" from the end of the words
def remove_plural(dict):
    dict2={}
    for word in dict.keys():
        if word.endswith("ها") and word != "ها" :
            #print(word,dict[word])
            word2=word.replace("ها","")
            if word2 in dict.keys():
                tmp=dict[word2][1:]
                dict2[word2]=tmp
                for i in dict[word][1:]:
                    if i not in dict2[word2]:
                        dict2[word2].append(i)
                dict2[word2]=add_freq(word2,dict2)
            elif word2 != "\u200c":
                list=dict[word]
                dict2[word2]=list
            #dict.pop(word)
            #print(word2,dict2[word2])
        else:
            dict2[word]=dict[word]
    return dict2

def remove_plural_y(dict):
    dict2={}
    for word in dict.keys():
        if word.endswith("های") and word != "های" :
            #print(word,dict[word])
            word2=word.replace("های","")
            if word2 in dict.keys():
                tmp=dict[word2][1:]
                dict2[word2]=tmp
                for i in dict[word][1:]:
                    if i not in dict2[word2]:
                        dict2[word2].append(i)
                dict2[word2]=add_freq(word2,dict2)
            elif word2 != "\u200c":
                list=dict[word]
                dict2[word2]=list
            #dict.pop(word)
            #print(word2,dict2[word2])
        else:
            dict2[word]=dict[word]
    return dict2
dict=remove_plural(dict)
#print("first",dict["آمار"])
dict=remove_plural_y(dict)
#print("first",dict["آمار"])
#3.4
#removing "تر" and "ترین"
def remove_comperative(dict):
    dict2={}
    for word in dict.keys():
        if word.endswith("تر") and word != "تر" :
            #print(word,dict[word])
            word2=word.replace("تر","")
            if word2 in dict.keys():
                tmp=dict[word2][1:]
                dict2[word2]=tmp
                for i in dict[word][1:]:
                    if i not in dict2[word2]:
                        dict2[word2].append(i)
                dict2[word2]=add_freq(word2,dict2)
            elif word2 != "\u200c":
                list=dict[word]
                dict2[word2]=list
            #dict.pop(word)
            #print(word2,dict2[word2])
        else:
            dict2[word]=dict[word]
    return dict2
dict=remove_comperative(dict)

def remove_superlative(dict):
    dict2={}
    for word in dict.keys():
        if word.endswith("ترین") and word != "ترین" :
            #print(word,dict[word])
            word2=word.replace("ترین","")
            if word2 in dict.keys():
                tmp=dict[word2][1:]
                dict2[word2]=tmp
                for i in dict[word][1:]:
                    if i not in dict2[word2]:
                        dict2[word2].append(i)
                dict2[word2]=add_freq(word2,dict2)
            elif word2 != "\u200c":
                list=dict[word]
                dict2[word2]=list
            #dict.pop(word)
            #print(word2,dict2[word2])
        else:
            dict2[word]=dict[word]
    return dict2
dict=remove_superlative(dict)

#3.5 convert words with half-space to completely joint words
def remove_halfspace(dict):
    dict2={}
    for word in dict.keys():
        if "\u200c" in word:
           # print(word,dict[word])
            if word.endswith("ای"):
                word3=word.replace("ای","")
            else:
                word3=word
            word2=word3.replace("\u200c","")
            if word2 in dict.keys():
                tmp=dict[word2][1:]
                dict2[word2]=tmp
                for i in dict[word][1:]:
                    if i not in dict2[word2]:
                        dict2[word2].append(i)
                dict2[word2]=add_freq(word2,dict2)
            elif word2 != "\u200c":
                list=dict[word]
                dict2[word2]=list
            #dict.pop(word)
           # print(word2,dict2[word2])
        else:
            dict2[word]=dict[word]
    return dict2
dict=remove_halfspace(dict)


#step 5
# Python program for implementation of heap Sort


# heap sort
def heapify(arr, n, i):
    largest = i # Initialize largest as root
    l = 2 * i + 1	 # left = 2*i + 1
    r = 2 * i + 2	 # right = 2*i + 2
    if l < n and arr[i] < arr[l]:
        largest = l
    if r < n and arr[largest] < arr[r]:
        largest = r
    if largest != i:
        arr[i],arr[largest] = arr[largest],arr[i] # swap .
        heapify(arr, n, largest)
    # The main function to sort an array of given size
def heapSort(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i] # swap
        heapify(arr, i, 0)

descending_freq=[]
ten_frequent_terms=[]
for k in sorted(term_freq, key=lambda k: len(term_freq[k]), reverse=True):
    descending_freq.append(k)
ten_frequent_terms=descending_freq[1:11]
#print(ten_frequent_terms)
#removing the frequent items from inverted index
def remove_freq(dict):
    dict2={}
    for word in dict.keys():
        list=dict[word]
        if word not in ten_frequent_terms:
            dict2[word]=list
    return dict2

dict=remove_freq(dict)


#all the operations unified

def hamsansaz(dict):
    dict2={}
    dict3={}
    dict4={}
    dict5={}
    dict6={}
    dict7={}
    dict8={}
    dict2=remove_pre_suf(dict)
    dict3=remove_plural(dict2)
    dict4=remove_plural_y(dict3)
    dict5=remove_comperative(dict4)
    dict6=remove_superlative(dict5)
    dict7=remove_halfspace(dict6)
    dict8=remove_freq(dict7)
    return dict8
def sort_values(dict):
    dict2={}
    for word in dict:
        list=dict[word][1:]
        heapSort(list)
        list.insert(0,dict[word][0])
        dict2[word]=list
    return dict2
dict=sort_values(dict)

#sort the dictionary by the keys
#dict["آمار"]=[7,20,39,40,68,87,90,99]
inverted_index=OrderedDict(sorted(dict.items(),key=lambda word: [ persian_dict[x] if x in persian_dict else len(persian)+1 for x in word[0] ]))
print(inverted_index)

#testing fro phase 1 with different docs in phase 2 ast is removed
print("phase1 test",inverted_index['پیشبینی'])
#answer is [1,10]
print("phase1 test",inverted_index['صورت'])
#answer is [3,7,8,10]
#print(inverted_index['است'])
#answer is [6,1,3,7,8,9,10]
result=[]
#part6

#returns the normalized word
def word_normalize(word2):
    word=''.join(ch for ch in word2 if ch.isalpha() or ch=="\u200c")
    if "می‌" in word :
        word2=word.replace("می‌","")
        word2=get_suffix(word2)
    else:
        word2=word
    if word2.endswith("ها") and word2 != "ها" :
        word3=word2.replace("ها","")
    else:
        word3=word2
    if word3.endswith("های") and word3 != "های" :
        word4=word3.replace("های","")
    else:
        word4=word3
    if word4.endswith("تر") and word4 != "تر" :
        word5=word4.replace("تر","")
    else:
        word5=word4
    if word5.endswith("ترین") and word5 != "ترین" :
        word6=word5.replace("ترین","")
    else:
        word6=word5
    if "\u200c" in word6:
        word7=word6.replace("\u200c","")
        if word7.endswith("ای"):
            word8=word7.replace("ای","")
        else:
            word8=word7
    else:
        word8=word6
    return word8

def sentence_normalize(sentence):
    s=[]
    sen=sentence.split(" ")
    for word in sen:
        if word_normalize(word) != "":
            s.append(word_normalize(word))
    return s
def sentence_output_phase1(query):
    #key is docID values is the number of query words in that doc
    multi_word_query_dict={}
    for i in range(docSize):
        multi_word_query_dict[i+1]=0
    sentence=sentence_normalize(query)
    for s in sentence:
        if s in inverted_index.keys():
            value_list=inverted_index[s][1:]
        else:
            value_list=[]
        for v in value_list:
            multi_word_query_dict[v]=multi_word_query_dict[v]+1
    multi_word_query_dict={x:y for x,y in multi_word_query_dict.items() if y!=0}
    #print(multi_word_query_dict)
    for k in sorted(multi_word_query_dict, key=lambda k: multi_word_query_dict[k], reverse=True):
        result.append(k)

    #print(result)
    if not result:
        print("result not found")
    else:
        print("results: ")
        for r in result:
            print(r,".txt")



#####phase 2
#the normaliation for one tf[doc]
def mi_tf_doc(tf_doc):
    tf_doc2={}
    for word in tf_doc.keys():
        if "می‌" in word:
            flg=True
            miword2=word.replace("می\u200c","")
            miword2=get_suffix(miword2)
            if miword2 not in tf_doc.keys():
                tf_doc2[miword2]=tf_doc[word]
            elif miword2 != "\u200c":
                tf_doc2[miword2]=tf_doc[word]+tf_doc[miword2]
        else:
            tf_doc2[word]=tf_doc[word]
    return tf_doc2

def ha_tf_doc(tf_doc):
    tf_doc2={}
    for word in tf_doc.keys():
        if word.endswith("ها") and word != "ها" :
            flg=True
            #print(word,dict[word])
            haword2=word.replace("ها","")
            if haword2 not in tf_doc.keys():
                tf_doc2[haword2]=tf_doc[word]
            elif word2 != "\u200c":
                tf_doc2[haword2]=tf_doc[word]+tf_doc[haword2]
        else:
            tf_doc2[word]=tf_doc[word]
    return tf_doc2

def hay_tf_doc(tf_doc):
    tf_doc2={}
    for word in tf_doc.keys():
        if word.endswith("های") and word != "های" :
            #print(word,dict[word])
            hayword2=word.replace("های","")
            if hayword2 not in tf_doc.keys():
                tf_doc2[hayword2]=tf_doc[word]
            elif word2 != "\u200c":
                tf_doc2[hayword2]=tf_doc[word]+tf_doc[hayword2]
        else:
            tf_doc2[word]=tf_doc[word]
    return tf_doc2

def tar_tf_doc(tf_doc):
    tf_doc2={}
    for word in tf_doc.keys():
        if word.endswith("تر") and word != "تر" :
            #print(word,dict[word])
            trword2=word.replace("تر","")
            if trword2 not in tf_doc.keys():
                tf_doc2[trword2]=tf_doc[word]
            elif trword2 != "\u200c":
                tf_doc2[trword2]=tf_doc[word]+tf_doc[trword2]
        else:
            tf_doc2[word]=tf_doc[word]
    return tf_doc2
def tarin_tf_doc(tf_doc):
    tf_doc2={}
    for word in tf_doc.keys():
        if word.endswith("ترین") and word != "ترین" :
            #print(word,dict[word])
            trnword2=word.replace("ترین","")
            if trnword2 not in tf_doc.keys():
                tf_doc2[trnword2]=tf_doc[word]
            elif trnword2 != "\u200c":
                tf_doc2[trnword2]=tf_doc[word]+tf_doc[trnword2]
        else:
            tf_doc2[word]=tf_doc[word]
    return tf_doc2

def halfspace_tf_doc(tf_doc):
    tf_doc2={}
    for word in tf_doc.keys():
        if "\u200c" in word:
            flg=True
            # print(word,dict[word])
            if word.endswith("ای"):
                 word3=word.replace("ای","")
            else:
                word3=word
            halfword2=word3.replace("\u200c","")
            if halfword2 not in tf_doc.keys():
                tf_doc2[halfword2]=tf_doc[word]
            elif halfword2 != "\u200c":
                tf_doc2[halfword2]=tf_doc[word]+tf_doc[halfword2]
        else:
            tf_doc2[word]=tf_doc[word]
    return tf_doc2
def remove_freq_tf_doc(tf_doc):
    tf_doc2={}
    for word in tf_doc:
        if word not in ten_frequent_terms:
            tf_doc2[word]=tf_doc[word]
    return tf_doc2


def normalize_tf_doc(tf_doc):
    tf_doc1=mi_tf_doc(tf_doc)
    tf_doc2=ha_tf_doc(tf_doc1)
    tf_doc3=hay_tf_doc(tf_doc2)
    tf_doc4=tar_tf_doc(tf_doc3)
    tf_doc5=tarin_tf_doc(tf_doc4)
    tf_doc6=halfspace_tf_doc(tf_doc5)
    tf_doc7=remove_freq_tf_doc(tf_doc6)
    return tf_doc7
#print(tf['2'])
#print(normalize_tf_doc(tf['2']))
def normalize_tf(tf):
    tf2={}
    for doc in tf.keys():
        tf2[doc]=normalize_tf_doc(tf[doc])
    return tf2
tf=normalize_tf(tf)


##filling the championlist
def get_best_docs(term):
    #docs=[]
    #result=[]
    champ_candidate={}
    docs=inverted_index[term][1:]
    for doc in docs:
        #print(doc,":",tf[str(doc)][term])
        champ_candidate[doc]=tf[str(doc)][term]
        #champ_candidate.setdefault(doc,tf[str(doc)][term])
    if championlist_r<len(inverted_index[term]):
        result = nlargest(championlist_r, champ_candidate, key = champ_candidate.get)
    else:
        result=inverted_index[term][1:]
    return result
#print("ugh",get_best_docs("آن"))
#print("an:",inverted_index["آن"])
#print("amar:",inverted_index["آمار"])
#for d in inverted_index["آن"][2:]:
#    print(d)
#    print(tf[str(d)]["آن"])
#print(ten_frequent_terms)
for word in inverted_index.keys():
    champion[word]=get_best_docs(word)
    #champion.setdefault(word,[]).append()
#print(champion)

#print(tf["89"])
#for doc in tf:

###making the tf-idf table
inverted_series=pd.Series(inverted_index)
def get_tfidf(term,doc):
    if term not in tf[doc].keys():
        tfdt=-1
    else:
        tfdt=math.log10(tf[doc][term])
    tfidf=(1+tfdt)*math.log10( len(inverted_index) / len(inverted_index[term][1:]) )
    return tfidf

tfidf=np.zeros([len(inverted_index),docSize])
for i in range( len(inverted_index) ):
    for j in range(docSize):
        tfidf[i][j]=get_tfidf(inverted_series.index[i],str(j+1))

#getting the vectors
#a=np.arange(1,10).reshape(3,3)
#print(a)
#print(a[:,1])
#print((a.T)[0])

def word_count_query(sentence,word):
    count=0
    for s in sentence:
        if s==word:
            count=count+1
    return count
##sentence is the input query output is the long vector
##elimination happens here
def query2vec(sentence):
    vec=[]
    sentence=sentence_normalize(sentence)
    for word in inverted_index:
        if word in sentence:
            if math.log10( len(inverted_index) / len(inverted_index[word][1:]) )>elimination_idf_ts:
                vec.append(word_count_query(sentence,word))
        else:
            vec.append(0)
    return vec
#returns tekrari items of a list
def Repeat(x):
    _size = len(x)
    repeated = []
    for i in range(_size):
        k = i + 1
        for j in range(k, _size):
            if x[i] == x[j] and x[i] not in repeated:
                repeated.append(x[i])
    return repeated
#returns documents who have just one query word
def doc_eliminate(query):
    doc_count={}
    result=[]
    result3=[]
    query2=sentence_normalize(query)

    #print(query2)
    for word in query2:
        docs=inverted_index[word][1:]
        for doc in docs:
            result.append(doc)
    result2=Repeat(result)
    #print(result)
    for r in result:
        if r not in result2:
            result3.append(r)
    return result3


def cosine(vec1,vec2):
    dot=0
    sum1=0
    sum2=0
    sim=0
    for i in range(len(vec1)):
        if vec1[i] !=0 and vec2[i]!=0 :
            dot=dot+vec1[i]*vec2[i]
        sum1=sum1+vec1[i]*vec1[i]
        sum2=sum2+vec2[i]*vec2[i]
    sim=dot/( math.sqrt(sum1)*math.sqrt(sum2) )
    return sim


#print(query2vec("علی کتاب کتاب‌ها می‌رود علی")[1574]) #ro
#print(query2vec("علی کتاب کتاب‌ها می‌رود علی")[2062]) #ali
#print(query2vec("علی کتاب کتاب‌ها می‌رود علی")[2306]) #ketab
#print( cosine( query2vec("علی کتاب کتاب‌ها می‌رود علی"),tfidf[:,68] ) )
#print(inverted_index["علی"])
#print(inverted_index["کتاب"])
#print(inverted_index["رو"])



#input is query output is the championlist of query
def get_champ(query):
    query_champ=[]
    query2=sentence_normalize(query)
    for word in query2:
        for c in champion[word]:
            if c not in query_champ:
                query_champ.append(c)
    return query_champ

#("hi",get_champ("علی کتاب کتاب‌ها می‌رود علی"))


def sentence_output_phase2(query):
    query_vec=query2vec(query)
    positive_similarity={}
    heapsize=0
    result=[]
    for i in range(docSize):
        similarity=cosine(query_vec,(tfidf.T)[i])
        if similarity > 0 and i+1 not in doc_eliminate(query) :
            heapsize=heapsize+1
            positive_similarity[str(i+1)]=similarity
    ####print(positive_similarity)

    if pagesize<len(positive_similarity):
        result = nlargest(pagesize, positive_similarity, key = positive_similarity.get)
    elif pagesize>=len(positive_similarity):
        result = nlargest(len(positive_similarity), positive_similarity, key = positive_similarity.get)
        #print(result)
    if not result:
        print("result not found")
    else:
        print("results: ")
        for r in result:
            print(r,".txt")
def sentence_output_phase2_champion(query):
    query_champion=get_champ(query) #list of docID
    query_vec=query2vec(query)
    positive_similarity={}
    heapsize=0
    result=[]
    for i in range(len(query_champion)):
        similarity=cosine(query_vec,(tfidf.T)[query_champion[i]-1])
        if similarity > 0 and query_champion[i] not in doc_eliminate(query) :
            positive_similarity[query_champion[i]]=similarity
    ########print(positive_similarity)

    if pagesize<len(positive_similarity):
        result = nlargest(pagesize, positive_similarity, key = positive_similarity.get)
    elif pagesize>=len(positive_similarity):
        result = nlargest(len(positive_similarity), positive_similarity, key = positive_similarity.get)
        #print(result)
    if not result:
        print("result not found")
    else:
        print("results: ")
        for r in result:
            print(r,".txt")
######showing result-ouput
#getting the queries
query=input()
if " " in query:
    #sentence
    #sentence_output_phase1(query)
    sentence_output_phase2(query)
    #sentence_output_phase2_champion(query)

else:
    #word
    word=word_normalize(query)
    print("actual searched word: ",word)
    if word in inverted_index.keys():
        result=inverted_index[word][1:]
        print("result list:")
        for r in result:
            print(r,".txt")
    else:
        print("not found")

#sentence_output_phase2("علی کتاب کتاب‌ها می‌رود علی")
#print("this",doc_eliminate("علی کتاب کتاب‌ها می‌رود علی"))
#print(inverted_index["علی"][1:])
#print(inverted_index["کتاب"][1:])
#print(inverted_index["رو"][1:])
#print("------")
#sentence_output_phase2_champion("علی کتاب کتاب‌ها می‌رود علی")
#sentence_output_phase2("علی کتاب کتاب‌ها می‌رود علی آباداندرصد")
#for w in sentence_normalize("علی کتاب کتاب‌ها می‌رود علی آباداندرصد"):
#    print(inverted_index[w])



