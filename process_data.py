import numpy as np
import pickle as cPickle
from collections import defaultdict
import sys, re
#import pandas as pd

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()
    
vocab = defaultdict(float)

with open('Sentences_Truelabel_new', "r") as f1:
    data1=f1.read().split('\n')    
with open('Sentences_Falselabel_revised','r')as f2:
    data2=f2.read().split('\n')

data=data1+data2  
#clean_string=True
    
for i in range(len(data)):
    sen=data[i].split("\t")[-1]
    words = set(sen.split(" "))
    for word in words:
        vocab[word] += 1
   

print "vocab size: " + str(len(vocab))
print "loading word2vec vectors..."

#def load_bin_vec(fname, vocab):
"""
Loads 200x1 word vecs from pubmed wv
"""
word_vecs = {}
with open('wikipedia-pubmed-and-PMC-w2v.bin', "rb") as f:
    header = f.readline()
    vocab_size, layer1_size = map(int, header.split())
    binary_len = np.dtype('float32').itemsize * layer1_size
    for line in xrange(vocab_size):
        word = []
        while True:
            ch = f.read(1)
            if ch == ' ':
                word = ''.join(word)
                break
            if ch != '\n':
                word.append(ch)   
        if word in vocab:
           word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
        else:
            f.read(binary_len)
#    return word_vecs

print "num words already in word2vec: " + str(len(word_vecs))


#def add_unknown_words(word_vecs, vocab, min_df=1, k=200):
"""
For words that occur in at least min_df documents, create a separate word vector.    
0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
"""
k=200
min_df=1
for word in vocab:
    if word not in word_vecs and vocab[word] >= min_df:
        word_vecs[word] = np.random.uniform(-0.25,0.25,k)  

#print("num words already in word2vec: " + str(len(word_vecs)))

##################
###Prepare file###
##################


file1=open("dict.txt","w")
file2=open("wv.txt","w")
for key, value in word_vecs.items():
    file1.write(key+'\n')
    L=list(value)
    for t in range(len(L)):
        if t==len(L)-1:
            file2.write(str(L[t]))
        else:
            file2.write(str(L[t])+' ')
    file2.write('\n')

file1.close()
file2.close()

################################
###prepare LOOK UP Dictionary###
################################

LOOKUP={}
C=1
for k in word_vecs.keys():
    LOOKUP[k]=C
    C+=1

################################
###Generate the Training data###
################################

def generate_filtered_data(filename,Data,REL=True):
    file3 = open(filename,'w')    
    D={}
    for i in range(len(Data)):
        Pair=Data[i].split('\t')[0]
        D[Pair]=[]
    
    for j in range(len(Data)):
        sen=Data[j].split('\t')[-1]
        Pair=Data[j].split('\t')[0]
        Pos=Data[j].split('\t')[2]
        Dic=defaultdict(list)
        Pro=Pos.split('|')
        for m in range(1,len(Pro)):
            Name=Pro[m].split('_')[0]
            Position=Pro[m].split('_')[1]
            Dic[Name].append(Position)        
        En1=Data[j].split("\t")[0].split("'")[1]
        En2=Data[j].split("\t")[0].split("'")[3]
        if len(Dic[En1])>1 or len(Dic[En1])>1:
            continue    
        D[Pair].append(Dic[En1][0]+'\t'+Dic[En2][0]+'\t'+sen)
    
    
    for key, value in D.items():
        if len(value)==0:
            continue
        else:        
            E1=key.split("'")[1]
            E2=key.split("'")[3]
            try:
                file3.write(str(LOOKUP[E1])+' '+str(LOOKUP[E2])+'\n')
                if REL:
                    file3.write(str(1)+' '+str(len(value))+'\n')
                else:
                    file3.write(str(0)+' '+str(len(value))+'\n')
                for k in range(len(value)):
                    PP1=value[k].split('\t')[0]
                    PP2=value[k].split('\t')[1]
                    SS=value[k].split('\t')[-1]
                    WW=SS.split(' ')
                    file3.write(str(PP1)+' '+str(PP2))
                    for n in range(len(WW)):
                        if n==len(WW)-1:
                            file3.write(str(LOOKUP[WW[n]]))
                        else:
                            file3.write(str(LOOKUP[WW[n]])+' ')
                    file3.write('\n')
            except:
                    continue
    file3.close()   
    return None

generate_filtered_data("data_filtered_True",data1,REL=True)
generate_filtered_data("data_filtered_False",data2,REL=False)
