# -*- coding: utf-8 -*-
from utils import GeneSeg
import csv,pickle,random,json
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf

vec_dir="./file/word2vec.pickle"
data_train="./file/pre_datas_train.csv"
data_test="./file/pre_datas_test.csv"
#process_dir="./file/process_datas.pickle"

def pre_process():
    with open(vec_dir,"rb") as f :
        #print f.readlines()
        word2vec=pickle.load(f)
        #print type(word2vec)
        #print len(word2vec)
        #print word2vec
        dictionary=word2vec["dictionary"]
        reverse_dictionary=word2vec["reverse_dictionary"]
        embeddings=word2vec["embeddings"]

    sql_data=[]
    normal_data=[]
    with open("./data/sql.csv","r",encoding='UTF-8') as f:
        reader = f.readlines()
        for row in reader: ##################
            word=GeneSeg(row)
            sql_data.append(word)

    with open("./data/normal_less.csv","r",encoding='UTF-8') as f:
        reader=f.readlines()
        for row in reader:
            word=GeneSeg(row)
            normal_data.append(word)

    sql_num=len(sql_data)
    normal_num=len(normal_data)
    #print xssed_num,normal_num
    sql_labels=[1]*sql_num
    normal_labels=[0]*normal_num
    data=sql_data+normal_data
    labels=sql_labels+normal_labels
    labels=to_categorical(labels)
    def to_index(data): #去word2vec里查询
        d_index=[]
        for word in data:
            #print word
            if word in dictionary.keys():
                d_index.append(dictionary[word])
            else:

                d_index.append(dictionary["UNK"])
        #print d_index
        return d_index
    data_index=[to_index(d) for d in data[0:]]
    data_index=pad_sequences(data_index,value=-1) # 变长序列
    rand=random.sample(range(len(data_index)),len(data_index)) #打乱重采样
    data=[data_index[index] for index in rand]
    labels=[labels[index] for index in rand]
    train_data,test_data,train_labels,test_labels=train_test_split(data,labels,test_size=0.1)
    #print 'train data:',train_datas
    train_size=len(train_labels)
    test_size=len(test_labels)
    input_num=len(train_data[0])
    #print 'sb',type(embeddings)
    #print embeddings
    dims_num = embeddings["UNK"]
    word2vec["train_size"]=train_size
    word2vec["test_size"]=test_size
    word2vec["input_num"]=input_num
    word2vec["dims_num"]=dims_num
    with open(vec_dir,"wb") as f :
        pickle.dump(word2vec,f)
    print("Saved word2vec to:",vec_dir)
    print("Write train data to:",data_train)
    with open(data_train,"w") as f:
        for i in range(train_size):
            data_line=str(train_data[i].tolist())+"|"+str(train_labels[i].tolist())+"\n"
            f.write(data_line)
    print("Write test data to:",data_test)
    with open(data_test,"w") as f:
        for i in range(test_size):
            data_line=str(test_data[i].tolist())+"|"+str(test_labels[i].tolist())+"\n"
            f.write(data_line)
    print("Write data over!")
def data_generator(data_dir):
    reader = tf.TextLineReader()
    queue = tf.train.string_input_producer([data_dir])
    _, value = reader.read(queue)
    sess = tf.Session()

    while True:
        v = sess.run(value)
        [data, label] = v.split(b"|")
        data = np.array(json.loads(data.decode("utf-8")))
        label = np.array(json.loads(label.decode("utf-8")))
        yield (data, label)
    coord.request_stop()
    coord.join(threads)
    sess.close()
def batch_generator(data_dir,data_size,batch_size,embeddings,reverse_dictionary,train=True):
    batch_data = []
    batch_label = []
    generator=data_generator(data_dir)
    n=0
    while True:
        for i in range(batch_size):
            data,label=next(generator)
            data_embed = []
            for d in data:
                if d != -1:
                    #print '1',reverse_dictionary[d]
                    #print '1',embeddings[reverse_dictionary[d]]
                    data_embed.append(embeddings[reverse_dictionary[d]])
                else:
                    #print '2',embeddings[0]
                    data_embed.append([0.0] * len(embeddings["UNK"]))
            batch_data.append(data_embed)
            batch_label.append(label)
            n+=1
            if not train and n==data_size:
                break
        if not train and n == data_size:
            yield (np.array(batch_data), np.array(batch_label))
            break
        else:
            yield (np.array(batch_data),np.array(batch_label))
            batch_data = []
            batch_label = []
def build_dataset(batch_size):
    with open(vec_dir, "rb") as f:
        word2vec = pickle.load(f)
    embeddings = word2vec["embeddings"]
    reverse_dictionary = word2vec["reverse_dictionary"]
    train_size=word2vec["train_size"]
    test_size=word2vec["test_size"]
    dims_num = word2vec["dims_num"]
    input_num =word2vec["input_num"]
    train_generator = batch_generator(data_train,train_size,batch_size,embeddings,reverse_dictionary)
    test_generator = batch_generator(data_test,test_size,batch_size,embeddings,reverse_dictionary,train=False)
    return train_generator,test_generator,train_size,test_size,input_num,dims_num
if __name__=="__main__":
    pre_process()
