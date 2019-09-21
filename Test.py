
from keras.models import load_model
from utils import GeneSeg
import numpy as np

import pickle

model_dir = "./file/LSTM_model"
model_dir1 = "./file/MLP_model"
model_dir2 = "./file/Conv_model"
vec_dir="./file/word2vec.pickle"
model = load_model(model_dir)
model_1 = load_model(model_dir1)
model_2 = load_model(model_dir2)

with open(vec_dir, "rb") as f:
    # print f.readlines()
    word2vec = pickle.load(f)
    # print type(word2vec)
    # print len(word2vec)
    # print word2vec
    dictionary = word2vec["dictionary"]
    reverse_dictionary = word2vec["reverse_dictionary"]
    embeddings = word2vec["embeddings"]

test_data = input()
data_test = []
data = GeneSeg(test_data)
print (u"split words result:" + str(data))
for word in data:
    # print word
    if word in dictionary.keys():
        data_test.append(dictionary[word])
    else:
        data_test.append(dictionary["UNK"])
#data_test=pad_sequences(data_test,value=-1)
#plot_model(model,to_file='./model.png',show_shapes=True,show_layer_names=True)
data_test2 = []
for i in range(0,591-len(data_test)):
    data_test2.append(-1);
data_test2+= data_test
data_embed = []
for d in data_test2:
    if d != -1:
        # print '1',reverse_dictionary[d]
        # print '1',embeddings[reverse_dictionary[d]]
        data_embed.append(embeddings[reverse_dictionary[d]])
    else:
        # print '2',embeddings[0]
        data_embed.append([0.0] * len(embeddings["UNK"]))
batch_data = []
print (u"data embedding result:" + str(data_embed))
for i in range(0,128):
    batch_data.append(data_embed)

answer = model.predict_on_batch(np.array(batch_data))
answer1 = model_1.predict_on_batch(np.array(batch_data))
answer2 = model_2.predict_on_batch(np.array(batch_data))

print (u"final result")

if answer[1][0] > 0.5:
    print (u"LSTM:good")
else:
    print( u"LSTM:bad")

if answer1[1][0] > 0.5:
    print (u"MLP:good")
else:
    print (u"MLP:bad")

if answer2[1][0] > 0.5:
    print (u"Conv:good")
else:
    print (u"Conv:bad")
