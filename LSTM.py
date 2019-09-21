# -*- coding: utf-8 -*-
import time

from keras.engine.saving import load_model
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Dropout, LSTM, Bidirectional, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from sklearn.metrics import precision_score, recall_score
from keras.models import model_from_json
from processing import build_dataset
from utils import init_session
import numpy as np
init_session()
batch_size=350
epochs_num=1
process_datas_dir=".\\file\\process_datas.pickle"
log_dir=".\\log\\LSTM.log"
model_dir=".\\file\\LSTM_model"
model=Sequential()
def train(train_generator,train_size,input_num,dims_num,test_generator,test_size,batch_size):
    print("Start Train Job! ")
    start=time.time()
    inputs=InputLayer(input_shape=(input_num,dims_num.shape[0]),batch_size=batch_size)
    layer1=LSTM(128,return_sequences=True)
    output=Dense(2,activation="softmax",name="Output")
    optimizer=Adam()
    model=Sequential()
    model.add(inputs)
    model.add(Bidirectional(layer1))
    #model.add(layer1)
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(output)
    call=TensorBoard(log_dir=log_dir,write_grads=True,histogram_freq=1)
    model.compile(optimizer,loss="categorical_crossentropy",metrics=["accuracy"])
    model.fit_generator(train_generator,steps_per_epoch=train_size//batch_size,epochs=epochs_num)

    test(model, test_generator, test_size, input_num, dims_num, batch_size)
   # model.fit_generator(train_generator, steps_per_epoch=5, epochs=5, callbacks=[call])
    # serialize model to JSON
    model.save(model_dir)
    model.save_weights("model.h5")
    print("Saved model to disk")
    '''
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    '''
    end=time.time()
    print("Overall job in %f s"%(end-start))

def test(loaded_model,test_generator,test_size,input_num,dims_num,batch_size):
    # load json and create model
    labels_pre=[]
    labels_true=[]
    batch_num=test_size//batch_size+1
    steps=0
    for batch,labels in test_generator:
        if len(labels)==batch_size:
            labels_pre.extend(loaded_model.predict_on_batch(batch))
        else:
            #print input_num, dims_num,batch_size - len(labels)
            batch = np.concatenate((batch, np.zeros((batch_size - len(labels), input_num, len(dims_num)))))
            labels_pre.extend(loaded_model.predict_on_batch(batch)[0:len(labels)])
        labels_true.extend(labels)
        steps+=1
        print("%d/%d batch"%(steps,batch_num))
    labels_pre=np.array(labels_pre).round()
    def to_y(labels):
        y=[]
        for i in range(len(labels)):
            if labels[i][0]==1:
                y.append(0)
            else:
                y.append(1)
        return y
    y_true=to_y(labels_true)
    y_pre=to_y(labels_pre)
    precision=precision_score(y_true,y_pre)
    recall=recall_score(y_true,y_pre)
    print("Precision score is :",precision)
    print("Recall score is :",recall)

if __name__=="__main__":
    train_generator, test_generator, train_size, test_size, input_num, dims_num=build_dataset(batch_size)
    train(train_generator,train_size,input_num,dims_num,test_generator,test_size,batch_size)
    #test(model_dir,test_generator,test_size,input_num,dims_num,batch_size)
