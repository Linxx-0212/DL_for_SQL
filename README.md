Detection  SQL injection  with Deep Learning 
====

* environment

> tensorflow
> numpy
> keras
> sklearn
> matplotlib
> nltk
>

* instruction

> 1. in ./data is the normal samples and injection samples used for training and testing
> 2. source is in ./src

* RUN

> 1. run word2vec_ginsim.py to train the word vectors
> 2. run processing.py to do pre-processing and generate the training and testing data set
> 3. run LSTM.py to train a bidirection LSTM model, do the cross-validation and save the model to disk
